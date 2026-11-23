"""Thin JS-facing API.  The page calls these functions by name through
pyodide.runPython and gets JSON strings back.  This is the only module that
holds state (the per-star light-curve cache) or builds JSON; everything else
is pure and independently testable.
"""
import json
import numpy as np

import generate
import analysis
import plots
import imaging
import rv
from plot_utils import to_b64

__all__ = ['generate_starfield', 'make_lightcurve', 'find_best_period',
           'make_folded_plot', 'make_oddeven_plot', 'make_psf_image',
           'p3_calendar', 'p3_ephemeris', 'p3_observe', 'p3_fit']

# Per-star light-curve cache: filled by make_lightcurve, read by the others.
_star_data = {}

LABELS = dict(boring='Quiet Star', pulsating='Pulsating Star',
              transit='Transit Candidate', eb='Eclipsing Binary',
              blend_eb='Blended Background EB')


def generate_starfield(seed=42, n_stars=25, debug=False):
    return generate.generate_starfield(seed, n_stars, debug)


def make_lightcurve(star_id, star_type, brightness, seed):
    """Plot the full light curve; also returns the axes bounds JS needs to map
    clicks to times, the noise level, and (debug) the true signal parameters."""
    t, flux, sigma, params = generate.gen_flux(star_id, star_type, brightness, seed)
    _star_data[star_id]    = dict(t=t, flux=flux, sigma=sigma, params=params)

    fig, ax = plots.lightcurve_fig(
        t, flux, f"Star #{star_id}  \u2014  {LABELS.get(star_type, star_type)}")
    bbox = ax.get_position()      # figure-fraction bounding box of the axes
    b64  = to_b64(fig)

    return json.dumps(dict(
        img  = b64,
        ax_x0 = float(bbox.x0), ax_x1 = float(bbox.x1),
        ax_y0 = float(bbox.y0), ax_y1 = float(bbox.y1),
        t_min = 0.0, t_max = 27.0,
        sigma = float(sigma),
        true_period = float(params.get('period', -1)),
        true_t0     = float(params.get('t0', -1)),
        true_depth  = float(params.get('depth', -1)),
        true_depth_obs = float(params.get('depth_obs', -1)),
        dilution    = float(params.get('dilution', 1.0)),
    ))


def find_best_period(star_id, t0_guess, period_guess, n_periods=200):
    """Iterative period search + fold plot.
    Returns JSON: { img, best_period, snr, depth, dur_frac, chi2r }"""
    d = _star_data.get(star_id)
    if not d:
        return json.dumps(dict(error='no data'))
    t, flux, sigma = d['t'], d['flux'], d['sigma']

    r = analysis.search_period(t, flux, sigma, t0_guess, period_guess, n_periods)
    base, depth = analysis.refine_box(r['ph'], flux, r['center'], r['width'],
                                      r['depth'], fallback='mean')
    fig = plots.fold_fig(r['ph'], flux, (r['center'], r['width'], base, depth),
                         r['period'], r['snr'])
    return json.dumps(dict(
        img         = to_b64(fig),
        best_period = r['period'],
        snr         = r['snr'],
        depth       = depth,
        dur_frac    = r['width'] * 2,
        chi2r       = r['chi2r'],
    ))


def make_folded_plot(star_id, period, t0):
    """Fold the cached light curve on (period, t0) with a BLS box fit.
    Returns JSON: { img, snr, depth, dur_frac, chi2r }"""
    d = _star_data.get(star_id)
    if not d:
        return json.dumps(dict(error='no data'))
    t, flux, sigma = d['t'], d['flux'], d['sigma']

    ph = analysis.fold_phase(t, t0, period)
    c0, w, dep, chi2, n_in = analysis.bls_vectorized(ph, flux, sigma)
    chi2r = chi2 / max(len(flux) - 2, 1)
    snr   = dep / (sigma / max(n_in ** 0.5, 1))

    base, depth = analysis.refine_box(ph, flux, c0, w, dep, fallback='median')
    fig = plots.fold_fig(ph, flux, (c0, w, base, depth), period, snr)
    return json.dumps(dict(
        img      = to_b64(fig),
        snr      = snr,
        depth    = dep,
        dur_frac = w * 2,
        chi2r    = chi2r,
    ))


def make_oddeven_plot(star_id, period, t0):
    """Fold and colour by even/odd transit number (EB diagnostic)."""
    d = _star_data.get(star_id)
    if not d:
        return json.dumps(dict(error='no data'))
    t, flux = d['t'], d['flux']

    ph   = analysis.fold_phase(t, t0, period)
    tnum = np.floor((t - t0) / period).astype(int)
    even = (tnum % 2 == 0)
    return json.dumps(dict(img=to_b64(plots.oddeven_fig(ph, flux, even))))


def make_psf_image(star_id, seed):
    """Phase 2 high-resolution image of one star.  Pixels are float32 counts,
    base64-encoded (little endian).  No truth is returned: the page only sees
    what an observer would see.  `debug` carries true positions for the overlay."""
    import base64
    s = generate._stars.get(star_id)
    if s is None:
        return json.dumps(dict(error='no star'))
    r = imaging.render(s, seed)
    enc = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype='<f4').tobytes()).decode('ascii')
    return json.dumps(dict(
        n=r['n'], ref_n=r['ref_n'], pix_scale=r['pix_scale'], fwhm=r['fwhm'],
        img=enc(r['img']), ref=enc(r['ref']),
        debug=r['truth'],
    ))


# ── Phase 3: reconnaissance radial velocities ──────────────────────────────
def _lc(star_id, seed):
    """Cached light curve for a star (regenerated if Phase 1 never loaded it)."""
    d = _star_data.get(star_id)
    if d is None:
        s = generate._stars[star_id]
        t, flux, sigma, params = generate.gen_flux(star_id, s['type'], s['brightness'], seed)
        d = _star_data[star_id] = dict(t=t, flux=flux, sigma=sigma, params=params)
    return d


def p3_calendar():
    """Observing constants, so the page never duplicates them."""
    return json.dumps(dict(months=rv.MONTHS, n_nights=rv.N_NIGHTS, cal0=rv.CAL0,
                           win_start=rv.WIN_START, win_len=rv.WIN_LEN, n_obs=rv.N_OBS,
                           max_per_night=rv.MAX_PER_NIGHT, sigma_rv=rv.SIGMA_RV,
                           fiber=rv.FIBER_ARCSEC, seeing=rv.SEEING_ARCSEC))


def p3_ephemeris(star_id, seed, period, t0):
    """Student's ephemeris (P from Phase 1, T0 = box-fit centre) with
    Delta-chi^2 = 1 uncertainties.  JSON: {period, t0, sig_t0, sig_p, ...}"""
    d = _lc(star_id, seed)
    return json.dumps(rv.ephemeris(d['t'], d['flux'], d['sigma'], float(period), float(t0)))


def p3_observe(star_id, seed, nights_json):
    """Run a 5-night schedule.  Returns times, RVs, per-observation CCFs and
    the reconnaissance stellar parameters.  `debug` holds the truth."""
    s = generate._stars[star_id]
    r = rv.observe(s, seed, json.loads(nights_json))
    rc = rv.recon(s, seed, r['system']['comps'])
    comps = [dict(name=c['name'], light=c['light'], fwhm=c['fwhm'])
             for c in r['system']['comps']]
    return json.dumps(dict(
        times=[float(x) for x in r['times']],
        rv=[round(float(x), 4) for x in r['rv']],
        err=[float(x) for x in r['err']],
        nights=r['nights'],
        labels=[rv.night_label(n) for n in r['nights']],
        clocks=[rv.clock(t) for t in r['times']],
        v=[round(float(x), 3) for x in r['v']],
        ccfs=[[round(float(x), 4) for x in row] for row in r['ccfs']],
        recon=rc,
        debug=dict(truth=r['system']['truth'], comps=comps),
    ), default=float)


def p3_fit(payload_json):
    """Sinusoid fit at a given ephemeris + minimum companion mass.
    payload: {t, rv, err, period, t0, mass}."""
    p = json.loads(payload_json)
    f = rv.fit_sine(p['t'], p['rv'], p['err'], p['period'], p['t0'])
    f.update(rv.mass_summary(f['K'], f['eK'], p['period'], p['mass']))
    return json.dumps(f)
