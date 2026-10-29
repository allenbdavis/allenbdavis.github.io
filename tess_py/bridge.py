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
from plot_utils import to_b64

__all__ = ['generate_starfield', 'make_lightcurve', 'find_best_period',
           'make_folded_plot', 'make_oddeven_plot']

# Per-star light-curve cache: filled by make_lightcurve, read by the others.
_star_data = {}

LABELS = dict(boring='Quiet Star', pulsating='Pulsating Star',
              transit='Transit Candidate', eb='Eclipsing Binary',
              blend_eb='Blended Background EB')


def generate_starfield(seed=42, n_stars=25):
    return generate.generate_starfield(seed, n_stars)


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
