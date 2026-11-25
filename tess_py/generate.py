"""Star field + light-curve generation (ground truth).

Standalone:   python generate.py --seed 42 [--n 25]
Nothing here knows about the browser, Pyodide, or Matplotlib.
"""
import json
import numpy as np
import stars as stars_mod

# ─────────────────────────────────────────────────────────────────────────────
# Companion / distance model. Ground truth lives here (and in the star dicts);
# what a student is allowed to see is decided by the UI, not by this module.
#
# Calibration notes (sources: Raghavan+2010; Ziegler+2020,2021 SOAR TESS surveys):
#   - Physical separation: log-normal in a, peak ~50 AU (Raghavan: orbital-period
#     peak ~300 yr).  Angular sep = a / distance, so nearer stars look wider.
#   - Mass ratio q ~ flat on [0.1, 1];  delta-mag(I) ~ 7.5*log10(1/q)  (0..7.5)
#   - Imaging window: 0.03" .. 4" (speckle FOV ~6").  Companions inside the
#     window are real and dilute the light curve whether or not they resolve.
#   - Detectability: SOAR-like contrast curve; ~5 mag beyond ~0.5", worse
#     closer in and for fainter hosts; unresolved below ~0.05".
COMP_FRAC      = 0.30
LOGA_MEAN      = 1.7      # log10(a/AU)
LOGA_SIGMA     = 1.5
SEP_MIN, SEP_MAX = 0.03, 4.0     # arcsec
RESOLVE_MIN    = 0.05            # arcsec (~0.85 x the 0.06" imager FWHM)
DMAG_PER_DEX   = 7.5

# Per-star truth that is not part of the JSON the UI sees (blend EB parameters,
# dilution factor).  Rebuilt by generate_starfield.
_truth = {}
# id -> star dict (as returned in the JSON), so other modules can look a star up
_stars = {}

def _contrast_limit(sep, brightness):
    """5-sigma delta-mag limit of the (synthetic) imager at separation sep."""
    return 5.0 - 4.0 * np.exp(-sep / 0.25) - 1.5 * (1.0 - brightness)

def _detectable(sep, dmag, brightness):
    return bool(sep >= RESOLVE_MIN and dmag <= _contrast_limit(sep, brightness))

BG_FRAC        = 0.06     # chance alignment with an unrelated (non-eclipsing) background star


def _bound_companion(srng, dist_pc, brightness):
    """Physically bound companion, or None.  Same distance as the primary.
    The separation is provisional: _finalize() rescales it once the primary's
    final distance is known (the companion's light changes the distance)."""
    if srng.random() >= COMP_FRAC:
        return None
    sep = a = proj = None
    for _ in range(200):                       # truncated log-normal in a
        a    = 10 ** srng.normal(LOGA_MEAN, LOGA_SIGMA)
        proj = srng.uniform(0.3, 1.0)          # projection factor
        s    = a * proj / dist_pc
        if SEP_MIN <= s <= SEP_MAX:
            sep = s
            break
    if sep is None:
        return None
    q    = srng.uniform(0.1, 1.0)
    dmag = DMAG_PER_DEX * np.log10(1.0 / q)
    pa   = srng.uniform(0.0, 360.0)
    inc  = float(np.degrees(np.arccos(srng.uniform(0.0, 1.0))))   # isotropic orbit
    return dict(sep_arcsec=float(sep), pa_deg=float(pa), dmag=float(dmag),
                bound=True, is_eb=False, dist_pc=float(dist_pc),
                q=float(q), a_au=float(a), proj=float(proj), inc_deg=inc,
                detectable=_detectable(sep, dmag, brightness))


def _chance_sep(rng):
    """Separation of a chance alignment: uniform per unit sky area in the imaging window."""
    u = rng.uniform((SEP_MIN / SEP_MAX) ** 2, 1.0)
    return float(SEP_MAX * np.sqrt(u))


def _background(rng, sep, dmag, is_eb):
    """Unbound star far behind the target (eclipsing or not)."""
    return dict(sep_arcsec=float(sep), pa_deg=float(rng.uniform(0.0, 360.0)), dmag=float(dmag),
                bound=False, is_eb=bool(is_eb), dist_factor=float(rng.uniform(1.5, 5.0)),
                dist_pc=0.0, detectable=False)                 # dist/detectable set in _finalize


def _make_blend(s, brng, sep, dmag):
    """Turn star s into a blended background eclipsing binary."""
    f_eb = 1.0 / (1.0 + 10 ** (0.4 * dmag))                    # EB's share of the light
    obs  = float(brng.uniform(0.004, 0.02))                     # depth the student sees
    d1   = float(min(obs / f_eb, 0.7))                          # intrinsic eclipse depth
    P    = float(brng.uniform(1.0, 10.0))
    s['type']      = 'blend_eb'
    s['companion'] = _background(brng, sep, dmag, True)
    _truth[s['id']] = dict(dilution=f_eb,
                           blend=dict(period=P, t0=float(brng.uniform(0, P)),
                                      dur_frac=float(brng.uniform(0.03, 0.08)),
                                      d1=d1, d2=float(d1 * brng.uniform(0.05, 0.9))))


def _set_spectral_type(s, label, trng):
    """Copy true stellar properties for `label` onto star s, plus [Fe/H] and v sin i."""
    p = stars_mod.props(label)
    s.update(spt=p['spt'], teff=p['teff'], mass=p['mass'], radius=p['radius'],
             logg=p['logg'], lum_class=p['lum_class'])
    teff = p['teff']
    if p['lum_class'] == 'III':
        vsini = trng.uniform(1.0, 5.0)
    elif teff > 7800:
        vsini = trng.uniform(80.0, 150.0)
    elif teff > 6900:
        vsini = trng.uniform(30.0, 80.0)
    elif teff > 6300:
        vsini = trng.uniform(8.0, 20.0)
    elif teff > 6000:
        vsini = trng.uniform(4.0, 10.0)
    else:
        vsini = trng.uniform(1.0, 5.0)
    s['vsini'] = float(vsini)
    s['feh']   = float(np.clip(trng.normal(-0.05, 0.2), -0.8, 0.5))


PULSATOR_TYPES = ('A5 V', 'F0 V', 'F5 V')          # gamma Dor / delta Sct territory


def _finalize(s, z_plx):
    """Distance, parallax, final companion geometry, dilution truth.

    `brightness` is the total light in the TESS aperture.  The primary alone is
    fainter by its companion's share, and its distance follows from that, so a
    student who later uses the total brightness overestimates the luminosity."""
    T = stars_mod.tmag_from_brightness(s['brightness'])
    c = s['companion']
    F = 10 ** (-0.4 * c['dmag']) if c else 0.0
    T1 = T + 2.5 * np.log10(1.0 + F)
    d = float(np.clip(stars_mod.distance_pc(T1, s['radius'], s['teff']), 5.0, 3000.0))
    s['tmag'] = float(T)
    s['distance_pc'] = d
    perr = float(0.02 + 0.08 * (1.0 - s['brightness']))            # mas, Gaia-like
    s['parallax_err_mas'] = perr
    s['parallax_mas'] = float(1000.0 / d + z_plx * perr)
    if c:
        if c['bound']:
            if c.get('fix_sep'):                                    # debug scenarios fix the angle
                c['a_au'] = float(c['sep_arcsec'] * d / c['proj'])
            sep = c['a_au'] * c['proj'] / d
            sep = float(np.clip(sep, SEP_MIN, SEP_MAX))
            c['a_au'] = float(sep * d / c['proj'])
            c['sep_arcsec'] = sep
            c['dist_pc'] = d
        else:
            c['dist_pc'] = float(d * c['dist_factor'])
        c['detectable'] = _detectable(c['sep_arcsec'], c['dmag'], s['brightness'])
        if not c['is_eb']:                    # any non-EB neighbour dilutes the host's own signal
            _truth[s['id']] = dict(dilution=1.0 / (1.0 + F))


# Debug field: one star per scenario, so every path through the pipeline can be exercised.
SCENARIOS = ['clean_planet', 'planet_bound', 'planet_bg', 'eb', 'beb_bright',
             'beb_faint', 'pulsator', 'quiet', 'giant_transit']


def _force_scenarios(stars, seed):
    if len(stars) < len(SCENARIOS):
        return
    frng = np.random.default_rng([seed, 10])
    ids  = [int(i) for i in frng.permutation(len(stars))][:len(SCENARIOS)]
    for role, i in zip(SCENARIOS, ids):
        s    = stars[i]
        rr   = np.random.default_rng([seed, 11, i])
        T    = stars_mod.tmag_from_brightness(s['brightness'])
        _truth.pop(i, None)
        s['companion'] = None
        s['scenario']  = role
        spt = None
        if role == 'clean_planet':
            s['type'] = 'transit'; spt = 'G2 V'
        elif role == 'planet_bound':
            s['type'] = 'transit'; spt = 'K0 V'
            q = 10 ** (-2.0 / DMAG_PER_DEX)
            s['companion'] = dict(sep_arcsec=0.6, pa_deg=float(rr.uniform(0, 360)), dmag=2.0,
                                  bound=True, is_eb=False, dist_pc=0.0, q=float(q),
                                  a_au=0.0, proj=0.8, inc_deg=60.0, fix_sep=True, detectable=False)
        elif role == 'planet_bg':
            s['type'] = 'transit'; spt = 'F8 V'
            s['companion'] = _background(rr, 1.0, 3.0, False)
        elif role == 'eb':
            s['type'] = 'eb'; spt = 'G5 V'
        elif role == 'beb_bright':
            spt = 'G0 V'; _make_blend(s, rr, 0.6, 1.5)
        elif role == 'beb_faint':
            spt = 'K0 V'; _make_blend(s, rr, 2.8, 3.5)
        elif role == 'pulsator':
            s['type'] = 'pulsating'; spt = 'F0 V'
        elif role == 'quiet':
            s['type'] = 'boring'; spt = stars_mod.draw_type(rr, T, giants=False)
        elif role == 'giant_transit':
            s['type'] = 'transit'; spt = 'K1 III'
        _set_spectral_type(s, spt, rr)


def generate_starfield(seed=42, n_stars=25, debug=False):
    """JSON list of stars.  Each carries its ground truth (system type, spectral
    type, companion, distance); what a student may see is decided by the UI.
    debug=True forces one star per entry of SCENARIOS."""
    rng   = np.random.default_rng(seed)
    types = ['boring']*5 + ['pulsating', 'transit', 'transit', 'eb']
    stars = []
    zs    = {}
    _truth.clear()
    _stars.clear()
    for i in range(n_stars):
        # Main stream: positions, brightness, system type.
        s = {
            'id':         i,
            'x':          float(rng.uniform(0.05, 0.95)),
            'y':          float(rng.uniform(0.05, 0.95)),
            'brightness': float(rng.uniform(0.2, 1.0)),
            'type':       str(rng.choice(types)),
        }
        # Per-star streams, so adding a property never perturbs the others.
        srng = np.random.default_rng([seed, 1, i])
        trng = np.random.default_rng([seed, 5, i])
        zs[i] = float(srng.normal(0.0, 1.0))                       # parallax error draw
        T = stars_mod.tmag_from_brightness(s['brightness'])
        allowed = PULSATOR_TYPES if s['type'] == 'pulsating' else None
        giants  = s['type'] not in ('pulsating', 'eb')
        _set_spectral_type(s, stars_mod.draw_type(trng, T, giants=giants, allowed=allowed), trng)
        d0 = stars_mod.distance_pc(T, s['radius'], s['teff'])
        s['companion'] = _bound_companion(srng, d0, s['brightness'])
        stars.append(s)

    # Blended background eclipsing binaries, converted from 'boring' stars.  The
    # first one is forced to be an easy imaging detection so there is always
    # one to debug.
    brng   = np.random.default_rng([seed, 2])
    n_bl   = max(2, int(round(0.08 * n_stars)))
    boring = [s['id'] for s in stars if s['type'] == 'boring']
    others = [s['id'] for s in stars if s['type'] != 'boring']      # fallback only
    pool   = boring if len(boring) >= n_bl else boring + others
    chosen = [int(i) for i in brng.permutation(pool)][:n_bl]
    for k, i in enumerate(chosen):
        if k == 0:
            sep  = float(brng.uniform(0.5, 2.5));  dmag = float(brng.uniform(1.5, 3.5))
        else:                                       # chance alignment: uniform on the sky
            sep  = _chance_sep(brng);              dmag = float(brng.uniform(1.0, 3.8))
        _make_blend(stars[i], brng, sep, dmag)

    # Unrelated (non-eclipsing) background stars: they only dilute.
    for s in stars:
        if s['companion'] is None:
            bg = np.random.default_rng([seed, 9, s['id']])
            if bg.random() < BG_FRAC:
                s['companion'] = _background(bg, _chance_sep(bg), float(bg.uniform(1.0, 5.0)), False)

    if debug:
        _force_scenarios(stars, seed)
    for s in stars:
        _finalize(s, zs[s['id']])
        _stars[s['id']] = s
    return json.dumps(stars)


# ─────────────────────────────────────────────────────────────────────────────
def gen_flux(star_id, star_type, brightness, seed):
    """
    Generate the time array and flux for one star.
    This is the single source of truth — called once per star, result cached.
    """
    rng   = np.random.default_rng(seed * 1000 + star_id)
    t     = np.linspace(0, 27, 1500)            # 27-day TESS sector
    sigma = 0.0006 + (1.0 - brightness) * 0.0024  # DEBUG: 5x reduced noise
    flux  = np.ones(len(t)) + rng.normal(0, sigma, len(t))
    noise = flux - 1.0
    params = {}

    if star_type == 'pulsating':
        P = float(rng.uniform(0.5, 3.0))
        A = float(rng.uniform(0.005, 0.02))
        flux += A * np.sin(2 * np.pi * t / P)
        params = dict(period=P, amp=A)

    elif star_type == 'transit':
        P        = float(rng.uniform(2.0, 15.0))
        depth    = float(rng.uniform(0.005, 0.025))
        t0       = float(rng.uniform(0, P))
        dur_frac = float(rng.uniform(0.03, 0.08))  # transit duration as fraction of P
        ph       = ((t - t0) % P) / P
        ph       = np.where(ph > 0.5, ph - 1.0, ph)
        flux[np.abs(ph) < dur_frac / 2] -= depth
        params   = dict(period=P, depth=depth, t0=t0, dur_frac=dur_frac)

    elif star_type == 'eb':
        P  = float(rng.uniform(1.0, 8.0))
        d1 = float(rng.uniform(0.05, 0.25))        # primary depth
        d2 = float(d1 * rng.uniform(0.3, 0.7))     # secondary depth
        t0 = float(rng.uniform(0, P))
        ph = ((t - t0) % P) / P
        # Primary eclipse (V-shape centred at phase 0)
        dp = np.minimum(np.abs(ph), 1.0 - np.abs(ph))
        w1 = 0.08
        m1 = dp < w1
        flux[m1] -= d1 * (1.0 - dp[m1] / w1)
        # Secondary eclipse (phase 0.5)
        ds = np.abs(ph - 0.5)
        w2 = 0.06
        m2 = ds < w2
        flux[m2] -= d2 * (1.0 - ds[m2] / w2)
        params = dict(period=P, d1=d1, d2=d2, t0=t0)

    elif star_type == 'blend_eb':
        # Background EB: box-like eclipses (primary at phase 0, secondary at 0.5),
        # diluted below by the foreground star's light.
        b  = _truth[star_id]['blend']
        P, t0, df = b['period'], b['t0'], b['dur_frac']
        ph1 = ((t - t0) % P) / P;            ph1 = np.where(ph1 > 0.5, ph1 - 1.0, ph1)
        ph2 = ((t - t0 - P / 2) % P) / P;    ph2 = np.where(ph2 > 0.5, ph2 - 1.0, ph2)
        flux[np.abs(ph1) < df / 2] -= b['d1']
        flux[np.abs(ph2) < df / 2] -= b['d2']
        params = dict(period=P, depth=b['d1'], t0=t0, dur_frac=df, d2=b['d2'])

    # Light from a companion inside the aperture dilutes every signal; noise is
    # untouched (it was drawn first, so the random stream is unchanged).
    dil = _truth.get(star_id, {}).get('dilution', 1.0)
    if dil != 1.0:
        flux = 1.0 + noise + dil * (flux - 1.0 - noise)
    params['dilution'] = float(dil)
    if 'depth' in params:
        params['depth_obs'] = float(params['depth'] * dil)

    return t, flux, sigma, params


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Generate a TESS-game star field and print a summary.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n', type=int, default=25)
    ap.add_argument('--debug', action='store_true', help='force one star per debug scenario')
    a = ap.parse_args()
    for s in json.loads(generate_starfield(a.seed, a.n, a.debug)):
        c = s['companion']
        kind = lambda c: 'BG-EB' if c['is_eb'] else ('bound' if c['bound'] else 'BG-star')
        comp = '-' if not c else (
            f"{kind(c)} sep={c['sep_arcsec']:.2f}\" "
            f"dm={c['dmag']:.1f} {'detectable' if c['detectable'] else 'hidden'}")
        print(f"#{s['id']:<3d} {s['type']:<9s} {s['spt']:<7s} b={s['brightness']:.2f} "
              f"d={s['distance_pc']:5.0f}pc plx={s['parallax_mas']:6.2f}  {comp}"
              f"  {s.get('scenario', '')}")
