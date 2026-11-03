"""Star field + light-curve generation (ground truth).

Standalone:   python generate.py --seed 42 [--n 25]
Nothing here knows about the browser, Pyodide, or Matplotlib.
"""
import json
import numpy as np

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
#     closer in and for fainter hosts; unresolved below ~0.04".
COMP_FRAC      = 0.30
LOGA_MEAN      = 1.7      # log10(a/AU)
LOGA_SIGMA     = 1.5
SEP_MIN, SEP_MAX = 0.03, 4.0     # arcsec
RESOLVE_MIN    = 0.04            # arcsec
DMAG_PER_DEX   = 7.5
DIST_MEDIAN_PC = 200.0           # typical TESS planet host (Ziegler+2021)
DIST_SIGMA_DEX = 0.30

# Per-star truth that is not part of the JSON the UI sees (blend EB parameters,
# dilution factor).  Rebuilt by generate_starfield.
_truth = {}

def _contrast_limit(sep, brightness):
    """5-sigma delta-mag limit of the (synthetic) imager at separation sep."""
    return 5.0 - 4.0 * np.exp(-sep / 0.25) - 1.5 * (1.0 - brightness)

def _detectable(sep, dmag, brightness):
    return bool(sep >= RESOLVE_MIN and dmag <= _contrast_limit(sep, brightness))

def _bound_companion(srng, dist_pc, brightness):
    """Physically bound companion, or None.  Same distance as the primary."""
    if srng.random() >= COMP_FRAC:
        return None
    sep = None
    for _ in range(200):                       # truncated log-normal in a
        a   = 10 ** srng.normal(LOGA_MEAN, LOGA_SIGMA)
        s   = a * srng.uniform(0.3, 1.0) / dist_pc     # projection factor
        if SEP_MIN <= s <= SEP_MAX:
            sep = s
            break
    if sep is None:
        return None
    q    = srng.uniform(0.1, 1.0)
    dmag = DMAG_PER_DEX * np.log10(1.0 / q)
    pa   = srng.uniform(0.0, 360.0)
    return dict(sep_arcsec=float(sep), pa_deg=float(pa), dmag=float(dmag),
                bound=True, is_eb=False, dist_pc=float(dist_pc),
                detectable=_detectable(sep, dmag, brightness))

def generate_starfield(seed=42, n_stars=25):
    rng   = np.random.default_rng(seed)
    types = ['boring']*5 + ['pulsating', 'transit', 'transit', 'eb']
    stars = []
    _truth.clear()
    for i in range(n_stars):
        # Existing draws, unchanged -> same positions/brightness/types as before.
        s = {
            'id':         i,
            'x':          float(rng.uniform(0.05, 0.95)),
            'y':          float(rng.uniform(0.05, 0.95)),
            'brightness': float(rng.uniform(0.2, 1.0)),
            'type':       str(rng.choice(types)),
        }
        # New truth, from per-star streams so it can never perturb the above.
        srng  = np.random.default_rng([seed, 1, i])
        dist  = float(np.clip(10 ** srng.normal(np.log10(DIST_MEDIAN_PC), DIST_SIGMA_DEX), 25.0, 800.0))
        perr  = float(0.02 + 0.08 * (1.0 - s['brightness']))        # mas, Gaia-like
        s['distance_pc']       = dist                                # truth (foreground/primary)
        s['parallax_mas']      = float(1000.0 / dist + srng.normal(0.0, perr))   # measured
        s['parallax_err_mas']  = perr
        s['companion']         = _bound_companion(srng, dist, s['brightness'])
        stars.append(s)

    # Blended background eclipsing binaries.  Converted from 'boring' stars so
    # no planet / EB / pulsator light curve in the field changes.  The first one
    # is forced to be an easy imaging detection so there is always one to debug.
    brng   = np.random.default_rng([seed, 2])
    n_bl   = max(2, int(round(0.08 * n_stars)))
    boring = [s['id'] for s in stars if s['type'] == 'boring']
    others = [s['id'] for s in stars if s['type'] != 'boring']      # fallback only
    pool   = boring if len(boring) >= n_bl else boring + others
    chosen = [int(i) for i in brng.permutation(pool)][:n_bl]
    for k, i in enumerate(chosen):
        s = stars[i]
        if k == 0:
            sep  = float(brng.uniform(0.5, 2.5));  dmag = float(brng.uniform(1.5, 3.5))
        else:                                       # chance alignment: uniform on the sky
            u    = brng.uniform((SEP_MIN / SEP_MAX) ** 2, 1.0)
            sep  = float(SEP_MAX * np.sqrt(u));    dmag = float(brng.uniform(1.0, 3.8))
        f_eb   = 1.0 / (1.0 + 10 ** (0.4 * dmag))                 # EB's share of the light
        obs    = float(brng.uniform(0.004, 0.02))                 # depth the student sees
        d1     = float(min(obs / f_eb, 0.7))                      # intrinsic eclipse depth
        P      = float(brng.uniform(1.0, 10.0))
        s['type']      = 'blend_eb'
        s['companion'] = dict(sep_arcsec=sep, pa_deg=float(brng.uniform(0.0, 360.0)),
                              dmag=dmag, bound=False, is_eb=True,
                              dist_pc=float(s['distance_pc'] * brng.uniform(1.5, 5.0)),
                              detectable=_detectable(sep, dmag, s['brightness']))
        _truth[i] = dict(dilution=f_eb,
                         blend=dict(period=P, t0=float(brng.uniform(0, P)),
                                    dur_frac=float(brng.uniform(0.03, 0.08)),
                                    d1=d1, d2=float(d1 * brng.uniform(0.05, 0.9))))
    # Bound companions dilute the host's own signal: host keeps 1/(1+10^(-0.4 dm)).
    for s in stars:
        c = s['companion']
        if c and c['bound']:
            _truth[s['id']] = dict(dilution=1.0 / (1.0 + 10 ** (-0.4 * c['dmag'])))
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
    a = ap.parse_args()
    for s in json.loads(generate_starfield(a.seed, a.n)):
        c = s['companion']
        comp = '-' if not c else (
            f"{'BG-EB' if c['is_eb'] else 'bound'} sep={c['sep_arcsec']:.2f}\" "
            f"dm={c['dmag']:.1f} {'detectable' if c['detectable'] else 'hidden'}")
        print(f"#{s['id']:<3d} {s['type']:<9s} b={s['brightness']:.2f} "
              f"d={s['distance_pc']:4.0f}pc plx={s['parallax_mas']:5.2f}  {comp}")
