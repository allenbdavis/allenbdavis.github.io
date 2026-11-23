"""Phase 3: reconnaissance (low-precision) radial velocities.

Standalone:   python rv.py --seed 0 --star 13 --debug
              python rv.py --seed 0 --star 13 --debug --nights 0 0 10 20 40

Everything about a system's spectrum is modelled as a set of *components* that
put light into the spectrograph fiber (the target, plus any companion star,
background eclipsing binary or background star close enough to leak in).
Each component has a velocity law v(t) and a Gaussian cross-correlation (CCF)
dip.  The measured RV is the centre of a single-Gaussian fit to the summed CCF
near the target's dip, plus Gaussian noise of SIGMA_RV.  So a blend only pulls
the velocity when its lines overlap the target's (see the design notes).

Pure numpy.  Truth comes from generate.py; nothing here builds JSON.
"""
import numpy as np
import generate
import analysis
import stars as st

# ── Physical constants (SI) ─────────────────────────────────────────────────
G, MSUN, MJUP = 6.674e-11, 1.989e30, 1.898e27
RSUN, RJUP, DAY = 6.957e8, 7.1492e7, 86400.0

# ── Observing set-up ────────────────────────────────────────────────────────
MONTHS     = [('Month 1', 30), ('Month 2', 31), ('Month 3', 30)]
N_NIGHTS   = sum(n for _, n in MONTHS)        # 91
CAL0       = 57.0           # TESS time (d) of Month 1 Day 1, 00:00 -- a month after the sector (t=0..27)
WIN_START  = 19.5 / 24.0    # each night's window opens at 19:30 ...
WIN_LEN    = 6.0 / 24.0     # ... and lasts 6 h
N_OBS      = 5
MAX_PER_NIGHT = 2
SIGMA_RV   = 0.200          # km/s, per observation
FIBER_ARCSEC = 2.3          # TRES-like reconnaissance spectrograph
SEEING_ARCSEC = 1.5
FWHM_INST  = 6.8            # km/s, c / R for R ~ 44,000
FWHM_MACRO = 5.0            # km/s, thermal + macroturbulence
LINE_DEPTH = 0.5            # CCF contrast of an unblended, slowly rotating star
CCF_NOISE  = 0.004          # per-pixel noise on the displayed CCF


# ── Calendar ────────────────────────────────────────────────────────────────
def night_window(n):
    """(start, end) in TESS days of night n (0 = Month 1 Day 1)."""
    s = CAL0 + n + WIN_START
    return s, s + WIN_LEN


def night_label(n):
    for name, days in MONTHS:
        if n < days:
            return f'{name}, Day {n + 1}'
        n -= days
    raise ValueError('night outside the calendar')


def clock(t):
    """Local clock time hh:mm of TESS time t."""
    m = int(round(((t % 1.0) * 24 * 60))) % (24 * 60)
    return f'{m // 60:02d}:{m % 60:02d}'


def validate_nights(nights):
    nights = [int(n) for n in nights]
    if len(nights) != N_OBS:
        raise ValueError(f'need exactly {N_OBS} nights')
    for n in nights:
        if not 0 <= n < N_NIGHTS:
            raise ValueError(f'night {n} outside the calendar')
        if nights.count(n) > MAX_PER_NIGHT:
            raise ValueError(f'night {n} used more than {MAX_PER_NIGHT} times')
    return sorted(nights)


def obs_times(seed, star_id, nights):
    """Actual observation times: random inside each night's window; two
    observations on the same night are at least half a window apart."""
    nights = validate_nights(nights)
    out = []
    for n in sorted(set(nights)):
        rng = np.random.default_rng([seed, 7, star_id, n])
        a, _ = night_window(n)
        if nights.count(n) == 1:
            out.append(a + rng.uniform(0, WIN_LEN))
        else:
            while True:
                t1, t2 = rng.uniform(0, WIN_LEN, 2)
                if abs(t1 - t2) >= WIN_LEN / 2:
                    out += sorted([a + t1, a + t2])
                    break
    return np.array(out)


# ── Orbits ──────────────────────────────────────────────────────────────────
def semi_amplitude(P_days, m1_sun, m2_sun, sini=1.0):
    """K of body 1 (km/s) for a circular orbit with companion mass m2."""
    m1, m2 = m1_sun * MSUN, m2_sun * MSUN
    return (2 * np.pi * G / (P_days * DAY)) ** (1 / 3) * m2 * sini / (m1 + m2) ** (2 / 3) / 1e3


def msini(K_kms, P_days, m1_sun):
    """Minimum companion mass (Msun) from the full mass function (not the
    planet approximation), so it stays right for stellar companions."""
    K = abs(K_kms) * 1e3
    if K <= 0 or P_days <= 0 or m1_sun <= 0:
        return 0.0
    f = P_days * DAY * K ** 3 / (2 * np.pi * G)
    m1 = m1_sun * MSUN
    lo, hi = 0.0, 1000 * MSUN
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid ** 3 / (m1 + mid) ** 2 < f:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) / MSUN


# ── Fiber coupling ──────────────────────────────────────────────────────────
_coupling_cache = {}


def fiber_coupling(sep_arcsec, fiber=FIBER_ARCSEC, seeing=SEEING_ARCSEC):
    """Fraction of a star's light entering a circular fiber centred sep away,
    relative to the same star on axis (Gaussian seeing)."""
    key = (fiber, seeing)
    if key not in _coupling_cache:
        s = seeing / 2.3548
        seps = np.linspace(0, fiber / 2 + 4 * seeing, 120)
        x = np.linspace(-fiber / 2, fiber / 2, 161)
        X, Y = np.meshgrid(x, x)
        inside = X ** 2 + Y ** 2 <= (fiber / 2) ** 2
        frac = np.array([np.exp(-((X[inside] - d) ** 2 + Y[inside] ** 2) / (2 * s * s)).sum()
                         for d in seps])
        _coupling_cache[key] = (seps, frac / frac[0])
    seps, frac = _coupling_cache[key]
    return float(np.interp(sep_arcsec, seps, frac, right=0.0))


# ── System model ────────────────────────────────────────────────────────────
def _fwhm(vsini):
    return float(np.sqrt(FWHM_INST ** 2 + FWHM_MACRO ** 2 + (1.5 * vsini) ** 2))


def _comp(name, light, teff, logg, fwhm, vel):
    return dict(name=name, light=float(light), teff=float(teff), logg=float(logg),
                fwhm=float(fwhm), vel=vel)


def _ms_teff_from_mass(m):
    d = st.DWARFS
    return float(np.interp(m, [r[2] for r in d][::-1], [r[1] for r in d][::-1]))


def _ms_lum(m):
    return float(m) ** 4


def system(star, seed):
    """Components that put light into the fiber, plus a truth summary.

    Velocities are km/s.  Transit / primary eclipse at T0, with the target
    moving away from us just before and toward us just after:
    v_target = gamma - K1 sin(2 pi (t - T0) / P)."""
    sid, typ = star['id'], star['type']
    _, _, _, p = generate.gen_flux(sid, typ, star['brightness'], seed)
    rng   = np.random.default_rng([seed, 6, sid])
    gamma = float(rng.normal(0.0, 25.0))
    M1, R1 = star['mass'], star['radius']
    L1    = float(st.luminosity(R1, star['teff']))
    terms = []                                 # target velocity terms: functions of t
    comps = []
    truth = dict(gamma=gamma, K1=0.0, companion=None)

    def sin_term(K, P, T0, sign=-1.0):
        return lambda t: sign * K * np.sin(2 * np.pi * (np.asarray(t) - T0) / P)

    if typ == 'transit':
        P, T0 = p['period'], p['t0']
        rc = R1 * np.sqrt(p['depth'])                      # companion radius, Rsun
        rj = rc * RSUN / RJUP
        if rj <= 1.8:                                      # a planet
            if rj > 0.8:
                mp = float(np.clip(10 ** rng.normal(0.0, 0.35), 0.2, 8.0))
            elif rj > 0.4:
                mp = float(rng.uniform(0.1, 0.4))
            else:
                mp = float((rj * 11.21) ** 1.7 / 317.8)
            m2 = mp * MJUP / MSUN
            K1 = semi_amplitude(P, M1, m2)
            terms.append(sin_term(K1, P, T0))
            truth.update(K1=K1, companion=dict(kind='planet', mass_mjup=mp, radius_rjup=float(rj),
                                               period=P, t0=T0))
        else:                                              # too big: a low-mass star
            m2 = float(np.clip(rc, 0.08, 0.9 * M1))
            K1 = semi_amplitude(P, M1, m2)
            K2 = K1 * M1 / m2
            terms.append(sin_term(K1, P, T0))
            comps.append(_comp('eclipsing companion', _ms_lum(m2) / L1, _ms_teff_from_mass(m2), 4.8,
                               _fwhm(3.0), (lambda g=gamma, f=sin_term(K2, P, T0, +1): lambda t: g + f(t))()))
            truth.update(K1=K1, companion=dict(kind='star', mass_sun=m2, radius_rjup=float(rj),
                                               period=P, t0=T0))
    elif typ == 'eb':
        P, T0 = p['period'], p['t0']
        m2 = float(M1 * rng.uniform(0.4, 0.95))
        K1 = semi_amplitude(P, M1, m2)
        K2 = K1 * M1 / m2
        terms.append(sin_term(K1, P, T0))
        comps.append(_comp('EB secondary', _ms_lum(m2) / L1, _ms_teff_from_mass(m2), 4.6,
                           _fwhm(3.0), (lambda g=gamma, f=sin_term(K2, P, T0, +1): lambda t: g + f(t))()))
        truth.update(K1=K1, companion=dict(kind='star', mass_sun=m2, period=P, t0=T0))
    elif typ == 'pulsating':
        Kp = 100.0 * p['amp']                              # ~1 km/s per 1% flux amplitude
        Pp = p['period']
        terms.append(lambda t, Kp=Kp, Pp=Pp: Kp * np.cos(2 * np.pi * np.asarray(t) / Pp))
        truth.update(K1=Kp, companion=dict(kind='pulsation', period=Pp))

    c = star.get('companion')
    if c and c['is_eb']:                                   # blended background EB
        b = generate._truth[sid]['blend']
        P, T0 = b['period'], b['t0']
        g_bg = gamma + float(rng.normal(0.0, 20.0))
        ma = float(rng.uniform(0.7, 1.3)); mb = float(ma * rng.uniform(0.5, 0.95))
        Ka = semi_amplitude(P, ma, mb); Kb = Ka * ma / mb
        lt = 10 ** (-0.4 * c['dmag']) * fiber_coupling(c['sep_arcsec'])
        la, lb = lt * _ms_lum(ma) / (_ms_lum(ma) + _ms_lum(mb)), lt * _ms_lum(mb) / (_ms_lum(ma) + _ms_lum(mb))
        for nm, l, m, f in [('background EB primary', la, ma, sin_term(Ka, P, T0)),
                            ('background EB secondary', lb, mb, sin_term(Kb, P, T0, +1))]:
            comps.append(_comp(nm, l, _ms_teff_from_mass(m), 4.5, _fwhm(3.0),
                               (lambda g=g_bg, f=f: lambda t: g + f(t))()))
        truth['blend'] = dict(gamma=g_bg, Ka=Ka, Kb=Kb, period=P, t0=T0,
                              light_in_fiber=lt, coupling=fiber_coupling(c['sep_arcsec']))
    elif c and not c['bound']:                             # static background star
        g_bg = gamma + float(rng.normal(0.0, 30.0))
        l = 10 ** (-0.4 * c['dmag']) * fiber_coupling(c['sep_arcsec'])
        comps.append(_comp('background star', l, float(rng.uniform(4500, 6500)), 4.4, _fwhm(3.0),
                           (lambda g=g_bg: lambda t: g + 0.0 * np.asarray(t))()))
        truth['background'] = dict(gamma=g_bg, light_in_fiber=l)
    elif c and c['bound']:                                 # wide bound companion
        m2 = c['q'] * M1
        mt = M1 + m2
        vrel = 29.78 * np.sqrt(mt / c['a_au']) * np.sin(np.radians(c['inc_deg']))
        Pb = 365.25 * np.sqrt(c['a_au'] ** 3 / mt)
        ph0 = float(rng.uniform(0, 2 * np.pi))
        orb = lambda t, Pb=Pb, ph0=ph0: np.sin(2 * np.pi * np.asarray(t) / Pb + ph0)
        terms.append(lambda t, k=vrel * m2 / mt, orb=orb: -k * orb(t))
        l = 10 ** (-0.4 * c['dmag']) * fiber_coupling(c['sep_arcsec'])
        comps.append(_comp('bound companion', l, _ms_teff_from_mass(m2), 4.6, _fwhm(3.0),
                           (lambda g=gamma, k=vrel * M1 / mt, orb=orb: lambda t: g + k * orb(t))()))
        truth['bound'] = dict(period_yr=Pb / 365.25, K_target=vrel * m2 / mt, light_in_fiber=l)

    target_vel = lambda t, terms=tuple(terms): gamma + sum((f(t) for f in terms), 0.0 * np.asarray(t, float))
    comps.insert(0, _comp('target', 1.0, star['teff'], star['logg'], _fwhm(star['vsini']), target_vel))
    return dict(comps=comps, truth=truth, gamma=gamma)


def weights(comps):
    L = np.array([c['light'] for c in comps])
    return L / L.sum()


def ccf_model(comps, t, v):
    """Noise-free summed CCF at time t on velocity grid v."""
    w = weights(comps)
    out = np.ones_like(v, dtype=float)
    for c, wk in zip(comps, w):
        sig = c['fwhm'] / 2.3548
        depth = min(LINE_DEPTH * wk * (9.0 / c['fwhm']) ** 0.5, 0.95)
        out -= depth * np.exp(-0.5 * ((v - float(c['vel'](t))) / sig) ** 2)
    return out


def fit_ccf_centre(comps, t):
    """Centre of a single-Gaussian (target-width) fit to the summed CCF,
    searched near the target's dip -- what a pipeline would report."""
    tgt = comps[0]
    v0 = float(tgt['vel'](t)); F = tgt['fwhm']; sig = F / 2.3548
    v = np.arange(v0 - 4 * F, v0 + 4 * F, 0.05)
    y = 1.0 - ccf_model(comps, t, v)
    cs = np.arange(v0 - 1.5 * F, v0 + 1.5 * F, 0.02)
    # least squares with fixed width & free depth: maximise sum(y g)^2 / sum(g^2)
    score = np.array([np.dot(y, g) ** 2 / np.dot(g, g)
                      for g in (np.exp(-0.5 * ((v - c) / sig) ** 2) for c in cs)])
    i = int(np.argmax(score))
    if 0 < i < len(cs) - 1:                                # parabolic refinement
        a, b, c = score[i - 1], score[i], score[i + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
        return float(cs[i] + off * (cs[1] - cs[0]))
    return float(cs[i])


# ── Observing ───────────────────────────────────────────────────────────────
def observe(star, seed, nights):
    """Run a 5-night schedule: times, RVs, and the CCF of each observation."""
    nights = validate_nights(nights)
    sysm = system(star, seed)
    comps = sysm['comps']
    times = obs_times(seed, star['id'], nights)
    nrng = np.random.default_rng([seed, 12, star['id']])
    rv = np.array([fit_ccf_centre(comps, t) for t in times]) + nrng.normal(0, SIGMA_RV, len(times))

    # Display grid: centred on the measured RVs, wide enough for every visible dip.
    centre = float(np.median(rv))
    half = 3.0 * comps[0]['fwhm']
    w = weights(comps)
    for c, wk in zip(comps[1:], w[1:]):
        if LINE_DEPTH * wk > CCF_NOISE:
            vs = [float(c['vel'](t)) for t in times]
            half = max(half, max(abs(x - centre) for x in vs) + 2.5 * c['fwhm'])
    half = float(np.clip(half, 60.0, 300.0))
    v = np.linspace(centre - half, centre + half, 401)
    ccfs = [ccf_model(comps, t, v) + nrng.normal(0, CCF_NOISE, v.size) for t in times]
    return dict(times=times, rv=rv, err=np.full(len(times), SIGMA_RV), nights=nights,
                v=v, ccfs=np.array(ccfs), system=sysm)


# ── Reconnaissance stellar parameters ───────────────────────────────────────
def recon(star, seed, comps):
    """Spectroscopic Teff/log g/[Fe/H]/v sin i, and the derived radius and
    mass (spectrum + Gaia parallax + total TESS magnitude).  Light from any
    companion in the fiber biases the spectrum; light from any companion in
    the TESS aperture biases the luminosity."""
    rng = np.random.default_rng([seed, 13, star['id']])
    w = weights(comps)
    teff = float(np.dot(w, [c['teff'] for c in comps]) + rng.normal(0, 100))
    lg = float(np.dot(w, [c['logg'] for c in comps]) + rng.normal(0, 0.10))
    feh = float(star['feh'] + rng.normal(0, 0.08))
    vsini = float(max(0.5, star['vsini'] * (1 + rng.normal(0, 0.08))))
    plx, eplx = star['parallax_mas'], star['parallax_err_mas']
    d = 1000.0 / plx
    T = star['tmag']
    MT = T - 5 * np.log10(d / 10)
    bc = float(st.bc_tess(teff))
    mbol = MT + bc
    L = 10 ** (-0.4 * (mbol - st.MBOL_SUN))
    R = float(np.sqrt(L) / (teff / st.TEFF_SUN) ** 2)
    eR = R * float(np.hypot(2 * 100 / teff, eplx / plx))
    if lg >= 3.9:
        M = st.ms_mass_from_teff(teff); how = 'main-sequence models (Teff)'
    else:
        M = float(10 ** (lg - 4.438) * R ** 2); how = 'surface gravity and radius (M = g R² / G)'
    return dict(spt=st.label_from(teff, lg), teff=teff, eteff=100.0, logg=lg, elogg=0.10,
                feh=feh, efeh=0.08, vsini=vsini, mass=float(M), emass=0.1 * float(M), mass_method=how,
                radius=R, eradius=eR, parallax=plx, eparallax=eplx, distance=d, tmag=T,
                abs_tmag=float(MT), bc=bc, mbol=float(mbol), lum=float(L))


# ── Fitting ─────────────────────────────────────────────────────────────────
def fit_sine(t, rv, err, P, T0):
    """Linear least squares for v = gamma - K sin(2 pi (t - T0) / P).
    K is signed: K < 0 means the RVs move opposite to a transiting companion."""
    t, rv, err = map(lambda a: np.asarray(a, float), (t, rv, err))
    s = -np.sin(2 * np.pi * (t - T0) / P)
    A = np.vstack([np.ones_like(t), s]).T
    W = 1.0 / err ** 2
    AtWA = A.T @ (A * W[:, None])
    cov = np.linalg.inv(AtWA)
    gamma, K = cov @ (A.T @ (W * rv))
    resid = rv - (gamma + K * s)
    chi2 = float(np.sum(resid ** 2 * W))
    dof = max(len(t) - 2, 1)
    return dict(gamma=float(gamma), K=float(K), egamma=float(np.sqrt(cov[0, 0])),
                eK=float(np.sqrt(cov[1, 1])), chi2=chi2, chi2r=chi2 / dof, dof=dof)


def mass_summary(K, eK, P, m1):
    m = msini(K, P, m1)
    hi = msini(abs(K) + eK, P, m1)
    lo = msini(max(abs(K) - eK, 0.0), P, m1)
    return dict(msini_sun=m, msini_mjup=m * MSUN / MJUP,
                emsini_mjup=0.5 * (hi - lo) * MSUN / MJUP)


# ── Ephemeris from the Phase 1 light curve ─────────────────────────────────
def ephemeris(t, flux, sigma, period, t0, n=61):
    """The student's ephemeris with uncertainties from the box-model chi^2.

    Period stays the student's; T0 becomes the box-fit centre.  sigma_T0 and
    sigma_P are the half-widths of the Delta chi^2 <= 1 region of each profile
    (the other parameter minimised over).  T0 is quoted at the epoch nearest
    the middle of the data, where it is least correlated with P."""
    t, flux = np.asarray(t, float), np.asarray(flux, float)
    P = float(period)
    ph = analysis.fold_phase(t, t0, P)
    c0, w, _, _, _ = analysis.bls_vectorized(ph, flux, sigma)
    tc = t0 + c0 * P
    mid = 0.5 * (t.min() + t.max())
    tref = tc + np.round((mid - tc) / P) * P
    half = w * P                                           # box half-duration (d)
    ncyc = max((t.max() - t.min()) / 2 / P, 1.0)
    dT = np.linspace(-half, half, n)
    dP = np.linspace(-1, 1, n) * half / ncyc
    var = sigma ** 2
    C = np.empty((n, n))
    for i, dp in enumerate(dP):
        Pi = P + dp
        x = (t[None, :] - (tref + dT[:, None])) / Pi
        x = x - np.round(x)
        inb = np.abs(x) * Pi < half
        nin = inb.sum(1); nout = t.size - nin
        sin_ = (inb * flux).sum(1)
        mi = np.where(nin > 0, sin_ / np.maximum(nin, 1), 1.0)
        mo = (flux.sum() - sin_) / np.maximum(nout, 1)
        model = np.where(inb, mi[:, None], mo[:, None])
        C[i] = ((flux[None, :] - model) ** 2).sum(1) / var
    cmin = C.min()
    i0 = n // 2                                            # dP = 0: the student's period
    tref = float(tref + dT[int(np.argmin(C[i0]))])

    def half_width(prof, grid):
        ok = grid[prof <= prof.min() + 1.0]
        step = grid[1] - grid[0]
        return float(max((ok.max() - ok.min()) / 2, step / 2)), bool(ok.min() <= grid[0] or ok.max() >= grid[-1])

    sT, capT = half_width(C.min(axis=0), dT)
    sP, capP = half_width(C.min(axis=1), dP)
    return dict(period=P, t0=tref, sig_t0=sT, sig_p=sP, capped=capT or capP,
                duration=2 * half, chi2_min=float(cmin))


def phase_sigma(t, eph):
    """1-sigma uncertainty in predicted orbital phase at time t."""
    n = (np.asarray(t, float) - eph['t0']) / eph['period']
    return np.sqrt(eph['sig_t0'] ** 2 + (n * eph['sig_p']) ** 2) / eph['period']


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser(description='Simulate one star\'s reconnaissance RVs.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n', type=int, default=25)
    ap.add_argument('--star', type=int, default=0)
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--nights', type=int, nargs=N_OBS, default=[0, 6, 13, 27, 45])
    a = ap.parse_args()
    S = json.loads(generate.generate_starfield(a.seed, a.n, a.debug))
    s = S[a.star]
    print(f"star {s['id']}  {s['type']}  {s['spt']}  {s.get('scenario', '')}  companion={s['companion']}")
    r = observe(s, a.seed, a.nights)
    for c in r['system']['comps']:
        print(f"  component {c['name']:<24s} light={c['light']:.4f} fwhm={c['fwhm']:.1f} km/s")
    print('  truth', {k: v for k, v in r['system']['truth'].items()})
    for n, t, v in zip(r['nights'], r['times'], r['rv']):
        print(f"  {night_label(n):<16s} {clock(t)}  t={t:8.4f}  RV={v:9.3f} km/s")
    rc = recon(s, a.seed, r['system']['comps'])
    print(f"  recon: {rc['spt']}  Teff={rc['teff']:.0f}  logg={rc['logg']:.2f}  "
          f"R={rc['radius']:.2f}  M={rc['mass']:.2f}   (true {s['spt']} R={s['radius']} M={s['mass']})")
    p = r['system']['truth'].get('companion') or {}
    if 'period' in p:
        t, f, sig, _ = generate.gen_flux(s['id'], s['type'], s['brightness'], a.seed)
        eph = ephemeris(t, f, sig, p['period'], p.get('t0', 0.0))
        fit = fit_sine(r['times'], r['rv'], r['err'], eph['period'], eph['t0'])
        ms = mass_summary(fit['K'], fit['eK'], eph['period'], rc['mass'])
        print(f"  ephemeris {eph}")
        print(f"  fit K={fit['K']:.3f}±{fit['eK']:.3f} km/s chi2r={fit['chi2r']:.2f}  "
              f"m sin i = {ms['msini_mjup']:.2f} MJup ({ms['msini_sun']:.3f} Msun)")
