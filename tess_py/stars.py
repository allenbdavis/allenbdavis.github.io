"""Stellar properties: spectral-type table, magnitudes, bolometric corrections,
and the inverse (spectral-type label from a measured Teff and log g).

Standalone:   python stars.py          # prints the table with absolute TESS mags
Pure numpy; used by generate.py (truth) and rv.py (recon "measurements").

Values are rounded main-sequence / giant-branch numbers in the spirit of
Pecaut & Mamajek (2013).  They are teaching-grade, not a calibration.
"""
import numpy as np

TEFF_SUN = 5772.0
MBOL_SUN = 4.74

# label, Teff [K], mass [Msun], radius [Rsun], sampling weight, luminosity class
TABLE = [
    ('A5 V',  8100, 1.85, 1.70, 0.04, 'V'),
    ('F0 V',  7220, 1.59, 1.60, 0.08, 'V'),
    ('F5 V',  6510, 1.33, 1.40, 0.13, 'V'),
    ('F8 V',  6170, 1.18, 1.20, 0.09, 'V'),
    ('G0 V',  5920, 1.06, 1.10, 0.10, 'V'),
    ('G2 V',  5770, 1.00, 1.00, 0.09, 'V'),
    ('G5 V',  5660, 0.97, 0.93, 0.08, 'V'),
    ('K0 V',  5270, 0.88, 0.85, 0.09, 'V'),
    ('K3 V',  4830, 0.78, 0.75, 0.07, 'V'),
    ('K5 V',  4440, 0.70, 0.68, 0.05, 'V'),
    ('M0 V',  3850, 0.57, 0.59, 0.03, 'V'),
    ('M2 V',  3560, 0.44, 0.44, 0.02, 'V'),
    ('M4 V',  3210, 0.23, 0.26, 0.01, 'V'),
    ('G8 III', 4950, 2.00, 9.0, 0.03, 'III'),
    ('K1 III', 4600, 1.30, 11.0, 0.03, 'III'),
]
DWARFS = [r for r in TABLE if r[5] == 'V']
GIANTS = [r for r in TABLE if r[5] == 'III']

# Finer Teff grid used only to name a measured Teff (dwarf scale; fine for giants too)
SUBTYPES = [('A0', 9700), ('A5', 8100), ('F0', 7220), ('F2', 6810), ('F5', 6510),
            ('F8', 6170), ('G0', 5920), ('G2', 5770), ('G5', 5660), ('G8', 5490),
            ('K0', 5270), ('K2', 5040), ('K3', 4830), ('K5', 4440), ('K7', 4050),
            ('M0', 3850), ('M1', 3680), ('M2', 3560), ('M3', 3430), ('M4', 3210),
            ('M5', 3060)]


def tmag_from_brightness(b):
    """The game's 0..1 'brightness' -> apparent TESS magnitude of everything in
    the TESS aperture (b=1 -> T=7.0, b=0.2 -> T=11.4)."""
    return 12.5 - 5.5 * b


def bc_tess(teff):
    """Bolometric correction in the TESS band (M_bol - M_T).  Smooth toy fit,
    ~0.5 mag for Sun-like stars, smaller for very hot or very cool stars."""
    x = np.log10(np.asarray(teff, float) / 5800.0)
    return 0.5 - 3.0 * x ** 2


def luminosity(radius, teff):
    return radius ** 2 * (teff / TEFF_SUN) ** 4


def abs_tmag(radius, teff):
    mbol = MBOL_SUN - 2.5 * np.log10(luminosity(radius, teff))
    return mbol - bc_tess(teff)


def logg(mass, radius):
    return 4.438 + np.log10(mass) - 2.0 * np.log10(radius)


def distance_pc(app_tmag, radius, teff):
    return 10.0 ** ((app_tmag - abs_tmag(radius, teff) + 5.0) / 5.0)


def row(label):
    return next(r for r in TABLE if r[0] == label)


def props(label):
    """dict of true stellar properties for a table entry."""
    lab, teff, m, r, _, lc = row(label)
    return dict(spt=lab, teff=float(teff), mass=float(m), radius=float(r),
                logg=float(logg(m, r)), lum_class=lc)


def draw_type(rng, app_tmag, giants=True, allowed=None, dmin=10.0, dmax=2000.0):
    """Draw a spectral type from the weighted table, rejecting types that would
    put a star of this apparent magnitude closer than dmin or beyond dmax."""
    rows = [r for r in TABLE if (giants or r[5] == 'V') and (allowed is None or r[0] in allowed)]
    ok = [r for r in rows if dmin <= distance_pc(app_tmag, r[3], r[1]) <= dmax]
    pool = ok or rows
    w = np.array([r[4] for r in pool]); w = w / w.sum()
    return pool[int(rng.choice(len(pool), p=w))][0]


def label_from(teff, logg_val):
    """Spectral type + luminosity class from measured Teff and log g."""
    name = min(SUBTYPES, key=lambda s: abs(s[1] - teff))[0]
    lc = 'V' if logg_val >= 3.9 else 'IV' if logg_val >= 3.4 else 'III'
    return f'{name} {lc}'


def ms_mass_from_teff(teff):
    """Main-sequence mass from Teff (the 'stellar models' step for dwarfs)."""
    t = np.array([r[1] for r in DWARFS], float)[::-1]
    m = np.array([r[2] for r in DWARFS], float)[::-1]
    return float(np.interp(teff, t, m))


def mass_lum_ratio(q):
    """Rough flux ratio of a main-sequence companion of mass ratio q (L ~ M^4)."""
    return float(np.clip(q, 0.0, 1.0)) ** 4


if __name__ == '__main__':
    print(f"{'type':7s} {'Teff':>5s} {'M':>5s} {'R':>5s} {'logg':>5s} {'M_T':>6s}   d(T=7..11.4)")
    for lab, teff, m, r, w, lc in TABLE:
        print(f"{lab:7s} {teff:5d} {m:5.2f} {r:5.2f} {logg(m, r):5.2f} {abs_tmag(r, teff):6.2f}   "
              f"{distance_pc(7.0, r, teff):6.0f}..{distance_pc(11.4, r, teff):6.0f} pc")
