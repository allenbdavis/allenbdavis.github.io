"""Synthetic high-resolution images for Phase 2 (speckle / adaptive-optics style).

Standalone:   python imaging.py --seed 0 --star 3 --out star3.png

Conventions (shared with the page):
  * pixel (i, j) covers [i, i+1) x [j, j+1); coordinates are continuous, so a
    pixel centre is at i + 0.5.  x grows to the right, y grows DOWN.
  * North is up, East is LEFT.  Position angle is measured from North toward East.
  * Images are float32 counts.  Nothing here reveals the companion beyond what
    the pixels show; `truth` is returned separately for debug overlays and tests.

Noise is built from the same contrast curve that decides which companions are
"detectable" in generate.py: the per-pixel noise at distance r from the primary
is set so a companion sitting exactly at the contrast limit has a peak of ~5
sigma.  That keeps the pixels and the `detectable` flag consistent by
construction (and tests/test_imaging.py checks it).
"""
import numpy as np
import generate

PIX_SCALE = 0.02        # arcsec / pixel
N_PIX     = 256         # 5.12" field of view
FWHM      = 0.06        # arcsec  (3 pixels)
SIGMA_PX  = FWHM / 2.3548 / PIX_SCALE

HALO_FRAC  = 0.06       # fraction of the light in a broad Moffat halo
HALO_ALPHA = 0.25       # arcsec
HALO_BETA  = 2.5

FLUX0     = 4.0e5       # total electrons from a brightness-1.0 star
READ_NOISE = 5.0        # electrons
SNR_LIMIT = 5.0         # peak S/N of a companion exactly at the contrast limit
REF_N     = 32          # size of the reference point-source cutout
EDGE      = 12          # keep both stars at least this many pixels from the edge


def star_flux(brightness):
    """Total counts; the 1.5-mag span matches the faint-host term in generate._contrast_limit."""
    return FLUX0 * 10 ** (-0.6 * (1.0 - brightness))


def _grid(n):
    ax = np.arange(n) + 0.5
    return np.meshgrid(ax, ax)          # X (right), Y (down)


def psf(n, cx, cy):
    """Unit-flux PSF (Gaussian core + Moffat halo) centred at (cx, cy) pixels."""
    X, Y = _grid(n)
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    core = np.exp(-r2 / (2 * SIGMA_PX ** 2)) / (2 * np.pi * SIGMA_PX ** 2)
    a = HALO_ALPHA / PIX_SCALE
    halo = (HALO_BETA - 1) / (np.pi * a ** 2) * (1 + r2 / a ** 2) ** (-HALO_BETA)
    return (1 - HALO_FRAC) * core + HALO_FRAC * halo


PSF_PEAK = float(psf(41, 20.5, 20.5).max())      # peak pixel of a unit-flux PSF


def offset_pixels(sep_arcsec, pa_deg):
    """(dx, dy) in pixels of a companion relative to its primary: N up, E left."""
    pa = np.radians(pa_deg)
    return (-sep_arcsec * np.sin(pa) / PIX_SCALE,
            -sep_arcsec * np.cos(pa) / PIX_SCALE)


def noise_sigma(r_arcsec, brightness):
    """Per-pixel noise implied by the contrast curve at distance r from the primary."""
    lim  = generate._contrast_limit(r_arcsec, brightness)
    peak = star_flux(brightness) * PSF_PEAK * 10 ** (-0.4 * lim)
    return peak / SNR_LIMIT


def _positions(rng, companion):
    """Primary near the centre (small random offset); shift the pair to stay in frame."""
    ang, rad = rng.uniform(0, 2 * np.pi), 0.3 / PIX_SCALE * np.sqrt(rng.uniform())
    p = np.array([N_PIX / 2 + rad * np.cos(ang), N_PIX / 2 + rad * np.sin(ang)])
    pts = [p]
    if companion:
        pts.append(p + np.array(offset_pixels(companion['sep_arcsec'], companion['pa_deg'])))
    pts = np.array(pts)
    lo, hi = EDGE - pts.min(axis=0), (N_PIX - EDGE) - pts.max(axis=0)
    shift = np.where(lo > 0, lo, np.where(hi < 0, hi, 0.0))
    return pts + shift


def render(star, seed):
    """Render one star. Returns dict(img, ref, truth, ...) with float32 arrays."""
    rng  = np.random.default_rng([seed, 3, star['id']])
    comp = star.get('companion')
    b    = star['brightness']
    F    = star_flux(b)
    pts  = _positions(rng, comp)
    px, py = pts[0]
    model = F * psf(N_PIX, px, py)
    if comp:
        cx, cy = pts[1]
        model = model + F * 10 ** (-0.4 * comp['dmag']) * psf(N_PIX, cx, cy)

    X, Y = _grid(N_PIX)
    r_as = np.hypot(X - px, Y - py) * PIX_SCALE          # distance from the primary
    var  = noise_sigma(r_as, b) ** 2 + np.clip(model, 0, None) + READ_NOISE ** 2
    img  = model + rng.normal(size=model.shape) * np.sqrt(var)

    # Reference point source: same brightness, same noise recipe, no companion.
    r2   = np.random.default_rng([seed, 4, star['id']])
    rx   = REF_N / 2 + r2.uniform(-0.5, 0.5)
    ry   = REF_N / 2 + r2.uniform(-0.5, 0.5)
    rX, rY = _grid(REF_N)
    rmodel = F * psf(REF_N, rx, ry)
    rr   = np.hypot(rX - rx, rY - ry) * PIX_SCALE
    rvar = noise_sigma(rr, b) ** 2 + np.clip(rmodel, 0, None) + READ_NOISE ** 2
    ref  = rmodel + r2.normal(size=rmodel.shape) * np.sqrt(rvar)

    return dict(
        img=img.astype('<f4'), ref=ref.astype('<f4'), n=N_PIX, ref_n=REF_N,
        pix_scale=PIX_SCALE, fwhm=FWHM,
        truth=dict(primary=[float(px), float(py)],
                   companion=[float(pts[1][0]), float(pts[1][1])] if comp else None),
    )


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser(description='Render a Phase 2 image to a PNG (log + linear stretch).')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--star', type=int, default=0)
    ap.add_argument('--n', type=int, default=25)
    ap.add_argument('--out', default='star.png')
    a = ap.parse_args()
    stars = json.loads(generate.generate_starfield(a.seed, a.n))
    s = stars[a.star]
    r = render(s, a.seed)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    x = np.clip(r['img'] / r['img'].max(), 0, 1)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(x, cmap='gray', origin='upper'); axs[0].set_title('linear')
    axs[1].imshow(np.log10(1000 * x + 1) / 3, cmap='gray', origin='upper'); axs[1].set_title('log')
    fig.suptitle(f"star {s['id']}  b={s['brightness']:.2f}  companion={s['companion']}", fontsize=8)
    fig.savefig(a.out, dpi=100)
    print('wrote', a.out)
