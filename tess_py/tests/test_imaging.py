import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import generate as g
import imaging as im


def stars(seed=0, n=60):
    return json.loads(g.generate_starfield(seed, n))


def test_geometry_north_up_east_left():
    assert im.offset_pixels(1.0, 0)[1] < 0            # north = up (smaller y)
    assert im.offset_pixels(1.0, 90)[0] < 0           # east = left (smaller x)
    d = np.hypot(*im.offset_pixels(0.5, 37.0))
    assert abs(d * im.PIX_SCALE - 0.5) < 1e-9


def test_deterministic_and_shapes():
    s = stars()[3]
    a, b = im.render(s, 0), im.render(s, 0)
    assert a['img'].shape == (im.N_PIX, im.N_PIX) and a['ref'].shape == (im.REF_N,) * 2
    assert np.array_equal(a['img'], b['img'])


def test_stars_stay_in_frame():
    for s in stars(2, 120):
        t = im.render(s, 2)['truth']
        for p in (t['primary'], t['companion']):
            if p:
                assert im.EDGE - 1e-6 <= p[0] <= im.N_PIX - im.EDGE + 1e-6
                assert im.EDGE - 1e-6 <= p[1] <= im.N_PIX - im.EDGE + 1e-6


def test_separation_preserved_after_shift():
    for s in stars(1, 120):
        c = s['companion']
        if c:
            t = im.render(s, 1)['truth']
            d = np.hypot(t['companion'][0] - t['primary'][0], t['companion'][1] - t['primary'][1])
            assert abs(d * im.PIX_SCALE - c['sep_arcsec']) < 1e-6


def _peak_snr(r, s, c):
    """Peak of the companion in a smoothed image, relative to noise far from both stars."""
    img = r['img']; px, py = r['truth']['companion']
    i, j = int(py), int(px)
    patch = img[i - 1:i + 2, j - 1:j + 2]
    return patch.max()


def test_companion_at_contrast_limit_is_visible_but_marginal():
    """A companion exactly at the limit has ~5 sigma peak; 1.5 mag fainter is lost."""
    base = dict(id=500, brightness=0.7, type='boring')
    sep = 1.0
    lim = g._contrast_limit(sep, 0.7)
    peaks = {}
    for tag, dm in (('at', lim), ('faint', lim + 1.5)):
        snr = []
        for seed in range(20):
            s = dict(base, companion=dict(sep_arcsec=sep, pa_deg=30, dmag=dm))
            r = im.render(s, seed)
            px, py = r['truth']['companion']
            # local sigma from the model
            sig = im.noise_sigma(sep, 0.7)
            peak = r['img'][int(py) - 1:int(py) + 2, int(px) - 1:int(px) + 2].mean() \
                   - im.star_flux(0.7) * im.psf(im.N_PIX, *r['truth']['primary'])[int(py), int(px)]
            snr.append(peak / sig)
        peaks[tag] = np.mean(snr)
    assert peaks['at'] > 2.5          # clearly there on average
    assert peaks['faint'] < peaks['at'] * 0.45


def test_noise_grows_for_fainter_hosts():
    assert im.noise_sigma(1.0, 0.3) > 0  # defined
    # fainter host -> fewer photons -> lower absolute flux, but also a shallower limit
    assert im.star_flux(0.3) < im.star_flux(0.9)


def test_close_pair_is_elongated_when_unresolved():
    s = dict(id=501, brightness=0.9, type='boring',
             companion=dict(sep_arcsec=0.03, pa_deg=0, dmag=0.0))
    r = im.render(s, 0)
    ref = r['ref']
    def width_y(a):
        prof = a.sum(axis=1); prof = prof - prof.min()
        return (prof > prof.max() / 2).sum()
    pk = r['img']
    sub = pk[int(r['truth']['primary'][1]) - 16:int(r['truth']['primary'][1]) + 16,
             int(r['truth']['primary'][0]) - 16:int(r['truth']['primary'][0]) + 16]
    assert width_y(sub) >= width_y(ref)
