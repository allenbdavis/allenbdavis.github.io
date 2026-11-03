import json
import numpy as np
import generate as g
import analysis as an


def test_fold_phase_range_and_transit_at_zero():
    t = np.linspace(0, 27, 1500)
    ph = an.fold_phase(t, 2.0, 5.0)
    assert ph.min() >= -0.5 and ph.max() <= 0.5
    assert abs(an.fold_phase(np.array([12.0]), 2.0, 5.0)[0]) < 1e-12


def test_search_recovers_injected_transit_period():
    S = json.loads(g.generate_starfield(0, 25))
    s = next(s for s in S if s['type'] == 'transit' and s['companion'] is None)
    t, flux, sigma, p = g.gen_flux(s['id'], 'transit', s['brightness'], 0)
    r = an.search_period(t, flux, sigma, p['t0'], p['period'] * 1.03)
    assert abs(r['period'] - p['period']) < 0.02 * p['period']
    assert r['snr'] > 5 and r['depth'] > 0


def test_refine_box_returns_positive_depth_for_a_real_dip():
    ph = np.linspace(-0.5, 0.5, 2000)
    flux = np.ones_like(ph); flux[np.abs(ph) < 0.03] -= 0.01
    base, depth = an.refine_box(ph, flux, 0.0, 0.03, 0.01)
    assert abs(base - 1.0) < 1e-9 and abs(depth - 0.01) < 1e-9
