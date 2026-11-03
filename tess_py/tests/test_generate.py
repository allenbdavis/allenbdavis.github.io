import json
import numpy as np
import generate as g


def field(seed=0, n=25):
    return json.loads(g.generate_starfield(seed, n))


def test_deterministic():
    assert g.generate_starfield(3, 30) == g.generate_starfield(3, 30)
    assert g.generate_starfield(3, 30) != g.generate_starfield(4, 30)


def test_at_least_one_blend_eb_even_in_small_fields():
    for n in (5, 10, 25, 100):
        assert any(s['type'] == 'blend_eb' for s in field(0, n)), n


def test_first_blend_is_detectable_for_debugging():
    blends = [s for s in field(0, 25) if s['type'] == 'blend_eb']
    assert any(s['companion']['detectable'] for s in blends)


def test_companion_fields_and_ranges():
    for s in field(1, 400):
        assert 25 <= s['distance_pc'] <= 800
        assert s['parallax_err_mas'] > 0
        c = s['companion']
        if c:
            assert g.SEP_MIN <= c['sep_arcsec'] <= g.SEP_MAX
            assert 0 <= c['pa_deg'] < 360 and c['dmag'] >= 0
            assert c['is_eb'] == (s['type'] == 'blend_eb')
            if c['bound']:
                assert c['dist_pc'] == s['distance_pc']
            else:
                assert c['dist_pc'] > s['distance_pc']      # background


def test_parallax_matches_distance_within_error():
    S = field(2, 400)
    z = [(s['parallax_mas'] - 1000 / s['distance_pc']) / s['parallax_err_mas'] for s in S]
    assert abs(np.mean(z)) < 0.2 and 0.8 < np.std(z) < 1.2


def test_detectability_rules():
    assert not g._detectable(0.02, 0.0, 1.0)      # inside the resolution limit
    assert g._detectable(1.0, 2.0, 1.0)           # bright, wide, modest contrast
    assert not g._detectable(1.0, 6.5, 1.0)       # too faint at any separation


def test_bound_companion_dilutes_transit_depth():
    S = field(0, 100)
    for s in S:
        c = s['companion']
        if s['type'] == 'transit' and c and c['bound']:
            _, _, _, p = g.gen_flux(s['id'], s['type'], s['brightness'], 0)
            expect = 1 / (1 + 10 ** (-0.4 * c['dmag']))
            assert abs(p['dilution'] - expect) < 1e-12
            assert abs(p['depth_obs'] - p['depth'] * expect) < 1e-12
            return
    raise AssertionError('no transit host with a companion in this field')


def test_blend_eb_looks_like_a_shallow_transit():
    for s in field(0, 100):
        if s['type'] == 'blend_eb':
            _, _, _, p = g.gen_flux(s['id'], s['type'], s['brightness'], 0)
            assert 0.003 <= p['depth_obs'] <= 0.021       # planet-like after dilution
            assert p['depth'] > p['depth_obs']            # intrinsically deeper


def test_dilution_does_not_touch_noise():
    """Same star, same seed: noise realisation is identical with or without dilution."""
    s = next(s for s in field(0, 25) if s['type'] == 'boring' and s['companion'] is None)
    _, f1, _, _ = g.gen_flux(s['id'], 'boring', s['brightness'], 0)
    g._truth[s['id']] = dict(dilution=0.4)
    _, f2, _, _ = g.gen_flux(s['id'], 'boring', s['brightness'], 0)
    assert np.array_equal(f1, f2)       # boring star has no signal to dilute
    g._truth.pop(s['id'])
