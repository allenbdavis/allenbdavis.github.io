import json
import numpy as np
import generate as g
import stars as st


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
        assert 5 <= s['distance_pc'] <= 3000
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


def test_debug_field_has_every_scenario():
    S = json.loads(g.generate_starfield(0, 25, debug=True))
    roles = {s.get('scenario'): s for s in S if s.get('scenario')}
    assert set(roles) == set(g.SCENARIOS)
    assert roles['giant_transit']['lum_class'] == 'III' and roles['giant_transit']['type'] == 'transit'
    assert roles['planet_bound']['companion']['bound']
    bg = roles['planet_bg']['companion']
    assert not bg['bound'] and not bg['is_eb']
    assert roles['beb_bright']['type'] == roles['beb_faint']['type'] == 'blend_eb'
    assert roles['beb_bright']['companion']['sep_arcsec'] < 1.0 < roles['beb_faint']['companion']['sep_arcsec']
    assert roles['quiet']['companion'] is None and roles['quiet']['type'] == 'boring'


def test_spectral_mix_is_fgk_heavy_with_a_few_giants():
    S = field(3, 400)
    letters = [s['spt'][0] for s in S]
    fgk = sum(l in 'FGK' for l in letters) / len(S)
    giants = sum(s['lum_class'] == 'III' for s in S) / len(S)
    assert fgk > 0.6
    assert 0.01 < giants < 0.2


def test_pulsators_are_hot_and_ebs_are_dwarfs():
    for s in field(4, 300):
        if s['type'] == 'pulsating':
            assert s['spt'] in g.PULSATOR_TYPES
        if s['type'] == 'eb':
            assert s['lum_class'] == 'V'


def test_distance_follows_from_primary_magnitude():
    """The primary's own magnitude (total minus companion light) and its
    absolute magnitude give back the stored distance."""
    for s in field(5, 200):
        if not 5 < s['distance_pc'] < 3000:
            continue                                   # clipped extremes
        c = s['companion']
        T1 = s['tmag'] + (2.5 * np.log10(1 + 10 ** (-0.4 * c['dmag'])) if c else 0.0)
        assert abs(st.distance_pc(T1, s['radius'], s['teff']) - s['distance_pc']) < 1e-6 * s['distance_pc']


def test_bound_separation_matches_orbit_and_distance():
    for s in field(6, 400):
        c = s['companion']
        if c and c['bound']:
            assert abs(c['a_au'] * c['proj'] / s['distance_pc'] - c['sep_arcsec']) < 1e-9
