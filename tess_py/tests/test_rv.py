import json
import numpy as np
import pytest
import generate as g
import rv


def debug_field(seed=0):
    S = json.loads(g.generate_starfield(seed, 25, debug=True))
    return {s.get('scenario'): s for s in S}


# ── orbits ──────────────────────────────────────────────────────────────────
def test_semi_amplitude_jupiter_and_hot_jupiter():
    assert abs(rv.semi_amplitude(365.25 * 11.86, 1.0, rv.MJUP / rv.MSUN) - 0.01247) < 3e-4
    assert abs(rv.semi_amplitude(365.25, 1.0, rv.MJUP / rv.MSUN) - 0.02843) < 3e-4


@pytest.mark.parametrize('m2', [0.0003, 0.001, 0.01, 0.1, 0.5, 1.0])
def test_msini_inverts_semi_amplitude(m2):
    K = rv.semi_amplitude(3.3, 1.1, m2)
    assert abs(rv.msini(K, 3.3, 1.1) - m2) < 1e-4 * max(m2, 0.01)


def test_fit_sine_recovers_parameters():
    t = np.array([57.9, 61.3, 66.0, 70.2, 80.7])
    P, T0, gam, K = 3.7, 1.2, -12.0, 4.5
    v = gam - K * np.sin(2 * np.pi * (t - T0) / P)
    f = rv.fit_sine(t, v, np.full(5, 0.2), P, T0)
    assert abs(f['gamma'] - gam) < 1e-9 and abs(f['K'] - K) < 1e-9 and f['chi2'] < 1e-12


# ── calendar ────────────────────────────────────────────────────────────────
def test_night_labels():
    assert rv.night_label(0) == 'Month 1, Day 1'
    assert rv.night_label(30) == 'Month 2, Day 1'
    assert rv.night_label(90) == 'Month 3, Day 30'
    with pytest.raises(ValueError):
        rv.night_label(91)


def test_obs_times_inside_windows_and_pairs_separated():
    nights = [3, 3, 10, 40, 90]
    t = rv.obs_times(0, 5, nights)
    assert np.array_equal(t, rv.obs_times(0, 5, nights))          # deterministic
    for n, tt in zip(sorted(nights), t):
        a, b = rv.night_window(n)
        assert a <= tt <= b
    assert t[1] - t[0] >= rv.WIN_LEN / 2


@pytest.mark.parametrize('bad', [[0, 1, 2, 3], [0, 0, 0, 1, 2], [0, 1, 2, 3, 91]])
def test_schedule_rules(bad):
    with pytest.raises(ValueError):
        rv.validate_nights(bad)


# ── fiber and blends ────────────────────────────────────────────────────────
def test_fiber_coupling_matches_design_table():
    assert rv.fiber_coupling(0.0) == pytest.approx(1.0)
    assert rv.fiber_coupling(1.0) == pytest.approx(0.58, abs=0.04)
    assert rv.fiber_coupling(2.0) == pytest.approx(0.08, abs=0.03)
    assert rv.fiber_coupling(3.5) < 0.005


def _two(light, dv, fwhm=12.0):
    return [rv._comp('target', 1.0, 5800, 4.4, fwhm, lambda t: 0.0 * np.asarray(t)),
            rv._comp('blend', light, 5800, 4.4, fwhm, lambda t: dv + 0.0 * np.asarray(t))]


def test_blend_pull_follows_ccf_formula():
    """shift ~ f dv exp(-dv^2 / 4 sigma^2) for a faint blend (design notes)."""
    for dv in (5.0, 10.0, 20.0, 40.0):
        f = 0.05
        comps = _two(f / (1 - f), dv)
        sig = 12.0 / 2.3548
        expect = f * dv * np.exp(-dv ** 2 / (4 * sig ** 2))
        assert rv.fit_ccf_centre(comps, 0.0) == pytest.approx(expect, abs=0.03)


def test_overlapping_equal_twin_halves_the_shift():
    comps = _two(1.0, 1.0)
    assert rv.fit_ccf_centre(comps, 0.0) == pytest.approx(0.5, abs=0.05)


# ── scenarios ───────────────────────────────────────────────────────────────
NIGHTS = [0, 6, 13, 27, 45]


def test_on_target_eb_is_kilometres_per_second_and_stellar_mass():
    s = debug_field()['eb']
    r = rv.observe(s, 0, NIGHTS)
    tr = r['system']['truth']
    c = tr['companion']
    f = rv.fit_sine(r['times'], r['rv'], r['err'], c['period'], c['t0'])
    assert abs(f['K'] - tr['K1']) < 0.05 * tr['K1']
    assert rv.msini(f['K'], c['period'], s['mass']) > 0.08         # a star, not a planet


def test_faint_background_eb_leaves_rv_flat():
    s = debug_field()['beb_faint']
    r = rv.observe(s, 0, NIGHTS)
    assert np.std(r['rv']) < 0.5


def test_bright_background_eb_puts_light_in_the_fiber():
    s = debug_field()['beb_bright']
    tr = rv.system(s, 0)['truth']['blend']
    assert tr['light_in_fiber'] > 0.1


def test_giant_host_makes_a_stellar_companion():
    s = debug_field()['giant_transit']
    assert rv.system(s, 0)['truth']['companion']['kind'] == 'star'


def test_ccf_shape_and_continuum():
    s = debug_field()['quiet']
    r = rv.observe(s, 0, NIGHTS)
    assert r['ccfs'].shape == (5, r['v'].size)
    assert abs(np.median(r['ccfs'][:, :20]) - 1.0) < 0.02         # continuum at the edges
    assert r['ccfs'].min() < 0.7                                    # the target's dip


def test_recon_values_and_luminosity_class():
    F = debug_field()
    q = F['quiet']
    rq = rv.recon(q, 0, rv.system(q, 0)['comps'])
    assert abs(rq['teff'] - q['teff']) < 400
    assert abs(rq['radius'] - q['radius']) < 0.25 * q['radius']
    assert rq['spt'].endswith(' V')
    gi = F['giant_transit']
    rg = rv.recon(gi, 0, rv.system(gi, 0)['comps'])
    assert rg['spt'].endswith(' III') and rg['radius'] > 5


def test_companion_light_inflates_derived_radius():
    s = debug_field()['planet_bound']
    rc = rv.recon(s, 0, rv.system(s, 0)['comps'])
    assert rc['radius'] > s['radius']


# ── ephemeris ───────────────────────────────────────────────────────────────
def test_ephemeris_contains_truth():
    s = debug_field()['clean_planet']
    t, f, sig, p = g.gen_flux(s['id'], s['type'], s['brightness'], 0)
    e = rv.ephemeris(t, f, sig, p['period'], p['t0'] + 0.05)       # sloppy click
    off = (e['t0'] - p['t0']) / p['period']
    off = (off - round(off)) * p['period']
    assert abs(off) < max(5 * e['sig_t0'], 0.02)
    assert e['sig_t0'] > 0 and e['sig_p'] > 0
    assert rv.phase_sigma(e['t0'] + 100, e) > rv.phase_sigma(e['t0'], e)
