"""Fails if the Python embedded in tess.html has drifted from tess_py/*.py."""
import os, pytest
import build

HTML = os.path.abspath(build.HTML)


@pytest.mark.skipif(not os.path.exists(HTML), reason='tess.html not found')
def test_embedded_bundle_matches_sources():
    embedded = build.extract_bundle(open(HTML, encoding='utf-8').read())
    assert sorted(embedded) == sorted(build.MODULES)
    for name in build.MODULES:
        assert embedded[name] == build.read_module(name), \
            f'{name}.py differs from the copy in tess.html -- run: python3 tess_py/build.py'


def test_escaping_round_trips_awkward_text():
    nasty = "a = `b` + '${x}' + '${\\'`\\'}' \\n \\u2014 \\\\ end"
    assert build.js_unraw(build.js_raw(nasty)) == nasty
    assert '`' not in build.js_raw(nasty) or "${'`'}" in build.js_raw(nasty)


def test_refuses_script_terminator():
    with pytest.raises(ValueError):
        build.js_raw('x = "</script>"')
