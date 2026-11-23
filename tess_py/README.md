# tess_py — Python for the TESS game

`tess.html` is a single self-contained page. The Python it runs lives here as
ordinary modules and is **embedded** into `tess.html` by a build script.

## Workflow
1. Edit the `.py` files in this folder (never the embedded block in `tess.html`).
2. `python3 tess_py/build.py` — refreshes the embedded copy (`--check` just verifies).
3. `python3 -m pytest tess_py/tests` — includes a test that fails if the embedded
   copy is stale.

Run any module on its own, e.g. `python3 tess_py/generate.py --seed 0 --debug`
or `python3 tess_py/rv.py --seed 0 --star 24 --debug`.

## Modules
| file | role |
|---|---|
| `stars.py` | spectral-type table, magnitudes, bolometric correction, type label from Teff/log g |
| `generate.py` | star field, spectral types, companions, distances/parallax, light curves (ground truth); `debug=True` forces one star per scenario |
| `analysis.py` | phase folding, BLS box fit, period search (pure numpy) |
| `plots.py` | Matplotlib figures |
| `plot_utils.py` | palette, axis styling, PNG → base64 |
| `imaging.py` | Phase 2 synthetic images: PSF + noise tied to the contrast limit, reference star |
| `rv.py` | Phase 3 reconnaissance RVs: true RV model, fiber blending, CCFs, schedule → times, fit, m sin i, recon stellar values, ephemeris uncertainty |
| `bridge.py` | the only module the page calls; JSON in/out, per-star cache |
| `build.py` | embeds the modules into `tess.html` |
