# tess_py — Python for the TESS game

`tess.html` is a single self-contained page. The Python it runs lives here as
ordinary modules and is **embedded** into `tess.html` by a build script.

## Workflow
1. Edit the `.py` files in this folder (never the embedded block in `tess.html`).
2. `python3 tess_py/build.py` — refreshes the embedded copy (`--check` just verifies).
3. `python3 -m pytest tess_py/tests` — includes a test that fails if the embedded
   copy is stale.

Run any module on its own, e.g. `python3 tess_py/generate.py --seed 42`.

## Modules
| file | role |
|---|---|
| `generate.py` | star field, companions, parallax, light curves (ground truth) |
| `analysis.py` | phase folding, BLS box fit, period search (pure numpy) |
| `plots.py` | Matplotlib figures |
| `plot_utils.py` | palette, axis styling, PNG → base64 |
| `bridge.py` | the only module the page calls; JSON in/out, per-star cache |
| `build.py` | embeds the modules into `tess.html` |
