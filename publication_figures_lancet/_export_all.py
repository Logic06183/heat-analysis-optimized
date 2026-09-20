"""Export every Lancet figure to editable SVG + 400 dpi PNG + print-ready PDF.

Each figure module builds its figure inside a `build_figure()` (or runs as a
script) and calls `fig.savefig(...)`. We monkeypatch `Figure.savefig` to capture
the Figure object on its first save, then re-emit all three formats into
FIGURES_FOR_MANUSCRIPT/ with a clean, journal-style stem (Figure_1, ...,
Figure_S1, ..., Figure_S2_DAG).

Run from the publication_figures_lancet/ dir with the standard env:
  RP2_PRIMARY_EXPOSURE=era5_land PYTHONPATH=<root>:. MPLCONFIGDIR=/tmp/mpl
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg

from _lancet_style import save_all_formats

HERE = Path(__file__).resolve().parent
OUT = HERE / "FIGURES_FOR_MANUSCRIPT"
OUT.mkdir(exist_ok=True)

# module name -> clean journal stem
FIGURES = [
    ("fig01_lag_signatures_and_r2",           "Figure_1"),
    ("fig02_cluster_profiles",                "Figure_2"),
    ("fig03_cluster_biomarker_signatures",    "Figure_3"),
    ("figS1_pca_diagnostic",                  "Figure_S1"),
    ("figS2_lag_signatures_all_biomarkers",   "Figure_S2"),
    ("figS3_sensitivity_analyses",            "Figure_S3"),
    ("figS4_interaction_heatmap",             "Figure_S4"),
    ("figS5_cohort_descriptive",              "Figure_S5"),
    ("figS6_climate_descriptive",             "Figure_S6"),
    ("figS7_cluster_stability",               "Figure_S7"),
    ("figS_DAG",                              "Figure_S2_DAG"),
]


def _capture_and_export(modname: str, stem: str) -> dict:
    captured = []
    orig = mpl.figure.Figure.savefig

    def cap(self, *a, **k):
        if not captured:
            captured.append(self)
        # suppress the module's own in-place writes during capture
        return None

    mpl.figure.Figure.savefig = cap
    try:
        mod = importlib.import_module(modname)
        mod = importlib.reload(mod)  # fresh build each run
        entry = None
        for fn in ("build_figure", "main", "make"):
            if hasattr(mod, fn):
                entry = getattr(mod, fn)
                break
        if entry is not None:
            entry()
        else:
            exec(compile(Path(mod.__file__).read_text(), mod.__file__, "exec"),
                 {"__name__": "__main__", "__file__": mod.__file__})
    finally:
        mpl.figure.Figure.savefig = orig

    if not captured:
        raise RuntimeError(f"{modname}: no figure captured")
    fig = captured[0]
    FigureCanvasAgg(fig)  # ensure a raster canvas for PNG
    paths = save_all_formats(fig, OUT, stem, dpi=400)
    mpl.pyplot.close(fig)
    return paths


if __name__ == "__main__":
    for modname, stem in FIGURES:
        try:
            paths = _capture_and_export(modname, stem)
            sizes = {k: Path(v).stat().st_size for k, v in paths.items()}
            print(f"OK  {stem:16s} <- {modname}   "
                  f"svg={sizes['svg']//1024}K png={sizes['png']//1024}K pdf={sizes['pdf']//1024}K")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {stem:16s} <- {modname}: {type(e).__name__}: {e}")
            sys.exit(1)
    print(f"\nAll formats written to {OUT}")
