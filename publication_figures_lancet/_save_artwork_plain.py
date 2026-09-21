"""Render a figure script with a plain tight-bbox save.

The supplementary artwork set (figS1 to figS7) was produced before
`_lancet_style.save_at_width` began enforcing Lancet column widths, so those
files are plain 300 dpi tight-bbox saves and come out wider than 180 mm. Use
this wrapper to regenerate a supplementary figure in that same form, so a
regenerated figure matches its neighbours in artwork_for_submission/.

    RP2_ARTWORK=1 RP2_FIG_OUT=<dir> RP2_PRIMARY_EXPOSURE=era5_land \
        PYTHONPATH=..:. python3 _save_artwork_plain.py figS2_lag_signatures_all_biomarkers.py

Main-text figures (fig01 to fig03) use save_at_width directly and should not
be rendered through this wrapper.
"""
import pathlib
import runpy
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, ".")
import _lancet_style as ls  # noqa: E402


def _plain(fig, out_dir, stem, span="double", formats=("svg", "png", "pdf"), verbose=True):
    out_dir = pathlib.Path(out_dir)
    paths = []
    for ext in formats:
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        paths.append(p)
    if verbose:
        print("Saved:", *[str(p) for p in paths], sep="\n  ")
    return paths


ls.save_at_width = _plain
runpy.run_path(sys.argv[1], run_name="__main__")
