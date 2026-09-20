"""Lancet Regional Health - Africa figure style.

Centralised matplotlib style + colour palettes so every figure in the submission
shares typography, weights, and semantic colours. Import as:

    from _lancet_style import apply_style, ORGAN_COLOURS, CLUSTER_COLOURS, MM
"""
from __future__ import annotations

import os as _os

import matplotlib as mpl
import matplotlib.pyplot as plt

# Physical sizing - Lancet accepts 88 mm (single col) and 180 mm (double col)
MM = 1.0 / 25.4  # mm -> inches

# Organ-system palette (8 systems, colour-blind safe, consistent across figures)
ORGAN_COLOURS = {
    "renal":            "#1F78B4",   # blue
    "cardiovascular":   "#E6550D",   # orange
    "immunological":    "#1B9E77",   # teal-green
    "body_composition": "#7F7F7F",   # neutral grey
    "hepatic":          "#DAA520",   # goldenrod
    "lipid":            "#8B5A2B",   # sienna
    "inflammatory":     "#B22246",   # deep crimson (CVD-nudged 2026-07: separates from immunological teal under deuteranopia)
    "metabolic":        "#6A3D9A",   # purple
}

ORGAN_PRETTY = {
    "renal":            "Renal",
    "cardiovascular":   "Cardiovascular",
    "immunological":    "Immunological",
    "body_composition": "Body composition",
    "hepatic":          "Hepatic",
    "lipid":            "Lipid",
    "inflammatory":     "Haematological",
    "metabolic":        "Metabolic",
}

# Biomarker -> organ system (only the 14 retained)
BIOMARKER_SYSTEM = {
    "creatinine":       "renal",
    "diastolic_bp":     "cardiovascular",
    "systolic_bp":      "cardiovascular",
    "heart_rate":       "cardiovascular",
    "viral_load":       "immunological",
    "cd4_count":        "immunological",
    "body_fat_percent": "body_composition",
    "waist_hip_ratio":  "body_composition",
    "bmi":              "body_composition",
    "hip_circumference":"body_composition",
    "albumin":          "hepatic",
    "ldl_cholesterol":  "lipid",
    "hematocrit":       "inflammatory",
    "hemoglobin":       "inflammatory",
}

BIOMARKER_PRETTY = {
    "creatinine":       "Creatinine",
    "diastolic_bp":     "Diastolic BP",
    "systolic_bp":      "Systolic BP",
    "heart_rate":       "Heart rate",
    "viral_load":       "Viral load",
    "cd4_count":        "CD4 count",
    "body_fat_percent": "Body fat %",
    "waist_hip_ratio":  "Waist-hip ratio",
    "bmi":              "BMI",
    "hip_circumference":"Hip circumference",
    "albumin":          "Albumin",
    "ldl_cholesterol":  "LDL-cholesterol",
    "hematocrit":       "Haematocrit",
    "hemoglobin":       "Haemoglobin",
}

# GMM cluster palette (under ERA5-Land primary)
# Cluster 0 = younger marginalised (47.8%)
# Cluster 1 = older employed (13.6%)
# Cluster 2 = HIV-burdened (38.6%) -- highlight colour
CLUSTER_COLOURS = {
    0: "#4E79A7",   # steel blue
    1: "#F28E2B",   # warm orange
    2: "#C94F7C",   # rose  -- HIV-burdened cluster, visually prominent
}
def _cluster_pct() -> dict[int, float]:
    """Cluster proportions for the run currently being plotted.

    Read from _data.CLUSTER_CHARACTERISTICS rather than frozen here, because
    hardcoding them is what left Figure 2's legend showing one run's
    proportions while every data-driven panel beside it showed another's.
    _data supplies these already mapped to canonical cluster ids (0 younger
    marginalised, 1 older employed, 2 high HIV burden) -- the raw GMM ids in
    cluster_profiles.csv are NOT in that order and must not be used directly.
    """
    try:
        from _data import CLUSTER_CHARACTERISTICS
        return {int(c): float(v["prevalence_pct"][0])
                for c, v in CLUSTER_CHARACTERISTICS.items()}
    except Exception:
        return {0: 47.1, 1: 13.4, 2: 39.5}


CLUSTER_PCT = _cluster_pct()

def _md1(v: float) -> str:
    """Midline decimal, 1 dp -- defined here because CLUSTER_LABELS is built
    above the general md() helper."""
    return f"{v:.1f}".replace(".", "\u00b7")


CLUSTER_LABELS = {
    0: f"Cluster 1  Younger marginalised ({_md1(CLUSTER_PCT[0])}%)",
    1: f"Cluster 2  Older, employed ({_md1(CLUSTER_PCT[1])}%)",
    2: f"Cluster 3  High HIV burden ({_md1(CLUSTER_PCT[2])}%)",
}
CLUSTER_SHORT = {
    0: "Younger marginalised",
    1: "Older, employed",
    2: "High HIV burden",
}

# Sequential heatmap for |SHAP| magnitudes (perceptually uniform, manuscript convention: warm = more impact)
SHAP_CMAP = "YlOrRd"
# Diverging for signed SHAP (negative vs positive contribution to biomarker)
SHAP_DIVERGING = "RdBu_r"

RETENTION_COLOUR = "#2A9D8F"    # green - "adequate" cue
INADEQUATE_COLOUR = "#B0B0B0"   # grey - below threshold
ANNOT_COLOUR = "#333333"
GRID_COLOUR = "#CFCFCF"


def apply_style(base_font_size: float = 8.0) -> None:
    """Apply Lancet-compatible rcParams. 8 pt is the standard figure font size."""
    mpl.rcParams.update({
        # Fonts - Arial-compatible stack; SVG will retain as editable text
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": base_font_size,
        "axes.titlesize": base_font_size + 1,
        "axes.labelsize": base_font_size,
        "xtick.labelsize": base_font_size - 1,
        "ytick.labelsize": base_font_size - 1,
        "legend.fontsize": base_font_size - 1,
        "figure.titlesize": base_font_size + 2,
        # Lines + spines
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,
        # No top/right spines by default
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Grid
        "axes.grid": False,
        "grid.color": GRID_COLOUR,
        "grid.linewidth": 0.4,
        # SVG - keep text as text (critical for Figma editing)
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # White background
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def panel_label(ax, letter: str, x: float = -0.10, y: float = 1.05, fontsize: float = 10):
    """Lancet-style panel label: bold lowercase letter, top-left."""
    ax.text(
        x, y, letter, transform=ax.transAxes,
        fontsize=fontsize, fontweight="bold", color="#111111",
        ha="left", va="bottom",
    )


def save_all_formats(fig, out_dir, stem: str, dpi: int = 400):
    """Save a figure as editable SVG + high-res PNG + print-ready PDF.

    - SVG: svg.fonttype='none' (set in apply_style) keeps text as editable
      text, so the file can be opened and hand-edited in Illustrator / Figma /
      Inkscape ("pencil"-editable).
    - PNG: 400 dpi raster preview, exceeds the Lancet 300 dpi floor with margin.
    - PDF: vector, fonttype 42 (embedded TrueType) for print submission.

    Returns dict of the three written paths.
    """
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": out_dir / f"{stem}.svg",
        "png": out_dir / f"{stem}.png",
        "pdf": out_dir / f"{stem}.pdf",
    }
    fig.savefig(paths["svg"], format="svg")
    fig.savefig(paths["png"], format="png", dpi=dpi)
    fig.savefig(paths["pdf"], format="pdf")
    return paths


# ---------------------------------------------------------------------------
# Submission mode, house style, and caption handling  (added 2026-09-04)
# ---------------------------------------------------------------------------
# The Lancet redraws every figure in-house and wants the legend as manuscript
# text; PLOS forbids captions embedded in the image file. So the in-figure
# "Figure N | ..." banner and any footnote belong OUT of the artwork we submit.
#
#   RP2_ARTWORK=1   -> submission artwork: no banner, no in-figure footnotes.
#   unset (default) -> working preview: banner and footnotes drawn, which is
#                      what you want when reading the figures yourself.
#
# Either way the caption text is written to <stem>.caption.txt so it can be
# pasted into the manuscript rather than retyped.
ARTWORK_MODE = _os.environ.get("RP2_ARTWORK", "") not in ("", "0", "false", "False")


def md(value, dp: int = 2, strip_leading_zero: bool = False) -> str:
    """Format a number in Lancet house style: midline decimal point (0·124).

    Lancet sets decimals midline throughout. Mixing "0.124" in a panel with
    "0·05" in a footnote -- which these figures did -- is a house-style error a
    copy-editor will catch, and it is invisible until someone looks for it.
    """
    if value is None:
        return ""
    txt = f"{value:.{dp}f}".replace(".", "\u00b7")
    if strip_leading_zero and txt.startswith("0\u00b7"):
        txt = txt[1:]
    return txt


def figure_banner(fig, text: str, **kw):
    """Draw the 'Figure N | title' banner only when NOT producing artwork.

    Returns the artist, or None in artwork mode, so callers can no-op safely.
    """
    if ARTWORK_MODE:
        return None
    kw.setdefault("fontsize", 10)
    kw.setdefault("fontweight", "bold")
    kw.setdefault("ha", "left")
    return fig.suptitle(text, **kw)


def in_figure_note(fig_or_ax, *args, **kw):
    """Draw an explanatory note inside the figure only in preview mode.

    Use for anything that is really caption text -- retention criteria, variance
    caveats, method asides. In artwork mode it is suppressed and should appear
    in the caption passed to write_caption() instead.
    """
    if ARTWORK_MODE:
        return None
    return fig_or_ax.text(*args, **kw)


# Lag windows: the last one is a 30-day cumulative mean, not a single day.
# Labelling it "30" on an axis of single-day lags invites the reader to see a
# seventh day, and to read a rise into 30 as a slow single-day response.
LAG_DAYS = [0, 1, 3, 7, 14, 21, 30]
LAG_CUMULATIVE = {30}


def lag_tick_labels(lags=None, mark: str = "\u2020") -> list[str]:
    """Tick labels for the lag axis, daggering any cumulative window."""
    lags = LAG_DAYS if lags is None else lags
    return [f"{d}{mark}" if d in LAG_CUMULATIVE else f"{d}" for d in lags]


LAG_FOOTNOTE = ("\u2020 30-day cumulative mean, not a single-day lag.")

# Category labels (biomarker names, cluster names) are printed in ink, not in
# their category colour: the adjacent colour bar / marker already encodes the
# category, coloured text is the one thing both Lancet and Nature advise
# against, and it is the first thing to fail in greyscale.
COLOUR_CATEGORY_LABELS = False
LABEL_INK = "#222222"


def category_label_colour(colour: str) -> str:
    """Colour for a category text label -- ink unless explicitly re-enabled."""
    return colour if COLOUR_CATEGORY_LABELS else LABEL_INK


def write_caption(out_dir, stem: str, caption: str, notes: str = "") -> "Path":
    """Write the caption/legend beside the figure for pasting into the paper."""
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    body = caption.strip()
    if notes:
        body += "\n\n" + notes.strip()
    path = out_dir / f"{stem}.caption.txt"
    path.write_text(body + "\n")
    return path


def check_width(fig, span: str = "double", tol_mm: float = 1.5) -> None:
    """Warn if the figure is not at a Lancet column width.

    savefig(bbox_inches='tight') -- set in apply_style -- silently changes the
    output width, so a figure sized to 180 mm can leave as 174 mm. Lancet
    redraws anyway, but a figure that is not at column width is a figure whose
    type sizes are not what you think they are after scaling.
    """
    target = {"single": 88.0, "double": 180.0}[span]
    got = fig.get_size_inches()[0] * 25.4
    if abs(got - target) > tol_mm:
        print(f"  ! width {got:.1f} mm vs Lancet {span}-column {target:.0f} mm "
              f"(bbox_inches='tight' will shift this again on save)")


def midline_tick_labels(fig) -> None:
    """Rewrite numeric tick labels in Lancet house style (0·25, not 0.25).

    Called after all plotting and before saving: it renders once to let
    matplotlib choose the ticks, then freezes those labels with midline
    decimals. Do not call before the axes are final -- it fixes the ticks.
    """
    from matplotlib.ticker import FixedFormatter, FixedLocator
    fig.canvas.draw()
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            labels = [t.get_text() for t in axis.get_ticklabels()]
            if not labels or not any("." in t for t in labels):
                continue
            locs = list(axis.get_ticklocs())
            if len(locs) != len(labels):
                continue
            axis.set_major_locator(FixedLocator(locs))
            axis.set_major_formatter(
                FixedFormatter([t.replace(".", "\u00b7") for t in labels]))


LANCET_WIDTH_MM = {"single": 88.0, "double": 180.0}


def save_at_width(fig, out_dir, stem: str, span: str = "double",
                  dpi: int = 300, pad_in: float = 0.05,
                  tol_mm: float = 0.4, min_font_pt: float = 5.0,
                  max_type_shrink: float = 0.95,
                  formats=("svg", "png", "pdf"), verbose: bool = True):
    """Save so the FINAL FILE is exactly a Lancet column wide.

    apply_style() sets savefig.bbox='tight', which crops to content -- so the
    file width is whatever the content happened to occupy, not the figsize you
    asked for. Every figure here was drifting: 176 to 221 mm against a 180 mm
    column. That matters because the journal scales the file to the column, and
    a figure 20 mm too wide has all its type silently reduced by 11%.

    Two levers, applied in that order:
      1. Scale the canvas. Cheap and lossless -- but it cannot fix a figure
         whose text alone is wider than the column, because point sizes do not
         shrink with the canvas. Chasing that case makes the figure worse.
      2. Scale every text object. Applied only when the canvas alone will not
         converge, floored at min_font_pt so nothing drops below legibility.

    Returns the written paths; reports the achieved width and any type scaling.
    """
    from pathlib import Path
    import matplotlib.text as _mtext
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_in = LANCET_WIDTH_MM[span] / 25.4
    content_target = target_in - 2 * pad_in

    texts = [t for t in fig.findobj(_mtext.Text) if t.get_text().strip()]
    base_sizes = [t.get_fontsize() for t in texts]
    font_scale = 1.0

    def measured_in():
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        return None if bb is None or bb.width <= 0 else bb.width

    best = (float("inf"), fig.get_size_inches().copy(), 1.0)

    for i in range(14):
        w_in = measured_in()
        if w_in is None:
            break
        err = abs(w_in - content_target)
        if err < best[0]:
            best = (err, fig.get_size_inches().copy(), font_scale)
        if err * 25.4 < tol_mm:
            break

        k = content_target / w_in
        if i < 6:
            # lever 1: canvas
            w, h = fig.get_size_inches()
            kk = max(0.8, min(1.25, k))
            fig.set_size_inches(w * kk, h * kk, forward=True)
        else:
            # lever 2: type. Only reached when the canvas alone stalls.
            # Never mangle typography to hit a number: a few percent is a
            # rounding accommodation, 15% is a different figure. Beyond the cap
            # the honest answer is that the layout is too wide, not the type.
            kk = max(0.97, min(1.0, k))
            if font_scale * kk < max_type_shrink:
                break
            if min(bs * font_scale * kk for bs in base_sizes) < min_font_pt:
                break
            font_scale *= kk
            for t, bs in zip(texts, base_sizes):
                t.set_fontsize(bs * font_scale)

    # fall back to the best attempt rather than leaving the last, worst one
    if measured_in() is None or abs(measured_in() - content_target) > best[0]:
        fig.set_size_inches(*best[1], forward=True)
        font_scale = best[2]
        for t, bs in zip(texts, base_sizes):
            t.set_fontsize(bs * font_scale)
        fig.canvas.draw()

    paths = {}
    for ext in formats:
        paths[ext] = out_dir / f"{stem}.{ext}"
        fig.savefig(paths[ext], format=ext, dpi=dpi,
                    bbox_inches="tight", pad_inches=pad_in)

    if verbose:
        final = None
        try:
            import subprocess, re as _re
            o = subprocess.run(["pdfinfo", str(paths["pdf"])],
                               capture_output=True, text=True).stdout
            m = _re.search(r"Page size:\s+([\d.]+) x", o)
            if m:
                final = float(m.group(1)) * 25.4 / 72
        except Exception:
            pass
        if final is None:
            final = (measured_in() + 2 * pad_in) * 25.4
        note = "" if font_scale == 1.0 else f", type x{font_scale:.2f}"
        delta = final - LANCET_WIDTH_MM[span]
        flag = ("" if abs(delta) < 1.0 else
                f"   ! {abs(delta):.0f} mm {'over' if delta > 0 else 'under'} "
                f"-- needs a layout change, not more scaling")
        print(f"  {stem}: {final:.1f} mm ({span} column{note}){flag}")
    return paths
