"""Figure 1 - Multi-system temperature lag-response signatures and model adequacy.

Spec (manuscript v1, Section S9):
  (A) Mean |SHAP| for the temperature feature at each of seven lag windows
      (0, 1, 3, 7, 14, 21, 30 days) for the 13 retained biomarkers, grouped
      by organ system. Row-normalised; colour intensity encodes magnitude.
  (B) Biomarker-level cross-validated R² (5-fold GroupKFold) with 95% CI
      computed from fold variance (mean ± 1·96 × SE). Dashed line marks
      the R²≥0 retention threshold.

Outputs:
  fig01_lag_signatures_and_r2.svg   (Figma-editable - text preserved)
  fig01_lag_signatures_and_r2.png   (300 dpi preview)
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from _data import (
    FOLD_R2, LAG_DAYS, LAG_MEAN_ABS_SHAP, N_SAMPLES, RETAINED, ALL_BIOMARKERS,
    RETENTION_CRITERION_LABEL, RETENTION_THRESHOLD,
)
from _lancet_style import (
    ARTWORK_MODE, md, in_figure_note, lag_tick_labels, LAG_FOOTNOTE,
    category_label_colour, write_caption, check_width, midline_tick_labels,
    BIOMARKER_PRETTY, BIOMARKER_SYSTEM, MM, ORGAN_COLOURS, ORGAN_PRETTY,
    SHAP_CMAP, apply_style, panel_label,
)
from _lancet_style import save_at_width

# Artwork runs write elsewhere so submission files do not overwrite the
# working previews. Set RP2_FIG_OUT to redirect.
OUT_DIR = Path(os.environ.get("RP2_FIG_OUT",
                              str(Path(__file__).resolve().parent)))
OUT_DIR.mkdir(parents=True, exist_ok=True)


def order_retained_by_system() -> list[str]:
    system_order = [
        "renal", "cardiovascular", "immunological",
        "body_composition", "hepatic", "lipid", "inflammatory",
    ]
    ordered = []
    for sys in system_order:
        bms = [b for b in RETAINED if BIOMARKER_SYSTEM.get(b) == sys]
        # Within system, preserve canonical order from _data.RETAINED list
        ordered.extend([b for b in RETAINED if b in bms])
    return ordered


def compute_r2_ci(fold_r2: list[float]) -> tuple[float, float, float]:
    arr = np.array(fold_r2)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def build_figure():
    order = order_retained_by_system()
    lag_mat = np.array([LAG_MEAN_ABS_SHAP[b] for b in order])
    # Per-row min-max scaling: each biomarker's temporal profile spans the full
    # colour range, so the *shape* (which lag matters most) is legible. Dividing
    # only by the row max (previous approach) left every cell in the top of the
    # ramp because within-row minima sit at 0.3-0.8 of the max, not near zero.
    # Absolute magnitude is carried by panel (b); panel (a) shows relative shape.
    row_min = lag_mat.min(axis=1, keepdims=True)
    row_max = lag_mat.max(axis=1, keepdims=True)
    lag_norm = (lag_mat - row_min) / (row_max - row_min)

    apply_style(base_font_size=8.0)

    # 180 mm double-column
    fig_w_in = 180 * MM
    fig_h_in = 135 * MM
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # GridSpec: heatmap | forest | colourbar
    gs = fig.add_gridspec(
        nrows=1, ncols=3,
        width_ratios=[1.00, 0.90, 0.035],
        left=0.22, right=0.965, top=0.83, bottom=0.11,
        wspace=0.45,
    )
    ax_hm = fig.add_subplot(gs[0, 0])
    ax_fp = fig.add_subplot(gs[0, 1], sharey=ax_hm)
    ax_cb = fig.add_subplot(gs[0, 2])

    # --- Panel A : heatmap ---
    im = ax_hm.imshow(
        lag_norm, aspect="auto", cmap=SHAP_CMAP, vmin=0, vmax=1,
        interpolation="nearest",
    )
    ax_hm.set_yticks(np.arange(len(order)))
    ax_hm.set_yticklabels([BIOMARKER_PRETTY[b] for b in order])
    ax_hm.tick_params(axis="y", pad=3)
    # Colour each tick label by its organ system (removes need for side strips)
    for tick, bm in zip(ax_hm.get_yticklabels(), order):
        tick.set_color(category_label_colour(ORGAN_COLOURS[BIOMARKER_SYSTEM[bm]]))
        tick.set_fontweight("semibold")
    ax_hm.set_xticks(np.arange(len(LAG_DAYS)))
    ax_hm.set_xticklabels(lag_tick_labels(LAG_DAYS))
    ax_hm.set_xlabel("Temperature exposure lag (days)")
    ax_hm.set_title("Relative temperature impact across lag windows  (peak lag outlined)",
                    fontsize=8.2, pad=6, loc="center")

    # Peak lag marker: outline the lag window with the highest mean |SHAP|
    peak_j = lag_mat.argmax(axis=1)
    for i, j in enumerate(peak_j):
        ax_hm.add_patch(Rectangle(
            (j - 0.48, i - 0.48), 0.96, 0.96,
            fill=False, edgecolor="#111111", linewidth=1.0, zorder=5,
        ))

    # Organ-system strip on the left
    systems = [BIOMARKER_SYSTEM[b] for b in order]
    groups = []
    start = 0
    for i in range(1, len(systems) + 1):
        if i == len(systems) or systems[i] != systems[start]:
            groups.append((systems[start], start, i - 1))
            start = i

    # Small coloured strip immediately left of each biomarker row (system cue)
    from matplotlib.transforms import blended_transform_factory
    tform = blended_transform_factory(ax_hm.transAxes, ax_hm.transData)
    for sysname, i0, i1 in groups:
        col = ORGAN_COLOURS.get(sysname, "#999999")
        strip = Rectangle(
            xy=(-0.025, i0 - 0.45), width=0.018, height=(i1 - i0) + 0.9,
            facecolor=col, edgecolor="none", clip_on=False, zorder=6,
            transform=tform,
        )
        ax_hm.add_patch(strip)

    # Full box around heatmap
    for side in ("top", "right", "left", "bottom"):
        ax_hm.spines[side].set_visible(True)
        ax_hm.spines[side].set_color("#666666")
        ax_hm.spines[side].set_linewidth(0.5)
    ax_hm.tick_params(axis="both", length=2.5, width=0.5)

    # Colourbar for heatmap
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("Relative |SHAP| within biomarker\n(min–max scaled per row)", fontsize=7.0)
    cb.ax.tick_params(labelsize=6.8, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#666666")

    panel_label(ax_hm, "a", x=-0.52, y=1.06)

    # --- Panel B : R² forest plot ---
    y_positions = np.arange(len(order))

    x_max = 0.0
    for ypos, bm in zip(y_positions, order):
        mean_r2, ci_lo, ci_hi = compute_r2_ci(FOLD_R2[bm])
        col = ORGAN_COLOURS[BIOMARKER_SYSTEM[bm]]
        # CI interval
        ax_fp.hlines(ypos, ci_lo, ci_hi, color=col, linewidth=1.2, zorder=2, alpha=0.85)
        # caps
        ax_fp.vlines([ci_lo, ci_hi], ypos - 0.22, ypos + 0.22,
                     color=col, linewidth=0.8, zorder=2)
        # point
        ax_fp.scatter(mean_r2, ypos, s=28, color=col, edgecolor="white",
                      linewidth=0.7, zorder=4)
        # numeric label to the right
        ax_fp.text(
            ci_hi + 0.013, ypos, md(mean_r2, 3),
            va="center", ha="left", fontsize=6.8, color="#333333",
        )
        x_max = max(x_max, ci_hi)

    # Retention threshold
    ax_fp.axvline(RETENTION_THRESHOLD, color="#666666", linestyle="--",
                  linewidth=0.8, zorder=1)
    ax_fp.text(RETENTION_THRESHOLD - 0.004, len(order) / 2,
               f"{RETENTION_CRITERION_LABEL} retention threshold",
               fontsize=6.2, color="#666666", ha="right", va="center",
               style="italic", rotation=90)

    ax_fp.invert_yaxis()
    ax_fp.set_ylim(len(order) - 0.5, -0.9)
    ax_fp.set_xlim(-0.06, x_max * 1.20)
    ax_fp.set_xlabel("Cross-validated R²  (5-fold GroupKFold)")
    ax_fp.set_title("Model adequacy", fontsize=8.5, pad=6, loc="center")
    ax_fp.tick_params(axis="y", left=False, labelleft=False)
    ax_fp.grid(axis="x", color="#EFEFEF", linewidth=0.4, zorder=0)
    ax_fp.set_axisbelow(True)
    panel_label(ax_fp, "b", x=-0.04, y=1.06)

    # Bottom retention annotation
    n_adequate = len(RETAINED)
    n_total = len(ALL_BIOMARKERS)
    in_figure_note(
        ax_fp, 0.99, -0.17,
        f"{n_adequate} of {n_total} biomarkers met {RETENTION_CRITERION_LABEL} retention criterion"
        f"    {LAG_FOOTNOTE}",
        transform=ax_fp.transAxes, ha="right", va="top",
        fontsize=6.9, color="#333333", style="italic",
    )

    # Overall title -- suppressed in artwork mode; it belongs in the legend
    if not ARTWORK_MODE:
        fig.text(
            0.02, 0.97,
            "Figure 1  |  Multi-system temperature lag-response signatures and model adequacy",
            fontsize=9.8, fontweight="bold", ha="left", va="top",
        )

    # Save
    svg_path = OUT_DIR / "fig01_lag_signatures_and_r2.svg"
    png_path = OUT_DIR / "fig01_lag_signatures_and_r2.png"
    midline_tick_labels(fig)
    save_at_width(fig, OUT_DIR, "fig01_lag_signatures_and_r2", span="double")
    write_caption(
        OUT_DIR, "fig01_lag_signatures_and_r2",
        "Figure 1: Multi-system temperature lag-response signatures and model adequacy. "
        "(a) Heatmap of mean absolute SHAP values for temperature exposure at seven lag "
        "windows, six single days (0, 1, 3, 7, 14, 21 days) and a 30-day cumulative mean "
        "(daggered on the axis), shown for the "
        f"{len(RETAINED)} retained biomarkers and grouped by physiological domain; the side "
        "bar encodes physiological domain and shading within each row encodes row-normalised "
        "SHAP magnitude, so shading is comparable within a row but not between rows. "
        "The outlined cell in each row marks that biomarker's peak lag. "
        "(b) Cross-validated R-squared by biomarker, from 5-fold cross-validation grouped by "
        "participant, with 50-replicate bootstrap 95% confidence intervals; the dashed line "
        f"marks the {RETENTION_CRITERION_LABEL} retention threshold.",
        f"{len(RETAINED)} of {len(ALL_BIOMARKERS)} biomarkers met the "
        f"{RETENTION_CRITERION_LABEL} retention criterion. {LAG_FOOTNOTE}",
    )
    plt.close(fig)

    return svg_path, png_path


if __name__ == "__main__":
    svg, png = build_figure()
    print(f"Saved: {svg}")
    print(f"Saved: {png}")
