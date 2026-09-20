"""Figure S2 - Bootstrap lag signatures across all 13 retained biomarkers.

Scientific purpose:
    Figure 1 collapses each biomarker's 7-lag curve to (peak |SHAP|, R², n).
    Figure 3 zooms into the 4 cluster-differentiating biomarkers. This
    supplementary figure shows the full 7-lag profile for every retained
    biomarker so readers can see (a) which biomarkers carry monotonic
    lag-response signal, (b) which ones are dominated by acute (lag 0)
    or delayed (lag 30) effects, and (c) the bootstrap stability of each
    curve (50 replicates, 95% percentile CI).

Layout:
    4 × 4 grid (14 used, 2 blank) in organ-system order so that cardiovascular,
    haematological, body-composition etc. cluster visually. Each panel shows
    mean |SHAP| with a CI ribbon on the left y-axis and peak-lag annotation.
    Panel border colour encodes organ system. Figure is 180 mm × ~200 mm.

Source data:
    mcd_outputs/stage1/{biomarker}/lag_response_summary.csv
    (7 rows × {lag_day, mean_abs_shap, ci_lower, ci_upper, pct_positive})
"""
from __future__ import annotations

import os
from mcd_pipeline import config as _pipeline_config
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _data import FOLD_R2, N_SAMPLES, RETAINED
from _lancet_style import (
    BIOMARKER_PRETTY, BIOMARKER_SYSTEM, MM, ORGAN_COLOURS, ORGAN_PRETTY,
    apply_style, panel_label,
)
from _lancet_style import ARTWORK_MODE, midline_tick_labels, md  # house style (2026-09-04)
from _lancet_style import save_at_width

STAGE1_DIR = Path(str(_pipeline_config.OUTPUT_ROOT.parent / _pipeline_config.OUTPUT_ROOT.name / "stage1"))
# Artwork runs write elsewhere so submission files do not overwrite the
# working previews. Set RP2_FIG_OUT to redirect.
OUT_DIR = Path(os.environ.get("RP2_FIG_OUT",
                              str(Path(__file__).resolve().parent)))

# Ordering: group by organ system so panels cluster visually
ORGAN_ORDER = [
    "cardiovascular",   # 3 biomarkers
    "renal",            # 1
    "inflammatory",     # 2 (haematological)
    "immunological",    # 2
    "hepatic",          # 1
    "lipid",            # 1
    "body_composition", # 4
    "metabolic",        # 0 (all dropped -- included for key)
]

# Build panel sequence in organ order
def panel_sequence() -> list[str]:
    buckets: dict[str, list[str]] = {k: [] for k in ORGAN_ORDER}
    for bm in RETAINED:
        buckets[BIOMARKER_SYSTEM[bm]].append(bm)
    seq: list[str] = []
    for org in ORGAN_ORDER:
        seq.extend(sorted(buckets[org]))
    return seq


def load_lag_summary(bm: str) -> pd.DataFrame:
    return pd.read_csv(STAGE1_DIR / bm / "lag_response_summary.csv")


def build_figure():
    apply_style(base_font_size=7.8)

    panels = panel_sequence()
    # Retained biomarker count varies by primary exposure (14 under ERA5, 13 under ERA5-Land).
    n_panels = len(panels)

    n_rows, n_cols = 4, 4
    fig_w_in = 180 * MM
    fig_h_in = 235 * MM
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    gs = fig.add_gridspec(
        nrows=n_rows, ncols=n_cols,
        left=0.085, right=0.975, top=0.885, bottom=0.06,
        hspace=1.10, wspace=0.50,
    )

    axes: list[plt.Axes] = []
    for i, bm in enumerate(panels):
        r, c = divmod(i, n_cols)
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)

        df = load_lag_summary(bm)
        lags = df["lag_day"].to_numpy(dtype=float)
        mu = df["mean_abs_shap"].to_numpy()
        lo = df["ci_lower"].to_numpy()
        hi = df["ci_upper"].to_numpy()
        pp = df["pct_positive"].to_numpy()

        organ = BIOMARKER_SYSTEM[bm]
        col = ORGAN_COLOURS[organ]

        # CI ribbon + mean line
        ax.fill_between(lags, lo, hi, color=col, alpha=0.22, linewidth=0, zorder=2)
        ax.plot(lags, mu, color=col, linewidth=1.3, zorder=3)
        # Directional markers: hollow if pct_positive < 0.5, filled if > 0.5, grey if ~0.5
        for x, y, p in zip(lags, mu, pp):
            if p >= 0.55:
                face = col
                edge = col
            elif p <= 0.45:
                face = "white"
                edge = col
            else:
                face = "#BBBBBB"
                edge = col
            ax.scatter(x, y, s=22, facecolor=face, edgecolor=edge, linewidth=0.9,
                       zorder=5)

        # Peak lag annotation
        peak_idx = int(np.argmax(mu))
        peak_x = lags[peak_idx]
        ymax_axis = max(hi) * 1.35
        ymin_axis = 0 if min(lo) >= 0 else min(lo) * 1.1
        ax.axvspan(peak_x - 1.0, peak_x + 1.0, color=col, alpha=0.08, zorder=1)

        # Panel header: biomarker on top line, n + R² on subtitle line
        mean_r2 = float(np.mean(FOLD_R2[bm]))
        n = N_SAMPLES[bm]
        pretty = BIOMARKER_PRETTY[bm]
        ax.text(
            0.0, 1.19, pretty,
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.0, color=col, fontweight="semibold",
        )
        ax.text(
            0.0, 1.05,
            f"n={n:,}   R²={mean_r2:.2f}   peak {int(peak_x)}d",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.4, color="#555555",
        )

        # Axes cosmetics
        ax.set_xlim(-1.5, 32)
        ax.set_ylim(ymin_axis, ymax_axis)
        ax.set_xticks([0, 7, 14, 21, 30])
        ax.tick_params(axis="x", labelsize=6.3)
        ax.tick_params(axis="y", labelsize=6.3)
        ax.grid(axis="y", color="#EFEFEF", linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)
        # coloured panel accent via spine bottom + left
        ax.spines["bottom"].set_color("#444444")
        ax.spines["left"].set_color("#444444")

        # Axis labels only on perimeter
        if r == n_rows - 1 or i >= 12:  # bottom row
            ax.set_xlabel("Lag (days)", fontsize=7.0)
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel("mean |SHAP|", fontsize=7.0)

    # Hide the 2 unused axes (positions 14 and 15)
    # We only created 14 - but we also want the last row's bottom-most two
    # cells to carry the legend and the narrative key. Create placeholder axes
    # in those cells.
    legend_ax = fig.add_subplot(gs[3, 2])
    key_ax = fig.add_subplot(gs[3, 3])
    for _ax in (legend_ax, key_ax):
        _ax.set_xticks([])
        _ax.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            _ax.spines[side].set_visible(False)

    # ---- Organ-system legend (bottom-left of the two spare cells) ----
    legend_ax.text(0.0, 1.00, "Organ system", fontsize=8.0, fontweight="bold",
                   transform=legend_ax.transAxes, va="top")
    y = 0.88
    # Only list systems that actually appear among the panels drawn, so the key
    # cannot advertise an organ system whose biomarkers are no longer retained.
    _systems_present = {BIOMARKER_SYSTEM[bm] for bm in panels}
    for organ in [o for o in ORGAN_ORDER if o in _systems_present]:
        legend_ax.add_patch(plt.Rectangle(
            (0.03, y - 0.03), 0.10, 0.06,
            facecolor=ORGAN_COLOURS[organ], edgecolor="none",
            transform=legend_ax.transAxes, clip_on=False,
        ))
        label = ORGAN_PRETTY[organ]
        legend_ax.text(
            0.18, y, label,
            fontsize=7.1, color="#222222",
            transform=legend_ax.transAxes, va="center",
        )
        y -= 0.11
    # Directional marker key
    legend_ax.text(
        0.0, y - 0.02, "Markers", fontsize=7.4, fontweight="semibold",
        transform=legend_ax.transAxes, va="top",
    )
    y -= 0.12
    # Filled circle
    legend_ax.scatter(0.07, y, s=32, facecolor="#555555", edgecolor="#555555",
                      linewidth=1.0, transform=legend_ax.transAxes, clip_on=False)
    legend_ax.text(0.18, y, "temp ↑ biomarker (>55%)",
                   transform=legend_ax.transAxes, fontsize=6.7, va="center")
    y -= 0.09
    legend_ax.scatter(0.07, y, s=32, facecolor="#BBBBBB", edgecolor="#555555",
                      linewidth=1.0, transform=legend_ax.transAxes, clip_on=False)
    legend_ax.text(0.18, y, "mixed direction (45-55%)",
                   transform=legend_ax.transAxes, fontsize=6.7, va="center")
    y -= 0.09
    legend_ax.scatter(0.07, y, s=32, facecolor="white", edgecolor="#555555",
                      linewidth=1.0, transform=legend_ax.transAxes, clip_on=False)
    legend_ax.text(0.18, y, "temp ↓ biomarker (<45%)",
                   transform=legend_ax.transAxes, fontsize=6.7, va="center")

    # ---- Narrative key (bottom-right spare cell) ----
    key_ax.text(0.0, 1.00, "Takeaways", fontsize=8.0, fontweight="bold",
                transform=key_ax.transAxes, va="top")
    # Counted from the panels actually drawn rather than asserted, so the
    # sentence cannot drift when the retained panel changes.
    _peaks = []
    for bm in panels:
        _df = load_lag_summary(bm)
        _peaks.append(int(_df["lag_day"].to_numpy()[
            int(np.argmax(_df["mean_abs_shap"].to_numpy()))
        ]))
    _n_edge = sum(1 for pk in _peaks if pk <= 3 or pk >= 21)
    bullets = [
        f"• {_n_edge} of {len(panels)} biomarkers peak at acute (0-3 d) or delayed (21-30 d) lags.",
        "• Cardiovascular markers show the most symmetrical responses with",
        "  SHAP balanced across positive and negative directions.",
        "• Haematological markers (Hct, Hb) are dominated by lag 30, consistent",
        "  with chronic haemoconcentration over sub-acute heat exposures.",
        "• Body-composition markers track medium-horizon exposure (7-14 d).",
        "• CI ribbons: 50 bootstrap replicates, 95% percentile interval.",
    ]
    y = 0.88
    for line in bullets:
        key_ax.text(
            0.0, y, line,
            fontsize=6.8, color="#222222",
            transform=key_ax.transAxes, va="top",
        )
        y -= 0.095

    # ---- Panel letters on the first panel only (sub-panels not lettered individually) ----
    panel_label(axes[0], "a", x=-0.30, y=1.18, fontsize=10)

    # ---- Titles ----
    if not ARTWORK_MODE:
        fig.text(
            0.02, 0.978,
            f"Figure S2  |  Bootstrap lag signatures for all {len(panels)} retained biomarkers",
            fontsize=10.0, fontweight="bold", ha="left", va="top",
        )
    fig.text(
        0.02, 0.955,
        "Mean |SHAP| of temperature at each lag (0, 1, 3, 7, 14, 21, 30 d) from interventional TreeSHAP, 50 bootstrap replicates.",
        fontsize=7.1, color="#444444", ha="left", va="top",
    )
    fig.text(
        0.02, 0.940,
        "Ribbon = 95% percentile CI. Marker fill encodes directional asymmetry (pct_positive: fraction of patient-visits with SHAP > 0).",
        fontsize=7.1, color="#444444", ha="left", va="top",
    )

    # ---- Save ----
    svg_path = OUT_DIR / "figS2_lag_signatures_all_biomarkers.svg"
    png_path = OUT_DIR / "figS2_lag_signatures_all_biomarkers.png"
    midline_tick_labels(fig)
    save_at_width(fig, OUT_DIR, "figS2_lag_signatures_all_biomarkers", span="double")
    plt.close(fig)
    return svg_path, png_path


if __name__ == "__main__":
    svg, png = build_figure()
    print(f"Saved: {svg}")
    print(f"Saved: {png}")
