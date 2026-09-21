"""Figure 3 - Temperature SHAP signatures for the four cluster-differentiating biomarkers.

Scientific purpose (manuscript v1, Sections 3.4-3.5):
    The three GMM clusters separate primarily on four biomarkers whose lag-response
    profiles drive the composite heat-sensitivity score: haematocrit, viral load,
    haemoglobin and body fat %. This figure shows each biomarker's mean |SHAP|
    across all seven temperature lags with 95% bootstrap CI, directional
    asymmetry (fraction of SHAP values that are positive) and the peak lag at
    which temperature exerts its strongest influence.

Why not a classic beeswarm:
    Per-patient SHAP arrays sit behind a mount that intermittently blocks access;
    we therefore render a bootstrap-replicated lag spectrum which carries the same
    directional information (pct_positive replaces the swarm asymmetry) while
    preserving the 7-lag temporal structure that a single-lag beeswarm would lose.

Layout:
    2 × 2 panel grid, 180 mm double-column, 8 pt typography.
    Top strip per panel     - mean |SHAP| ± 95% CI ribbon, peak-lag band.
    Bottom strip per panel  - pct_positive diverging bar (centred at 0.5).
    Marker colour           - diverging RdBu_r on pct_positive.
    Annotation              - n_samples, model R², temperature-feature rank.

Outputs:
    fig03_cluster_biomarker_signatures.svg
    fig03_cluster_biomarker_signatures.png   (300 dpi)
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib import cm
from matplotlib.colors import Normalize

from _data import CLUSTER_BIOMARKER_LAG_SIGNATURE
from _lancet_style import (
    ARTWORK_MODE, md, in_figure_note, lag_tick_labels, LAG_FOOTNOTE,
    category_label_colour, write_caption, check_width, midline_tick_labels,
    BIOMARKER_PRETTY, BIOMARKER_SYSTEM, MM, ORGAN_COLOURS,
    SHAP_DIVERGING, apply_style, panel_label,
)
from _lancet_style import save_at_width

# Artwork runs write elsewhere so submission files do not overwrite the
# working previews. Set RP2_FIG_OUT to redirect.
OUT_DIR = Path(os.environ.get("RP2_FIG_OUT",
                              str(Path(__file__).resolve().parent)))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Panel order mirrors the paper's narrative: haematological (inflammatory),
# immunological, haematological, body composition.
PANEL_ORDER = ["hematocrit", "viral_load", "hemoglobin", "body_fat_percent"]


def build_figure():
    apply_style(base_font_size=8.0)

    fig_w_in = 180 * MM
    fig_h_in = 165 * MM
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # Two rows × two columns of panels; within each panel, a 2-row gridspec for
    # the |SHAP| ribbon (taller) and pct_positive strip (shorter).
    outer = fig.add_gridspec(
        nrows=2, ncols=2,
        left=0.085, right=0.965, top=0.83, bottom=0.07,
        hspace=0.62, wspace=0.32,
    )

    panel_letters = ["a", "b", "c", "d"]
    shap_norm = Normalize(vmin=0.30, vmax=0.70)
    cmap_dir = plt.get_cmap(SHAP_DIVERGING)

    # Pre-compute per-panel y limits (ribbon panel) using common logic
    panel_ylims = {}
    for bm in PANEL_ORDER:
        sig = CLUSTER_BIOMARKER_LAG_SIGNATURE[bm]
        ci_hi = max(sig["ci_upper"]) * 1.32
        panel_ylims[bm] = (0, ci_hi)

    for idx, bm in enumerate(PANEL_ORDER):
        row, col = divmod(idx, 2)
        inner = outer[row, col].subgridspec(
            nrows=2, ncols=1,
            height_ratios=[3.2, 1.0],
            hspace=0.08,
        )
        ax_top = fig.add_subplot(inner[0, 0])
        ax_bot = fig.add_subplot(inner[1, 0], sharex=ax_top)

        sig = CLUSTER_BIOMARKER_LAG_SIGNATURE[bm]
        lags = np.array(sig["lag_day"], dtype=float)
        mu = np.array(sig["mean_abs_shap"])
        lo = np.array(sig["ci_lower"])
        hi = np.array(sig["ci_upper"])
        pp = np.array(sig["pct_positive"])
        n = sig["n_samples"]
        r2 = sig["model_r2"]
        tr_min, tr_max = sig["temp_rank_min"], sig["temp_rank_max"]

        organ = BIOMARKER_SYSTEM.get(bm, "body_composition")
        base_colour = ORGAN_COLOURS[organ]

        # ---- Top strip: mean |SHAP| ribbon ----
        ax_top.fill_between(
            lags, lo, hi, color=base_colour, alpha=0.22,
            linewidth=0, zorder=2, label="95% bootstrap CI",
        )
        ax_top.plot(
            lags, mu, color=base_colour, linewidth=1.5,
            zorder=3, alpha=0.85,
        )
        # Markers coloured by directional asymmetry (pct_positive)
        for x, y, p in zip(lags, mu, pp):
            ax_top.scatter(
                x, y, s=58,
                color=cmap_dir(shap_norm(p)),
                edgecolor=base_colour, linewidth=1.0,
                zorder=5,
            )

        # Peak-lag shading
        peak_idx = int(np.argmax(mu))
        # width of shaded band in log-like x coords
        peak_x = lags[peak_idx]
        # half-widths of a visual "bar" depend on neighbours
        left = (lags[peak_idx - 1] + peak_x) / 2 if peak_idx > 0 else peak_x - 0.8
        right = (peak_x + lags[peak_idx + 1]) / 2 if peak_idx < len(lags) - 1 else peak_x + 1.8
        ax_top.axvspan(
            left, right, color=base_colour, alpha=0.08, zorder=1,
        )
        # Peak annotation - offset horizontally if peak sits at edge of x-axis
        if peak_idx <= 1:
            ha_peak = "left"
            xoff = 9
        elif peak_idx == len(lags) - 1:
            ha_peak = "right"
            xoff = -6
        else:
            ha_peak = "center"
            xoff = 0
        ax_top.annotate(
            f"peak lag = {int(peak_x)} d",
            xy=(peak_x, mu[peak_idx]),
            xytext=(xoff, 7),
            textcoords="offset points",
            ha=ha_peak, va="bottom",
            fontsize=6.8, color="#222222",
            fontweight="semibold",
        )

        # Titles + axes
        title = f"{BIOMARKER_PRETTY[bm]}"
        subtitle = f"n = {n:,} observations   •   CV R² = {md(r2, 2)}"
        # Subtitle sits ABOVE the title (panel letter → subtitle → title → plot)
        ax_top.text(
            0.0, 1.20, subtitle, transform=ax_top.transAxes,
            fontsize=6.9, color="#555555", ha="left", va="bottom",
        )
        ax_top.set_title(title, fontsize=8.6, pad=5, fontweight="semibold",
                         color=category_label_colour(base_colour), loc="left")
        ax_top.set_ylim(panel_ylims[bm])
        ax_top.set_xlim(-1.5, 32)
        ax_top.set_xticks(sig["lag_day"])
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.set_ylabel("mean |SHAP|", fontsize=7.8)
        ax_top.grid(axis="y", color="#EFEFEF", linewidth=0.4, zorder=0)
        ax_top.set_axisbelow(True)

        panel_label(ax_top, panel_letters[idx], x=-0.17, y=1.20, fontsize=10)

        # ---- Bottom strip: pct_positive divergence from 0.5 ----
        centred = pp - 0.5
        bar_colours = [cmap_dir(shap_norm(p)) for p in pp]
        ax_bot.bar(
            lags, centred, width=1.6,
            color=bar_colours, edgecolor="#444444", linewidth=0.4,
            zorder=3,
        )
        ax_bot.axhline(0, color="#888888", linewidth=0.6, zorder=2)
        ax_bot.set_ylim(-0.22, 0.22)
        ax_bot.set_yticks([-0.2, 0, 0.2])
        ax_bot.set_yticklabels(["30%", "50%", "70%"], fontsize=6.6)
        ax_bot.set_ylabel("fraction\n+ve SHAP", fontsize=6.8)
        ax_bot.set_xlabel("Temperature exposure lag (days)", fontsize=7.8)
        ax_bot.set_xticks(sig["lag_day"])
        ax_bot.set_xticklabels(lag_tick_labels(list(sig["lag_day"])))
        ax_bot.grid(axis="y", color="#F3F3F3", linewidth=0.3, zorder=0)
        ax_bot.set_axisbelow(True)
        for side in ("top", "right"):
            ax_bot.spines[side].set_visible(False)

    # ---- Overall title + narrative subtitle ----
    if not ARTWORK_MODE:
        fig.text(
            0.02, 0.985,
            "Figure 3  |  Temperature SHAP signatures for the four biomarkers driving GMM cluster separation",
            fontsize=9.8, fontweight="bold", ha="left", va="top",
        )
    # ---- Shared colour bar for the directional axis ----
    cbar_ax = fig.add_axes([0.50, 0.955, 0.25, 0.011])
    sm = cm.ScalarMappable(cmap=cmap_dir, norm=shap_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_ticks([0.30, 0.50, 0.70])
    cb.set_ticklabels(["30% (temp ↓ biomarker)", "50%", "70% (temp ↑ biomarker)"])
    cb.ax.tick_params(labelsize=6.2, length=2, pad=2)
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor("#666666")
    fig.text(
        0.49, 0.960, "Fraction of participant-visits with SHAP > 0:",
        fontsize=6.9, color="#333333", ha="right", va="center",
    )

    svg_path = OUT_DIR / "fig03_cluster_biomarker_signatures.svg"
    png_path = OUT_DIR / "fig03_cluster_biomarker_signatures.png"
    midline_tick_labels(fig)
    save_at_width(fig, OUT_DIR, "fig03_cluster_biomarker_signatures", span="double")
    write_caption(
        OUT_DIR, "fig03_cluster_biomarker_signatures",
        "Figure 3: Temperature SHAP signatures for the four biomarkers driving GMM cluster "
        "separation. For (a) haematocrit, (b) viral load, (c) haemoglobin and (d) body fat "
        "percentage \u2014 the four biomarkers that most strongly separate Cluster 3 from "
        "Clusters 1 and 2 \u2014 the upper panel shows the mean absolute SHAP value for the "
        "temperature feature across the seven lag windows (six single-day lags from 0 to 21 "
        "days plus a 30-day cumulative mean, 50 bootstrap replicates), with the peak lag "
        "marked. The lower panel shows the fraction of participant-visits with a positive "
        "temperature SHAP value, so bars above 50% indicate that higher temperature is "
        "associated with a higher value of that biomarker.",
        "Vertical scales differ between panels because absolute SHAP magnitudes are not "
        "comparable across biomarkers; compare the shape of each curve, not its height "
        "against another panel. " + LAG_FOOTNOTE,
    )
    plt.close(fig)

    return svg_path, png_path


if __name__ == "__main__":
    svg, png = build_figure()
    print(f"Saved: {svg}")
    print(f"Saved: {png}")
