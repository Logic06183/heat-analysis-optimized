"""Figure S4 - Temperature × modifier interactions (Stage 2, FDR-tested).

Scientific purpose:
    Stage 2 complements the Stage 1 main effects by testing whether any
    demographic, socioeconomic or exposure variable *modifies* the
    temperature-biomarker relationship. For each biomarker we rank the top-50
    SHAP interaction pairs, run a 500-permutation null on the top ranks, and
    apply BH FDR at q = 0.10 to identify modifiers that meaningfully change
    the per-patient temperature effect. This figure summarises which
    modifiers emerge where.

Panels:
    a. Modifier category × biomarker heat-map. Rows = 13 retained biomarkers,
       columns = 5 modifier categories (Age, Sex, HIV status, SES, Temp×Temp).
       Cell = max |SHAP interaction| across all seven temperature lags (any
       modifier feature belonging to that category). Bold outline indicates
       that at least one lag in this category passes FDR q=0.10.
    b. Per-biomarker FDR-significant interaction count (horizontal bar). The
       companion answer to "who had the strongest biomarker-level modifier
       signal".
    c. Which modifier category drives the most FDR-significant interactions
       (stacked bar across biomarkers). Answers "sex or age or SES?".
    d. Top-10 strongest FDR-significant biomarker × modifier × temperature-lag
       triplets - a compact forest that surfaces the most reportable
       interactions.

Source data:
    mcd_outputs/stage2/stage2_summary.json       (per-biomarker top modifiers)
"""
from __future__ import annotations

import json
import os
from mcd_pipeline import config as _pipeline_config
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from _data import RETAINED
from _lancet_style import (
    BIOMARKER_PRETTY, BIOMARKER_SYSTEM, MM, ORGAN_COLOURS,
    apply_style, panel_label,
)
from _lancet_style import ARTWORK_MODE, midline_tick_labels, md  # house style (2026-09-04)
from _lancet_style import save_at_width

STAGE2_SUMMARY = Path(
    str(_pipeline_config.OUTPUT_ROOT.parent / _pipeline_config.OUTPUT_ROOT.name / "stage2" / "stage2_summary.json")
)
# Artwork runs write elsewhere so submission files do not overwrite the
# working previews. Set RP2_FIG_OUT to redirect.
OUT_DIR = Path(os.environ.get("RP2_FIG_OUT",
                              str(Path(__file__).resolve().parent)))

# Row order, grouped by organ system to match S2. The template fixes the
# ORDER only; which rows appear follows the retained panel, so this figure
# cannot show biomarkers that were screened out of Stage 3.
_ROW_ORDER_TEMPLATE = [
    "systolic_bp", "diastolic_bp", "heart_rate",
    "creatinine",
    "hematocrit", "hemoglobin",
    "cd4_count", "viral_load",
    "albumin",
    "ldl_cholesterol",
    "body_fat_percent", "hip_circumference", "waist_hip_ratio",
    # bmi dropped under ERA5-Land - did not meet model adequacy.
]
BIOMARKER_ORDER = (
    [b for b in _ROW_ORDER_TEMPLATE if b in RETAINED]
    + [b for b in RETAINED if b not in _ROW_ORDER_TEMPLATE]
)

# Modifier categorisation (based on feature name prefix/pattern)
MODIFIER_CATEGORIES = [
    ("Age", "#7B3294"),
    ("Sex", "#C94F7C"),
    ("HIV status", "#1B9E77"),
    ("SES", "#DAA520"),
    ("Temp×Temp", "#1F78B4"),
]

FDR_THRESHOLD = 0.10  # BH q-value threshold (MCD spec)


def classify_modifier(mod_name: str) -> str | None:
    """Map a feature name to a modifier category label."""
    if mod_name.startswith("age"):
        return "Age"
    if mod_name.startswith("sex") or mod_name == "sex_Male":
        return "Sex"
    if mod_name.startswith("hiv_status"):
        return "HIV status"
    if mod_name.startswith("gcro_") or mod_name.startswith("study_source"):
        # Study source = cohort fixed effect - we count as SES/structural
        return "SES"
    if mod_name.startswith("temp_lag_") or mod_name.startswith("era5_") or mod_name == "year":
        return "Temp×Temp"
    return None


def load_stage2() -> dict:
    with open(STAGE2_SUMMARY) as fh:
        return json.load(fh)


def build_interaction_matrix(stage2: dict):
    """Return:
    max_mag: [14 biomarkers × 5 categories] - max |interaction| across lags
    fdr_hit: [14 × 5] bool - True if any lag passes FDR
    n_sig:   [14] - total FDR-sig interactions per biomarker
    cat_counts: [5] - total FDR-sig interactions per modifier category
    cat_by_bm: [14 × 5] - FDR-sig count per biomarker × category (for panel c)
    top_triplets: list of dicts for panel d (top-10 by mean_abs_interaction
                  among FDR-sig)
    """
    n_bm = len(BIOMARKER_ORDER)
    n_cat = len(MODIFIER_CATEGORIES)
    cat_names = [c[0] for c in MODIFIER_CATEGORIES]

    max_mag = np.full((n_bm, n_cat), np.nan)
    fdr_hit = np.zeros((n_bm, n_cat), dtype=bool)
    cat_by_bm = np.zeros((n_bm, n_cat), dtype=int)
    n_sig = np.zeros(n_bm, dtype=int)
    all_fdr_triplets: list = []

    for i, bm in enumerate(BIOMARKER_ORDER):
        rec = stage2.get(bm)
        if rec is None:
            continue
        top_mods = rec.get("top_temperature_modifiers", [])
        for entry in top_mods:
            mod = entry["modifier"]
            cat = classify_modifier(mod)
            if cat is None:
                continue
            j = cat_names.index(cat)
            mag = float(entry["mean_abs_interaction"])
            p_adj = float(entry["p_adjusted"])

            # Record max magnitude regardless of FDR (for the heatmap fill)
            if np.isnan(max_mag[i, j]) or mag > max_mag[i, j]:
                max_mag[i, j] = mag

            if p_adj <= FDR_THRESHOLD:
                fdr_hit[i, j] = True
                cat_by_bm[i, j] += 1
                n_sig[i] += 1
                all_fdr_triplets.append({
                    "biomarker": bm,
                    "modifier": mod,
                    "category": cat,
                    "temp_feature": entry["temp_feature"],
                    "mean_abs": mag,
                    "p_adj": p_adj,
                })

    # Normalise max_mag within each biomarker so rows are comparable (biomarkers
    # span 3+ orders of magnitude in raw SHAP magnitude).
    max_mag_norm = np.full_like(max_mag, np.nan)
    for i in range(n_bm):
        row = max_mag[i]
        if np.all(np.isnan(row)):
            continue
        peak = np.nanmax(row)
        if peak > 0:
            max_mag_norm[i] = row / peak

    cat_counts = np.nansum(cat_by_bm, axis=0)

    # Top-10 by magnitude, but scale each by the within-biomarker max
    for t in all_fdr_triplets:
        bm_i = BIOMARKER_ORDER.index(t["biomarker"])
        bm_peak = np.nanmax(max_mag[bm_i])
        t["scaled_mag"] = t["mean_abs"] / bm_peak if bm_peak > 0 else 0
    all_fdr_triplets.sort(key=lambda d: d["scaled_mag"], reverse=True)

    return max_mag_norm, fdr_hit, n_sig, cat_counts, cat_by_bm, all_fdr_triplets[:10]


def build_figure():
    apply_style(base_font_size=8.0)

    fig_w_in = 190 * MM
    fig_h_in = 235 * MM
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    outer = fig.add_gridspec(
        nrows=2, ncols=2,
        left=0.090, right=0.975, top=0.830, bottom=0.055,
        height_ratios=[1.30, 1.00],
        width_ratios=[1.55, 1.00],
        hspace=0.62, wspace=0.45,
    )

    stage2 = load_stage2()
    max_mag_norm, fdr_hit, n_sig, cat_counts, cat_by_bm, top10 = build_interaction_matrix(stage2)

    # ============ Panel a: modifier category × biomarker heatmap ============
    ax_a = fig.add_subplot(outer[0, 0])
    cmap = LinearSegmentedColormap.from_list(
        "mod_cmap",
        ["#FDFDFD", "#E8E6D3", "#C2C07E", "#6F8B4D", "#2A6F3F"],
        N=256,
    )
    im = ax_a.imshow(max_mag_norm, cmap=cmap, vmin=0.0, vmax=1.0,
                     aspect="auto", zorder=2)
    ax_a.set_xticks(range(len(MODIFIER_CATEGORIES)))
    ax_a.set_xticklabels([c[0] for c in MODIFIER_CATEGORIES], fontsize=7.3)
    for tick, (_name, col) in zip(ax_a.get_xticklabels(), MODIFIER_CATEGORIES):
        tick.set_color(col)
        tick.set_fontweight("semibold")
    ax_a.set_yticks(range(len(BIOMARKER_ORDER)))
    ax_a.set_yticklabels(
        [BIOMARKER_PRETTY[b] for b in BIOMARKER_ORDER], fontsize=7.2,
    )
    for tick, bm in zip(ax_a.get_yticklabels(), BIOMARKER_ORDER):
        tick.set_color(ORGAN_COLOURS[BIOMARKER_SYSTEM[bm]])
    # Annotate cells: FDR hits get bold rectangle + small check glyph
    for i in range(max_mag_norm.shape[0]):
        for j in range(max_mag_norm.shape[1]):
            val = max_mag_norm[i, j]
            if np.isnan(val):
                ax_a.text(j, i, "-", ha="center", va="center",
                          fontsize=6.8, color="#AAAAAA")
                continue
            if fdr_hit[i, j]:
                ax_a.add_patch(Rectangle(
                    (j - 0.46, i - 0.46), 0.92, 0.92,
                    facecolor="none", edgecolor="#111111",
                    linewidth=1.1, zorder=4,
                ))
                ax_a.plot(j + 0.30, i - 0.30, marker="o", markersize=3.2,
                          markerfacecolor="#111111", markeredgecolor="none",
                          zorder=5)
            txt_colour = "#111111" if val < 0.55 else "white"
            ax_a.text(j, i, f"{val:.2f}", ha="center", va="center",
                      fontsize=6.3, color=txt_colour, zorder=5)
    ax_a.set_title(
        "|SHAP interaction| per biomarker × modifier category",
        fontsize=7.7, fontweight="semibold", color="#333333",
        pad=4, loc="left",
    )
    ax_a.tick_params(length=0)
    for side in ("top", "right", "bottom", "left"):
        ax_a.spines[side].set_visible(False)
    panel_label(ax_a, "a", x=-0.24, y=1.04, fontsize=11)

    # Compact colourbar above panel a, tucked to the right of the panel header
    ax_a_pos = ax_a.get_position()
    cax_w = 0.14
    cax_h = 0.008
    cax = fig.add_axes([
        ax_a_pos.x1 - cax_w,
        ax_a_pos.y1 + 0.035,
        cax_w, cax_h,
    ])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=6.0, length=2, pad=1.6)
    cb.outline.set_linewidth(0.3)
    cb.outline.set_edgecolor("#666666")
    fig.text(
        ax_a_pos.x1 - cax_w, ax_a_pos.y1 + 0.050,
        "Row-normalised |interaction|   dot + outline = FDR q ≤ 0.10",
        fontsize=6.2, color="#555555", ha="left", va="bottom",
    )

    # ============ Panel b: FDR-significant interaction count per biomarker ============
    ax_b = fig.add_subplot(outer[0, 1])
    y = np.arange(len(BIOMARKER_ORDER))
    # Order bars to mirror heatmap rows
    sorted_idx = list(range(len(BIOMARKER_ORDER)))
    ax_b.barh(
        y, [n_sig[i] for i in sorted_idx],
        color=[ORGAN_COLOURS[BIOMARKER_SYSTEM[BIOMARKER_ORDER[i]]] for i in sorted_idx],
        edgecolor="#222222", linewidth=0.3, zorder=3,
    )
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([BIOMARKER_PRETTY[BIOMARKER_ORDER[i]] for i in sorted_idx],
                         fontsize=7.0)
    for tick, i in zip(ax_b.get_yticklabels(), sorted_idx):
        tick.set_color(ORGAN_COLOURS[BIOMARKER_SYSTEM[BIOMARKER_ORDER[i]]])
    ax_b.set_xlabel("FDR-significant interactions (n)", fontsize=7.2)
    from matplotlib.ticker import MaxNLocator
    ax_b.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_b.invert_yaxis()
    ax_b.grid(axis="x", color="#F0F0F0", linewidth=0.4, zorder=0)
    ax_b.set_axisbelow(True)
    for side in ("top", "right"):
        ax_b.spines[side].set_visible(False)
    # Annotate counts at bar end
    for yi, i in enumerate(sorted_idx):
        cnt = int(n_sig[i])
        if cnt == 0:
            ax_b.text(0.25, yi, "0", ha="left", va="center",
                      fontsize=6.5, color="#999999")
        else:
            ax_b.text(cnt + 0.25, yi, f"{cnt}", ha="left", va="center",
                      fontsize=6.6, color="#222222", fontweight="semibold")
    ax_b.set_title("FDR-significant interactions per biomarker", fontsize=8.0, fontweight="semibold",
                   loc="left", pad=5)
    panel_label(ax_b, "b", x=-0.40, y=1.08, fontsize=11)

    # ============ Panel c: modifier category stacked bar ============
    ax_c = fig.add_subplot(outer[1, 0])
    # Per biomarker, stacked by modifier category
    n_bm = len(BIOMARKER_ORDER)
    x = np.arange(n_bm)
    bottom = np.zeros(n_bm)
    for j, (cat_name, cat_col) in enumerate(MODIFIER_CATEGORIES):
        h = cat_by_bm[:, j]
        ax_c.bar(x, h, bottom=bottom, color=cat_col, edgecolor="white",
                 linewidth=0.3, width=0.72, label=cat_name, zorder=3)
        bottom += h
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([BIOMARKER_PRETTY[b] for b in BIOMARKER_ORDER],
                         rotation=38, ha="right", fontsize=6.5)
    for tick, bm in zip(ax_c.get_xticklabels(), BIOMARKER_ORDER):
        tick.set_color(ORGAN_COLOURS[BIOMARKER_SYSTEM[bm]])
    ax_c.set_ylabel("FDR-significant interactions (n)", fontsize=7.2)
    ax_c.grid(axis="y", color="#F0F0F0", linewidth=0.4, zorder=0)
    ax_c.set_axisbelow(True)
    for side in ("top", "right"):
        ax_c.spines[side].set_visible(False)
    ax_c.set_title(
        "Modifier-category composition of FDR-significant interactions",
        fontsize=7.8, fontweight="semibold", color="#333333",
        pad=5, loc="left",
    )
    # Summary totals as a mini-legend at top-right
    total = int(cat_counts.sum())
    y_legend = ax_c.get_ylim()[1]
    handles = []
    for j, (cat_name, cat_col) in enumerate(MODIFIER_CATEGORIES):
        cnt = int(cat_counts[j])
        pct = 100 * cnt / total if total else 0
        handles.append(plt.Rectangle((0, 0), 1, 1, color=cat_col))
    leg = ax_c.legend(
        handles,
        [f"{c[0]}  {int(cat_counts[i])} ({100*int(cat_counts[i])/total:.0f}%)"
         if total else c[0]
         for i, c in enumerate(MODIFIER_CATEGORIES)],
        loc="upper left", frameon=False, fontsize=6.6, handlelength=1.0,
        handleheight=0.6, borderaxespad=0.2,
    )
    panel_label(ax_c, "c", x=-0.09, y=1.10, fontsize=11)

    # ============ Panel d: Top-10 FDR-significant triplets (forest-like) ============
    ax_d = fig.add_subplot(outer[1, 1])
    for side in ("top", "right"):
        ax_d.spines[side].set_visible(False)

    if not top10:
        ax_d.text(0.5, 0.5, "No FDR-significant\ninteractions", ha="center",
                  va="center", fontsize=9, transform=ax_d.transAxes)
        ax_d.set_xticks([])
        ax_d.set_yticks([])
    else:
        y = np.arange(len(top10))[::-1]  # top of list at top
        cat_colour_map = {c[0]: c[1] for c in MODIFIER_CATEGORIES}
        values = [t["scaled_mag"] for t in top10]
        cats = [t["category"] for t in top10]
        bm_pretty = [BIOMARKER_PRETTY[t["biomarker"]] for t in top10]
        mod_name_clean = [t["modifier"].replace("gcro_", "")
                                     .replace("_", " ")
                                     .replace("sex Male", "Male (vs F)")
                                     .replace("hiv status Positive", "HIV+")
                                     for t in top10]
        temp_name_clean = [t["temp_feature"].replace("temp_lag_", "")
                                          .replace("d", "d")
                           for t in top10]
        rowcols = [cat_colour_map[c] for c in cats]
        ax_d.barh(y, values, color=rowcols, edgecolor="#222222", linewidth=0.3,
                  zorder=3, height=0.34)
        # Single combined descriptor above each bar, left-aligned inside the
        # panel's own axes and coloured by organ system (was a wide left-hanging
        # y-tick label that overflowed into panel c and clipped biomarker names).
        # The bar fill already encodes modifier category, so organ colour on the
        # label adds the orthogonal biomarker-system cue.
        for yi, t, bm, mod, tf in zip(y, top10, bm_pretty, mod_name_clean,
                                      temp_name_clean):
            organ_col = ORGAN_COLOURS[BIOMARKER_SYSTEM[t["biomarker"]]]
            ax_d.text(0.0, yi + 0.25, f"{bm}  ·  {mod} × {tf}",
                      ha="left", va="bottom", fontsize=6.3, color=organ_col)
        for yi, t, v in zip(y, top10, values):
            ax_d.text(v + 0.015, yi,
                      f"|SHAP|={t['mean_abs']:.3f}, q={t['p_adj']:.3f}",
                      ha="left", va="center", fontsize=6.0, color="#333333")
        ax_d.set_yticks([])
        ax_d.set_ylim(-0.7, len(top10) - 0.02)
        ax_d.set_xlim(0, max(values) * 1.42)
        ax_d.set_xlabel("Scaled interaction magnitude\n(within-biomarker)",
                        fontsize=6.8)
        ax_d.grid(axis="x", color="#F0F0F0", linewidth=0.4, zorder=0)
        ax_d.set_axisbelow(True)

    ax_d.set_title("Ten strongest FDR-significant interactions",
                   fontsize=8.0, fontweight="semibold", loc="left", pad=5)
    panel_label(ax_d, "d", x=-0.08, y=1.10, fontsize=11)

    # ============ Titles ============
    if not ARTWORK_MODE:
        fig.text(
            0.02, 0.985,
            "Figure S4  |  Temperature × modifier interactions (Stage 2)",
            fontsize=10.0, fontweight="bold", ha="left", va="top",
        )
    fig.text(
        0.02, 0.964,
        "Top-50 SHAP interaction pairs per biomarker tested against a 500-permutation null with BH-FDR at q = 0.10.",
        fontsize=7.2, color="#444444", ha="left", va="top",
    )
    fig.text(
        0.02, 0.949,
        "Sex dominates body-composition and haematological biomarkers; age modifies cardiovascular and renal biomarkers.",
        fontsize=7.2, color="#444444", ha="left", va="top",
    )

    # ---- Save ----
    svg_path = OUT_DIR / "figS4_interaction_heatmap.svg"
    png_path = OUT_DIR / "figS4_interaction_heatmap.png"
    midline_tick_labels(fig)
    save_at_width(fig, OUT_DIR, "figS4_interaction_heatmap", span="double")
    plt.close(fig)
    return svg_path, png_path


if __name__ == "__main__":
    svg, png = build_figure()
    print(f"Saved: {svg}")
    print(f"Saved: {png}")
