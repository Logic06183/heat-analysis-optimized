"""Figure S3 - Sensitivity analyses summary.

Scientific purpose:
    The primary Stage 1 results (Figure 1) rest on modelling choices that
    could each plausibly drive the findings. This figure summarises the
    exposure-definition and HIV-stratification checks as a single
    robust-vs-not visual: would a reviewer's preferred alternative
    specification change the story?

Panels (appendix numbering as of 21 September 2026):
    a. Concordance heatmap (retained biomarkers x 5 specifications:
       SA4 heatwave-day thresholds at p95/p90, SA5 three humidity-aware
       indices). Cell = Spearman rho of the per-lag SHAP profile against
       the primary specification; rho >= 0.70 outlined.
    b. Per-specification mean rho (95% range across biomarkers) and the
       share of retained biomarkers concordant.
    c. Humidity-metric comparison (SA5) by biomarker.
    d. HIV-stratified concordance (SA3): retained biomarkers whose
       HIV-positive and HIV-negative lag profiles disagree, those that
       agree, and those with no HIV-negative stratum.

The biomarker rows follow RETAINED from the Stage 3 run in force.

Source data:
    mcd_outputs_era5_land/sensitivity/{sa8_heatwave,sa10_humidity,sa7_hiv_stratified}/
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

SENS_DIR = Path(str(_pipeline_config.OUTPUT_ROOT.parent / _pipeline_config.OUTPUT_ROOT.name / "sensitivity"))
# Artwork runs write elsewhere so submission files do not overwrite the
# working previews. Set RP2_FIG_OUT to redirect.
OUT_DIR = Path(os.environ.get("RP2_FIG_OUT",
                              str(Path(__file__).resolve().parent)))

SA_COLUMNS = [
    ("SA4 p95", "sa8_heatwave/sa8_summary.json", ("p95_vs_primary", None)),
    ("SA4 p90", "sa8_heatwave/sa8_summary.json", ("p90_vs_primary", None)),
    ("SA5 apparent T", "sa10_humidity/sa10_summary.json", ("variants", "apparent_temperature")),
    ("SA5 heat index", "sa10_humidity/sa10_summary.json", ("variants", "heat_index")),
    ("SA5 wet bulb", "sa10_humidity/sa10_summary.json", ("variants", "wet_bulb_stull")),
]

# Organ-system display order for the retained panel. Filtered against
# RETAINED (read from the Stage 3 run) so the figure follows the retention
# rule in force rather than a hard-coded list.
_ORGAN_DISPLAY_ORDER = [
    "systolic_bp", "diastolic_bp", "heart_rate",
    "creatinine",
    "hematocrit", "hemoglobin",
    "cd4_count", "viral_load",
    "albumin",
    "ldl_cholesterol",
    "body_fat_percent", "hip_circumference", "waist_hip_ratio",
]
BIOMARKER_ORDER = [bm for bm in _ORGAN_DISPLAY_ORDER if bm in RETAINED] + \
                  [bm for bm in RETAINED if bm not in _ORGAN_DISPLAY_ORDER]


def get_spearman(sa_json: dict, bm: str, locator: tuple):
    key1, key2 = locator
    if key1 == "variants":
        node = sa_json["variants"][key2]["per_biomarker"]
    elif key1 == "per_biomarker":
        node = sa_json["per_biomarker"]
    else:
        node = sa_json["per_biomarker"]
    if bm not in node:
        return None
    entry = node[bm]
    if key1 in ("p90_vs_primary", "p95_vs_primary"):
        entry = entry.get(key1)
        if entry is None:
            return None
    if not isinstance(entry, dict):
        return None
    return entry.get("spearman_rho")


def load_concordance_matrix() -> np.ndarray:
    M = np.full((len(BIOMARKER_ORDER), len(SA_COLUMNS)), np.nan)
    for j, (_label, rel, locator) in enumerate(SA_COLUMNS):
        with open(SENS_DIR / rel) as fh:
            data = json.load(fh)
        for i, bm in enumerate(BIOMARKER_ORDER):
            rho = get_spearman(data, bm, locator)
            if rho is not None:
                M[i, j] = rho
    return M


def load_summary_stats() -> list:
    M = load_concordance_matrix()
    out = []
    for j, (label, _rel, _loc) in enumerate(SA_COLUMNS):
        vec = M[:, j]
        vec = vec[~np.isnan(vec)]
        mean = float(np.mean(vec))
        lo = float(np.percentile(vec, 2.5)) if len(vec) > 3 else float(np.min(vec))
        hi = float(np.percentile(vec, 97.5)) if len(vec) > 3 else float(np.max(vec))
        n_concordant = int(np.sum(vec >= 0.70))
        n_total = len(vec)
        out.append({
            "label": label, "mean": mean, "ci_lo": lo, "ci_hi": hi,
            "n_concordant": n_concordant, "n_total": n_total,
        })
    return out


def build_figure():
    apply_style(base_font_size=8.0)

    fig_w_in = 190 * MM
    fig_h_in = 240 * MM
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # Outer layout: top row = heatmap + forest; bottom row = humidity strip + HIV callout
    outer = fig.add_gridspec(
        nrows=2, ncols=2,
        left=0.085, right=0.975, top=0.845, bottom=0.055,
        height_ratios=[1.25, 1.0],
        width_ratios=[1.70, 1.0],
        hspace=0.60, wspace=0.40,
    )

    # ============ Panel a: concordance heatmap ============
    ax_a = fig.add_subplot(outer[0, 0])
    M = load_concordance_matrix()
    cmap = LinearSegmentedColormap.from_list(
        "sa_cmap",
        ["#B24745", "#F2F2F2", "#E5D4B1", "#8AB07D", "#2A9D8F"],
        N=256,
    )
    im = ax_a.imshow(M, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto", zorder=2)
    ax_a.set_xticks(range(len(SA_COLUMNS)))
    ax_a.set_xticklabels(
        [c[0] for c in SA_COLUMNS], rotation=35, ha="right", fontsize=6.9,
    )
    ax_a.set_yticks(range(len(BIOMARKER_ORDER)))
    ax_a.set_yticklabels(
        [BIOMARKER_PRETTY[b] for b in BIOMARKER_ORDER], fontsize=7.1,
    )
    for tick, bm in zip(ax_a.get_yticklabels(), BIOMARKER_ORDER):
        tick.set_color(ORGAN_COLOURS[BIOMARKER_SYSTEM[bm]])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            if np.isnan(val):
                ax_a.text(j, i, "-", ha="center", va="center",
                          fontsize=6.2, color="#999999")
                continue
            if val >= 0.70:
                ax_a.add_patch(Rectangle(
                    (j - 0.46, i - 0.46), 0.92, 0.92,
                    facecolor="none", edgecolor="#111111",
                    linewidth=1.0, zorder=4,
                ))
            colour = "#111111" if abs(val) < 0.55 else "white"
            ax_a.text(j, i, f"{val:.2f}", ha="center", va="center",
                      fontsize=5.8, color=colour, zorder=5)
    ax_a.set_title(
        "Spearman ρ of SHAP lag profile vs primary specification",
        fontsize=7.8, fontweight="semibold", color="#333333",
        pad=4, loc="left",
    )
    ax_a.tick_params(length=0)
    for side in ("top", "right", "bottom", "left"):
        ax_a.spines[side].set_visible(False)
    panel_label(ax_a, "a", x=-0.20, y=1.10, fontsize=11)

    # Colourbar - placed cleanly above panel a, well clear of the subtitle
    cax = fig.add_axes([0.085, 0.898, 0.48, 0.010])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([-1.0, -0.5, 0, 0.5, 0.7, 1.0])
    cb.set_ticklabels(["−1", "−0.5", "0", "0.5", "0.7*", "1"])
    cb.ax.tick_params(labelsize=6.2, length=2, pad=2)
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor("#666666")
    fig.text(
        0.085, 0.917,
        "* ρ ≥ 0.70 counted as concordant (bold outline in panel a).",
        fontsize=6.6, color="#555555", ha="left", va="bottom",
    )

    # ============ Panel b: summary forest ============
    ax_b = fig.add_subplot(outer[0, 1])
    stats = load_summary_stats()
    stats.sort(key=lambda s: s["mean"])
    labels = [s["label"] for s in stats]
    means = np.array([s["mean"] for s in stats])
    los = np.array([s["ci_lo"] for s in stats])
    his = np.array([s["ci_hi"] for s in stats])
    n_conc = [s["n_concordant"] for s in stats]
    n_tot = [s["n_total"] for s in stats]

    y = np.arange(len(stats))
    for i in range(len(stats)):
        ax_b.plot([los[i], his[i]], [y[i], y[i]],
                  color="#999999", linewidth=0.9, zorder=2)

    def _col(v):
        if v >= 0.70:
            return "#2A9D8F"
        if v >= 0.40:
            return "#E9A93A"
        return "#B24745"

    point_cols = [_col(m) for m in means]
    ax_b.scatter(means, y, s=54, facecolor=point_cols, edgecolor="#222222",
                 linewidth=0.6, zorder=4)
    ax_b.axvline(0.70, color="#2A9D8F", linestyle="--", linewidth=0.7,
                 zorder=1, alpha=0.55)
    ax_b.axvline(0, color="#CCCCCC", linewidth=0.5, zorder=1)

    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labels, fontsize=7.0)
    ax_b.set_xlim(-0.6, 1.05)
    ax_b.set_xticks([-0.5, 0, 0.5, 0.7, 1.0])
    ax_b.set_xticklabels(["-0.5", "0", "0.5", "0.7", "1"], fontsize=6.5)
    ax_b.set_xlabel("Mean Spearman ρ (95% range across biomarkers)",
                    fontsize=7.2)
    ax_b.grid(axis="x", color="#F0F0F0", linewidth=0.4, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.set_title("Sensitivity ranking", fontsize=8.0, fontweight="semibold",
                   loc="left", pad=4)

    # Right-side concordance count (placed just inside axes)
    for i, (nc, nt) in enumerate(zip(n_conc, n_tot)):
        pct = 100 * nc / nt if nt else 0
        ax_b.text(1.04, y[i], f"{nc}/{nt} ({pct:.0f}%)",
                  transform=ax_b.get_yaxis_transform(),
                  ha="left", va="center", fontsize=6.3, color="#333333")
    panel_label(ax_b, "b", x=-0.32, y=1.10, fontsize=11)

    # ============ Panel c: humidity-metric SA5 strip ============
    ax_c = fig.add_subplot(outer[1, 0])
    with open(SENS_DIR / "sa10_humidity/sa10_summary.json") as fh:
        sa10 = json.load(fh)
    variants = [
        ("Dry-bulb (primary)", None, "#2A9D8F"),
        ("Apparent temp", "apparent_temperature", "#F28E2B"),
        ("Heat index (NOAA)", "heat_index", "#E25C3B"),
        ("Wet-bulb (Stull)", "wet_bulb_stull", "#9F2B68"),
    ]
    x_positions = np.arange(len(BIOMARKER_ORDER))
    for vi, (_vname, key, colour) in enumerate(variants):
        if key is None:
            vec = np.ones(len(BIOMARKER_ORDER))
        else:
            vec = np.array([
                get_spearman(sa10, bm, ("variants", key))
                if get_spearman(sa10, bm, ("variants", key)) is not None else np.nan
                for bm in BIOMARKER_ORDER
            ])
        y_vals = np.full_like(vec, vi, dtype=float)
        sizes = [55 if (not np.isnan(v) and v >= 0.70) else 26 for v in vec]
        facecols = [colour if not np.isnan(v) else "white" for v in vec]
        ax_c.scatter(x_positions, y_vals, s=sizes,
                     facecolor=facecols, edgecolor="#222222",
                     linewidth=0.4, zorder=3)
        for xi, v in zip(x_positions, vec):
            if np.isnan(v):
                continue
            ax_c.text(xi, vi + 0.30, f"{v:.2f}",
                      ha="center", va="bottom", fontsize=5.6, color="#333333")

    for vi in range(len(variants) + 1):
        ax_c.axhline(vi - 0.5, color="#EFEFEF", linewidth=0.4, zorder=1)

    ax_c.set_yticks(range(len(variants)))
    ax_c.set_yticklabels([v[0] for v in variants], fontsize=7.0)
    ax_c.set_xticks(x_positions)
    ax_c.set_xticklabels([BIOMARKER_PRETTY[b] for b in BIOMARKER_ORDER],
                         rotation=38, ha="right", fontsize=6.4)
    for tick, bm in zip(ax_c.get_xticklabels(), BIOMARKER_ORDER):
        tick.set_color(ORGAN_COLOURS[BIOMARKER_SYSTEM[bm]])
    ax_c.set_ylim(-0.7, len(variants) - 0.3)
    ax_c.set_xlim(-0.8, len(BIOMARKER_ORDER) - 0.2)
    ax_c.grid(axis="y", visible=False)
    ax_c.invert_yaxis()
    for side in ("top", "right"):
        ax_c.spines[side].set_visible(False)
    ax_c.set_title(
        "Exposure-metric sensitivity (SA5) - dry-bulb reference vs humidity-aware indices",
        fontsize=7.6, fontweight="semibold", color="#333333",
        pad=4, loc="left",
    )
    panel_label(ax_c, "c", x=-0.12, y=1.13, fontsize=11)

    # ============ Panel d: HIV effect modifier callout (SA5) ============
    ax_d = fig.add_subplot(outer[1, 1])
    ax_d.set_xticks([]); ax_d.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax_d.spines[side].set_visible(False)

    with open(SENS_DIR / "sa7_hiv_stratified/sa7_summary.json") as fh:
        sa7 = json.load(fh)
    # Schema-tolerant: old (adequate_biomarkers_only nested) vs new (flat) format.
    # Restricted to the retained panel (RETAINED, from the Stage 3 run).
    # Biomarkers recorded only in people living with HIV (CD4 count, viral
    # load) have no HIV-negative stratum and are listed separately rather
    # than counted as "not modified".
    retained = set(RETAINED)
    per_bm = sa7.get("per_biomarker", {})
    no_comparator = [b for b in BIOMARKER_ORDER
                     if isinstance(per_bm.get(b), dict) and "hiv_pos_vs_neg" not in per_bm[b]]
    if "adequate_biomarkers_only" in sa7:
        adequate = sa7["adequate_biomarkers_only"]
        effect_mods = sorted(b for b in adequate["effect_modifiers"] if b in retained)
        not_mod = [b for b in adequate.get("not_modified", [])
                   if b in retained and b not in no_comparator]
    else:
        effect_mods = sorted(b for b in sa7.get("effect_modifiers", []) if b in retained)
        not_mod = sorted(retained - set(effect_mods) - set(no_comparator))

    ax_d.text(0.0, 1.02,
              "SA3  |  HIV status as effect modifier",
              fontsize=8.8, fontweight="bold", color="#333333",
              transform=ax_d.transAxes, va="top")

    n_effect = len(effect_mods)
    n_total = len(BIOMARKER_ORDER)
    # Restrained headline: moderate-weight figure + single descriptor line beneath
    # (was a 26pt display number with text wrapped awkwardly beside it).
    ax_d.text(0.02, 0.925, f"{n_effect} of {n_total}",
              fontsize=15, color="#B22246", fontweight="bold",
              transform=ax_d.transAxes, va="top")
    ax_d.text(0.02, 0.80,
              "retained biomarkers show distinct lag-response\nprofiles by HIV status",
              fontsize=7.0, color="#222222", linespacing=1.35,
              transform=ax_d.transAxes, va="top")

    ax_d.text(0.0, 0.64,
              "Effect modifiers (HIV+ vs HIV- ρ < 0.70):",
              fontsize=7.1, fontweight="semibold", color="#333333",
              transform=ax_d.transAxes, va="top")

    # Two-column list - 6 per column. Increased row spacing 0.055 -> 0.062
    # so the 6.8 px font (8 px effective with leading) doesn't crowd.
    for i, bm in enumerate(effect_mods):
        c, r = divmod(i, 6)
        pretty = BIOMARKER_PRETTY.get(bm, bm)
        organ = BIOMARKER_SYSTEM.get(bm, "body_composition")
        col = ORGAN_COLOURS[organ]
        ax_d.text(
            0.02 + c * 0.50, 0.57 - r * 0.062,
            f"•  {pretty}",
            fontsize=6.8, color=col,
            transform=ax_d.transAxes, va="top",
        )

    # Not modified: lowered to 0.18 (was 0.22) so it clears the last bullet
    # row of the effect-modifier column.
    ax_d.text(0.0, 0.18, "Not modified:", fontsize=7.0, fontweight="semibold",
              color="#555555", transform=ax_d.transAxes, va="top")
    for i, bm in enumerate(not_mod):
        pretty = BIOMARKER_PRETTY.get(bm, bm)
        organ = BIOMARKER_SYSTEM.get(bm, "body_composition")
        col = ORGAN_COLOURS[organ]
        ax_d.text(
            0.02 + i * 0.34, 0.12,
            f"•  {pretty}",
            fontsize=6.8, color=col,
            transform=ax_d.transAxes, va="top",
        )
    if no_comparator:
        ax_d.text(
            0.0, 0.07,
            "No HIV-negative comparator: "
            + ", ".join(BIOMARKER_PRETTY.get(bm, bm) for bm in no_comparator),
            fontsize=6.6, color="#555555",
            transform=ax_d.transAxes, va="top",
        )

    ax_d.text(
        0.0, 0.00,
        "→ motivates HIV-stratified primary analysis.",
        fontsize=6.9, color="#444444", fontstyle="italic",
        transform=ax_d.transAxes, va="top",
    )
    panel_label(ax_d, "d", x=-0.04, y=1.07, fontsize=11)

    # ============ Titles ============
    if not ARTWORK_MODE:
        fig.text(
            0.02, 0.982,
            "Figure S3  |  Sensitivity-analysis summary",
            fontsize=10.0, fontweight="bold", ha="left", va="top",
        )
    fig.text(
        0.02, 0.962,
        "Concordance of each sensitivity specification with the primary Stage 1 model.",
        fontsize=7.2, color="#444444", ha="left", va="top",
    )
    fig.text(
        0.02, 0.945,
        "Findings are robust to heat-wave threshold (SA4 p95/p90) but sensitive to exposure-metric choice",
        fontsize=7.2, color="#444444", ha="left", va="top",
    )
    fig.text(
        0.02, 0.929,
        "(SA5 humidity-aware indices) and to HIV-status stratification (SA3 panel d).",
        fontsize=7.2, color="#444444", ha="left", va="top",
    )

    # ---- Save ----
    svg_path = OUT_DIR / "figS3_sensitivity_analyses.svg"
    png_path = OUT_DIR / "figS3_sensitivity_analyses.png"
    midline_tick_labels(fig)
    save_at_width(fig, OUT_DIR, "figS3_sensitivity_analyses", span="double")
    plt.close(fig)
    return svg_path, png_path


if __name__ == "__main__":
    svg, png = build_figure()
    print(f"Saved: {svg}")
    print(f"Saved: {png}")
