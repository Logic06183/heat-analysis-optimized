"""Figure S5 - Cohort descriptive summary (14 Johannesburg cohorts).

Scientific purpose:
    The analysis ready dataset pools 14 prospective cohorts recruited from
    Johannesburg between 2004 and 2024. Sample sizes, demographic profiles
    and HIV prevalence differ substantially between studies (ACTG trials,
    WRHI HIV treatment cohorts, VIDA/Aurum population cohorts, SCHARP,
    DPHRU ageing, Ezintsha long-term ART). This figure gives reviewers and
    readers a single visual for "what goes into the analysis" - per-cohort
    sample size, sex and HIV mix, and age distribution.

Panels:
    a. Per-cohort sample size (horizontal bar). Bar colour encodes HIV
       prevalence (teal = HIV-negative cohorts, rose = HIV-positive
       cohorts, neutral = mixed).
    b. Age distribution per cohort - mean ± SD dot-whisker, showing the
       span from SCHARP-006 adolescents to DPHRU-053 ageing adults.
    c. Sex × HIV composition - each cohort is a bubble positioned by
       (% female, % HIV+), bubble size ∝ n_patients. Cohort names
       labelled.
    d. Summary statistics box.

Source data:
    FINAL_DATASETS/TIDY_DATASETS/TIDY_demographics.csv   (frozen snapshot).
    Cohort metadata extracted 2026-04-23 from the tidy long-format file.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

from _lancet_style import MM, apply_style, panel_label
from _lancet_style import ARTWORK_MODE, midline_tick_labels, md  # house style (2026-09-04)
from _lancet_style import save_at_width

# Artwork runs write elsewhere so submission files do not overwrite the
# working previews. Set RP2_FIG_OUT to redirect.
OUT_DIR = Path(os.environ.get("RP2_FIG_OUT",
                              str(Path(__file__).resolve().parent)))

# Per-cohort summary for the participants who contribute biomarker observations (the
# Table 1 denominators: unique patient_id per study in TIDY_clinical_biomarkers.csv,
# n = 9,759), with age, sex and HIV status from TIDY_demographics.csv. Computed
# 2026-09-21. Study names follow Table 1 of the paper. pct_female/pct_hiv_pos in %.
COHORTS = [
    # study name, n, mean_age, sd_age, pct_female, pct_hiv_pos, programme_label
    ("Thol'Impilo",   2186, 35.21, 10.14,  62.21, 100.00, "HIV cohort (Aurum)"),
    ("COV005",        2130, 32.88, 11.06,  45.02,   0.00, "Population (VIDA)"),
    ("ADVANCE",       1053, 32.46,  7.73,  59.16, 100.00, "ART (Ezintsha)"),
    ("MASC",          1009, 53.50,  5.99,  49.55,  19.72, "Ageing (DPHRU)"),
    ("PEARLS",         600, 34.72,  7.85,  57.50, 100.00, "HIV treatment (WRHI)"),
    ("HCW cohort",     535, 39.05,  9.38,  82.43,   3.55, "Population (VIDA)"),
    ("Ezintsha 025",   487, 34.78, 10.10,  48.46,   2.87, "Population (Ezintsha)"),
    ("HPTN 082",       451, 20.84,  2.28, 100.00,   0.00, "Adolescent (SCHARP)"),
    ("HPTN 075",       401, 24.23,  5.48,   0.00,  17.96, "Male cohort (SCHARP)"),
    ("WRHI 052",       300, 41.99,  7.91,  68.00, 100.00, "HIV treatment (WRHI)"),
    ("WBS",            247, 32.05,  7.23, 100.00,  60.32, "Women's health (DPHRU)"),
    ("OCTANE",         152, np.nan, np.nan, 100.00, 100.00, "ACTG clinical trial"),
    ("STRIDE",         108, 34.39,  8.02, 100.00, 100.00, "ACTG clinical trial"),
    ("ACTG 5175",      100, 38.17,  7.61, 100.00, 100.00, "ACTG clinical trial"),
]

PROGRAMME_COLOURS = {
    "ACTG clinical trial":       "#C94F7C",  # rose
    "ART (Ezintsha)":             "#C94F7C",
    "HIV cohort (Aurum)":         "#C94F7C",
    "HIV treatment (WRHI)":       "#C94F7C",
    "Adolescent (SCHARP)":        "#1F78B4",  # blue
    "Male cohort (SCHARP)":       "#1F78B4",
    "Ageing (DPHRU)":             "#DAA520",  # gold
    "Population (Ezintsha)":      "#2A9D8F",  # teal
    "Population (VIDA)":          "#2A9D8F",
    "Women's health (DPHRU)":     "#6A3D9A",  # purple
}


def hiv_colour(pct: float) -> str:
    """Teal→grey→rose gradient on HIV prevalence."""
    if pct >= 75:
        return "#C94F7C"   # HIV-positive-dominant
    if pct <= 25:
        return "#2A9D8F"   # HIV-negative-dominant
    return "#DAA520"        # mixed (>25% and <75% HIV+)


def build_figure():
    apply_style(base_font_size=8.0)

    fig_w_in = 190 * MM
    fig_h_in = 220 * MM
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # ---- Figure title + subtitle (consistent with Figs S1-S4, S7) ----
    if not ARTWORK_MODE:
        fig.text(
            0.02, 0.978,
            "Figure S5  |  Cohort descriptive summary across 14 Johannesburg studies",
            fontsize=10.0, fontweight="bold", ha="left", va="top",
        )
    fig.text(
        0.02, 0.957,
        "Per-cohort sample size, age distribution, and sex × HIV mix for the prospective studies pooled in the primary analysis (ERA5-Land linkage).",
        fontsize=7.0, color="#444444", ha="left", va="top",
    )

    outer = fig.add_gridspec(
        nrows=2, ncols=2,
        left=0.100, right=0.975, top=0.910, bottom=0.060,
        height_ratios=[1.35, 1.00],
        width_ratios=[1.30, 1.00],
        hspace=0.50, wspace=0.45,
    )

    cohorts = COHORTS
    n_cohorts = len(cohorts)

    # ---- Panel a: sample size bars ----
    ax_a = fig.add_subplot(outer[0, 0])
    y = np.arange(n_cohorts)
    n_arr = np.array([c[1] for c in cohorts])
    hiv_pct = np.array([c[5] for c in cohorts])
    bar_cols = [hiv_colour(p) for p in hiv_pct]
    ax_a.barh(y, n_arr, color=bar_cols, edgecolor="#222222",
              linewidth=0.35, zorder=3, height=0.72)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([c[0].replace("JHB_", "") for c in cohorts],
                         fontsize=7.1)
    for tick, c in zip(ax_a.get_yticklabels(), cohorts):
        tick.set_color(hiv_colour(c[5]))
        tick.set_fontweight("semibold")
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Participants", fontsize=7.4)
    ax_a.grid(axis="x", color="#F0F0F0", linewidth=0.4, zorder=0)
    ax_a.set_axisbelow(True)
    for side in ("top", "right"):
        ax_a.spines[side].set_visible(False)

    # Annotate each bar: n and %HIV
    x_max = n_arr.max()
    for yi, c in zip(y, cohorts):
        ax_a.text(
            c[1] + x_max * 0.015, yi,
            f"n={c[1]:,}   HIV+ {c[5]:.0f}%   F {c[4]:.0f}%",
            ha="left", va="center", fontsize=6.4, color="#333333",
        )
    ax_a.set_xlim(0, x_max * 1.38)
    ax_a.set_title(
        "Participants per cohort",
        fontsize=8.0, fontweight="semibold", color="#333333",
        pad=4, loc="left",
    )
    panel_label(ax_a, "a", x=-0.22, y=1.05, fontsize=11)

    # Small legend
    leg_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#C94F7C"),
        plt.Rectangle((0, 0), 1, 1, color="#DAA520"),
        plt.Rectangle((0, 0), 1, 1, color="#2A9D8F"),
    ]
    ax_a.legend(
        leg_handles,
        ["HIV-positive dominant (≥75%)", "Mixed (25-75%)", "HIV-negative dominant (≤25%)"],
        loc="lower right", frameon=False, fontsize=6.5, handlelength=1.0,
        handleheight=0.6, borderaxespad=0.3,
    )

    # ---- Panel b: age mean ± SD per cohort ----
    ax_b = fig.add_subplot(outer[0, 1])
    for yi, c in zip(y, cohorts):
        mean = c[2]
        sd = c[3]
        if np.isnan(mean):
            continue
        ax_b.plot([mean - sd, mean + sd], [yi, yi],
                  color="#999999", linewidth=1.0, zorder=2)
        ax_b.scatter(mean, yi, s=36, color=hiv_colour(c[5]),
                     edgecolor="#222222", linewidth=0.5, zorder=4)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([])
    ax_b.invert_yaxis()
    ax_b.set_xlim(10, 65)
    ax_b.set_xticks([15, 25, 35, 45, 55, 65])
    ax_b.set_xlabel("Age (years, mean ± SD)", fontsize=7.4)
    ax_b.grid(axis="x", color="#F0F0F0", linewidth=0.4, zorder=0)
    ax_b.set_axisbelow(True)
    for side in ("top", "right"):
        ax_b.spines[side].set_visible(False)
    ax_b.axvline(18, color="#E0C0C0", linewidth=0.6, linestyle="--",
                 alpha=0.6, zorder=1)
    ax_b.axvline(60, color="#E0C0C0", linewidth=0.6, linestyle="--",
                 alpha=0.6, zorder=1)
    ax_b.set_title(
        "Age profile",
        fontsize=8.0, fontweight="semibold", color="#333333",
        pad=4, loc="left",
    )
    panel_label(ax_b, "b", x=-0.10, y=1.05, fontsize=11)

    # Missing-age note
    missing = [c[0] for c in cohorts if np.isnan(c[2])]
    if missing:
        ax_b.text(
            0.02, 0.02,
            f"missing: {', '.join(m.replace('JHB_', '') for m in missing)}",
            transform=ax_b.transAxes, fontsize=6.0, color="#999999",
            ha="left", va="bottom", fontstyle="italic",
        )

    # ---- Panel c: sex × HIV composition bubble ----
    ax_c = fig.add_subplot(outer[1, 0])
    # Manual label offsets to avoid collisions in the dense 100/100 corner.
    # Key: study_code without JHB_ prefix.  (dx, dy, ha, va)  in data coords.
    LABEL_OFFSETS = {
        "Thol'Impilo":   ( +2.0, -7.5, "left",   "top"),
        "ADVANCE":    ( -5.0, -7.5, "right",  "top"),
        "PEARLS":    (-15.0,  0.0, "right",  "center"),
        "WRHI 052":    ( +4.0,  0.0, "left",   "center"),
        "WBS":   ( +4.0, -4.0, "left",   "top"),
        "OCTANE":    ( -2.0, +9.0, "right",  "bottom"),
        "STRIDE":    ( +3.0, +9.0, "left",   "bottom"),
        "ACTG 5175":    (  0.0, -8.5, "center", "top"),
        "HPTN 082":  ( -3.0, -4.0, "right",  "top"),
        "HPTN 075":  ( +4.0,  0.0, "left",   "center"),
        "COV005":    ( -3.0, +3.5, "right",  "bottom"),
        "HCW cohort":    ( +4.0,  0.0, "left",   "center"),
        "Ezintsha 025":    ( -3.0, -4.0, "right",  "top"),
        "MASC":   ( +4.0,  0.0, "left",   "center"),
    }
    for c in cohorts:
        pct_fem = c[4]
        pct_hiv = c[5]
        n = c[1]
        label = c[0].replace("JHB_", "")
        # bubble size: scale n from 100 to 2550 into 30..250 pts²
        size = 30 + (n - 100) / (2550 - 100) * 260
        ax_c.scatter(pct_fem, pct_hiv, s=size,
                     facecolor=hiv_colour(pct_hiv),
                     alpha=0.75,
                     edgecolor="#222222", linewidth=0.5, zorder=3)
        dx, dy, ha, va = LABEL_OFFSETS.get(label, (+3.0, 0.0, "left", "center"))
        ax_c.text(
            pct_fem + dx, pct_hiv + dy,
            label,
            fontsize=6.3, color="#222222", va=va, ha=ha,
            zorder=5,
        )
    ax_c.set_xlabel("% Female", fontsize=7.4)
    ax_c.set_ylabel("% HIV-positive", fontsize=7.4)
    ax_c.set_xlim(-10, 118)
    ax_c.set_ylim(-15, 118)
    ax_c.set_xticks([0, 25, 50, 75, 100])
    ax_c.set_yticks([0, 25, 50, 75, 100])
    ax_c.grid(color="#F0F0F0", linewidth=0.4, zorder=0)
    ax_c.set_axisbelow(True)
    for side in ("top", "right"):
        ax_c.spines[side].set_visible(False)
    # Quadrant shading
    ax_c.axhspan(75, 112, facecolor="#C94F7C", alpha=0.04, zorder=0)
    ax_c.axhspan(-8, 25, facecolor="#2A9D8F", alpha=0.04, zorder=0)
    ax_c.set_title(
        "Sex × HIV composition per cohort (bubble size = n)",
        fontsize=8.0, fontweight="semibold", color="#333333",
        pad=4, loc="left",
    )
    panel_label(ax_c, "c", x=-0.14, y=1.05, fontsize=11)

    # ---- Panel d: summary stats box ----
    ax_d = fig.add_subplot(outer[1, 1])
    ax_d.set_xticks([]); ax_d.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax_d.spines[side].set_visible(False)

    total_n = sum(c[1] for c in cohorts)
    n_hiv_dominant = sum(1 for c in cohorts if c[5] >= 75)
    n_mixed = sum(1 for c in cohorts if 25 < c[5] < 75)
    n_hivneg_dominant = sum(1 for c in cohorts if c[5] <= 25)
    n_female_only = sum(1 for c in cohorts if c[4] == 100)
    n_pop_cohorts = sum(1 for c in cohorts if "Population" in c[6])
    n_treatment = sum(1 for c in cohorts if "HIV" in c[6] or "ART" in c[6] or "ACTG" in c[6])

    ax_d.text(0.0, 1.00, "At a glance", fontsize=9.2, fontweight="bold",
              color="#333333", transform=ax_d.transAxes, va="top")

    y0 = 0.91
    # Live counts from ERA5_LAND_LINKAGE.csv (computed below).
    _linkage_path = Path(__file__).resolve().parents[1] / "FINAL_DATASETS" / "CLIMATE_HEALTH_LINKAGE" / "ERA5_LAND_LINKAGE.csv"
    if _linkage_path.exists():
        import pandas as pd
        _link = pd.read_csv(_linkage_path, low_memory=False, usecols=["patient_id", "visit_date"])
        _link["visit_date"] = pd.to_datetime(_link["visit_date"])
        n_linked = _link["patient_id"].nunique()
        n_visits = len(_link)
        yr_min = int(_link["visit_date"].dt.year.min())
        yr_max = int(_link["visit_date"].dt.year.max())
    else:
        n_linked, n_visits, yr_min, yr_max = 10130, 67826, 2005, 2022

    rows = [
        (f"{len(cohorts)}", "prospective cohorts"),
        (f"{total_n:,}", "participants (demographics snapshot)"),
        (f"{n_linked:,}", "after ERA5-Land climate linkage"),
        (f"{n_visits:,}", "patient-visits"),
        (f"{yr_max - yr_min} years", f"follow-up window ({yr_min}-{yr_max})"),
    ]
    # Restrained key-value table (was a stack of five 13.5pt display numbers that
    # read like a marketing 'at a glance' slide). Bold right-aligned figures in a
    # narrow left column, descriptor on the same baseline.
    for num, desc in rows:
        ax_d.text(0.30, y0, num, fontsize=9.5, fontweight="bold",
                  color="#B22246", transform=ax_d.transAxes, va="top", ha="right")
        ax_d.text(0.36, y0, desc, fontsize=7.1, color="#222222",
                  transform=ax_d.transAxes, va="top", ha="left")
        y0 -= 0.075

    ax_d.text(0.0, y0 - 0.00, "Programme mix:", fontsize=7.4,
              fontweight="semibold", color="#333333",
              transform=ax_d.transAxes, va="top")
    y0 -= 0.06
    def _plural(n, noun):
        return f"{noun}" if n == 1 else f"{noun}s"
    mix_lines = [
        f"•  {n_hiv_dominant} HIV-positive {_plural(n_hiv_dominant, 'cohort')}",
        f"•  {n_mixed} mixed {_plural(n_mixed, 'cohort')}",
        f"•  {n_hivneg_dominant} HIV-negative {_plural(n_hivneg_dominant, 'cohort')}",
        f"•  {n_female_only} female-only {_plural(n_female_only, 'cohort')}",
    ]
    for line in mix_lines:
        ax_d.text(0.02, y0, line, fontsize=6.9, color="#222222",
                  transform=ax_d.transAxes, va="top")
        y0 -= 0.052

    ax_d.text(
        0.0, y0 - 0.015,
        "Pipeline aggregates across cohorts with GroupKFold by patient_id,\n"
        "ensuring no participant appears in both train and test folds.",
        fontsize=6.5, color="#555555", fontstyle="italic",
        transform=ax_d.transAxes, va="top", linespacing=1.25,
    )
    panel_label(ax_d, "d", x=-0.05, y=1.05, fontsize=11)

    # Inline figure title removed - Lancet typesets the caption separately.

    # ---- Save ----
    svg_path = OUT_DIR / "figS5_cohort_descriptive.svg"
    png_path = OUT_DIR / "figS5_cohort_descriptive.png"
    ax_d.set_visible(False)  # summary text panel removed; the caption and Table 1 carry the counts
    midline_tick_labels(fig)
    save_at_width(fig, OUT_DIR, "figS5_cohort_descriptive", span="double")
    plt.close(fig)
    return svg_path, png_path


if __name__ == "__main__":
    svg, png = build_figure()
    print(f"Saved: {svg}")
    print(f"Saved: {png}")
