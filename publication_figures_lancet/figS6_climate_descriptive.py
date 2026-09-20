"""Figure S6 - Climate descriptive summary (JHB ERA5 exposure).

Scientific purpose:
    Johannesburg sits at 1,753m elevation on the South African highveld, which
    gives it a temperate climate with a cool-dry winter (May-Aug) and a warm-wet
    summer (Oct-Mar). Our analysis links ERA5 Land reanalysis temperatures to
    biomarker visits across 2005-2022. This figure provides reviewers with a
    single visual summary of the exposure distribution that feeds Stage 1.

Panels:
    a. Monthly climatology box-whisker (p5-p95 envelope, median line) -
       shows the temperate seasonal cycle and absence of extreme summer
       heat (max p95 ≈ 24°C).
    b. Annual means and p95 across 2005-2022, with visit count bars on a
       secondary axis - shows no monotonic warming trend over the
       recruitment window and identifies 2020-21 as the peak recruitment
       years (COVID-era VIDA-007/008).
    c. Overall lag-0 exposure distribution (histogram) with tercile shading
       labelled cool / moderate / warm - makes the point that 'heat' in
       this cohort is highveld warm (p95 ≈ 22.5°C, max ≈ 26.6°C), not
       tropical.
    d. Lag correlation matrix (lag-0, lag-7, lag-14, lag-30) - supports the
       methodological choice of interventional TreeSHAP for correlated lag
       features (ρ ≈ 0.82-0.97 between lag pairs).

Source data:
    mcd_outputs/dlnm_r_input.csv (ERA5 extraction, 59,896 patient-visit rows,
    2005-2022). Summaries frozen as constants below for reproducibility.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from _lancet_style import MM, apply_style, panel_label
from _lancet_style import ARTWORK_MODE, midline_tick_labels, md  # house style (2026-09-04)
from _lancet_style import save_at_width

# Artwork runs write elsewhere so submission files do not overwrite the
# working previews. Set RP2_FIG_OUT to redirect.
OUT_DIR = Path(os.environ.get("RP2_FIG_OUT",
                              str(Path(__file__).resolve().parent)))

# ----------------------------------------------------------------------
# Compute climate stats live from the ERA5-Land linkage CSV.
# Falls back to frozen snapshot if the linkage CSV is not present.
# ----------------------------------------------------------------------
import pandas as pd

_LINKAGE = Path(__file__).resolve().parents[1] / "FINAL_DATASETS" / "CLIMATE_HEALTH_LINKAGE" / "ERA5_LAND_LINKAGE.csv"

def _compute_live_climate() -> dict:
    df = pd.read_csv(_LINKAGE, low_memory=False)
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    df["month"] = df["visit_date"].dt.month
    df["year"] = df["visit_date"].dt.year
    t = df["era5land_temp_lag0d_c"]

    monthly = {}
    for m in range(1, 13):
        sub = t[df["month"] == m].dropna()
        if len(sub) == 0:
            continue
        p5, p25, p50, p75, p95 = np.nanpercentile(sub, [5, 25, 50, 75, 95])
        monthly[m] = (float(np.nanmean(sub)), float(p5), float(p25), float(p50), float(p75), float(p95))

    yearly = []
    for y in sorted(df["year"].unique()):
        sub = df[df["year"] == y]["era5land_temp_lag0d_c"].dropna()
        if len(sub) == 0:
            continue
        yearly.append((int(y), float(np.nanmean(sub)),
                       float(np.nanpercentile(sub, 95)), int(len(sub))))

    P = np.nanpercentile(t.dropna(), [5, 25, 50, 75, 95])
    p_levels = {"p5": float(P[0]), "p25": float(P[1]),
                "p50": float(P[2]), "p75": float(P[3]), "p95": float(P[4])}

    counts, _ = np.histogram(t.dropna(), bins=np.arange(1, 27, 1))

    lag_cols = ["era5land_temp_lag0d_c", "era5land_temp_lag7d_c",
                "era5land_temp_lag14d_c", "era5land_temp_lag30d_c"]
    lag_corr = df[lag_cols].corr().values

    return {
        "MONTHLY_CLIM": monthly,
        "YEARLY": yearly,
        "P_LEVELS": p_levels,
        "HIST_COUNTS": counts.astype(int),
        "LAG_CORR": lag_corr,
        "n_visits": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()),
        "date_min": str(df["visit_date"].min().date()),
        "date_max": str(df["visit_date"].max().date()),
    }


_LIVE = _compute_live_climate() if _LINKAGE.exists() else None

if _LIVE is not None:
    MONTHLY_CLIM = _LIVE["MONTHLY_CLIM"]
    YEARLY = _LIVE["YEARLY"]
    P_LEVELS = _LIVE["P_LEVELS"]
    HIST_COUNTS = _LIVE["HIST_COUNTS"]
    LAG_CORR = _LIVE["LAG_CORR"]
else:
    # Frozen snapshot fallback (2026-04-23 ERA5 extraction)
    MONTHLY_CLIM = {
        1:  (20.66, 17.76, 19.56, 20.57, 21.67, 23.66),
        2:  (20.15, 17.09, 19.15, 20.26, 21.18, 22.56),
        3:  (18.80, 15.60, 17.74, 18.87, 20.05, 21.55),
        4:  (15.76, 12.38, 14.58, 15.69, 17.03, 19.19),
        5:  (13.02,  7.95, 11.70, 13.44, 14.63, 16.17),
        6:  (10.25,  6.27,  8.64, 10.39, 11.86, 13.83),
        7:  (10.19,  5.79,  8.77, 10.37, 11.71, 13.60),
        8:  (13.28,  8.08, 11.23, 13.22, 15.63, 18.19),
        9:  (17.58, 12.21, 15.60, 17.77, 20.01, 22.13),
        10: (18.81, 13.86, 16.69, 18.72, 21.02, 24.03),
        11: (19.65, 15.07, 17.98, 19.82, 21.33, 23.65),
        12: (20.23, 15.85, 18.94, 20.29, 21.84, 23.90),
    }
    YEARLY = []
    P_LEVELS = {"p5": 8.34, "p25": 11.82, "p50": 16.06, "p75": 19.55, "p95": 22.49}
    HIST_COUNTS = np.array([
           3,   14,  167,  240,  494,  604, 1047, 2267, 2350, 4066, 4620,
        3511, 2725, 3866, 3848, 3784, 3854, 4659, 4823, 5108, 3637, 2118,
        1375,  571,  129,   16,
    ])
    LAG_CORR = np.array([
        [1.000, 0.860, 0.850, 0.816],
        [0.860, 1.000, 0.932, 0.926],
        [0.850, 0.932, 1.000, 0.967],
        [0.816, 0.926, 0.967, 1.000],
    ])

MONTH_ABBR = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
HIST_EDGES = np.arange(1, 27, 1)
LAG_NAMES = ["lag-0", "lag-7", "lag-14", "lag-30"]


def build_figure():
    apply_style(base_font_size=8.0)

    fig_w_in = 190 * MM
    fig_h_in = 210 * MM
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # ---- Figure title + subtitle (consistent with Figs S1-S5, S7) ----
    if not ARTWORK_MODE:
        fig.text(
            0.02, 0.978,
            "Figure S6  |  ERA5-Land climate exposure across the study period",
            fontsize=10.0, fontweight="bold", ha="left", va="top",
        )
    fig.text(
        0.02, 0.957,
        "Monthly climatology, year-on-year temperature trend, lag-0 distribution and lag-feature correlation at the 0·1° ERA5-Land grid linked to clinic visits.",
        fontsize=7.0, color="#444444", ha="left", va="top",
    )

    outer = fig.add_gridspec(
        nrows=2, ncols=2,
        left=0.075, right=0.970, top=0.910, bottom=0.060,
        height_ratios=[1.00, 1.00],
        width_ratios=[1.20, 1.00],
        hspace=0.55, wspace=0.38,
    )

    # Custom SA colourmap: cool-blue → cream → warm-ochre
    sa_cmap = LinearSegmentedColormap.from_list(
        "sa_warm",
        ["#2A6EB5", "#6FAAD5", "#E6D8A8", "#E89A5A", "#C94F7C"],
        N=256,
    )

    # ------------------------------------------------------------------
    # Panel a: monthly climatology box-whisker
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(outer[0, 0])
    months = list(range(1, 13))
    colours_month = [sa_cmap((MONTHLY_CLIM[m][0] - 8) / 14) for m in months]

    for m, col in zip(months, colours_month):
        mean, p5, p25, p50, p75, p95 = MONTHLY_CLIM[m]
        # Whisker (p5-p95)
        ax_a.plot([m, m], [p5, p95], color="#555555", linewidth=0.8,
                  zorder=2, solid_capstyle="round")
        # Box (p25-p75)
        ax_a.add_patch(plt.Rectangle(
            (m - 0.32, p25), 0.64, p75 - p25,
            facecolor=col, edgecolor="#222222", linewidth=0.5,
            alpha=0.85, zorder=3,
        ))
        # Median
        ax_a.plot([m - 0.32, m + 0.32], [p50, p50],
                  color="#222222", linewidth=1.1, zorder=4)
        # Mean dot
        ax_a.scatter(m, mean, s=14, marker="D",
                     facecolor="white", edgecolor="#222222",
                     linewidth=0.6, zorder=5)

    # Season band shading
    ax_a.axvspan(0.5, 3.5, facecolor="#F6CFCF", alpha=0.18, zorder=0)   # summer Jan-Mar
    ax_a.axvspan(5.5, 8.5, facecolor="#CFE4F6", alpha=0.22, zorder=0)   # winter Jun-Aug
    ax_a.axvspan(9.5, 12.5, facecolor="#F6CFCF", alpha=0.18, zorder=0)  # summer Oct-Dec

    ax_a.text(2, 5.2, "summer (Oct-Mar)", fontsize=6.4, color="#8B4F63",
              ha="center", va="bottom", fontstyle="italic")
    ax_a.text(7, 5.2, "winter (Jun-Aug)", fontsize=6.4, color="#4D6B8A",
              ha="center", va="bottom", fontstyle="italic")
    ax_a.text(11, 5.2, "summer", fontsize=6.4, color="#8B4F63",
              ha="center", va="bottom", fontstyle="italic")

    ax_a.set_xticks(months)
    ax_a.set_xticklabels(MONTH_ABBR, fontsize=7.4)
    ax_a.set_ylabel("Daily temperature (°C, ERA5-Land lag-0)", fontsize=7.4)
    ax_a.set_ylim(3, 27)
    ax_a.set_xlim(0.4, 12.6)
    ax_a.set_yticks([5, 10, 15, 20, 25])
    ax_a.grid(axis="y", color="#F0F0F0", linewidth=0.4, zorder=0)
    ax_a.set_axisbelow(True)
    for side in ("top", "right"):
        ax_a.spines[side].set_visible(False)
    ax_a.set_title(
        "Monthly climatology (2005-2022)",
        fontsize=8.0, fontweight="semibold", color="#333333",
        pad=4, loc="left",
    )
    panel_label(ax_a, "a", x=-0.11, y=1.06, fontsize=11)

    # Legend under panel a
    leg_handles = [
        plt.Line2D([0], [0], color="#555555", linewidth=0.8, label="p5-p95"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#A0B0BE", edgecolor="#222222",
                      linewidth=0.4, label="p25-p75"),
        plt.Line2D([0], [0], color="#222222", linewidth=1.1, label="median"),
        plt.Line2D([0], [0], marker="D", linestyle="None",
                   markerfacecolor="white", markeredgecolor="#222222",
                   markersize=4, label="mean"),
    ]
    ax_a.legend(handles=leg_handles, loc="upper right", frameon=False,
                fontsize=6.6, ncol=4, handlelength=1.2, handleheight=0.7,
                columnspacing=0.8, borderaxespad=0.3)

    # ------------------------------------------------------------------
    # Panel b: yearly trend + visit count
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(outer[0, 1])
    years = np.array([y[0] for y in YEARLY])
    y_mean = np.array([y[1] for y in YEARLY])
    y_p95 = np.array([y[2] for y in YEARLY])
    y_n = np.array([y[3] for y in YEARLY])

    # Visits on secondary axis (grey bars behind)
    ax_bn = ax_b.twinx()
    ax_bn.bar(years, y_n, color="#D9D9D9", edgecolor="none",
              width=0.78, zorder=1)
    ax_bn.set_ylabel("n visits", fontsize=7.0, color="#666666")
    ax_bn.tick_params(axis="y", labelsize=6.4, colors="#666666", length=2)
    ax_bn.set_ylim(0, 25000)
    for side in ("top",):
        ax_bn.spines[side].set_visible(False)
    ax_bn.spines["right"].set_color("#999999")

    # Temperature lines
    ax_b.plot(years, y_p95, color="#C94F7C", marker="o", markersize=3.2,
              linewidth=1.2, label="p95 (hot-day temp)", zorder=4)
    ax_b.plot(years, y_mean, color="#2A6EB5", marker="s", markersize=3.0,
              linewidth=1.2, label="annual mean", zorder=3)

    ax_b.set_xlabel("Year", fontsize=7.4)
    ax_b.set_ylabel("Temperature (°C)", fontsize=7.4, color="#333333")
    ax_b.set_ylim(12, 26)
    ax_b.set_yticks([14, 18, 22, 26])
    ax_b.set_xlim(2004.3, 2022.7)
    ax_b.set_xticks([2006, 2010, 2014, 2018, 2022])
    ax_b.grid(axis="y", color="#F0F0F0", linewidth=0.4, zorder=0)
    ax_b.set_axisbelow(True)
    for side in ("top",):
        ax_b.spines[side].set_visible(False)
    ax_b.set_zorder(ax_bn.get_zorder() + 1)
    ax_b.patch.set_visible(False)

    ax_b.legend(loc="upper left", frameon=False, fontsize=6.6,
                handlelength=1.4, borderaxespad=0.3)
    ax_b.set_title(
        "Annual trend & recruitment (2005-2022)",
        fontsize=8.0, fontweight="semibold", color="#333333",
        pad=4, loc="left",
    )
    panel_label(ax_b, "b", x=-0.14, y=1.06, fontsize=11)

    # ------------------------------------------------------------------
    # Panel c: exposure histogram
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(outer[1, 0])
    edges = HIST_EDGES  # length 26: bins 1..26
    centres = (edges[:-1] + edges[1:]) / 2
    counts = HIST_COUNTS[:len(centres)]

    # Colour bars by their temperature
    bar_cols = [sa_cmap((c - 3) / 24) for c in centres]
    ax_c.bar(centres, counts, width=0.95, color=bar_cols,
             edgecolor="#222222", linewidth=0.3, zorder=3)

    # Percentile markers - staggered y-positions to avoid label crowding
    for lbl, v, ypos in [
        ("p25", P_LEVELS["p25"], 5700),
        ("median", P_LEVELS["p50"], 5400),
        ("p75", P_LEVELS["p75"], 5700),
        ("p95", P_LEVELS["p95"], 5400),
    ]:
        ax_c.axvline(v, color="#222222", linewidth=0.5,
                     linestyle="--", alpha=0.55, zorder=4)
        ax_c.text(v, ypos, lbl, fontsize=6.3, color="#222222",
                  ha="center", va="bottom",
                  bbox=dict(boxstyle="round,pad=0.18",
                            facecolor="white", edgecolor="none",
                            alpha=0.8))
    # p95-rect for "heat" (≥p95) zone
    ax_c.axvspan(P_LEVELS["p95"], 27, facecolor="#C94F7C",
                 alpha=0.12, zorder=1)
    ax_c.text(
        P_LEVELS["p95"] + 1.3, 4800,
        "heat-exposure\nregion (≥p95)",
        fontsize=6.4, color="#9A3A5E", ha="left", va="top",
        fontstyle="italic",
    )

    ax_c.set_xlabel("Daily temperature (°C, ERA5-Land lag-0)", fontsize=7.4)
    ax_c.set_ylabel("Patient-visit days", fontsize=7.4)
    ax_c.set_xlim(2, 27)
    ax_c.set_ylim(0, 6000)
    ax_c.set_yticks([0, 1000, 2000, 3000, 4000, 5000])
    ax_c.grid(axis="y", color="#F0F0F0", linewidth=0.4, zorder=0)
    ax_c.set_axisbelow(True)
    for side in ("top", "right"):
        ax_c.spines[side].set_visible(False)
    ax_c.set_title(
        "Lag-0 exposure distribution (all patient-visits)",
        fontsize=8.0, fontweight="semibold", color="#333333",
        pad=4, loc="left",
    )
    panel_label(ax_c, "c", x=-0.11, y=1.06, fontsize=11)

    # ------------------------------------------------------------------
    # Panel d: lag correlation matrix (lower triangle only - the matrix
    # is symmetric, so the upper triangle would duplicate the lower).
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(outer[1, 1])
    n = len(LAG_NAMES)
    # Mask upper triangle so only lower triangle + diagonal renders.
    masked = np.ma.array(LAG_CORR, mask=np.triu(np.ones_like(LAG_CORR), k=1).astype(bool))
    cmap_obj = plt.cm.RdBu_r.copy()
    cmap_obj.set_bad("white", alpha=0.0)
    im = ax_d.imshow(masked, cmap=cmap_obj, vmin=0.5, vmax=1.0,
                     aspect="equal", zorder=2)

    # Annotate cells (lower triangle only)
    for i in range(n):
        for j in range(i + 1):
            r = LAG_CORR[i, j]
            fontcol = "white" if r < 0.68 or r > 0.92 else "#222222"
            ax_d.text(j, i, f"{r:.2f}", ha="center", va="center",
                      fontsize=7.6, color=fontcol, fontweight="semibold",
                      zorder=3)

    ax_d.set_xticks(range(n))
    ax_d.set_yticks(range(n))
    ax_d.set_xticklabels(LAG_NAMES, fontsize=7.2)
    ax_d.set_yticklabels(LAG_NAMES, fontsize=7.2)
    ax_d.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax_d.spines[side].set_visible(False)

    # Colourbar
    ax_d_pos = ax_d.get_position()
    cax = fig.add_axes([ax_d_pos.x0, ax_d_pos.y0 - 0.042,
                        ax_d_pos.width, 0.009])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_ticks([0.5, 0.7, 0.9, 1.0])
    cbar.ax.tick_params(labelsize=6.4)
    cbar.set_label("Pearson ρ", fontsize=7.0)
    cbar.outline.set_visible(False)

    ax_d.set_title(
        "Lag correlation structure",
        fontsize=8.0, fontweight="semibold", color="#333333",
        pad=4, loc="left",
    )
    panel_label(ax_d, "d", x=-0.14, y=1.06, fontsize=11)

    # Inline figure title removed - Lancet typesets the caption separately.

    # Save
    svg_path = OUT_DIR / "figS6_climate_descriptive.svg"
    png_path = OUT_DIR / "figS6_climate_descriptive.png"
    midline_tick_labels(fig)
    save_at_width(fig, OUT_DIR, "figS6_climate_descriptive", span="double")
    plt.close(fig)
    return svg_path, png_path


if __name__ == "__main__":
    svg, png = build_figure()
    print(f"Saved: {svg}")
    print(f"Saved: {png}")
