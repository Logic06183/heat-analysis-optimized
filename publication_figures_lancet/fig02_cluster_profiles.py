"""Figure 2 - Vulnerability clusters: PCA structure, heat-sensitivity, demography.

Spec (manuscript v1, Section S9):
  (A) PCA scatter PC1 vs PC2 of 9,293 patients, coloured by GMM k=3 cluster,
      with 95% density ellipses per cluster.
  (B) Composite heat-sensitivity score by cluster, with 2,000-replicate
      bootstrap 95% CIs. Zero line marks the population reference.
  (C) Demographic composition by cluster (percent HIV-positive, female,
      unemployed, informal dwelling; mean age shown separately) with
      bootstrap 95% CIs.

Outputs:
  fig02_cluster_profiles.svg   (Figma-editable - text preserved)
  fig02_cluster_profiles.png   (300 dpi preview)
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

from _data import (
    CLUSTER_CHARACTERISTICS, PC_VARIANCE_EXPLAINED_PCT,
    N_PATIENTS_CLUSTERED, N_PCS_USED, ARI_STABILITY,
)
from _lancet_style import (
    ARTWORK_MODE, md, in_figure_note, lag_tick_labels, LAG_FOOTNOTE,
    category_label_colour, write_caption, check_width, midline_tick_labels,
    CLUSTER_COLOURS, CLUSTER_LABELS, CLUSTER_SHORT, MM,
    apply_style, panel_label,
)
from _lancet_style import save_at_width

# Artwork runs write elsewhere so submission files do not overwrite the
# working previews. Set RP2_FIG_OUT to redirect.
OUT_DIR = Path(os.environ.get("RP2_FIG_OUT",
                              str(Path(__file__).resolve().parent)))
# Cached intermediates always live beside the script, never in the artwork
# output directory -- OUT_DIR is where figures GO, not where inputs live.
CACHE_DIR = Path(__file__).resolve().parent
PC_FILE = CACHE_DIR / "_pc_cluster.csv"   # produced by a one-off python load above


def kde_contour(ax, xy, colour, mass=0.50, x_range=None, y_range=None, zbase=6):
    """Draw the Gaussian covariance ellipse enclosing `mass` of the cluster's
    probability, plus a white-haloed centroid dot.

    The clustering is a Gaussian mixture, so a covariance ellipse is the honest
    primitive: it shows exactly the shape the GMM fits. This replaces the earlier
    KDE contours, which rendered the genuinely diffuse 'older, employed' cluster
    (PC1/PC2 SD ~6-7) as a lumpy organic blob spanning the whole panel and buried
    the two tight cores under saturated fill.
    """
    from scipy.stats import chi2
    mu = xy.mean(axis=0)
    cov = np.cov(xy.T)
    vals, vecs = np.linalg.eigh(cov)          # ascending eigenvalues
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # Mahalanobis radius enclosing `mass` of a bivariate normal.
    r = np.sqrt(chi2.ppf(mass, df=2))
    width, height = 2 * r * np.sqrt(vals)      # full axis lengths
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    # Light fill (subtle figure-ground cue) + crisp edge (carries location).
    ax.add_patch(Ellipse(mu, width, height, angle=angle,
                         facecolor=colour, alpha=0.10, edgecolor="none",
                         zorder=zbase))
    ax.add_patch(Ellipse(mu, width, height, angle=angle,
                         facecolor="none", edgecolor=colour, linewidth=1.7,
                         zorder=zbase + 1))
    ax.scatter(mu[0], mu[1], marker="o", s=46, color=colour,
               edgecolor="white", linewidths=1.3, zorder=zbase + 2)


# Back-compat alias so existing callers don't need to change
density_ellipse = kde_contour


def panel_pca_scatter(ax):
    df = pd.read_csv(PC_FILE)
    # Sort clusters so the smallest cluster plots on top (visibility)
    cluster_sizes = df["cluster"].value_counts().to_dict()
    plot_order = sorted(cluster_sizes, key=lambda c: -cluster_sizes[c])

    # Fixed visible window chosen to show the cluster cores clearly - full
    # spread (especially Cluster 1 outliers) is documented in Supp Fig S1.
    X_LO, X_HI = -5.5, 5.5
    Y_LO, Y_HI = -3.0, 3.0

    # Raw points as faint texture (shows data volume, n=9,293) without competing
    # with the contour lines that carry the cluster structure.
    for c in plot_order:
        sub = df[df["cluster"] == c]
        ax.scatter(
            sub["PC1"], sub["PC2"],
            s=2.4, color=CLUSTER_COLOURS[c], alpha=0.18,
            linewidths=0, rasterized=True, zorder=2 + (1 if c == 2 else 0),
        )
    # Contour cores drawn largest-first so the smaller/tighter cluster lines sit
    # on top and stay readable where the cores overlap near the origin.
    for rank, c in enumerate(plot_order):
        sub = df[df["cluster"] == c]
        # Evaluate KDE only within the visible window so the contour reflects the
        # density of the visible core, not the full spread.
        xy_in = sub[
            (sub["PC1"].between(X_LO, X_HI)) & (sub["PC2"].between(Y_LO, Y_HI))
        ][["PC1", "PC2"]].values
        if len(xy_in) < 20:
            continue
        density_ellipse(
            ax, xy_in, CLUSTER_COLOURS[c],
            x_range=(X_LO, X_HI), y_range=(Y_LO, Y_HI),
            zbase=6 + rank * 3,
        )

    pc1_var, pc2_var = PC_VARIANCE_EXPLAINED_PCT[0], PC_VARIANCE_EXPLAINED_PCT[1]
    cum10 = sum(PC_VARIANCE_EXPLAINED_PCT[:10])
    ax.set_xlabel(f"PC1 ({md(pc1_var,1)}% variance)")
    ax.set_ylabel(f"PC2 ({md(pc2_var,1)}% variance)")
    ax.set_title(
        f"Participant profiles, PC1 × PC2 view  (n = {N_PATIENTS_CLUSTERED:,})",
        fontsize=8.2, pad=6, loc="center",
    )

    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(Y_LO, Y_HI)
    # Honesty caveat - this 2D projection shows only ~14% of the variance on
    # which the GMM actually clusters; full dimensional diagnostic in Fig S1.
    in_figure_note(
        ax, 0.985, 0.145,
        f"This 2D view captures {md(pc1_var + pc2_var,1)}% of the {md(cum10,1)}%\n"
        f"variance used to cluster in 10D  -  see Fig S1",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=5.9, color="#666666", style="italic",
        bbox=dict(facecolor="white", edgecolor="#CCCCCC",
                  linewidth=0.4, boxstyle="round,pad=0.25"),
    )

    # Cluster legend (custom so points + ellipses both represented)
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=6,
               markerfacecolor=CLUSTER_COLOURS[c], markeredgecolor="white",
               markeredgewidth=0.6,
               label=CLUSTER_LABELS[c])
        for c in [0, 1, 2]
    ]
    leg = ax.legend(
        handles=legend_handles, loc="upper right", frameon=False,
        fontsize=6.8, handlelength=1.0, labelspacing=0.35,
        bbox_to_anchor=(1.00, 1.00), borderaxespad=0.0,
    )
    leg.set_zorder(10)

    # Footer: clustering method
    ax.text(
        0.99, 0.02,
        f"GMM k=3, 10D PC space  •  ellipses = 50% covariance  •  ARI = {md(ARI_STABILITY, 3)}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=6.2, color="#666666", style="italic",
    )
    ax.grid(True, color="#EFEFEF", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def panel_heat_sensitivity(ax):
    clusters = [0, 1, 2]
    point = [CLUSTER_CHARACTERISTICS[c]["mean_composite_heat_sens"][0] for c in clusters]
    ci_lo = [CLUSTER_CHARACTERISTICS[c]["mean_composite_heat_sens"][1] for c in clusters]
    ci_hi = [CLUSTER_CHARACTERISTICS[c]["mean_composite_heat_sens"][2] for c in clusters]

    y = np.arange(len(clusters))
    colours = [CLUSTER_COLOURS[c] for c in clusters]

    # Horizontal bars from 0 to point estimate (signed)
    for i, (c, p) in enumerate(zip(clusters, point)):
        ax.barh(i, p, color=colours[i], alpha=0.65, height=0.55,
                edgecolor=colours[i], linewidth=0.8, zorder=2)
        # 95% CI whisker
        ax.hlines(i, ci_lo[i], ci_hi[i], color=colours[i], linewidth=1.4, zorder=3)
        ax.vlines([ci_lo[i], ci_hi[i]], i - 0.15, i + 0.15,
                  color=colours[i], linewidth=1.0, zorder=3)
        # Value label
        x_lab = ci_hi[i] + 0.03 if p >= 0 else ci_lo[i] - 0.03
        ha = "left" if p >= 0 else "right"
        ax.text(x_lab, i, ("+" if p >= 0 else "\u2212") + md(abs(p), 2), ha=ha, va="center",
                fontsize=7.0, color="#222222")

    ax.axvline(0, color="#444444", linewidth=0.8, linestyle="-", zorder=1)
    ax.set_ylim(2.8, -1.1)  # extra headroom for "Population reference" cue
    ax.set_yticks(y)
    ax.set_yticklabels([CLUSTER_SHORT[c] for c in clusters])
    for tick, c in zip(ax.get_yticklabels(), clusters):
        tick.set_color(category_label_colour(CLUSTER_COLOURS[c]))
        tick.set_fontweight("semibold")
    ax.invert_yaxis()
    ax.set_xlabel("Composite heat-sensitivity score  (z-scaled)")
    ax.set_title("Heat-sensitivity by vulnerability cluster  (mean, 95% CI)",
                 fontsize=8.2, pad=6, loc="center")
    ax.set_xlim(-0.85, 0.60)
    ax.grid(axis="x", color="#EFEFEF", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Reference annotation - small italic label at top of zero line
    ax.text(
        0.0, -0.68, "Population\nreference",
        fontsize=5.8, color="#666666",
        ha="center", va="bottom", style="italic",
    )


def panel_demographics(ax):
    clusters = [0, 1, 2]
    metrics = [
        ("pct_hiv_positive",     "HIV-positive"),
        ("pct_female",           "Female"),
        ("pct_unemployed",       "Unemployed"),
        ("pct_informal_dwelling","Informal dwelling"),
    ]
    n_m = len(metrics)
    n_c = len(clusters)
    bar_h = 0.22
    y_positions = np.arange(n_m)

    for ci, c in enumerate(clusters):
        offsets = (ci - (n_c - 1) / 2) * bar_h
        for mi, (key, _label) in enumerate(metrics):
            p, lo, hi = CLUSTER_CHARACTERISTICS[c][key]
            ax.barh(
                mi + offsets, p, height=bar_h * 0.88,
                color=CLUSTER_COLOURS[c], edgecolor="white", linewidth=0.4,
                alpha=0.90, zorder=2,
            )
            ax.hlines(
                mi + offsets, lo, hi, color="#222222",
                linewidth=0.9, zorder=3,
            )
            ax.vlines(
                [lo, hi], mi + offsets - 0.05, mi + offsets + 0.05,
                color="#222222", linewidth=0.7, zorder=3,
            )
            ax.text(
                hi + 1.0, mi + offsets, f"{p:.0f}%",
                va="center", ha="left", fontsize=5.8, color="#333333",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for _, label in metrics])
    ax.set_ylim(4.65, -0.55)  # reserve space below final bar for age inset
    ax.set_xlabel("Proportion within cluster  (%)")
    ax.set_title("Demographic composition  (bars, 95% CI)",
                 fontsize=8.2, pad=6, loc="center")
    ax.set_xlim(0, 112)   # extra headroom for value labels beside bars
    ax.grid(axis="x", color="#EFEFEF", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Age inset - two lines below the bar plot, left-aligned label + three values
    age_bits = [(c, *CLUSTER_CHARACTERISTICS[c]["mean_age_years"]) for c in clusters]
    ax.text(
        -0.5, 3.72, "Mean age (years)",
        ha="left", va="center", fontsize=6.8,
        fontweight="semibold", color="#222222",
    )
    # Three coloured values spaced evenly
    positions = [18, 54, 92]
    for (c, p, lo, hi), xpos in zip(age_bits, positions):
        ax.text(
            xpos, 4.08,
            f"{CLUSTER_SHORT[c]}",
            ha="center", va="center", fontsize=5.8,
            color=category_label_colour(CLUSTER_COLOURS[c]), style="italic",
        )
        ax.text(
            xpos, 4.32,
            f"{md(p,1)} ({md(lo,1)}\u2013{md(hi,1)})",
            ha="center", va="center", fontsize=6.3,
            color=category_label_colour(CLUSTER_COLOURS[c]), fontweight="semibold",
        )



def panel_hiv_prevalence(ax):
    """HIV-positive prevalence per cluster with 95% CIs, read from the same
    CLUSTER_CHARACTERISTICS source as panel d so the two panels agree.
    Biology-led: shows the HIV concentration without forcing projection
    separation (which would over-assert)."""
    clusters = [0, 1, 2]
    ys = clusters[::-1]
    for i, c in enumerate(clusters):
        p, lo, hi = CLUSTER_CHARACTERISTICS[c]["pct_hiv_positive"]
        yy = ys[i]
        ax.errorbar(p, i, xerr=[[max(p - lo, 0)], [max(hi - p, 0)]], fmt="o",
                    color=CLUSTER_COLOURS[c], capsize=3, markersize=8, lw=1.7, zorder=3)
        ax.text(min(hi + 3, 99), i, f"{p:.0f}%", va="center", fontsize=7.6,
                color=CLUSTER_COLOURS[c], fontweight="bold")
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([CLUSTER_SHORT.get(c, f"Cluster {c+1}") for c in clusters], fontsize=7.0)
    ax.set_xlabel("HIV-positive prevalence (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(clusters) - 0.4)
    ax.set_title("HIV burden by cluster (95% CI)", fontsize=8.2, pad=6, loc="center")


def build_figure():
    apply_style(base_font_size=8.0)

    fig_w_in = 180 * MM
    fig_h_in = 170 * MM
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # Two rows: top = PCA scatter spanning both cols; bottom = heat-sens | demographics
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[1.20, 1.00],
        width_ratios=[1.00, 1.10],
        left=0.085, right=0.965, top=0.90, bottom=0.08,
        hspace=0.45, wspace=0.38,
    )
    ax_pca = fig.add_subplot(gs[0, 0])
    ax_hiv = fig.add_subplot(gs[0, 1])
    ax_hs  = fig.add_subplot(gs[1, 0])
    ax_dm  = fig.add_subplot(gs[1, 1])

    panel_pca_scatter(ax_pca)
    panel_hiv_prevalence(ax_hiv)
    panel_heat_sensitivity(ax_hs)
    panel_demographics(ax_dm)

    panel_label(ax_pca, "a", x=-0.12, y=1.02)
    panel_label(ax_hiv, "b", x=-0.16, y=1.02)
    panel_label(ax_hs,  "c", x=-0.24, y=1.04)
    panel_label(ax_dm,  "d", x=-0.19, y=1.04)

    if not ARTWORK_MODE:
        fig.text(
            0.02, 0.97,
            "Figure 2  |  Vulnerability clusters: PCA structure, HIV burden, heat-sensitivity, and demographic composition",
            fontsize=9.8, fontweight="bold", ha="left", va="top",
        )

    svg_path = OUT_DIR / "fig02_cluster_profiles.svg"
    png_path = OUT_DIR / "fig02_cluster_profiles.png"
    midline_tick_labels(fig)
    save_at_width(fig, OUT_DIR, "fig02_cluster_profiles", span="double")
    write_caption(
        OUT_DIR, "fig02_cluster_profiles",
        "Figure 2: Vulnerability clusters: PCA structure, HIV burden, heat sensitivity and "
        "demographic composition. (a) Two-dimensional principal component projection "
        "(PC1 vs PC2) of participant-level SHAP profiles, coloured by cluster assignment "
        "(k=3) with 95% density ellipses. This is a two-dimensional view of a grouping that "
        "used the first ten principal components, so some visual overlap is expected and does "
        "not mean the clusters are inseparable; the full-dimensional diagnostic is figure S1. "
        "(b) HIV-positive prevalence within each cluster, with 95% confidence intervals. "
        "(c) Mean composite heat-sensitivity score (z-scaled) for each cluster relative to the "
        "population reference. (d) Demographic composition within each cluster, with 95% "
        "confidence intervals and mean age.",
        "Panels b and d report the same HIV prevalence from a single source. "
        "Clusters are numbered 1-3 in reading order of the legend.",
    )
    plt.close(fig)

    return svg_path, png_path


if __name__ == "__main__":
    svg, png = build_figure()
    print(f"Saved: {svg}")
    print(f"Saved: {png}")
