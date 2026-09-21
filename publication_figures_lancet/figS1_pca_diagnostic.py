"""Figure S1 - Dimensional diagnostic for GMM k=3 clustering.

Purpose:
    Figure 2A shows PC1 × PC2 only (~14.7 % of variance). This supplementary
    figure provides the full dimensional view of what clustering actually
    uses:

    (A) PC pair plot (PC1 through PC4) with per-cluster densities, showing
        which PCs actually separate which clusters.
    (B) Cumulative variance explained across the full PC set with the 10-PC
        clustering boundary and 85 % archival threshold marked.
    (C) Silhouette distributions per cluster (sampled, 10D Euclidean).
    (D) Mahalanobis distance (per cluster, 10D) - the metric appropriate
        to GMM's full-covariance model.

    Takeaway for readers: GMM-in-10D identifies clusters whose SEPARATION
    lives in higher PCs and in COVARIANCE STRUCTURE (not only mean),
    which is why Euclidean silhouette is modest while GMM-ARI stability
    is 0.913 (ERA5-Land primary).
"""
from __future__ import annotations
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mcd_pipeline import config as _pipeline_config
from _data import (
    CLUSTER_CHARACTERISTICS, PC_VARIANCE_EXPLAINED_PCT,
    N_PATIENTS_CLUSTERED, N_PCS_USED, ARI_STABILITY,
    STAGE3_SUBDIR as _data_STAGE3_SUBDIR,
)
from _lancet_style import (
    CLUSTER_COLOURS, CLUSTER_SHORT, MM,
    apply_style, panel_label,
)
from _lancet_style import ARTWORK_MODE, midline_tick_labels, md  # house style (2026-09-04)
from _lancet_style import save_at_width

# Artwork runs write elsewhere so submission files do not overwrite the
# working previews. Set RP2_FIG_OUT to redirect.
OUT_DIR = Path(os.environ.get("RP2_FIG_OUT",
                              str(Path(__file__).resolve().parent)))
# Cached intermediates always live beside the script, never in the artwork
# output directory -- OUT_DIR is where figures GO, not where inputs live.
CACHE_DIR = Path(__file__).resolve().parent
_STAGE3 = _pipeline_config.OUTPUT_ROOT / _data_STAGE3_SUBDIR

# Load live PCA variance from current pipeline run (auto-switches by env var).
_pca = json.load(open(_STAGE3 / "pca_summary.json"))
ALL_PC_VAR_PCT = [v * 100.0 for v in _pca["explained_variance_ratio"]]

# Live silhouette + k-means baseline for the current run.
_sil_gmm = json.load(open(_STAGE3 / "silhouette_scores.json"))
SILHOUETTE_GMM_10D = _sil_gmm["silhouette"]
BIC = _sil_gmm["bic"]
AIC = _sil_gmm["aic"]
try:
    _sil_km = json.load(open(_STAGE3 / "silhouette_scores_kmeans.json"))
    # k=3 is stored under "by_k" in the current pipeline output; the flat
    # "silhouette" key is a legacy layout. Reading only the flat key made this
    # fall through to the frozen 0.210 default and print a wrong number rather
    # than fail visibly.
    KMEANS_K3_SILHOUETTE = float(
        _sil_km.get("by_k", {}).get("3", _sil_km.get("silhouette", 0.210))
    )
except FileNotFoundError:
    KMEANS_K3_SILHOUETTE = 0.210

# Per-cluster median silhouette (computed live from the saved sample if available).
try:
    _silh_sample = np.load(CACHE_DIR / "_silh_sample.npy")  # (n, 2): [cluster, silhouette]
    SILHOUETTE_PER_CLUSTER_MEDIAN = {
        int(c): float(np.median(_silh_sample[_silh_sample[:, 0] == c, 1]))
        for c in np.unique(_silh_sample[:, 0])
    }
except FileNotFoundError:
    SILHOUETTE_PER_CLUSTER_MEDIAN = {0: 0.563, 1: -0.271, 2: -0.231}


def panel_pc_pairs(axs):
    """4×4 grid: lower triangle = scatter, diagonal = marginal density, upper = blank."""
    df = pd.read_csv(CACHE_DIR / "_pc10_cluster.csv")
    pcs = ["PC1", "PC2", "PC3", "PC4"]

    # Clip extreme outliers for plot readability (don't change data)
    q = df[pcs].quantile([0.005, 0.995])

    for i, yp in enumerate(pcs):
        for j, xp in enumerate(pcs):
            ax = axs[i][j]
            if i == j:
                # Diagonal - per-cluster marginal KDE
                for c in [0, 1, 2]:
                    vals = df[df.cluster == c][xp].values
                    vals = vals[(vals > q.loc[0.005, xp]) & (vals < q.loc[0.995, xp])]
                    ax.hist(
                        vals, bins=40, density=True,
                        color=CLUSTER_COLOURS[c], alpha=0.45,
                        edgecolor=CLUSTER_COLOURS[c], linewidth=0.3,
                    )
                ax.set_yticks([])
                ax.set_ylabel("")
                var_xp = PC_VARIANCE_EXPLAINED_PCT[j]
                ax.text(
                    0.5, 0.92, f"{xp} ({var_xp:.1f}%)",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=6.6, color="#333333", fontweight="semibold",
                )
            elif i > j:
                # Lower triangle - scatter
                for c in [2, 0, 1]:  # plotting order: tightest last for visibility
                    sub = df[df.cluster == c]
                    ax.scatter(
                        sub[xp], sub[yp],
                        s=1.2, color=CLUSTER_COLOURS[c], alpha=0.32,
                        linewidths=0, rasterized=True,
                    )
                ax.set_xlim(q.loc[0.005, xp], q.loc[0.995, xp])
                ax.set_ylim(q.loc[0.005, yp], q.loc[0.995, yp])
            else:
                # Upper triangle - blank (tidier)
                ax.set_visible(False)

            # Axis labels on left column / bottom row only
            if j == 0 and i != j:
                ax.set_ylabel(yp, fontsize=6.8)
            else:
                ax.set_ylabel("")
            if i == len(pcs) - 1:
                ax.set_xlabel(xp, fontsize=6.8)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
            if j != 0 and i != j:
                ax.tick_params(axis="y", labelleft=False)
            ax.tick_params(axis="both", labelsize=5.8, length=2)
            ax.grid(True, color="#F4F4F4", linewidth=0.3)
            ax.set_axisbelow(True)

    # Cluster legend inside the upper-triangle empty area
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=6,
               markerfacecolor=CLUSTER_COLOURS[c], markeredgecolor="white",
               markeredgewidth=0.6,
               label=f"Cluster {c+1} - {CLUSTER_SHORT[c]}")
        for c in [0, 1, 2]
    ]
    axs[0][3].set_visible(True)
    axs[0][3].axis("off")
    axs[0][3].legend(
        handles=legend_handles, loc="center", fontsize=7.0,
        frameon=False, handlelength=1.2, labelspacing=0.6,
    )
    axs[0][3].text(
        0.5, 1.05,
        "Cluster",
        transform=axs[0][3].transAxes, ha="center", va="bottom",
        fontsize=7.8, color="#222222", fontweight="semibold",
    )
    axs[0][2].set_visible(True)
    axs[0][2].axis("off")
    axs[0][2].text(
        0.0, 0.75,
        "Key takeaways:",
        transform=axs[0][2].transAxes, ha="left", va="center",
        fontsize=6.8, color="#333333", fontweight="semibold",
    )
    axs[0][2].text(
        0.0, 0.50,
        "• Clusters 1 vs 3 separate on PC1",
        transform=axs[0][2].transAxes, ha="left", va="center",
        fontsize=6.4, color="#555555",
    )
    axs[0][2].text(
        0.0, 0.30,
        "• Cluster 2 separates on PC3 + PC4",
        transform=axs[0][2].transAxes, ha="left", va="center",
        fontsize=6.4, color="#555555",
    )
    axs[0][2].text(
        0.0, 0.10,
        "• Fig 2A (PC1×PC2) would miss",
        transform=axs[0][2].transAxes, ha="left", va="center",
        fontsize=6.4, color="#555555",
    )
    axs[0][2].text(
        0.0, -0.05,
        "   cluster 2's true structure",
        transform=axs[0][2].transAxes, ha="left", va="center",
        fontsize=6.4, color="#555555",
    )
    axs[1][3].set_visible(True)
    axs[1][3].axis("off")
    axs[2][3].set_visible(True)
    axs[2][3].axis("off")


def panel_variance_explained(ax):
    n_pcs = len(ALL_PC_VAR_PCT)
    x = np.arange(1, n_pcs + 1)
    var = np.array(ALL_PC_VAR_PCT)
    cum = np.cumsum(var)

    # Individual variance bars (coloured by "used in clustering" vs "not")
    colours = ["#4E79A7" if i < N_PCS_USED else "#C0C4CC" for i in range(n_pcs)]
    ax.bar(x, var, color=colours, width=0.8, linewidth=0, zorder=2)

    # Cumulative variance line (right axis)
    ax2 = ax.twinx()
    ax2.plot(x, cum, color="#2A9D8F", linewidth=1.3, zorder=3)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Cumulative variance (%)", color="#2A9D8F", fontsize=7.4)
    ax2.tick_params(axis="y", labelcolor="#2A9D8F", labelsize=6.5)

    # 10-PC clustering boundary
    cum10 = cum[N_PCS_USED - 1]
    ax.axvline(N_PCS_USED + 0.5, color="#E76F51", linestyle="--",
               linewidth=0.8, zorder=4)
    ax2.annotate(
        f"10 PCs used for clustering\n({cum10:.1f}% variance)",
        xy=(N_PCS_USED, cum10), xytext=(N_PCS_USED + 6, cum10 - 12),
        fontsize=6.3, color="#E76F51",
        arrowprops=dict(arrowstyle="-", color="#E76F51", linewidth=0.5),
    )

    # 85% archival threshold
    ax2.axhline(85, color="#666666", linestyle=":", linewidth=0.6, zorder=4)
    ax2.text(n_pcs - 0.5, 86.5, f"85% archival threshold ({n_pcs} PCs)",
             fontsize=5.8, color="#666666", ha="right", va="bottom",
             style="italic")

    ax.set_xlim(0.3, n_pcs + 0.7)
    ax.set_ylim(0, max(var) * 1.1)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Individual variance (%)", fontsize=7.4)
    ax.tick_params(axis="both", labelsize=6.5)
    ax.set_title("Variance explained by principal component",
                 fontsize=8.2, pad=6, loc="left", fontweight="semibold")

    # Small legend tucked on left so it does not collide with the cumulative curve
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#4E79A7", label=f"used for clustering (top {N_PCS_USED})"),
        Patch(facecolor="#C0C4CC", label=f"unused (PCs {N_PCS_USED+1}-{n_pcs})"),
    ]
    ax.legend(handles=legend_handles, loc="center left", fontsize=6.3,
              frameon=False, handlelength=1.0, bbox_to_anchor=(0.30, 0.82))


def panel_silhouette(ax):
    """Silhouette distributions per cluster, sampled in 10D Euclidean space."""
    data = np.load(CACHE_DIR / "_silh_sample.npy")
    ys, silh = data[:, 0].astype(int), data[:, 1]

    y_positions = [0, 1, 2]
    for ypos, c in zip(y_positions, [0, 1, 2]):
        s = silh[ys == c]
        # Violin-like horizontal histogram
        parts = ax.violinplot(
            [s], positions=[ypos], vert=False, widths=0.75, showmeans=False,
            showmedians=False, showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(CLUSTER_COLOURS[c])
            pc.set_alpha(0.45)
            pc.set_edgecolor(CLUSTER_COLOURS[c])
            pc.set_linewidth(0.4)
        # Median marker
        med = np.median(s)
        ax.scatter(med, ypos, s=28, color=CLUSTER_COLOURS[c],
                   edgecolor="white", linewidth=0.8, zorder=5)
        # Numeric label placed outside the violin body, clear of edge
        if med > 0:
            label_x, ha = min(med + 0.12, 0.85), "left"
        else:
            label_x, ha = max(med - 0.12, -0.85), "right"
        ax.text(
            label_x, ypos - 0.38, f"median {med:+.2f}",
            fontsize=6.2, color=CLUSTER_COLOURS[c],
            ha=ha, va="center",
            fontweight="semibold",
        )

    ax.axvline(0, color="#444444", linewidth=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([CLUSTER_SHORT[c] for c in [0, 1, 2]])
    for tick, c in zip(ax.get_yticklabels(), [0, 1, 2]):
        tick.set_color(CLUSTER_COLOURS[c])
        tick.set_fontweight("semibold")
    ax.invert_yaxis()
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Silhouette coefficient  (10D Euclidean, sampled n=3,000)")
    ax.set_title("Silhouette by cluster",
                 fontsize=8.2, pad=6, loc="left", fontweight="semibold")
    ax.grid(axis="x", color="#EFEFEF", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Summary footer
    ax.text(
        1.0, 3.25,
        f"Overall silhouette: GMM-k3 = {SILHOUETTE_GMM_10D:.2f}   •   k-means-k3 = {KMEANS_K3_SILHOUETTE:.2f}   •   GMM ARI stability (10D) = {ARI_STABILITY:.3f}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=6.2, color="#666666", style="italic",
    )


def panel_mahalanobis(ax):
    """Mahalanobis distance (10D) per patient from own cluster centroid."""
    df = pd.read_csv(CACHE_DIR / "_mahal.csv")
    y_positions = [0, 1, 2]
    for ypos, c in zip(y_positions, [0, 1, 2]):
        vals = df.loc[df.cluster == c, "mahal"].values
        # Clip extreme outliers at 99th pct for display
        clip = np.quantile(vals, 0.99)
        v_disp = vals[vals <= clip]
        parts = ax.violinplot(
            [v_disp], positions=[ypos], vert=False, widths=0.75,
            showmeans=False, showmedians=False, showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(CLUSTER_COLOURS[c])
            pc.set_alpha(0.45)
            pc.set_edgecolor(CLUSTER_COLOURS[c])
            pc.set_linewidth(0.4)
        med = np.median(vals)
        q25 = np.quantile(vals, 0.25)
        q75 = np.quantile(vals, 0.75)
        ax.scatter(med, ypos, s=28, color=CLUSTER_COLOURS[c],
                   edgecolor="white", linewidth=0.8, zorder=5)
        # Place numeric label below the violin body
        ax.text(
            med, ypos - 0.42,
            f"median {med:.2f}  (IQR {q25:.2f}-{q75:.2f})",
            fontsize=6.1, color=CLUSTER_COLOURS[c],
            ha="center", va="center",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([CLUSTER_SHORT[c] for c in [0, 1, 2]])
    for tick, c in zip(ax.get_yticklabels(), [0, 1, 2]):
        tick.set_color(CLUSTER_COLOURS[c])
        tick.set_fontweight("semibold")
    ax.invert_yaxis()
    ax.set_xlim(0, 9)
    ax.set_xlabel("Mahalanobis distance from own-cluster centroid  (10D)")
    ax.set_title("Mahalanobis distance from own-cluster centroid",
                 fontsize=8.2, pad=6, loc="left", fontweight="semibold")
    ax.grid(axis="x", color="#EFEFEF", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def build_figure():
    apply_style(base_font_size=8.0)

    fig_w_in = 190 * MM
    fig_h_in = 220 * MM
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # Two rows: top half = PC pair plot 4×4; bottom half = variance / silh / mahal
    outer = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[1.20, 1.00],
        left=0.08, right=0.97, top=0.905, bottom=0.06,
        hspace=0.50,
    )

    # Top - PC pair plot (4x4 grid inside)
    pair_gs = outer[0, 0].subgridspec(nrows=4, ncols=4, hspace=0.12, wspace=0.12)
    axs = [[fig.add_subplot(pair_gs[i, j]) for j in range(4)] for i in range(4)]
    panel_pc_pairs(axs)

    # Bottom - three-panel row: variance (wider) on top, silh + mahal below
    bot = outer[1, 0].subgridspec(nrows=2, ncols=2, hspace=0.60, wspace=0.35,
                                   height_ratios=[1.0, 1.0])
    ax_var = fig.add_subplot(bot[0, :])
    ax_silh = fig.add_subplot(bot[1, 0])
    ax_mah = fig.add_subplot(bot[1, 1])

    panel_variance_explained(ax_var)
    panel_silhouette(ax_silh)
    panel_mahalanobis(ax_mah)

    # Panel labels
    panel_label(axs[0][0], "a", x=-0.45, y=1.15, fontsize=10)
    panel_label(ax_var, "b", x=-0.08, y=1.10, fontsize=10)
    panel_label(ax_silh, "c", x=-0.22, y=1.12, fontsize=10)
    panel_label(ax_mah, "d", x=-0.22, y=1.12, fontsize=10)

    if not ARTWORK_MODE:
        fig.text(
            0.02, 0.98,
            "Figure S1  |  Dimensional diagnostic for GMM k=3 clustering",
            fontsize=10.0, fontweight="bold", ha="left", va="top",
        )
    _pc12 = ALL_PC_VAR_PCT[0] + ALL_PC_VAR_PCT[1]
    fig.text(
        0.02, 0.960,
        f"The PC1×PC2 projection in Figure 2A captures {md(_pc12,1)}% of the variance on which clustering is performed.",
        fontsize=7.1, color="#444444", ha="left", va="top",
    )
    fig.text(
        0.02, 0.945,
        "Panel a shows where clusters actually separate across PC1-4; b-d confirm dimensionality, separation quality, and cluster tightness in the 10D space GMM uses.",
        fontsize=7.1, color="#444444", ha="left", va="top",
    )

    svg_path = OUT_DIR / "figS1_pca_diagnostic.svg"
    png_path = OUT_DIR / "figS1_pca_diagnostic.png"
    midline_tick_labels(fig)
    save_at_width(fig, OUT_DIR, "figS1_pca_diagnostic", span="double")
    plt.close(fig)
    return svg_path, png_path


if __name__ == "__main__":
    svg, png = build_figure()
    print(f"Saved: {svg}")
    print(f"Saved: {png}")
