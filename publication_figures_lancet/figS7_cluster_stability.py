"""Figure S7 - Cluster-method stability comparison.

Scientific purpose:
    The main-text Figure 2 reports a 3-cluster GMM solution fit on the top 10
    principal components of the Stage 1 biomarker profile. That choice reflects
    a methodological refinement motivated by the instability of the original
    all-component k-means pipeline, well below the 0.80 reproducibility
    threshold. This supplementary figure compares five candidate clustering
    strategies along two axes - within-sample stability (500-iteration ARI) and
    silhouette - and makes the case that PC10 + GMM + k=3 dominates on stability
    without sacrificing interpretability.

    All five candidates are measured on ONE profile. The published version mixed
    sources (four rows from the original ERA5 refinement at n=9,736, the primary
    row from the 13-biomarker ERA5-Land run), which is why its primary ARI of
    0.913 disagreed with Figures 2 and S1. Regenerate the inputs with
    rerun_figS7_selection_panel.py after any change to the Stage 3 panel.

Panels:
    a. ARI stability forest - mean 500-iteration ARI with 95% CI for each
       candidate method; the 0.80 reproducibility threshold is marked.
       Highlights the dominance of PC10_gmm_k3.
    b. Stability × silhouette scatter - each candidate plotted on the
       (silhouette, ARI) plane with bubble size ∝ cumulative PCA variance
       retained. The Pareto frontier is annotated.
    c. Cluster-size composition - stacked bar showing the three or four
       partition sizes produced by each candidate; demonstrates that
       PC10_gmm_k3 gives a clinically interpretable split.
    d. Decision narrative text box - articulates why PC10_gmm_k3 was chosen
       as the primary specification.

Source data:
    config.OUTPUT_ROOT / RP2_REFINEMENT_DIR / refinement_summary.json
    (default stage3_refinement_r2_005, produced by rerun_figS7_selection_panel.py).
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

# ----------------------------------------------------------------------
# Candidate panel, read from the refinement run rather than frozen here.
# ----------------------------------------------------------------------
# The published panel mixed sources: four comparison rows came from the
# original ERA5 refinement (n=9,736) while the primary row carried the
# 13-biomarker ERA5-Land stability. Reading one refinement_summary.json means
# every row is measured on the same profile, and the panel cannot drift out of
# step with Figures 2 and S1 again.
import json as _json

_REFINE_SUBDIR = os.environ.get("RP2_REFINEMENT_DIR", "stage3_refinement_r2_005")


def _pretty(name: str) -> tuple[str, str]:
    """'PC10_gmm_k3' -> ('PC10 GMM k=3', 'PC10·GMM·k3')."""
    pc, meth, kk = name.split("_", 2)
    k = kk.lstrip("k")
    disp_meth = {"gmm": "GMM", "kmeans": "k-means"}.get(meth, meth)
    short_meth = {"gmm": "GMM", "kmeans": "km"}.get(meth, meth)
    return f"{pc} {disp_meth} k={k}", f"{pc}·{short_meth}·k{k}"


def _load_candidates() -> list[dict]:
    from mcd_pipeline import config as _cfg
    path = _cfg.OUTPUT_ROOT / _REFINE_SUBDIR / "refinement_summary.json"
    summary = _json.loads(path.read_text())
    rows = []
    for r in summary["all_results"]:
        disp, short = _pretty(r["name"])
        primary = bool(r.get("is_primary"))
        rows.append({
            "name": disp + ("  (primary)" if primary else ""),
            "short": short,
            "n_dims": r["n_dims"],
            "cum_var": r["cum_var"],
            "method": r["method"],
            "k": r["k"],
            "silhouette": r["silhouette"],
            "mean_ari": r["mean_ari"],
            "std_ari": r["std_ari"],
            "ci_lower": r["ci_lower"],
            "ci_upper": r["ci_upper"],
            "sizes": sorted(r["sizes"], reverse=True),
            "is_primary": primary,
            "is_original": bool(r.get("is_original")),
        })
    rows.sort(key=lambda c: c["mean_ari"])   # panel a reads bottom-up by ARI
    return rows, summary


CANDIDATES, REFINEMENT = _load_candidates()
_PRIMARY = next(c for c in CANDIDATES if c["is_primary"])

ARI_THRESHOLD = 0.80

# Cluster palette from the main text (consistent with Fig 2)
CLUSTER_PALETTE_3 = ["#2A9D8F", "#E89A5A", "#C94F7C"]   # teal / ochre / rose
CLUSTER_PALETTE_4 = ["#2A9D8F", "#E89A5A", "#C94F7C", "#6A3D9A"]

METHOD_FILL = {
    "kmeans": "#A6CEE3",   # light blue
    "gmm":    "#FB9A99",   # light rose
}
METHOD_EDGE = {
    "kmeans": "#1F78B4",
    "gmm":    "#C94F7C",
}


def build_figure():
    apply_style(base_font_size=8.0)

    fig_w_in = 190 * MM
    fig_h_in = 220 * MM
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    outer = fig.add_gridspec(
        nrows=2, ncols=2,
        left=0.115, right=0.975, top=0.840, bottom=0.070,
        height_ratios=[1.00, 0.95],
        width_ratios=[1.10, 1.00],
        hspace=0.60, wspace=0.50,
    )

    # ------------------------------------------------------------------
    # Panel a: ARI stability forest
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(outer[0, 0])
    n = len(CANDIDATES)
    y_pos = np.arange(n)
    for yi, c in enumerate(CANDIDATES):
        fill = METHOD_FILL[c["method"]]
        edge = METHOD_EDGE[c["method"]]
        lw = 1.4 if c["is_primary"] else 0.5
        # CI whisker
        ax_a.plot([c["ci_lower"], c["ci_upper"]], [yi, yi],
                  color="#555555", linewidth=0.8, zorder=2,
                  solid_capstyle="round")
        # Mean dot
        marker_size = 90 if c["is_primary"] else 55
        ax_a.scatter(c["mean_ari"], yi, s=marker_size,
                     facecolor=fill, edgecolor=edge,
                     linewidth=lw, zorder=4,
                     marker="D" if c["is_primary"] else "o")
        # Label
        fontweight = "bold" if c["is_primary"] else "normal"
        col = "#C94F7C" if c["is_primary"] else ("#999999" if c["is_original"] else "#333333")
        ax_a.text(
            -0.05, yi, c["short"],
            fontsize=7.2, fontweight=fontweight, color=col,
            ha="right", va="center",
            transform=ax_a.get_yaxis_transform(),
        )
        # Right-side numeric (two-line: mean on top, CI below for readability)
        ax_a.text(
            1.02, yi + 0.12, f"{c['mean_ari']:.3f}",
            fontsize=7.0, color=col, fontweight=fontweight,
            ha="left", va="center",
            transform=ax_a.get_yaxis_transform(),
        )
        ax_a.text(
            1.02, yi - 0.22, f"[{c['ci_lower']:.2f}, {c['ci_upper']:.2f}]",
            fontsize=6.2, color="#666666",
            ha="left", va="center",
            transform=ax_a.get_yaxis_transform(),
        )

    ax_a.axvline(ARI_THRESHOLD, color="#C94F7C", linewidth=0.9,
                 linestyle="--", alpha=0.65, zorder=1)
    ax_a.text(ARI_THRESHOLD - 0.01, n - 0.4,
              "reproducibility\nthreshold 0.80",
              fontsize=6.3, color="#9A3A5E", ha="right", va="top",
              fontstyle="italic")

    ax_a.set_yticks([])
    ax_a.set_xlim(-0.05, 1.05)
    ax_a.set_ylim(-0.7, n - 0.3)
    ax_a.set_xlabel("Mean ARI across 500 resamples (95% CI)", fontsize=7.4)
    ax_a.grid(axis="x", color="#F0F0F0", linewidth=0.4, zorder=0)
    ax_a.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax_a.spines[side].set_visible(False)
    ax_a.set_title(
        "Bootstrap stability (ARI) across 5 candidate clusterings",
        fontsize=8.0, fontweight="semibold", color="#333333",
        pad=4, loc="left",
    )
    panel_label(ax_a, "a", x=-0.22, y=1.06, fontsize=11)

    # Method legend
    leg_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=METHOD_FILL["kmeans"],
                   markeredgecolor=METHOD_EDGE["kmeans"],
                   markersize=6, label="k-means"),
        plt.Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=METHOD_FILL["gmm"],
                   markeredgecolor=METHOD_EDGE["gmm"],
                   markersize=6, label="GMM"),
        plt.Line2D([0], [0], marker="D", linestyle="None",
                   markerfacecolor=METHOD_FILL["gmm"],
                   markeredgecolor=METHOD_EDGE["gmm"],
                   markersize=6, markeredgewidth=1.4, label="primary"),
    ]
    ax_a.legend(handles=leg_handles, loc="lower left", frameon=False,
                fontsize=6.6, handlelength=0.9, borderaxespad=0.4)

    # ------------------------------------------------------------------
    # Panel b: stability × silhouette scatter
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(outer[0, 1])
    for c in CANDIDATES:
        fill = METHOD_FILL[c["method"]]
        edge = METHOD_EDGE[c["method"]]
        size = 80 + c["cum_var"] * 400  # var-retained encoded in bubble
        marker = "D" if c["is_primary"] else "o"
        lw = 1.5 if c["is_primary"] else 0.6
        ax_b.scatter(c["silhouette"], c["mean_ari"], s=size,
                     facecolor=fill, edgecolor=edge, linewidth=lw,
                     alpha=0.85, marker=marker, zorder=4)

    # Label each point
    # Keyed by the short label so it survives the leading-PC count changing
    # with the run (PC56 under ERA5, PC43 on the ten-biomarker profile).
    B_LABEL_OFFSETS = {
        "PC5·GMM·k3":  (+0.016, -0.012, "left",  "top"),
        "PC5·km·k4":   (+0.014, -0.030, "left",  "top"),
        "PC5·km·k3":   (+0.014, -0.028, "left",  "top"),
        "PC10·GMM·k3": (+0.020, +0.012, "left",  "bottom"),
    }
    for c in CANDIDATES:
        dx, dy, ha, va = B_LABEL_OFFSETS.get(
            c["short"], (+0.012, -0.045, "left", "top")
        )
        fw = "bold" if c["is_primary"] else "normal"
        col = "#C94F7C" if c["is_primary"] else "#222222"
        ax_b.text(
            c["silhouette"] + dx, c["mean_ari"] + dy,
            c["short"],
            fontsize=6.5, color=col, fontweight=fw,
            ha=ha, va=va,
        )

    # Threshold lines
    ax_b.axhline(ARI_THRESHOLD, color="#C94F7C", linewidth=0.8,
                 linestyle="--", alpha=0.55, zorder=1)
    ax_b.text(0.435, ARI_THRESHOLD + 0.008, "ARI = 0.80",
              fontsize=6.2, color="#9A3A5E", ha="right", va="bottom",
              fontstyle="italic")

    ax_b.set_xlabel("Silhouette score", fontsize=7.4)
    ax_b.set_ylabel("Mean ARI (500 resamples)", fontsize=7.4)
    ax_b.set_xlim(0.05, 0.44)
    ax_b.set_ylim(0.20, 1.10)
    ax_b.grid(color="#F0F0F0", linewidth=0.4, zorder=0)
    ax_b.set_axisbelow(True)
    for side in ("top", "right"):
        ax_b.spines[side].set_visible(False)
    ax_b.set_title(
        "Stability × silhouette (bubble size = PCA variance retained)",
        fontsize=8.0, fontweight="semibold", color="#333333",
        pad=4, loc="left",
    )
    panel_label(ax_b, "b", x=-0.15, y=1.06, fontsize=11)

    # ------------------------------------------------------------------
    # Panel c: cluster-size composition (stacked bar)
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(outer[1, 0])
    n_cand = len(CANDIDATES)
    bar_h = 0.68
    ypos = np.arange(n_cand)
    for yi, c in enumerate(CANDIDATES):
        sizes = np.array(c["sizes"])
        total = sizes.sum()
        props = sizes / total
        # Sort clusters descending for stable visual ordering
        order = np.argsort(-sizes)
        cum = 0.0
        palette = CLUSTER_PALETTE_4 if c["k"] == 4 else CLUSTER_PALETTE_3
        for idx, oi in enumerate(order):
            p = props[oi]
            col = palette[idx % len(palette)]
            ax_c.barh(yi, p, left=cum, height=bar_h,
                      color=col, edgecolor="#222222", linewidth=0.3,
                      zorder=3)
            # Annotate cluster size — only where the segment is wide enough
            # to hold the two-line label without colliding with its neighbour.
            if p >= 0.08:
                ax_c.text(
                    cum + p / 2, yi,
                    f"n={sizes[oi]:,}\n{p*100:.0f}%",
                    fontsize=6.3, color="white",
                    ha="center", va="center", fontweight="semibold",
                    zorder=4,
                )
            cum += p

    ax_c.set_yticks(ypos)
    ax_c.set_yticklabels(
        [f"{c['short']}  (k={c['k']})" for c in CANDIDATES],
        fontsize=7.2,
    )
    for tick, c in zip(ax_c.get_yticklabels(), CANDIDATES):
        if c["is_primary"]:
            tick.set_color("#C94F7C")
            tick.set_fontweight("bold")
        elif c["is_original"]:
            tick.set_color("#999999")
    ax_c.set_xlim(0, 1.0)
    ax_c.set_xlabel("Partition proportion", fontsize=7.4)
    ax_c.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_c.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=7.0)
    ax_c.grid(axis="x", color="#F0F0F0", linewidth=0.4, zorder=0)
    ax_c.set_axisbelow(True)
    for side in ("top", "right"):
        ax_c.spines[side].set_visible(False)
    ax_c.set_title(
        "Cluster-size composition (ranked largest → smallest)",
        fontsize=8.0, fontweight="semibold", color="#333333",
        pad=4, loc="left",
    )
    panel_label(ax_c, "c", x=-0.22, y=1.06, fontsize=11)

    # ------------------------------------------------------------------
    # Panel d: decision narrative
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(outer[1, 1])
    ax_d.set_xticks([]); ax_d.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax_d.spines[side].set_visible(False)

    ax_d.text(0.0, 1.00, "Why PC10 + GMM + k=3",
              fontsize=9.4, fontweight="bold", color="#333333",
              transform=ax_d.transAxes, va="top")

    # Narrative numbers follow the loaded panel so they cannot contradict it.
    _prim = next(c for c in CANDIDATES if c["is_primary"])
    _orig = next((c for c in CANDIDATES if c["is_original"]), None)
    _others = [c for c in CANDIDATES if not c["is_primary"] and not c["is_original"]]
    _lo, _hi = (min(c["mean_ari"] for c in _others),
                max(c["mean_ari"] for c in _others)) if _others else (0.0, 0.0)

    y0 = 0.91
    items = [
        (f"Stability ARI {_prim['mean_ari']:.3f}",
         f"above the 0.80 reproducibility threshold under ERA5-Land - comparator "
         f"candidates lie in the {_lo:.2f}-{_hi:.2f} range with wide CIs."),
        (f"Original {_orig['short'].split(chr(183))[0]} k-means ARI {_orig['mean_ari']:.3f}"
         if _orig else "Original all-component k-means",
         "failed reproducibility; refinement was motivated by this explicit quantitative failure, not post-hoc tuning."),
        ("GMM soft boundaries",
         "allow probabilistic cluster assignment, consistent with continuous vulnerability scoring used in the supplementary."),
        ("k = 3 is interpretable",
         "splits into HIV-positive progressive, HIV-negative metabolic, and low-vulnerability strata; k = 4 adds noise."),
        (f"PC10 retains {_prim['cum_var']*100:.1f} % variance",
         f"sufficient signal for clustering while avoiding the curse-of-dimensionality "
         f"penalty suffered by {_orig['short'].split(chr(183))[0] if _orig else 'the full PC set'}."),
    ]
    for head, body in items:
        ax_d.text(0.00, y0, head, fontsize=7.4, fontweight="semibold",
                  color="#C94F7C", transform=ax_d.transAxes, va="top")
        y0 -= 0.05
        ax_d.text(0.03, y0, body, fontsize=6.6, color="#333333",
                  transform=ax_d.transAxes, va="top",
                  wrap=True, linespacing=1.3)
        # Approximate line count for spacing
        n_chars = len(body)
        est_lines = max(1, int(np.ceil(n_chars / 48)))
        y0 -= 0.058 * est_lines + 0.02

    ax_d.text(
        0.0, y0 - 0.01,
        "Primary methods cite the 500-iteration ARI benchmark\n"
        "and reference this figure for full transparency.",
        fontsize=6.4, color="#777777", fontstyle="italic",
        transform=ax_d.transAxes, va="top", linespacing=1.3,
    )
    panel_label(ax_d, "d", x=-0.05, y=1.06, fontsize=11)

    # ------------------------------------------------------------------
    # Titles
    # ------------------------------------------------------------------
    if not ARTWORK_MODE:
        fig.text(
            0.02, 0.982,
            "Figure S7  |  Cluster-method stability comparison",
            fontsize=10.0, fontweight="bold", ha="left", va="top",
        )
    fig.text(
        0.02, 0.963,
        "500-iteration bootstrap stability (Adjusted Rand Index) across five candidate clusterings of the Stage 1 biomarker profile.",
        fontsize=7.2, color="#444444", ha="left", va="top",
    )
    fig.text(
        0.02, 0.947,
        f"The primary method (PC10 + GMM + k=3) dominates on stability "
        f"(ARI {_PRIMARY['mean_ari']:.3f}, ERA5-Land) and produces a clinically interpretable 3-cluster split.",
        fontsize=7.2, color="#444444", ha="left", va="top",
    )
    fig.text(
        0.02, 0.929,
        "Dashed line marks the 0.80 reproducibility threshold; 95 % CIs are bootstrap-derived over 500 resamples of the patient × PC matrix.",
        fontsize=7.2, color="#444444", ha="left", va="top",
    )
    fig.text(
        0.02, 0.907,
        f"All five candidates fitted on the same profile ({REFINEMENT['source_run']}; "
        f"n={REFINEMENT['n_patients']:,}, {REFINEMENT['n_biomarkers']} biomarkers, "
        f"{REFINEMENT['n_components_total']} components), "
        f"{REFINEMENT['n_bootstrap_iterations']}-iteration bootstrap, seed {REFINEMENT['seed']}.",
        fontsize=6.6, color="#777777", fontstyle="italic",
        ha="left", va="top",
    )

    svg_path = OUT_DIR / "figS7_cluster_stability.svg"
    png_path = OUT_DIR / "figS7_cluster_stability.png"
    midline_tick_labels(fig)
    save_at_width(fig, OUT_DIR, "figS7_cluster_stability", span="double")
    plt.close(fig)
    return svg_path, png_path


if __name__ == "__main__":
    svg, png = build_figure()
    print(f"Saved: {svg}")
    print(f"Saved: {png}")
