"""
Stage 3 — Clustering Parsimony Diagnostics
===========================================

Addresses the Reviewer 2 (Lancet Planetary Health) request for
clustering-parsimony evidence beyond the bootstrap ARI comparison
already reported in Table S1.

Computes:
  1. Tibshirani-Walther-Hastie (2001) gap statistic for k = 2..8 using
     uniform reference distributions on the GMM clustering subspace.
     Reports the standard 1-SE rule for choosing k as well as the raw
     elbow.
  2. PC loadings — the contribution of each biomarker SHAP profile to
     PC1, PC2, and PC3. Renders a heatmap and exports the loading
     matrix for the supplementary material so reviewers can interpret
     *what physiologically* differentiates the clusters.
  3. Per-cluster mean SHAP profile by biomarker, supporting clinician-
     driven cluster naming.

Inputs
------
mcd_outputs/stage3/pc_scores.csv             (run_stage3.py output)
mcd_outputs/stage3/pca_components.npy
mcd_outputs/stage3/pca_summary.json
mcd_outputs/stage3/cluster_labels.csv

Outputs
-------
mcd_outputs/stage3/parsimony/gap_statistic.csv
mcd_outputs/stage3/parsimony/gap_statistic.png
mcd_outputs/stage3/parsimony/pc_loadings.csv
mcd_outputs/stage3/parsimony/pc_loadings_heatmap.png
mcd_outputs/stage3/parsimony/per_cluster_biomarker_mean.csv
mcd_outputs/stage3/parsimony/parsimony_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from mcd_pipeline import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gap statistic
# ---------------------------------------------------------------------------


def _within_cluster_dispersion(X: np.ndarray, labels: np.ndarray) -> float:
    """Sum of within-cluster dispersion (W_k) used in the gap statistic."""
    total = 0.0
    for cid in np.unique(labels):
        members = X[labels == cid]
        if len(members) < 2:
            continue
        # Sum of pairwise squared distances divided by 2*n_r equals the
        # within-cluster sum of squares; scikit-learn's GMM does not expose
        # this directly so we compute it explicitly.
        diffs = members - members.mean(axis=0, keepdims=True)
        total += float(np.sum(diffs ** 2))
    return total


def _fit_gmm_labels(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fit a GMM with k components and return hard assignments."""
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=seed,
        n_init=3,
        max_iter=200,
    )
    return gmm.fit_predict(X)


def gap_statistic(
    X: np.ndarray,
    k_values: Iterable[int],
    n_references: int = 50,
    seed: int = config.MASTER_SEED,
) -> pd.DataFrame:
    """Tibshirani-Walther-Hastie (2001) gap statistic for GMM clustering.

    Reference distribution is uniform within the bounding box of X (the
    "uniform" choice in the original paper). Computed in the PCA-reduced
    clustering space for direct comparability with run_stage3.py.

    Returns
    -------
    DataFrame with one row per k:
        k, log_W_k, ref_log_W_k_mean, gap, sk, gap_minus_sk_next
    The 1-SE rule chooses the smallest k such that
        gap[k] >= gap[k+1] - sk[k+1].
    """
    rng = np.random.default_rng(seed)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    rows = []
    for k in k_values:
        labels = _fit_gmm_labels(X, k=k, seed=int(rng.integers(0, 2**31)))
        log_w = np.log(_within_cluster_dispersion(X, labels) + 1e-12)

        ref_log_w = np.empty(n_references)
        for i in range(n_references):
            X_ref = rng.uniform(low=mins, high=maxs, size=X.shape)
            ref_labels = _fit_gmm_labels(X_ref, k=k, seed=int(rng.integers(0, 2**31)))
            ref_log_w[i] = np.log(_within_cluster_dispersion(X_ref, ref_labels) + 1e-12)

        gap = float(np.mean(ref_log_w) - log_w)
        sd = float(np.std(ref_log_w, ddof=1))
        sk = sd * np.sqrt(1 + 1.0 / n_references)
        rows.append({
            "k": int(k),
            "log_W_k": float(log_w),
            "ref_log_W_k_mean": float(np.mean(ref_log_w)),
            "ref_log_W_k_sd": sd,
            "gap": gap,
            "sk": float(sk),
        })

        logger.info(
            "  Gap stat k=%d: log W=%.3f, gap=%.3f, sk=%.3f",
            k, log_w, gap, sk,
        )

    df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

    df["gap_next"] = df["gap"].shift(-1)
    df["sk_next"] = df["sk"].shift(-1)
    df["one_se_rule_satisfied"] = df["gap"] >= (df["gap_next"] - df["sk_next"])
    return df


def select_optimal_k_one_se(gap_df: pd.DataFrame) -> int | None:
    """Smallest k that satisfies the 1-SE rule, or argmax(gap) as fallback."""
    eligible = gap_df[gap_df["one_se_rule_satisfied"].fillna(False)]
    if len(eligible) > 0:
        return int(eligible["k"].iloc[0])
    return int(gap_df.loc[gap_df["gap"].idxmax(), "k"])


def _plot_gap(gap_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.errorbar(
        gap_df["k"], gap_df["gap"], yerr=gap_df["sk"],
        fmt="o-", color="#1f3a93", capsize=4, linewidth=1.4,
    )
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Gap statistic")
    ax.set_title("GMM gap statistic with reference SE bars")
    ax.set_xticks(sorted(gap_df["k"].unique()))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    optimal_k = select_optimal_k_one_se(gap_df)
    ax.axvline(optimal_k, color="#c0392b", linestyle="--", linewidth=1.0,
               label=f"1-SE rule k = {optimal_k}")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# PC loadings
# ---------------------------------------------------------------------------


def _load_biomarker_feature_names(stage3_dir: Path) -> list[str]:
    """Reconstruct the SHAP feature names from the PCA summary."""
    summary_path = stage3_dir / "pca_summary.json"
    if not summary_path.exists():
        return []
    with open(summary_path) as f:
        meta = json.load(f)
    biomarkers = meta.get("included_biomarkers") or []
    # The Stage 3 PCA is built from per-biomarker SHAP profiles concatenated
    # across the 7 lag windows, so the features are biomarker x lag pairs.
    # We label by biomarker only (averaging across lags) for legibility.
    return list(biomarkers)


def export_pc_loadings(
    stage3_dir: Path,
    n_pcs: int = 3,
) -> pd.DataFrame:
    """Export top-N PC loadings averaged within each biomarker."""
    components_path = stage3_dir / "pca_components.npy"
    if not components_path.exists():
        raise FileNotFoundError(f"pca_components.npy missing in {stage3_dir}")
    components = np.load(components_path)  # shape (n_components, n_features)

    biomarkers = _load_biomarker_feature_names(stage3_dir)
    n_features = components.shape[1]
    if not biomarkers:
        # Fallback to indexed labels if metadata is missing
        biomarker_labels = [f"feature_{i}" for i in range(n_features)]
    else:
        # Even split across biomarkers (each biomarker contributes the same
        # number of lag features in the SHAP fingerprint)
        per_bm = n_features // len(biomarkers)
        biomarker_labels = []
        for bm in biomarkers:
            biomarker_labels.extend([bm] * per_bm)
        # Pad if there is rounding error in the split
        while len(biomarker_labels) < n_features:
            biomarker_labels.append(biomarker_labels[-1])

    loading_df = pd.DataFrame(
        components[:n_pcs].T,
        columns=[f"PC{i+1}" for i in range(n_pcs)],
        index=biomarker_labels,
    )
    loading_df.index.name = "biomarker"

    # Mean absolute loading per biomarker per PC for the heatmap
    loading_summary = (
        loading_df.abs().groupby(loading_df.index).mean().sort_values("PC1", ascending=False)
    )
    return loading_summary


def _plot_loadings_heatmap(loadings: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 0.35 * len(loadings) + 1.5))
    im = ax.imshow(loadings.values, aspect="auto", cmap="RdBu_r",
                   vmin=-loadings.values.max(), vmax=loadings.values.max())
    ax.set_yticks(range(len(loadings)))
    ax.set_yticklabels(loadings.index, fontsize=9)
    ax.set_xticks(range(len(loadings.columns)))
    ax.set_xticklabels(loadings.columns, fontsize=9)
    ax.set_title("Mean |loading| of biomarker SHAP profile on PC1-3", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("Mean |loading|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-cluster biomarker means (clinician interpretability)
# ---------------------------------------------------------------------------


def per_cluster_biomarker_mean(
    stage3_dir: Path,
) -> pd.DataFrame | None:
    """Compute the per-cluster mean of patient-level SHAP fingerprints.

    Reads the PC scores rather than the underlying SHAP matrix because
    that is what was actually clustered. For interpretation we report
    each cluster's mean PC1, PC2, PC3 score so clinicians can inspect
    cluster-by-cluster differences in the explanation space.
    """
    pc_scores_path = stage3_dir / "pc_scores.csv"
    cluster_labels_path = stage3_dir / "cluster_labels.csv"
    if not (pc_scores_path.exists() and cluster_labels_path.exists()):
        return None
    _pc = pd.read_csv(pc_scores_path)
    pc_scores = _pc.set_index(_pc.columns[0])
    _cl = pd.read_csv(cluster_labels_path)
    cluster_labels = _cl.set_index(_cl.columns[0])
    cluster_labels.columns = ["cluster"]
    joined = pc_scores.join(cluster_labels, how="inner")
    means = joined.groupby("cluster").mean()
    return means


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_parsimony_diagnostics(
    stage3_dir: Path | None = None,
    k_values: Iterable[int] = range(2, 9),
    n_references: int = 50,
    n_pcs_for_loadings: int = 3,
    seed: int = config.MASTER_SEED,
) -> dict:
    """Run the parsimony diagnostics and write artefacts."""
    stage3_dir = stage3_dir or (config.OUTPUT_ROOT / "stage3")
    out_dir = stage3_dir / "parsimony"
    out_dir.mkdir(parents=True, exist_ok=True)

    pc_scores_path = stage3_dir / "pc_scores.csv"
    if not pc_scores_path.exists():
        raise FileNotFoundError(
            f"pc_scores.csv not found at {pc_scores_path}; run Stage 3 first"
        )
    _tmp = pd.read_csv(pc_scores_path)
    pc_scores = _tmp.set_index(_tmp.columns[0])

    n_dims = min(config.PCA_CLUSTERING_DIMS, pc_scores.shape[1])
    X_cluster = pc_scores.iloc[:, :n_dims].values
    logger.info(
        "Running gap statistic on %d patients in %d-dim PC space",
        X_cluster.shape[0], n_dims,
    )

    gap_df = gap_statistic(
        X=X_cluster, k_values=list(k_values),
        n_references=n_references, seed=seed,
    )
    gap_df.to_csv(out_dir / "gap_statistic.csv", index=False)
    _plot_gap(gap_df, out_dir / "gap_statistic.png")
    optimal_k = select_optimal_k_one_se(gap_df)

    loadings = export_pc_loadings(
        stage3_dir=stage3_dir, n_pcs=n_pcs_for_loadings,
    )
    loadings.to_csv(out_dir / "pc_loadings.csv")
    _plot_loadings_heatmap(loadings, out_dir / "pc_loadings_heatmap.png")

    per_cluster = per_cluster_biomarker_mean(stage3_dir=stage3_dir)
    if per_cluster is not None:
        per_cluster.to_csv(out_dir / "per_cluster_biomarker_mean.csv")

    summary = {
        "diagnostic": "stage3_parsimony_diagnostics",
        "n_patients": int(X_cluster.shape[0]),
        "n_clustering_dims": int(n_dims),
        "gap_statistic": {
            "k_values": list(map(int, k_values)),
            "n_references_per_k": int(n_references),
            "optimal_k_1se_rule": int(optimal_k),
            "argmax_gap_k": int(gap_df.loc[gap_df["gap"].idxmax(), "k"]),
            "table_csv": str(out_dir / "gap_statistic.csv"),
            "plot_png": str(out_dir / "gap_statistic.png"),
        },
        "pc_loadings": {
            "n_pcs_reported": int(n_pcs_for_loadings),
            "table_csv": str(out_dir / "pc_loadings.csv"),
            "heatmap_png": str(out_dir / "pc_loadings_heatmap.png"),
        },
        "per_cluster_pc_means_csv": (
            str(out_dir / "per_cluster_biomarker_mean.csv")
            if per_cluster is not None else None
        ),
    }
    with open(out_dir / "parsimony_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(
        "Parsimony diagnostics complete: 1-SE optimal k=%d, argmax-gap k=%d",
        summary["gap_statistic"]["optimal_k_1se_rule"],
        summary["gap_statistic"]["argmax_gap_k"],
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stage3-dir", type=Path, default=None)
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--n-references", type=int, default=50)
    p.add_argument("--n-pcs-loadings", type=int, default=3)
    p.add_argument("--seed", type=int, default=config.MASTER_SEED)
    return p


def main() -> int:
    from mcd_pipeline.utils.logging_config import configure_logging

    configure_logging()
    args = _build_arg_parser().parse_args()
    run_parsimony_diagnostics(
        stage3_dir=args.stage3_dir,
        k_values=range(args.k_min, args.k_max + 1),
        n_references=args.n_references,
        n_pcs_for_loadings=args.n_pcs_loadings,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
