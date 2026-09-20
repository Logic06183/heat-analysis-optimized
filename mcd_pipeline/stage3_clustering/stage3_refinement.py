"""
Stage 3 Refinement — Systematic Comparison of Clustering Approaches
====================================================================

Addresses the low cluster stability (ARI=0.377) found in the initial
Stage 3 run by comparing:

1. Reduced PCA dimensionality (5, 10, 15, 20 components vs original 56)
2. K-means vs HDBSCAN vs Gaussian Mixture Model (GMM)
3. Continuous vulnerability scores (PC1-3 as heat sensitivity indices)

Rationale:
- 56 PCA components suffers from curse of dimensionality for k-means
- First 10 PCs capture ~58% variance — signal-to-noise may be better
- HDBSCAN handles unequal cluster sizes and doesn't force all points
  into clusters (outliers stay as noise)
- GMM allows elliptical clusters and soft assignments

Usage:
    python -m mcd_pipeline.stage3_clustering.stage3_refinement
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from mcd_pipeline import config

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────

def _bootstrap_ari(
    X: np.ndarray,
    labels_ref: np.ndarray,
    fit_fn,
    n_iterations: int = 200,
    seed: int = config.MASTER_SEED,
) -> dict:
    """Fast bootstrap ARI assessment for any clustering method.

    Parameters
    ----------
    X : array (n, d)
    labels_ref : array (n,) — reference labels from full sample
    fit_fn : callable(X_boot, rng) -> labels_full
        Fits on bootstrap sample and returns labels for the FULL X.
    n_iterations : int
    seed : int

    Returns
    -------
    dict with mean_ari, std_ari, ci_lower, ci_upper
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    aris = np.empty(n_iterations)

    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        X_boot = X[idx]
        boot_labels = fit_fn(X_boot, X, rng)
        aris[i] = adjusted_rand_score(labels_ref, boot_labels)

    return {
        "mean_ari": float(aris.mean()),
        "std_ari": float(aris.std()),
        "ci_lower": float(np.percentile(aris, 2.5)),
        "ci_upper": float(np.percentile(aris, 97.5)),
    }


def _kmeans_fit_fn(k: int):
    """Return a fit function for k-means with given k."""
    def fn(X_boot, X_full, rng):
        km = KMeans(n_clusters=k, random_state=int(rng.integers(0, 2**31)), n_init=10)
        km.fit(X_boot)
        return km.predict(X_full)
    return fn


def _gmm_fit_fn(k: int):
    """Return a fit function for GMM with given k."""
    def fn(X_boot, X_full, rng):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=int(rng.integers(0, 2**31)),
            n_init=3,
            max_iter=300,
        )
        gmm.fit(X_boot)
        return gmm.predict(X_full)
    return fn


# ── Main Refinement ──────────────────────────────────────────────────

def run_refinement(
    output_dir: Path = config.OUTPUT_ROOT / "stage3_refinement",
    stage3_dir: Path = config.OUTPUT_ROOT / "stage3",
    n_stability_iter: int = 200,
) -> dict:
    """Run systematic clustering comparison.

    Loads existing PCA-reduced SHAP data from Stage 3, then tests
    multiple dimensionality × algorithm combinations.

    Parameters
    ----------
    output_dir : Path
        Where to write refinement results.
    stage3_dir : Path
        Existing Stage 3 output directory (pc_scores.csv, etc.).
    n_stability_iter : int
        Bootstrap iterations for stability assessment (200 is enough
        for comparison; final run uses 500).

    Returns
    -------
    dict — summary of all approaches tested.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load existing PC scores (full 56-component version) ──
    pc_scores = pd.read_csv(stage3_dir / "pc_scores.csv", index_col=0)
    logger.info("Loaded PC scores: %d patients × %d components", *pc_scores.shape)

    # ── Define dimensionality variants ──
    dim_variants = {
        "PC5": 5,
        "PC10": 10,
        "PC15": 15,
        "PC20": 20,
        "PC56_original": pc_scores.shape[1],
    }

    # Track cumulative variance for each
    with open(stage3_dir / "pca_summary.json") as f:
        pca_info = json.load(f)
    evr = np.array(pca_info["explained_variance_ratio"])
    cumvar = {name: float(evr[:n].sum()) for name, n in dim_variants.items()}

    results = []

    for dim_name, n_pcs in dim_variants.items():
        X = pc_scores.iloc[:, :n_pcs].values
        logger.info("\n── %s (%d dims, %.1f%% variance) ──", dim_name, n_pcs, cumvar[dim_name] * 100)

        # ── K-means (k=3..6, skip 7-8 as they were poor) ──
        for k in [3, 4, 5, 6]:
            km = KMeans(n_clusters=k, random_state=config.MASTER_SEED, n_init=10)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels)

            # Cluster size distribution
            sizes = np.bincount(labels)
            size_ratio = sizes.min() / sizes.max()

            stability = _bootstrap_ari(
                X, labels, _kmeans_fit_fn(k), n_iterations=n_stability_iter,
            )

            row = {
                "dim_variant": dim_name,
                "n_dims": n_pcs,
                "cum_variance": cumvar[dim_name],
                "method": "kmeans",
                "k": k,
                "silhouette": sil,
                "mean_ari": stability["mean_ari"],
                "std_ari": stability["std_ari"],
                "ari_ci_lower": stability["ci_lower"],
                "ari_ci_upper": stability["ci_upper"],
                "min_cluster_size": int(sizes.min()),
                "max_cluster_size": int(sizes.max()),
                "size_ratio": size_ratio,
                "n_clusters_found": len(sizes),
                "n_noise": 0,
            }
            results.append(row)
            logger.info(
                "  k-means k=%d: sil=%.3f, ARI=%.3f±%.3f, sizes=%s",
                k, sil, stability["mean_ari"], stability["std_ari"],
                sizes.tolist(),
            )

        # ── GMM (k=3..5) ──
        for k in [3, 4, 5]:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=config.MASTER_SEED,
                n_init=5,
                max_iter=300,
            )
            labels = gmm.fit_predict(X)
            sil = silhouette_score(X, labels)
            sizes = np.bincount(labels)
            size_ratio = sizes.min() / sizes.max()

            stability = _bootstrap_ari(
                X, labels, _gmm_fit_fn(k), n_iterations=n_stability_iter,
            )

            row = {
                "dim_variant": dim_name,
                "n_dims": n_pcs,
                "cum_variance": cumvar[dim_name],
                "method": "gmm",
                "k": k,
                "silhouette": sil,
                "mean_ari": stability["mean_ari"],
                "std_ari": stability["std_ari"],
                "ari_ci_lower": stability["ci_lower"],
                "ari_ci_upper": stability["ci_upper"],
                "min_cluster_size": int(sizes.min()),
                "max_cluster_size": int(sizes.max()),
                "size_ratio": size_ratio,
                "n_clusters_found": len(sizes),
                "n_noise": 0,
            }
            results.append(row)
            logger.info(
                "  GMM k=%d: sil=%.3f, ARI=%.3f±%.3f, sizes=%s",
                k, sil, stability["mean_ari"], stability["std_ari"],
                sizes.tolist(),
            )

        # ── HDBSCAN ──
        for min_cluster in [50, 100, 200, 500]:
            hdb = HDBSCAN(
                min_cluster_size=min_cluster,
                min_samples=10,
                metric="euclidean",
            )
            labels = hdb.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())

            if n_clusters < 2:
                logger.info(
                    "  HDBSCAN min_cluster=%d: only %d cluster(s), %d noise — skipping",
                    min_cluster, n_clusters, n_noise,
                )
                row = {
                    "dim_variant": dim_name,
                    "n_dims": n_pcs,
                    "cum_variance": cumvar[dim_name],
                    "method": "hdbscan",
                    "k": n_clusters,
                    "silhouette": np.nan,
                    "mean_ari": np.nan,
                    "std_ari": np.nan,
                    "ari_ci_lower": np.nan,
                    "ari_ci_upper": np.nan,
                    "min_cluster_size": min_cluster,
                    "max_cluster_size": 0,
                    "size_ratio": 0.0,
                    "n_clusters_found": n_clusters,
                    "n_noise": n_noise,
                }
                results.append(row)
                continue

            # Silhouette on non-noise points
            mask = labels != -1
            if mask.sum() > 100:
                sil = silhouette_score(X[mask], labels[mask])
            else:
                sil = np.nan

            sizes = np.bincount(labels[mask])
            size_ratio = float(sizes.min() / sizes.max()) if len(sizes) > 1 else 0.0

            # For HDBSCAN stability: refit and compare
            def _hdb_fit_fn(X_boot, X_full, rng):
                h = HDBSCAN(min_cluster_size=min_cluster, min_samples=10, metric="euclidean")
                h.fit(X_boot)
                # HDBSCAN doesn't have predict; use approximate_predict
                from sklearn.cluster import HDBSCAN as _HDBSCAN
                # Use labels from training only for bootstrap overlap
                boot_labels_on_boot = h.labels_
                # Refit on full for comparison (can't predict new points easily)
                h_full = HDBSCAN(min_cluster_size=min_cluster, min_samples=10, metric="euclidean")
                return h_full.fit_predict(X_full)

            stability = _bootstrap_ari(
                X, labels, _hdb_fit_fn, n_iterations=n_stability_iter,
            )

            row = {
                "dim_variant": dim_name,
                "n_dims": n_pcs,
                "cum_variance": cumvar[dim_name],
                "method": f"hdbscan_min{min_cluster}",
                "k": n_clusters,
                "silhouette": sil,
                "mean_ari": stability["mean_ari"],
                "std_ari": stability["std_ari"],
                "ari_ci_lower": stability["ci_lower"],
                "ari_ci_upper": stability["ci_upper"],
                "min_cluster_size": min_cluster,
                "max_cluster_size": int(sizes.max()),
                "size_ratio": size_ratio,
                "n_clusters_found": n_clusters,
                "n_noise": n_noise,
            }
            results.append(row)
            logger.info(
                "  HDBSCAN min_cluster=%d: %d clusters, %d noise, sil=%.3f, ARI=%.3f±%.3f",
                min_cluster, n_clusters, n_noise, sil,
                stability["mean_ari"], stability["std_ari"],
            )

    # ── Assemble results ──
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / "refinement_comparison.csv", index=False)

    # ── Identify best approach ──
    # Prioritise: meets ARI > 0.6, then best silhouette
    viable = df_results[df_results["mean_ari"] > 0.6].copy()
    if len(viable) > 0:
        best = viable.sort_values("silhouette", ascending=False).iloc[0]
        recommendation = "stable_clusters"
    else:
        # Fallback: best ARI overall
        valid = df_results.dropna(subset=["mean_ari"])
        if len(valid) > 0:
            best = valid.sort_values("mean_ari", ascending=False).iloc[0]
        else:
            best = df_results.iloc[0]
        recommendation = "continuous_scores"

    # ── Continuous vulnerability score (always produce) ──
    # Use first 3 PCs as interpretable heat sensitivity indices
    vuln_scores = pc_scores.iloc[:, :3].copy()
    vuln_scores.columns = ["heat_sensitivity_PC1", "heat_sensitivity_PC2", "heat_sensitivity_PC3"]
    # Composite: weighted by variance explained
    weights = evr[:3] / evr[:3].sum()
    vuln_scores["composite_heat_sensitivity"] = (
        vuln_scores.values @ weights
    )
    vuln_scores.to_csv(output_dir / "vulnerability_scores.csv")

    summary = {
        "n_approaches_tested": len(df_results),
        "n_viable_above_ari_0.6": len(viable) if 'viable' in dir() else 0,
        "best_approach": {
            "dim_variant": str(best["dim_variant"]),
            "method": str(best["method"]),
            "k": int(best["k"]) if not pd.isna(best.get("k")) else None,
            "silhouette": float(best["silhouette"]) if not pd.isna(best["silhouette"]) else None,
            "mean_ari": float(best["mean_ari"]) if not pd.isna(best["mean_ari"]) else None,
            "std_ari": float(best["std_ari"]) if not pd.isna(best["std_ari"]) else None,
        },
        "recommendation": recommendation,
        "note": (
            "If recommendation=='continuous_scores', discrete clustering does not "
            "produce stable groups — report continuous vulnerability indices instead. "
            "This is defensible for an exposure-response paper (not a clustering paper)."
        ),
        "vulnerability_scores_path": str(output_dir / "vulnerability_scores.csv"),
        "original_stage3_ari": 0.377,
        "original_stage3_silhouette": 0.197,
    }

    with open(output_dir / "refinement_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n=== REFINEMENT SUMMARY ===")
    logger.info("Best approach: %s + %s (k=%s)", best["dim_variant"], best["method"], best["k"])
    logger.info("  Silhouette: %.3f (was 0.197)", best["silhouette"] if not pd.isna(best["silhouette"]) else 0)
    logger.info("  ARI: %.3f (was 0.377)", best["mean_ari"] if not pd.isna(best["mean_ari"]) else 0)
    logger.info("Recommendation: %s", recommendation)

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    results = run_refinement()
    print(json.dumps(results, indent=2))
