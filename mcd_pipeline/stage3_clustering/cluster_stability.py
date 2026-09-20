"""
Cluster Stability Assessment — Bootstrap Validation
====================================================

Assesses stability of the vulnerability clusters via bootstrap
resampling. For each of 500 iterations, resamples persons with
replacement, re-runs k-means, and computes adjusted Rand index
against the full-sample clustering.

Stability threshold: >0.80 mean adjusted Rand index.

MCD reference: "bootstrap stability >0.8 over 500 iterations"
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from mcd_pipeline import config

logger = logging.getLogger(__name__)


def bootstrap_cluster_stability(
    pc_scores: pd.DataFrame,
    optimal_k: int,
    n_iterations: int = config.CLUSTER_STABILITY_ITERATIONS,
    threshold: float = config.CLUSTER_STABILITY_THRESHOLD,
    seed: int = config.MASTER_SEED,
) -> dict:
    """Assess cluster stability via bootstrap resampling.

    For each iteration:
    1. Resample patients with replacement (same N as original)
    2. Re-run k-means with optimal_k on the bootstrap sample
    3. Predict labels for the *full* sample using the bootstrap model
    4. Compute adjusted Rand index vs. full-sample labels

    Using predict on the full sample (rather than fitting on out-of-bag)
    avoids ARI being undefined for patients not in the bootstrap sample.

    Parameters
    ----------
    pc_scores : pd.DataFrame
        PCA scores from fit_shap_pca (n_patients × n_components).
    optimal_k : int
        Number of clusters.
    n_iterations : int
        Bootstrap iterations (default: 500).
    threshold : float
        Stability threshold (default: 0.80).
    seed : int
        Master random seed.

    Returns
    -------
    dict with keys:
        mean_ari       : float — mean adjusted Rand index
        std_ari        : float
        ci_lower       : float — 2.5th percentile
        ci_upper       : float — 97.5th percentile
        meets_threshold: bool
        per_iteration_ari: np.ndarray
    """
    rng = np.random.default_rng(seed)
    X = pc_scores.values
    n = len(X)

    # Full-sample reference labels
    ref_km = KMeans(n_clusters=optimal_k, random_state=seed, n_init=10)
    ref_labels = ref_km.fit_predict(X)

    ari_scores = np.empty(n_iterations)

    for i in range(n_iterations):
        # Bootstrap resample (with replacement)
        idx = rng.integers(0, n, size=n)
        X_boot = X[idx]

        # Fit k-means on bootstrap sample
        km_boot = KMeans(n_clusters=optimal_k, random_state=int(rng.integers(0, 2**31)), n_init=10)
        km_boot.fit(X_boot)

        # Predict labels for FULL sample using bootstrap centroids
        boot_labels = km_boot.predict(X)
        ari_scores[i] = adjusted_rand_score(ref_labels, boot_labels)

        if (i + 1) % 100 == 0:
            logger.info(
                "  Stability bootstrap %d/%d: running mean ARI=%.3f",
                i + 1, n_iterations, ari_scores[:i + 1].mean(),
            )

    mean_ari = float(ari_scores.mean())
    std_ari = float(ari_scores.std())
    ci_lower = float(np.percentile(ari_scores, 2.5))
    ci_upper = float(np.percentile(ari_scores, 97.5))
    meets = mean_ari >= threshold

    logger.info(
        "  Cluster stability: mean ARI=%.3f±%.3f [%.3f, %.3f] — %s threshold %.2f",
        mean_ari, std_ari, ci_lower, ci_upper,
        "MEETS" if meets else "FAILS", threshold,
    )

    return {
        "mean_ari": mean_ari,
        "std_ari": std_ari,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "meets_threshold": meets,
        "per_iteration_ari": ari_scores,
    }
