"""
Stage 3 Refinement — Focused Comparison on Promising Approaches
================================================================

Based on initial screen, tests the most promising dimension/method
combinations with full 500-iteration stability assessment.

Key finding from initial screen:
- PC5 + k-means (k=3-4): silhouette ~0.34, ARI ~0.89
- PC10 + GMM (k=3): ARI=0.994 (!)
- Original PC56 + k-means (k=4): silhouette=0.197, ARI=0.377

Reducing PCA dimensions from 56 to 5-10 dramatically improves
cluster stability by removing noise dimensions.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from mcd_pipeline import config

logger = logging.getLogger(__name__)


def bootstrap_ari_full(
    X: np.ndarray,
    labels_ref: np.ndarray,
    fit_predict_fn,
    n_iterations: int = 500,
    seed: int = config.MASTER_SEED,
) -> dict:
    """Full bootstrap stability assessment."""
    rng = np.random.default_rng(seed)
    n = len(X)
    aris = np.empty(n_iterations)

    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        boot_labels = fit_predict_fn(X[idx], X, rng)
        aris[i] = adjusted_rand_score(labels_ref, boot_labels)
        if (i + 1) % 100 == 0:
            logger.info("    Bootstrap %d/%d: running ARI=%.3f", i + 1, n_iterations, aris[:i+1].mean())

    return {
        "mean_ari": float(aris.mean()),
        "std_ari": float(aris.std()),
        "ci_lower": float(np.percentile(aris, 2.5)),
        "ci_upper": float(np.percentile(aris, 97.5)),
        "per_iteration": aris,
    }


def run_focused_refinement(
    output_dir: Path = config.OUTPUT_ROOT / "stage3_refinement",
    stage3_dir: Path = config.OUTPUT_ROOT / "stage3",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing PC scores
    pc_scores = pd.read_csv(stage3_dir / "pc_scores.csv", index_col=0)
    with open(stage3_dir / "pca_summary.json") as f:
        pca_info = json.load(f)
    evr = np.array(pca_info["explained_variance_ratio"])

    logger.info("Loaded %d patients × %d PCs", *pc_scores.shape)

    results = []

    # ── Approach 1: PC5 + k-means k=3 ──
    logger.info("\n=== Approach 1: PC5 + k-means k=3 ===")
    X5 = pc_scores.iloc[:, :5].values
    km = KMeans(n_clusters=3, random_state=config.MASTER_SEED, n_init=10)
    labels = km.fit_predict(X5)
    sil = silhouette_score(X5, labels)
    sizes = np.bincount(labels)
    logger.info("  Silhouette: %.3f, sizes: %s", sil, sizes.tolist())

    def km3_fn(Xb, Xf, rng):
        m = KMeans(n_clusters=3, random_state=int(rng.integers(0, 2**31)), n_init=10)
        m.fit(Xb)
        return m.predict(Xf)

    stab = bootstrap_ari_full(X5, labels, km3_fn, n_iterations=500)
    results.append({
        "name": "PC5_kmeans_k3", "n_dims": 5, "cum_var": float(evr[:5].sum()),
        "method": "kmeans", "k": 3, "silhouette": sil,
        **{k: v for k, v in stab.items() if k != "per_iteration"},
        "sizes": sizes.tolist(),
    })
    np.save(output_dir / "ari_PC5_kmeans_k3.npy", stab["per_iteration"])
    logger.info("  ARI: %.3f ± %.3f [%.3f, %.3f]",
                stab["mean_ari"], stab["std_ari"], stab["ci_lower"], stab["ci_upper"])

    # ── Approach 2: PC5 + k-means k=4 ──
    logger.info("\n=== Approach 2: PC5 + k-means k=4 ===")
    km4 = KMeans(n_clusters=4, random_state=config.MASTER_SEED, n_init=10)
    labels4 = km4.fit_predict(X5)
    sil4 = silhouette_score(X5, labels4)
    sizes4 = np.bincount(labels4)
    logger.info("  Silhouette: %.3f, sizes: %s", sil4, sizes4.tolist())

    def km4_fn(Xb, Xf, rng):
        m = KMeans(n_clusters=4, random_state=int(rng.integers(0, 2**31)), n_init=10)
        m.fit(Xb)
        return m.predict(Xf)

    stab4 = bootstrap_ari_full(X5, labels4, km4_fn, n_iterations=500)
    results.append({
        "name": "PC5_kmeans_k4", "n_dims": 5, "cum_var": float(evr[:5].sum()),
        "method": "kmeans", "k": 4, "silhouette": sil4,
        **{k: v for k, v in stab4.items() if k != "per_iteration"},
        "sizes": sizes4.tolist(),
    })
    np.save(output_dir / "ari_PC5_kmeans_k4.npy", stab4["per_iteration"])
    logger.info("  ARI: %.3f ± %.3f [%.3f, %.3f]",
                stab4["mean_ari"], stab4["std_ari"], stab4["ci_lower"], stab4["ci_upper"])

    # ── Approach 3: PC10 + GMM k=3 ──
    logger.info("\n=== Approach 3: PC10 + GMM k=3 ===")
    X10 = pc_scores.iloc[:, :10].values
    gmm3 = GaussianMixture(n_components=3, covariance_type="full",
                           random_state=config.MASTER_SEED, n_init=5, max_iter=300)
    labels_g3 = gmm3.fit_predict(X10)
    sil_g3 = silhouette_score(X10, labels_g3)
    sizes_g3 = np.bincount(labels_g3)
    logger.info("  Silhouette: %.3f, sizes: %s", sil_g3, sizes_g3.tolist())

    def gmm3_fn(Xb, Xf, rng):
        m = GaussianMixture(n_components=3, covariance_type="full",
                            random_state=int(rng.integers(0, 2**31)), n_init=3, max_iter=300)
        m.fit(Xb)
        return m.predict(Xf)

    stab_g3 = bootstrap_ari_full(X10, labels_g3, gmm3_fn, n_iterations=500)
    results.append({
        "name": "PC10_gmm_k3", "n_dims": 10, "cum_var": float(evr[:10].sum()),
        "method": "gmm", "k": 3, "silhouette": sil_g3,
        **{k: v for k, v in stab_g3.items() if k != "per_iteration"},
        "sizes": sizes_g3.tolist(),
    })
    np.save(output_dir / "ari_PC10_gmm_k3.npy", stab_g3["per_iteration"])
    logger.info("  ARI: %.3f ± %.3f [%.3f, %.3f]",
                stab_g3["mean_ari"], stab_g3["std_ari"], stab_g3["ci_lower"], stab_g3["ci_upper"])

    # ── Approach 4: PC5 + GMM k=3 ──
    logger.info("\n=== Approach 4: PC5 + GMM k=3 ===")
    gmm3_5 = GaussianMixture(n_components=3, covariance_type="full",
                              random_state=config.MASTER_SEED, n_init=5, max_iter=300)
    labels_g35 = gmm3_5.fit_predict(X5)
    sil_g35 = silhouette_score(X5, labels_g35)
    sizes_g35 = np.bincount(labels_g35)
    logger.info("  Silhouette: %.3f, sizes: %s", sil_g35, sizes_g35.tolist())

    def gmm3_5_fn(Xb, Xf, rng):
        m = GaussianMixture(n_components=3, covariance_type="full",
                            random_state=int(rng.integers(0, 2**31)), n_init=3, max_iter=300)
        m.fit(Xb)
        return m.predict(Xf)

    stab_g35 = bootstrap_ari_full(X5, labels_g35, gmm3_5_fn, n_iterations=500)
    results.append({
        "name": "PC5_gmm_k3", "n_dims": 5, "cum_var": float(evr[:5].sum()),
        "method": "gmm", "k": 3, "silhouette": sil_g35,
        **{k: v for k, v in stab_g35.items() if k != "per_iteration"},
        "sizes": sizes_g35.tolist(),
    })
    np.save(output_dir / "ari_PC5_gmm_k3.npy", stab_g35["per_iteration"])
    logger.info("  ARI: %.3f ± %.3f [%.3f, %.3f]",
                stab_g35["mean_ari"], stab_g35["std_ari"], stab_g35["ci_lower"], stab_g35["ci_upper"])

    # ── Continuous vulnerability scores ──
    logger.info("\n=== Computing continuous vulnerability scores ===")
    vuln = pc_scores.iloc[:, :3].copy()
    vuln.columns = ["heat_sensitivity_PC1", "heat_sensitivity_PC2", "heat_sensitivity_PC3"]
    weights = evr[:3] / evr[:3].sum()
    vuln["composite_heat_sensitivity"] = vuln.values @ weights
    vuln.to_csv(output_dir / "vulnerability_scores.csv")
    logger.info("  Saved continuous vulnerability scores for %d patients", len(vuln))

    # ── Summary ──
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / "refinement_comparison.csv", index=False)

    # Find best
    best = df_results.sort_values("mean_ari", ascending=False).iloc[0]
    meets_threshold = best["mean_ari"] >= 0.80

    summary = {
        "approaches_tested": [r["name"] for r in results],
        "original_performance": {
            "method": "PC56_kmeans_k4",
            "silhouette": 0.197,
            "mean_ari": 0.377,
        },
        "best_approach": {
            "name": best["name"],
            "n_dims": int(best["n_dims"]),
            "method": best["method"],
            "k": int(best["k"]),
            "silhouette": float(best["silhouette"]),
            "mean_ari": float(best["mean_ari"]),
            "std_ari": float(best["std_ari"]),
            "ci_lower": float(best["ci_lower"]),
            "ci_upper": float(best["ci_upper"]),
            "sizes": best["sizes"],
            "meets_stability_threshold": bool(meets_threshold),
        },
        "all_results": results,
        "continuous_scores_saved": True,
        "recommendation": (
            f"Use {best['name']} — ARI {best['mean_ari']:.3f} vs original 0.377. "
            f"{'Meets' if meets_threshold else 'Does not meet'} 0.80 threshold. "
            "Also produced continuous heat sensitivity scores as supplementary."
        ),
    }

    with open(output_dir / "refinement_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\n=== REFINEMENT COMPLETE ===")
    logger.info("Best: %s — ARI=%.3f (was 0.377), Sil=%.3f (was 0.197)",
                best["name"], best["mean_ari"], best["silhouette"])
    logger.info("Meets 0.80 threshold: %s", meets_threshold)

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    results = run_focused_refinement()
    print(json.dumps(results, indent=2, default=str))
