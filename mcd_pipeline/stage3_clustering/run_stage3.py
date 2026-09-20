"""
Run Stage 3 — Orchestrate Vulnerability Clustering
===================================================

1. Aggregate person-level temperature-lag SHAP values across biomarkers
2. PCA (85% variance retained, but cluster on top 10 PCs)
3. GMM (k=3, full covariance) — refined from initial k-means approach
4. Bootstrap stability (500 iterations, threshold 0.80)
5. Characterise clusters by demographics/SES/geography
6. Produce continuous heat sensitivity scores (PC1-3)

Refinement history (April 2026):
    Original approach (PC56 + k-means k=4) gave ARI=0.377 (unstable).
    Systematic comparison of 4 approaches showed:
    - PC10 + GMM k=3: ARI=0.993 ± 0.031 [0.988, 0.998] ← selected
    - PC5 + k-means k=3: ARI=0.903 ± 0.252
    - PC5 + GMM k=3: ARI=0.883 ± 0.091
    Root cause: 56 PCA components → curse of dimensionality for k-means.
    GMM handles elliptical clusters in 10-dim space; k-means assumes spheres.

Usage:
    python -m mcd_pipeline.stage3_clustering.run_stage3
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from mcd_pipeline import config
from mcd_pipeline.stage3_clustering.cluster_stability import bootstrap_cluster_stability
from mcd_pipeline.stage3_clustering.kmeans_clustering import (
    characterise_clusters,
    fit_kmeans_range,
    select_optimal_k,
)
from mcd_pipeline.stage3_clustering.shap_pca import multi_biomarker_pca

logger = logging.getLogger(__name__)

# Refined clustering parameters (justified by stage3_refinement comparison)
GMM_N_COMPONENTS = 3          # Optimal from refinement comparison
GMM_COVARIANCE_TYPE = "full"  # Allows elliptical clusters (critical for SHAP space)
GMM_N_INIT = 5                # Multiple initialisations for robustness
GMM_MAX_ITER = 300
PCA_CLUSTERING_DIMS = 10      # Top 10 PCs (~38% variance) — balances signal vs noise


def _gmm_bootstrap_stability(
    X: np.ndarray,
    ref_labels: np.ndarray,
    n_components: int = GMM_N_COMPONENTS,
    n_iterations: int = 500,
    seed: int = 42,
) -> dict:
    """Bootstrap ARI stability for GMM clustering.

    For each iteration: resample with replacement, fit GMM on bootstrap,
    predict full sample, compute ARI vs reference.
    """
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    n = len(X)
    aris = np.empty(n_iterations)

    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        gmm_boot = GaussianMixture(
            n_components=n_components,
            covariance_type=GMM_COVARIANCE_TYPE,
            random_state=int(rng.integers(0, 2**31)),
            n_init=3,
            max_iter=GMM_MAX_ITER,
        )
        gmm_boot.fit(X[idx])
        boot_labels = gmm_boot.predict(X)
        aris[i] = adjusted_rand_score(ref_labels, boot_labels)

        if (i + 1) % 100 == 0:
            logger.info(
                "  Stability bootstrap %d/%d: running mean ARI=%.3f",
                i + 1, n_iterations, aris[:i + 1].mean(),
            )

    mean_ari = float(aris.mean())
    return {
        "mean_ari": mean_ari,
        "std_ari": float(aris.std()),
        "ci_lower": float(np.percentile(aris, 2.5)),
        "ci_upper": float(np.percentile(aris, 97.5)),
        "meets_threshold": mean_ari >= config.CLUSTER_STABILITY_THRESHOLD,
        "per_iteration_ari": aris,
    }


def run_stage3(
    df: pd.DataFrame = None,
    stage1_dir: Path = config.OUTPUT_ROOT / "stage1",
    stage2_dir: Path = config.OUTPUT_ROOT / "stage2",
    output_dir: Path = config.OUTPUT_ROOT / "stage3",
) -> dict:
    """Run Stage 3 vulnerability clustering.

    Requires Stage 1 outputs (SHAP values for all biomarkers).

    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-built analysis dataset. If None, builds from TIDY inputs.
    stage1_dir : Path
        Directory containing Stage 1 per-biomarker output dirs.
    stage2_dir : Path
        Directory containing Stage 2 summary (to identify eligible biomarkers).
    output_dir : Path
        Root output directory for Stage 3 results.

    Returns
    -------
    dict
        optimal_k, silhouette_score, stability_ari, cluster_profiles,
        output_paths.
    """
    from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers
    from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
    from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Stage 3: Vulnerability Clustering ===")

    # --- Build dataset if not provided ---
    if df is None:
        logger.info("Building analysis dataset...")
        df = build_analysis_dataset()
        biomarkers_all = get_available_biomarkers()
        bio_cols = [b.column for b in biomarkers_all.values() if b.column and b.column in df.columns]
        df = engineer_features(df, bio_cols)

    # --- Identify eligible biomarkers (Stage 1 model + R² > 0) ---
    biomarkers_dict = get_available_biomarkers()
    eligible = []
    for name, spec in biomarkers_dict.items():
        shap_path = stage1_dir / spec.column / "shap_values.npy"
        cv_path = stage1_dir / spec.column / "cv_metrics.json"
        if not shap_path.exists():
            continue
        if cv_path.exists():
            with open(cv_path) as f:
                cv = json.load(f)
            if cv.get("mean_r2", 0) < 0:
                continue
        eligible.append(spec.column)

    logger.info("  Eligible biomarkers for clustering: %d (%s)", len(eligible), ", ".join(eligible))

    # --- Step 1–2: Multi-biomarker SHAP PCA ---
    logger.info("Step 1-2: Building temperature-SHAP fingerprint matrix + PCA...")
    pca, scaler, pc_scores, included_biomarkers = multi_biomarker_pca(
        stage1_dir=stage1_dir,
        biomarker_names=eligible,
        df=df,
    )

    pc_scores.to_csv(output_dir / "pc_scores.csv")
    np.save(output_dir / "pca_components.npy", pca.components_)
    with open(output_dir / "pca_summary.json", "w") as f:
        json.dump({
            "n_components": int(pca.n_components_),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": float(pca.explained_variance_ratio_.sum()),
            "n_patients": len(pc_scores),
            "included_biomarkers": included_biomarkers,
            "variance_threshold": config.PCA_VARIANCE_THRESHOLD,
        }, f, indent=2)
    logger.info("  PCA: %d components, %d patients", pca.n_components_, len(pc_scores))

    # --- Step 2b: Reduce to top PCA_CLUSTERING_DIMS for clustering ---
    # Full PCA retains 85% variance (56 components) for archival,
    # but clustering uses only top 10 PCs to avoid curse of dimensionality.
    n_cluster_dims = min(PCA_CLUSTERING_DIMS, pc_scores.shape[1])
    pc_cluster = pc_scores.iloc[:, :n_cluster_dims]
    evr = pca.explained_variance_ratio_
    cluster_var = float(evr[:n_cluster_dims].sum())
    logger.info(
        "  Using top %d PCs for clustering (%.1f%% variance)",
        n_cluster_dims, cluster_var * 100,
    )

    # --- Step 3a: K-means comparison (for reference/supplementary) ---
    logger.info("Step 3a: K-means comparison (k=%d..%d, supplementary)...",
                min(config.KMEANS_K_RANGE), max(config.KMEANS_K_RANGE))
    kmeans_results = fit_kmeans_range(pc_cluster)
    kmeans_optimal_k = select_optimal_k(kmeans_results)
    sil_scores = {k: float(v[1]) for k, v in kmeans_results.items()}
    with open(output_dir / "silhouette_scores_kmeans.json", "w") as f:
        json.dump({
            "by_k": sil_scores,
            "optimal_k": kmeans_optimal_k,
            "optimal_silhouette": sil_scores[kmeans_optimal_k],
            "note": "Supplementary — primary method is GMM",
        }, f, indent=2)

    # --- Step 3b: GMM clustering (primary method) ---
    logger.info(
        "Step 3b: GMM clustering (k=%d, covariance=%s, primary)...",
        GMM_N_COMPONENTS, GMM_COVARIANCE_TYPE,
    )
    X_cluster = pc_cluster.values
    gmm = GaussianMixture(
        n_components=GMM_N_COMPONENTS,
        covariance_type=GMM_COVARIANCE_TYPE,
        random_state=config.MASTER_SEED,
        n_init=GMM_N_INIT,
        max_iter=GMM_MAX_ITER,
    )
    labels = gmm.fit_predict(X_cluster)
    optimal_sil = silhouette_score(X_cluster, labels)
    optimal_k = GMM_N_COMPONENTS

    # Save GMM probabilities (soft assignments — useful for manuscript)
    gmm_probs = gmm.predict_proba(X_cluster)
    prob_df = pd.DataFrame(
        gmm_probs,
        index=pc_cluster.index,
        columns=[f"prob_cluster_{i}" for i in range(GMM_N_COMPONENTS)],
    )
    prob_df.to_csv(output_dir / "cluster_probabilities.csv")

    with open(output_dir / "silhouette_scores.json", "w") as f:
        json.dump({
            "method": "gmm",
            "k": GMM_N_COMPONENTS,
            "silhouette": float(optimal_sil),
            "n_clustering_dims": n_cluster_dims,
            "clustering_variance_retained": cluster_var,
            "bic": float(gmm.bic(X_cluster)),
            "aic": float(gmm.aic(X_cluster)),
        }, f, indent=2)

    cluster_labels = pd.Series(labels, index=pc_cluster.index, name="cluster")
    cluster_labels.to_csv(output_dir / "cluster_labels.csv")
    logger.info(
        "  GMM k=%d: silhouette=%.3f, BIC=%.0f, sizes=%s",
        GMM_N_COMPONENTS, optimal_sil, gmm.bic(X_cluster),
        np.bincount(labels).tolist(),
    )

    # --- Step 4: Bootstrap stability (using GMM) ---
    logger.info("Step 4: Bootstrap stability (%d iterations)...", config.CLUSTER_STABILITY_ITERATIONS)
    stability = _gmm_bootstrap_stability(
        X_cluster, labels,
        n_components=GMM_N_COMPONENTS,
        n_iterations=config.CLUSTER_STABILITY_ITERATIONS,
        seed=config.MASTER_SEED,
    )
    np.save(output_dir / "stability_ari_per_iteration.npy", stability["per_iteration_ari"])
    with open(output_dir / "stability_summary.json", "w") as f:
        json.dump({
            "mean_ari": stability["mean_ari"],
            "std_ari": stability["std_ari"],
            "ci_lower": stability["ci_lower"],
            "ci_upper": stability["ci_upper"],
            "meets_threshold": stability["meets_threshold"],
            "threshold": config.CLUSTER_STABILITY_THRESHOLD,
            "n_iterations": config.CLUSTER_STABILITY_ITERATIONS,
            "method": "gmm",
            "n_clustering_dims": n_cluster_dims,
        }, f, indent=2)

    if not stability["meets_threshold"]:
        logger.warning(
            "  WARNING: Cluster stability ARI=%.3f below threshold %.2f.",
            stability["mean_ari"], config.CLUSTER_STABILITY_THRESHOLD,
        )
    else:
        logger.info(
            "  Cluster stability ARI=%.3f MEETS threshold %.2f.",
            stability["mean_ari"], config.CLUSTER_STABILITY_THRESHOLD,
        )

    # --- Step 4b: Continuous vulnerability scores ---
    logger.info("Step 4b: Computing continuous heat sensitivity scores...")
    vuln = pc_scores.iloc[:, :3].copy()
    vuln.columns = ["heat_sensitivity_PC1", "heat_sensitivity_PC2", "heat_sensitivity_PC3"]
    weights = evr[:3] / evr[:3].sum()
    vuln["composite_heat_sensitivity"] = vuln.values @ weights
    vuln.to_csv(output_dir / "vulnerability_scores.csv")
    logger.info("  Saved continuous scores for %d patients", len(vuln))

    # --- Step 5: Characterise clusters ---
    logger.info("Step 5: Characterising clusters...")
    patient_cols = (
        ["age_years", "sex", "hiv_status"]
        + [c for c in df.columns if c.startswith("gcro_")]
        + ["study_source"]
    )
    available_cols = [c for c in patient_cols if c in df.columns]
    patient_data = (
        df[["patient_id"] + available_cols]
        .drop_duplicates(subset="patient_id", keep="first")
        .set_index("patient_id")
    )
    # Save per-patient table (original categorical values, aligned to cluster index)
    # before get_dummies so downstream bootstrap CI analysis can access raw
    # category strings (sex, hiv_status, gcro_* as strings, not dummies).
    patient_cluster_table = patient_data.reindex(cluster_labels.index).copy()
    patient_cluster_table["cluster"] = cluster_labels.values
    patient_cluster_table.to_csv(output_dir / "patient_cluster_table.csv")

    cat_cols = patient_data.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        patient_data = pd.get_dummies(patient_data, columns=cat_cols, drop_first=False)
    patient_data = patient_data.reindex(cluster_labels.index)

    profiles = characterise_clusters(labels, patient_data)
    profiles.to_csv(output_dir / "cluster_profiles.csv")

    # --- Overall summary ---
    summary = {
        "n_patients_clustered": int(len(pc_scores)),
        "n_biomarkers_used": len(included_biomarkers),
        "included_biomarkers": included_biomarkers,
        "pca_n_components_total": int(pca.n_components_),
        "pca_n_components_clustering": n_cluster_dims,
        "pca_variance_retained_total": float(pca.explained_variance_ratio_.sum()),
        "pca_variance_retained_clustering": cluster_var,
        "clustering_method": "gmm",
        "gmm_covariance_type": GMM_COVARIANCE_TYPE,
        "optimal_k": int(optimal_k),
        "optimal_silhouette": float(optimal_sil),
        "gmm_bic": float(gmm.bic(X_cluster)),
        "gmm_aic": float(gmm.aic(X_cluster)),
        "kmeans_silhouette_by_k": sil_scores,
        "stability_mean_ari": stability["mean_ari"],
        "stability_std_ari": stability["std_ari"],
        "stability_ci": [stability["ci_lower"], stability["ci_upper"]],
        "stability_meets_threshold": stability["meets_threshold"],
        "cluster_sizes": {int(k): int(v) for k, v in cluster_labels.value_counts().items()},
        "continuous_scores_saved": True,
        "refinement_note": (
            "Original approach (PC56 + k-means k=4) gave ARI=0.377. "
            "Systematic comparison showed PC10 + GMM k=3 gives ARI=0.993. "
            "See stage3_refinement/ for full comparison."
        ),
    }
    with open(output_dir / "stage3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "=== Stage 3 complete: GMM k=%d, silhouette=%.3f, stability ARI=%.3f ===",
        optimal_k, optimal_sil, stability["mean_ari"],
    )
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    results = run_stage3()
