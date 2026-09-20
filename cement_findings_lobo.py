"""LOBO + PC-dimension sensitivity for the ERA5-Land cluster finding.

Diagnostic 5: Leave-one-biomarker-out (LOBO) Stage 3 — refit clustering 13 times,
              each time dropping one of the 13 retained biomarkers from the
              SHAP-fingerprint matrix. Tests whether the HIV cluster depends
              on any single biomarker.

Diagnostic 6: PC-dimension sensitivity — refit GMM on top 5, 10, and 20 PCs.
              Tests whether the clustering subspace dimension matters.

Outputs land alongside the earlier diagnostics in mcd_outputs_era5_land/diagnostics/.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from mcd_pipeline import config
from mcd_pipeline.config import (
    GMM_COVARIANCE_TYPE,
    GMM_N_COMPONENTS,
    PCA_CLUSTERING_DIMS,
)
from mcd_pipeline.stage3_clustering.run_stage3 import GMM_MAX_ITER, GMM_N_INIT

STAGE1 = config.OUTPUT_ROOT / "stage1"
STAGE3 = config.OUTPUT_ROOT / "stage3"
OUT = config.OUTPUT_ROOT / "diagnostics"


def fit_gmm(X: np.ndarray, seed: int = config.MASTER_SEED) -> np.ndarray:
    return GaussianMixture(
        n_components=GMM_N_COMPONENTS,
        covariance_type=GMM_COVARIANCE_TYPE,
        random_state=seed,
        n_init=GMM_N_INIT,
        max_iter=GMM_MAX_ITER,
    ).fit_predict(X)


def _patient_meta() -> pd.DataFrame:
    from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
    from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers
    from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features
    df = build_analysis_dataset()
    biomarkers = get_available_biomarkers()
    bio_cols = [b.column for b in biomarkers.values() if b.column and b.column in df.columns]
    df = engineer_features(df, bio_cols)
    df = df.assign(_hiv_pos=df["hiv_status"].astype(str).str.lower().eq("positive"))
    meta = (
        df.groupby("patient_id")
        .agg(cohort=("study_source", "first"), hiv_pos=("_hiv_pos", "max"))
    )
    meta["hiv_pos"] = meta["hiv_pos"].fillna(0).astype(int)
    return df, meta


def hiv_burden_cluster(labels: pd.Series, meta: pd.DataFrame) -> tuple[int, int, float]:
    """Return (cluster_id_with_highest_hiv, size, hiv_prevalence)."""
    aligned = meta.reindex(labels.index)
    by_cluster = pd.DataFrame({"cluster": labels.values, "hiv": aligned["hiv_pos"].values})
    summary = by_cluster.groupby("cluster")["hiv"].agg(["count", "sum", "mean"])
    top = summary["mean"].idxmax()
    return int(top), int(summary.loc[top, "count"]), float(summary.loc[top, "mean"])


# ----------------------------------------------------------------------------
# Diagnostic 5: LOBO
# ----------------------------------------------------------------------------
def diagnostic_5_lobo(df: pd.DataFrame, meta: pd.DataFrame, retained: list[str]) -> dict:
    logger.info("=== 5. Leave-one-biomarker-out Stage 3 (n=%d biomarkers) ===", len(retained))
    from mcd_pipeline.stage3_clustering.shap_pca import multi_biomarker_pca

    # Primary clustering for ARI baseline
    primary_labels = pd.read_csv(STAGE3 / "cluster_labels.csv").set_index("patient_id")["cluster"]

    results: list[dict] = []
    for dropped in retained:
        kept = [b for b in retained if b != dropped]
        logger.info("  Dropping %s — refitting with %d biomarkers...", dropped, len(kept))
        pca, scaler, pc_scores, _ = multi_biomarker_pca(
            stage1_dir=STAGE1, biomarker_names=kept, df=df,
        )
        n_dims = min(PCA_CLUSTERING_DIMS, pc_scores.shape[1])
        X = pc_scores.iloc[:, :n_dims].values
        labels = fit_gmm(X)
        loco_series = pd.Series(labels, index=pc_scores.index)
        # ARI on patients in both
        common = primary_labels.index.intersection(loco_series.index)
        ari = adjusted_rand_score(primary_labels.loc[common].values, loco_series.loc[common].values)
        cid, csize, chiv = hiv_burden_cluster(loco_series, meta)
        result = {
            "biomarker_dropped": dropped,
            "n_biomarkers_kept": len(kept),
            "n_patients_clustered": int(len(loco_series)),
            "ari_vs_primary": float(ari),
            "hiv_burden_cluster_size": csize,
            "hiv_burden_cluster_prevalence": float(chiv),
        }
        results.append(result)
        logger.info(
            "    ARI=%.3f, HIV-burden cluster: n=%d, HIV=%.1f%%",
            ari, csize, 100 * chiv,
        )

    out = {
        "diagnostic": "leave_one_biomarker_out_stage3",
        "n_lobo_runs": len(results),
        "summary_stats": {
            "ari_min": float(min(r["ari_vs_primary"] for r in results)),
            "ari_max": float(max(r["ari_vs_primary"] for r in results)),
            "ari_mean": float(np.mean([r["ari_vs_primary"] for r in results])),
            "ari_median": float(np.median([r["ari_vs_primary"] for r in results])),
            "hiv_prev_min": float(min(r["hiv_burden_cluster_prevalence"] for r in results)),
            "hiv_prev_max": float(max(r["hiv_burden_cluster_prevalence"] for r in results)),
            "hiv_prev_mean": float(np.mean([r["hiv_burden_cluster_prevalence"] for r in results])),
        },
        "per_biomarker": results,
    }
    return out


# ----------------------------------------------------------------------------
# Diagnostic 6: PC-dimension sensitivity
# ----------------------------------------------------------------------------
def diagnostic_6_pc_dims(meta: pd.DataFrame, dims_to_try: list[int]) -> dict:
    logger.info("=== 6. PC-dimension sensitivity (n_dims in %s) ===", dims_to_try)
    pc_scores = pd.read_csv(STAGE3 / "pc_scores.csv").set_index("patient_id")
    primary_labels = pd.read_csv(STAGE3 / "cluster_labels.csv").set_index("patient_id")["cluster"]

    results = []
    for n_dims in dims_to_try:
        if n_dims > pc_scores.shape[1]:
            logger.warning("  Skipping n_dims=%d (only %d PCs available)", n_dims, pc_scores.shape[1])
            continue
        X = pc_scores.iloc[:, :n_dims].values
        labels = fit_gmm(X)
        loco_series = pd.Series(labels, index=pc_scores.index)
        ari = adjusted_rand_score(primary_labels.values, loco_series.values)
        cid, csize, chiv = hiv_burden_cluster(loco_series, meta)
        results.append({
            "n_pcs": n_dims,
            "ari_vs_primary": float(ari),
            "hiv_burden_cluster_size": csize,
            "hiv_burden_cluster_prevalence": float(chiv),
        })
        logger.info(
            "  n_pcs=%d: ARI=%.3f, HIV-burden cluster: n=%d, HIV=%.1f%%",
            n_dims, ari, csize, 100 * chiv,
        )

    return {
        "diagnostic": "pc_dimension_sensitivity",
        "canonical_n_pcs": PCA_CLUSTERING_DIMS,
        "per_n_pcs": results,
    }


def main() -> None:
    df, meta = _patient_meta()

    retained = [
        "creatinine", "albumin", "systolic_bp", "diastolic_bp", "heart_rate",
        "cd4_count", "viral_load", "ldl_cholesterol", "hip_circumference",
        "waist_hip_ratio", "body_fat_percent", "hemoglobin", "hematocrit",
    ]
    # Restrict to those actually present on disk
    retained = [b for b in retained if (STAGE1 / b / "shap_values.npy").exists()]
    logger.info("Retained biomarkers found on disk: %d", len(retained))

    r5 = diagnostic_5_lobo(df, meta, retained)
    (OUT / "leave_one_biomarker_out.json").write_text(json.dumps(r5, indent=2))

    r6 = diagnostic_6_pc_dims(meta, dims_to_try=[5, 10, 15, 20])
    (OUT / "pc_dimension_sensitivity.json").write_text(json.dumps(r6, indent=2))

    logger.info("\nLOBO + PC-dimension diagnostics saved to %s", OUT)


if __name__ == "__main__":
    main()
