"""
SA11 — Leave-one-cohort-out Stage 3 clustering stress test.

Question
--------
Is the three-cluster solution — and specifically the HIV-burden Cluster 2 —
driven by patients from any single contributing cohort?

Design
------
Reuse the existing Stage 1 SHAP fingerprints (already PCA-projected in
``mcd_outputs/stage3/pc_scores.csv``). For each of the 14 Johannesburg cohorts,
drop all patients from that cohort and refit GMM (k = 3, full covariance) on
the remaining patient-level SHAP fingerprints with the same hyperparameters
as the primary run. Report:

  * Adjusted Rand index (ARI) between the full clustering and the LOCO
    clustering, computed on patients present in *both* (i.e., everyone not in
    the dropped cohort).
  * Cluster size distribution for each LOCO clustering.
  * HIV prevalence in each LOCO cluster (the headline test: does the
    HIV-burden cluster persist in every LOCO refit?).

Complementary to SA9 (per-cohort meta-analysis of Stage 1 SHAP), which tests
Stage 1 stability. This SA tests Stage 3 stability under cohort removal.

Output: ``mcd_outputs/sensitivity/sa11_leave_one_cohort_out/sa11_summary.json``
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from mcd_pipeline import config
from mcd_pipeline.config import (
    GMM_COVARIANCE_TYPE,
    GMM_N_COMPONENTS,
    PCA_CLUSTERING_DIMS,
)
# These two live as module-level constants in run_stage3.py, not in config.
from mcd_pipeline.stage3_clustering.run_stage3 import GMM_MAX_ITER, GMM_N_INIT

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _fit_gmm(X: np.ndarray, seed: int = config.MASTER_SEED) -> np.ndarray:
    gmm = GaussianMixture(
        n_components=GMM_N_COMPONENTS,
        covariance_type=GMM_COVARIANCE_TYPE,
        random_state=seed,
        n_init=GMM_N_INIT,
        max_iter=GMM_MAX_ITER,
    )
    return gmm.fit_predict(X)


def main() -> None:
    stage3_dir = config.OUTPUT_ROOT / "stage3"
    out_dir = config.OUTPUT_ROOT / "sensitivity" / "sa11_leave_one_cohort_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== SA11: Leave-one-cohort-out Stage 3 clustering ===")

    # --- Load patient-level PC scores (one row per patient, from Stage 3) ---
    pc = pd.read_csv(stage3_dir / "pc_scores.csv").set_index("patient_id")
    n_dims = min(PCA_CLUSTERING_DIMS, pc.shape[1])
    pc_cluster = pc.iloc[:, :n_dims]
    logger.info("  Loaded %d patients × %d PCs for clustering", *pc_cluster.shape)

    # --- Load primary cluster labels (the canonical k=3 GMM result) ---
    labels = pd.read_csv(stage3_dir / "cluster_labels.csv").set_index("patient_id")["cluster"]
    logger.info("  Primary cluster sizes: %s", dict(labels.value_counts().sort_index()))

    # --- Link each patient to their cohort + HIV status from the analysis dataset ---
    from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers
    from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
    from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features

    logger.info("  Building analysis dataset (to attach cohort + HIV status to patient_id)...")
    df = build_analysis_dataset()
    biomarkers_all = get_available_biomarkers()
    bio_cols = [b.column for b in biomarkers_all.values() if b.column and b.column in df.columns]
    df = engineer_features(df, bio_cols)

    # One row per patient: take first observed study_source and HIV status.
    # HIV status in the dataset is encoded "Positive" / "Negative" / NaN (visit-level).
    # Use case-insensitive matching, and define patient-level HIV+ as "ever recorded positive".
    patient_meta = (
        df.assign(_hiv_pos=df["hiv_status"].astype(str).str.lower().eq("positive"))
        .groupby("patient_id")
        .agg(study_source=("study_source", "first"),
             hiv_status=("_hiv_pos", "max"))
        .reindex(pc.index)
    )
    patient_meta["hiv_status"] = patient_meta["hiv_status"].fillna(0).astype(int)
    cohorts = sorted(patient_meta["study_source"].dropna().unique().tolist())
    logger.info("  Cohorts: %d (%s)", len(cohorts), ", ".join(cohorts))

    primary_hiv_by_cluster = (
        patient_meta.assign(cluster=labels)
        .groupby("cluster")["hiv_status"].agg(["count", "sum"])
        .assign(hiv_prevalence=lambda d: d["sum"] / d["count"])
    )
    logger.info("  Primary HIV prevalence by cluster:\n%s", primary_hiv_by_cluster)

    # --- For each cohort: drop, refit, compute ARI vs primary on matched patients ---
    results: list[dict] = []
    for cohort in cohorts:
        keep = patient_meta["study_source"] != cohort
        n_kept = int(keep.sum())
        n_dropped = int((~keep).sum())
        if n_kept < 100:
            logger.warning("  Skipping %s: only %d patients would remain", cohort, n_kept)
            continue

        X = pc_cluster.loc[keep].values
        loco_labels = _fit_gmm(X)
        loco_series = pd.Series(loco_labels, index=pc_cluster.loc[keep].index)

        # ARI on matched patients (everyone not in dropped cohort, in both clusterings)
        matched = labels.loc[loco_series.index]
        ari = adjusted_rand_score(matched.values, loco_series.values)

        # Cluster sizes and HIV prevalence per LOCO cluster
        meta_kept = patient_meta.loc[loco_series.index].assign(cluster=loco_series.values)
        size_by_cluster = meta_kept["cluster"].value_counts().sort_index().to_dict()
        hiv_by_cluster = (
            meta_kept.groupby("cluster")["hiv_status"]
            .agg(["count", "sum"])
            .assign(hiv_prev=lambda d: d["sum"] / d["count"])
            ["hiv_prev"].to_dict()
        )

        # Does the HIV-burden cluster survive? Identify the LOCO cluster with the
        # highest HIV prevalence; report its size and HIV%.
        hiv_burden_cluster_id = max(hiv_by_cluster, key=hiv_by_cluster.get)
        hiv_burden_prev = hiv_by_cluster[hiv_burden_cluster_id]
        hiv_burden_size = size_by_cluster[hiv_burden_cluster_id]

        result = {
            "cohort_dropped": cohort,
            "n_patients_kept": n_kept,
            "n_patients_dropped": n_dropped,
            "ari_vs_primary": float(ari),
            "cluster_sizes": {int(k): int(v) for k, v in size_by_cluster.items()},
            "hiv_prevalence_by_cluster": {int(k): float(v) for k, v in hiv_by_cluster.items()},
            "hiv_burden_cluster_id": int(hiv_burden_cluster_id),
            "hiv_burden_cluster_size": int(hiv_burden_size),
            "hiv_burden_cluster_prevalence": float(hiv_burden_prev),
        }
        results.append(result)
        logger.info(
            "  Drop %-30s n=%5d  ARI=%.3f  HIV-burden cluster: id=%d, n=%d, HIV%%=%.1f",
            cohort, n_kept, ari, hiv_burden_cluster_id, hiv_burden_size, 100 * hiv_burden_prev,
        )

    # --- Summary statistics ---
    aris = [r["ari_vs_primary"] for r in results]
    burden_prevs = [r["hiv_burden_cluster_prevalence"] for r in results]
    summary = {
        "sensitivity_analysis": "sa11_leave_one_cohort_out",
        "design": (
            "For each of the 14 Johannesburg cohorts, all patients from that "
            "cohort were dropped and Stage 3 GMM (k=3, full covariance) was "
            "refit on the remaining patient-level SHAP fingerprints. ARI is "
            "computed between full and LOCO clustering on patients present in "
            "both (i.e., everyone not in the dropped cohort). The headline "
            "test is whether the HIV-burden cluster (highest HIV prevalence) "
            "persists in every LOCO refit."
        ),
        "primary_hiv_prevalence_by_cluster": {
            int(k): {"n": int(primary_hiv_by_cluster.loc[k, "count"]),
                     "hiv_positive": int(primary_hiv_by_cluster.loc[k, "sum"]),
                     "hiv_prevalence": float(primary_hiv_by_cluster.loc[k, "hiv_prevalence"])}
            for k in primary_hiv_by_cluster.index
        },
        "summary_stats": {
            "n_loco_runs": len(results),
            "ari_min": float(min(aris)),
            "ari_max": float(max(aris)),
            "ari_mean": float(np.mean(aris)),
            "ari_median": float(np.median(aris)),
            "hiv_burden_prevalence_min": float(min(burden_prevs)),
            "hiv_burden_prevalence_max": float(max(burden_prevs)),
            "hiv_burden_prevalence_mean": float(np.mean(burden_prevs)),
        },
        "per_cohort": results,
    }
    out_path = out_dir / "sa11_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("\n  SA11 complete; ARI range %.3f–%.3f, HIV-burden prevalence range %.1f–%.1f%%",
                summary["summary_stats"]["ari_min"], summary["summary_stats"]["ari_max"],
                100 * summary["summary_stats"]["hiv_burden_prevalence_min"],
                100 * summary["summary_stats"]["hiv_burden_prevalence_max"])
    logger.info("  Written: %s", out_path)


if __name__ == "__main__":
    main()
