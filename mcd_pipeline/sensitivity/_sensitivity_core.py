"""
Sensitivity Core — Shared Infrastructure for Subset Sensitivity Analyses
=========================================================================

Most sensitivity analyses follow the same pattern:
1. Build/load the analysis dataset
2. Apply a data transformation (subset rows, modify features, stratify)
3. Re-run Stage 1 on the modified data (lightweight: fewer bootstrap reps)
4. Compare SHAP lag profiles against the primary (full-data) results
5. Report concordance metrics

This module provides the shared infrastructure so each SA module only
needs to define its data transformation.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import (
    BiomarkerSpec,
    MIN_SAMPLE_SIZE,
    get_available_biomarkers,
)

logger = logging.getLogger(__name__)

# Lightweight bootstrap for sensitivity analyses — use config constant so a
# single place controls all replication counts.
SA_BOOTSTRAP_REPLICATES = config.N_BOOTSTRAP_SENSITIVITY

# Adequate biomarkers (Stage 1 CV R² > 0) — concordance statistics should be
# reported for these 14 only; the 10 inadequate biomarkers (negative R²) are
# excluded to avoid spurious concordance from models that failed to learn
# temperature-biomarker relationships.  Derived at runtime from Stage 1 summary
# so it stays in sync if the primary analysis is re-run.
def _get_adequate_biomarkers(stage1_dir: Path = None) -> set[str]:
    """Return set of biomarker names with model_adequate=True from Stage 1."""
    stage1_dir = stage1_dir or (config.OUTPUT_ROOT / "stage1")
    summary_path = stage1_dir / "stage1_summary.json"
    if not summary_path.exists():
        logger.warning(
            "stage1_summary.json not found at %s — adequate biomarker filter disabled",
            summary_path,
        )
        return set()
    with open(summary_path) as f:
        summary = json.load(f)
    return {b for b, v in summary.items() if v.get("model_adequate") is True}


def load_primary_lag_profiles(
    stage1_dir: Path = None,
) -> dict[str, pd.DataFrame]:
    """Load primary Stage 1 SHAP lag-response profiles.

    Returns
    -------
    dict
        biomarker_name -> DataFrame with columns [lag_day, mean_abs_shap]
    """
    stage1_dir = stage1_dir or (config.OUTPUT_ROOT / "stage1")
    profiles = {}

    for bio_dir in stage1_dir.iterdir():
        if not bio_dir.is_dir():
            continue
        lag_path = bio_dir / "lag_response_summary.csv"
        if lag_path.exists():
            df = pd.read_csv(lag_path)
            profiles[bio_dir.name] = df

    logger.info("Loaded %d primary lag profiles", len(profiles))
    return profiles


def compare_lag_profiles(
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> dict:
    """Compare two lag-response profiles.

    Computes Spearman and Kendall correlation between lag-response
    curves (mean |SHAP| per lag day).

    Parameters
    ----------
    primary : pd.DataFrame
        Primary lag profile with columns [lag_day, mean_abs_shap].
    sensitivity : pd.DataFrame
        Sensitivity lag profile with same structure.

    Returns
    -------
    dict with spearman_rho, kendall_tau, max_abs_diff, concordant
    """
    merged = primary.merge(
        sensitivity, on="lag_day", suffixes=("_primary", "_sensitivity")
    )

    if len(merged) < 3:
        return {
            "spearman_rho": np.nan,
            "kendall_tau": np.nan,
            "max_abs_diff": np.nan,
            "concordant": False,
            "n_lags_compared": len(merged),
        }

    p = merged["mean_abs_shap_primary"].values
    s = merged["mean_abs_shap_sensitivity"].values

    rho, p_val_rho = spearmanr(p, s)
    tau, p_val_tau = kendalltau(p, s)

    # Normalise to compare relative importance (0-1 scale)
    p_norm = p / (p.sum() + 1e-12)
    s_norm = s / (s.sum() + 1e-12)
    max_diff = float(np.abs(p_norm - s_norm).max())

    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p_val_rho),
        "kendall_tau": float(tau),
        "kendall_p": float(p_val_tau),
        "max_abs_diff_normalised": max_diff,
        "concordant": bool(rho >= config.CONCORDANCE_RHO_THRESHOLD),
        "n_lags_compared": len(merged),
    }


def run_lightweight_stage1(
    df: pd.DataFrame,
    biomarker_names: list[str] = None,
    output_dir: Path = None,
    n_bootstrap: int = SA_BOOTSTRAP_REPLICATES,
    label: str = "sensitivity",
) -> dict:
    """Run a lightweight Stage 1 on a (possibly modified) dataset.

    Trains models for specified biomarkers with fewer bootstrap
    replicates than the primary analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-engineered analysis dataset.
    biomarker_names : list[str], optional
        Biomarkers to process. If None, uses all eligible.
    output_dir : Path
        Where to save results.
    n_bootstrap : int
        Number of bootstrap replicates (default 10).
    label : str
        Label for logging.

    Returns
    -------
    dict
        Per-biomarker results including lag profiles.
    """
    from mcd_pipeline.stage1_lag_profiling.xgboost_trainer import (
        prepare_features_and_target,
        train_single_model,
    )
    from mcd_pipeline.stage1_lag_profiling.shap_profiler import compute_shap_values
    from mcd_pipeline.stage1_lag_profiling.lag_response import (
        extract_lag_shap_profile,
        summarise_lag_response,
        classify_temporal_window,
    )

    biomarkers_dict = get_available_biomarkers()
    if biomarker_names is None:
        biomarker_names = list(biomarkers_dict.keys())

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for bio_name in biomarker_names:
        spec = biomarkers_dict.get(bio_name)
        if spec is None or spec.column is None:
            continue
        if spec.column not in df.columns:
            continue

        n_valid = df[spec.column].notna().sum()
        if n_valid < MIN_SAMPLE_SIZE:
            logger.info("  [%s] Skipping %s: %d samples < %d", label, bio_name, n_valid, MIN_SAMPLE_SIZE)
            continue

        try:
            X, y, groups = prepare_features_and_target(df, spec)
            model, cv_metrics = train_single_model(X, y, groups)
            shap_vals = compute_shap_values(model, X)
            feature_names = X.columns.tolist()

            lag_profile = extract_lag_shap_profile(shap_vals, feature_names)
            lag_summary = summarise_lag_response(lag_profile)
            temporal_window = classify_temporal_window(lag_summary)

            # Save
            bio_dir = output_dir / spec.column
            bio_dir.mkdir(parents=True, exist_ok=True)
            lag_summary.to_csv(bio_dir / "lag_response_summary.csv", index=False)

            import json as _json
            with open(bio_dir / "cv_metrics.json", "w") as f:
                _json.dump(cv_metrics, f, indent=2, default=str)

            results[bio_name] = {
                "cv_r2": cv_metrics["mean_r2"],
                "n_samples": len(X),
                "temporal_window": temporal_window,
                "lag_summary": lag_summary,
            }
            logger.info(
                "  [%s] %s: R²=%.4f, n=%d, window=%s",
                label, bio_name, cv_metrics["mean_r2"], len(X), temporal_window,
            )

        except Exception as e:
            logger.warning("  [%s] %s failed: %s", label, bio_name, e)
            results[bio_name] = {"error": str(e)}

    return results


def run_subset_sensitivity(
    df_full: pd.DataFrame,
    subset_mask: pd.Series,
    sa_name: str,
    sa_label: str,
    output_dir: Path,
    biomarker_names: list[str] = None,
    primary_profiles: dict = None,
) -> dict:
    """Run a complete subset sensitivity analysis.

    1. Apply mask to get subset
    2. Run lightweight Stage 1
    3. Compare lag profiles with primary
    4. Save results

    Parameters
    ----------
    df_full : pd.DataFrame
        Full analysis dataset (pre-engineered).
    subset_mask : pd.Series[bool]
        Boolean mask for which rows to include.
    sa_name : str
        Short name (e.g., "sa3_pollution").
    sa_label : str
        Human-readable label for logging.
    output_dir : Path
        Where to save results.
    biomarker_names : list[str], optional
        Specific biomarkers. None = all eligible.
    primary_profiles : dict, optional
        Primary lag profiles. Loaded from stage1/ if None.

    Returns
    -------
    dict — summary with concordance metrics per biomarker.
    """
    df_subset = df_full[subset_mask].copy()
    n_total = len(df_full)
    n_subset = len(df_subset)
    n_patients = df_subset["patient_id"].nunique() if "patient_id" in df_subset.columns else 0

    logger.info(
        "=== %s: %s ===\n  %d/%d rows (%.1f%%), %d patients",
        sa_name, sa_label, n_subset, n_total, 100 * n_subset / n_total, n_patients,
    )

    if n_subset < config.MIN_SUBSET_ROWS:
        logger.warning(
            "  Too few rows (%d < %d) for %s — skipping",
            n_subset, config.MIN_SUBSET_ROWS, sa_name,
        )
        return {"status": "skipped", "reason": f"Too few rows: {n_subset} < {config.MIN_SUBSET_ROWS}"}

    # Run lightweight Stage 1
    sa_output = output_dir / sa_name
    sa_results = run_lightweight_stage1(
        df_subset,
        biomarker_names=biomarker_names,
        output_dir=sa_output,
        label=sa_name,
    )

    # Load primary profiles for comparison
    if primary_profiles is None:
        primary_profiles = load_primary_lag_profiles()

    # Compare
    comparisons = {}
    for bio_name, result in sa_results.items():
        if "error" in result:
            comparisons[bio_name] = {"status": "failed", "error": result["error"]}
            continue

        lag_summary = result.get("lag_summary")
        primary_lag = primary_profiles.get(bio_name)

        if lag_summary is None or primary_lag is None:
            comparisons[bio_name] = {"status": "no_comparison"}
            continue

        comparison = compare_lag_profiles(primary_lag, lag_summary)
        comparison["cv_r2_primary"] = None  # Could load from primary
        comparison["cv_r2_sensitivity"] = result.get("cv_r2")
        comparison["temporal_window"] = result.get("temporal_window")
        comparisons[bio_name] = comparison

    # Summary statistics — all biomarkers
    failed = [b for b, c in comparisons.items() if c.get("status") == "failed"]
    if failed:
        logger.warning("  %s: %d biomarkers failed: %s", sa_name, len(failed), failed)

    concordant = [c for c in comparisons.values() if c.get("concordant", False)]
    rhos = [c["spearman_rho"] for c in comparisons.values()
            if isinstance(c.get("spearman_rho"), float) and not np.isnan(c["spearman_rho"])]

    # Adequate-biomarker-only concordance stats (primary reporting metric)
    adequate_set = _get_adequate_biomarkers()
    if adequate_set:
        adq_comps = {b: c for b, c in comparisons.items() if b in adequate_set}
        adq_concordant = sum(1 for c in adq_comps.values() if c.get("concordant", False))
        adq_rhos = [c["spearman_rho"] for c in adq_comps.values()
                    if isinstance(c.get("spearman_rho"), float) and not np.isnan(c["spearman_rho"])]
        adequate_stats = {
            "n_adequate": len(adq_comps),
            "n_concordant": adq_concordant,
            "pct_concordant": round(100 * adq_concordant / len(adq_comps), 1) if adq_comps else None,
            "mean_spearman_rho": round(float(np.mean(adq_rhos)), 3) if adq_rhos else None,
            "min_spearman_rho": round(float(np.min(adq_rhos)), 3) if adq_rhos else None,
            "concordant_biomarkers": sorted(b for b, c in adq_comps.items() if c.get("concordant")),
            "note": (
                f"Statistics for the {len(adq_comps)} model-adequate biomarkers "
                f"(Stage 1 CV R²>0) only. 10 inadequate biomarkers excluded."
            ),
        }
    else:
        adequate_stats = {"note": "stage1_summary.json not found; adequate filter not applied"}

    summary = {
        "sensitivity_analysis": sa_name,
        "description": sa_label,
        "concordance_threshold": config.CONCORDANCE_RHO_THRESHOLD,
        "n_rows_original": n_total,
        "n_rows_subset": n_subset,
        "n_patients_subset": n_patients,
        "pct_data_retained": round(100 * n_subset / n_total, 1),
        "n_biomarkers_tested": len(sa_results),
        "n_biomarkers_failed": len(failed),
        "n_concordant": len(concordant),
        "n_total_compared": len(rhos),
        "mean_spearman_rho": float(np.mean(rhos)) if rhos else None,
        "min_spearman_rho": float(np.min(rhos)) if rhos else None,
        "adequate_biomarkers_only": adequate_stats,
        "per_biomarker": comparisons,
    }

    # Save
    sa_output.mkdir(parents=True, exist_ok=True)
    with open(sa_output / f"{sa_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(
        "  %s complete: %d/%d concordant (ρ≥%.1f) all | %d/%d adequate | mean ρ=%.3f",
        sa_name,
        len(concordant), len(rhos),
        config.CONCORDANCE_RHO_THRESHOLD,
        adequate_stats.get("n_concordant", "?"), adequate_stats.get("n_adequate", "?"),
        np.mean(rhos) if rhos else 0,
    )
    return summary
