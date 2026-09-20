"""
SA5: Imputation Comparison — KNN vs MICE vs Complete-Case
=========================================================

Compares three approaches to handling missing socioeconomic data:
1. KNN imputation (current approach — baseline)
2. MICE (Multiple Imputation by Chained Equations via IterativeImputer)
3. Complete-case analysis (drop rows missing any gcro_ SES column)

Key question: Do Stage 1 SHAP lag-response profiles change when the
imputation strategy for ward-level GCRO SES variables changes? If
Spearman ρ ≥ 0.7 across methods, the primary results are robust to
imputation choice.

MCD reference: "KNN vs MICE vs complete-case"
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from mcd_pipeline import config
from mcd_pipeline.sensitivity._sensitivity_core import (
    compare_lag_profiles,
    load_primary_lag_profiles,
    run_lightweight_stage1,
)

logger = logging.getLogger(__name__)

# SES columns subject to imputation comparison
GCRO_NUMERIC_COLS = [
    "gcro_income_bracket",
]
GCRO_DUMMY_PREFIXES = [
    "gcro_dwelling_type",
    "gcro_employment_status",
    "gcro_education_level",
]


def _gcro_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric gcro_ columns present in df.

    KNN/MICE imputers require numeric inputs. String-valued SES columns
    (e.g. gcro_dwelling_type = 'Formal') are excluded — they will be
    one-hot encoded downstream in prepare_features_and_target. Only
    columns already numeric (int/float) are imputed here.
    """
    return [
        c for c in df.columns
        if c.startswith("gcro_")
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def _apply_knn(df: pd.DataFrame, gcro_cols: list[str], n_neighbors: int = 5) -> pd.DataFrame:
    """Apply KNN imputation to gcro_ columns (current primary approach)."""
    out = df.copy()
    if not gcro_cols or not out[gcro_cols].isnull().any().any():
        return out
    imputer = KNNImputer(n_neighbors=n_neighbors)
    out[gcro_cols] = imputer.fit_transform(out[gcro_cols])
    return out


def _apply_mice(df: pd.DataFrame, gcro_cols: list[str]) -> pd.DataFrame:
    """Apply MICE (IterativeImputer) to gcro_ columns."""
    out = df.copy()
    if not gcro_cols or not out[gcro_cols].isnull().any().any():
        return out
    imputer = IterativeImputer(random_state=config.MASTER_SEED, max_iter=10)
    out[gcro_cols] = imputer.fit_transform(out[gcro_cols])
    return out


def _apply_complete_case(df: pd.DataFrame, gcro_cols: list[str]) -> pd.DataFrame:
    """Drop rows missing any gcro_ column."""
    if not gcro_cols:
        return df
    return df.dropna(subset=gcro_cols).copy()


def run_imputation_comparison(
    df: pd.DataFrame = None,
    output_dir: Path = None,
    primary_profiles: dict = None,
) -> dict:
    """Compare KNN vs MICE vs complete-case imputation for SES covariates.

    Re-runs Stage 1 under each strategy and compares SHAP lag profiles
    against the primary (KNN) results.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-engineered analysis dataset. Built from TIDY inputs if None.
    output_dir : Path
        Where to save results (default: sensitivity/).
    primary_profiles : dict, optional
        Primary lag profiles. Loaded from stage1/ if None.

    Returns
    -------
    dict
        Per-strategy and per-biomarker concordance metrics.
    """
    from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers
    from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
    from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features

    output_dir = output_dir or (config.OUTPUT_ROOT / "sensitivity")
    sa_output = output_dir / "sa5_imputation_compare"
    sa_output.mkdir(parents=True, exist_ok=True)

    # --- Build dataset ---
    if df is None:
        logger.info("Building analysis dataset for SA5...")
        df = build_analysis_dataset()
        biomarkers_all = get_available_biomarkers()
        bio_cols = [b.column for b in biomarkers_all.values() if b.column and b.column in df.columns]
        df = engineer_features(df, bio_cols)

    gcro_cols = _gcro_cols(df)
    n_missing = int(df[gcro_cols].isnull().sum().sum()) if gcro_cols else 0
    n_rows = len(df)
    n_patients = df["patient_id"].nunique() if "patient_id" in df.columns else 0

    logger.info(
        "=== SA5: Imputation Comparison ===\n"
        "  %d rows, %d patients, %d gcro_ columns, %d missing values",
        n_rows, n_patients, len(gcro_cols), n_missing,
    )

    if not gcro_cols:
        return {"status": "skipped", "reason": "No gcro_ columns found in dataset"}

    if n_missing == 0:
        logger.info("  No missing SES values — all three strategies will be identical.")
        logger.info("  Proceeding to confirm robustness (identical results expected).")

    # --- Load primary profiles ---
    if primary_profiles is None:
        primary_profiles = load_primary_lag_profiles()

    # --- Strategy 1: KNN (primary approach — reference for this comparison) ---
    logger.info("  Strategy 1/3: KNN imputation (primary)...")
    df_knn = _apply_knn(df, gcro_cols)
    results_knn = run_lightweight_stage1(
        df_knn,
        output_dir=sa_output / "knn",
        label="sa5_knn",
    )

    # --- Strategy 2: MICE ---
    logger.info("  Strategy 2/3: MICE imputation...")
    df_mice = _apply_mice(df, gcro_cols)
    results_mice = run_lightweight_stage1(
        df_mice,
        output_dir=sa_output / "mice",
        label="sa5_mice",
    )

    # --- Strategy 3: Complete-case ---
    df_cc = _apply_complete_case(df, gcro_cols)
    n_cc = len(df_cc)
    n_dropped = n_rows - n_cc
    logger.info(
        "  Strategy 3/3: Complete-case (%d/%d rows retained, %d dropped)...",
        n_cc, n_rows, n_dropped,
    )
    if n_cc < 500:
        logger.warning("  Complete-case too small (%d rows) — skipping.", n_cc)
        results_cc = {}
        cc_skipped = True
    else:
        results_cc = run_lightweight_stage1(
            df_cc,
            output_dir=sa_output / "complete_case",
            label="sa5_cc",
        )
        cc_skipped = False

    # --- Compare strategies vs primary Stage 1 profiles ---
    def _build_comparisons(results: dict, label: str) -> dict:
        comparisons = {}
        for bio_name, result in results.items():
            if "error" in result:
                comparisons[bio_name] = {"status": "failed", "error": result["error"]}
                continue
            lag_summary = result.get("lag_summary")
            primary_lag = primary_profiles.get(bio_name)
            if lag_summary is None or primary_lag is None:
                comparisons[bio_name] = {"status": "no_comparison"}
                continue
            cmp = compare_lag_profiles(primary_lag, lag_summary)
            cmp["cv_r2"] = result.get("cv_r2")
            cmp["temporal_window"] = result.get("temporal_window")
            comparisons[bio_name] = cmp
        return comparisons

    cmp_knn = _build_comparisons(results_knn, "knn")
    cmp_mice = _build_comparisons(results_mice, "mice")
    cmp_cc = _build_comparisons(results_cc, "complete_case") if not cc_skipped else {}

    # --- Also compare KNN vs MICE directly (same N, different fill values) ---
    knn_mice_direct = {}
    for bio_name, r_knn in results_knn.items():
        r_mice = results_mice.get(bio_name, {})
        if "lag_summary" in r_knn and "lag_summary" in r_mice:
            knn_mice_direct[bio_name] = compare_lag_profiles(
                r_knn["lag_summary"], r_mice["lag_summary"]
            )

    def _concordance_summary(comps: dict) -> dict:
        rhos = [c["spearman_rho"] for c in comps.values()
                if isinstance(c.get("spearman_rho"), float) and not np.isnan(c["spearman_rho"])]
        n_concordant = sum(1 for c in comps.values() if c.get("concordant", False))
        return {
            "n_biomarkers": len(comps),
            "n_concordant": n_concordant,
            "mean_spearman_rho": float(np.mean(rhos)) if rhos else None,
            "min_spearman_rho": float(np.min(rhos)) if rhos else None,
        }

    summary = {
        "sensitivity_analysis": "sa5_imputation_compare",
        "description": "KNN vs MICE vs complete-case imputation for gcro_ SES variables",
        "n_rows": n_rows,
        "n_patients": n_patients,
        "n_gcro_cols": len(gcro_cols),
        "n_missing_values": n_missing,
        "complete_case_rows": n_cc if not cc_skipped else None,
        "complete_case_skipped": cc_skipped,
        "strategies": {
            "knn_vs_primary": _concordance_summary(cmp_knn),
            "mice_vs_primary": _concordance_summary(cmp_mice),
            "complete_case_vs_primary": _concordance_summary(cmp_cc) if not cc_skipped else {"status": "skipped"},
            "knn_vs_mice_direct": _concordance_summary(knn_mice_direct),
        },
        "per_biomarker": {
            "knn_vs_primary": cmp_knn,
            "mice_vs_primary": cmp_mice,
            "complete_case_vs_primary": cmp_cc,
            "knn_vs_mice_direct": knn_mice_direct,
        },
    }

    with open(sa_output / "sa5_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(
        "=== SA5 complete: KNN ρ=%.3f, MICE ρ=%.3f vs primary ===",
        summary["strategies"]["knn_vs_primary"].get("mean_spearman_rho") or 0,
        summary["strategies"]["mice_vs_primary"].get("mean_spearman_rho") or 0,
    )
    return summary
