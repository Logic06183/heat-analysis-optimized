"""
Run Stage 1 — Orchestrate Lag-Response Profiling Across All Biomarkers
======================================================================

Loops over all available biomarkers, trains XGBoost models with bootstrap
replication, computes interventional SHAP values, and extracts lag-response
profiles. Saves all results to mcd_outputs/stage1/.

Usage:
    python -m mcd_pipeline.stage1_lag_profiling.run_stage1

MCD reference: "One XGBoost model per biomarker (n=27)"
"""

import json
import logging
from pathlib import Path

import pandas as pd

from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import (
    MIN_SAMPLE_SIZE,
    get_available_biomarkers,
)
from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features

logger = logging.getLogger(__name__)


def run_stage1(
    df: pd.DataFrame = None,
    output_dir: Path = None,
) -> dict:
    """Run Stage 1 for all biomarkers meeting the minimum sample size.

    For each biomarker:
    1. Train XGBoost with GroupKFold + 50 bootstrap replicates
    2. Compute interventional SHAP values
    3. Extract lag-response profiles
    4. Save results

    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-built analysis dataset. If None, builds from scratch.
    output_dir : Path, optional
        Output directory. Defaults to mcd_outputs/stage1.

    Returns
    -------
    dict
        Summary keyed by biomarker name with cv_metrics, bootstrap_stability,
        dominant_temporal_window, and output_paths.
    """
    from mcd_pipeline.stage1_lag_profiling.xgboost_trainer import train_biomarker

    output_dir = output_dir or (config.OUTPUT_ROOT / "stage1")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset if not provided
    if df is None:
        logger.info("Building analysis dataset...")
        df = build_analysis_dataset()
        biomarkers_dict = get_available_biomarkers()
        bio_cols = [
            b.column
            for b in biomarkers_dict.values()
            if b.column and b.column in df.columns
        ]
        df = engineer_features(df, bio_cols)

    # Get biomarkers that meet minimum sample size
    biomarkers_dict = get_available_biomarkers()
    eligible = []
    for name, spec in biomarkers_dict.items():
        if not spec.column or spec.column not in df.columns:
            continue
        n_valid = df[spec.column].notna().sum()
        if n_valid >= MIN_SAMPLE_SIZE:
            eligible.append(spec)
        else:
            logger.info(
                "Skipping %s: %d samples (min %d)", name, n_valid, MIN_SAMPLE_SIZE
            )

    logger.info(
        "=== Stage 1: Processing %d/%d biomarkers ===",
        len(eligible),
        len(biomarkers_dict),
    )

    results = {}
    for i, biomarker in enumerate(eligible):
        logger.info(
            "[%d/%d] Processing %s...", i + 1, len(eligible), biomarker.column
        )
        try:
            result = train_biomarker(
                df,
                biomarker,
                output_dir=output_dir / biomarker.column,
            )
            results[biomarker.column] = result
        except Exception as e:
            logger.error("  FAILED: %s — %s", biomarker.column, e)
            results[biomarker.column] = {"error": str(e)}

    # Save overall summary
    summary_path = output_dir / "stage1_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Stage 1 summary saved to %s", summary_path)

    # Print overview
    n_success = sum(1 for r in results.values() if "error" not in r)
    logger.info(
        "=== Stage 1 complete: %d/%d biomarkers succeeded ===",
        n_success,
        len(eligible),
    )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_stage1()
