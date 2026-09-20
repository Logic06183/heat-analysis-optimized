"""
SA8: Heat Wave Threshold Comparison — 95th vs 90th Percentile
==============================================================

Compares heat wave definitions using 90th vs 95th percentile thresholds
to assess sensitivity of results to heat wave classification.

The primary analysis uses the 95th percentile. This SA re-runs with
90th percentile to check whether a less extreme threshold changes
which biomarkers show heat sensitivity and which lag windows dominate.

MCD reference: "heat wave threshold 95th vs 90th percentile"
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mcd_pipeline import config
from mcd_pipeline.sensitivity._sensitivity_core import (
    compare_lag_profiles,
    load_primary_lag_profiles,
    run_lightweight_stage1,
)

logger = logging.getLogger(__name__)


def _compute_heatwave_flag(
    df: pd.DataFrame,
    temp_col: str = "temp_lag_0d",
    percentile: int = 90,
    min_consecutive_days: int = 3,
) -> pd.Series:
    """Compute a heatwave indicator at a given percentile threshold.

    A heatwave day is defined as any day within a period of ≥3
    consecutive days where the mean temperature exceeds the
    percentile threshold of the study-period distribution.

    For this sensitivity analysis, we use a simplified binary flag:
    1 if daily temp exceeds the percentile threshold, 0 otherwise.
    This captures whether the observation occurred on an extreme
    heat day without requiring consecutive-day tracking (which
    would need the full daily time series, not available per-visit).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain temp_col.
    temp_col : str
        Temperature column to threshold.
    percentile : int
        Percentile threshold (90 or 95).

    Returns
    -------
    pd.Series — binary heatwave flag.
    """
    if len(df) < 100:
        logger.warning(
            "SA8: only %d rows — %dth percentile threshold may be unreliable (estimated from <100 rows)",
            len(df), percentile,
        )
    threshold = df[temp_col].quantile(percentile / 100)
    flag = (df[temp_col] >= threshold).astype(int)
    logger.info(
        "  Heatwave flag (p%d): threshold=%.1f°C, %d/%d days flagged (%.1f%%)",
        percentile, threshold, flag.sum(), len(flag), 100 * flag.mean(),
    )
    return flag


def run_heatwave_threshold_comparison(
    df: pd.DataFrame,
    output_dir: Path = None,
    primary_profiles: dict = None,
) -> dict:
    """Compare Stage 1 results using 90th vs 95th percentile heat wave flags.

    For each percentile:
    1. Create heatwave binary flag
    2. Add as a feature to the dataset
    3. Run lightweight Stage 1
    4. Compare lag profiles vs primary

    Then compare 90th vs 95th directly.
    """
    output_dir = output_dir or (config.OUTPUT_ROOT / "sensitivity")
    sa_output = output_dir / "sa8_heatwave"
    sa_output.mkdir(parents=True, exist_ok=True)

    if "temp_lag_0d" not in df.columns:
        return {"status": "skipped", "reason": "temp_lag_0d not found"}

    # Load primary profiles
    if primary_profiles is None:
        primary_profiles = load_primary_lag_profiles()

    # _build_feature_columns() in xgboost_trainer uses config.CLINICAL_COVARIATES.
    # Temporarily add heatwave_flag so it enters the Stage 1 model as a covariate.
    # This tests whether controlling for heatwave context (at each threshold) alters
    # the temperature lag SHAP profiles — the key SA8 question.
    orig_clinical = list(config.CLINICAL_COVARIATES)

    results_by_pct = {}
    for pct in config.HEATWAVE_PERCENTILES:
        logger.info("=== SA8: Heatwave threshold p%d ===", pct)

        df_sa = df.copy()
        df_sa["heatwave_flag"] = _compute_heatwave_flag(df_sa, percentile=pct)

        # Wire heatwave_flag into the model feature set for this run only
        config.CLINICAL_COVARIATES = orig_clinical + ["heatwave_flag"]

        try:
            pct_output = sa_output / f"p{pct}"
            sa_results = run_lightweight_stage1(
                df_sa,
                output_dir=pct_output,
                label=f"sa8_p{pct}",
            )
        finally:
            config.CLINICAL_COVARIATES = orig_clinical  # always restore

        results_by_pct[pct] = sa_results

    # Compare each percentile vs primary
    comparisons = {}
    for pct, sa_results in results_by_pct.items():
        for bio_name, result in sa_results.items():
            if "error" in result or "lag_summary" not in result:
                continue
            primary_lag = primary_profiles.get(bio_name)
            if primary_lag is None:
                continue

            comp = compare_lag_profiles(primary_lag, result["lag_summary"])
            comp["percentile"] = pct
            comparisons.setdefault(bio_name, {})[f"p{pct}_vs_primary"] = comp

    # Compare 90th vs 95th directly
    for bio_name in comparisons:
        r90 = results_by_pct.get(90, {}).get(bio_name, {})
        r95 = results_by_pct.get(95, {}).get(bio_name, {})
        if "lag_summary" in r90 and "lag_summary" in r95:
            comparisons[bio_name]["p90_vs_p95"] = compare_lag_profiles(
                r90["lag_summary"], r95["lag_summary"]
            )

    summary = {
        "sensitivity_analysis": "sa8_heatwave_threshold",
        "description": f"Heatwave threshold comparison: {config.HEATWAVE_PERCENTILES}",
        "percentiles_tested": config.HEATWAVE_PERCENTILES,
        "n_biomarkers_tested": len(comparisons),
        "per_biomarker": comparisons,
    }

    with open(sa_output / "sa8_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("  SA8 complete: %d biomarkers tested", len(comparisons))
    return summary
