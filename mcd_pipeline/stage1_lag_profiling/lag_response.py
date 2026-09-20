"""
Lag-Response Curve Extraction — From SHAP to Temporal Dynamics
==============================================================

Extracts lag-specific SHAP importance for each biomarker, producing
the equivalent of a DLNM lag-response curve but heterogeneous across
individuals rather than population-averaged.

For each biomarker model, the SHAP values for lag features
(temp_lag_0d, temp_lag_1d, ..., temp_lag_30d) describe how much
each lag window contributes to the prediction.

MCD reference: "Stage 1 — Lag-response profiling: SHAP feature
importance identifies which lag windows drive prediction for each
biomarker; analogous to a DLNM lag-response curve but heterogeneous
across individuals rather than population-averaged"
"""

import re

import numpy as np
import pandas as pd

import logging

from mcd_pipeline import config

logger = logging.getLogger(__name__)


def extract_lag_shap_profile(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract SHAP values for lag features only.

    Returns a DataFrame with one row per observation and one column per
    lag day, containing the SHAP attribution for that lag.

    Returns
    -------
    pd.DataFrame
        Columns: lag_0, lag_1, lag_3, lag_7, lag_14, lag_21, lag_30
        Rows: one per observation
    """
    # Find standardised lag feature indices
    lag_data = {}
    for lag_day in config.LAG_DAYS:
        col_name = f"temp_lag_{lag_day}d"
        if col_name in feature_names:
            idx = feature_names.index(col_name)
            lag_data[f"lag_{lag_day}"] = shap_values[:, idx]

    return pd.DataFrame(lag_data)


def summarise_lag_response(lag_shap_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise the lag-response curve: mean |SHAP| per lag with CIs.

    Returns
    -------
    pd.DataFrame
        Columns: lag_day, mean_abs_shap, ci_lower, ci_upper, pct_positive
    """
    if lag_shap_df.empty:
        logger.warning("summarise_lag_response: empty lag SHAP DataFrame — returning empty summary")
        return pd.DataFrame(columns=["lag_day", "mean_abs_shap", "ci_lower", "ci_upper", "pct_positive"])

    rows = []
    for col in lag_shap_df.columns:
        # Column format is "lag_{day}" — use regex to avoid fragile split indexing
        match = re.match(r"lag_(\d+)$", col)
        if not match:
            logger.warning("summarise_lag_response: unrecognised column name '%s', skipping", col)
            continue
        lag_day = int(match.group(1))
        vals = lag_shap_df[col].dropna()

        if len(vals) == 0:
            continue

        abs_vals = vals.abs()
        # Bootstrap CI on mean |SHAP|.
        # Seed from config so all CI computations are reproducible under the
        # same global seed without re-seeding on each iteration (which would
        # make the sequence fragile to iteration count changes).
        boot_means = []
        rng = np.random.RandomState(config.BOOTSTRAP_RANDOM_SEED)
        for _ in range(1000):
            sample = abs_vals.sample(n=len(abs_vals), replace=True, random_state=rng)
            boot_means.append(sample.mean())

        rows.append(
            {
                "lag_day": lag_day,
                "mean_abs_shap": abs_vals.mean(),
                "ci_lower": np.percentile(boot_means, 2.5),
                "ci_upper": np.percentile(boot_means, 97.5),
                "pct_positive": (vals > 0).mean(),
            }
        )

    return pd.DataFrame(rows).sort_values("lag_day").reset_index(drop=True)


def classify_temporal_window(lag_summary: pd.DataFrame) -> str:
    """Classify the dominant temporal window for a biomarker.

    Categories (from MCD):
    - 'acute' — dominant at lags 0-3 days
    - 'sub_acute' — dominant at lags 7-14 days
    - 'extended' — dominant at lags 21-30 days
    - 'mixed' — no single dominant window

    The dominant window is the one with the highest total mean |SHAP|.
    A window is "dominant" if it accounts for >40% of the total lag SHAP.
    """
    if lag_summary.empty:
        return "unknown"

    # Group into windows
    windows = {
        "acute": [0, 1, 3],
        "sub_acute": [7, 14],
        "extended": [21, 30],
    }

    window_importance = {}
    for name, lags in windows.items():
        mask = lag_summary["lag_day"].isin(lags)
        window_importance[name] = lag_summary.loc[mask, "mean_abs_shap"].sum()

    total = sum(window_importance.values())
    if total == 0:
        return "unknown"

    # Find dominant window
    dominant = max(window_importance, key=window_importance.get)
    proportion = window_importance[dominant] / total

    if proportion > 0.40:
        return dominant
    return "mixed"
