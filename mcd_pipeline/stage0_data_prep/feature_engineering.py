"""
Feature Engineering — Person-Mean Centering & Fourier Seasonality
=================================================================

Implements the MCD-specified feature transformations:

1. **Fourier seasonality terms**: sin(2*pi*m/12) and cos(2*pi*m/12) where m=month.
   Captures cyclical seasonal confounding without requiring month dummies.
   Reference: Bhaskaran et al. (2013) IJE 42:1187-1195

2. **Month fixed effects**: One-hot encoded month (1-12), month 1 as reference.

3. **Lag feature selection**: Extract the specific lag windows required by
   the MCD (0, 1, 3, 7, 14, 21, 30 days) from the full DLNM lag set.

4. **Person-mean centering**: For within-person correlation, subtract each
   person's mean biomarker value from their individual observations.
   Reference: Curran & Bauer (2011) Psych Methods 16:1-16

MCD reference: "Within-person correlation addressed via person-mean centering"
and "temporal confounders (month fixed effects, Fourier seasonality terms,
calendar year, Study_ID)"
"""

import logging

import numpy as np
import pandas as pd

from mcd_pipeline import config

logger = logging.getLogger(__name__)


def add_fourier_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Add Fourier sin/cos seasonality terms from month column.

    Creates columns: fourier_sin_month, fourier_cos_month

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'month' column (1-12).

    Returns
    -------
    pd.DataFrame
        With two new Fourier columns appended.
    """
    month = df["month"].astype(float)
    df["fourier_sin_month"] = np.sin(2 * np.pi * month / 12)
    df["fourier_cos_month"] = np.cos(2 * np.pi * month / 12)
    logger.info("Added Fourier seasonality terms")
    return df


def person_mean_center(
    df: pd.DataFrame,
    biomarker_columns: list[str],
    patient_id_col: str = "patient_id",
) -> pd.DataFrame:
    """Apply person-mean centering to biomarker columns.

    For each patient, subtracts their personal mean from each observation.
    Also adds the person-mean as a between-person feature.

    Creates columns:
        {biomarker}_centered — within-person deviation
        {biomarker}_person_mean — between-person level

    Parameters
    ----------
    df : pd.DataFrame
        Analysis dataset with patient_id and biomarker columns.
    biomarker_columns : list[str]
        Column names to center.
    patient_id_col : str
        Patient identifier column.

    Returns
    -------
    pd.DataFrame
        With centered and person-mean columns added.
    """
    centered_count = 0
    for col in biomarker_columns:
        if col not in df.columns:
            continue
        person_mean = df.groupby(patient_id_col)[col].transform("mean")
        df[f"{col}_centered"] = df[col] - person_mean
        df[f"{col}_person_mean"] = person_mean
        centered_count += 1

    logger.info("Person-mean centered %d biomarker columns", centered_count)
    return df


def select_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the MCD-specified lag columns from the full DLNM lag set.

    The climate-linked datasets contain dlnm_lag0_c through dlnm_lag21_c.
    This selects only the lags specified in config.LAG_DAYS: [0, 1, 3, 7, 14, 21, 30].

    For lag 30, uses era5_temp_lag30d_c (rolling 30-day mean) since
    individual day lag 30 is not in the DLNM set (which goes to lag 21).

    Also creates standardised lag feature names (temp_lag_0d ... temp_lag_30d)
    for consistent reference in models and SHAP output.

    Returns
    -------
    pd.DataFrame
        With standardised lag feature columns added.
    """
    present = []
    missing = []
    for col in config.LAG_FEATURE_COLUMNS:
        if col in df.columns:
            present.append(col)
        else:
            missing.append(col)

    if missing:
        logger.warning("Missing lag columns: %s", missing)

    # Create standardised lag feature names for the model
    for lag_day, col in zip(config.LAG_DAYS, config.LAG_FEATURE_COLUMNS):
        if col in df.columns:
            df[f"temp_lag_{lag_day}d"] = df[col]

    logger.info(
        "Selected %d/%d lag features: %s",
        len(present),
        len(config.LAG_FEATURE_COLUMNS),
        [f"lag{d}" for d in config.LAG_DAYS],
    )
    return df


def add_month_fixed_effects(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode month as fixed effects.

    Creates 11 month dummy columns (month_2 through month_12;
    month_1 as reference category).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'month' column.

    Returns
    -------
    pd.DataFrame
        With 11 month dummy columns added.
    """
    for m in range(2, 13):
        df[f"month_{m}"] = (df["month"] == m).astype(int)
    logger.info("Added 11 month fixed effects (month_2 through month_12)")
    return df


def engineer_features(
    df: pd.DataFrame,
    biomarker_columns: list[str],
) -> pd.DataFrame:
    """Full feature engineering pipeline.

    Applies all transformations in sequence:
    1. Fourier seasonality
    2. Month fixed effects
    3. Lag feature selection
    4. Person-mean centering

    Parameters
    ----------
    df : pd.DataFrame
        Raw analysis dataset from build_analysis_dataset.
    biomarker_columns : list[str]
        Biomarker columns to center.

    Returns
    -------
    pd.DataFrame
        Fully engineered dataset ready for Stage 1.
    """
    logger.info("=== Feature engineering ===")
    df = add_fourier_seasonality(df)
    df = add_month_fixed_effects(df)
    df = select_lag_features(df)
    df = person_mean_center(df, biomarker_columns)

    logger.info(
        "Feature engineering complete: %d rows × %d cols",
        len(df),
        len(df.columns),
    )
    return df
