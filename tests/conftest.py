"""Shared fixtures for MCD pipeline tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_analysis_df():
    """Minimal DataFrame that mimics ANALYSIS_READY structure.

    Contains all required columns with realistic dummy data.
    Suitable for testing validation and biomarker helpers.
    """
    n = 200
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        # Core identifiers
        "patient_id": [f"P{i:04d}" for i in range(n)],
        "visit_date": pd.date_range("2010-01-01", periods=n, freq="7D"),
        "study_source": rng.choice(
            ["JHB_ACTG_015", "JHB_WRHI_001", "JHB_DPHRU_053"], n
        ),
        "year": 2010 + np.arange(n) % 12,
        "month": 1 + np.arange(n) % 12,

        # Demographics
        "age_years": rng.normal(35, 10, n).clip(18, 80),
        "sex": rng.choice(["M", "F"], n),
        "hiv_status": rng.choice(["positive", "negative"], n),

        # Coordinates (Johannesburg)
        "latitude": rng.uniform(-26.5, -26.0, n),
        "longitude": rng.uniform(27.8, 28.2, n),

        # Biomarkers — all available ones with realistic ranges
        "creatinine": rng.normal(80, 20, n),
        "creatinine_clearance": rng.normal(90, 25, n),
        "albumin": rng.normal(40, 5, n),
        "fasting_glucose": rng.normal(90, 15, n),
        "hba1c": rng.normal(5.5, 0.8, n),
        "fasting_insulin": rng.lognormal(2, 0.5, n),
        "systolic_bp": rng.normal(125, 15, n),
        "diastolic_bp": rng.normal(80, 10, n),
        "heart_rate": rng.normal(72, 10, n),
        "hs_crp": rng.lognormal(0.5, 1.0, n),
        "cd4_count": rng.normal(500, 200, n).clip(10, 2000),
        "viral_load": rng.lognormal(5, 3, n),
        "alt": rng.lognormal(3, 0.5, n),
        "ast": rng.lognormal(3, 0.5, n),
        "total_cholesterol": rng.normal(190, 35, n),
        "hdl_cholesterol": rng.normal(50, 12, n),
        "ldl_cholesterol": rng.normal(120, 30, n),
        "triglycerides": rng.lognormal(4.5, 0.5, n),
        "bmi": rng.normal(27, 5, n).clip(15, 50),
        "waist_circumference": rng.normal(85, 12, n),
        "hip_circumference": rng.normal(100, 10, n),
        "waist_hip_ratio": rng.normal(0.85, 0.08, n),
        "body_fat_percent": rng.normal(30, 8, n),
        "total_fat_mass": rng.normal(20, 8, n),
        "fat_mass_index": rng.normal(7, 3, n),
        "hemoglobin": rng.normal(13, 2, n),
        "hematocrit": rng.normal(40, 4, n),
    })

    # Sprinkle some NaNs to be realistic
    for col in ["hba1c", "fasting_insulin", "hs_crp", "cd4_count", "viral_load"]:
        mask = rng.random(n) < 0.3
        df.loc[mask, col] = np.nan

    return df


@pytest.fixture
def sample_climate_df(sample_analysis_df):
    """Extend analysis_df with climate-linked columns."""
    df = sample_analysis_df.copy()
    rng = np.random.default_rng(99)
    n = len(df)

    # DLNM lag columns
    for lag in range(22):  # 0-21
        df[f"dlnm_lag{lag}_c"] = rng.normal(22, 5, n)

    # ERA5 rolling means
    for lag in ["1d", "3d", "7d", "14d", "30d"]:
        df[f"era5_temp_lag{lag}_c"] = rng.normal(22, 4, n)

    # Additional climate
    df["era5_temp_mean_c"] = rng.normal(22, 5, n)
    df["era5_temp_max_c"] = rng.normal(28, 5, n)
    df["era5_temp_min_c"] = rng.normal(16, 4, n)
    df["era5_temp_range_c"] = df["era5_temp_max_c"] - df["era5_temp_min_c"]

    return df
