"""
Stage 0 Logic Tests — Data Prep Scientific Correctness
========================================================

Tests run against ACTUAL data via build_analysis_dataset() which
loads from TIDY_DATASETS/. Validates the pipeline's scientific
integrity before committing to a multi-hour Stage 1 run.

Covers:
- TIDY inputs: all 4 CSVs exist and have expected structure
- Built dataset: correct shape, studies, patients, lag columns
- SES merge: gcro_ columns present, non-degenerate values
- Biomarker coverage: sample sizes, completeness
- Feature engineering: Fourier identity, month FE, person-mean centering,
  lag feature mapping, NaN propagation
- GroupKFold integrity: no patient leakage across folds
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import (
    MIN_SAMPLE_SIZE,
    get_available_biomarkers,
    validate_biomarker_columns,
)
from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features
from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset

EXPECTED_STUDIES = sorted([
    "JHB_ACTG_015", "JHB_ACTG_016", "JHB_ACTG_019",
    "JHB_Aurum_009", "JHB_DPHRU_013", "JHB_DPHRU_053",
    "JHB_Ezin_002", "JHB_EZIN_025",
    "JHB_SCHARP_004", "JHB_SCHARP_006",
    "JHB_VIDA_007", "JHB_VIDA_008",
    "JHB_WRHI_001", "JHB_WRHI_003",
])


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def built_df():
    """Build the analysis dataset from TIDY inputs (module-scoped)."""
    return build_analysis_dataset()


@pytest.fixture(scope="module")
def engineered_df(built_df):
    """Apply feature engineering to the built dataset."""
    biomarker_cols = [
        b.column for b in get_available_biomarkers().values()
        if b.column and b.column in built_df.columns
    ]
    return engineer_features(built_df.copy(), biomarker_cols)


# ============================================================================
# 1. TIDY Input Tests
# ============================================================================

class TestTidyInputs:
    """Validate TIDY dataset files exist and have expected structure."""

    def test_tidy_biomarkers_exists(self):
        assert config.TIDY_BIOMARKERS_CSV.exists()

    def test_tidy_climate_exists(self):
        assert config.TIDY_CLIMATE_CSV.exists()

    def test_tidy_demographics_exists(self):
        assert config.TIDY_DEMOGRAPHICS_CSV.exists()

    def test_tidy_socioeconomic_exists(self):
        assert config.TIDY_SOCIOECONOMIC_CSV.exists()

    def test_tidy_biomarkers_has_required_columns(self):
        df = pd.read_csv(config.TIDY_BIOMARKERS_CSV, nrows=5)
        required = {"study_source", "patient_id", "visit_date", "biomarker", "value"}
        assert required.issubset(df.columns), f"Missing: {required - set(df.columns)}"

    def test_tidy_climate_has_required_columns(self):
        df = pd.read_csv(config.TIDY_CLIMATE_CSV, nrows=5)
        required = {"study_source", "patient_id", "visit_date", "climate_variable", "value"}
        assert required.issubset(df.columns), f"Missing: {required - set(df.columns)}"


# ============================================================================
# 2. Built Dataset Tests
# ============================================================================

class TestBuiltDataset:
    """Validate the dataset built from TIDY inputs."""

    def test_nonempty(self, built_df):
        assert len(built_df) > 10000, f"Only {len(built_df)} rows"

    def test_14_studies_present(self, built_df):
        actual = sorted(built_df["study_source"].unique())
        assert actual == EXPECTED_STUDIES

    def test_patient_id_no_nan(self, built_df):
        assert built_df["patient_id"].isna().sum() == 0

    def test_visit_date_in_range(self, built_df):
        dates = built_df["visit_date"].dropna()
        assert dates.min() >= pd.Timestamp("2003-01-01")
        assert dates.max() <= pd.Timestamp("2024-01-01")

    def test_lag_columns_present(self, built_df):
        missing = [c for c in config.LAG_FEATURE_COLUMNS if c not in built_df.columns]
        assert not missing, f"Missing lag columns: {missing}"

    def test_ses_columns_present(self, built_df):
        missing = [c for c in config.SOCIOECONOMIC_COVARIATES if c not in built_df.columns]
        assert not missing, f"Missing SES columns: {missing}"

    def test_demographic_columns_present(self, built_df):
        for col in config.DEMOGRAPHIC_COVARIATES + config.CLINICAL_COVARIATES:
            assert col in built_df.columns, f"Missing: {col}"

    def test_ses_values_not_degenerate(self, built_df):
        for col in config.SOCIOECONOMIC_COVARIATES:
            if col in built_df.columns:
                n_unique = built_df[col].dropna().nunique()
                assert n_unique > 1, f"{col} has only {n_unique} unique value"

    def test_no_duplicate_patient_visit(self, built_df):
        dupes = built_df.duplicated(
            subset=["patient_id", "visit_date", "study_source"]
        ).sum()
        assert dupes == 0, f"{dupes} duplicate (patient_id, visit_date) rows"

    def test_biomarker_columns_present(self, built_df):
        biomarkers = get_available_biomarkers()
        present = [
            b.column for b in biomarkers.values()
            if b.column and b.column in built_df.columns
        ]
        assert len(present) >= 15, f"Only {len(present)} biomarkers found"


# ============================================================================
# 3. Feature Engineering Tests
# ============================================================================

class TestFeatureEngineering:
    """Validate mathematical correctness of feature transformations."""

    def test_fourier_trig_identity(self, engineered_df):
        """sin^2 + cos^2 = 1 for all rows with valid month."""
        valid = engineered_df.dropna(subset=["fourier_sin_month", "fourier_cos_month"])
        identity = valid["fourier_sin_month"]**2 + valid["fourier_cos_month"]**2
        np.testing.assert_allclose(identity.values, 1.0, atol=1e-10)

    def test_fourier_terms_bounded(self, engineered_df):
        for col in ["fourier_sin_month", "fourier_cos_month"]:
            vals = engineered_df[col].dropna()
            assert vals.min() >= -1.0 - 1e-10
            assert vals.max() <= 1.0 + 1e-10

    def test_month_fixed_effects(self, engineered_df):
        month_cols = [f"month_{m}" for m in range(2, 13)]
        present = [c for c in month_cols if c in engineered_df.columns]
        assert len(present) == 11

        valid = engineered_df.dropna(subset=["month"]).copy()
        row_sums = valid[present].sum(axis=1)
        month1_mask = valid["month"] == 1
        assert (row_sums[month1_mask] == 0).all()
        assert (row_sums[~month1_mask] == 1).all()

    def test_person_mean_centering_zero_mean(self, engineered_df):
        test_col = None
        for col in ["hemoglobin", "bmi", "cd4_count"]:
            centered = f"{col}_centered"
            if centered in engineered_df.columns and engineered_df[centered].notna().sum() > 100:
                test_col = col
                break
        if test_col is None:
            pytest.skip("No centered biomarker with sufficient data")

        means = (
            engineered_df.dropna(subset=[f"{test_col}_centered"])
            .groupby("patient_id")[f"{test_col}_centered"]
            .mean()
        )
        np.testing.assert_allclose(means.values, 0.0, atol=1e-8)

    def test_person_mean_column_equals_actual_mean(self, engineered_df):
        test_col = None
        for col in ["hemoglobin", "bmi", "cd4_count"]:
            pm = f"{col}_person_mean"
            if pm in engineered_df.columns and col in engineered_df.columns:
                if engineered_df[col].notna().sum() > 100:
                    test_col = col
                    break
        if test_col is None:
            pytest.skip("No person_mean column with sufficient data")

        sub = engineered_df.dropna(subset=[test_col])
        actual = sub.groupby("patient_id")[test_col].transform("mean")
        np.testing.assert_allclose(sub[f"{test_col}_person_mean"].values, actual.values, atol=1e-8)

    def test_all_7_lag_features_created(self, engineered_df):
        expected = [f"temp_lag_{d}d" for d in config.LAG_DAYS]
        missing = [c for c in expected if c not in engineered_df.columns]
        assert not missing, f"Missing: {missing}"

    def test_lag_feature_mapping(self, engineered_df):
        if "temp_lag_0d" in engineered_df.columns and "dlnm_lag0_c" in engineered_df.columns:
            v = engineered_df[["temp_lag_0d", "dlnm_lag0_c"]].dropna()
            np.testing.assert_array_equal(v["temp_lag_0d"].values, v["dlnm_lag0_c"].values)

    def test_no_spurious_nan(self, built_df):
        """Feature engineering should not add NaN to existing columns."""
        before_nan = built_df.isna().sum()
        bio_cols = [
            b.column for b in get_available_biomarkers().values()
            if b.column and b.column in built_df.columns
        ]
        after = engineer_features(built_df.copy(), bio_cols)
        for col in built_df.columns:
            if col in after.columns:
                assert after[col].isna().sum() <= before_nan.get(col, 0), (
                    f"NaN introduced in {col}"
                )


# ============================================================================
# 4. GroupKFold Integrity Tests
# ============================================================================

class TestGroupKFoldIntegrity:
    """Validate GroupKFold produces non-leaking folds."""

    def test_no_patient_leakage(self, engineered_df):
        test_col = None
        for col in ["hemoglobin", "bmi", "cd4_count"]:
            if col in engineered_df.columns and engineered_df[col].notna().sum() > 200:
                test_col = col
                break
        if test_col is None:
            pytest.skip("No biomarker with sufficient data")

        sub = engineered_df.dropna(subset=[test_col]).reset_index(drop=True)
        groups = sub["patient_id"].astype(str)
        gkf = GroupKFold(n_splits=config.CV_FOLDS)
        for fold, (train_idx, val_idx) in enumerate(gkf.split(sub, groups=groups)):
            overlap = set(groups.iloc[train_idx]) & set(groups.iloc[val_idx])
            assert len(overlap) == 0, f"Fold {fold}: {len(overlap)} leaked patients"

    def test_all_folds_nonempty(self, engineered_df):
        test_col = None
        for col in ["hemoglobin", "bmi"]:
            if col in engineered_df.columns and engineered_df[col].notna().sum() > 200:
                test_col = col
                break
        if test_col is None:
            pytest.skip("No biomarker with sufficient data")

        sub = engineered_df.dropna(subset=[test_col]).reset_index(drop=True)
        groups = sub["patient_id"].astype(str)
        gkf = GroupKFold(n_splits=config.CV_FOLDS)
        for fold, (train_idx, val_idx) in enumerate(gkf.split(sub, groups=groups)):
            assert len(train_idx) > 0
            assert len(val_idx) > 0

    def test_folds_cover_all_data(self, engineered_df):
        test_col = None
        for col in ["hemoglobin", "bmi"]:
            if col in engineered_df.columns and engineered_df[col].notna().sum() > 200:
                test_col = col
                break
        if test_col is None:
            pytest.skip("No biomarker with sufficient data")

        sub = engineered_df.dropna(subset=[test_col]).reset_index(drop=True)
        groups = sub["patient_id"].astype(str)
        gkf = GroupKFold(n_splits=config.CV_FOLDS)
        all_val = set()
        for _, val_idx in gkf.split(sub, groups=groups):
            all_val.update(val_idx)
        assert len(all_val) == len(sub)
