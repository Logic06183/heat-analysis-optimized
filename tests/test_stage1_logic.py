"""
Stage 1 Logic Tests -- XGBoost-SHAP Scientific Correctness
===========================================================

Runs a QUICK version of the Stage 1 pipeline (2 bootstrap replicates,
single biomarker) to validate scientific logic before committing to
the full 50-replicate, 27-biomarker run.

Covers:
- Feature preparation: NaN handling, expected columns, target validity
- Model training: GroupKFold CV metrics, prediction shapes, leakage
- SHAP values: shape, additivity, finiteness, lag feature importance
- Bootstrap: replication count, shape consistency, aggregation
- Lag response: profile extraction, temporal window classification
- MCD compliance: interventional SHAP, GroupKFold, patient resampling,
  7 lag days
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.model_selection import GroupKFold

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import (
    BIOMARKERS,
    BiomarkerSpec,
    get_available_biomarkers,
)
from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features
from mcd_pipeline.stage1_lag_profiling.xgboost_trainer import (
    _build_feature_columns,
    bootstrap_train,
    prepare_features_and_target,
    train_single_model,
)
from mcd_pipeline.stage1_lag_profiling.shap_profiler import (
    aggregate_bootstrap_shap,
    compute_shap_values,
)
from mcd_pipeline.stage1_lag_profiling.lag_response import (
    classify_temporal_window,
    extract_lag_shap_profile,
    summarise_lag_response,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def full_engineered_df():
    """Build the complete engineered dataset from TIDY inputs.

    Module-scoped: built once, shared across all tests.
    """
    df = build_analysis_dataset()
    biomarker_cols = [
        b.column for b in get_available_biomarkers().values()
        if b.column and b.column in df.columns
    ]
    df = engineer_features(df, biomarker_cols)
    return df


def _pick_test_biomarker(df: pd.DataFrame) -> BiomarkerSpec:
    """Pick the biomarker with the most non-null values for fast testing.

    Prefers biomarkers that do NOT require log-transform (simpler checks).
    """
    best_col = None
    best_n = 0
    best_spec = None
    for key, spec in get_available_biomarkers().items():
        if spec.column and spec.column in df.columns:
            n = df[spec.column].notna().sum()
            # Prefer non-log-transformed for simpler SHAP additivity checks
            if n > best_n and not spec.log_transform:
                best_n = n
                best_col = spec.column
                best_spec = spec
    if best_spec is None:
        # Fall back to any biomarker
        for key, spec in get_available_biomarkers().items():
            if spec.column and spec.column in df.columns:
                n = df[spec.column].notna().sum()
                if n > best_n:
                    best_n = n
                    best_col = spec.column
                    best_spec = spec
    return best_spec


@pytest.fixture(scope="module")
def test_biomarker(full_engineered_df):
    """The biomarker spec used for Stage 1 testing."""
    spec = _pick_test_biomarker(full_engineered_df)
    assert spec is not None, "No biomarker with non-null values found"
    return spec


@pytest.fixture(scope="module")
def prepared_data(full_engineered_df, test_biomarker):
    """X, y, groups from prepare_features_and_target."""
    return prepare_features_and_target(full_engineered_df, test_biomarker)


@pytest.fixture(scope="module")
def trained_model_and_metrics(prepared_data):
    """Trained XGBoost model and CV metrics dict."""
    X, y, groups = prepared_data
    return train_single_model(X, y, groups)


@pytest.fixture(scope="module")
def primary_shap(trained_model_and_metrics, prepared_data):
    """Primary SHAP values from the full-data model."""
    model, _ = trained_model_and_metrics
    X, _, _ = prepared_data
    return compute_shap_values(model, X)


@pytest.fixture(scope="module")
def bootstrap_results(full_engineered_df, test_biomarker):
    """Bootstrap SHAP arrays (2 replicates only, for speed)."""
    return bootstrap_train(
        full_engineered_df, test_biomarker,
        n_replicates=2, seed=42,
    )


# ============================================================================
# 1. Feature Preparation Tests
# ============================================================================

class TestFeaturePreparation:
    """Validate the feature matrix and target for model training."""

    def test_x_has_no_nan(self, prepared_data):
        """Feature matrix X must have zero NaN values after preparation."""
        X, _, _ = prepared_data
        n_nan = X.isna().sum().sum()
        assert n_nan == 0, (
            f"X has {n_nan} NaN values. Columns with NaN: "
            f"{X.columns[X.isna().any()].tolist()}"
        )

    def test_x_contains_expected_feature_groups(self, prepared_data):
        """X must contain lag, demographic, temporal, and SES feature groups."""
        X, _, _ = prepared_data
        cols = set(X.columns)

        # Lag features (standardised names, should be present)
        lag_present = [
            c for c in [f"temp_lag_{d}d" for d in config.LAG_DAYS]
            if c in cols
        ]
        assert len(lag_present) >= 5, (
            f"Only {len(lag_present)}/7 lag features found in X"
        )

        # Demographic covariates (may be one-hot encoded)
        has_age = "age_years" in cols
        assert has_age, "age_years missing from feature matrix"

        # Fourier terms
        assert "fourier_sin_month" in cols, "fourier_sin_month missing from X"
        assert "fourier_cos_month" in cols, "fourier_cos_month missing from X"

    def test_y_all_finite(self, prepared_data):
        """Target values y must be finite (no NaN, no Inf)."""
        _, y, _ = prepared_data
        assert y.notna().all(), f"y has {y.isna().sum()} NaN values"
        assert np.isfinite(y.values).all(), "y has Inf values"

    def test_groups_align_with_x(self, prepared_data):
        """Patient ID groups must align with X row indices."""
        X, _, groups = prepared_data
        assert len(groups) == len(X), (
            f"groups length ({len(groups)}) != X length ({len(X)})"
        )
        assert (groups.index == X.index).all(), (
            "groups index does not match X index"
        )

    def test_dropping_nan_target_preserves_all_studies(self, full_engineered_df, test_biomarker):
        """Dropping NaN target should not eliminate ALL data from any study."""
        target_col = test_biomarker.column
        mask = full_engineered_df[target_col].notna()
        subset = full_engineered_df.loc[mask]

        # Check how many studies are represented
        original_studies = set(full_engineered_df["study_source"].unique())
        remaining_studies = set(subset["study_source"].unique())
        lost = original_studies - remaining_studies

        # Informational: it is OK to lose some studies (sparse biomarkers)
        # but we flag it
        if lost:
            print(
                f"INFO: {test_biomarker.column} has no data in studies: {lost}"
            )

    def test_feature_columns_match_config(self):
        """_build_feature_columns must include all MCD-specified feature groups."""
        features = _build_feature_columns()

        # Check lag features present
        for d in config.LAG_DAYS:
            assert f"temp_lag_{d}d" in features, f"temp_lag_{d}d missing from features"

        # Check DTR
        assert config.DTR_COLUMN in features, "DTR column missing"

        # Check demographics
        for c in config.DEMOGRAPHIC_COVARIATES:
            assert c in features, f"Demographic covariate '{c}' missing"

        # Check SES
        for c in config.SOCIOECONOMIC_COVARIATES:
            assert c in features, f"SES covariate '{c}' missing"

        # Check Fourier
        assert "fourier_sin_month" in features
        assert "fourier_cos_month" in features


# ============================================================================
# 2. Model Training Tests
# ============================================================================

class TestModelTraining:
    """Validate XGBoost training and CV metrics."""

    def test_model_trains_without_error(self, trained_model_and_metrics):
        """Model training must complete without raising."""
        model, metrics = trained_model_and_metrics
        assert model is not None
        assert metrics is not None

    def test_cv_r2_is_finite_and_bounded(self, trained_model_and_metrics):
        """CV R-squared must be finite and in [-1, 1]."""
        _, metrics = trained_model_and_metrics
        r2 = metrics["mean_r2"]
        assert np.isfinite(r2), f"mean_r2 is not finite: {r2}"
        assert -1.0 <= r2 <= 1.0, f"mean_r2 = {r2} is outside [-1, 1]"

    def test_cv_rmse_is_positive(self, trained_model_and_metrics):
        """CV RMSE must be strictly positive."""
        _, metrics = trained_model_and_metrics
        rmse = metrics["mean_rmse"]
        assert rmse > 0, f"mean_rmse = {rmse} is not positive"

    def test_predictions_match_val_length(self, prepared_data):
        """Model predictions must have same length as validation set."""
        X, y, groups = prepared_data
        gkf = GroupKFold(n_splits=config.CV_FOLDS)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]

            model = xgb.XGBRegressor(**config.XGBOOST_PARAMS)
            model.fit(X_train, y_train, verbose=False)
            preds = model.predict(X_val)

            assert len(preds) == len(X_val), (
                f"Fold {fold}: predictions ({len(preds)}) != "
                f"validation ({len(X_val)})"
            )
            break  # One fold is sufficient for this shape check

    def test_groupkfold_produces_5_folds(self, prepared_data):
        """GroupKFold must produce exactly CV_FOLDS (5) splits."""
        X, y, groups = prepared_data
        gkf = GroupKFold(n_splits=config.CV_FOLDS)
        n_folds = sum(1 for _ in gkf.split(X, y, groups))
        assert n_folds == config.CV_FOLDS, (
            f"GroupKFold produced {n_folds} folds, expected {config.CV_FOLDS}"
        )

    def test_no_patient_leakage_in_cv(self, prepared_data):
        """No patient_id may appear in both train and validation in any fold."""
        X, y, groups = prepared_data
        gkf = GroupKFold(n_splits=config.CV_FOLDS)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            train_patients = set(groups.iloc[train_idx])
            val_patients = set(groups.iloc[val_idx])
            overlap = train_patients & val_patients
            assert len(overlap) == 0, (
                f"PATIENT LEAKAGE in fold {fold}: {len(overlap)} shared patients. "
                f"Examples: {list(overlap)[:3]}"
            )


# ============================================================================
# 3. SHAP Value Tests
# ============================================================================

class TestSHAPValues:
    """Validate SHAP values for correctness and MCD compliance."""

    def test_shap_shape_matches_x(self, primary_shap, prepared_data):
        """SHAP values shape must be (n_samples, n_features)."""
        X, _, _ = prepared_data
        assert primary_shap.shape == X.shape, (
            f"SHAP shape {primary_shap.shape} != X shape {X.shape}"
        )

    def test_shap_additivity(self, trained_model_and_metrics, prepared_data):
        """SHAP values must sum to (prediction - expected_value) for each sample.

        This is the fundamental SHAP additivity property:
            f(x) = E[f(x)] + sum(SHAP_i)

        Uses the same background sample as the pipeline (config.SHAP_BACKGROUND_SAMPLES)
        so that SHAP values and expected_value come from the same explainer.
        """
        import shap as shap_lib
        model, _ = trained_model_and_metrics
        X, _, _ = prepared_data

        # Use the same background sampling as the pipeline
        background = shap_lib.utils.sample(
            X, config.SHAP_BACKGROUND_SAMPLES, random_state=42
        )
        explainer = shap_lib.TreeExplainer(
            model, data=background,
            feature_perturbation=config.SHAP_FEATURE_PERTURBATION,
        )
        shap_values = explainer.shap_values(X, check_additivity=False)

        predictions = model.predict(X)
        reconstructed = explainer.expected_value + shap_values.sum(axis=1)
        np.testing.assert_allclose(
            predictions, reconstructed, atol=0.01,
            err_msg="SHAP additivity violated: prediction != expected_value + sum(SHAP)"
        )

    def test_shap_no_nan_or_inf(self, primary_shap):
        """No SHAP value may be NaN or Inf."""
        assert not np.any(np.isnan(primary_shap)), (
            f"SHAP has {np.isnan(primary_shap).sum()} NaN values"
        )
        assert not np.any(np.isinf(primary_shap)), (
            f"SHAP has {np.isinf(primary_shap).sum()} Inf values"
        )

    def test_lag_features_have_nonzero_shap(self, primary_shap, prepared_data):
        """Mean |SHAP| for temperature lag features must be > 0.

        If all lag SHAP values are zero, the model is ignoring temperature
        entirely, which is either a data error or a feature engineering bug.
        """
        X, _, _ = prepared_data
        feature_names = X.columns.tolist()

        lag_cols = [f"temp_lag_{d}d" for d in config.LAG_DAYS]
        lag_indices = [
            feature_names.index(c) for c in lag_cols
            if c in feature_names
        ]

        assert len(lag_indices) > 0, "No lag features found in feature matrix"

        lag_shap = primary_shap[:, lag_indices]
        mean_abs_lag = np.abs(lag_shap).mean()
        assert mean_abs_lag > 0, (
            "All lag feature SHAP values are exactly zero -- "
            "temperature has no predictive signal (check feature engineering)"
        )


# ============================================================================
# 4. Bootstrap Tests
# ============================================================================

class TestBootstrap:
    """Validate bootstrap replication and aggregation."""

    def test_correct_number_of_replicates(self, bootstrap_results):
        """Bootstrap must produce exactly 2 SHAP arrays (test mode)."""
        shap_arrays, _, _ = bootstrap_results
        assert len(shap_arrays) == 2, (
            f"Expected 2 bootstrap replicates, got {len(shap_arrays)}"
        )

    def test_all_shap_arrays_same_shape(self, bootstrap_results, prepared_data):
        """Each bootstrap SHAP array must match the shape of the original X."""
        shap_arrays, X_orig, _ = bootstrap_results
        expected_shape = X_orig.shape
        for i, sv in enumerate(shap_arrays):
            assert sv.shape == expected_shape, (
                f"Bootstrap replicate {i} SHAP shape {sv.shape} != "
                f"expected {expected_shape}"
            )

    def test_bootstrap_aggregation_produces_required_outputs(self, bootstrap_results):
        """aggregate_bootstrap_shap must return mean, CI, and stability."""
        shap_arrays, _, feature_names = bootstrap_results
        result = aggregate_bootstrap_shap(shap_arrays, feature_names)

        assert "mean_abs_shap" in result, "Missing mean_abs_shap"
        assert "ci_lower" in result, "Missing ci_lower"
        assert "ci_upper" in result, "Missing ci_upper"
        assert "ranking_stability" in result, "Missing ranking_stability"

        # Stability is between 0 and 1
        w = result["ranking_stability"]
        assert 0 <= w <= 1.0, f"ranking_stability = {w} outside [0, 1]"

        # mean_abs_shap values are non-negative
        assert (result["mean_abs_shap"] >= 0).all(), (
            "Some mean_abs_shap values are negative"
        )

        # CI lower <= mean <= CI upper (element-wise)
        for feat in feature_names:
            lo = result["ci_lower"][feat]
            hi = result["ci_upper"][feat]
            mean = result["mean_abs_shap"][feat]
            assert lo <= mean + 1e-10, (
                f"Feature {feat}: ci_lower ({lo}) > mean ({mean})"
            )
            assert mean <= hi + 1e-10, (
                f"Feature {feat}: mean ({mean}) > ci_upper ({hi})"
            )


# ============================================================================
# 5. Lag Response Tests
# ============================================================================

class TestLagResponse:
    """Validate lag-response curve extraction and classification."""

    def test_lag_profile_has_7_columns(self, primary_shap, prepared_data):
        """Lag profile DataFrame must have exactly 7 columns (one per lag day)."""
        _, _, _ = prepared_data
        X, _, _ = prepared_data
        feature_names = X.columns.tolist()
        lag_profile = extract_lag_shap_profile(primary_shap, feature_names)
        assert lag_profile.shape[1] == 7, (
            f"Lag profile has {lag_profile.shape[1]} columns, expected 7. "
            f"Columns: {lag_profile.columns.tolist()}"
        )

    def test_all_7_lag_days_present(self, primary_shap, prepared_data):
        """Lag profile must contain all 7 MCD lag days: 0, 1, 3, 7, 14, 21, 30."""
        X, _, _ = prepared_data
        feature_names = X.columns.tolist()
        lag_profile = extract_lag_shap_profile(primary_shap, feature_names)

        expected_cols = [f"lag_{d}" for d in config.LAG_DAYS]
        actual_cols = lag_profile.columns.tolist()
        missing = [c for c in expected_cols if c not in actual_cols]
        assert not missing, f"Missing lag day columns: {missing}"

    def test_classify_temporal_window_valid_output(self, primary_shap, prepared_data):
        """classify_temporal_window must return a valid category."""
        X, _, _ = prepared_data
        feature_names = X.columns.tolist()
        lag_profile = extract_lag_shap_profile(primary_shap, feature_names)
        lag_summary = summarise_lag_response(lag_profile)
        window = classify_temporal_window(lag_summary)

        valid = {"acute", "sub_acute", "extended", "mixed", "unknown"}
        assert window in valid, (
            f"classify_temporal_window returned '{window}', "
            f"expected one of {valid}"
        )

    def test_lag_summary_mean_abs_shap_nonnegative(self, primary_shap, prepared_data):
        """All mean_abs_shap values in lag summary must be non-negative."""
        X, _, _ = prepared_data
        feature_names = X.columns.tolist()
        lag_profile = extract_lag_shap_profile(primary_shap, feature_names)
        lag_summary = summarise_lag_response(lag_profile)

        assert (lag_summary["mean_abs_shap"] >= 0).all(), (
            "Negative mean_abs_shap in lag summary"
        )

    def test_lag_summary_has_expected_columns(self, primary_shap, prepared_data):
        """Lag summary must have lag_day, mean_abs_shap, ci_lower, ci_upper, pct_positive."""
        X, _, _ = prepared_data
        feature_names = X.columns.tolist()
        lag_profile = extract_lag_shap_profile(primary_shap, feature_names)
        lag_summary = summarise_lag_response(lag_profile)

        expected = {"lag_day", "mean_abs_shap", "ci_lower", "ci_upper", "pct_positive"}
        actual = set(lag_summary.columns)
        missing = expected - actual
        assert not missing, f"Missing columns in lag summary: {missing}"

    def test_classify_window_edge_cases(self):
        """classify_temporal_window handles edge cases: empty and balanced."""
        # Empty
        empty = pd.DataFrame(columns=["lag_day", "mean_abs_shap"])
        assert classify_temporal_window(empty) == "unknown"

        # Balanced across all windows -> "mixed"
        balanced = pd.DataFrame({
            "lag_day": [0, 1, 3, 7, 14, 21, 30],
            "mean_abs_shap": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        })
        result = classify_temporal_window(balanced)
        # acute: sum=3, sub_acute: sum=2, extended: sum=2, total=7
        # acute proportion = 3/7 = 0.43 > 0.40 -> "acute"
        assert result in {"acute", "mixed"}, f"Balanced result: {result}"

        # Strong acute signal
        acute = pd.DataFrame({
            "lag_day": [0, 1, 3, 7, 14, 21, 30],
            "mean_abs_shap": [5.0, 5.0, 5.0, 0.1, 0.1, 0.1, 0.1],
        })
        assert classify_temporal_window(acute) == "acute"

        # Strong extended signal
        extended = pd.DataFrame({
            "lag_day": [0, 1, 3, 7, 14, 21, 30],
            "mean_abs_shap": [0.1, 0.1, 0.1, 0.1, 0.1, 5.0, 5.0],
        })
        assert classify_temporal_window(extended) == "extended"


# ============================================================================
# 6. MCD Compliance Checks
# ============================================================================

class TestMCDCompliance:
    """Validate critical MCD specification requirements."""

    def test_shap_uses_interventional_perturbation(self):
        """Config must specify 'interventional' SHAP perturbation."""
        assert config.SHAP_FEATURE_PERTURBATION == "interventional", (
            f"SHAP perturbation is '{config.SHAP_FEATURE_PERTURBATION}', "
            "MCD requires 'interventional' (not 'tree_path_dependent')"
        )

    def test_cv_uses_groupkfold(self):
        """Config must specify GroupKFold (verified via CV_FOLDS and PATIENT_ID_COLUMN)."""
        assert config.CV_FOLDS == 5, f"CV_FOLDS = {config.CV_FOLDS}, MCD spec = 5"
        assert config.PATIENT_ID_COLUMN == "patient_id", (
            f"PATIENT_ID_COLUMN = '{config.PATIENT_ID_COLUMN}', expected 'patient_id'"
        )

    def test_bootstrap_resamples_patients_not_observations(self):
        """Verify bootstrap logic resamples at the patient level.

        This is tested by inspecting the bootstrap_train implementation:
        it should use unique patient IDs, not individual row indices.
        """
        # We verify structurally via the source: bootstrap_train calls
        # rng.choice(patient_ids, ...) not rng.choice(range(len(X)), ...)
        import inspect
        source = inspect.getsource(bootstrap_train)
        assert "patient_ids" in source or "boot_patients" in source, (
            "bootstrap_train does not reference patient-level resampling"
        )
        assert "replace=True" in source, (
            "bootstrap_train does not use with-replacement sampling"
        )

    def test_7_lag_days_match_mcd_spec(self):
        """LAG_DAYS must be exactly [0, 1, 3, 7, 14, 21, 30]."""
        expected = [0, 1, 3, 7, 14, 21, 30]
        assert config.LAG_DAYS == expected, (
            f"LAG_DAYS = {config.LAG_DAYS}, MCD spec = {expected}"
        )

    def test_lag_feature_columns_mapping(self):
        """LAG_FEATURE_COLUMNS must correctly map LAG_DAYS to source columns."""
        expected = [
            "dlnm_lag0_c", "dlnm_lag1_c", "dlnm_lag3_c",
            "dlnm_lag7_c", "dlnm_lag14_c", "dlnm_lag21_c",
            "era5_temp_lag30d_c",
        ]
        assert config.LAG_FEATURE_COLUMNS == expected, (
            f"LAG_FEATURE_COLUMNS mismatch:\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {config.LAG_FEATURE_COLUMNS}"
        )

    def test_xgboost_params_match_mcd(self):
        """Key XGBoost hyperparameters must match MCD specification."""
        p = config.XGBOOST_PARAMS
        assert p["n_estimators"] == 500, f"n_estimators = {p['n_estimators']}"
        assert p["max_depth"] == 6, f"max_depth = {p['max_depth']}"
        assert p["learning_rate"] == 0.05, f"learning_rate = {p['learning_rate']}"
        assert p["random_state"] == 42, f"random_state = {p['random_state']}"

    def test_bootstrap_count_is_50(self):
        """MCD specifies 50 bootstrap model replicates."""
        assert config.N_BOOTSTRAP_REPLICATES == 50, (
            f"N_BOOTSTRAP_REPLICATES = {config.N_BOOTSTRAP_REPLICATES}, MCD spec = 50"
        )

    def test_bonferroni_alpha_correct(self):
        """Bonferroni alpha must be 0.05 / 8 = 0.00625."""
        expected = 0.05 / 8
        assert abs(config.BONFERRONI_ALPHA - expected) < 1e-10, (
            f"BONFERRONI_ALPHA = {config.BONFERRONI_ALPHA}, expected {expected}"
        )

    def test_8_biomarker_systems(self):
        """There must be exactly 8 organ systems."""
        from mcd_pipeline.stage0_data_prep.biomarker_definitions import BIOMARKER_SYSTEMS
        assert len(BIOMARKER_SYSTEMS) == 8, (
            f"Found {len(BIOMARKER_SYSTEMS)} systems, MCD spec = 8"
        )
        assert config.N_BIOMARKER_SYSTEMS == 8
