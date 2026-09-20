"""Tests for mcd_pipeline.config — verify MCD parameters are correct."""

import pytest

from mcd_pipeline import config


class TestPaths:

    def test_project_root_exists(self):
        assert config.PROJECT_ROOT.exists()

    def test_final_datasets_dir_exists(self):
        assert config.FINAL_DATASETS.exists()

    def test_tidy_datasets_dir_exists(self):
        assert config.TIDY_DIR.exists()

    def test_tidy_biomarkers_csv_exists(self):
        assert config.TIDY_BIOMARKERS_CSV.exists()

    def test_tidy_climate_csv_exists(self):
        assert config.TIDY_CLIMATE_CSV.exists()

    def test_tidy_demographics_csv_exists(self):
        assert config.TIDY_DEMOGRAPHICS_CSV.exists()

    def test_tidy_socioeconomic_csv_exists(self):
        assert config.TIDY_SOCIOECONOMIC_CSV.exists()


class TestLagParameters:

    def test_lag_days_values(self):
        assert config.LAG_DAYS == [0, 1, 3, 7, 14, 21, 30]

    def test_lag_feature_columns_count(self):
        # 6 DLNM lags (0,1,3,7,14,21) + 1 ERA5 rolling (30d) = 7
        assert len(config.LAG_FEATURE_COLUMNS) == 7

    def test_no_dlnm_lag30(self):
        # dlnm_lag30_c does NOT exist — lag 30 uses era5_temp_lag30d_c
        assert "dlnm_lag30_c" not in config.LAG_FEATURE_COLUMNS
        assert "era5_temp_lag30d_c" in config.LAG_FEATURE_COLUMNS

    def test_dlnm_columns_end_at_21(self):
        dlnm_cols = [c for c in config.LAG_FEATURE_COLUMNS if c.startswith("dlnm_")]
        max_lag = max(int(c.split("lag")[1].split("_")[0]) for c in dlnm_cols)
        assert max_lag == 21

    def test_dtr_column_defined(self):
        assert config.DTR_COLUMN == "era5_temp_range_c"


class TestCovariates:

    def test_socioeconomic_use_gcro_prefix(self):
        for cov in config.SOCIOECONOMIC_COVARIATES:
            assert cov.startswith("gcro_"), (
                f"SES covariate '{cov}' should use gcro_ prefix"
            )

    def test_demographic_covariates(self):
        assert "age_years" in config.DEMOGRAPHIC_COVARIATES
        assert "sex" in config.DEMOGRAPHIC_COVARIATES


class TestMCDParameters:

    def test_bootstrap_replicates(self):
        assert config.N_BOOTSTRAP_REPLICATES == 50

    def test_shap_interventional(self):
        assert config.SHAP_FEATURE_PERTURBATION == "interventional"

    def test_permutations(self):
        assert config.N_PERMUTATIONS == 500

    def test_fdr_threshold(self):
        assert config.FDR_THRESHOLD == 0.10

    def test_pca_variance(self):
        assert config.PCA_VARIANCE_THRESHOLD == 0.85

    def test_kmeans_range(self):
        assert list(config.KMEANS_K_RANGE) == [3, 4, 5, 6, 7, 8]

    def test_cluster_stability(self):
        assert config.CLUSTER_STABILITY_ITERATIONS == 500
        assert config.CLUSTER_STABILITY_THRESHOLD == 0.80

    def test_bonferroni_alpha(self):
        assert config.N_BIOMARKER_SYSTEMS == 8
        assert abs(config.BONFERRONI_ALPHA - 0.00625) < 1e-10

    def test_fdr_within_systems(self):
        assert config.FDR_Q_WITHIN_SYSTEMS == 0.10


class TestXGBoostParams:

    def test_required_keys(self):
        required = {"n_estimators", "max_depth", "learning_rate", "random_state"}
        assert required.issubset(set(config.XGBOOST_PARAMS.keys()))

    def test_reasonable_values(self):
        assert 100 <= config.XGBOOST_PARAMS["n_estimators"] <= 2000
        assert 1 <= config.XGBOOST_PARAMS["max_depth"] <= 15
        assert 0.001 <= config.XGBOOST_PARAMS["learning_rate"] <= 1.0


class TestSensitivityFlags:

    def test_sa5_imputation_enabled(self):
        assert config.SENSITIVITY_ANALYSES["sa5_imputation"] is True  # implemented: returns not_applicable for categorical SES

    def test_eight_sensitivity_analyses(self):
        assert len(config.SENSITIVITY_ANALYSES) == 8
