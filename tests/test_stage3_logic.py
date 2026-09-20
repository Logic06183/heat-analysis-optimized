"""
Tests for Stage 3 — Vulnerability Clustering logic.

Uses synthetic data to keep tests fast (no real SHAP files needed).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from mcd_pipeline.stage3_clustering.cluster_stability import bootstrap_cluster_stability
from mcd_pipeline.stage3_clustering.kmeans_clustering import (
    characterise_clusters,
    fit_kmeans_range,
    select_optimal_k,
)
from mcd_pipeline.stage3_clustering.shap_pca import (
    TEMP_LAG_FEATURES,
    aggregate_person_shap,
    fit_shap_pca,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_shap(rng):
    """50 observations × 10 features, random SHAP values."""
    n_obs, n_feat = 50, 10
    feature_names = [f"feat_{i}" for i in range(n_feat)]
    shap_vals = rng.standard_normal((n_obs, n_feat))
    patient_ids = pd.Series([f"P{i % 15:03d}" for i in range(n_obs)])
    return shap_vals, patient_ids, feature_names


@pytest.fixture
def synthetic_pc_scores(rng):
    """100 patients × 5 PCA components, with 3 clear clusters."""
    centers = np.array([[3, 0, 0, 0, 0], [-3, 0, 0, 0, 0], [0, 3, 0, 0, 0]])
    X = np.vstack([
        centers[i] + rng.standard_normal((33, 5)) * 0.3
        for i in range(3)
    ])
    ids = [f"P{i:04d}" for i in range(len(X))]
    return pd.DataFrame(X, index=ids, columns=[f"PC{i+1}" for i in range(5)])


# ---------------------------------------------------------------------------
# aggregate_person_shap
# ---------------------------------------------------------------------------

class TestAggregatePersonShap:
    def test_returns_dataframe(self, synthetic_shap):
        shap_vals, patient_ids, feature_names = synthetic_shap
        result = aggregate_person_shap(shap_vals, patient_ids, feature_names)
        assert isinstance(result, pd.DataFrame)

    def test_index_is_patient_id(self, synthetic_shap):
        shap_vals, patient_ids, feature_names = synthetic_shap
        result = aggregate_person_shap(shap_vals, patient_ids, feature_names)
        assert result.index.name == "patient_id"

    def test_one_row_per_patient(self, synthetic_shap):
        shap_vals, patient_ids, feature_names = synthetic_shap
        result = aggregate_person_shap(shap_vals, patient_ids, feature_names)
        assert len(result) == patient_ids.nunique()

    def test_columns_match_features(self, synthetic_shap):
        shap_vals, patient_ids, feature_names = synthetic_shap
        result = aggregate_person_shap(shap_vals, patient_ids, feature_names)
        assert list(result.columns) == feature_names

    def test_mean_aggregation(self):
        """Two visits for P001: mean SHAP should be average of both rows."""
        shap_vals = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
        patient_ids = pd.Series(["P001", "P001", "P002"])
        feature_names = ["f1", "f2"]
        result = aggregate_person_shap(shap_vals, patient_ids, feature_names)
        assert result.loc["P001", "f1"] == pytest.approx(2.0)
        assert result.loc["P001", "f2"] == pytest.approx(3.0)
        assert result.loc["P002", "f1"] == pytest.approx(10.0)

    def test_no_nan_in_output(self, synthetic_shap):
        shap_vals, patient_ids, feature_names = synthetic_shap
        result = aggregate_person_shap(shap_vals, patient_ids, feature_names)
        assert not result.isnull().any().any()


# ---------------------------------------------------------------------------
# fit_shap_pca
# ---------------------------------------------------------------------------

class TestFitShapPCA:
    def test_returns_three_objects(self, rng):
        person_shap = pd.DataFrame(rng.standard_normal((80, 20)),
                                   index=[f"P{i}" for i in range(80)])
        result = fit_shap_pca(person_shap, variance_threshold=0.85)
        assert len(result) == 3

    def test_pc_scores_shape(self, rng):
        person_shap = pd.DataFrame(rng.standard_normal((80, 20)),
                                   index=[f"P{i}" for i in range(80)])
        pca, scaler, pc_df = fit_shap_pca(person_shap, variance_threshold=0.85)
        assert pc_df.shape[0] == 80
        assert pc_df.shape[1] == pca.n_components_

    def test_index_preserved(self, rng):
        ids = [f"PAT_{i:04d}" for i in range(60)]
        person_shap = pd.DataFrame(rng.standard_normal((60, 15)), index=ids)
        _, _, pc_df = fit_shap_pca(person_shap)
        assert list(pc_df.index) == ids

    def test_variance_threshold_respected(self, rng):
        person_shap = pd.DataFrame(rng.standard_normal((100, 30)),
                                   index=[f"P{i}" for i in range(100)])
        pca, _, _ = fit_shap_pca(person_shap, variance_threshold=0.85)
        assert pca.explained_variance_ratio_.sum() >= 0.84  # allow tiny float diff

    def test_column_names_are_pc_numbered(self, rng):
        person_shap = pd.DataFrame(rng.standard_normal((60, 10)),
                                   index=[f"P{i}" for i in range(60)])
        _, _, pc_df = fit_shap_pca(person_shap)
        assert all(c.startswith("PC") for c in pc_df.columns)

    def test_no_nan_in_scores(self, rng):
        person_shap = pd.DataFrame(rng.standard_normal((60, 10)),
                                   index=[f"P{i}" for i in range(60)])
        _, _, pc_df = fit_shap_pca(person_shap)
        assert not pc_df.isnull().any().any()


# ---------------------------------------------------------------------------
# fit_kmeans_range
# ---------------------------------------------------------------------------

class TestFitKmeansRange:
    def test_returns_dict_for_each_k(self, synthetic_pc_scores):
        results = fit_kmeans_range(synthetic_pc_scores, k_range=range(2, 5), seed=42)
        assert set(results.keys()) == {2, 3, 4}

    def test_silhouette_between_minus1_and_1(self, synthetic_pc_scores):
        results = fit_kmeans_range(synthetic_pc_scores, k_range=range(2, 5), seed=42)
        for k, (km, sil) in results.items():
            assert -1.0 <= sil <= 1.0

    def test_clear_clusters_get_high_silhouette(self, synthetic_pc_scores):
        """The 3-cluster fixture should give silhouette > 0.5 at k=3."""
        results = fit_kmeans_range(synthetic_pc_scores, k_range=range(3, 4), seed=42)
        assert results[3][1] > 0.5

    def test_reproducible_with_same_seed(self, synthetic_pc_scores):
        r1 = fit_kmeans_range(synthetic_pc_scores, k_range=range(3, 4), seed=0)
        r2 = fit_kmeans_range(synthetic_pc_scores, k_range=range(3, 4), seed=0)
        assert r1[3][1] == r2[3][1]


# ---------------------------------------------------------------------------
# select_optimal_k
# ---------------------------------------------------------------------------

class TestSelectOptimalK:
    def test_selects_highest_silhouette(self):
        from sklearn.cluster import KMeans
        dummy_km = KMeans(n_clusters=2, random_state=0)
        results = {3: (dummy_km, 0.4), 4: (dummy_km, 0.7), 5: (dummy_km, 0.5)}
        assert select_optimal_k(results) == 4

    def test_single_k(self):
        from sklearn.cluster import KMeans
        dummy_km = KMeans(n_clusters=3, random_state=0)
        assert select_optimal_k({3: (dummy_km, 0.6)}) == 3


# ---------------------------------------------------------------------------
# characterise_clusters
# ---------------------------------------------------------------------------

class TestCharacteriseClusters:
    @pytest.fixture
    def patient_data(self, rng):
        n = 90
        return pd.DataFrame({
            "age_years": rng.uniform(25, 70, n),
            "sex_Male": rng.choice([0, 1], n),
            "hiv_status_Positive": rng.choice([0, 1], n),
            "gcro_dwelling_type_Informal": rng.choice([0, 1], n),
            "gcro_employment_status_Not employed": rng.choice([0, 1], n),
        }, index=[f"P{i:04d}" for i in range(n)])

    def test_returns_dataframe(self, patient_data, rng):
        labels = np.repeat([0, 1, 2], 30)
        result = characterise_clusters(labels, patient_data)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_cluster(self, patient_data, rng):
        labels = np.repeat([0, 1, 2], 30)
        result = characterise_clusters(labels, patient_data)
        assert len(result) == 3

    def test_n_column_sums_to_total(self, patient_data, rng):
        labels = np.repeat([0, 1, 2], 30)
        result = characterise_clusters(labels, patient_data)
        assert result["n"].sum() == len(patient_data)

    def test_pct_total_sums_to_100(self, patient_data, rng):
        labels = np.repeat([0, 1, 2], 30)
        result = characterise_clusters(labels, patient_data)
        assert result["pct_total"].sum() == pytest.approx(100.0, abs=0.5)


# ---------------------------------------------------------------------------
# bootstrap_cluster_stability
# ---------------------------------------------------------------------------

class TestBootstrapClusterStability:
    def test_returns_required_keys(self, synthetic_pc_scores):
        result = bootstrap_cluster_stability(
            synthetic_pc_scores, optimal_k=3, n_iterations=10, seed=42
        )
        for key in ["mean_ari", "std_ari", "ci_lower", "ci_upper",
                    "meets_threshold", "per_iteration_ari"]:
            assert key in result

    def test_ari_bounded(self, synthetic_pc_scores):
        result = bootstrap_cluster_stability(
            synthetic_pc_scores, optimal_k=3, n_iterations=10, seed=42
        )
        assert 0.0 <= result["mean_ari"] <= 1.0

    def test_per_iteration_length(self, synthetic_pc_scores):
        result = bootstrap_cluster_stability(
            synthetic_pc_scores, optimal_k=3, n_iterations=20, seed=42
        )
        assert len(result["per_iteration_ari"]) == 20

    def test_clear_clusters_meet_threshold(self, synthetic_pc_scores):
        """Well-separated 3-cluster fixture should be very stable."""
        result = bootstrap_cluster_stability(
            synthetic_pc_scores, optimal_k=3, n_iterations=50,
            threshold=0.80, seed=42
        )
        assert result["meets_threshold"]

    def test_ci_lower_le_mean_le_ci_upper(self, synthetic_pc_scores):
        result = bootstrap_cluster_stability(
            synthetic_pc_scores, optimal_k=3, n_iterations=20, seed=42
        )
        assert result["ci_lower"] <= result["mean_ari"] <= result["ci_upper"]

    def test_reproducible_with_same_seed(self, synthetic_pc_scores):
        r1 = bootstrap_cluster_stability(synthetic_pc_scores, optimal_k=3, n_iterations=10, seed=7)
        r2 = bootstrap_cluster_stability(synthetic_pc_scores, optimal_k=3, n_iterations=10, seed=7)
        assert r1["mean_ari"] == pytest.approx(r2["mean_ari"])


# ---------------------------------------------------------------------------
# TEMP_LAG_FEATURES constant
# ---------------------------------------------------------------------------

class TestTempLagFeatures:
    def test_seven_features(self):
        assert len(TEMP_LAG_FEATURES) == 7

    def test_all_expected_lags_present(self):
        for d in [0, 1, 3, 7, 14, 21, 30]:
            assert f"temp_lag_{d}d" in TEMP_LAG_FEATURES
