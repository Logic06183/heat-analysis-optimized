"""
Tests for Stage 2 — Interaction Detection logic.

Uses synthetic data and tiny models to keep tests fast.
Permutation counts are reduced to n_permutations=3 for speed.
"""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from mcd_pipeline import config
from mcd_pipeline.stage2_interactions.fdr_correction import (
    apply_fdr_to_interactions,
    benjamini_hochberg,
)
from mcd_pipeline.stage2_interactions.interaction_detector import (
    compute_interaction_values,
    detect_temperature_interactions,
    rank_interactions,
)
from mcd_pipeline.stage2_interactions.permutation_null import (
    compute_permutation_pvalue,
    generate_permutation_null,
    test_top_interactions as run_top_interactions_test,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_X():
    """Small feature matrix with temp_lag_ prefixed columns for interaction tests."""
    rng = np.random.RandomState(42)
    n = 50
    return pd.DataFrame({
        "temp_lag_0d": rng.randn(n),
        "temp_lag_7d": rng.randn(n),
        "age_years": rng.uniform(20, 60, n),
        "gcro_income_bracket": rng.randint(0, 4, n).astype(float),
    })


@pytest.fixture(scope="module")
def trained_model(synthetic_X):
    """Tiny XGBoost model trained on synthetic data."""
    rng = np.random.RandomState(42)
    y = synthetic_X["temp_lag_0d"] * 2 + rng.randn(len(synthetic_X)) * 0.1
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=42, n_jobs=1)
    model.fit(synthetic_X, y)
    return model


@pytest.fixture(scope="module")
def interaction_tensor(trained_model, synthetic_X):
    """Pre-computed interaction tensor for reuse."""
    tensor, X_sub = compute_interaction_values(trained_model, synthetic_X, sample_size=20)
    return tensor, X_sub, synthetic_X.columns.tolist()


# ---------------------------------------------------------------------------
# interaction_detector tests
# ---------------------------------------------------------------------------

class TestComputeInteractionValues:

    def test_returns_3d_tensor(self, trained_model, synthetic_X):
        tensor, X_sub = compute_interaction_values(trained_model, synthetic_X, sample_size=20)
        assert tensor.ndim == 3

    def test_tensor_shape(self, trained_model, synthetic_X):
        n_sub = 20
        tensor, X_sub = compute_interaction_values(trained_model, synthetic_X, sample_size=n_sub)
        n_features = synthetic_X.shape[1]
        assert tensor.shape == (len(X_sub), n_features, n_features)

    def test_tensor_symmetric(self, trained_model, synthetic_X):
        """Interaction tensor should be symmetric: [i,j] == [j,i] for each sample."""
        tensor, _ = compute_interaction_values(trained_model, synthetic_X, sample_size=15)
        for s in range(tensor.shape[0]):
            np.testing.assert_allclose(tensor[s], tensor[s].T, atol=1e-5)

    def test_subsamples_when_large(self, trained_model, synthetic_X):
        """Should subsample to sample_size when X is larger."""
        _, X_sub = compute_interaction_values(trained_model, synthetic_X, sample_size=10)
        assert len(X_sub) == 10

    def test_uses_full_when_small(self, trained_model, synthetic_X):
        """Should use all rows when X is smaller than sample_size."""
        _, X_sub = compute_interaction_values(trained_model, synthetic_X, sample_size=1000)
        assert len(X_sub) == len(synthetic_X)


class TestRankInteractions:

    def test_returns_dataframe(self, interaction_tensor):
        tensor, _, feature_names = interaction_tensor
        ranked = rank_interactions(tensor, feature_names)
        assert isinstance(ranked, pd.DataFrame)

    def test_required_columns(self, interaction_tensor):
        tensor, _, feature_names = interaction_tensor
        ranked = rank_interactions(tensor, feature_names)
        assert {"feature_i", "feature_j", "mean_abs_interaction", "rank"}.issubset(ranked.columns)

    def test_upper_triangle_only(self, interaction_tensor):
        """feature_i index should always be less than feature_j index."""
        tensor, _, feature_names = interaction_tensor
        ranked = rank_interactions(tensor, feature_names, top_n=100)
        for _, row in ranked.iterrows():
            assert row["idx_i"] < row["idx_j"]

    def test_sorted_descending(self, interaction_tensor):
        tensor, _, feature_names = interaction_tensor
        ranked = rank_interactions(tensor, feature_names)
        vals = ranked["mean_abs_interaction"].values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_top_n_respected(self, interaction_tensor):
        tensor, _, feature_names = interaction_tensor
        ranked = rank_interactions(tensor, feature_names, top_n=3)
        assert len(ranked) <= 3


class TestDetectTemperatureInteractions:

    def test_finds_temp_pairs(self, interaction_tensor):
        tensor, _, feature_names = interaction_tensor
        result = detect_temperature_interactions(tensor, feature_names)
        assert not result.empty

    def test_required_columns(self, interaction_tensor):
        tensor, _, feature_names = interaction_tensor
        result = detect_temperature_interactions(tensor, feature_names)
        assert {"temp_feature", "modifier", "mean_abs_interaction", "direction"}.issubset(
            result.columns
        )

    def test_temp_features_have_correct_prefix(self, interaction_tensor):
        tensor, _, feature_names = interaction_tensor
        result = detect_temperature_interactions(tensor, feature_names)
        assert all(r.startswith("temp_lag_") for r in result["temp_feature"])

    def test_direction_values(self, interaction_tensor):
        tensor, _, feature_names = interaction_tensor
        result = detect_temperature_interactions(tensor, feature_names)
        assert set(result["direction"].unique()).issubset({"positive", "negative"})

    def test_empty_when_no_temp_features(self, trained_model):
        """Returns empty DataFrame when no temp_lag_ features present."""
        X_no_temp = pd.DataFrame({
            "age": np.random.randn(20),
            "income": np.random.randn(20),
        })
        model = xgb.XGBRegressor(n_estimators=5, random_state=42, n_jobs=1)
        model.fit(X_no_temp, np.random.randn(20))
        tensor, _ = compute_interaction_values(model, X_no_temp, sample_size=20)
        result = detect_temperature_interactions(tensor, X_no_temp.columns.tolist())
        assert result.empty


# ---------------------------------------------------------------------------
# permutation_null tests
# ---------------------------------------------------------------------------

class TestPermutationNull:

    def test_null_length(self, trained_model, synthetic_X):
        null = generate_permutation_null(
            trained_model, synthetic_X,
            "temp_lag_0d", "gcro_income_bracket",
            n_permutations=3, seed=42,
        )
        assert len(null) == 3

    def test_null_non_negative(self, trained_model, synthetic_X):
        """Mean absolute interaction values should always be >= 0."""
        null = generate_permutation_null(
            trained_model, synthetic_X,
            "temp_lag_0d", "gcro_income_bracket",
            n_permutations=3, seed=42,
        )
        assert np.all(null >= 0)

    def test_missing_feature_returns_zeros(self, trained_model, synthetic_X):
        null = generate_permutation_null(
            trained_model, synthetic_X,
            "nonexistent_feature", "gcro_income_bracket",
            n_permutations=3,
        )
        np.testing.assert_array_equal(null, np.zeros(3))


class TestComputePValue:

    def test_observed_above_all_null(self):
        null = np.array([0.1, 0.2, 0.3])
        p = compute_permutation_pvalue(observed=1.0, null_distribution=null)
        assert p == pytest.approx(1.0 / 3)  # floor at 1/n

    def test_observed_below_all_null(self):
        null = np.array([1.0, 2.0, 3.0])
        p = compute_permutation_pvalue(observed=0.1, null_distribution=null)
        assert p == pytest.approx(1.0)

    def test_empty_null_returns_one(self):
        p = compute_permutation_pvalue(observed=1.0, null_distribution=np.array([]))
        assert p == 1.0

    def test_p_value_in_range(self):
        rng = np.random.RandomState(42)
        null = rng.uniform(0, 1, 100)
        p = compute_permutation_pvalue(observed=0.5, null_distribution=null)
        assert 0 < p <= 1.0


class TestTopInteractions:

    def test_adds_p_value_column(self, trained_model, synthetic_X, interaction_tensor):
        tensor, _, feature_names = interaction_tensor
        ranked = rank_interactions(tensor, feature_names, top_n=5)
        result = run_top_interactions_test(
            trained_model, synthetic_X, ranked, n_permutations=3
        )
        assert "p_value" in result.columns

    def test_non_temp_pairs_get_nan_pvalue(self, trained_model, synthetic_X, interaction_tensor):
        """Non-temperature pairs should not be permutation-tested (p_value=NaN)."""
        tensor, _, feature_names = interaction_tensor
        ranked = rank_interactions(tensor, feature_names, top_n=10)
        result = run_top_interactions_test(
            trained_model, synthetic_X, ranked, n_permutations=3
        )
        non_temp = result[
            ~result["feature_i"].str.startswith("temp_lag_") &
            ~result["feature_j"].str.startswith("temp_lag_")
        ]
        assert non_temp["p_value"].isna().all()


# ---------------------------------------------------------------------------
# fdr_correction tests
# ---------------------------------------------------------------------------

class TestBenjaminiHochberg:

    def test_returns_two_arrays(self):
        rejected, adjusted = benjamini_hochberg(np.array([0.01, 0.05, 0.5, 0.9]))
        assert len(rejected) == 4
        assert len(adjusted) == 4

    def test_empty_input(self):
        rejected, adjusted = benjamini_hochberg(np.array([]))
        assert len(rejected) == 0
        assert len(adjusted) == 0

    def test_very_small_p_rejected(self):
        rejected, _ = benjamini_hochberg(np.array([0.001, 0.8, 0.9]), q=0.10)
        assert rejected[0]

    def test_large_p_not_rejected(self):
        rejected, _ = benjamini_hochberg(np.array([0.5, 0.8, 0.9]), q=0.10)
        assert not any(rejected)

    def test_adjusted_p_geq_raw(self):
        """BH-adjusted p-values should generally be >= raw p-values."""
        p_raw = np.array([0.01, 0.02, 0.03, 0.05])
        _, adjusted = benjamini_hochberg(p_raw, q=0.10)
        assert np.all(adjusted >= p_raw - 1e-10)


class TestApplyFDRToInteractions:

    def test_adds_required_columns(self):
        df = pd.DataFrame({
            "feature_i": ["a", "b"],
            "feature_j": ["c", "d"],
            "p_value": [0.001, 0.9],
        })
        result = apply_fdr_to_interactions(df)
        assert "p_adjusted" in result.columns
        assert "significant_fdr" in result.columns

    def test_raises_without_p_value_column(self):
        df = pd.DataFrame({"feature_i": ["a"], "feature_j": ["b"]})
        with pytest.raises(ValueError, match="p_value"):
            apply_fdr_to_interactions(df)

    def test_does_not_modify_input(self):
        df = pd.DataFrame({"p_value": [0.01, 0.5]})
        original_cols = df.columns.tolist()
        apply_fdr_to_interactions(df)
        assert df.columns.tolist() == original_cols
