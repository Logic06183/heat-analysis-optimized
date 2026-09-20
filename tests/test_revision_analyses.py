"""
Tests for revision-analysis modules (Lancet Planetary Health peer review).

Tests cover:
  - parsimony_diagnostics: gap statistic, 1-SE rule, PC loadings export
  - cluster_proportion_cis: cluster summarisation, bootstrap CI computation
  - sa10_humidity: pre-computed ERA5 lag column helpers

All tests use synthetic data; no TIDY files or stage3 outputs are required.
"""

import numpy as np
import pandas as pd
import pytest

from mcd_pipeline.stage3_clustering.cluster_proportion_cis import (
    _ensure_features,
    _summarise_cluster,
    bootstrap_cluster_characteristics,
)
from mcd_pipeline.stage3_clustering.parsimony_diagnostics import (
    _within_cluster_dispersion,
    gap_statistic,
    select_optimal_k_one_se,
    export_pc_loadings,
)
from mcd_pipeline.sensitivity.sa10_humidity import (
    _precomputed_col,
    _copy_precomputed_lag_columns,
)
from mcd_pipeline import config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def three_cluster_data(rng):
    """200 points in 5-dim space with three well-separated clusters."""
    centers = np.array([[4, 0, 0, 0, 0], [-4, 0, 0, 0, 0], [0, 4, 0, 0, 0]])
    X = np.vstack([centers[i] + rng.standard_normal((66, 5)) * 0.4 for i in range(3)])
    labels = np.repeat([0, 1, 2], 66)
    return X, labels


@pytest.fixture
def patient_cluster_df(rng):
    """200-patient DataFrame with cluster labels and demographics."""
    n = 200
    return pd.DataFrame({
        "cluster": np.repeat([0, 1, 2], [80, 70, 50]),
        "age_years": rng.uniform(25, 65, n),
        "sex": rng.choice(["female", "male"], n),
        "hiv_status": rng.choice(["positive", "negative"], n),
        "gcro_dwelling_type": rng.choice(["formal", "informal"], n),
        "gcro_employment_status": rng.choice(["Employed", "Not employed"], n),
        "composite_heat_sensitivity": rng.standard_normal(n),
    })


# ---------------------------------------------------------------------------
# _within_cluster_dispersion
# ---------------------------------------------------------------------------

class TestWithinClusterDispersion:
    def test_zero_for_identical_points(self):
        X = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        labels = np.array([0, 0, 0])
        assert _within_cluster_dispersion(X, labels) == pytest.approx(0.0)

    def test_positive_for_spread_points(self):
        X = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
        labels = np.array([0, 0, 1, 1])
        assert _within_cluster_dispersion(X, labels) > 0.0

    def test_multiple_clusters(self, three_cluster_data):
        X, labels = three_cluster_data
        w = _within_cluster_dispersion(X, labels)
        assert w > 0.0

    def test_merging_clusters_increases_dispersion(self):
        """If we treat two separated groups as one, dispersion should grow."""
        X = np.array([[0.0], [0.0], [10.0], [10.0]])
        labels_merged = np.array([0, 0, 0, 0])
        labels_split = np.array([0, 0, 1, 1])
        w_merged = _within_cluster_dispersion(X, labels_merged)
        w_split = _within_cluster_dispersion(X, labels_split)
        assert w_merged > w_split


# ---------------------------------------------------------------------------
# gap_statistic
# ---------------------------------------------------------------------------

class TestGapStatistic:
    def test_returns_dataframe(self, three_cluster_data):
        X, _ = three_cluster_data
        df = gap_statistic(X, k_values=[2, 3], n_references=3, seed=0)
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_k(self, three_cluster_data):
        X, _ = three_cluster_data
        df = gap_statistic(X, k_values=[2, 3, 4], n_references=3, seed=0)
        assert len(df) == 3

    def test_required_columns(self, three_cluster_data):
        X, _ = three_cluster_data
        df = gap_statistic(X, k_values=[2, 3], n_references=3, seed=0)
        for col in ["k", "gap", "sk", "log_W_k", "ref_log_W_k_mean", "one_se_rule_satisfied"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_gap_positive_for_structured_data(self, three_cluster_data):
        """Uniform reference should have higher dispersion than structured clusters."""
        X, _ = three_cluster_data
        df = gap_statistic(X, k_values=[3], n_references=5, seed=0)
        assert df.loc[df["k"] == 3, "gap"].values[0] > 0.0

    def test_sk_positive(self, three_cluster_data):
        X, _ = three_cluster_data
        df = gap_statistic(X, k_values=[2, 3], n_references=3, seed=0)
        assert (df["sk"] > 0).all()

    def test_k_sorted_ascending(self, three_cluster_data):
        X, _ = three_cluster_data
        df = gap_statistic(X, k_values=[4, 2, 3], n_references=3, seed=0)
        assert list(df["k"]) == sorted(df["k"])

    def test_last_row_has_nan_gap_next(self, three_cluster_data):
        X, _ = three_cluster_data
        df = gap_statistic(X, k_values=[2, 3], n_references=3, seed=0)
        assert pd.isna(df["gap_next"].iloc[-1])


# ---------------------------------------------------------------------------
# select_optimal_k_one_se
# ---------------------------------------------------------------------------

class TestSelectOptimalKOneSe:
    def _make_gap_df(self, data):
        """Build a minimal gap DataFrame from (k, gap, sk) tuples."""
        rows = [{"k": k, "gap": g, "sk": s} for k, g, s in data]
        df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
        df["gap_next"] = df["gap"].shift(-1)
        df["sk_next"] = df["sk"].shift(-1)
        df["one_se_rule_satisfied"] = df["gap"] >= (df["gap_next"] - df["sk_next"])
        return df

    def test_selects_smallest_k_satisfying_rule(self):
        # k=2 satisfies: gap[2]=3.5 >= gap[3]-sk[3] = 3.4-0.01 = 3.39
        df = self._make_gap_df([(2, 3.5, 0.01), (3, 3.4, 0.01), (4, 3.3, 0.01)])
        assert select_optimal_k_one_se(df) == 2

    def test_falls_back_to_argmax_when_no_rule_satisfied(self):
        # Monotonically increasing gap — no k satisfies the 1-SE rule
        df = self._make_gap_df([(2, 1.0, 0.5), (3, 2.0, 0.5), (4, 3.0, 0.5)])
        # No k satisfies gap[k] >= gap[k+1] - sk[k+1]; fallback to argmax
        result = select_optimal_k_one_se(df)
        assert result == 4  # argmax gap is k=4 (last k, also no next to beat)

    def test_returns_int(self, three_cluster_data):
        X, _ = three_cluster_data
        df = gap_statistic(X, k_values=[2, 3, 4], n_references=3, seed=0)
        k = select_optimal_k_one_se(df)
        assert isinstance(k, int)


# ---------------------------------------------------------------------------
# export_pc_loadings (unit tests against synthetic arrays via tmp_path)
# ---------------------------------------------------------------------------

class TestExportPcLoadings:
    def test_returns_dataframe(self, tmp_path, rng):
        # Write synthetic pca_components.npy and pca_summary.json
        n_components, n_features = 5, 14  # 2 biomarkers × 7 lags
        components = rng.standard_normal((n_components, n_features))
        np.save(tmp_path / "pca_components.npy", components)
        import json
        summary = {
            "n_components": n_components,
            "included_biomarkers": ["hematocrit", "hemoglobin"],
        }
        (tmp_path / "pca_summary.json").write_text(json.dumps(summary))

        result = export_pc_loadings(tmp_path, n_pcs=3)
        assert isinstance(result, pd.DataFrame)

    def test_columns_are_pc_numbered(self, tmp_path, rng):
        n_components, n_features = 5, 14
        components = rng.standard_normal((n_components, n_features))
        np.save(tmp_path / "pca_components.npy", components)
        import json
        summary = {"n_components": n_components, "included_biomarkers": ["hematocrit", "hemoglobin"]}
        (tmp_path / "pca_summary.json").write_text(json.dumps(summary))

        result = export_pc_loadings(tmp_path, n_pcs=3)
        assert list(result.columns) == ["PC1", "PC2", "PC3"]

    def test_index_is_biomarker_name(self, tmp_path, rng):
        n_components, n_features = 5, 14
        components = rng.standard_normal((n_components, n_features))
        np.save(tmp_path / "pca_components.npy", components)
        import json
        biomarkers = ["hematocrit", "hemoglobin"]
        summary = {"n_components": n_components, "included_biomarkers": biomarkers}
        (tmp_path / "pca_summary.json").write_text(json.dumps(summary))

        result = export_pc_loadings(tmp_path, n_pcs=3)
        assert set(result.index) == set(biomarkers)

    def test_loadings_are_non_negative(self, tmp_path, rng):
        """export_pc_loadings returns mean absolute loadings, so all >= 0."""
        n_components, n_features = 5, 14
        components = rng.standard_normal((n_components, n_features))
        np.save(tmp_path / "pca_components.npy", components)
        import json
        summary = {"n_components": n_components, "included_biomarkers": ["a", "b"]}
        (tmp_path / "pca_summary.json").write_text(json.dumps(summary))

        result = export_pc_loadings(tmp_path, n_pcs=3)
        assert (result.values >= 0).all()

    def test_raises_if_components_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            export_pc_loadings(tmp_path, n_pcs=3)


# ---------------------------------------------------------------------------
# _ensure_features
# ---------------------------------------------------------------------------

class TestEnsureFeatures:
    def test_adds_composite_if_missing(self):
        df = pd.DataFrame({"age_years": [30, 40]})
        out = _ensure_features(df)
        assert "composite_heat_sensitivity" in out.columns
        assert out["composite_heat_sensitivity"].isna().all()

    def test_does_not_modify_if_present(self):
        df = pd.DataFrame({
            "age_years": [30, 40],
            "composite_heat_sensitivity": [0.5, -0.5],
        })
        out = _ensure_features(df)
        assert list(out["composite_heat_sensitivity"]) == [0.5, -0.5]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"age_years": [30]})
        _ensure_features(df)
        assert "composite_heat_sensitivity" not in df.columns


# ---------------------------------------------------------------------------
# _summarise_cluster
# ---------------------------------------------------------------------------

class TestSummariseCluster:
    def test_prevalence_correct(self):
        df_cluster = pd.DataFrame({"age_years": [30, 40], "composite_heat_sensitivity": [0.1, 0.2]})
        result = _summarise_cluster(df_cluster, n_total=10)
        assert result["prevalence_pct"] == pytest.approx(20.0)

    def test_mean_age_correct(self):
        df_cluster = pd.DataFrame({"age_years": [30.0, 50.0], "composite_heat_sensitivity": [0.0, 0.0]})
        result = _summarise_cluster(df_cluster, n_total=4)
        assert result["mean_age_years"] == pytest.approx(40.0)

    def test_pct_female_correct(self):
        df_cluster = pd.DataFrame({
            "age_years": [30, 40, 50],
            "composite_heat_sensitivity": [0.0, 0.0, 0.0],
            "sex": ["female", "male", "female"],
        })
        result = _summarise_cluster(df_cluster, n_total=3)
        assert result["pct_female"] == pytest.approx(200.0 / 3)

    def test_pct_hiv_positive(self):
        df_cluster = pd.DataFrame({
            "age_years": [30, 40],
            "composite_heat_sensitivity": [0.0, 0.0],
            "hiv_status": ["positive", "negative"],
        })
        result = _summarise_cluster(df_cluster, n_total=2)
        assert result["pct_hiv_positive"] == pytest.approx(50.0)

    def test_pct_unemployed_with_space_variant(self):
        """Data uses 'Not employed' (with space) — ensure the match works."""
        df_cluster = pd.DataFrame({
            "age_years": [30, 40, 50],
            "composite_heat_sensitivity": [0.0, 0.0, 0.0],
            "gcro_employment_status": ["Not employed", "Employed", "Not employed"],
        })
        result = _summarise_cluster(df_cluster, n_total=3)
        assert result["pct_unemployed"] == pytest.approx(200.0 / 3)

    def test_nan_for_missing_column(self):
        df_cluster = pd.DataFrame({"age_years": [30]})
        result = _summarise_cluster(df_cluster, n_total=5)
        assert np.isnan(result["pct_female"])


# ---------------------------------------------------------------------------
# bootstrap_cluster_characteristics
# ---------------------------------------------------------------------------

class TestBootstrapClusterCharacteristics:
    def test_returns_dataframe_and_array(self, patient_cluster_df):
        table, raw = bootstrap_cluster_characteristics(
            patient_cluster_df, n_bootstrap=50, seed=0
        )
        assert isinstance(table, pd.DataFrame)
        assert isinstance(raw, np.ndarray)

    def test_one_row_per_cluster_per_metric(self, patient_cluster_df):
        table, _ = bootstrap_cluster_characteristics(
            patient_cluster_df, n_bootstrap=50, seed=0
        )
        n_clusters = patient_cluster_df["cluster"].nunique()
        n_metrics = len(table["metric"].unique())
        assert len(table) == n_clusters * n_metrics

    def test_ci_lower_le_mean_le_ci_upper(self, patient_cluster_df):
        table, _ = bootstrap_cluster_characteristics(
            patient_cluster_df, n_bootstrap=100, seed=0
        )
        valid = table[table["n_bootstrap_used"] > 0]
        assert (valid["ci_lower_2.5"] <= valid["mean"]).all()
        assert (valid["mean"] <= valid["ci_upper_97.5"]).all()

    def test_n_bootstrap_used_equals_n_bootstrap(self, patient_cluster_df):
        n_boot = 80
        table, _ = bootstrap_cluster_characteristics(
            patient_cluster_df, n_bootstrap=n_boot, seed=0
        )
        assert (table["n_bootstrap_used"] == n_boot).all()

    def test_raw_array_shape(self, patient_cluster_df):
        n_boot = 50
        table, raw = bootstrap_cluster_characteristics(
            patient_cluster_df, n_bootstrap=n_boot, seed=0
        )
        n_clusters = patient_cluster_df["cluster"].nunique()
        n_metrics = len(table["metric"].unique())
        assert raw.shape == (n_clusters, n_metrics, n_boot)

    def test_prevalence_sum_is_100(self, patient_cluster_df):
        table, _ = bootstrap_cluster_characteristics(
            patient_cluster_df, n_bootstrap=50, seed=0
        )
        prev = table[table["metric"] == "prevalence_pct"]
        assert prev["point_estimate"].sum() == pytest.approx(100.0, abs=0.5)

    def test_reproducible_with_same_seed(self, patient_cluster_df):
        t1, _ = bootstrap_cluster_characteristics(patient_cluster_df, n_bootstrap=30, seed=7)
        t2, _ = bootstrap_cluster_characteristics(patient_cluster_df, n_bootstrap=30, seed=7)
        assert (t1["ci_lower_2.5"].values == t2["ci_lower_2.5"].values).all()

    def test_different_seeds_different_cis(self, patient_cluster_df):
        t1, _ = bootstrap_cluster_characteristics(patient_cluster_df, n_bootstrap=50, seed=1)
        t2, _ = bootstrap_cluster_characteristics(patient_cluster_df, n_bootstrap=50, seed=99)
        # CIs should differ (not identical) with different seeds
        assert not (t1["ci_lower_2.5"].values == t2["ci_lower_2.5"].values).all()


# ---------------------------------------------------------------------------
# SA10 humidity: pre-computed ERA5 lag column helpers
# ---------------------------------------------------------------------------

class TestPrecomputedCol:
    def test_lag0_has_no_d_suffix(self):
        """TIDY convention: lag-0 column is 'apparent_temp_lag0_c', not 'lag0d_c'."""
        assert _precomputed_col("apparent_temp", 0) == "apparent_temp_lag0_c"

    def test_lag_nonzero_has_d_suffix(self):
        assert _precomputed_col("apparent_temp", 7) == "apparent_temp_lag7d_c"

    def test_all_lag_days_well_formed(self):
        for lag in config.LAG_DAYS:
            col = _precomputed_col("heat_index", lag)
            assert col.startswith("heat_index_lag")
            assert col.endswith("_c")

    def test_wet_bulb_prefix(self):
        assert _precomputed_col("wet_bulb", 30) == "wet_bulb_lag30d_c"


class TestCopyPrecomputedLagColumns:
    @pytest.fixture
    def df_with_precomputed(self):
        """Minimal DataFrame with pre-computed ERA5 lag columns for apparent_temp."""
        rng = np.random.default_rng(42)
        n = 50
        data = {"patient_id": range(n), "visit_date": pd.date_range("2020-01-01", periods=n)}
        for lag in config.LAG_DAYS:
            col = _precomputed_col("apparent_temp", lag)
            data[col] = rng.uniform(15.0, 35.0, n)
        return pd.DataFrame(data)

    def test_target_columns_created(self, df_with_precomputed):
        result = _copy_precomputed_lag_columns(df_with_precomputed, "apparent_temp", "apptemp")
        for lag in config.LAG_DAYS:
            assert f"apptemp_lag{lag}d_c" in result.columns

    def test_values_preserved(self, df_with_precomputed):
        result = _copy_precomputed_lag_columns(df_with_precomputed, "apparent_temp", "apptemp")
        for lag in config.LAG_DAYS:
            src = _precomputed_col("apparent_temp", lag)
            dst = f"apptemp_lag{lag}d_c"
            pd.testing.assert_series_equal(
                result[src].reset_index(drop=True),
                result[dst].reset_index(drop=True),
                check_names=False,
            )

    def test_does_not_modify_input(self, df_with_precomputed):
        cols_before = set(df_with_precomputed.columns)
        _copy_precomputed_lag_columns(df_with_precomputed, "apparent_temp", "apptemp")
        assert set(df_with_precomputed.columns) == cols_before

    def test_lag0_normalised_to_lag0d(self, df_with_precomputed):
        """lag0_c (no 'd') must map to lag0d_c (with 'd') for the swap helper."""
        result = _copy_precomputed_lag_columns(df_with_precomputed, "apparent_temp", "apptemp")
        # Source uses "lag0_c", destination must use "lag0d_c"
        assert "apptemp_lag0d_c" in result.columns
        assert "apptemp_lag0_c" not in result.columns

    def test_raises_on_missing_column(self):
        df_incomplete = pd.DataFrame({"some_col": [1, 2, 3]})
        with pytest.raises(KeyError, match="missing"):
            _copy_precomputed_lag_columns(df_incomplete, "apparent_temp", "apptemp")

    def test_no_nulls_in_output(self, df_with_precomputed):
        result = _copy_precomputed_lag_columns(df_with_precomputed, "apparent_temp", "apptemp")
        for lag in config.LAG_DAYS:
            assert result[f"apptemp_lag{lag}d_c"].isna().sum() == 0
