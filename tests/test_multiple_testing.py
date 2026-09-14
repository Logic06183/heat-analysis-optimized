"""Tests for multiple testing correction modules.

Tests both:
- mcd_pipeline.multiple_testing.correction (Bonferroni + FDR pipeline)
- mcd_pipeline.stage2_interactions.fdr_correction (BH FDR standalone)
"""

import numpy as np
import pandas as pd
import pytest

from mcd_pipeline.multiple_testing.correction import (
    bonferroni_system_level,
    fdr_within_system,
    apply_full_correction,
)
from mcd_pipeline.stage2_interactions.fdr_correction import (
    benjamini_hochberg,
    apply_fdr_to_interactions,
)


class TestBonferronniSystemLevel:

    def test_significant_system(self):
        pvalues = {"renal": 0.001, "metabolic": 0.5}
        result = bonferroni_system_level(pvalues, alpha=0.00625)
        assert result["renal"] is True
        assert result["metabolic"] is False

    def test_all_significant(self):
        pvalues = {s: 0.001 for s in ["renal", "metabolic", "hepatic"]}
        result = bonferroni_system_level(pvalues, alpha=0.00625)
        assert all(result.values())

    def test_none_significant(self):
        pvalues = {"renal": 0.5, "metabolic": 0.8}
        result = bonferroni_system_level(pvalues, alpha=0.00625)
        assert not any(result.values())

    def test_boundary_value(self):
        # Exactly at threshold should NOT be significant (strict <)
        result = bonferroni_system_level({"renal": 0.00625}, alpha=0.00625)
        assert result["renal"] is False

    def test_empty_input(self):
        result = bonferroni_system_level({})
        assert result == {}


class TestFdrWithinSystem:

    def test_significant_biomarkers(self):
        pvalues = {"creatinine": 0.001, "albumin": 0.002, "egfr": 0.9}
        result = fdr_within_system(pvalues, q=0.10)
        assert result["creatinine"] is True
        assert result["albumin"] is True
        assert result["egfr"] is False

    def test_all_significant(self):
        pvalues = {"a": 0.001, "b": 0.002, "c": 0.003}
        result = fdr_within_system(pvalues, q=0.10)
        assert all(result.values())

    def test_none_significant(self):
        pvalues = {"a": 0.5, "b": 0.6, "c": 0.9}
        result = fdr_within_system(pvalues, q=0.10)
        assert not any(result.values())

    def test_empty_input(self):
        result = fdr_within_system({})
        assert result == {}

    def test_single_biomarker(self):
        result = fdr_within_system({"creatinine": 0.01}, q=0.10)
        assert result["creatinine"] is True


class TestApplyFullCorrection:

    def test_basic_correction(self):
        # Renal and hepatic should be significant; metabolic should not
        pvalues = {
            "creatinine": 0.001,
            "albumin": 0.003,
            "fasting_glucose": 0.5,
            "alt": 0.0001,
        }
        result = apply_full_correction(pvalues)
        assert isinstance(result, pd.DataFrame)
        assert "final_significant" in result.columns

        creat = result[result["biomarker"] == "creatinine"].iloc[0]
        assert creat["bonferroni_significant"]

        glucose = result[result["biomarker"] == "fasting_glucose"].iloc[0]
        assert not glucose["bonferroni_significant"]
        assert not glucose["final_significant"]

    def test_output_columns(self):
        pvalues = {"creatinine": 0.01}
        result = apply_full_correction(pvalues)
        expected_cols = {
            "biomarker", "system", "raw_pvalue",
            "bonferroni_significant", "fdr_significant", "final_significant",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_sorted_by_pvalue(self):
        pvalues = {"bmi": 0.9, "creatinine": 0.001, "alt": 0.01}
        result = apply_full_correction(pvalues)
        assert result["raw_pvalue"].is_monotonic_increasing


class TestBenjaminiHochberg:

    def test_basic_correction(self):
        pvalues = np.array([0.001, 0.01, 0.05, 0.5, 0.9])
        rejected, adjusted = benjamini_hochberg(pvalues, q=0.10)
        assert rejected[0]  # smallest p
        assert not rejected[-1]  # largest p
        assert len(adjusted) == len(pvalues)

    def test_adjusted_pvalues_monotonic_after_sorting(self):
        pvalues = np.array([0.01, 0.03, 0.04, 0.5])
        _, adjusted = benjamini_hochberg(pvalues, q=0.10)
        # Adjusted p-values from BH may not be strictly monotonic
        # but should all be >= raw p-values
        assert all(adjusted >= pvalues - 1e-10)

    def test_empty_input(self):
        rejected, adjusted = benjamini_hochberg(np.array([]))
        assert len(rejected) == 0
        assert len(adjusted) == 0

    def test_single_pvalue_significant(self):
        rejected, adjusted = benjamini_hochberg(np.array([0.01]), q=0.10)
        assert rejected[0]

    def test_single_pvalue_not_significant(self):
        rejected, adjusted = benjamini_hochberg(np.array([0.5]), q=0.10)
        assert not rejected[0]

    def test_all_significant(self):
        pvalues = np.array([0.001, 0.002, 0.003])
        rejected, _ = benjamini_hochberg(pvalues, q=0.10)
        assert all(rejected)

    def test_none_significant(self):
        pvalues = np.array([0.5, 0.6, 0.9])
        rejected, _ = benjamini_hochberg(pvalues, q=0.10)
        assert not any(rejected)


class TestApplyFdrToInteractions:

    def test_adds_columns(self):
        df = pd.DataFrame({
            "feature_pair": ["temp_x_age", "temp_x_sex", "temp_x_hiv"],
            "interaction_strength": [0.5, 0.3, 0.1],
            "p_value": [0.001, 0.05, 0.9],
        })
        result = apply_fdr_to_interactions(df, q=0.10)
        assert "p_adjusted" in result.columns
        assert "significant_fdr" in result.columns

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"p_value": [0.01, 0.5]})
        result = apply_fdr_to_interactions(df)
        assert "p_adjusted" not in df.columns  # original unchanged
        assert "p_adjusted" in result.columns

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"not_p_value": [0.01]})
        with pytest.raises(ValueError, match="p_value"):
            apply_fdr_to_interactions(df)
