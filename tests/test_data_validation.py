"""Tests for mcd_pipeline.stage0_data_prep.data_validation."""

import numpy as np
import pandas as pd
import pytest

from mcd_pipeline.stage0_data_prep.data_validation import (
    validate_required_columns,
    validate_biomarker_coverage,
    validate_date_ranges,
    validate_coordinates,
    run_all_validations,
)


class TestValidateRequiredColumns:

    def test_all_present(self, sample_analysis_df):
        missing = validate_required_columns(sample_analysis_df)
        assert missing == []

    def test_missing_patient_id(self, sample_analysis_df):
        df = sample_analysis_df.drop(columns=["patient_id"])
        missing = validate_required_columns(df)
        assert "patient_id" in missing

    def test_missing_biomarker_column(self, sample_analysis_df):
        df = sample_analysis_df.drop(columns=["creatinine"])
        missing = validate_required_columns(df)
        assert "creatinine" in missing

    def test_extra_columns_checked(self, sample_analysis_df):
        missing = validate_required_columns(
            sample_analysis_df, extra_columns=["nonexistent_col"]
        )
        assert "nonexistent_col" in missing


class TestValidateBiomarkerCoverage:

    def test_returns_dataframe(self, sample_analysis_df):
        result = validate_biomarker_coverage(sample_analysis_df)
        assert isinstance(result, pd.DataFrame)
        assert "biomarker" in result.columns
        assert "n_observations" in result.columns
        assert "meets_threshold" in result.columns

    def test_all_biomarkers_in_dataset(self, sample_analysis_df):
        result = validate_biomarker_coverage(sample_analysis_df)
        assert result["in_dataset"].all()

    def test_sparse_biomarker_flagged(self, sample_analysis_df):
        # Make cd4_count very sparse (< MIN_SAMPLE_SIZE)
        df = sample_analysis_df.copy()
        df["cd4_count"] = np.nan
        df.loc[:5, "cd4_count"] = 500  # Only 6 non-null

        result = validate_biomarker_coverage(df)
        cd4_row = result[result["biomarker"] == "cd4_count"].iloc[0]
        assert cd4_row["n_observations"] == 6
        assert cd4_row["meets_threshold"] == False  # noqa: E712 (np.bool_)

    def test_missing_column_flagged(self):
        df = pd.DataFrame({"creatinine": [1.0, 2.0]})
        result = validate_biomarker_coverage(df)
        n_not_in_dataset = (~result["in_dataset"]).sum()
        assert n_not_in_dataset > 0


class TestValidateDateRanges:

    def test_valid_dates(self, sample_analysis_df):
        result = validate_date_ranges(sample_analysis_df)
        assert result["all_valid"]

    def test_out_of_range_dates(self, sample_analysis_df):
        df = sample_analysis_df.copy()
        df.loc[0, "visit_date"] = "1990-01-01"  # Before 2003
        result = validate_date_ranges(df)
        assert not result["all_valid"]
        assert len(result["out_of_range"]) > 0

    def test_no_visit_date_column(self):
        df = pd.DataFrame({"patient_id": ["P1"]})
        result = validate_date_ranges(df)
        assert result["studies_checked"] == 0


class TestValidateCoordinates:

    def test_valid_coordinates(self, sample_analysis_df):
        result = validate_coordinates(sample_analysis_df)
        assert result["all_valid"]
        assert result["n_out_of_bounds"] == 0

    def test_out_of_bounds_coordinates(self, sample_analysis_df):
        df = sample_analysis_df.copy()
        df.loc[0, "latitude"] = -20.0  # Way north of JHB
        df.loc[1, "longitude"] = 35.0  # Way east
        result = validate_coordinates(df)
        assert not result["all_valid"]
        assert result["n_out_of_bounds"] == 2

    def test_no_coordinate_columns(self):
        df = pd.DataFrame({"patient_id": ["P1"]})
        result = validate_coordinates(df)
        assert result["n_checked"] == 0


class TestRunAllValidations:

    def test_passes_for_good_data(self, sample_analysis_df):
        report = run_all_validations(sample_analysis_df)
        assert report["overall_pass"]

    def test_fails_for_missing_columns(self, sample_analysis_df):
        df = sample_analysis_df.drop(columns=["patient_id", "visit_date"])
        report = run_all_validations(df)
        assert not report["overall_pass"]
        assert "patient_id" in report["missing_columns"]

    def test_report_structure(self, sample_analysis_df):
        report = run_all_validations(sample_analysis_df)
        assert "missing_columns" in report
        assert "biomarker_coverage" in report
        assert "date_ranges" in report
        assert "coordinates" in report
        assert "overall_pass" in report
