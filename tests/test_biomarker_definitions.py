"""Tests for mcd_pipeline.stage0_data_prep.biomarker_definitions."""

import pytest

from mcd_pipeline.stage0_data_prep.biomarker_definitions import (
    BIOMARKERS,
    BIOMARKER_SYSTEMS,
    BiomarkerSpec,
    MIN_SAMPLE_SIZE,
    get_available_biomarkers,
    get_biomarkers_by_system,
    get_derivable_biomarkers,
    get_logtransformed_biomarkers,
    validate_biomarker_columns,
)


class TestBiomarkerRegistry:
    """Tests for the BIOMARKERS dict and BIOMARKER_SYSTEMS list."""

    def test_eight_organ_systems(self):
        assert len(BIOMARKER_SYSTEMS) == 8

    def test_system_names(self):
        expected = {
            "renal", "metabolic", "cardiovascular", "inflammatory",
            "immunological", "hepatic", "lipid", "body_composition",
        }
        assert set(BIOMARKER_SYSTEMS) == expected

    def test_all_biomarkers_have_valid_system(self):
        for key, spec in BIOMARKERS.items():
            assert spec.system in BIOMARKER_SYSTEMS, (
                f"Biomarker '{key}' has invalid system '{spec.system}'"
            )

    def test_all_biomarkers_have_positive_threshold(self):
        for key, spec in BIOMARKERS.items():
            if spec.clinical_threshold is not None:
                assert spec.clinical_threshold > 0, (
                    f"Biomarker '{key}' has non-positive threshold"
                )

    def test_unavailable_biomarkers_have_no_column(self):
        for key, spec in BIOMARKERS.items():
            if not spec.available:
                assert spec.column is None, (
                    f"Unavailable biomarker '{key}' should have column=None"
                )

    def test_frozen_dataclass(self):
        spec = BIOMARKERS["creatinine"]
        with pytest.raises(AttributeError):
            spec.name = "something else"

    def test_known_unavailable_biomarkers(self):
        assert not BIOMARKERS["esr"].available
        assert not BIOMARKERS["cd8_count"].available

    def test_known_derivable_biomarkers(self):
        assert BIOMARKERS["egfr"].derivable
        assert BIOMARKERS["egfr"].column is None

    def test_viral_load_present(self):
        assert "viral_load" in BIOMARKERS
        assert BIOMARKERS["viral_load"].column == "viral_load"
        assert BIOMARKERS["viral_load"].log_transform is True

    def test_dxa_biomarkers_present(self):
        for key in ["body_fat_percent", "total_fat_mass", "fat_mass_index"]:
            assert key in BIOMARKERS, f"DXA biomarker '{key}' missing"
            assert BIOMARKERS[key].system == "body_composition"


class TestGetAvailableBiomarkers:
    """Tests for get_available_biomarkers()."""

    def test_returns_only_available_with_columns(self):
        available = get_available_biomarkers()
        for key, spec in available.items():
            assert spec.available is True
            assert spec.column is not None

    def test_excludes_unavailable(self):
        available = get_available_biomarkers()
        assert "esr" not in available
        assert "cd8_count" not in available

    def test_excludes_derivable_without_column(self):
        available = get_available_biomarkers()
        assert "egfr" not in available

    def test_count_reasonable(self):
        available = get_available_biomarkers()
        assert 20 <= len(available) <= 35


class TestGetBiomarkersBySystem:

    def test_renal_system(self):
        renal = get_biomarkers_by_system("renal")
        assert "creatinine" in renal
        assert "albumin" in renal

    def test_empty_for_invalid_system(self):
        result = get_biomarkers_by_system("nonexistent")
        assert len(result) == 0

    def test_each_system_has_biomarkers(self):
        for system in BIOMARKER_SYSTEMS:
            bms = get_biomarkers_by_system(system)
            assert len(bms) > 0, f"System '{system}' has no biomarkers"


class TestGetLogtransformedBiomarkers:

    def test_returns_only_logtransform(self):
        lt = get_logtransformed_biomarkers()
        for key, spec in lt.items():
            assert spec.log_transform is True

    def test_includes_known_logtransform(self):
        lt = get_logtransformed_biomarkers()
        assert "viral_load" in lt
        assert "hs_crp" in lt
        assert "triglycerides" in lt

    def test_excludes_non_logtransform(self):
        lt = get_logtransformed_biomarkers()
        assert "bmi" not in lt
        assert "systolic_bp" not in lt


class TestValidateBiomarkerColumns:

    def test_all_columns_present(self, sample_analysis_df):
        found, missing = validate_biomarker_columns(sample_analysis_df.columns)
        assert len(missing) == 0
        assert len(found) == len(get_available_biomarkers())

    def test_missing_columns_detected(self):
        columns = ["patient_id", "creatinine", "albumin"]
        found, missing = validate_biomarker_columns(columns)
        assert "creatinine" in found
        assert "albumin" in found
        assert len(missing) > 0  # many biomarkers not in this small set

    def test_empty_columns(self):
        found, missing = validate_biomarker_columns([])
        assert len(found) == 0
        assert len(missing) == len(get_available_biomarkers())
