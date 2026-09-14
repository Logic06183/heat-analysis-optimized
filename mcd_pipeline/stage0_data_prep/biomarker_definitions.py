"""
Biomarker Definitions — 27 Biomarkers Across 8 Organ Systems
=============================================================

Maps every MCD-specified biomarker to its column name in the
JOHANNESBURG_CLINICAL_ANALYSIS_READY dataset, organ system, clinical
significance threshold, and unit. Biomarkers not present in the dataset
are flagged as unavailable or derivable.

MCD reference: "...27+ biomarkers across 8 systems (renal: creatinine/eGFR;
metabolic: glucose/HbA1c; cardiovascular: BP/HR; inflammatory: CRP/ESR;
immunological: CD4/CD8; hepatic: ALT/AST; lipid: cholesterol panel;
body composition: BMI/DXA)"
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BiomarkerSpec:
    """Specification for a single biomarker."""

    name: str                          # Human-readable name
    column: Optional[str]              # Column name in ANALYSIS_READY CSV (None if unavailable)
    system: str                        # Organ system (one of BIOMARKER_SYSTEMS)
    unit: str                          # Measurement unit
    clinical_threshold: Optional[float]  # Clinically meaningful difference
    log_transform: bool = False        # Whether to log-transform for analysis
    derivable: bool = False            # True if computable from other columns
    available: bool = True             # False if not in any study dataset
    notes: str = ""


# The 8 organ systems defined in the MCD
BIOMARKER_SYSTEMS = [
    "renal",
    "metabolic",
    "cardiovascular",
    "inflammatory",
    "immunological",
    "hepatic",
    "lipid",
    "body_composition",
]

# Complete biomarker registry
# Clinical significance thresholds adapted from:
#   - climate_health_evaluation.py (archived reference)
#   - Standard clinical laboratory reference ranges
BIOMARKERS = {
    # --- Renal ---
    "creatinine": BiomarkerSpec(
        name="Serum Creatinine",
        column="creatinine",
        system="renal",
        unit="µmol/L",
        clinical_threshold=26.5,  # ~0.3 mg/dL (26.5 µmol/L)
        notes="Primary renal biomarker. All 7 contributing studies store values in "
              "µmol/L (typical adult range 45-90). To convert to mg/dL: divide by 88.4",
    ),
    "creatinine_clearance": BiomarkerSpec(
        name="Creatinine Clearance",
        column="creatinine_clearance",
        system="renal",
        unit="mL/min",
        clinical_threshold=15.0,
        notes="Estimated from creatinine",
    ),
    "egfr": BiomarkerSpec(
        name="Estimated GFR",
        column=None,
        system="renal",
        unit="mL/min/1.73m²",
        clinical_threshold=15.0,
        derivable=True,
        notes="Derive from creatinine using CKD-EPI equation (Levey et al. 2009). "
              "CKD-EPI requires creatinine in mg/dL; stored values are in µmol/L — "
              "convert first: mg/dL = µmol/L / 88.4",
    ),
    "albumin": BiomarkerSpec(
        name="Serum Albumin",
        column="albumin",
        system="renal",
        unit="g/L",
        clinical_threshold=5.0,
    ),

    # --- Metabolic ---
    "fasting_glucose": BiomarkerSpec(
        name="Fasting Glucose",
        column="fasting_glucose",
        system="metabolic",
        unit="mg/dL",
        clinical_threshold=18.0,  # ~1 mmol/L
        notes="Units standardised to mg/dL in ANALYSIS_READY",
    ),
    "hba1c": BiomarkerSpec(
        name="HbA1c",
        column="hba1c",
        system="metabolic",
        unit="%",
        clinical_threshold=0.5,
        notes="Glycated haemoglobin; sparse coverage",
    ),
    "fasting_insulin": BiomarkerSpec(
        name="Fasting Insulin",
        column="fasting_insulin",
        system="metabolic",
        unit="µIU/mL",
        clinical_threshold=5.0,
        log_transform=True,
    ),

    # --- Cardiovascular ---
    "systolic_bp": BiomarkerSpec(
        name="Systolic Blood Pressure",
        column="systolic_bp",
        system="cardiovascular",
        unit="mmHg",
        clinical_threshold=5.0,
    ),
    "diastolic_bp": BiomarkerSpec(
        name="Diastolic Blood Pressure",
        column="diastolic_bp",
        system="cardiovascular",
        unit="mmHg",
        clinical_threshold=3.0,
    ),
    "heart_rate": BiomarkerSpec(
        name="Heart Rate",
        column="heart_rate",
        system="cardiovascular",
        unit="bpm",
        clinical_threshold=5.0,
    ),

    # --- Inflammatory ---
    "hs_crp": BiomarkerSpec(
        name="High-Sensitivity CRP",
        column="hs_crp",
        system="inflammatory",
        unit="mg/L",
        clinical_threshold=1.0,
        log_transform=True,
        notes="C-reactive protein",
    ),
    "esr": BiomarkerSpec(
        name="Erythrocyte Sedimentation Rate",
        column=None,
        system="inflammatory",
        unit="mm/hr",
        clinical_threshold=10.0,
        available=False,
        notes="NOT available in any study dataset",
    ),

    # --- Immunological ---
    "cd4_count": BiomarkerSpec(
        name="CD4 Count",
        column="cd4_count",
        system="immunological",
        unit="cells/µL",
        clinical_threshold=50.0,
        notes="Primary immunological marker; n=4,555 records",
    ),
    "viral_load": BiomarkerSpec(
        name="HIV Viral Load",
        column="viral_load",
        system="immunological",
        unit="copies/mL",
        clinical_threshold=1000.0,
        log_transform=True,
        notes="n=2,520 records; use log10_viral_load for analysis. "
              "Required for SA6 case-crossover. ACTG_016 and WRHI_003 "
              "values set to NA in ANALYSIS_READY (placeholder codes removed).",
    ),
    "cd8_count": BiomarkerSpec(
        name="CD8 Count",
        column=None,
        system="immunological",
        unit="cells/µL",
        clinical_threshold=50.0,
        available=False,
        notes="NOT available in current harmonised dataset",
    ),

    # --- Hepatic ---
    "alt": BiomarkerSpec(
        name="Alanine Aminotransferase",
        column="alt",
        system="hepatic",
        unit="U/L",
        clinical_threshold=10.0,
        log_transform=True,
    ),
    "ast": BiomarkerSpec(
        name="Aspartate Aminotransferase",
        column="ast",
        system="hepatic",
        unit="U/L",
        clinical_threshold=10.0,
        log_transform=True,
    ),

    # --- Lipid ---
    "total_cholesterol": BiomarkerSpec(
        name="Total Cholesterol",
        column="total_cholesterol",
        system="lipid",
        unit="mg/dL",
        clinical_threshold=20.0,
    ),
    "hdl_cholesterol": BiomarkerSpec(
        name="HDL Cholesterol",
        column="hdl_cholesterol",
        system="lipid",
        unit="mg/dL",
        clinical_threshold=10.0,
    ),
    "ldl_cholesterol": BiomarkerSpec(
        name="LDL Cholesterol",
        column="ldl_cholesterol",
        system="lipid",
        unit="mg/dL",
        clinical_threshold=15.0,
    ),
    "triglycerides": BiomarkerSpec(
        name="Triglycerides",
        column="triglycerides",
        system="lipid",
        unit="mg/dL",
        clinical_threshold=25.0,
        log_transform=True,
    ),

    # --- Body Composition ---
    "bmi": BiomarkerSpec(
        name="Body Mass Index",
        column="bmi",
        system="body_composition",
        unit="kg/m²",
        clinical_threshold=1.0,
        notes="n=6,611 records (highest coverage)",
    ),
    "waist_circumference": BiomarkerSpec(
        name="Waist Circumference",
        column="waist_circumference",
        system="body_composition",
        unit="cm",
        clinical_threshold=5.0,
    ),
    "hip_circumference": BiomarkerSpec(
        name="Hip Circumference",
        column="hip_circumference",
        system="body_composition",
        unit="cm",
        clinical_threshold=5.0,
    ),
    "waist_hip_ratio": BiomarkerSpec(
        name="Waist-Hip Ratio",
        column="waist_hip_ratio",
        system="body_composition",
        unit="ratio",
        clinical_threshold=0.05,
    ),

    # --- Body Composition: DXA measures ---
    "body_fat_percent": BiomarkerSpec(
        name="Body Fat Percentage",
        column="body_fat_percent",
        system="body_composition",
        unit="%",
        clinical_threshold=3.0,
        notes="DXA-derived; sparse coverage (WRHI_001, DPHRU_053)",
    ),
    "total_fat_mass": BiomarkerSpec(
        name="Total Fat Mass",
        column="total_fat_mass",
        system="body_composition",
        unit="kg",
        clinical_threshold=2.0,
        notes="DXA-derived; sparse coverage",
    ),
    "fat_mass_index": BiomarkerSpec(
        name="Fat Mass Index",
        column="fat_mass_index",
        system="body_composition",
        unit="kg/m²",
        clinical_threshold=1.0,
        notes="DXA-derived: total_fat_mass / height_m²",
    ),

    # --- Hematological (grouped with body composition per MCD's 8-system framework) ---
    "hemoglobin": BiomarkerSpec(
        name="Hemoglobin",
        column="hemoglobin",
        system="body_composition",
        unit="g/dL",
        clinical_threshold=1.0,
        notes="n=3,036 records",
    ),
    "hematocrit": BiomarkerSpec(
        name="Hematocrit",
        column="hematocrit",
        system="body_composition",
        unit="%",
        clinical_threshold=3.0,
    ),
}


# Minimum sample size to include a biomarker in analysis
MIN_SAMPLE_SIZE = 100


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_available_biomarkers() -> dict[str, BiomarkerSpec]:
    """Return only biomarkers that have a column in the dataset."""
    return {k: v for k, v in BIOMARKERS.items() if v.available and v.column is not None}


def get_biomarkers_by_system(system: str) -> dict[str, BiomarkerSpec]:
    """Return biomarkers belonging to a specific organ system."""
    return {k: v for k, v in BIOMARKERS.items() if v.system == system}


def get_derivable_biomarkers() -> dict[str, BiomarkerSpec]:
    """Return biomarkers that must be computed from other columns."""
    return {k: v for k, v in BIOMARKERS.items() if v.derivable}


def get_logtransformed_biomarkers() -> dict[str, BiomarkerSpec]:
    """Return available biomarkers that should be log-transformed for analysis."""
    return {k: v for k, v in get_available_biomarkers().items() if v.log_transform}


def validate_biomarker_columns(
    columns: list[str] | set[str],
) -> tuple[dict[str, BiomarkerSpec], list[str]]:
    """Check which available biomarkers actually exist in a set of columns.

    Parameters
    ----------
    columns : list or set of str
        Column names from a DataFrame (e.g., ``df.columns.tolist()``).

    Returns
    -------
    tuple of (found, missing)
        found : dict mapping biomarker key -> BiomarkerSpec for those
            whose ``column`` is present in the provided columns.
        missing : list of column names that are registered as available
            but not found in the provided columns.
    """
    columns = set(columns)
    found = {}
    missing = []

    for key, spec in get_available_biomarkers().items():
        if spec.column in columns:
            found[key] = spec
        else:
            missing.append(spec.column)

    return found, missing
