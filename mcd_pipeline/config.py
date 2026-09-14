"""
Central Configuration — MCD Pipeline Parameters
================================================

All tuneable parameters from the Manuscript Concept Document in one place.
Import this module rather than hard-coding values in analysis scripts.

MCD reference: "Heat, Vulnerability, and the Body: Multi-System Biomarker
Signatures of Temperature Exposure in Johannesburg Revealed Through
Explainable Machine Learning" — Parker et al. (2026)
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FINAL_DATASETS = PROJECT_ROOT / "FINAL_DATASETS"
ANALYSIS_READY_CSV = FINAL_DATASETS / "JOHANNESBURG_CLINICAL_ANALYSIS_READY.csv"
CLINICAL_FULL_CSV = FINAL_DATASETS / "JOHANNESBURG_CLINICAL_FULL.csv"
GCRO_CSV = FINAL_DATASETS / "GCRO_SOCIOECONOMIC_CLIMATE_ENHANCED_LABELED.csv"

INDIVIDUAL_STUDIES_DIR = FINAL_DATASETS / "INDIVIDUAL_STUDIES"
CLIMATE_LINKED_DIR = (
    FINAL_DATASETS / "CLIMATE_HEALTH_LINKAGE" / "JHB_CLIMATE_LINKAGE"
    / "linked_datasets"
)
SES_MATCHED_DIR = FINAL_DATASETS / "SOCIOECONOMIC_LINKAGE" / "matched_datasets"
METADATA_DIR = FINAL_DATASETS / "metadata"

# Tidy (long-format) datasets — canonical pipeline inputs
TIDY_DIR = FINAL_DATASETS / "TIDY_DATASETS"
TIDY_BIOMARKERS_CSV = TIDY_DIR / "TIDY_clinical_biomarkers.csv"
TIDY_CLIMATE_CSV = TIDY_DIR / "TIDY_climate_variables.csv"
TIDY_DEMOGRAPHICS_CSV = TIDY_DIR / "TIDY_demographics.csv"
TIDY_SOCIOECONOMIC_CSV = TIDY_DIR / "TIDY_socioeconomic.csv"
TIDY_HVI_CSV = TIDY_DIR / "TIDY_hvi.csv"

CLIMATE_ZARR_BASE = Path("/home/cparker/selected_data_all/data/RP2_subsets/JHB")

OUTPUT_ROOT = PROJECT_ROOT / "mcd_outputs"
PUBLICATION_ROOT = PROJECT_ROOT / "mcd_publication"

# ---------------------------------------------------------------------------
# Temperature Lag Windows (MCD Section: Analysis Plan)
# ---------------------------------------------------------------------------

# Individual day lags used as XGBoost features
LAG_DAYS = [0, 1, 3, 7, 14, 21, 30]

# Column names in climate-linked datasets for these lags.
# DLNM-extracted lags go up to 21 days; lag 30 uses the ERA5 rolling mean.
_DLNM_LAG_COLUMNS = [f"dlnm_lag{d}_c" for d in LAG_DAYS if d <= 21]
LAG_FEATURE_COLUMNS = _DLNM_LAG_COLUMNS + ["era5_temp_lag30d_c"]

# Rolling mean lag columns (supplementary features)
ROLLING_LAG_COLUMNS = [
    "era5_temp_lag1d_c",
    "era5_temp_lag3d_c",
    "era5_temp_lag7d_c",
    "era5_temp_lag14d_c",
    "era5_temp_lag30d_c",
]

# Diurnal temperature range (additional climate feature per MCD)
DTR_COLUMN = "era5_temp_range_c"

# ---------------------------------------------------------------------------
# Covariates (MCD Section: Analysis Plan)
# ---------------------------------------------------------------------------

DEMOGRAPHIC_COVARIATES = ["age_years", "sex"]
CLINICAL_COVARIATES = ["hiv_status"]
SOCIOECONOMIC_COVARIATES = [
    "gcro_dwelling_type",
    "gcro_income_bracket",
    "gcro_education_level",
    "gcro_employment_status",
]

# Temporal confounders
# Month fixed effects: one-hot encoded month (1–12)
# Fourier seasonality: sin(2πm/12), cos(2πm/12) for m = month
# Calendar year: integer year
# Study ID: categorical study identifier
TEMPORAL_CONFOUNDERS = ["month", "year", "study_source"]
# Fourier terms are computed in feature_engineering.py

# ---------------------------------------------------------------------------
# XGBoost Model Parameters (MCD: Primary Analysis)
# ---------------------------------------------------------------------------

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

# Cross-validation: GroupKFold by patient_id (prevents patient leakage)
CV_FOLDS = 5
PATIENT_ID_COLUMN = "patient_id"

# ---------------------------------------------------------------------------
# Bootstrap Parameters (MCD: 50 model replicates)
# ---------------------------------------------------------------------------

N_BOOTSTRAP_REPLICATES = 50  # Model replicates for SHAP stability (MCD spec)
N_BOOTSTRAP_FAST = 10        # Exploratory run (use --fast flag)
BOOTSTRAP_RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# SHAP Configuration (MCD: TreeSHAP with interventional conditional expectation)
# ---------------------------------------------------------------------------

SHAP_FEATURE_PERTURBATION = "interventional"  # NOT "tree_path_dependent"
# This marginalises over the empirical feature distribution rather than
# conditioning on correlated neighbours — critical for correlated lag features.
# Reference: Janzing, Minorics & Blöbaum (2020) "Feature relevance
# quantification in explainable AI"

# ---------------------------------------------------------------------------
# Stage 2: Interaction Detection (MCD Section)
# ---------------------------------------------------------------------------

N_PERMUTATIONS = 500  # Per biomarker model
FDR_THRESHOLD = 0.10  # Benjamini-Hochberg q-value
TOP_INTERACTIONS = 50  # Rank top N interactions per biomarker

# ---------------------------------------------------------------------------
# Stage 3: Vulnerability Clustering (MCD Section)
# ---------------------------------------------------------------------------

PCA_VARIANCE_THRESHOLD = 0.85  # Retain components explaining 85% variance
KMEANS_K_RANGE = range(3, 9)  # k = 3, 4, 5, 6, 7, 8
CLUSTER_STABILITY_ITERATIONS = 500
CLUSTER_STABILITY_THRESHOLD = 0.80

# ---------------------------------------------------------------------------
# Multiple Testing Correction (MCD Section)
# ---------------------------------------------------------------------------

# Primary: Bonferroni across 8 biomarker systems
N_BIOMARKER_SYSTEMS = 8
BONFERRONI_ALPHA = 0.05 / N_BIOMARKER_SYSTEMS  # = 0.00625

# Secondary: FDR within significant systems and for interactions
FDR_Q_WITHIN_SYSTEMS = 0.10

# ---------------------------------------------------------------------------
# Sensitivity Analysis Flags
# ---------------------------------------------------------------------------

SENSITIVITY_ANALYSES = {
    "sa1_dlnm": True,           # DLNM validation for top 5 biomarkers
    "sa2_modis_vs_era5": True,  # MODIS LST vs ERA5
    "sa3_pollution": True,      # Pollution-adjusted subset (2007–2021)
    "sa4_covid": True,          # COVID-19 period exclusion (2020–2021)
    "sa5_imputation": False,    # BLOCKED: imputation work in progress
    "sa6_case_crossover": True, # Case-crossover for CD4/viral load
    "sa7_hiv_stratified": True, # HIV cohorts vs general population
    "sa8_heatwave": True,       # 95th vs 90th percentile threshold
}

# SA3: pollution-adjusted subset year range
POLLUTION_DATA_START_YEAR = 2007
POLLUTION_DATA_END_YEAR = 2021

# SA4: COVID exclusion years
COVID_EXCLUSION_YEARS = [2020, 2021]

# SA8: heatwave percentile thresholds to compare
HEATWAVE_PERCENTILES = [90, 95]

# ---------------------------------------------------------------------------
# Random Seeds for Reproducibility
# ---------------------------------------------------------------------------

MASTER_SEED = 42
