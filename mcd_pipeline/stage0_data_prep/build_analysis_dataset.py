"""
Build Analysis Dataset — From TIDY Datasets to ML-Ready Wide Format
====================================================================

Pivots the TIDY long-format datasets into a single wide-format DataFrame
ready for the XGBoost-SHAP pipeline.

Inputs (all from FINAL_DATASETS/TIDY_DATASETS/):
- TIDY_clinical_biomarkers.csv  (biomarker, value per patient-visit)
- TIDY_climate_variables.csv    (climate_variable, value per patient-visit)
- TIDY_demographics.csv         (demographic, value per patient)
- TIDY_socioeconomic.csv        (ses_variable, value per patient)

Output:
- Wide DataFrame with one row per (patient_id, visit_date), columns for
  each biomarker, climate variable, demographic, and SES covariate.

MCD reference: "Data to be used" section
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from mcd_pipeline import config

logger = logging.getLogger(__name__)


def _load_and_pivot_biomarkers() -> pd.DataFrame:
    """Load TIDY biomarkers and pivot to wide format."""
    logger.info("  Loading TIDY_clinical_biomarkers.csv...")
    bio = pd.read_csv(config.TIDY_BIOMARKERS_CSV, low_memory=False)

    # Drop non-numeric biomarkers (e.g. viral_load_category)
    bio["value"] = pd.to_numeric(bio["value"], errors="coerce")
    bio = bio.dropna(subset=["value"])

    # Apply physiological plausibility caps (set implausible values to NaN)
    n_capped = 0
    for biomarker_name, (lo, hi) in config.BIOMARKER_CAPS.items():
        mask = bio["biomarker"] == biomarker_name
        out_of_range = mask & ((bio["value"] < lo) | (bio["value"] > hi))
        n_out = out_of_range.sum()
        if n_out > 0:
            logger.warning(
                "  Capping %d implausible %s values outside [%g, %g]",
                n_out, biomarker_name, lo, hi,
            )
            bio.loc[out_of_range, "value"] = np.nan
            n_capped += n_out
    if n_capped:
        bio = bio.dropna(subset=["value"])
        logger.info("  Removed %d physiologically implausible values", n_capped)

    # Pivot: one column per biomarker
    bio_wide = bio.pivot_table(
        index=["study_source", "patient_id", "visit_date", "latitude", "longitude"],
        columns="biomarker",
        values="value",
        aggfunc="first",
    ).reset_index()

    bio_wide.columns.name = None
    logger.info(
        "  Biomarkers: %d rows × %d biomarker columns",
        len(bio_wide),
        len(bio_wide.columns) - 5,
    )
    return bio_wide


def _load_and_pivot_climate() -> pd.DataFrame:
    """Load TIDY climate and pivot to wide format."""
    logger.info("  Loading TIDY_climate_variables.csv...")
    clim = pd.read_csv(config.TIDY_CLIMATE_CSV, low_memory=False)

    clim["value"] = pd.to_numeric(clim["value"], errors="coerce")
    clim = clim.dropna(subset=["value"])

    clim_wide = clim.pivot_table(
        index=["study_source", "patient_id", "visit_date"],
        columns="climate_variable",
        values="value",
        aggfunc="first",
    ).reset_index()

    clim_wide.columns.name = None
    logger.info(
        "  Climate: %d rows × %d climate columns",
        len(clim_wide),
        len(clim_wide.columns) - 3,
    )
    return clim_wide


def _load_and_pivot_demographics() -> pd.DataFrame:
    """Load TIDY demographics and pivot to wide format."""
    logger.info("  Loading TIDY_demographics.csv...")
    dem = pd.read_csv(config.TIDY_DEMOGRAPHICS_CSV)

    dem_wide = dem.pivot_table(
        index=["study_source", "patient_id"],
        columns="demographic",
        values="value",
        aggfunc="first",
    ).reset_index()

    dem_wide.columns.name = None
    logger.info("  Demographics: %d patients", len(dem_wide))
    return dem_wide


def _load_and_pivot_ses() -> pd.DataFrame:
    """Load TIDY socioeconomic and pivot to wide format."""
    logger.info("  Loading TIDY_socioeconomic.csv...")
    ses = pd.read_csv(config.TIDY_SOCIOECONOMIC_CSV)

    ses_wide = ses.pivot_table(
        index=["study_source", "patient_id"],
        columns="ses_variable",
        values="value",
        aggfunc="first",
    ).reset_index()

    ses_wide.columns.name = None

    # Rename to match config.SOCIOECONOMIC_COVARIATES (gcro_ prefix)
    rename_map = {
        "dwelling_type": "gcro_dwelling_type",
        "education_level": "gcro_education_level",
        "employment_status": "gcro_employment_status",
        "income_bracket": "gcro_income_bracket",
    }
    ses_wide = ses_wide.rename(columns=rename_map)
    logger.info("  SES: %d patients", len(ses_wide))
    return ses_wide


def build_analysis_dataset(
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build wide-format analysis dataset from TIDY inputs.

    Pivots all four TIDY datasets and joins on (study_source, patient_id)
    and (study_source, patient_id, visit_date).

    Parameters
    ----------
    output_path : Path, optional
        If provided, save the combined dataset to this path.

    Returns
    -------
    pd.DataFrame
        Wide-format dataset ready for feature engineering.
    """
    logger.info("=== Stage 0: Building analysis dataset from TIDY inputs ===")

    # Step 1: Pivot each TIDY dataset to wide format
    bio_wide = _load_and_pivot_biomarkers()
    clim_wide = _load_and_pivot_climate()
    dem_wide = _load_and_pivot_demographics()
    ses_wide = _load_and_pivot_ses()

    # Step 2: Merge biomarkers + climate on (study_source, patient_id, visit_date)
    logger.info("Step 2: Merging biomarkers + climate...")
    merge_keys = ["study_source", "patient_id", "visit_date"]
    df = bio_wide.merge(clim_wide, on=merge_keys, how="inner")
    logger.info("  After bio+climate merge: %d rows", len(df))

    # Step 3: Merge demographics on (study_source, patient_id)
    logger.info("Step 3: Merging demographics...")
    patient_keys = ["study_source", "patient_id"]
    df = df.merge(dem_wide, on=patient_keys, how="left")

    # Step 4: Merge SES on (study_source, patient_id)
    logger.info("Step 4: Merging socioeconomic data...")
    df = df.merge(ses_wide, on=patient_keys, how="left")

    n_ses = df[config.SOCIOECONOMIC_COVARIATES[0]].notna().sum() if config.SOCIOECONOMIC_COVARIATES[0] in df.columns else 0
    logger.info(
        "  SES coverage: %d/%d rows (%.1f%%)",
        n_ses, len(df), 100 * n_ses / max(len(df), 1),
    )

    # Step 5: Temporal columns
    logger.info("Step 5: Deriving temporal columns...")
    df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")
    df["year"] = df["visit_date"].dt.year
    df["month"] = df["visit_date"].dt.month

    # Step 6: Quality filters
    before = len(df)
    df = df.dropna(subset=["visit_date"]).reset_index(drop=True)
    n_dropped = before - len(df)
    if n_dropped:
        logger.info("  Dropped %d rows without visit_date", n_dropped)

    # Deduplicate (keep first per patient-date-study)
    before = len(df)
    df = df.drop_duplicates(
        subset=["patient_id", "visit_date", "study_source"], keep="first"
    ).reset_index(drop=True)
    n_deduped = before - len(df)
    if n_deduped:
        logger.info("  Deduplicated %d rows", n_deduped)

    # Convert age_years to numeric
    if "age_years" in df.columns:
        df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")

    logger.info(
        "Final dataset: %d rows × %d cols, %d patients, %d studies",
        len(df),
        len(df.columns),
        df["patient_id"].nunique(),
        df["study_source"].nunique(),
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved to %s", output_path)

    return df
