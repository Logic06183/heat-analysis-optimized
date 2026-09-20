"""Export an ERA5-Land-flavoured dlnm_r_input.csv with matching column names
so the existing R DLNM script runs without modification.

The R script reads columns named ``dlnm_lag{0,1,3,7,14,21}_c`` and
``era5_temp_lag30d_c``. We replace these with the ERA5-Land equivalents while
keeping the same column names so the R code does not need to change."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers
from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUT_CSV = config.OUTPUT_ROOT / "dlnm_r_input.csv"

# ERA5-Land columns to inject as if they were the original dlnm_lag*_c / era5_temp_lag30d_c
ERA5_LAND_LAG_MAP = {
    "era5land_temp_lag0d_c":  "dlnm_lag0_c",
    "era5land_temp_lag1d_c":  "dlnm_lag1_c",
    "era5land_temp_lag3d_c":  "dlnm_lag3_c",
    "era5land_temp_lag7d_c":  "dlnm_lag7_c",
    "era5land_temp_lag14d_c": "dlnm_lag14_c",
    "era5land_temp_lag21d_c": "dlnm_lag21_c",
    "era5land_temp_lag30d_c": "era5_temp_lag30d_c",
}


def main() -> None:
    logger.info("Primary exposure source: %s", config.PRIMARY_EXPOSURE_SOURCE)
    logger.info("Writing R DLNM input to %s", OUT_CSV)

    df = build_analysis_dataset()
    biomarkers = get_available_biomarkers()
    bio_cols = [b.column for b in biomarkers.values() if b.column and b.column in df.columns]
    df = engineer_features(df, bio_cols)

    # Merge ERA5-Land lag columns from the linkage file (they're already in df after
    # engineer_features with PRIMARY_EXPOSURE_SOURCE=era5_land, but for safety re-read).
    land = pd.read_csv(config.ERA5_LAND_CSV, low_memory=False)
    land["visit_date"] = land["visit_date"].astype(str)
    df["visit_date"] = df["visit_date"].astype(str)
    keep = ["study_source", "patient_id", "visit_date"] + list(ERA5_LAND_LAG_MAP)
    land = land[keep].drop_duplicates(subset=["study_source", "patient_id", "visit_date"])
    # Drop the original ERA5 31 km lag columns first — they would otherwise collide
    # with the renamed ERA5-Land columns after the merge+rename below.
    df = df.drop(
        columns=list(ERA5_LAND_LAG_MAP.values())
        + [c for c in ERA5_LAND_LAG_MAP if c in df.columns],
        errors="ignore",
    )
    merged = df.merge(land, on=["study_source", "patient_id", "visit_date"], how="left")
    # Rename ERA5-Land columns to the R script's expected names.
    merged = merged.rename(columns=ERA5_LAND_LAG_MAP)

    # Columns to keep in the R input — covariates + lag columns + biomarkers
    cov_cols = ["patient_id", "visit_date", "year", "month", "age_years", "sex", "hiv_status"]
    cov_cols = [c for c in cov_cols if c in merged.columns]
    lag_cols = list(ERA5_LAND_LAG_MAP.values())
    # 14 adequate biomarkers under primary ERA5; bmi dropped under ERA5-Land.
    adequate = [
        "body_fat_percent", "hematocrit", "hemoglobin", "viral_load",
        "waist_hip_ratio", "creatinine", "cd4_count", "diastolic_bp",
        "ldl_cholesterol", "systolic_bp", "heart_rate", "albumin",
        "hip_circumference",  # bmi excluded — not retained under ERA5-Land
    ]
    bio_cols_out = [c for c in adequate if c in merged.columns]

    out = merged[cov_cols + lag_cols + bio_cols_out].copy()
    out.to_csv(OUT_CSV, index=False)
    logger.info(
        "Wrote %s — %d rows, %d cols (%d biomarkers, %d lag cols)",
        OUT_CSV, len(out), len(out.columns), len(bio_cols_out), len(lag_cols),
    )


if __name__ == "__main__":
    main()
