"""
Data Validation — Pre-Analysis Quality Assurance
=================================================

Validates the analysis dataset before any modelling begins:
- Required columns present
- Biomarker sample sizes meet minimum thresholds
- Date ranges within expected bounds (2003-2021)
- Coordinate validation (Johannesburg bounds)
- Missing data summary per biomarker and study

MCD reference: General quality assurance before primary analysis.
"""

import logging
from typing import Optional

import pandas as pd

from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import (
    BIOMARKERS,
    MIN_SAMPLE_SIZE,
    get_available_biomarkers,
)

logger = logging.getLogger(__name__)

# Johannesburg bounding box (generous)
JHB_LAT_RANGE = (-27.0, -25.0)
JHB_LON_RANGE = (27.0, 29.0)

# Expected study year range
YEAR_RANGE = (2003, 2021)


def validate_required_columns(
    df: pd.DataFrame,
    extra_columns: Optional[list[str]] = None,
) -> list[str]:
    """Check that all required columns are present.

    Checks for:
    - Patient identifier and temporal columns
    - Demographic and clinical covariates from config
    - All available biomarker columns from the registry

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.
    extra_columns : list[str], optional
        Additional columns to require beyond the standard set.

    Returns
    -------
    list[str]
        Missing column names (empty if all present).
    """
    required = set()

    # Core identifiers
    required.add(config.PATIENT_ID_COLUMN)
    required.add("visit_date")
    required.add("study_source")

    # Covariates from config
    required.update(config.DEMOGRAPHIC_COVARIATES)
    required.update(config.CLINICAL_COVARIATES)

    # Available biomarker columns
    for spec in get_available_biomarkers().values():
        if spec.column:
            required.add(spec.column)

    if extra_columns:
        required.update(extra_columns)

    present = set(df.columns)
    missing = sorted(required - present)

    if missing:
        logger.warning(f"Missing {len(missing)} required columns: {missing}")
    else:
        logger.info(f"All {len(required)} required columns present")

    return missing


def validate_biomarker_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Report non-null counts for each biomarker, flagging sparse ones.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.

    Returns
    -------
    pd.DataFrame
        Summary with columns: biomarker, system, column, n_observations,
        pct_missing, meets_threshold, in_dataset.
    """
    rows = []
    n_total = len(df)

    for key, spec in get_available_biomarkers().items():
        in_dataset = spec.column in df.columns
        n_obs = int(df[spec.column].notna().sum()) if in_dataset else 0
        pct_missing = round(100 * (1 - n_obs / n_total), 1) if n_total > 0 else 100.0

        rows.append({
            "biomarker": key,
            "system": spec.system,
            "column": spec.column,
            "n_observations": n_obs,
            "pct_missing": pct_missing,
            "meets_threshold": n_obs >= MIN_SAMPLE_SIZE,
            "in_dataset": in_dataset,
        })

    summary = pd.DataFrame(rows)

    n_below = (~summary["meets_threshold"]).sum()
    n_missing_col = (~summary["in_dataset"]).sum()

    if n_missing_col > 0:
        logger.warning(
            f"{n_missing_col} biomarker column(s) not found in dataset: "
            f"{summary.loc[~summary['in_dataset'], 'biomarker'].tolist()}"
        )
    if n_below > 0:
        logger.warning(
            f"{n_below} biomarker(s) below MIN_SAMPLE_SIZE ({MIN_SAMPLE_SIZE}): "
            f"{summary.loc[~summary['meets_threshold'], 'biomarker'].tolist()}"
        )
    else:
        logger.info("All available biomarkers meet minimum sample size")

    return summary


def validate_date_ranges(df: pd.DataFrame) -> dict:
    """Check visit_date ranges per study are within expected bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'visit_date' and 'study_source' columns.

    Returns
    -------
    dict
        Keys: studies_checked, out_of_range (list of dicts with study,
        min_date, max_date), all_valid (bool).
    """
    result = {"studies_checked": 0, "out_of_range": [], "all_valid": True}

    if "visit_date" not in df.columns:
        logger.warning("No visit_date column — skipping date validation")
        return result

    dates = pd.to_datetime(df["visit_date"], errors="coerce")

    if "study_source" in df.columns:
        groups = df.groupby("study_source")
    else:
        groups = [("all", df)]

    for study, group in groups:
        result["studies_checked"] += 1
        study_dates = pd.to_datetime(group["visit_date"], errors="coerce").dropna()

        if study_dates.empty:
            continue

        min_year = study_dates.dt.year.min()
        max_year = study_dates.dt.year.max()

        if min_year < YEAR_RANGE[0] or max_year > YEAR_RANGE[1]:
            result["out_of_range"].append({
                "study": study,
                "min_date": str(study_dates.min().date()),
                "max_date": str(study_dates.max().date()),
            })
            result["all_valid"] = False
            logger.warning(
                f"Study {study}: dates {study_dates.min().date()} to "
                f"{study_dates.max().date()} outside expected {YEAR_RANGE}"
            )

    if result["all_valid"]:
        logger.info(
            f"All {result['studies_checked']} studies within "
            f"expected date range {YEAR_RANGE}"
        )

    return result


def validate_coordinates(df: pd.DataFrame) -> dict:
    """Verify coordinates are within Johannesburg bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'latitude' and 'longitude' columns.

    Returns
    -------
    dict
        Keys: n_checked, n_out_of_bounds, pct_out_of_bounds, all_valid.
    """
    result = {
        "n_checked": 0,
        "n_out_of_bounds": 0,
        "pct_out_of_bounds": 0.0,
        "all_valid": True,
    }

    if "latitude" not in df.columns or "longitude" not in df.columns:
        logger.warning("No lat/lon columns — skipping coordinate validation")
        return result

    has_coords = df["latitude"].notna() & df["longitude"].notna()
    coords = df.loc[has_coords, ["latitude", "longitude"]]
    result["n_checked"] = len(coords)

    if result["n_checked"] == 0:
        return result

    out_of_bounds = (
        (coords["latitude"] < JHB_LAT_RANGE[0])
        | (coords["latitude"] > JHB_LAT_RANGE[1])
        | (coords["longitude"] < JHB_LON_RANGE[0])
        | (coords["longitude"] > JHB_LON_RANGE[1])
    )

    result["n_out_of_bounds"] = int(out_of_bounds.sum())
    result["pct_out_of_bounds"] = round(
        100 * result["n_out_of_bounds"] / result["n_checked"], 2
    )
    result["all_valid"] = result["n_out_of_bounds"] == 0

    if not result["all_valid"]:
        logger.warning(
            f"{result['n_out_of_bounds']} records ({result['pct_out_of_bounds']}%) "
            f"outside Johannesburg bounds "
            f"(lat {JHB_LAT_RANGE}, lon {JHB_LON_RANGE})"
        )
    else:
        logger.info(
            f"All {result['n_checked']} records within Johannesburg bounds"
        )

    return result


def run_all_validations(df: pd.DataFrame) -> dict:
    """Run all validation checks and return a summary report.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis dataset.

    Returns
    -------
    dict
        Validation report with keys: missing_columns, biomarker_coverage,
        date_ranges, coordinates, overall_pass.
    """
    logger.info(f"Running validations on dataset: {len(df)} rows, {len(df.columns)} columns")

    missing_cols = validate_required_columns(df)
    coverage = validate_biomarker_coverage(df)
    dates = validate_date_ranges(df)
    coords = validate_coordinates(df)

    overall_pass = (
        len(missing_cols) == 0
        and dates["all_valid"]
        and coords["all_valid"]
    )

    report = {
        "missing_columns": missing_cols,
        "biomarker_coverage": coverage,
        "date_ranges": dates,
        "coordinates": coords,
        "overall_pass": overall_pass,
    }

    if overall_pass:
        logger.info("All validations PASSED")
    else:
        logger.warning("Some validations FAILED — review report before proceeding")

    return report
