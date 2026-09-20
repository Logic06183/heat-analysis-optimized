"""
SA10: Humidity Sensitivity Analysis
====================================

Tests whether the primary temperature-biomarker findings hold when the
exposure variable is replaced with humidity-aware indices instead of
dry-bulb air temperature. Particularly relevant for the planned Abidjan
extension (tropical humid climate) where humidity is a first-order
modifier of perceived heat stress.

Approach
--------
1. Load the climate-linked dataset which already contains pre-computed
   daily-resolution humidity lag columns extracted from ERA5 (apparent
   temperature, NOAA heat index, Stull wet-bulb temperature), one column
   per lag day (lag 0, 1, 3, 7, 14, 21, 30).  These are stored in
   TIDY_climate_variables.csv as ``apparent_temp_lagNd_c`` etc. and
   pivoted wide at dataset-build time.
2. Copy those pre-computed lag columns into the internal
   ``{new_prefix}_lag{N}d_c`` format expected by the swap helper.
3. Refit Stage 1 with the alternative exposure and compare SHAP lag
   profiles against the primary (dry-bulb temperature) profiles.
4. Report concordance under the standard ρ ≥ 0.7 threshold.

Fallback behaviour
------------------
If the pre-computed ERA5 lag columns are absent (e.g., older dataset
build), the module falls back to the legacy rolling-window computation
from per-visit values.  This fallback produces degenerate results when
visits are spaced weeks apart (all lag windows collapse to the same
visit value) and should be treated as invalid.  Run the ERA5 humidity
extraction notebook to populate TIDY_climate_variables.csv with daily
lag series.

Outputs
-------
mcd_outputs/sensitivity/sa10_humidity/
  - sa10_summary.json  — concordance metrics, indexed by exposure variant
  - apparent_temperature/    Stage 1 outputs under apparent T
  - heat_index/              Stage 1 outputs under NOAA heat index
  - wet_bulb_stull/          Stage 1 outputs under Stull wet bulb
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from mcd_pipeline import config
from mcd_pipeline.sensitivity._sensitivity_core import (
    compare_lag_profiles,
    load_primary_lag_profiles,
    run_lightweight_stage1,
)

logger = logging.getLogger(__name__)


# Columns expected in the climate-linked dataset for the precomputed indices.
HUMIDITY_BASE_COLUMNS: dict[str, str] = {
    "apparent_temperature": "apparent_temp_c",
    "heat_index": "heat_index_c",
}

# Auxiliary columns used for the on-the-fly Stull wet-bulb computation.
RELATIVE_HUMIDITY_CANDIDATES: tuple[str, ...] = (
    "era5_relative_humidity_pct",
    "relative_humidity_pct",
    "rh_percent",
)
DEWPOINT_CANDIDATES: tuple[str, ...] = (
    "era5_dewpoint_2m_c",
    "dewpoint_c",
)
TEMPERATURE_PRIMARY_LAG = "dlnm_lag0_c"

# Pre-computed ERA5 lag column prefixes as stored in TIDY_climate_variables.csv.
# After the TIDY pivot, these appear as columns in the wide-format dataset.
# Note: lag-0 uses the suffix "_lag0_c" (no 'd'); lag-N > 0 uses "_lagNd_c".
PRECOMPUTED_TIDY_PREFIXES: dict[str, str] = {
    "apparent_temperature": "apparent_temp",
    "heat_index": "heat_index",
    "wet_bulb_stull": "wet_bulb",
}


# ---------------------------------------------------------------------------
# Climate index helpers
# ---------------------------------------------------------------------------


def stull_wet_bulb(t_air_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """Stull (2011) wet-bulb temperature approximation.

    Valid for relative humidity 5%–99% and air temperature -20°C–50°C.

    Reference
    ---------
    Stull, R. (2011) "Wet-bulb temperature from relative humidity and
    air temperature", J. Appl. Meteorol. Climatol. 50: 2267–2269.
    """
    rh = rh_pct.clip(lower=5.0, upper=99.0)
    tw = (
        t_air_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t_air_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * np.power(rh, 1.5) * np.arctan(0.023101 * rh)
        - 4.686035
    )
    return tw


def relative_humidity_from_dewpoint(t_air_c: pd.Series, dewpoint_c: pd.Series) -> pd.Series:
    """Magnus / August-Roche-Magnus approximation for RH from dew point."""
    a, b = 17.625, 243.04
    e_t = np.exp((a * t_air_c) / (b + t_air_c))
    e_td = np.exp((a * dewpoint_c) / (b + dewpoint_c))
    rh = 100.0 * (e_td / e_t)
    return rh.clip(lower=5.0, upper=100.0)


# ---------------------------------------------------------------------------
# Pre-computed lag column helpers
# ---------------------------------------------------------------------------


def _precomputed_col(tidy_prefix: str, lag: int) -> str:
    """Return the TIDY-pivot column name for a given lag day.

    TIDY convention: lag-0 → ``{prefix}_lag0_c`` (no 'd'),
                     lag-N → ``{prefix}_lagNd_c`` for N > 0.
    """
    suffix = "0" if lag == 0 else f"{lag}d"
    return f"{tidy_prefix}_lag{suffix}_c"


def _copy_precomputed_lag_columns(
    df: pd.DataFrame,
    tidy_prefix: str,
    new_prefix: str,
) -> pd.DataFrame:
    """Copy pre-computed daily ERA5 lag columns into the ``{new_prefix}_lagNd_c`` format.

    The TIDY pivot produces e.g.::

        apparent_temp_lag0_c   (lag 0, no 'd')
        apparent_temp_lag1d_c
        apparent_temp_lag7d_c  ...

    This copies them to::

        apptemp_lag0d_c
        apptemp_lag1d_c
        apptemp_lag7d_c  ...

    so ``_swap_temperature_for_humidity`` can proceed unchanged.

    Parameters
    ----------
    df : DataFrame
        Wide-format engineered dataset (output of build_engineered_dataset).
    tidy_prefix : str
        Column prefix in the TIDY pivot (e.g., ``"apparent_temp"``).
    new_prefix : str
        Target prefix for swap helper (e.g., ``"apptemp"``).

    Returns
    -------
    DataFrame with added ``{new_prefix}_lagNd_c`` columns.

    Raises
    ------
    KeyError
        If any expected pre-computed column is absent.
    """
    df_out = df.copy()
    missing = []
    for lag in config.LAG_DAYS:
        src_col = _precomputed_col(tidy_prefix, lag)
        dst_col = f"{new_prefix}_lag{lag}d_c"
        if src_col not in df_out.columns:
            missing.append(src_col)
        else:
            df_out[dst_col] = df_out[src_col]
    if missing:
        raise KeyError(
            f"Pre-computed humidity lag columns missing from dataset: {missing}. "
            "Re-run the ERA5 humidity extraction notebook and rebuild the engineered dataset."
        )
    return df_out


# ---------------------------------------------------------------------------
# Lag feature construction (legacy rolling-window fallback)
# ---------------------------------------------------------------------------


def _build_lag_features_from_series(
    df: pd.DataFrame,
    base_value_col: str,
    new_prefix: str,
) -> pd.DataFrame:
    """Compute lag features for a humidity index from per-visit values.

    For each visit (study_source, patient_id, visit_date) we compute the
    rolling N-day mean of the index values within the patient's history.
    Falls back to the same-day value if the patient has no history.

    The resulting columns mirror config.LAG_FEATURE_COLUMNS so Stage 1
    can be re-run by simply renaming columns at the entrypoint.
    """
    if base_value_col not in df.columns:
        raise KeyError(
            f"Humidity column '{base_value_col}' not present in climate-linked dataset"
        )

    df_local = df.sort_values(["patient_id", "visit_date"]).copy()
    df_local["visit_date"] = pd.to_datetime(df_local["visit_date"])

    # Per-patient daily series
    grouped = df_local.set_index("visit_date").groupby("patient_id")[base_value_col]

    out_cols: dict[str, pd.Series] = {}
    for lag in config.LAG_DAYS:
        if lag == 0:
            out_cols[f"{new_prefix}_lag0d_c"] = df_local[base_value_col].values
            continue
        # Rolling mean with min_periods=1 falls back to fewer days early
        rolled = grouped.rolling(window=f"{lag}D", min_periods=1).mean().reset_index()
        rolled = rolled.rename(columns={base_value_col: f"{new_prefix}_lag{lag}d_c"})
        # Re-index back into df_local order
        merged = df_local.reset_index(drop=True).merge(
            rolled,
            on=["patient_id", "visit_date"],
            how="left",
        )
        out_cols[f"{new_prefix}_lag{lag}d_c"] = merged[f"{new_prefix}_lag{lag}d_c"].values

    df_with_lags = df_local.reset_index(drop=True).copy()
    for col, vals in out_cols.items():
        df_with_lags[col] = vals
    return df_with_lags


def _swap_temperature_for_humidity(
    df: pd.DataFrame,
    new_prefix: str,
) -> pd.DataFrame:
    """Replace the temperature lag features with humidity-derived equivalents.

    The XGBoost trainer reads standardised ``temp_lag_{N}d`` columns created
    by ``engineer_features()`` — those are copies of ``dlnm_lag*_c`` made at
    dataset-build time.  Overwriting only ``dlnm_lag*_c`` (as done previously)
    leaves ``temp_lag_{N}d`` pointing at the original dry-bulb values, so the
    model never sees the humidity exposure.

    This function overwrites both the raw ``dlnm_lag*_c`` columns AND the
    standardised ``temp_lag_{N}d`` trainer features with humidity values.
    Original dry-bulb values are preserved under ``_dry_bulb_backup_*`` names.
    """
    df_swap = df.copy()

    for primary_col, lag in zip(config.LAG_FEATURE_COLUMNS, config.LAG_DAYS):
        humidity_col = f"{new_prefix}_lag{lag}d_c"
        if humidity_col not in df_swap.columns:
            raise KeyError(
                f"Expected humidity lag column {humidity_col} not present"
            )
        # Back up the original dry-bulb value for both representations.
        df_swap[f"_dry_bulb_backup_{primary_col}"] = df_swap[primary_col]
        std_col = f"temp_lag_{lag}d"
        if std_col in df_swap.columns:
            df_swap[f"_dry_bulb_backup_{std_col}"] = df_swap[std_col]

        # Overwrite raw lag column used for consistency checks.
        df_swap[primary_col] = df_swap[humidity_col]
        # Overwrite the standardised name that the XGBoost trainer actually reads.
        if std_col in df_swap.columns:
            df_swap[std_col] = df_swap[humidity_col]

    return df_swap


# ---------------------------------------------------------------------------
# Per-variant runner
# ---------------------------------------------------------------------------


def _run_variant(
    df_full: pd.DataFrame,
    variant_name: str,
    base_value_col: str,
    new_prefix: str,
    output_dir: Path,
    primary_profiles: dict,
    tidy_prefix: Optional[str] = None,
) -> dict:
    """Refit Stage 1 with one humidity-aware exposure variant.

    Parameters
    ----------
    tidy_prefix : str, optional
        TIDY-pivot column prefix (e.g. ``"apparent_temp"``).  When provided
        and the pre-computed ERA5 lag columns are present in ``df_full``, they
        are used directly.  Falls back to the legacy rolling-window approach if
        the columns are absent.
    """
    variant_dir = output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Prefer pre-computed daily-resolution ERA5 lag columns (correct).
    # Fall back to rolling-window approximation only if they are absent
    # (produces degenerate results for widely-spaced visit data).
    precomputed_available = (
        tidy_prefix is not None
        and _precomputed_col(tidy_prefix, 0) in df_full.columns
    )
    if precomputed_available:
        logger.info(
            "SA10 [%s]: using pre-computed ERA5 daily lag columns (tidy_prefix=%s)",
            variant_name, tidy_prefix,
        )
        df_with_lags = _copy_precomputed_lag_columns(df_full, tidy_prefix, new_prefix)
    else:
        if tidy_prefix is not None:
            logger.warning(
                "SA10 [%s]: pre-computed columns not found (checked %s); "
                "falling back to rolling-window approximation — results may be invalid",
                variant_name, _precomputed_col(tidy_prefix, 0),
            )
        df_with_lags = _build_lag_features_from_series(
            df=df_full, base_value_col=base_value_col, new_prefix=new_prefix
        )

    df_swapped = _swap_temperature_for_humidity(df_with_lags, new_prefix=new_prefix)

    n_rows = len(df_swapped)
    logger.info(
        "SA10 [%s]: refitting Stage 1 (%d rows, exposure=%s)",
        variant_name, n_rows, base_value_col,
    )

    sa_results = run_lightweight_stage1(
        df=df_swapped,
        output_dir=variant_dir,
        n_bootstrap=config.N_BOOTSTRAP_SENSITIVITY,
        label=f"sa10_{variant_name}",
    )

    comparisons: dict[str, dict] = {}
    rhos: list[float] = []
    for bio_name, result in sa_results.items():
        if "error" in result or "lag_summary" not in result:
            comparisons[bio_name] = {"status": result.get("error", "failed")}
            continue
        primary_lag = primary_profiles.get(bio_name)
        if primary_lag is None:
            comparisons[bio_name] = {"status": "no_primary_profile"}
            continue
        comparison = compare_lag_profiles(primary_lag, result["lag_summary"])
        comparison["cv_r2"] = result.get("cv_r2")
        comparisons[bio_name] = comparison
        if isinstance(comparison.get("spearman_rho"), float) and not np.isnan(comparison["spearman_rho"]):
            rhos.append(comparison["spearman_rho"])

    n_concordant = sum(1 for c in comparisons.values() if c.get("concordant", False))
    summary = {
        "variant": variant_name,
        "exposure_column": base_value_col,
        "n_rows_used": int(n_rows),
        "n_biomarkers_tested": len(comparisons),
        "n_biomarkers_concordant": n_concordant,
        "mean_spearman_rho": float(np.mean(rhos)) if rhos else None,
        "min_spearman_rho": float(np.min(rhos)) if rhos else None,
        "per_biomarker": comparisons,
    }
    with open(variant_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(
        "SA10 [%s] done: %d/%d concordant (mean ρ=%.3f)",
        variant_name,
        n_concordant,
        len(comparisons),
        float(np.mean(rhos)) if rhos else float("nan"),
    )
    return summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_humidity_sensitivity(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    primary_profiles: dict | None = None,
    include_wet_bulb: bool = True,
) -> dict:
    """Run all humidity-aware exposure variants and aggregate results.

    Parameters
    ----------
    df : DataFrame
        Engineered analysis dataset (output of build_engineered_dataset).
    output_dir : Path, optional
        Override output directory.
    primary_profiles : dict, optional
        Primary Stage 1 SHAP profiles for the dry-bulb exposure.
    include_wet_bulb : bool
        Compute Stull wet-bulb temperature on the fly when humidity inputs
        are present and refit Stage 1 with that exposure. Default True.
    """
    output_dir = output_dir or (config.OUTPUT_ROOT / "sensitivity" / "sa10_humidity")
    output_dir.mkdir(parents=True, exist_ok=True)

    if primary_profiles is None:
        primary_profiles = load_primary_lag_profiles()

    summary: dict = {
        "sensitivity_analysis": "sa10_humidity",
        "description": (
            "Refits the Stage 1 lag-response screen using humidity-aware "
            "exposure indices (apparent temperature, NOAA heat index, and "
            "Stull wet-bulb temperature where computable). Tests whether "
            "the dry-bulb temperature SHAP profiles are robust to the choice "
            "of heat-stress exposure metric."
        ),
        "concordance_threshold": config.CONCORDANCE_RHO_THRESHOLD,
        "variants": {},
    }

    # 1. Apparent temperature — use pre-computed ERA5 lag columns when available.
    app_temp_tidy_prefix = PRECOMPUTED_TIDY_PREFIXES["apparent_temperature"]
    app_temp_lag0_col = _precomputed_col(app_temp_tidy_prefix, 0)
    app_temp_base_col = HUMIDITY_BASE_COLUMNS["apparent_temperature"]
    if app_temp_lag0_col in df.columns or app_temp_base_col in df.columns:
        summary["variants"]["apparent_temperature"] = _run_variant(
            df_full=df,
            variant_name="apparent_temperature",
            base_value_col=app_temp_base_col,
            new_prefix="apptemp",
            output_dir=output_dir,
            primary_profiles=primary_profiles,
            tidy_prefix=app_temp_tidy_prefix,
        )
    else:
        summary["variants"]["apparent_temperature"] = {
            "status": "missing_column",
            "expected_column": app_temp_lag0_col,
        }

    # 2. NOAA heat index — use pre-computed ERA5 lag columns when available.
    heat_idx_tidy_prefix = PRECOMPUTED_TIDY_PREFIXES["heat_index"]
    heat_idx_lag0_col = _precomputed_col(heat_idx_tidy_prefix, 0)
    heat_idx_base_col = HUMIDITY_BASE_COLUMNS["heat_index"]
    if heat_idx_lag0_col in df.columns or heat_idx_base_col in df.columns:
        summary["variants"]["heat_index"] = _run_variant(
            df_full=df,
            variant_name="heat_index",
            base_value_col=heat_idx_base_col,
            new_prefix="heatidx",
            output_dir=output_dir,
            primary_profiles=primary_profiles,
            tidy_prefix=heat_idx_tidy_prefix,
        )
    else:
        summary["variants"]["heat_index"] = {
            "status": "missing_column",
            "expected_column": heat_idx_lag0_col,
        }

    # 3. Stull wet-bulb.
    #    Preferred path: use pre-computed ERA5 wet-bulb lag columns from TIDY.
    #    Fallback A: compute on the fly from RH/dewpoint columns in the dataset.
    #    Fallback B: skip if no humidity inputs are available.
    if include_wet_bulb:
        wb_tidy_prefix = PRECOMPUTED_TIDY_PREFIXES["wet_bulb_stull"]
        wb_lag0_col = _precomputed_col(wb_tidy_prefix, 0)
        if wb_lag0_col in df.columns:
            # Best path: pre-computed daily ERA5 wet-bulb.
            summary["variants"]["wet_bulb_stull"] = _run_variant(
                df_full=df,
                variant_name="wet_bulb_stull",
                base_value_col=wb_lag0_col,
                new_prefix="wb",
                output_dir=output_dir,
                primary_profiles=primary_profiles,
                tidy_prefix=wb_tidy_prefix,
            )
            summary["variants"]["wet_bulb_stull"]["source"] = "era5_precomputed"
        else:
            # Fallback: compute from RH or dewpoint columns.
            rh_col = next((c for c in RELATIVE_HUMIDITY_CANDIDATES if c in df.columns), None)
            dew_col = next((c for c in DEWPOINT_CANDIDATES if c in df.columns), None)
            if TEMPERATURE_PRIMARY_LAG in df.columns and (rh_col or dew_col):
                df_wb = df.copy()
                t_air = df_wb[TEMPERATURE_PRIMARY_LAG]
                if rh_col is None and dew_col is not None:
                    rh = relative_humidity_from_dewpoint(t_air, df_wb[dew_col])
                else:
                    rh = df_wb[rh_col]
                df_wb["wet_bulb_stull_c"] = stull_wet_bulb(t_air, rh)
                summary["variants"]["wet_bulb_stull"] = _run_variant(
                    df_full=df_wb,
                    variant_name="wet_bulb_stull",
                    base_value_col="wet_bulb_stull_c",
                    new_prefix="wb",
                    output_dir=output_dir,
                    primary_profiles=primary_profiles,
                )
                summary["variants"]["wet_bulb_stull"]["humidity_input_column"] = rh_col or dew_col
                summary["variants"]["wet_bulb_stull"]["source"] = "on_the_fly_stull"
            else:
                summary["variants"]["wet_bulb_stull"] = {
                    "status": "humidity_inputs_missing",
                    "tried_precomputed": wb_lag0_col,
                    "tried_relative_humidity": list(RELATIVE_HUMIDITY_CANDIDATES),
                    "tried_dewpoint": list(DEWPOINT_CANDIDATES),
                }
    else:
        summary["variants"]["wet_bulb_stull"] = {"status": "skipped_by_flag"}

    summary_path = output_dir / "sa10_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("SA10 complete: %d variants attempted", len(summary["variants"]))
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--no-wet-bulb",
        action="store_true",
        help="Skip the Stull wet-bulb on-the-fly variant.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    return p


def main() -> int:
    from mcd_pipeline.utils.logging_config import configure_logging
    from mcd_pipeline.stage0_data_prep.feature_engineering import (
        build_engineered_dataset,
    )

    args = _build_arg_parser().parse_args()
    configure_logging()

    df = build_engineered_dataset()
    run_humidity_sensitivity(
        df=df,
        output_dir=args.output_dir,
        include_wet_bulb=not args.no_wet_bulb,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
