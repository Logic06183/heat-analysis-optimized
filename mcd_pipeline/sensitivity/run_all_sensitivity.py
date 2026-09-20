"""
Run All Sensitivity Analyses
============================

Orchestrates all 8 sensitivity analyses. Skips blocked analyses
(SA5 until imputation complete) and disabled analyses per config.

Usage:
    python -m mcd_pipeline.sensitivity.run_all_sensitivity

The orchestrator builds the dataset once and passes it to each SA,
avoiding redundant I/O.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from mcd_pipeline import config

logger = logging.getLogger(__name__)


def run_all_sensitivity(
    df: pd.DataFrame = None,
    output_dir: Path = None,
) -> dict:
    """Run all enabled sensitivity analyses.

    Checks config.SENSITIVITY_ANALYSES flags. Skips disabled analyses
    with a logged warning.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-engineered analysis dataset. If None, builds from scratch.
    output_dir : Path, optional
        Output root. Defaults to mcd_outputs/sensitivity/.

    Returns
    -------
    dict
        Keyed by SA name with results or skip reason.
    """
    output_dir = output_dir or (config.OUTPUT_ROOT / "sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset once if not provided
    if df is None:
        logger.info("Building analysis dataset for sensitivity analyses...")
        from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
        from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features
        from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers

        df = build_analysis_dataset()
        biomarkers_dict = get_available_biomarkers()
        bio_cols = [b.column for b in biomarkers_dict.values() if b.column and b.column in df.columns]
        df = engineer_features(df, bio_cols)

    # Load primary lag profiles once
    from mcd_pipeline.sensitivity._sensitivity_core import load_primary_lag_profiles
    primary_profiles = load_primary_lag_profiles()

    results = {}
    flags = config.SENSITIVITY_ANALYSES

    # --- SA1: DLNM Validation ---
    if flags.get("sa1_dlnm", False):
        logger.info("\n" + "=" * 60)
        try:
            from mcd_pipeline.sensitivity.sa1_dlnm_validation import run_dlnm_validation
            results["sa1_dlnm"] = run_dlnm_validation(df, output_dir=output_dir)
        except Exception as e:
            logger.error("SA1 failed: %s", e)
            results["sa1_dlnm"] = {"status": "error", "error": str(e)}
    else:
        results["sa1_dlnm"] = {"status": "disabled"}

    # --- SA2: MODIS vs ERA5 ---
    if flags.get("sa2_modis_vs_era5", False):
        logger.info("\n" + "=" * 60)
        try:
            from mcd_pipeline.sensitivity.sa2_modis_vs_era5 import run_modis_comparison
            results["sa2_modis"] = run_modis_comparison(df, output_dir=output_dir)
        except Exception as e:
            logger.error("SA2 failed: %s", e)
            results["sa2_modis"] = {"status": "error", "error": str(e)}
    else:
        results["sa2_modis"] = {"status": "disabled"}

    # --- SA3: Pollution-Adjusted ---
    if flags.get("sa3_pollution", False):
        logger.info("\n" + "=" * 60)
        try:
            from mcd_pipeline.sensitivity.sa3_pollution_adjusted import run_pollution_adjusted
            results["sa3_pollution"] = run_pollution_adjusted(
                df, output_dir=output_dir, primary_profiles=primary_profiles
            )
        except Exception as e:
            logger.error("SA3 failed: %s", e)
            results["sa3_pollution"] = {"status": "error", "error": str(e)}
    else:
        results["sa3_pollution"] = {"status": "disabled"}

    # --- SA4: COVID Exclusion ---
    if flags.get("sa4_covid", False):
        logger.info("\n" + "=" * 60)
        try:
            from mcd_pipeline.sensitivity.sa4_covid_exclusion import run_covid_exclusion
            results["sa4_covid"] = run_covid_exclusion(
                df, output_dir=output_dir, primary_profiles=primary_profiles
            )
        except Exception as e:
            logger.error("SA4 failed: %s", e)
            results["sa4_covid"] = {"status": "error", "error": str(e)}
    else:
        results["sa4_covid"] = {"status": "disabled"}

    # --- SA5: Imputation (BLOCKED) ---
    if flags.get("sa5_imputation", False):
        logger.info("\n" + "=" * 60)
        try:
            from mcd_pipeline.sensitivity.sa5_imputation_compare import run_imputation_comparison
            results["sa5_imputation"] = run_imputation_comparison(df)
        except NotImplementedError as e:
            results["sa5_imputation"] = {"status": "blocked", "reason": str(e)}
        except Exception as e:
            logger.error("SA5 failed: %s", e)
            results["sa5_imputation"] = {"status": "error", "error": str(e)}
    else:
        results["sa5_imputation"] = {"status": "disabled_blocked"}

    # --- SA6: Case-Crossover ---
    if flags.get("sa6_case_crossover", False):
        logger.info("\n" + "=" * 60)
        try:
            from mcd_pipeline.sensitivity.sa6_case_crossover import run_case_crossover
            results["sa6_case_crossover"] = run_case_crossover(df, output_dir=output_dir)
        except Exception as e:
            logger.error("SA6 failed: %s", e)
            results["sa6_case_crossover"] = {"status": "error", "error": str(e)}
    else:
        results["sa6_case_crossover"] = {"status": "disabled"}

    # --- SA7: HIV Stratified ---
    if flags.get("sa7_hiv_stratified", False):
        logger.info("\n" + "=" * 60)
        try:
            from mcd_pipeline.sensitivity.sa7_hiv_vs_general import run_hiv_stratified
            results["sa7_hiv"] = run_hiv_stratified(
                df, output_dir=output_dir, primary_profiles=primary_profiles
            )
        except Exception as e:
            logger.error("SA7 failed: %s", e)
            results["sa7_hiv"] = {"status": "error", "error": str(e)}
    else:
        results["sa7_hiv"] = {"status": "disabled"}

    # --- SA8: Heatwave Threshold ---
    if flags.get("sa8_heatwave", False):
        logger.info("\n" + "=" * 60)
        try:
            from mcd_pipeline.sensitivity.sa8_heatwave_threshold import run_heatwave_threshold_comparison
            results["sa8_heatwave"] = run_heatwave_threshold_comparison(
                df, output_dir=output_dir, primary_profiles=primary_profiles
            )
        except Exception as e:
            logger.error("SA8 failed: %s", e)
            results["sa8_heatwave"] = {"status": "error", "error": str(e)}
    else:
        results["sa8_heatwave"] = {"status": "disabled"}

    # --- Save overall summary ---
    # Count successes
    completed = [k for k, v in results.items() if v.get("status") not in ("disabled", "disabled_blocked", "error", "blocked")]
    errored = [k for k, v in results.items() if v.get("status") == "error"]
    skipped = [k for k, v in results.items() if v.get("status") in ("disabled", "disabled_blocked", "blocked")]

    overall = {
        "n_completed": len(completed),
        "n_errored": len(errored),
        "n_skipped": len(skipped),
        "completed": completed,
        "errored": errored,
        "skipped": skipped,
        "per_analysis": {k: _summarise_sa(v) for k, v in results.items()},
    }

    with open(output_dir / "sensitivity_overall_summary.json", "w") as f:
        json.dump(overall, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("=== SENSITIVITY ANALYSES COMPLETE ===")
    logger.info("  Completed: %d (%s)", len(completed), ", ".join(completed))
    logger.info("  Errored:   %d (%s)", len(errored), ", ".join(errored))
    logger.info("  Skipped:   %d (%s)", len(skipped), ", ".join(skipped))

    return results


def _summarise_sa(result: dict) -> dict:
    """Extract a one-line summary from an SA result."""
    status = result.get("status", "completed")
    if status in ("disabled", "disabled_blocked", "blocked", "error", "skipped"):
        return {"status": status}

    summary = {"status": status}
    if "mean_spearman_rho" in result:
        summary["mean_rho"] = result["mean_spearman_rho"]
        summary["n_concordant"] = result.get("n_concordant")
    if "n_concordant" in result:
        summary["n_concordant"] = result["n_concordant"]
    if "n_tested" in result:
        summary["n_tested"] = result["n_tested"]
    if "effect_modifiers" in result:
        summary["effect_modifiers"] = result["effect_modifiers"]

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    results = run_all_sensitivity()
