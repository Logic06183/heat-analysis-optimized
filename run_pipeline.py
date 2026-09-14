"""
MCD Pipeline Orchestrator
=========================

Top-level script to run the complete MCD analysis pipeline.

Usage::

    python run_pipeline.py                    # Run all stages (full: 50 bootstrap)
    python run_pipeline.py --fast             # Exploratory run (10 bootstrap)
    python run_pipeline.py --stage 1          # Run only Stage 1
    python run_pipeline.py --stage 0 1        # Run Stages 0 and 1
    python run_pipeline.py --sensitivity      # Run sensitivity analyses
    python run_pipeline.py --dry-run          # Validate config and data only
    python run_pipeline.py --validate         # Run data validation only

MCD reference: "Heat, Vulnerability, and the Body: Multi-System Biomarker
Signatures of Temperature Exposure in Johannesburg Revealed Through
Explainable Machine Learning"
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import (
    get_available_biomarkers,
    BIOMARKER_SYSTEMS,
)
from mcd_pipeline.stage0_data_prep.data_validation import run_all_validations
from mcd_pipeline.utils.reproducibility import (
    set_global_seeds,
    capture_environment,
    compute_data_checksum,
)


def setup_logging() -> logging.Logger:
    """Configure pipeline logging."""
    log_dir = config.OUTPUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "pipeline.log"),
        ],
    )
    return logging.getLogger("mcd_pipeline")


def validate_config(logger: logging.Logger) -> bool:
    """Validate pipeline configuration and data availability."""
    ok = True

    # Check primary dataset exists
    if not config.ANALYSIS_READY_CSV.exists():
        logger.error(f"ANALYSIS_READY not found: {config.ANALYSIS_READY_CSV}")
        ok = False

    # Check climate-linked directory
    if not config.CLIMATE_LINKED_DIR.exists():
        logger.error(f"Climate-linked dir not found: {config.CLIMATE_LINKED_DIR}")
        ok = False

    # Report biomarker availability
    biomarkers = get_available_biomarkers()
    logger.info(
        f"{len(biomarkers)} biomarkers available across "
        f"{len(BIOMARKER_SYSTEMS)} systems"
    )

    # Checksum primary dataset for provenance
    if config.ANALYSIS_READY_CSV.exists():
        checksum = compute_data_checksum(config.ANALYSIS_READY_CSV)
        logger.info(f"ANALYSIS_READY SHA-256: {checksum[:16]}...")

    return ok


def validate_data(logger: logging.Logger) -> bool:
    """Load the primary dataset and run all validation checks."""
    if not config.ANALYSIS_READY_CSV.exists():
        logger.error("Cannot validate data — ANALYSIS_READY not found")
        return False

    logger.info("Loading ANALYSIS_READY for validation...")
    df = pd.read_csv(config.ANALYSIS_READY_CSV, low_memory=False)

    report = run_all_validations(df)

    if report["overall_pass"]:
        logger.info("Data validation PASSED")
    else:
        logger.warning("Data validation found issues — review output above")

    return report["overall_pass"]


def run_stage0(logger: logging.Logger, fast: bool = False) -> pd.DataFrame:
    """Stage 0: Data preparation and feature engineering."""
    logger.info("=" * 60)
    logger.info("STAGE 0: Data Preparation")
    logger.info("=" * 60)

    from mcd_pipeline.stage0_data_prep.build_analysis_dataset import (
        build_analysis_dataset,
    )
    from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features

    df = build_analysis_dataset()

    biomarkers = get_available_biomarkers()
    bio_cols = [
        b.column
        for b in biomarkers.values()
        if b.column and b.column in df.columns
    ]
    df = engineer_features(df, bio_cols)

    logger.info("Stage 0 complete: %d rows x %d cols", len(df), len(df.columns))
    return df


def run_stage1(
    logger: logging.Logger, df: pd.DataFrame = None, fast: bool = False,
) -> dict:
    """Stage 1: Lag-response profiling (XGBoost-SHAP per biomarker)."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Lag-Response Profiling")
    if fast:
        logger.info("  (FAST mode: %d bootstrap replicates)", config.N_BOOTSTRAP_FAST)
    else:
        logger.info(
            "  (FULL mode: %d bootstrap replicates)", config.N_BOOTSTRAP_REPLICATES
        )
    logger.info("=" * 60)

    # Override bootstrap count for fast mode
    if fast:
        config.N_BOOTSTRAP_REPLICATES = config.N_BOOTSTRAP_FAST

    from mcd_pipeline.stage1_lag_profiling.run_stage1 import run_stage1 as _run

    return _run(df=df)


def run_stage2(logger: logging.Logger, fast: bool = False) -> None:
    """Stage 2: Interaction detection."""
    logger.info("=" * 60)
    logger.info("STAGE 2: Interaction Detection")
    logger.info("=" * 60)
    raise NotImplementedError("Stage 2 not yet implemented")


def run_stage3(logger: logging.Logger, fast: bool = False) -> None:
    """Stage 3: Vulnerability clustering."""
    logger.info("=" * 60)
    logger.info("STAGE 3: Vulnerability Clustering")
    logger.info("=" * 60)
    raise NotImplementedError("Stage 3 not yet implemented")


def run_sensitivity(logger: logging.Logger) -> None:
    """Run enabled sensitivity analyses."""
    logger.info("=" * 60)
    logger.info("SENSITIVITY ANALYSES")
    logger.info("=" * 60)

    for sa_name, enabled in config.SENSITIVITY_ANALYSES.items():
        if enabled:
            logger.info(f"  Running {sa_name}...")
        else:
            logger.info(f"  Skipping {sa_name} (disabled)")

    raise NotImplementedError("Sensitivity analyses not yet implemented")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCD Pipeline — Multi-System Biomarker Heat Analysis"
    )
    parser.add_argument(
        "--stage",
        nargs="*",
        type=int,
        choices=[0, 1, 2, 3],
        help="Run specific stages (default: all)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Exploratory run: 10 bootstrap replicates instead of 50",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run sensitivity analyses",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration only, do not run analysis",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run data validation checks only",
    )
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("MCD Pipeline starting")
    logger.info(f"Project root: {config.PROJECT_ROOT}")

    # Set reproducibility seeds
    set_global_seeds()

    # Capture environment
    env = capture_environment()
    logger.info(f"Environment captured: {env['n_packages']} packages")

    if not validate_config(logger):
        logger.error("Configuration validation failed. Exiting.")
        sys.exit(1)

    if args.validate:
        ok = validate_data(logger)
        sys.exit(0 if ok else 1)

    if args.dry_run:
        logger.info("Dry run complete. Configuration is valid.")
        return

    stages = args.stage if args.stage is not None else [0, 1, 2, 3]

    # Run stages with data passing
    df = None

    if 0 in stages:
        df = run_stage0(logger, fast=args.fast)

    if 1 in stages:
        results = run_stage1(logger, df=df, fast=args.fast)
        logger.info(
            "Stage 1 results: %d biomarkers processed",
            sum(1 for r in results.values() if "error" not in r),
        )

    if 2 in stages:
        run_stage2(logger, fast=args.fast)

    if 3 in stages:
        run_stage3(logger, fast=args.fast)

    if args.sensitivity:
        run_sensitivity(logger)

    logger.info("MCD Pipeline complete")


if __name__ == "__main__":
    main()
