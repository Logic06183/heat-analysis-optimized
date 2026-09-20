"""
SA12 recovery script.

Why this exists: the original `mcd_pipeline.sensitivity.sa12_era5land` run was
killed by the harness with 22 of 24 biomarkers already complete on disk
(under `mcd_outputs/sensitivity/sa12_era5land/era5land_stage1/`). All the
expensive XGBoost+SHAP fitting for those 22 is persisted; only hematocrit and
hemoglobin are missing, plus the final concordance summary.

This script:
  1. Builds the analysis dataset, swaps primary ERA5 lag features for
     ERA5-Land (mirroring sa12_era5land.py).
  2. Runs lightweight Stage 1 ONLY for hematocrit and hemoglobin (the missing
     two) — `biomarker_names=["hematocrit", "hemoglobin"]`.
  3. Loads all 24 ERA5-Land lag profiles from disk and computes Spearman ρ
     concordance against the primary ERA5 lag profiles (same metric as SA2
     MODIS uses).
  4. Writes the final sa12_summary.json.

Run:
    python sa12_recover.py

Output:
    mcd_outputs/sensitivity/sa12_era5land/sa12_summary.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from mcd_pipeline import config
from mcd_pipeline.sensitivity._sensitivity_core import (
    compare_lag_profiles,
    load_primary_lag_profiles,
    run_lightweight_stage1,
)
from mcd_pipeline.sensitivity.sa12_era5land import _swap_primary_for_era5_land

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SA_DIR = config.OUTPUT_ROOT / "sensitivity" / "sa12_era5land"
STAGE1_DIR = SA_DIR / "era5land_stage1"
MISSING = ["hematocrit", "hemoglobin"]


def main() -> None:
    # --- 1. Pre-flight: which biomarkers actually need re-fitting? ---
    existing = sorted(p.name for p in STAGE1_DIR.iterdir() if p.is_dir())
    logger.info("Existing biomarker dirs: %d", len(existing))
    to_run = [b for b in MISSING if not (STAGE1_DIR / b / "lag_response_summary.csv").exists()]
    logger.info("Biomarkers still to fit: %s", to_run)

    if to_run:
        # --- 2. Build dataset and swap ERA5-Land in ---
        from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers
        from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
        from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features

        df = build_analysis_dataset()
        biomarkers = get_available_biomarkers()
        bio_cols = [b.column for b in biomarkers.values() if b.column and b.column in df.columns]
        df = engineer_features(df, bio_cols)

        land_df = pd.read_csv(config.ERA5_LAND_CSV, low_memory=False)
        df_land = _swap_primary_for_era5_land(df, land_df)
        logger.info("Dataset ready after ERA5-Land swap: %d rows", len(df_land))

        # --- 3. Fit ONLY the missing biomarkers ---
        run_lightweight_stage1(
            df_land,
            biomarker_names=to_run,
            output_dir=STAGE1_DIR,
            label="sa12_era5land_recover",
        )
        logger.info("Fitted missing biomarkers: %s", to_run)
    else:
        logger.info("All biomarkers already on disk — skipping refit.")

    # --- 4. Build concordance summary from all on-disk lag profiles ---
    logger.info("Building concordance summary from on-disk lag profiles...")
    primary_profiles = load_primary_lag_profiles()
    concordance: dict = {}
    for bio_dir in sorted(STAGE1_DIR.iterdir()):
        if not bio_dir.is_dir():
            continue
        bio = bio_dir.name
        lag_csv = bio_dir / "lag_response_summary.csv"
        cv_json = bio_dir / "cv_metrics.json"
        if not lag_csv.exists() or not cv_json.exists():
            concordance[bio] = {"status": "missing_files"}
            continue
        land_lag = pd.read_csv(lag_csv)
        primary_lag = primary_profiles.get(bio)
        if primary_lag is None:
            concordance[bio] = {"status": "no_primary_profile"}
            continue
        cmp = compare_lag_profiles(primary_lag, land_lag)
        # Annotate with R² from cv_metrics.json for context
        try:
            cv = json.loads(cv_json.read_text())
            folds = cv.get("folds", [])
            if folds:
                mean_r2 = sum(f.get("r2", 0) for f in folds) / len(folds)
                cmp["cv_r2_era5_land"] = float(mean_r2)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        concordance[bio] = cmp

    # --- 5. Summary ---
    concordant = [b for b, c in concordance.items() if c.get("concordant", False)]
    rhos = [c["spearman_rho"] for c in concordance.values()
            if isinstance(c.get("spearman_rho"), float)]
    summary = {
        "sensitivity_analysis": "sa12_era5_land",
        "description": (
            "Re-fit Stage 1 with ERA5-Land (~9 km) replacing primary ERA5 (~31 km) "
            "as the temperature exposure; SHAP lag profiles compared to primary "
            "via Spearman ρ (same metric as SA2 MODIS)."
        ),
        "n_biomarkers_compared": len(concordance),
        "n_biomarkers_concordant": len(concordant),
        "concordant_biomarkers": concordant,
        "mean_spearman_rho": float(sum(rhos) / len(rhos)) if rhos else None,
        "per_biomarker": concordance,
        "note_on_recovery": (
            "Original sa12_era5land run was killed at 22/24 biomarkers; "
            "hematocrit and hemoglobin re-fit by sa12_recover.py and the "
            "concordance summary built from all 24 on-disk lag profiles."
        ),
    }
    out_path = SA_DIR / "sa12_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info(
        "SA12 complete: %d/%d biomarkers concordant (ρ ≥ 0.7), mean ρ = %.3f",
        len(concordant),
        len(concordance),
        summary["mean_spearman_rho"] or 0.0,
    )
    logger.info("Written: %s", out_path)


if __name__ == "__main__":
    main()
