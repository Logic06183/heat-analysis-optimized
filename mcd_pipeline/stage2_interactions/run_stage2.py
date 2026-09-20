"""
Run Stage 2 — Orchestrate Interaction Detection Across Biomarkers
=================================================================

For each biomarker with a trained Stage 1 model:
1. Compute SHAP interaction values
2. Rank top 50 interactions
3. Test temperature × modifier pairs against permutation null (500 perms)
4. Apply BH FDR correction at q=0.10
5. Save results

Usage:
    python -m mcd_pipeline.stage2_interactions.run_stage2
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from mcd_pipeline import config
from mcd_pipeline.stage2_interactions.fdr_correction import apply_fdr_to_interactions
from mcd_pipeline.stage2_interactions.interaction_detector import (
    compute_interaction_values,
    detect_temperature_interactions,
    rank_interactions,
)
from mcd_pipeline.stage2_interactions.permutation_null import test_top_interactions

logger = logging.getLogger(__name__)


def run_stage2(
    df: pd.DataFrame = None,
    stage1_dir: Path = config.OUTPUT_ROOT / "stage1",
    output_dir: Path = config.OUTPUT_ROOT / "stage2",
    n_permutations: int = config.N_PERMUTATIONS,
) -> dict:
    """Run Stage 2 interaction detection for all biomarkers.

    Requires Stage 1 outputs (trained models and feature matrices).

    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-built analysis dataset. If None, builds from TIDY inputs.
    stage1_dir : Path
        Directory containing Stage 1 per-biomarker output dirs.
    output_dir : Path
        Root output directory for Stage 2 results.
    n_permutations : int
        Permutations for null distribution (default: 500).

    Returns
    -------
    dict
        Summary keyed by biomarker with significant_interactions,
        top_temperature_modifiers, and output_paths.
    """
    from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers
    from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
    from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features
    from mcd_pipeline.stage1_lag_profiling.xgboost_trainer import prepare_features_and_target

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Stage 2: Interaction Detection ===")
    logger.info("  Permutations per pair: %d", n_permutations)
    logger.info("  FDR threshold: q=%.2f", config.FDR_THRESHOLD)

    # Build dataset once
    if df is None:
        logger.info("Building analysis dataset...")
        df = build_analysis_dataset()
        biomarkers_all = get_available_biomarkers()
        bio_cols = [b.column for b in biomarkers_all.values()
                    if b.column and b.column in df.columns]
        df = engineer_features(df, bio_cols)

    biomarkers_dict = get_available_biomarkers()

    # Find biomarkers with Stage 1 model outputs
    eligible = []
    for name, spec in biomarkers_dict.items():
        model_path = stage1_dir / spec.column / "model.ubj"
        cv_path = stage1_dir / spec.column / "cv_metrics.json"
        if not model_path.exists():
            logger.info("  SKIP %s: no model.ubj (Stage 1 not run or failed)", spec.column)
            continue
        if cv_path.exists():
            with open(cv_path) as f:
                cv = json.load(f)
            if cv.get("mean_r2", 0) < 0:
                logger.info("  SKIP %s: R²=%.3f < 0", spec.column, cv["mean_r2"])
                continue
        eligible.append(spec)

    logger.info("=== Stage 2: Processing %d/%d eligible biomarkers ===",
                len(eligible), len(biomarkers_dict))

    # Load any previously completed results (enables resume after VM reboot)
    summary = {}
    overall_summary_path = output_dir / "stage2_summary.json"
    if overall_summary_path.exists():
        with open(overall_summary_path) as f:
            summary = json.load(f)
        logger.info("Resuming: %d biomarkers already completed", len(summary))

    for i, biomarker in enumerate(eligible):
        bio_name = biomarker.column

        # Skip if already completed
        checkpoint = output_dir / bio_name / "stage2_summary.json"
        if checkpoint.exists() and bio_name not in summary:
            with open(checkpoint) as f:
                summary[bio_name] = json.load(f)
        if bio_name in summary and "error" not in summary[bio_name]:
            logger.info("[%d/%d] %s — SKIPPING (already complete)", i + 1, len(eligible), bio_name)
            continue

        logger.info("[%d/%d] %s", i + 1, len(eligible), bio_name)

        bio_output = output_dir / bio_name
        bio_output.mkdir(parents=True, exist_ok=True)

        try:
            # Prepare features
            X, y, groups = prepare_features_and_target(df, biomarker)
            feature_names = X.columns.tolist()

            # Load trained model
            model = xgb.XGBRegressor()
            model.load_model(stage1_dir / bio_name / "model.ubj")

            # Step 1: Compute interaction tensor
            interaction_values, X_sub = compute_interaction_values(model, X)

            # Step 2: Rank all pairs
            ranked = rank_interactions(interaction_values, feature_names)
            ranked.to_csv(bio_output / "ranked_interactions.csv", index=False)

            # Step 3: Temperature × modifier interactions
            temp_interactions = detect_temperature_interactions(
                interaction_values, feature_names
            )
            if not temp_interactions.empty:
                temp_interactions.to_csv(
                    bio_output / "temperature_modifier_interactions.csv", index=False
                )

            # Step 4: Permutation test (temperature pairs in top-50 only)
            tested = test_top_interactions(
                model, X, ranked, n_permutations=n_permutations,
                checkpoint_path=bio_output / "pair_checkpoint.csv",
            )

            # Step 5: FDR correction on tested pairs
            has_pval = tested["p_value"].notna()
            if has_pval.sum() > 0:
                tested_with_pval = apply_fdr_to_interactions(
                    tested[has_pval].copy(), q=config.FDR_THRESHOLD
                )
                tested.loc[has_pval, "p_adjusted"] = tested_with_pval["p_adjusted"].values
                tested.loc[has_pval, "significant_fdr"] = tested_with_pval["significant_fdr"].values
            else:
                tested["p_adjusted"] = np.nan
                tested["significant_fdr"] = False

            tested.to_csv(bio_output / "interaction_test_results.csv", index=False)

            n_sig = int(tested["significant_fdr"].fillna(False).sum())
            n_tested = int(has_pval.sum())

            # Top temperature modifiers
            temp_mask = (
                tested["feature_i"].str.startswith("temp_lag_") |
                tested["feature_j"].str.startswith("temp_lag_")
            )
            sig_temp = tested[temp_mask].copy()
            top_modifiers = []
            if not sig_temp.empty:
                sig_mask = sig_temp["significant_fdr"].fillna(False)
                top_df = sig_temp[sig_mask] if sig_mask.any() else sig_temp.head(5)
                for _, row in top_df.iterrows():
                    fi, fj = row["feature_i"], row["feature_j"]
                    modifier = fj if fi.startswith("temp_lag_") else fi
                    top_modifiers.append({
                        "modifier": modifier,
                        "temp_feature": fi if fi.startswith("temp_lag_") else fj,
                        "mean_abs_interaction": float(row["mean_abs_interaction"]),
                        "p_adjusted": float(row["p_adjusted"]) if pd.notna(row.get("p_adjusted")) else None,
                    })

            bio_summary = {
                "biomarker": bio_name,
                "n_pairs_ranked": len(ranked),
                "n_pairs_tested": n_tested,
                "n_significant_fdr": n_sig,
                "top_temperature_modifiers": top_modifiers,
                "output_dir": str(bio_output),
            }
            with open(bio_output / "stage2_summary.json", "w") as f:
                json.dump(bio_summary, f, indent=2, default=str)

            summary[bio_name] = bio_summary
            logger.info(
                "  %s: %d/%d tested pairs, %d FDR-significant",
                bio_name, n_tested, len(ranked), n_sig,
            )

        except Exception as e:
            logger.error("  FAILED: %s — %s", bio_name, e, exc_info=True)
            summary[bio_name] = {"biomarker": bio_name, "error": str(e)}

        # Save overall summary after each biomarker (checkpoint)
        with open(overall_summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    # Save final overall summary
    with open(output_dir / "stage2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    n_success = sum(1 for v in summary.values() if "error" not in v and not v.get("skipped"))
    logger.info("=== Stage 2 complete: %d/%d biomarkers succeeded ===",
                n_success, len(eligible))
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    results = run_stage2()
