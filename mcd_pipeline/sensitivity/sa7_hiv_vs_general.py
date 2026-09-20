"""
SA7: HIV Cohorts vs General Population — Stratified Analysis
=============================================================

Stratifies Stage 1 analysis by HIV status to assess whether
heat-biomarker relationships differ between PLHIV and general population.

Key question: Does immunocompromise modify the temperature–biomarker
relationship? If SHAP profiles differ, it suggests HIV status is an
effect modifier — important for public health policy in high-prevalence
settings like Johannesburg (~17% adult HIV prevalence).

MCD reference: "HIV cohorts vs general population cohorts"
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mcd_pipeline import config
from mcd_pipeline.sensitivity._sensitivity_core import (
    compare_lag_profiles,
    load_primary_lag_profiles,
    run_lightweight_stage1,
)

logger = logging.getLogger(__name__)


def run_hiv_stratified(
    df: pd.DataFrame,
    output_dir: Path = None,
    primary_profiles: dict = None,
) -> dict:
    """Run Stage 1 separately for HIV+ and HIV- subgroups.

    Compares:
    1. HIV+ SHAP profiles vs primary (full-sample) profiles
    2. HIV- SHAP profiles vs primary profiles
    3. HIV+ vs HIV- profiles (direct comparison)
    """
    output_dir = output_dir or (config.OUTPUT_ROOT / "sensitivity")
    sa_output = output_dir / "sa7_hiv_stratified"
    sa_output.mkdir(parents=True, exist_ok=True)

    if "hiv_status" not in df.columns:
        return {"status": "skipped", "reason": "hiv_status column not found"}

    # Validate hiv_status column type and content
    if not pd.api.types.is_string_dtype(df["hiv_status"]) and \
       not pd.api.types.is_object_dtype(df["hiv_status"]):
        raise ValueError(
            f"hiv_status column has unexpected dtype {df['hiv_status'].dtype}; "
            "expected string/object with values 'positive'/'negative'."
        )
    known = {"positive", "negative"}
    val_counts = df["hiv_status"].str.lower().value_counts()
    unrecognised = set(val_counts.index) - known
    if unrecognised:
        logger.warning(
            "SA7: hiv_status contains unrecognised values %s (%d rows) — these will be "
            "excluded from both strata. Check data quality.",
            unrecognised,
            df["hiv_status"].str.lower().isin(unrecognised).sum(),
        )

    # Split by HIV status
    hiv_pos = df[df["hiv_status"].str.lower() == "positive"].copy()
    hiv_neg = df[df["hiv_status"].str.lower() == "negative"].copy()

    n_unclassified = len(df) - len(hiv_pos) - len(hiv_neg)
    if n_unclassified > 0:
        logger.warning(
            "SA7: %d rows (%.1f%%) not classified as HIV+/HIV- and excluded",
            n_unclassified, 100 * n_unclassified / len(df),
        )

    logger.info(
        "=== SA7: HIV Stratified Analysis ===\n"
        "  HIV+: %d rows, %d patients\n"
        "  HIV-: %d rows, %d patients",
        len(hiv_pos), hiv_pos["patient_id"].nunique(),
        len(hiv_neg), hiv_neg["patient_id"].nunique(),
    )

    # Run Stage 1 for each stratum
    results_pos = run_lightweight_stage1(
        hiv_pos,
        output_dir=sa_output / "hiv_positive",
        label="sa7_hiv+",
    )
    results_neg = run_lightweight_stage1(
        hiv_neg,
        output_dir=sa_output / "hiv_negative",
        label="sa7_hiv-",
    )

    # Load primary profiles
    if primary_profiles is None:
        primary_profiles = load_primary_lag_profiles()

    # Compare each stratum vs primary, and vs each other
    comparisons = {}
    for bio_name in set(list(results_pos.keys()) + list(results_neg.keys())):
        bio_comp = {}
        primary_lag = primary_profiles.get(bio_name)

        # HIV+ vs primary
        pos_result = results_pos.get(bio_name, {})
        if "lag_summary" in pos_result and primary_lag is not None:
            bio_comp["hiv_pos_vs_primary"] = compare_lag_profiles(
                primary_lag, pos_result["lag_summary"]
            )
            bio_comp["hiv_pos_r2"] = pos_result.get("cv_r2")
            bio_comp["hiv_pos_n"] = pos_result.get("n_samples")

        # HIV- vs primary
        neg_result = results_neg.get(bio_name, {})
        if "lag_summary" in neg_result and primary_lag is not None:
            bio_comp["hiv_neg_vs_primary"] = compare_lag_profiles(
                primary_lag, neg_result["lag_summary"]
            )
            bio_comp["hiv_neg_r2"] = neg_result.get("cv_r2")
            bio_comp["hiv_neg_n"] = neg_result.get("n_samples")

        # HIV+ vs HIV- (direct)
        if "lag_summary" in pos_result and "lag_summary" in neg_result:
            bio_comp["hiv_pos_vs_neg"] = compare_lag_profiles(
                pos_result["lag_summary"], neg_result["lag_summary"]
            )

        comparisons[bio_name] = bio_comp

    # Identify effect modification
    effect_modifiers = []
    for bio_name, comp in comparisons.items():
        pos_neg = comp.get("hiv_pos_vs_neg", {})
        # Effect modifier if HIV+/HIV- profiles are NOT concordant (ρ < threshold)
        if pos_neg.get("spearman_rho") is not None and pos_neg["spearman_rho"] < config.CONCORDANCE_RHO_THRESHOLD:
            effect_modifiers.append(bio_name)

    summary = {
        "sensitivity_analysis": "sa7_hiv_stratified",
        "description": "HIV cohorts vs general population",
        "n_hiv_positive": len(hiv_pos),
        "n_hiv_negative": len(hiv_neg),
        "n_patients_positive": hiv_pos["patient_id"].nunique(),
        "n_patients_negative": hiv_neg["patient_id"].nunique(),
        "n_biomarkers_tested": len(comparisons),
        "effect_modifiers": effect_modifiers,
        "per_biomarker": comparisons,
    }

    with open(sa_output / "sa7_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(
        "  SA7 complete: %d biomarkers tested, %d potential effect modifiers: %s",
        len(comparisons), len(effect_modifiers), effect_modifiers,
    )
    return summary
