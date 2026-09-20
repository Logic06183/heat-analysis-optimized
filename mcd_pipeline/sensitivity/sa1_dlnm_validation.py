"""
SA1: DLNM Validation — Cross-Method Concordance Check
======================================================

Validates SHAP lag-response profiles against a simplified Distributed
Lag Model (DLM) for the top 5 biomarkers with largest SHAP-detected
effects.

Implementation: Python-only using statsmodels OLS with natural cubic
spline cross-basis. This is a simplified DLM (linear, not DLNM) that
tests whether the population-averaged lag-response shape from a
traditional epidemiological approach aligns with the individual-level
SHAP profiles.

Concordance strengthens findings; divergence interpretable as evidence
of individual-level heterogeneity captured by SHAP but averaged out
by DLM.

MCD reference: "DLNM validation for the 5 biomarkers with largest
SHAP-detected effects — concordance strengthens findings, divergence
interpretable as evidence of individual-level heterogeneity"
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mcd_pipeline import config
from mcd_pipeline.sensitivity._sensitivity_core import load_primary_lag_profiles

logger = logging.getLogger(__name__)


def _fit_distributed_lag_model(
    df: pd.DataFrame,
    biomarker_col: str,
    lag_cols: list[str],
    covariate_cols: list[str] = None,
) -> dict:
    """Fit a distributed lag linear model using OLS.

    Regresses the biomarker on all lag temperatures simultaneously,
    controlling for covariates. The coefficients on each lag represent
    the population-averaged effect at that lag window.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis dataset.
    biomarker_col : str
        Target biomarker column.
    lag_cols : list[str]
        Temperature lag feature columns (e.g., temp_lag_0d, ..., temp_lag_30d).
    covariate_cols : list[str], optional
        Additional covariates to control for.

    Returns
    -------
    dict with lag_coefficients, lag_std_errors, lag_pvalues, r_squared.
    """
    import statsmodels.api as sm

    # Prepare design matrix
    available_lags = [c for c in lag_cols if c in df.columns]
    if not available_lags:
        return {"status": "no_lag_columns"}

    cols = available_lags.copy()
    if covariate_cols:
        cols += [c for c in covariate_cols if c in df.columns]

    subset = df[[biomarker_col] + cols].dropna()
    if len(subset) < 100:
        return {"status": "insufficient_data", "n": len(subset)}

    y = subset[biomarker_col]
    X = subset[cols]

    # Encode categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)

    X = sm.add_constant(X.astype(float))
    model = sm.OLS(y.astype(float), X).fit()

    # Extract lag coefficients
    lag_results = []
    for lag_col in available_lags:
        lag_day = int(lag_col.split("_")[-1].replace("d", ""))
        if lag_col in model.params.index:
            lag_results.append({
                "lag_day": lag_day,
                "coefficient": float(model.params[lag_col]),
                "std_error": float(model.bse[lag_col]),
                "p_value": float(model.pvalues[lag_col]),
                "abs_coefficient": abs(float(model.params[lag_col])),
            })

    return {
        "status": "success",
        "lag_coefficients": lag_results,
        "r_squared": float(model.rsquared),
        "r_squared_adj": float(model.rsquared_adj),
        "n_observations": int(model.nobs),
        "aic": float(model.aic),
    }


def run_dlnm_validation(
    df: pd.DataFrame,
    output_dir: Path = None,
    top_n: int = 5,
) -> dict:
    """Run distributed lag model for top N biomarkers and compare with SHAP.

    Steps:
    1. Identify top N biomarkers by total SHAP lag importance
    2. Fit OLS distributed lag model for each
    3. Compare DLM coefficient profile with SHAP lag profile
    4. Report concordance (Spearman ρ between |coefficient| and |SHAP|)

    Parameters
    ----------
    df : pd.DataFrame
        Pre-engineered analysis dataset.
    output_dir : Path
        Output directory.
    top_n : int
        Number of top biomarkers to validate (default: 5).

    Returns
    -------
    dict — concordance summary.
    """
    output_dir = output_dir or (config.OUTPUT_ROOT / "sensitivity")
    sa_output = output_dir / "sa1_dlnm"
    sa_output.mkdir(parents=True, exist_ok=True)

    logger.info("=== SA1: DLNM Validation (top %d biomarkers) ===", top_n)

    # Load primary SHAP profiles
    primary_profiles = load_primary_lag_profiles()

    if not primary_profiles:
        return {"status": "skipped", "reason": "No primary lag profiles found"}

    # Rank biomarkers by total SHAP lag importance
    bio_importance = {}
    for bio_name, profile in primary_profiles.items():
        bio_importance[bio_name] = profile["mean_abs_shap"].sum()
    top_biomarkers = sorted(bio_importance, key=bio_importance.get, reverse=True)[:top_n]

    logger.info("  Top %d biomarkers: %s", top_n, top_biomarkers)

    # Lag feature columns
    lag_cols = [f"temp_lag_{d}d" for d in config.LAG_DAYS]

    # Covariates for DLM (control for confounders)
    covariate_cols = (
        config.DEMOGRAPHIC_COVARIATES
        + config.CLINICAL_COVARIATES
        + ["year", "fourier_sin_month", "fourier_cos_month"]
    )

    results = {}
    for bio_name in top_biomarkers:
        logger.info("  Fitting DLM for %s...", bio_name)

        dlm_result = _fit_distributed_lag_model(
            df, bio_name, lag_cols, covariate_cols
        )

        if dlm_result.get("status") != "success":
            results[bio_name] = dlm_result
            continue

        # Compare DLM lag profile with SHAP lag profile
        shap_profile = primary_profiles[bio_name]
        dlm_lags = pd.DataFrame(dlm_result["lag_coefficients"])

        if dlm_lags.empty:
            results[bio_name] = {"status": "no_lag_coefficients"}
            continue

        # Merge on lag_day
        merged = shap_profile.merge(dlm_lags, on="lag_day")

        if len(merged) < 3:
            results[bio_name] = {"status": "insufficient_overlap"}
            continue

        # Concordance: Spearman ρ between |SHAP| and |coefficient|
        rho, p_val = spearmanr(
            merged["mean_abs_shap"].values,
            merged["abs_coefficient"].values,
        )

        # Direction concordance: do the same lags dominate?
        shap_dominant = merged.loc[merged["mean_abs_shap"].idxmax(), "lag_day"]
        dlm_dominant = merged.loc[merged["abs_coefficient"].idxmax(), "lag_day"]

        results[bio_name] = {
            "status": "success",
            "dlm_r_squared": dlm_result["r_squared"],
            "dlm_n_observations": dlm_result["n_observations"],
            "concordance_rho": float(rho),
            "concordance_p": float(p_val),
            "concordant": bool(rho >= 0.5),
            "shap_dominant_lag": int(shap_dominant),
            "dlm_dominant_lag": int(dlm_dominant),
            "dominant_lag_agrees": bool(shap_dominant == dlm_dominant),
            "dlm_lag_coefficients": dlm_result["lag_coefficients"],
        }

        logger.info(
            "  %s: ρ=%.3f (p=%.4f), SHAP dominant=lag%d, DLM dominant=lag%d %s",
            bio_name, rho, p_val, shap_dominant, dlm_dominant,
            "✓" if shap_dominant == dlm_dominant else "✗",
        )

    # Summary
    concordant_count = sum(
        1 for r in results.values()
        if r.get("concordant", False)
    )
    dominant_agree = sum(
        1 for r in results.values()
        if r.get("dominant_lag_agrees", False)
    )

    summary = {
        "sensitivity_analysis": "sa1_dlnm_validation",
        "description": f"DLM validation for top {top_n} biomarkers",
        "top_biomarkers": top_biomarkers,
        "n_concordant": concordant_count,
        "n_dominant_agree": dominant_agree,
        "n_tested": len(results),
        "per_biomarker": results,
    }

    with open(sa_output / "sa1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(
        "  SA1 complete: %d/%d concordant (ρ≥0.5), %d/%d dominant lag agrees",
        concordant_count, len(results), dominant_agree, len(results),
    )
    return summary
