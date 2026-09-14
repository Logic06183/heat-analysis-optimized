"""
SHAP Profiler — Interventional TreeSHAP for Lag Attribution
============================================================

Computes SHAP values using **interventional** conditional expectation,
which marginalises over the empirical feature distribution rather than
conditioning on correlated neighbours. This is critical for correlated
lag features where standard tree-path-dependent SHAP would produce
misleading attributions.

References:
- Janzing, Minorics & Blöbaum (2020) "Feature relevance quantification
  in explainable AI: A causality problem" AISTATS
- Lundberg et al. (2020) "From local explanations to global understanding
  with explainable AI for trees" Nature Machine Intelligence 2:56-67

MCD reference: "TreeSHAP with interventional conditional expectation to
handle correlated lag features: this marginalises over the empirical
feature distribution rather than conditioning on correlated neighbours"
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import kendalltau

from mcd_pipeline import config

logger = logging.getLogger(__name__)


def compute_shap_values(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    feature_perturbation: str = config.SHAP_FEATURE_PERTURBATION,
) -> np.ndarray:
    """Compute interventional TreeSHAP values.

    Parameters
    ----------
    model : xgb.XGBRegressor
        Trained XGBoost model.
    X : pd.DataFrame
        Feature matrix (same features as training).
    feature_perturbation : str
        "interventional" (default) or "tree_path_dependent".

    Returns
    -------
    np.ndarray
        SHAP values array of shape (n_samples, n_features).
    """
    explainer = shap.TreeExplainer(
        model,
        data=X,
        feature_perturbation=feature_perturbation,
    )
    shap_values = explainer.shap_values(X)
    return shap_values


def aggregate_bootstrap_shap(
    shap_arrays: list[np.ndarray],
    feature_names: list[str],
) -> dict:
    """Aggregate SHAP values across bootstrap replicates.

    Computes mean |SHAP| importance per feature with bootstrap confidence
    intervals. Reports feature ranking stability via Kendall's W.

    Parameters
    ----------
    shap_arrays : list[np.ndarray]
        List of SHAP arrays from N bootstrap replicates.
    feature_names : list[str]
        Feature names corresponding to columns.

    Returns
    -------
    dict with keys:
        mean_abs_shap: pd.Series (feature -> mean |SHAP|)
        ci_lower: pd.Series (2.5th percentile)
        ci_upper: pd.Series (97.5th percentile)
        ranking_stability: float (Kendall's W across replicates)
    """
    n_reps = len(shap_arrays)

    # Mean |SHAP| per feature per replicate
    importance_per_rep = np.array(
        [np.abs(sv).mean(axis=0) for sv in shap_arrays]
    )  # shape: (n_reps, n_features)

    # Aggregate across replicates
    mean_importance = importance_per_rep.mean(axis=0)
    ci_lower = np.percentile(importance_per_rep, 2.5, axis=0)
    ci_upper = np.percentile(importance_per_rep, 97.5, axis=0)

    mean_abs_shap = pd.Series(mean_importance, index=feature_names).sort_values(
        ascending=False
    )
    ci_lower_s = pd.Series(ci_lower, index=feature_names)
    ci_upper_s = pd.Series(ci_upper, index=feature_names)

    # Kendall's W for ranking stability
    # Each replicate ranks the features; W measures concordance across rankings
    rankings = np.array(
        [(-imp).argsort().argsort() for imp in importance_per_rep]
    )  # shape: (n_reps, n_features)

    n_features = rankings.shape[1]
    if n_reps > 1 and n_features > 1:
        mean_ranks = rankings.mean(axis=0)
        ss_total = np.sum((mean_ranks - mean_ranks.mean()) ** 2)
        w = (12 * ss_total) / (n_reps**2 * (n_features**3 - n_features))
        w = min(max(w, 0.0), 1.0)
    else:
        w = 1.0

    return {
        "mean_abs_shap": mean_abs_shap,
        "ci_lower": ci_lower_s,
        "ci_upper": ci_upper_s,
        "ranking_stability": w,
    }


def save_shap_results(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    biomarker_name: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Save SHAP values and metadata for downstream stages.

    Saves:
    - shap_values.npy — raw SHAP array
    - shap_summary.csv — mean |SHAP| per feature
    - feature_names.json — ordered feature list

    These are consumed by Stage 2 (interactions) and Stage 3 (clustering).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw SHAP values
    shap_path = output_dir / "shap_values.npy"
    np.save(shap_path, shap_values)

    # Save feature names
    feature_names = X.columns.tolist()
    names_path = output_dir / "feature_names.json"
    with open(names_path, "w") as f:
        json.dump(feature_names, f)

    # Save summary: mean |SHAP| per feature, sorted
    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names,
    ).sort_values(ascending=False)

    summary_path = output_dir / "shap_summary.csv"
    mean_abs.to_csv(summary_path, header=True)

    logger.info(
        "  Saved SHAP results for %s to %s", biomarker_name, output_dir
    )

    return {
        "shap_values": shap_path,
        "feature_names": names_path,
        "summary": summary_path,
    }
