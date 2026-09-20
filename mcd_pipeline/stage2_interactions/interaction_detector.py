"""
SHAP Interaction Detector — Temperature × Covariate Interactions
================================================================

Computes SHAP interaction values to quantify effect modification:
temperature × dwelling type, income, HIV status, age.

Uses TreeExplainer.shap_interaction_values() which returns an
(n_samples, n_features, n_features) tensor. The off-diagonal elements
[i, j, k] represent the interaction effect between features j and k
for observation i.

Top 50 interactions ranked by absolute magnitude per biomarker model.

MCD reference: "Stage 2 — Interaction detection: SHAP interaction values
ranked by absolute magnitude; exploratory framing with permutation-derived
null (500 permutations per biomarker model)"
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from mcd_pipeline import config

logger = logging.getLogger(__name__)

# Subsample size for computing interaction tensor — balances accuracy vs speed.
# Full (N, F, F) tensor at N=15K and F=47 would take ~10 min; 300 takes ~10 sec.
INTERACTION_SAMPLE_SIZE = 300


def compute_interaction_values(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    sample_size: int = INTERACTION_SAMPLE_SIZE,
    seed: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP interaction values tensor.

    Uses tree-path-dependent SHAP interactions (no background needed),
    on a subsample of X for computational tractability.

    Parameters
    ----------
    model : xgb.XGBRegressor
        Trained model from Stage 1.
    X : pd.DataFrame
        Feature matrix.
    sample_size : int
        Max rows to use for interaction computation.
    seed : int
        Random seed for subsampling.

    Returns
    -------
    tuple of (interaction_values, X_sub)
        interaction_values : np.ndarray, shape (n_sub, n_features, n_features)
        X_sub : pd.DataFrame, the subsampled rows used
    """
    if len(X) > sample_size:
        X_sub = X.sample(n=sample_size, random_state=seed)
    else:
        X_sub = X.copy()

    logger.info(
        "  Computing SHAP interaction values (N=%d, %d features)...",
        len(X_sub), X_sub.shape[1],
    )
    explainer = shap.TreeExplainer(model)
    interaction_values = explainer.shap_interaction_values(X_sub)
    logger.info("  Interaction tensor shape: %s", interaction_values.shape)
    return interaction_values, X_sub


def rank_interactions(
    interaction_values: np.ndarray,
    feature_names: list[str],
    top_n: int = config.TOP_INTERACTIONS,
) -> pd.DataFrame:
    """Rank feature interactions by mean absolute interaction value.

    Only considers upper triangle (i < j) since tensor is symmetric.

    Returns
    -------
    pd.DataFrame
        Columns: feature_i, feature_j, mean_abs_interaction, rank
        Sorted by mean_abs_interaction descending.
    """
    n_features = len(feature_names)
    rows = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            mean_abs = float(np.abs(interaction_values[:, i, j]).mean())
            rows.append({
                "feature_i": feature_names[i],
                "feature_j": feature_names[j],
                "mean_abs_interaction": mean_abs,
                "idx_i": i,
                "idx_j": j,
            })

    df = (
        pd.DataFrame(rows)
        .sort_values("mean_abs_interaction", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    df["rank"] = df.index + 1
    return df


def detect_temperature_interactions(
    interaction_values: np.ndarray,
    feature_names: list[str],
    temperature_features: Optional[list[str]] = None,
    modifier_features: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Extract interactions specifically between temperature lags and
    sociodemographic modifiers (dwelling_type, income, HIV status, age).

    MCD reference: "SHAP interaction values (temperature × dwelling type,
    income, HIV status, age)"

    Returns
    -------
    pd.DataFrame
        Columns: temp_feature, modifier, mean_interaction, mean_abs_interaction,
                 direction, idx_i, idx_j
        Sorted by mean_abs_interaction descending.
    """
    if temperature_features is None:
        temperature_features = [f for f in feature_names if f.startswith("temp_lag_")]

    if modifier_features is None:
        modifier_keywords = ["age", "sex", "hiv", "gcro_dwelling", "gcro_income",
                             "gcro_education", "gcro_employment"]
        modifier_features = [
            f for f in feature_names
            if any(kw in f.lower() for kw in modifier_keywords)
        ]

    logger.info(
        "  Temperature features: %s", temperature_features
    )
    logger.info(
        "  Modifier features found: %s", modifier_features
    )

    rows = []
    for temp_f in temperature_features:
        if temp_f not in feature_names:
            continue
        i = feature_names.index(temp_f)
        for mod_f in modifier_features:
            if mod_f not in feature_names:
                continue
            j = feature_names.index(mod_f)
            interactions = interaction_values[:, i, j]
            mean_int = float(interactions.mean())
            mean_abs = float(np.abs(interactions).mean())
            rows.append({
                "temp_feature": temp_f,
                "modifier": mod_f,
                "mean_interaction": mean_int,
                "mean_abs_interaction": mean_abs,
                "direction": "positive" if mean_int > 0 else "negative",
                "idx_i": i,
                "idx_j": j,
            })

    if not rows:
        logger.warning("  No temperature × modifier interactions found.")
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("mean_abs_interaction", ascending=False)
        .reset_index(drop=True)
    )
