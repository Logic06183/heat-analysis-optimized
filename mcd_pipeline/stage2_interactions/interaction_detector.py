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

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from mcd_pipeline import config


def compute_interaction_values(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
) -> np.ndarray:
    """Compute SHAP interaction values tensor.

    Parameters
    ----------
    model : xgb.XGBRegressor
        Trained model from Stage 1.
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    np.ndarray
        Shape (n_samples, n_features, n_features).
    """
    raise NotImplementedError("Stage 2: compute_interaction_values")


def rank_interactions(
    interaction_values: np.ndarray,
    feature_names: list[str],
    top_n: int = config.TOP_INTERACTIONS,
) -> pd.DataFrame:
    """Rank feature interactions by mean absolute interaction value.

    Returns
    -------
    pd.DataFrame
        Columns: feature_i, feature_j, mean_abs_interaction, rank
        Sorted by mean_abs_interaction descending.
    """
    raise NotImplementedError("Stage 2: rank_interactions")


def detect_temperature_interactions(
    interaction_values: np.ndarray,
    feature_names: list[str],
    temperature_features: Optional[list[str]] = None,
    modifier_features: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Extract interactions specifically between temperature lags and
    sociodemographic modifiers (dwelling_type, income, hiv_status, age).

    MCD reference: "SHAP interaction values (temperature × dwelling type,
    income, HIV status, age)"

    Returns
    -------
    pd.DataFrame
        Columns: temp_feature, modifier, mean_interaction, direction
    """
    raise NotImplementedError("Stage 2: detect_temperature_interactions")
