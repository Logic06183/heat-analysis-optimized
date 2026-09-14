"""
Permutation Null Distribution — Statistical Significance for Interactions
=========================================================================

Generates a null distribution of SHAP interaction values by permuting
the modifier variable (breaking the true interaction while preserving
marginal distributions). Compares observed interactions against this null
to compute p-values.

500 permutations per biomarker model, as specified in the MCD.

MCD reference: "exploratory framing with permutation-derived null
(500 permutations per biomarker model)"
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from mcd_pipeline import config


def generate_permutation_null(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    feature_i: str,
    feature_j: str,
    n_permutations: int = config.N_PERMUTATIONS,
    seed: int = config.MASTER_SEED,
) -> np.ndarray:
    """Generate null distribution of interaction values by permuting feature_j.

    For each permutation:
    1. Shuffle feature_j values (breaking interaction with feature_i)
    2. Recompute SHAP interaction values for the (i, j) pair
    3. Record the mean absolute interaction

    Parameters
    ----------
    model : xgb.XGBRegressor
        Trained model.
    X : pd.DataFrame
        Original feature matrix.
    feature_i, feature_j : str
        Feature pair to test.
    n_permutations : int
        Number of permutations (default: 500).

    Returns
    -------
    np.ndarray
        Null distribution of mean |interaction| values, length n_permutations.
    """
    raise NotImplementedError("Stage 2: generate_permutation_null")


def compute_permutation_pvalue(
    observed: float,
    null_distribution: np.ndarray,
) -> float:
    """Compute one-sided p-value: P(null >= observed).

    Returns
    -------
    float
        p-value in [0, 1].
    """
    raise NotImplementedError("Stage 2: compute_permutation_pvalue")


def test_top_interactions(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    ranked_interactions: pd.DataFrame,
    n_permutations: int = config.N_PERMUTATIONS,
) -> pd.DataFrame:
    """Test all top-ranked interactions against permutation null.

    Returns
    -------
    pd.DataFrame
        ranked_interactions with added columns: null_mean, null_std,
        p_value, significant_raw (p < 0.05).
    """
    raise NotImplementedError("Stage 2: test_top_interactions")
