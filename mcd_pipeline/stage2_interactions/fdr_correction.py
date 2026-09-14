"""
FDR Correction — Benjamini-Hochberg for Interaction P-Values
=============================================================

Applies Benjamini-Hochberg false discovery rate correction at q=0.10
to the permutation p-values from the top 50 interactions per biomarker.

MCD reference: "Benjamini-Hochberg FDR q=0.10 applied to permutation
p-values for top 50 interactions"
"""

import logging

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


def benjamini_hochberg(
    p_values: np.ndarray,
    q: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : np.ndarray
        Raw p-values.
    q : float
        FDR threshold (default: 0.10).

    Returns
    -------
    tuple of (rejected: bool array, adjusted_p: float array)
    """
    p_values = np.asarray(p_values, dtype=float)

    if len(p_values) == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    rejected, adjusted, _, _ = multipletests(p_values, alpha=q, method="fdr_bh")

    n_sig = rejected.sum()
    logger.info(f"BH FDR: {n_sig}/{len(p_values)} rejected at q={q}")

    return rejected, adjusted


def apply_fdr_to_interactions(
    interaction_results: pd.DataFrame,
    q: float = 0.10,
) -> pd.DataFrame:
    """Apply BH FDR to interaction test results.

    Parameters
    ----------
    interaction_results : pd.DataFrame
        Must have 'p_value' column from permutation testing.
    q : float
        FDR threshold (default: 0.10).

    Returns
    -------
    pd.DataFrame
        Copy with added columns: p_adjusted, significant_fdr.

    Raises
    ------
    ValueError
        If 'p_value' column is missing.
    """
    if "p_value" not in interaction_results.columns:
        raise ValueError("interaction_results must have a 'p_value' column")

    result = interaction_results.copy()
    rejected, adjusted = benjamini_hochberg(result["p_value"].values, q=q)
    result["p_adjusted"] = adjusted
    result["significant_fdr"] = rejected

    return result
