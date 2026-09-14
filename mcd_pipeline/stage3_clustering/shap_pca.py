"""
SHAP PCA — Dimensionality Reduction on Individual SHAP Vectors
===============================================================

Applies PCA to individual-level SHAP value vectors (one per person per
biomarker) to reduce dimensionality while retaining 85% of variance.
The resulting PC scores become the input for vulnerability clustering.

The SHAP vectors represent each person's unique pattern of heat
sensitivity across lag windows and covariates — a 'heat fingerprint'.

MCD reference: "Stage 3 — Vulnerability clustering: PCA (85% variance
retained) then k-means..."
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from mcd_pipeline import config


def aggregate_person_shap(
    shap_values: np.ndarray,
    patient_ids: pd.Series,
) -> pd.DataFrame:
    """Aggregate observation-level SHAP values to person-level.

    For patients with multiple visits, takes the mean SHAP value per feature.

    Returns
    -------
    pd.DataFrame
        Index: patient_id, Columns: feature names, Values: mean |SHAP|.
    """
    raise NotImplementedError("Stage 3: aggregate_person_shap")


def fit_shap_pca(
    person_shap: pd.DataFrame,
    variance_threshold: float = config.PCA_VARIANCE_THRESHOLD,
) -> tuple[PCA, pd.DataFrame]:
    """Fit PCA on person-level SHAP vectors, retaining 85% variance.

    Parameters
    ----------
    person_shap : pd.DataFrame
        Person-level mean SHAP values.
    variance_threshold : float
        Cumulative variance to retain.

    Returns
    -------
    tuple of (fitted PCA, PC scores DataFrame with patient_id index)
    """
    raise NotImplementedError("Stage 3: fit_shap_pca")


def multi_biomarker_pca(
    stage1_dir: Path,
    biomarker_names: list[str],
) -> tuple[PCA, pd.DataFrame]:
    """Concatenate SHAP vectors across biomarkers, then PCA.

    Each person gets a concatenated vector of their SHAP profiles
    across all analysed biomarkers — a multi-system heat fingerprint.

    Returns
    -------
    tuple of (fitted PCA, PC scores DataFrame)
    """
    raise NotImplementedError("Stage 3: multi_biomarker_pca")
