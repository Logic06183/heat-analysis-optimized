"""
K-Means Clustering — Identify Heat-Vulnerable Population Clusters
=================================================================

Applies k-means (k=3..8) to PCA-reduced SHAP vectors. Selects optimal
k using silhouette width. Characterises clusters by demographics,
socioeconomic status, and ward-level geography.

MCD reference: "...k-means (k=3–8) on individual SHAP vectors; optimal
k by silhouette width"
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from mcd_pipeline import config


def fit_kmeans_range(
    pc_scores: pd.DataFrame,
    k_range: range = config.KMEANS_K_RANGE,
    seed: int = config.MASTER_SEED,
) -> dict[int, tuple[KMeans, float]]:
    """Fit k-means for each k in range and compute silhouette scores.

    Returns
    -------
    dict
        k -> (fitted KMeans model, silhouette score)
    """
    raise NotImplementedError("Stage 3: fit_kmeans_range")


def select_optimal_k(results: dict[int, tuple[KMeans, float]]) -> int:
    """Select k with highest silhouette score.

    Returns
    -------
    int
        Optimal number of clusters.
    """
    raise NotImplementedError("Stage 3: select_optimal_k")


def characterise_clusters(
    labels: np.ndarray,
    patient_data: pd.DataFrame,
) -> pd.DataFrame:
    """Characterise each cluster by demographics and SES.

    For each cluster reports:
    - Mean age, sex distribution, HIV prevalence
    - Dwelling type distribution, income, education
    - Geographic distribution (ward-level if available)
    - Dominant biomarker response patterns

    Returns
    -------
    pd.DataFrame
        Summary table with one row per cluster.
    """
    raise NotImplementedError("Stage 3: characterise_clusters")
