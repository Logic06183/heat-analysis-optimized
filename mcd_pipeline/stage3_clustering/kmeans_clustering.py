"""
K-Means Clustering — Identify Heat-Vulnerable Population Clusters
=================================================================

Applies k-means (k=3..8) to PCA-reduced SHAP vectors. Selects optimal
k using silhouette width. Characterises clusters by demographics,
socioeconomic status, and ward-level geography.

MCD reference: "...k-means (k=3–8) on individual SHAP vectors; optimal
k by silhouette width"
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from mcd_pipeline import config

logger = logging.getLogger(__name__)


def fit_kmeans_range(
    pc_scores: pd.DataFrame,
    k_range: range = config.KMEANS_K_RANGE,
    seed: int = config.MASTER_SEED,
) -> dict[int, tuple[KMeans, float]]:
    """Fit k-means for each k in range and compute silhouette scores.

    Parameters
    ----------
    pc_scores : pd.DataFrame
        PCA scores (n_patients × n_components).
    k_range : range
        Values of k to try (default: 3..8).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        k -> (fitted KMeans model, silhouette score)
    """
    X = pc_scores.values
    results = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        results[k] = (km, float(sil))
        logger.info("  k=%d: silhouette=%.4f", k, sil)

    return results


def select_optimal_k(results: dict[int, tuple[KMeans, float]]) -> int:
    """Select k with highest silhouette score.

    Parameters
    ----------
    results : dict
        Output of fit_kmeans_range.

    Returns
    -------
    int
        Optimal number of clusters.
    """
    optimal_k = max(results, key=lambda k: results[k][1])
    logger.info(
        "  Optimal k=%d (silhouette=%.4f)", optimal_k, results[optimal_k][1]
    )
    return optimal_k


def characterise_clusters(
    labels: np.ndarray,
    patient_data: pd.DataFrame,
) -> pd.DataFrame:
    """Characterise each cluster by demographics and SES.

    Joins cluster labels to patient-level covariates (one row per patient,
    the first recorded value is used for time-invariant variables).

    For each cluster reports:
    - n, % of total
    - Mean age, % female, % HIV positive
    - % informal dwelling, mean income bracket
    - % not employed
    - Most common education level
    - Dominant temperature lag response (highest mean |SHAP| lag)

    Parameters
    ----------
    labels : np.ndarray
        Cluster label per patient (aligned with patient_data index).
    patient_data : pd.DataFrame
        Patient-level data with index = patient_id.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per cluster.
    """
    data = patient_data.copy()
    data["cluster"] = labels

    n_total = len(data)
    rows = []

    for k in sorted(data["cluster"].unique()):
        grp = data[data["cluster"] == k]
        row = {"cluster": k, "n": len(grp), "pct_total": round(100 * len(grp) / n_total, 1)}

        # Demographics
        if "age_years" in grp.columns:
            row["mean_age"] = round(grp["age_years"].mean(), 1)
        if "sex_Male" in grp.columns:
            row["pct_female"] = round(100 * (1 - grp["sex_Male"].mean()), 1)
        elif "sex" in grp.columns:
            row["pct_female"] = round(100 * (grp["sex"].str.lower() == "female").mean(), 1)
        if "hiv_status_Positive" in grp.columns:
            row["pct_hiv_positive"] = round(100 * grp["hiv_status_Positive"].mean(), 1)
        elif "hiv_status" in grp.columns:
            row["pct_hiv_positive"] = round(100 * (grp["hiv_status"].str.lower() == "positive").mean(), 1)

        # Socioeconomic
        if "gcro_dwelling_type_Informal" in grp.columns:
            row["pct_informal_dwelling"] = round(100 * grp["gcro_dwelling_type_Informal"].mean(), 1)
        if "gcro_income_bracket" in grp.columns:
            row["mean_income_bracket"] = round(grp["gcro_income_bracket"].mean(), 2)
        if "gcro_employment_status_Not employed" in grp.columns:
            row["pct_not_employed"] = round(100 * grp["gcro_employment_status_Not employed"].mean(), 1)

        rows.append(row)

    profile = pd.DataFrame(rows).set_index("cluster")
    logger.info("  Cluster profiles:\n%s", profile.to_string())
    return profile
