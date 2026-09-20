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

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from mcd_pipeline import config

logger = logging.getLogger(__name__)

# Temperature lag feature names produced by feature_engineering.py
TEMP_LAG_FEATURES = [f"temp_lag_{d}d" for d in config.LAG_DAYS]


def aggregate_person_shap(
    shap_values: np.ndarray,
    patient_ids: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    """Aggregate observation-level SHAP values to person-level.

    For patients with multiple visits, takes the mean SHAP value per feature.

    Parameters
    ----------
    shap_values : np.ndarray
        Shape (n_observations, n_features).
    patient_ids : pd.Series
        Patient ID for each observation row (aligned with shap_values).
    feature_names : list[str]
        Feature names corresponding to columns of shap_values.

    Returns
    -------
    pd.DataFrame
        Index: patient_id, Columns: feature names, Values: mean SHAP.
    """
    df = pd.DataFrame(shap_values, columns=feature_names)
    df.index = patient_ids.values
    df.index.name = "patient_id"
    return df.groupby("patient_id").mean()


def fit_shap_pca(
    person_shap: pd.DataFrame,
    variance_threshold: float = config.PCA_VARIANCE_THRESHOLD,
) -> tuple[PCA, StandardScaler, pd.DataFrame]:
    """Fit PCA on person-level SHAP vectors, retaining 85% variance.

    StandardScales before PCA so features with large SHAP magnitudes
    (e.g. body composition markers) don't dominate.

    Parameters
    ----------
    person_shap : pd.DataFrame
        Person-level SHAP values (index: patient_id).
    variance_threshold : float
        Cumulative variance to retain (default: 0.85).

    Returns
    -------
    tuple of (fitted PCA, fitted StandardScaler, PC scores DataFrame)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(person_shap)

    pca = PCA(n_components=variance_threshold, svd_solver="full", random_state=config.MASTER_SEED)
    pc_scores = pca.fit_transform(X_scaled)

    n_components = pca.n_components_
    logger.info(
        "  PCA: %d components retain %.1f%% variance (threshold %.0f%%)",
        n_components, pca.explained_variance_ratio_.sum() * 100, variance_threshold * 100,
    )

    pc_df = pd.DataFrame(
        pc_scores,
        index=person_shap.index,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    return pca, scaler, pc_df


def multi_biomarker_pca(
    stage1_dir: Path,
    biomarker_names: list[str],
    df: pd.DataFrame,
) -> tuple[PCA, StandardScaler, pd.DataFrame, list[str]]:
    """Concatenate temperature-lag SHAP vectors across biomarkers, then PCA.

    Each person gets a concatenated vector of their temperature-lag SHAP
    profiles across all analysed biomarkers — a multi-system heat fingerprint.

    Only temperature lag features are used (7 lag windows × n_biomarkers).
    This keeps clustering focused on differential heat sensitivity rather
    than confounders like age, sex, or study source.

    Parameters
    ----------
    stage1_dir : Path
        Directory containing per-biomarker Stage 1 outputs.
    biomarker_names : list[str]
        Eligible biomarker names (R² > 0, model.ubj exists).
    df : pd.DataFrame
        Analysis dataset from Stage 0 (needed to recover patient IDs).

    Returns
    -------
    tuple of (fitted PCA, fitted StandardScaler, PC scores DataFrame,
              list of biomarkers successfully included)
    """
    from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers
    from mcd_pipeline.stage1_lag_profiling.xgboost_trainer import prepare_features_and_target

    biomarkers_dict = get_available_biomarkers()
    per_biomarker: dict[str, pd.DataFrame] = {}

    for bio_name in biomarker_names:
        shap_path = stage1_dir / bio_name / "shap_values.npy"
        names_path = stage1_dir / bio_name / "feature_names.json"

        if not shap_path.exists() or not names_path.exists():
            logger.warning("  Skipping %s: missing shap_values.npy or feature_names.json", bio_name)
            continue

        spec = biomarkers_dict.get(bio_name)
        if spec is None:
            logger.warning("  Skipping %s: not in biomarker registry", bio_name)
            continue

        try:
            X, _, groups = prepare_features_and_target(df, spec)
        except Exception as e:
            logger.warning("  Skipping %s: prepare_features_and_target failed: %s", bio_name, e)
            continue

        shap_vals = np.load(shap_path)
        feature_names = json.load(open(names_path))

        if shap_vals.shape[0] != len(X):
            logger.warning(
                "  Skipping %s: shap_values rows (%d) != X rows (%d)",
                bio_name, shap_vals.shape[0], len(X),
            )
            continue

        # Filter to temperature lag features only
        temp_idx = [i for i, f in enumerate(feature_names) if f in TEMP_LAG_FEATURES]
        if not temp_idx:
            logger.warning("  Skipping %s: no temperature lag features found", bio_name)
            continue

        temp_names = [feature_names[i] for i in temp_idx]
        shap_temp = shap_vals[:, temp_idx]

        # Aggregate to person level
        person_df = aggregate_person_shap(
            shap_temp, groups.reset_index(drop=True), temp_names
        )
        # Prefix columns with biomarker name to avoid clashes
        person_df.columns = [f"{bio_name}__{c}" for c in person_df.columns]
        per_biomarker[bio_name] = person_df

        logger.info("  %s: %d patients, %d temp features", bio_name, len(person_df), len(temp_idx))

    if not per_biomarker:
        raise RuntimeError("No biomarkers available for Stage 3 PCA.")

    # Concatenate across biomarkers; patients missing a biomarker get 0
    combined = pd.concat(list(per_biomarker.values()), axis=1, join="outer").fillna(0.0)
    logger.info(
        "  Combined SHAP matrix: %d patients × %d features (%d biomarkers)",
        len(combined), combined.shape[1], len(per_biomarker),
    )

    pca, scaler, pc_df = fit_shap_pca(combined)
    return pca, scaler, pc_df, list(per_biomarker.keys())
