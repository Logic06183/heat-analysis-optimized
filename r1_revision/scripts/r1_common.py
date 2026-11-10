"""Shared helpers for the R1 revision analyses.

Every numbered script in this folder imports from here, so the steps that
repeat (load the data, fit the Stage 1 model with cross-validation, bootstrap
a difference, build the person-level SHAP profile, cluster it, permute a
label) are written once and can be checked once.

Nothing here changes the main pipeline. `config`, `build_analysis_dataset` and
`prepare_features_and_target` are the pipeline's own; they are imported here once and the scripts
take them from this module.

The model settings, folds, lags and
seeds all come from `mcd_pipeline.config`, the same file the published
analysis used.

Paths
-----
REPO        repository root (override with the R1_REPO_ROOT environment variable)
OUTPUT_DIR  where aggregate results are written (override with R1_OUTPUT_DIR)
STAGE1_DIR  published Stage 1 outputs (SHAP values per biomarker)
STAGE3_DIR  published Stage 3 outputs (cluster labels)
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# The paper's primary exposure is ERA5-Land. The pipeline reads this switch.
os.environ.setdefault("RP2_PRIMARY_EXPOSURE", "era5_land")

REPO = Path(os.environ.get("R1_REPO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO))
logging.basicConfig(level=logging.ERROR)

from mcd_pipeline import config  # noqa: E402
from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers  # noqa: E402
from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset  # noqa: E402
from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features  # noqa: E402
from mcd_pipeline.stage1_lag_profiling.xgboost_trainer import prepare_features_and_target  # noqa: E402

OUTPUT_DIR = Path(os.environ.get("R1_OUTPUT_DIR", Path(__file__).resolve().parents[1] / "outputs"))
OUTPUT_DIR.mkdir(exist_ok=True)
STAGE1_DIR = config.OUTPUT_ROOT / "stage1"
STAGE3_DIR = config.OUTPUT_ROOT / "stage3_r2_005"

PARTICIPANT = "patient_id"
SEED = 42
N_PERMUTATIONS = 10000
VIRAEMIA_THRESHOLD = 1000  # HIV RNA copies per mL, the WHO threshold for virological failure

# The ten biomarkers that passed the Stage 1 screen (cross-validated R-squared of 0.05 or more).
RETAINED = [
    "creatinine",
    "diastolic_bp",
    "heart_rate",
    "cd4_count",
    "viral_load",
    "ldl_cholesterol",
    "waist_hip_ratio",
    "body_fat_percent",
    "hemoglobin",
    "hematocrit",
]

# Temperature features in every Stage 1 model: one column per lag, plus the diurnal temperature range.
LAG_COLUMNS = [f"temp_lag_{days}d" for days in config.LAG_DAYS]
TEMPERATURE_COLUMNS = LAG_COLUMNS + [config.DTR_COLUMN]

# Stage 3 settings, as published.
PCA_VARIANCE_KEPT = 0.85
PCA_DIMENSIONS_CLUSTERED = 10


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_visits():
    """Return the visit-level analysis table and the biomarker definitions.

    One row per participant-visit, with the temperature lags and covariates
    built exactly as in the main pipeline.
    """
    visits = build_analysis_dataset()
    biomarkers = get_available_biomarkers()
    biomarker_columns = [b.column for b in biomarkers.values() if b.column and b.column in visits.columns]
    visits = engineer_features(visits, biomarker_columns)
    return visits, biomarkers


def cohort_of_each_participant(visits):
    """The cohort each participant belongs to (their most frequent study_source)."""
    return visits.groupby(PARTICIPANT)["study_source"].agg(lambda s: s.mode().iat[0]).astype(str)


def published_clusters():
    """Published Stage 3 result: cluster label and HIV status per participant.

    Returns (labels, hiv_positive, cluster3_label). `cluster3_label` is the
    label of the cluster with the highest HIV prevalence, which the paper
    calls Cluster 3.
    """
    labels = pd.read_csv(STAGE3_DIR / "cluster_labels.csv").set_index(PARTICIPANT)["cluster"]
    table = pd.read_csv(STAGE3_DIR / "patient_cluster_table.csv").set_index(PARTICIPANT)
    hiv_positive = table["hiv_status"].astype(str).str.lower().eq("positive").astype(int)
    prevalence = hiv_positive.reindex(labels.index).groupby(labels).mean()
    cluster3_label = int(prevalence.idxmax())
    return labels, hiv_positive, cluster3_label


# --------------------------------------------------------------------------
# Stage 1 model: cross-validation with and without temperature
# --------------------------------------------------------------------------


def without_temperature(X):
    """The same feature table with the temperature columns removed."""
    present = [c for c in TEMPERATURE_COLUMNS if c in X.columns]
    return X.drop(columns=present)


def cross_validate(X, y, participants, classifier=False):
    """Fit the Stage 1 XGBoost model in participant-grouped folds.

    All of a participant's visits fall in the same fold, so the score is for
    people the model has not seen. Returns the out-of-fold prediction for
    every row and one score per fold (R-squared, or AUC for a classifier).
    """
    predictions = np.full(len(y), np.nan)
    fold_scores = []
    folds = GroupKFold(n_splits=config.CV_FOLDS)
    for train, test in folds.split(X, y, participants):
        if classifier:
            model = xgb.XGBClassifier(**config.XGBOOST_PARAMS, eval_metric="logloss")
            model.fit(X.iloc[train], y.iloc[train], verbose=False)
            predicted = model.predict_proba(X.iloc[test])[:, 1]
            both_classes_present = y.iloc[test].nunique() > 1
            score = roc_auc_score(y.iloc[test], predicted) if both_classes_present else np.nan
        else:
            model = xgb.XGBRegressor(**config.XGBOOST_PARAMS)
            # eval_set is passed exactly as in the published trainer. The settings have no early
            # stopping, so the test fold does not influence the fit.
            model.fit(X.iloc[train], y.iloc[train], eval_set=[(X.iloc[test], y.iloc[test])], verbose=False)
            predicted = model.predict(X.iloc[test])
            score = r2_score(y.iloc[test], predicted)
        predictions[test] = predicted
        fold_scores.append(score)
    return predictions, np.array(fold_scores)


def bootstrap_intervals(y, predictions_a, predictions_b, participants, score, n_boot, rng):
    """Bootstrap 95% intervals for model A's score and for score(A) minus score(B).

    Participants are resampled with replacement (all of a person's visits
    stay together) and both models are scored on the same resample.
    Returns {"score_a": [low, high], "difference": [low, high], "n_resamples_used": n}.

    The intervals are for scores computed on all out-of-fold predictions pooled together.
    The point estimates in the output tables are means over the five folds, so an interval
    need not be centred exactly on its point estimate.
    """
    participants = pd.Series(participants.values)
    unique_participants = participants.unique()
    rows_of = participants.groupby(participants).indices
    y = np.asarray(y)
    scores_a = []
    differences = []
    for _ in range(n_boot):
        drawn = rng.choice(unique_participants, size=len(unique_participants), replace=True)
        rows = np.concatenate([rows_of[p] for p in drawn])
        try:
            score_a = score(y[rows], predictions_a[rows])
            score_b = score(y[rows], predictions_b[rows])
        except ValueError:
            # AUC is undefined when a resample has only one outcome class. Skip it.
            continue
        scores_a.append(score_a)
        differences.append(score_a - score_b)
    return {
        "score_a": [float(v) for v in np.percentile(scores_a, [2.5, 97.5])],
        "difference": [float(v) for v in np.percentile(differences, [2.5, 97.5])],
        "n_resamples_used": len(differences),
    }


# --------------------------------------------------------------------------
# Stage 3: person-level SHAP profile, clustering, permutation
# --------------------------------------------------------------------------


def person_level_temperature_shap(visits, biomarkers, name):
    """One row per participant: their mean temperature-lag SHAP values for one biomarker.

    Reads the published Stage 1 SHAP values (one row per visit), keeps the
    temperature-lag columns and averages each participant's visits. Also
    returns the number of visits each participant contributed.
    """
    shap_values = np.load(STAGE1_DIR / name / "shap_values.npy")
    with open(STAGE1_DIR / name / "feature_names.json") as handle:
        feature_names = json.load(handle)
    X, _, participants = prepare_features_and_target(visits, biomarkers[name])
    assert shap_values.shape[0] == len(X), "SHAP rows must match the model's rows"

    lag_positions = [i for i, feature in enumerate(feature_names) if feature in LAG_COLUMNS]
    per_visit = pd.DataFrame(
        shap_values[:, lag_positions],
        columns=[f"{name}__{feature_names[i]}" for i in lag_positions],
        index=pd.Index(participants.reset_index(drop=True).values, name=PARTICIPANT),
    )
    visits_per_participant = per_visit.groupby(PARTICIPANT).size()
    per_person = per_visit.groupby(PARTICIPANT).mean()
    return per_person, visits_per_participant


def cluster_profiles(profiles, k=3, standardise=True, seed=SEED, max_iter=300):
    """Stage 3 clustering, as published: standardise, PCA, Gaussian mixture.

    PCA keeps the components that explain 85% of the variance and the first
    ten of those are clustered. Returns a dict with the labels, the clustered
    components, the fitted mixture, the number of components at 85% and the
    share of variance carried by the clustered components.
    """
    if standardise:
        values = StandardScaler().fit_transform(profiles)
    else:
        values = np.asarray(profiles, float)
    pca = PCA(n_components=PCA_VARIANCE_KEPT, svd_solver="full")
    components = pca.fit_transform(values)
    n_used = min(PCA_DIMENSIONS_CLUSTERED, components.shape[1])
    components = components[:, :n_used]
    mixture = GaussianMixture(
        n_components=k, covariance_type="full", random_state=seed, n_init=5, max_iter=max_iter
    ).fit(components)
    return {
        "labels": pd.Series(mixture.predict(components), index=profiles.index),
        "components": components,
        "mixture": mixture,
        "n_components_at_85pct": int(pca.n_components_),
        "variance_in_clustered_components": float(pca.explained_variance_ratio_[:n_used].sum()),
    }


def highest_prevalence_permutation_test(labels, is_positive, n_permutations, rng):
    """Is the highest prevalence of a binary trait across clusters more than chance?

    Shuffles the trait across participants `n_permutations` times, keeping
    the clusters fixed, and records the highest cluster prevalence each time.
    The p value is upper-tail, (1 + number of shuffles at or above the observed value) / (1 + shuffles),
    so it is never exactly zero.
    Returns (observed highest prevalence, p value, largest value under shuffling).
    """
    cluster_codes = pd.factorize(np.asarray(labels))[0]
    cluster_sizes = np.bincount(cluster_codes)
    is_positive = np.asarray(is_positive)

    def highest_prevalence(trait):
        return (np.bincount(cluster_codes, weights=trait, minlength=len(cluster_sizes)) / cluster_sizes).max()

    observed = highest_prevalence(is_positive)
    shuffled = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled[i] = highest_prevalence(rng.permutation(is_positive))
    p_value = (1 + (shuffled >= observed).sum()) / (1 + n_permutations)
    return float(observed), float(p_value), float(shuffled.max())


def write_json(result, filename):
    with open(OUTPUT_DIR / filename, "w") as handle:
        json.dump(result, handle, indent=2, default=str)
