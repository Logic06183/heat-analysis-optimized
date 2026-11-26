"""02. How much does temperature add to each biomarker model? (appendix SA12)

Question    Reviewers 2 and 3 and the statistical reviewer asked how much of
            each model's cross-validated R-squared comes from temperature.
Data in     The visit-level analysis table. All 24 screened biomarkers.
Steps       For each biomarker:
            1. Fit the published model (temperature + covariates) in five
               participant-grouped folds.
            2. Fit the same model without the temperature columns, same folds.
            3. Incremental R-squared = mean over folds of (full - covariates only).
            4. 95% intervals for the full model's R-squared and for the
               increment, from 1,000 participant-level bootstrap resamples
               of the out-of-fold predictions.
            Then repeat for HIV viral load as a yes/no outcome (1,000 copies
            per mL or more), scored by AUC.
Files out   r1_7a_incremental_r2.csv        one row per biomarker
            r1_7d_viral_suppression.json    the viraemia classifier
            r1_7a_7d_run_info.json          software versions and settings
Reported    Results (incremental R-squared, viraemia AUC); appendix SA12 and
            its table, which also gives n participants and observations.

`r2_published` is read from the published Stage 1 output so the reader can
confirm that `r2_full` reproduces it.
"""

import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, roc_auc_score

from r1_common import (
    OUTPUT_DIR,
    STAGE1_DIR,
    TEMPERATURE_COLUMNS,
    VIRAEMIA_THRESHOLD,
    bootstrap_intervals,
    config,
    cross_validate,
    load_visits,
    prepare_features_and_target,
    without_temperature,
    write_json,
)
from mcd_pipeline.stage0_data_prep.biomarker_definitions import MIN_SAMPLE_SIZE

N_BOOT = int(os.environ.get("R1_NBOOT", "1000"))

# One random generator for the whole script, so the bootstrap draws follow on
# from one biomarker to the next and the run is reproducible.
rng = np.random.default_rng(config.BOOTSTRAP_RANDOM_SEED)

visits, biomarkers = load_visits()


def published_r2(column):
    path = STAGE1_DIR / str(column) / "cv_metrics.json"
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)["mean_r2"]


# ---- Incremental R-squared, every screened biomarker ----------------------

rows = []
for biomarker in biomarkers.values():
    column = biomarker.column
    if not column or column not in visits.columns:
        continue
    if visits[column].notna().sum() < MIN_SAMPLE_SIZE:
        continue

    X, y, participants = prepare_features_and_target(visits, biomarker)
    X_covariates_only = without_temperature(X)

    predictions_full, r2_full = cross_validate(X, y, participants)
    predictions_cov, r2_cov = cross_validate(X_covariates_only, y, participants)
    fold_differences = r2_full - r2_cov
    intervals = bootstrap_intervals(y, predictions_full, predictions_cov, participants, r2_score, N_BOOT, rng)

    rows.append(
        {
            "biomarker": column,
            "n_obs": len(y),
            "n_participants": participants.nunique(),
            "n_temp_features": X.shape[1] - X_covariates_only.shape[1],
            "r2_published": published_r2(column),
            "r2_full": float(r2_full.mean()),
            "r2_full_lo": intervals["score_a"][0],
            "r2_full_hi": intervals["score_a"][1],
            "r2_covariates_only": float(r2_cov.mean()),
            "delta_r2": float(fold_differences.mean()),
            "delta_r2_lo": intervals["difference"][0],
            "delta_r2_hi": intervals["difference"][1],
            "folds_temperature_better": int((fold_differences > 0).sum()),
            "fold_delta": ";".join(f"{d:.4f}" for d in fold_differences),
        }
    )
    # Written after every biomarker so a stopped run keeps what it finished.
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "r1_7a_incremental_r2.csv", index=False)
    print(rows[-1], flush=True)


# ---- HIV viral load as viraemic or not ------------------------------------

viral_load = biomarkers["viral_load"]
X, y_continuous, participants = prepare_features_and_target(visits, viral_load)
raw_copies = visits.loc[visits[viral_load.column].notna(), viral_load.column].astype(float).values
assert len(raw_copies) == len(y_continuous), "viral load rows must match the model's rows"
viraemic = pd.Series((raw_copies >= VIRAEMIA_THRESHOLD).astype(int), index=y_continuous.index)

predictions_full, auc_full = cross_validate(X, viraemic, participants, classifier=True)
predictions_cov, auc_cov = cross_validate(without_temperature(X), viraemic, participants, classifier=True)
intervals = bootstrap_intervals(
    viraemic, predictions_full, predictions_cov, participants, roc_auc_score, N_BOOT, rng
)

write_json(
    {
        "n_obs": int(len(viraemic)),
        "n_participants": int(participants.nunique()),
        "prevalence_viraemic": float(viraemic.mean()),
        "auc_full": float(np.nanmean(auc_full)),
        "auc_covariates_only": float(np.nanmean(auc_cov)),
        "delta_auc": float(np.nanmean(auc_full - auc_cov)),
        "delta_auc_ci": intervals["difference"],
        "auc_full_ci": intervals["score_a"],
        "fold_auc_full": auc_full.tolist(),
        "fold_auc_cov": auc_cov.tolist(),
    },
    "r1_7d_viral_suppression.json",
)

write_json(
    {
        "xgboost": xgb.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "n_boot": N_BOOT,
        "seed": config.BOOTSTRAP_RANDOM_SEED,
        "exposure": config.PRIMARY_EXPOSURE_SOURCE,
        "temperature_columns": TEMPERATURE_COLUMNS,
        "finished": time.ctime(),
    },
    "r1_7a_7d_run_info.json",
)
