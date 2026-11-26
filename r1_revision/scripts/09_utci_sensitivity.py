"""09. Repeat the screen with UTCI in place of air temperature (appendix SA16)

Question    Do the ten retained biomarkers give the same answer when exposure
            is the Universal Thermal Climate Index?
Data in     The visit-level analysis table; utci/utci_daily_cells.csv (script 08).
Steps       1. Attach UTCI to every visit at the same lags as air temperature
               (single days 0, 1, 3, 7, 14, 21; mean of the previous 30 days),
               and the daily UTCI range in place of the diurnal temperature range.
            2. For each retained biomarker fit three models on the same folds:
               air temperature, UTCI, covariates only.
            3. Report cross-validated R-squared for each, the increment over
               covariates only, and a bootstrap interval for the UTCI increment.
            4. Compare which lags matter: Spearman correlation between the
               air-temperature and UTCI lag profiles (mean absolute SHAP per lag).
Files out   r1_sa16_utci.csv        one row per biomarker
            r1_sa16_utci_info.json  visit coverage, UTCI/air correlation, source
Reported    SA16 and its table; one Results sentence.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from r1_common import (
    OUTPUT_DIR,
    REPO,
    RETAINED,
    SEED,
    bootstrap_intervals,
    config,
    cross_validate,
    load_visits,
    prepare_features_and_target,
    without_temperature,
    write_json,
)

UTCI_DIR = Path(os.environ.get("R1_UTCI_DIR", Path(__file__).resolve().parents[1] / "utci"))
GRID = 0.25
SINGLE_DAY_LAGS = [0, 1, 3, 7, 14, 21]
N_BOOT = 1000
VISIT_KEY = ["study_source", "patient_id", "visit_date"]

rng = np.random.default_rng(SEED)
visits, biomarkers = load_visits()


# ---- 1. UTCI at every visit ------------------------------------------------

# Where each visit's exposure was assigned, snapped to the UTCI grid.
linkage = pd.read_csv(
    REPO / "FINAL_DATASETS/CLIMATE_HEALTH_LINKAGE/ERA5_LAND_LINKAGE.csv",
    usecols=VISIT_KEY + ["latitude", "longitude"],
)
linkage["visit_date"] = pd.to_datetime(linkage["visit_date"]).dt.floor("D")
linkage["cell_lat"] = (linkage["latitude"] / GRID).round() * GRID
linkage["cell_lon"] = (linkage["longitude"] / GRID).round() * GRID

# Daily UTCI per grid cell, with the daily range and the trailing 30-day mean.
daily = pd.read_csv(UTCI_DIR / "utci_daily_cells.csv", parse_dates=["date"])
daily = daily.sort_values(["cell_lat", "cell_lon", "date"])
daily["utci_range"] = daily["utci_max"] - daily["utci_min"]
daily["utci_30day_mean"] = daily.groupby(["cell_lat", "cell_lon"])["utci_mean"].transform(
    lambda series: series.rolling(30, min_periods=25).mean()
)
daily = daily.set_index(["cell_lat", "cell_lon", "date"])


def utci_on(column, days_before):
    """Look up a daily UTCI column for each visit's cell, `days_before` the visit."""
    date = linkage["visit_date"] - pd.Timedelta(days=days_before)
    key = pd.MultiIndex.from_arrays([linkage["cell_lat"], linkage["cell_lon"], date])
    return daily[column].reindex(key).values


utci = linkage[VISIT_KEY].copy()
for lag in SINGLE_DAY_LAGS:
    utci[f"temp_lag_{lag}d"] = utci_on("utci_mean", lag)
utci["temp_lag_30d"] = utci_on("utci_30day_mean", 0)
utci[config.DTR_COLUMN] = utci_on("utci_range", 0)
utci = utci.drop_duplicates(VISIT_KEY)

# One UTCI row per visit, in the same order as `visits`.
visit_keys = visits[["study_source", "patient_id"]].copy()
visit_keys["visit_date"] = pd.to_datetime(visits["visit_date"]).dt.floor("D")
utci_by_visit = visit_keys.merge(utci, on=VISIT_KEY, how="left")
assert len(utci_by_visit) == len(visits), "one UTCI row per visit"
utci_columns = [c for c in utci_by_visit.columns if c not in VISIT_KEY]
coverage_by_column = {c: float(utci_by_visit[c].notna().mean()) for c in utci_columns}
coverage = coverage_by_column["temp_lag_0d"]
correlation_with_air = float(
    pd.Series(utci_by_visit["temp_lag_7d"].values).corr(pd.Series(visits["temp_lag_7d"].values))
)
print(
    "visits with UTCI:",
    round(coverage, 4),
    "| correlation with air temperature (lag 7):",
    correlation_with_air,
)

# The same visit table with UTCI written over the temperature columns. From here on, in
# `visits_with_utci` and `X_utci`, the columns named temp_lag_* and the daily range hold UTCI.
visits_with_utci = visits.copy()
for lag in config.LAG_DAYS:
    visits_with_utci[f"temp_lag_{lag}d"] = utci_by_visit[f"temp_lag_{lag}d"].values
visits_with_utci[config.DTR_COLUMN] = utci_by_visit[config.DTR_COLUMN].values


# ---- 2 to 4. Models ---------------------------------------------------------


def lag_profile(X, y):
    """Mean absolute SHAP value of each temperature lag, from one fit on all rows."""
    model = xgb.XGBRegressor(**config.XGBOOST_PARAMS)
    model.fit(X, y, verbose=False)
    contributions = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    columns = list(X.columns)
    return np.array(
        [np.abs(contributions[:, columns.index(f"temp_lag_{lag}d")]).mean() for lag in config.LAG_DAYS]
    )


rows = []
for name in RETAINED:
    X_air, y, participants = prepare_features_and_target(visits, biomarkers[name])
    X_utci, y_utci, participants_utci = prepare_features_and_target(visits_with_utci, biomarkers[name])

    predictions_air, r2_air = cross_validate(X_air, y, participants)
    predictions_utci, r2_utci = cross_validate(X_utci, y_utci, participants_utci)
    predictions_cov, r2_cov = cross_validate(without_temperature(X_air), y, participants)

    same_lags_matter = spearmanr(lag_profile(X_air, y), lag_profile(X_utci, y_utci)).correlation
    intervals = bootstrap_intervals(y, predictions_utci, predictions_cov, participants, r2_score, N_BOOT, rng)

    rows.append(
        {
            "biomarker": name,
            "n_obs": len(y),
            "r2_air": float(r2_air.mean()),
            "r2_utci": float(r2_utci.mean()),
            "r2_cov": float(r2_cov.mean()),
            "delta_air": float((r2_air - r2_cov).mean()),
            "delta_utci": float((r2_utci - r2_cov).mean()),
            "delta_utci_ci": intervals["difference"],
            "lag_profile_rho": float(same_lags_matter),
        }
    )
    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "r1_sa16_utci.csv", index=False)

write_json(
    {
        "coverage": coverage,
        "coverage_by_utci_column": coverage_by_column,
        "corr_utci7_air7": correlation_with_air,
        "source": "CDS derived-utci-historical-timeseries v1.1 consolidated, accessed 2026-11-10",
        "finished": time.ctime(),
    },
    "r1_sa16_utci_info.json",
)
