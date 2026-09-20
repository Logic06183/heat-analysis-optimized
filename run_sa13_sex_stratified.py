#!/usr/bin/env python3
"""SA13: sex-stratified replication of Stage 1 lag profiling.

Holds Stage 1 fixed in every respect except one: the model is fitted
within each sex stratum rather than once on the pooled sample with sex
in the adjustment set. Same a priori hyperparameters
(config.XGBOOST_PARAMS), same participant-grouped 5-fold CV, same
interventional TreeSHAP, same R^2 >= 0.05 retention rule. Sex is dropped
from the covariates since it is now the stratifier.

Per biomarker and per stratum this reports: mean cross-validated R^2
with a cluster-bootstrap 95% CI (patients resampled over pooled
out-of-fold predictions — no refitting), participant and observation
counts, whether the biomarker clears the retention cut, and the peak lag
from the SHAP profile, compared against the primary (pooled) peak lag.

Strata with fewer than MIN_OBS observations (or fewer patients than CV
folds) are reported as not estimable rather than fitted.
"""
from __future__ import annotations
import json, logging, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

logging.basicConfig(level=logging.ERROR)

REPO = Path(__file__).resolve().parent
CACHE = REPO / "_nestedcv_engineered.pkl"

os.environ.setdefault("RP2_PRIMARY_EXPOSURE", "era5_land")
sys.path.insert(0, str(REPO))

from mcd_pipeline import config                                    # noqa: E402
from mcd_pipeline.stage0_data_prep.biomarker_definitions import (  # noqa: E402
    get_available_biomarkers,
)
from mcd_pipeline.stage1_lag_profiling.xgboost_trainer import (    # noqa: E402
    prepare_features_and_target,
)
from mcd_pipeline.stage1_lag_profiling.shap_profiler import (      # noqa: E402
    compute_shap_values,
)
from mcd_pipeline.stage1_lag_profiling.lag_response import (       # noqa: E402
    extract_lag_shap_profile,
    summarise_lag_response,
)

OUT = config.OUTPUT_ROOT / "sensitivity" / "sa13_sex_stratified"
OUT.mkdir(parents=True, exist_ok=True)

STRATA = ["Female", "Male"]
MIN_OBS = 100
R2_CUT = 0.05
N_CI_BOOT = 1000


def log(m): print(f"[sa13] {m}", flush=True)


def _params():
    p = dict(config.XGBOOST_PARAMS)
    p["n_jobs"] = -1
    return p


def _primary_peak_lag(biomarker: str) -> int | None:
    f = config.OUTPUT_ROOT / "stage1" / biomarker / "lag_response_summary.csv"
    if not f.exists():
        return None
    s = pd.read_csv(f)
    return int(s.loc[s["mean_abs_shap"].idxmax(), "lag_day"])


def _primary_mean_r2(biomarker: str) -> float | None:
    f = config.OUTPUT_ROOT / "stage1" / biomarker / "cv_metrics.json"
    if not f.exists():
        return None
    return float(json.loads(f.read_text())["mean_r2"])


def _oof_r2_ci(y, oof_pred, groups, seed=config.BOOTSTRAP_RANDOM_SEED):
    """Cluster-bootstrap 95% CI for cross-validated R^2.

    Resamples patients (not observations) over the pooled out-of-fold
    predictions, so the interval respects the grouping structure without
    any refitting.
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y, dtype=float)
    oof_pred = np.asarray(oof_pred, dtype=float)
    codes, _ = pd.factorize(groups)
    by_patient: dict[int, np.ndarray] = {}
    for i, c in enumerate(codes):
        by_patient.setdefault(c, []).append(i)
    ids = list(by_patient)
    stats = []
    for _ in range(N_CI_BOOT):
        take = rng.choice(len(ids), size=len(ids), replace=True)
        idx = np.concatenate([by_patient[ids[t]] for t in take])
        if np.var(y[idx]) == 0:
            continue
        stats.append(r2_score(y[idx], oof_pred[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def run_stratum(name, spec, df_s, stratum):
    """Fit Stage 1 within one sex stratum. Returns a result dict."""
    n_target = int(df_s[spec.column].notna().sum())
    if n_target < MIN_OBS:
        return {"biomarker": name, "stratum": stratum, "status": "not_estimable",
                "reason": f"{n_target} observations < {MIN_OBS}", "n_obs": n_target}

    X, y, groups = prepare_features_and_target(df_s, spec)
    if groups.nunique() < config.CV_FOLDS:
        return {"biomarker": name, "stratum": stratum, "status": "not_estimable",
                "reason": f"{groups.nunique()} patients < {config.CV_FOLDS} folds",
                "n_obs": int(len(X))}

    gkf = GroupKFold(n_splits=config.CV_FOLDS)
    fold_r2 = []
    oof = pd.Series(index=X.index, dtype=float)
    for tr, te in gkf.split(X, y, groups):
        m = xgb.XGBRegressor(**_params())
        m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
        pred = m.predict(X.iloc[te])
        oof.iloc[te] = pred
        fold_r2.append(float(r2_score(y.iloc[te], pred)))

    mean_r2 = float(np.mean(fold_r2))
    ci_lo, ci_hi = _oof_r2_ci(y, oof, groups)

    # Final refit on the full stratum for SHAP, mirroring train_single_model
    final = xgb.XGBRegressor(**_params())
    final.fit(X, y, verbose=False)
    shap_vals = compute_shap_values(final, X)
    profile = extract_lag_shap_profile(shap_vals, X.columns.tolist())
    lag_summary = summarise_lag_response(profile)
    peak = int(lag_summary.loc[lag_summary["mean_abs_shap"].idxmax(), "lag_day"])
    lag_summary.to_csv(OUT / f"{name}_{stratum.lower()}_lag_response.csv", index=False)

    return {
        "biomarker": name,
        "stratum": stratum,
        "status": "ok",
        "n_obs": int(len(X)),
        "n_patients": int(groups.nunique()),
        "fold_r2": fold_r2,
        "mean_r2": mean_r2,
        "r2_ci_lower": ci_lo,
        "r2_ci_upper": ci_hi,
        "retained": bool(mean_r2 >= R2_CUT),
        "peak_lag_days": peak,
    }


def main():
    if not CACHE.exists():
        sys.exit(f"missing {CACHE}; build it first (see run_nested_cv_sensitivity.py)")
    df = pd.read_pickle(CACHE)
    df = df[df["sex"].isin(STRATA)]

    ten = json.loads(
        (config.OUTPUT_ROOT / "stage3_r2_005" / "stage3_summary.json").read_text()
    )["included_biomarkers"]
    specs = {b.column: b for b in get_available_biomarkers().values() if b.column}
    log(f"dataset {df.shape}; strata "
        + ", ".join(f"{s}={int((df.sex == s).sum())}" for s in STRATA)
        + f"; {len(ten)} biomarkers")

    results = []
    for name in ten:
        spec = specs.get(name)
        if spec is None:
            log(f"!! no BiomarkerSpec for {name}; skipping"); continue
        primary_peak = _primary_peak_lag(name)
        primary_r2 = _primary_mean_r2(name)
        for stratum in STRATA:
            df_s = df[df["sex"] == stratum].drop(columns=["sex"])
            t0 = time.time()
            r = run_stratum(name, spec, df_s, stratum)
            r["seconds"] = round(time.time() - t0, 1)
            r["primary_peak_lag_days"] = primary_peak
            r["primary_mean_r2"] = primary_r2
            if r["status"] == "ok":
                r["same_peak_as_primary"] = bool(r["peak_lag_days"] == primary_peak)
                log(f"{name} [{stratum}]: n={r['n_obs']} ({r['n_patients']} pts)  "
                    f"R2={r['mean_r2']:.4f} ({r['r2_ci_lower']:.4f},"
                    f"{r['r2_ci_upper']:.4f})  retained={r['retained']}  "
                    f"peak={r['peak_lag_days']}d (primary {primary_peak}d)  "
                    f"({r['seconds']}s)")
            else:
                log(f"{name} [{stratum}]: NOT ESTIMABLE — {r['reason']}")
            results.append(r)
            (OUT / "sa13_partial.json").write_text(json.dumps(results, indent=2))

    # ----- appendix summary -----
    by_bio = {}
    for r in results:
        by_bio.setdefault(r["biomarker"], {})[r["stratum"]] = r

    both, one, neither, not_est, same_peak, peak_pairs = [], [], [], [], [], []
    for b, d in by_bio.items():
        ok = {s: r for s, r in d.items() if r["status"] == "ok"}
        if len(ok) < 2:
            not_est.append(b)
        kept = [s for s, r in ok.items() if r["retained"]]
        if len(d) == len(ok):
            (both if len(kept) == 2 else one if len(kept) == 1 else neither).append(b)
        peaks = {s: r["peak_lag_days"] for s, r in ok.items()}
        if len(peaks) == 2:
            peak_pairs.append(b)
            if len(set(peaks.values())) == 1 and d["Female"]["same_peak_as_primary"]:
                same_peak.append(b)

    summary = {
        "sensitivity_analysis": "sa13_sex_stratified",
        "description": ("Sex-stratified replication of Stage 1. Identical "
                        "hyperparameters, participant-grouped 5-fold CV, "
                        "interventional TreeSHAP and R^2 >= 0.05 retention "
                        "rule; sex removed from covariates as the stratifier."),
        "min_obs": MIN_OBS,
        "retention_threshold": R2_CUT,
        "retained_in_both": sorted(both),
        "retained_in_one_only": sorted(one),
        "retained_in_neither": sorted(neither),
        "not_estimable_in_a_stratum": sorted(not_est),
        "same_peak_lag_both_strata_and_primary": sorted(same_peak),
        "per_stratum": results,
    }
    dest = OUT / "sa13_summary.json"
    dest.write_text(json.dumps(summary, indent=2))
    log(f"wrote {dest}")

    n = len(by_bio)
    log(f"Of {n} retained biomarkers: {len(both)} retained in both strata, "
        f"{len(one)} in one stratum only, {len(neither)} in neither; "
        f"{len(not_est)} not estimable in at least one stratum.")
    log(f"{len(same_peak)}/{len(peak_pairs)} estimable in both strata kept the "
        f"same peak lag in both strata as the pooled analysis.")


if __name__ == "__main__":
    sys.exit(main())
