#!/usr/bin/env python3
"""Nested cross-validation sensitivity analysis (Stage 1 hyperparameters).

Addresses the reviewer point (A. Waljee, Q2) that Stage 1 used hyperparameters
fixed a priori and applied identically to all 24 biomarkers. Fixed
hyperparameters avoid tuning on the data we then interpret, but may underfit
some outcomes and so understate predictive signal. This quantifies that.

Design
------
Outer loop  : 5-fold GroupKFold by patient_id -- the SAME scheme, and the same
              folds, as the primary Stage 1 analysis.
Inner loop  : 3-fold GroupKFold by patient_id, run strictly WITHIN each outer
              training portion. No outer-test data touches selection.
Search space: max_depth {3, 6, 9} x learning_rate {0.03, 0.05, 0.10}
              (the two capacity knobs the underfitting concern is about);
              all other hyperparameters held at config.XGBOOST_PARAMS.
Comparison  : the fixed-hyperparameter model is scored on the SAME outer folds,
              so the contrast is paired fold-by-fold rather than against the
              published number.

Interpretation: if nested-CV R-squared is not materially higher than the fixed
model's, the a priori hyperparameters were not underfitting and the primary
estimates are not conservative for that reason.
"""
from __future__ import annotations
import json, logging, os, sys, time
from itertools import product
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

OUT = config.OUTPUT_ROOT / "sensitivity" / "sa12_nested_cv"
OUT.mkdir(parents=True, exist_ok=True)

OUTER_FOLDS = config.CV_FOLDS      # 5
INNER_FOLDS = 3

# v1 was centred on the a priori values (depth 6, lr 0.05). Every biomarker
# selected the minimum of both axes, so the optimum lay on the boundary and v1's
# gain is only a lower bound. v2 extends downward to bracket it, keeping depth 6
# as the a priori reference point.
GRIDS = {
    "v1": {"max_depth": [3, 6, 9], "learning_rate": [0.03, 0.05, 0.10]},
    "v2": {"max_depth": [2, 3, 6], "learning_rate": [0.01, 0.02, 0.03, 0.05]},
}

# Biomarkers that missed the R^2 >= 0.05 retention cut under fixed
# hyperparameters. Whether tuning lifts them over it decides whether the
# retained panel is still ten.
NEAR_MISS = ["systolic_bp", "albumin", "hip_circumference"]


def log(m): print(f"[nestedcv] {m}", flush=True)


def _params(**over):
    p = {**config.XGBOOST_PARAMS, **over}
    p["n_jobs"] = -1
    return p


def _fit_score(params, Xtr, ytr, Xte, yte):
    m = xgb.XGBRegressor(**params)
    m.fit(Xtr, ytr, verbose=False)
    return float(r2_score(yte, m.predict(Xte)))


def nested_cv_one(name, X, y, groups, grid):
    outer = GroupKFold(n_splits=OUTER_FOLDS)
    combos = [dict(zip(grid, v)) for v in product(*grid.values())]

    fixed_r2, nested_r2, chosen = [], [], []
    for fold, (tr, te) in enumerate(outer.split(X, y, groups)):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        gtr = groups.iloc[tr]

        # (a) fixed hyperparameters, this outer fold
        fixed_r2.append(_fit_score(_params(), Xtr, ytr, Xte, yte))

        # (b) select within the training portion only
        inner = GroupKFold(n_splits=INNER_FOLDS)
        splits = list(inner.split(Xtr, ytr, gtr))
        best, best_score = None, -np.inf
        for c in combos:
            scores = []
            for itr, ite in splits:
                scores.append(_fit_score(
                    _params(**c),
                    Xtr.iloc[itr], ytr.iloc[itr], Xtr.iloc[ite], ytr.iloc[ite],
                ))
            s = float(np.mean(scores))
            if s > best_score:
                best, best_score = c, s

        # (c) refit on the full outer-training portion with the selected params,
        #     score once on the untouched outer test fold
        nested_r2.append(_fit_score(_params(**best), Xtr, ytr, Xte, yte))
        chosen.append({**best, "inner_mean_r2": best_score})
        log(f"    fold {fold}: fixed={fixed_r2[-1]:.4f}  nested={nested_r2[-1]:.4f}  "
            f"chose depth={best['max_depth']} lr={best['learning_rate']}")

    return {
        "biomarker": name,
        "n_obs": int(len(X)),
        "n_patients": int(groups.nunique()),
        "fixed_fold_r2": fixed_r2,
        "nested_fold_r2": nested_r2,
        "fixed_mean_r2": float(np.mean(fixed_r2)),
        "nested_mean_r2": float(np.mean(nested_r2)),
        "delta_r2": float(np.mean(nested_r2) - np.mean(fixed_r2)),
        "selected_per_fold": chosen,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="v1", choices=sorted(GRIDS))
    ap.add_argument("--include-near-miss", action="store_true",
                    help="also run the biomarkers that missed the 0.05 cut")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    grid = GRIDS[a.grid]
    tag = a.tag or a.grid

    if not CACHE.exists():
        sys.exit(f"missing {CACHE}; build it first")
    df = pd.read_pickle(CACHE)
    ten = json.loads(
        (config.OUTPUT_ROOT / "stage3_r2_005" / "stage3_summary.json").read_text()
    )["included_biomarkers"]
    panel = list(ten) + (NEAR_MISS if a.include_near_miss else [])
    specs = {b.column: b for b in get_available_biomarkers().values() if b.column}
    log(f"dataset {df.shape}; grid={a.grid} {grid}; {len(panel)} biomarkers")

    results = []
    for name in panel:
        spec = specs.get(name)
        if spec is None:
            log(f"!! no BiomarkerSpec for {name}; skipping"); continue
        X, y, groups = prepare_features_and_target(df, spec)
        log(f"--- {name}: n={len(X)} obs, {groups.nunique()} patients ---")
        t0 = time.time()
        r = nested_cv_one(name, X, y, groups, grid)
        r["seconds"] = round(time.time() - t0, 1)
        r["retained_in_primary"] = name in ten
        log(f"    fixed={r['fixed_mean_r2']:.4f}  nested={r['nested_mean_r2']:.4f}  "
            f"delta={r['delta_r2']:+.4f}  ({r['seconds']}s)")
        results.append(r)
        (OUT / f"sa12_nested_cv_partial_{tag}.json").write_text(json.dumps(results, indent=2))

    deltas = [r["delta_r2"] for r in results]
    summary = {
        "sensitivity_analysis": "sa12_nested_cv",
        "description": ("Nested cross-validation of Stage 1 XGBoost hyperparameters. "
                        "Outer 5-fold GroupKFold by patient; inner 3-fold GroupKFold "
                        "within each outer training portion only."),
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "search_space": grid,
        "grid_name": a.grid,
        "fixed_hyperparameters": {k: v for k, v in config.XGBOOST_PARAMS.items()
                                  if k not in ("n_jobs",)},
        "n_biomarkers": len(results),
        "mean_delta_r2": float(np.mean(deltas)) if deltas else None,
        "max_delta_r2": float(np.max(deltas)) if deltas else None,
        "min_delta_r2": float(np.min(deltas)) if deltas else None,
        "n_improved": int(sum(d > 0 for d in deltas)),
        "n_biomarkers_crossing_threshold": int(sum(
            r["fixed_mean_r2"] < 0.05 <= r["nested_mean_r2"] for r in results)),
        "per_biomarker": results,
    }
    dest = OUT / f"sa12_nested_cv_summary_{tag}.json"
    dest.write_text(json.dumps(summary, indent=2))
    log(f"wrote {dest}")
    log(f"mean delta R2 = {summary['mean_delta_r2']:+.4f}; "
        f"improved in {summary['n_improved']}/{len(results)}")


if __name__ == "__main__":
    sys.exit(main())
