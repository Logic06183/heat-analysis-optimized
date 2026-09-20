"""
SHAP Background Subsampling Verification
=========================================

Validates that using 500 background samples for interventional TreeSHAP
produces lag attributions consistent with using the full dataset as background.

Methodology
-----------
For each target biomarker:
1. Train ONE XGBoost model (fixed seed — no bootstrap)
2. Compute SHAP with FULL background  (original behaviour)
3. Compute SHAP with 500-sample background  (optimised behaviour)
4. Compare:
   - Spearman r of mean |SHAP| feature rankings (all features)
   - Spearman r of lag-feature importance rankings specifically
   - Dominant temporal window classification match (acute/sub_acute/extended/mixed)
   - Top-3 lag features in common

Pass criteria (conservative, publication threshold):
   - Spearman r (all features)  >= 0.95
   - Spearman r (lag features)  >= 0.90
   - Dominant window match      == True

Usage
-----
    python verify_shap_background.py

Run after `python run_pipeline.py --stage 0 1 --fast` completes.
Takes ~5-10 minutes on M4 Pro.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers
from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features
from mcd_pipeline.stage1_lag_profiling.lag_response import (
    classify_temporal_window,
    extract_lag_shap_profile,
    summarise_lag_response,
)
from mcd_pipeline.stage1_lag_profiling.xgboost_trainer import (
    prepare_features_and_target,
    train_single_model,
)

# Biomarkers to verify: pick the ones with best R² so signal is clearest
VERIFY_BIOMARKERS = ["hemoglobin", "hematocrit", "viral_load", "creatinine", "systolic_bp"]

SPEARMAN_ALL_THRESHOLD  = 0.95
SPEARMAN_LAG_THRESHOLD  = 0.90
BACKGROUND_N            = config.SHAP_BACKGROUND_SAMPLES  # 500


def _compute_shap(model, X, background):
    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation=config.SHAP_FEATURE_PERTURBATION,
    )
    return explainer.shap_values(X, check_additivity=False)


def _lag_window(shap_values, feature_names):
    profile = extract_lag_shap_profile(shap_values, feature_names)
    if profile.empty:
        return "unknown"
    summary = summarise_lag_response(profile)
    return classify_temporal_window(summary)


def _top_lag_features(mean_abs_shap, feature_names, n=3):
    lag_idx = [i for i, f in enumerate(feature_names) if f.startswith("temp_lag_")]
    lag_importance = [(feature_names[i], mean_abs_shap[i]) for i in lag_idx]
    lag_importance.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in lag_importance[:n]]


def run_verification(df, biomarker_name, biomarker_spec):
    print(f"\n{'='*60}")
    print(f"  {biomarker_name}")
    print(f"{'='*60}")

    X, y, groups = prepare_features_and_target(df, biomarker_spec)
    feature_names = X.columns.tolist()
    n = len(X)

    print(f"  N={n} samples, {len(feature_names)} features")

    # Train one model (fixed seed)
    model, cv = train_single_model(X, y, groups)
    r2_folds = [f["r2"] for f in cv.get("folds", [])]
    std_r2 = float(np.std(r2_folds)) if r2_folds else float("nan")
    print(f"  CV R²={cv['mean_r2']:.4f} (±{std_r2:.4f})")

    if cv["mean_r2"] < 0:
        print("  SKIP: R²<0 — lag attributions not interpretable")
        return None

    # --- Full background ---
    print(f"  Computing SHAP (full background, N={n})...", end=" ", flush=True)
    shap_full = _compute_shap(model, X, background=X)
    print("done")

    # --- Subsampled background ---
    background_sub = shap.utils.sample(X, BACKGROUND_N, random_state=42)
    print(f"  Computing SHAP (sub background, N={BACKGROUND_N})...", end=" ", flush=True)
    shap_sub = _compute_shap(model, X, background=background_sub)
    print("done")

    # --- Compare mean |SHAP| feature rankings ---
    mean_full = np.abs(shap_full).mean(axis=0)
    mean_sub  = np.abs(shap_sub).mean(axis=0)

    r_all, _ = spearmanr(mean_full, mean_sub)

    lag_idx = [i for i, f in enumerate(feature_names) if f.startswith("temp_lag_")]
    r_lag, _ = spearmanr(mean_full[lag_idx], mean_sub[lag_idx])

    # --- Window classification ---
    window_full = _lag_window(shap_full, feature_names)
    window_sub  = _lag_window(shap_sub,  feature_names)
    window_match = window_full == window_sub

    # --- Top-3 lag features ---
    top3_full = _top_lag_features(mean_full, feature_names)
    top3_sub  = _top_lag_features(mean_sub,  feature_names)
    top3_overlap = len(set(top3_full) & set(top3_sub))

    # --- Pass/fail ---
    pass_all    = r_all >= SPEARMAN_ALL_THRESHOLD
    pass_lag    = r_lag >= SPEARMAN_LAG_THRESHOLD
    pass_window = window_match

    print(f"\n  Spearman r (all features):  {r_all:.4f}  {'✓ PASS' if pass_all    else '✗ FAIL'} (threshold {SPEARMAN_ALL_THRESHOLD})")
    print(f"  Spearman r (lag features):  {r_lag:.4f}  {'✓ PASS' if pass_lag    else '✗ FAIL'} (threshold {SPEARMAN_LAG_THRESHOLD})")
    print(f"  Dominant window (full):     {window_full}")
    print(f"  Dominant window (sub):      {window_sub}  {'✓ MATCH' if pass_window else '✗ MISMATCH'}")
    print(f"  Top-3 lag overlap:          {top3_overlap}/3  {top3_full} vs {top3_sub}")

    # Show lag importance side-by-side
    lag_names = [feature_names[i] for i in lag_idx]
    print(f"\n  Lag importance (mean |SHAP|):")
    print(f"  {'Feature':<20} {'Full':>10} {'Sub-500':>10} {'Diff%':>8}")
    for i, name in zip(lag_idx, lag_names):
        diff_pct = 100 * (mean_sub[i] - mean_full[i]) / (mean_full[i] + 1e-10)
        print(f"  {name:<20} {mean_full[i]:>10.4f} {mean_sub[i]:>10.4f} {diff_pct:>7.1f}%")

    overall_pass = pass_all and pass_lag and pass_window
    print(f"\n  OVERALL: {'✓ PASS' if overall_pass else '✗ FAIL'}")

    return {
        "biomarker":        biomarker_name,
        "n_samples":        n,
        "cv_r2":            cv["mean_r2"],
        "spearman_all":     r_all,
        "spearman_lag":     r_lag,
        "window_full":      window_full,
        "window_sub":       window_sub,
        "window_match":     window_match,
        "top3_overlap":     top3_overlap,
        "pass":             overall_pass,
    }


def main():
    print("SHAP Background Subsampling Verification")
    print(f"Background N: {BACKGROUND_N} vs full dataset")
    print(f"Thresholds: Spearman(all)>={SPEARMAN_ALL_THRESHOLD}, "
          f"Spearman(lag)>={SPEARMAN_LAG_THRESHOLD}, window match")

    # Build dataset once
    print("\nBuilding analysis dataset...")
    df = build_analysis_dataset()
    biomarkers_dict = get_available_biomarkers()
    bio_cols = [b.column for b in biomarkers_dict.values()
                if b.column and b.column in df.columns]
    df = engineer_features(df, bio_cols)

    # Run verification for each target biomarker
    records = []
    for name in VERIFY_BIOMARKERS:
        spec = next((s for s in biomarkers_dict.values() if s.column == name), None)
        if spec is None:
            print(f"\nSkipping {name}: not found in biomarker registry")
            continue
        result = run_verification(df, name, spec)
        if result:
            records.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Biomarker':<22} {'r_all':>7} {'r_lag':>7} {'Window':>8} {'Result':>8}")
    print("-" * 60)
    all_pass = True
    for r in records:
        status = "✓ PASS" if r["pass"] else "✗ FAIL"
        win = "✓" if r["window_match"] else "✗"
        print(f"{r['biomarker']:<22} {r['spearman_all']:>7.4f} {r['spearman_lag']:>7.4f} "
              f"{'match' if r['window_match'] else 'MISMATCH':>8}  {status}")
        if not r["pass"]:
            all_pass = False

    print(f"\n{'='*60}")
    if all_pass:
        print("✓ ALL PASS — 500-sample background is sufficient.")
        print("  Lag attributions and temporal windows are consistent")
        print("  with full-background computation. Optimisation is valid.")
    else:
        print("✗ FAILURES DETECTED — review above before using optimised results.")
        print("  Consider increasing SHAP_BACKGROUND_SAMPLES in config.py.")
    print(f"{'='*60}")

    # Save results
    out_path = Path("mcd_outputs/shap_background_verification.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
