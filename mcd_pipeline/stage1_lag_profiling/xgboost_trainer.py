"""
XGBoost Trainer — Per-Biomarker Model with Bootstrap Replication
================================================================

Trains one XGBoost model per biomarker with:
- GroupKFold cross-validation (grouped by patient_id)
- 50 bootstrap model replicates for SHAP stability assessment
- Person-level resampling for bootstrap confidence intervals

Each bootstrap replicate resamples patients (not observations), trains
XGBoost on the resampled data, and produces SHAP values. The collection
of 50 SHAP arrays enables stability assessment of feature rankings.

MCD reference: "One XGBoost model per biomarker (n=27)... Bootstrap
stability of SHAP lag rankings reported across 50 model replicates."
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import BiomarkerSpec
from mcd_pipeline.stage1_lag_profiling.shap_profiler import compute_shap_values

logger = logging.getLogger(__name__)


def _build_feature_columns() -> list[str]:
    """Return the ordered list of feature columns for the XGBoost model."""
    features = []

    # 1. Temperature lag features (standardised names)
    features += [f"temp_lag_{d}d" for d in config.LAG_DAYS]

    # 2. Diurnal temperature range
    features.append(config.DTR_COLUMN)

    # 3. Demographic + clinical covariates
    features += config.DEMOGRAPHIC_COVARIATES
    features += config.CLINICAL_COVARIATES

    # 4. Socioeconomic covariates
    features += config.SOCIOECONOMIC_COVARIATES

    # 5. Temporal confounders
    features.append("year")
    features.append("study_source")
    features += ["fourier_sin_month", "fourier_cos_month"]
    features += [f"month_{m}" for m in range(2, 13)]

    return features


def prepare_features_and_target(
    df: pd.DataFrame,
    biomarker: BiomarkerSpec,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare X (features), y (target), and groups (patient_id) for a biomarker.

    Drops rows where the target biomarker is NaN. Constructs the feature
    matrix from lag columns + covariates + Fourier terms + month FE.

    Returns
    -------
    tuple of (X, y, groups)
    """
    target_col = biomarker.column
    if target_col not in df.columns:
        raise ValueError(f"Biomarker column '{target_col}' not found in dataset")

    # Drop rows where target is NaN
    mask = df[target_col].notna()
    subset = df.loc[mask].copy()

    if len(subset) == 0:
        raise ValueError(f"No non-null values for biomarker '{target_col}'")

    # Apply log transform if specified
    y = subset[target_col].astype(float)
    if biomarker.log_transform:
        y = np.log1p(y)

    # Build feature matrix
    feature_cols = _build_feature_columns()
    available = [c for c in feature_cols if c in subset.columns]
    X = subset[available].copy()

    # Encode categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)

    # Warn if any feature column has high NaN prevalence before imputation
    nan_pct = X.isna().mean()
    high_nan = nan_pct[nan_pct > 0.2]
    if not high_nan.empty:
        logger.warning(
            "  %s: high NaN prevalence in features (>20%%) — median imputation may be unreliable: %s",
            target_col,
            {c: f"{v:.1%}" for c, v in high_nan.items()},
        )

    # Fill remaining NaN in features with column median
    X = X.fillna(X.median(numeric_only=True))

    # float32: XGBoost uses float32 internally; halves memory bandwidth
    X = X.astype("float32")

    groups = subset[config.PATIENT_ID_COLUMN].astype(str)

    n_patients = groups.nunique()
    if n_patients < config.CV_FOLDS:
        raise ValueError(
            f"Biomarker '{target_col}' has only {n_patients} unique patients "
            f"but GroupKFold requires ≥ {config.CV_FOLDS}. "
            "Cannot create non-overlapping patient folds."
        )

    logger.info(
        "  %s: %d samples, %d features, %d patients",
        target_col,
        len(X),
        X.shape[1],
        n_patients,
    )
    return X, y, groups


def train_single_model(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    params: Optional[dict] = None,
) -> tuple[xgb.XGBRegressor, dict]:
    """Train XGBoost with GroupKFold CV and return model + metrics.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target biomarker values.
    groups : pd.Series
        Patient IDs for GroupKFold.
    params : dict, optional
        XGBoost parameters. Defaults to config.XGBOOST_PARAMS.

    Returns
    -------
    tuple of (fitted model, metrics dict with R², RMSE, MAE per fold)
    """
    params = params or config.XGBOOST_PARAMS

    gkf = GroupKFold(n_splits=config.CV_FOLDS)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = model.predict(X_val)
        fold_metrics.append(
            {
                "fold": fold,
                "r2": r2_score(y_val, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_val, y_pred)),
                "mae": mean_absolute_error(y_val, y_pred),
                "n_train": len(X_train),
                "n_val": len(X_val),
            }
        )

    # Refit on full data for SHAP computation
    final_model = xgb.XGBRegressor(**params)
    final_model.fit(X, y, verbose=False)

    metrics = {
        "folds": fold_metrics,
        "mean_r2": np.mean([m["r2"] for m in fold_metrics]),
        "mean_rmse": np.mean([m["rmse"] for m in fold_metrics]),
        "mean_mae": np.mean([m["mae"] for m in fold_metrics]),
        "n_samples": len(X),
        "n_features": X.shape[1],
    }

    logger.info(
        "  CV: R²=%.4f (±%.4f), RMSE=%.4f, MAE=%.4f",
        metrics["mean_r2"],
        np.std([m["r2"] for m in fold_metrics]),
        metrics["mean_rmse"],
        metrics["mean_mae"],
    )
    return final_model, metrics


def bootstrap_train(
    df: pd.DataFrame,
    biomarker: BiomarkerSpec,
    n_replicates: int = config.N_BOOTSTRAP_REPLICATES,
    seed: int = config.BOOTSTRAP_RANDOM_SEED,
) -> tuple[list[np.ndarray], pd.DataFrame, list[str]]:
    """Train N bootstrap replicates with person-level resampling.

    For each replicate:
    1. Resample patient IDs with replacement
    2. Select all observations for resampled patients
    3. Train XGBoost on resampled data
    4. Compute SHAP values on original (non-resampled) data

    Parameters
    ----------
    df : pd.DataFrame
        Full analysis dataset.
    biomarker : BiomarkerSpec
        Target biomarker specification.
    n_replicates : int
        Number of bootstrap replicates (default: 50).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (shap_arrays, X_original, feature_names)
        shap_arrays: list of SHAP arrays, one per replicate
        X_original: feature matrix for original (non-resampled) data
        feature_names: list of feature column names
    """
    rng = np.random.RandomState(seed)

    # Prepare original features/target (for SHAP computation)
    X_orig, y_orig, groups_orig = prepare_features_and_target(df, biomarker)
    feature_names = X_orig.columns.tolist()
    patient_ids = groups_orig.unique()

    # Pre-generate all bootstrap seeds for reproducibility
    rep_seeds = rng.randint(0, 2**31, size=n_replicates)

    def _one_replicate(rep_seed: int) -> np.ndarray:
        rep_rng = np.random.RandomState(rep_seed)
        boot_patients = rep_rng.choice(patient_ids, size=len(patient_ids), replace=True)
        boot_mask = groups_orig.isin(boot_patients)
        X_boot = X_orig.loc[boot_mask]
        y_boot = y_orig.loc[boot_mask]
        # XGBoost uses 1 thread per worker to avoid over-subscription
        params = {**config.XGBOOST_PARAMS, "n_jobs": 1}
        model = xgb.XGBRegressor(**params)
        model.fit(X_boot, y_boot, verbose=False)
        return compute_shap_values(model, X_orig)

    n_jobs = min(n_replicates, config.N_PARALLEL_JOBS)
    shap_arrays = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_replicate)(s) for s in rep_seeds
    )
    logger.info("  %d bootstrap replicates complete (parallel, %d workers)", n_replicates, n_jobs)

    return shap_arrays, X_orig, feature_names


def train_biomarker(
    df: pd.DataFrame,
    biomarker: BiomarkerSpec,
    output_dir: Optional[Path] = None,
) -> dict:
    """Full training pipeline for a single biomarker.

    1. Prepare features and target
    2. Train primary model with GroupKFold CV
    3. Train bootstrap replicates
    4. Save model, metrics, and SHAP arrays

    Returns
    -------
    dict
        Results including cv_metrics, bootstrap_metrics, output_paths.
    """
    from mcd_pipeline.stage1_lag_profiling.shap_profiler import (
        aggregate_bootstrap_shap,
        save_shap_results,
    )
    from mcd_pipeline.stage1_lag_profiling.lag_response import (
        extract_lag_shap_profile,
        summarise_lag_response,
        classify_temporal_window,
    )

    output_dir = output_dir or (config.OUTPUT_ROOT / "stage1" / biomarker.column)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training biomarker: %s (%s)", biomarker.name, biomarker.column)

    # Step 1: Primary model with CV
    X, y, groups = prepare_features_and_target(df, biomarker)
    model, cv_metrics = train_single_model(X, y, groups)

    # Step 2: Primary SHAP values
    primary_shap = compute_shap_values(model, X)

    # Step 3: Bootstrap replicates
    logger.info("  Running %d bootstrap replicates...", config.N_BOOTSTRAP_REPLICATES)
    shap_arrays, X_orig, feature_names = bootstrap_train(df, biomarker)

    # Step 4: Aggregate bootstrap SHAP
    boot_summary = aggregate_bootstrap_shap(shap_arrays, feature_names)

    # Step 5: Lag-response profile
    lag_profile = extract_lag_shap_profile(primary_shap, feature_names)
    lag_summary = summarise_lag_response(lag_profile)
    temporal_window = classify_temporal_window(lag_summary)

    # Step 6: Save results
    saved = save_shap_results(primary_shap, X, biomarker.column, output_dir)

    # Save trained model for Stage 2 interaction detection
    model.save_model(output_dir / "model.ubj")

    # Save lag summary
    lag_summary.to_csv(output_dir / "lag_response_summary.csv", index=False)

    # Save CV metrics
    import json

    with open(output_dir / "cv_metrics.json", "w") as f:
        json.dump(cv_metrics, f, indent=2, default=str)

    # Save bootstrap summary
    boot_summary["mean_abs_shap"].to_csv(
        output_dir / "bootstrap_mean_shap.csv", header=True
    )

    mean_r2 = cv_metrics["mean_r2"]
    model_adequate = bool(mean_r2 >= 0.0)  # cast: numpy.bool_ → Python bool for JSON
    if not model_adequate:
        logger.warning(
            "  CAUTION: %s R²=%.3f — model worse than mean predictor; "
            "lag attributions unreliable for this biomarker.",
            biomarker.column, mean_r2,
        )

    results = {
        "biomarker": biomarker.column,
        "cv_metrics": cv_metrics,
        "bootstrap_stability": boot_summary.get("ranking_stability"),
        "dominant_temporal_window": temporal_window,
        "n_samples": len(X),
        "model_adequate": model_adequate,
        "output_dir": str(output_dir),
    }

    logger.info(
        "  Complete: R²=%.4f, window=%s, stability=%.3f",
        mean_r2,
        temporal_window,
        boot_summary.get("ranking_stability", 0),
    )
    return results
