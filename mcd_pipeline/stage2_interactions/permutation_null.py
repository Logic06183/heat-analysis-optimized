"""
Permutation Null Distribution — Statistical Significance for Interactions
=========================================================================

Generates a null distribution of SHAP interaction values by permuting
the modifier variable (breaking the true interaction while preserving
marginal distributions). Compares observed interactions against this null
to compute p-values.

500 permutations per biomarker model, as specified in the MCD.

MCD reference: "exploratory framing with permutation-derived null
(500 permutations per biomarker model)"
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from joblib import Parallel, delayed

from mcd_pipeline import config

logger = logging.getLogger(__name__)

# Subsample size for permutation null — smaller than interaction tensor
# subsample to keep each permutation fast (~1-2 sec on M4 Pro).
PERM_SAMPLE_SIZE = 100


def generate_permutation_null(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    feature_i: str,
    feature_j: str,
    n_permutations: int = config.N_PERMUTATIONS,
    seed: int = config.MASTER_SEED,
) -> np.ndarray:
    """Generate null distribution of interaction values by permuting feature_j.

    For each permutation:
    1. Shuffle feature_j values (breaking interaction with feature_i)
    2. Recompute SHAP interaction values for the (i, j) pair
    3. Record the mean absolute interaction

    Uses a small subsample (PERM_SAMPLE_SIZE rows) per permutation for speed,
    and parallelises across all available cores.

    Parameters
    ----------
    model : xgb.XGBRegressor
        Trained model.
    X : pd.DataFrame
        Original feature matrix.
    feature_i, feature_j : str
        Feature pair to test.
    n_permutations : int
        Number of permutations (default: 500).

    Returns
    -------
    np.ndarray
        Null distribution of mean |interaction| values, length n_permutations.
    """
    n_sub = min(len(X), PERM_SAMPLE_SIZE)
    X_sub = X.sample(n=n_sub, random_state=seed) if len(X) > n_sub else X.copy()

    feature_names = X_sub.columns.tolist()
    if feature_i not in feature_names or feature_j not in feature_names:
        logger.warning("  Feature %s or %s not in feature matrix", feature_i, feature_j)
        return np.zeros(n_permutations)

    idx_i = feature_names.index(feature_i)
    idx_j = feature_names.index(feature_j)

    rng = np.random.RandomState(seed)
    perm_seeds = rng.randint(0, 2**31, size=n_permutations)

    # Build TreeExplainer once — expensive model parsing happens here, not per permutation
    explainer = shap.TreeExplainer(model)

    def _one_permutation(perm_seed: int) -> float:
        X_perm = X_sub.copy()
        perm_rng = np.random.RandomState(perm_seed)
        X_perm[feature_j] = perm_rng.permutation(X_perm[feature_j].values)
        interaction_vals = explainer.shap_interaction_values(X_perm)
        return float(np.abs(interaction_vals[:, idx_i, idx_j]).mean())

    n_jobs = min(n_permutations, config.N_PARALLEL_JOBS if config.N_PARALLEL_JOBS > 0 else 14)
    null_dist = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_permutation)(s) for s in perm_seeds
    )
    return np.array(null_dist)


def compute_permutation_pvalue(
    observed: float,
    null_distribution: np.ndarray,
) -> float:
    """Compute one-sided p-value: P(null >= observed).

    Returns
    -------
    float
        p-value in [0, 1]. Minimum is 1/n_permutations.
    """
    n = len(null_distribution)
    if n == 0:
        return 1.0
    p = float(np.sum(null_distribution >= observed) / n)
    # Floor at 1/n to avoid p=0 (not estimable below this resolution)
    return max(p, 1.0 / n)


def test_top_interactions(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    ranked_interactions: pd.DataFrame,
    n_permutations: int = config.N_PERMUTATIONS,
    checkpoint_path: Path = None,
) -> pd.DataFrame:
    """Test all top-ranked interactions against permutation null.

    Only runs permutation tests for temperature × modifier pairs
    (i.e., where feature_i or feature_j is a temperature lag feature),
    as these are the scientifically focal interactions in the MCD.
    Non-temperature pairs receive p_value=NaN.

    If checkpoint_path is given, saves progress after each pair and resumes
    from an existing checkpoint on restart (skipping already-tested pairs).

    Returns
    -------
    pd.DataFrame
        ranked_interactions with added columns: null_mean, null_std,
        p_value, significant_raw (p < 0.05).
    """
    result = ranked_interactions.copy()
    result["null_mean"] = np.nan
    result["null_std"] = np.nan
    result["p_value"] = np.nan
    result["significant_raw"] = False

    # Resume from checkpoint if available
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            ckpt = pd.read_csv(checkpoint_path)
            for col in ["null_mean", "null_std", "p_value", "significant_raw"]:
                if col not in ckpt.columns:
                    continue
                for _, ckpt_row in ckpt.iterrows():
                    if pd.isna(ckpt_row.get("p_value")):
                        continue
                    mask = (
                        (result["feature_i"] == ckpt_row["feature_i"]) &
                        (result["feature_j"] == ckpt_row["feature_j"])
                    )
                    result.loc[mask, col] = ckpt_row[col]
            n_loaded = result["p_value"].notna().sum()
            if n_loaded > 0:
                logger.info("  Resumed %d already-tested pairs from checkpoint", n_loaded)

    temp_prefix = "temp_lag_"

    for idx, row in result.iterrows():
        fi = row["feature_i"]
        fj = row["feature_j"]

        # Skip pairs already tested (loaded from checkpoint)
        if pd.notna(row["p_value"]):
            logger.info("  Skipping %s × %s (checkpoint)", fi, fj)
            continue

        # Only permutation-test pairs involving a temperature lag feature
        is_temp_pair = fi.startswith(temp_prefix) or fj.startswith(temp_prefix)
        if not is_temp_pair:
            continue

        # Permute the non-temperature feature (the modifier)
        modifier = fj if fi.startswith(temp_prefix) else fi

        logger.info("  Permutation test: %s × %s (%d perms)...", fi, fj, n_permutations)
        null_dist = generate_permutation_null(
            model, X, fi, modifier, n_permutations=n_permutations
        )
        p_val = compute_permutation_pvalue(row["mean_abs_interaction"], null_dist)

        result.at[idx, "null_mean"] = float(null_dist.mean())
        result.at[idx, "null_std"] = float(null_dist.std())
        result.at[idx, "p_value"] = p_val
        result.at[idx, "significant_raw"] = bool(p_val < 0.05)

        logger.info(
            "    observed=%.4f  null=%.4f±%.4f  p=%.4f  %s",
            row["mean_abs_interaction"],
            null_dist.mean(), null_dist.std(),
            p_val,
            "sig" if p_val < 0.05 else "ns",
        )

        # Save after each pair so progress survives a crash
        if checkpoint_path is not None:
            result.to_csv(checkpoint_path, index=False)

    return result
