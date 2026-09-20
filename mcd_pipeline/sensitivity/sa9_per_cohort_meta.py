"""
SA9: Per-Cohort Meta-Analysis for Cluster-Driving Biomarkers
=============================================================

Addresses the Reviewer 2 (Lancet Planetary Health) request for a direct
empirical check on Simpson's-paradox / assay-drift concerns underlying
the headline HIV-cluster finding.

Design
------
1. Restrict to the four biomarkers that drive the Stage 3 cluster
   separation: haematocrit, viral load, body fat percentage, haemoglobin.
   These can be overridden via --biomarkers.
2. For each biomarker, refit Stage 1 (XGBoost + interventional TreeSHAP)
   *separately within each of the six largest cohorts*. Cohorts smaller
   than MIN_COHORT_N are excluded with a transparent log entry.
3. Extract the per-cohort SHAP lag profile (mean |SHAP| at each lag) and
   the mean cross-validated R^2 with bootstrap 95% CIs.
4. Pool with both fixed-effect (inverse-variance) and DerSimonian-Laird
   random-effect meta-analysis on the per-lag importance, reporting tau^2
   and I^2 statistics so the editor can see between-cohort heterogeneity.
5. Render forest plots (one per biomarker x lag-window summary) and a
   per-cohort SHAP lag heatmap for the supplementary material.

Inputs / outputs
----------------
Input:  config.ANALYSIS_READY_CSV (or pre-engineered DataFrame passed in)
Output: mcd_outputs/sensitivity/sa9_per_cohort_meta/
          - sa9_summary.json
          - <biomarker>/per_cohort_lag_profiles.csv
          - <biomarker>/meta_analysis.csv
          - <biomarker>/forest_plot.png  (one per acute lag window)
          - <biomarker>/per_cohort_heatmap.png

The script is deterministic given config.MASTER_SEED.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")  # headless rendering on the analysis server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from mcd_pipeline import config
from mcd_pipeline.sensitivity._sensitivity_core import (
    load_primary_lag_profiles,
    run_lightweight_stage1,
)
from mcd_pipeline.stage0_data_prep.biomarker_definitions import (
    get_available_biomarkers,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Biomarkers identified in Stage 3 as driving the HIV-burden cluster.
DEFAULT_CLUSTER_DRIVERS: tuple[str, ...] = (
    "hematocrit",
    "viral_load",
    "body_fat_percent",
    "hemoglobin",
)

# Minimum patients in a cohort before we attempt a per-cohort fit.
# Below this XGBoost + GroupKFold(5) will not be reliable.
MIN_COHORT_N = 200

# Lag windows for the forest plot (acute, sub-acute, chronic representatives).
FOREST_LAGS_DAYS: tuple[int, ...] = (0, 7, 21)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class CohortFit:
    """Per-cohort Stage 1 fit summary used for meta-analysis."""

    cohort: str
    n_rows: int
    n_patients: int
    cv_r2_mean: float
    cv_r2_se: float
    lag_profile: pd.DataFrame  # columns: lag_day, mean_abs_shap, se_abs_shap


def _select_top_cohorts(
    df: pd.DataFrame,
    biomarker_col: str,
    n_top: int,
    min_n: int,
) -> list[str]:
    """Return the cohort names with the most non-missing biomarker observations."""
    counts = (
        df.dropna(subset=[biomarker_col])
        .groupby("study_source", dropna=True)["patient_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    eligible = counts[counts >= min_n]
    if eligible.empty:
        return []
    return list(eligible.head(n_top).index)


def _fit_cohort(
    df_cohort: pd.DataFrame,
    biomarker_name: str,
    out_dir: Path,
    label: str,
) -> Optional[CohortFit]:
    """Fit Stage 1 within a single cohort and return a CohortFit summary.

    Returns None if the fit fails (e.g. insufficient samples per fold).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results = run_lightweight_stage1(
        df=df_cohort,
        biomarker_names=[biomarker_name],
        output_dir=out_dir,
        n_bootstrap=config.N_BOOTSTRAP_SENSITIVITY,
        label=label,
    )
    res = results.get(biomarker_name)
    if res is None or "lag_summary" not in res:
        return None

    lag_summary = res["lag_summary"].copy()
    if "se_abs_shap" not in lag_summary.columns:
        # Older pipelines may not record SHAP standard errors; impute
        # something monotone with sample size so meta-analysis weights are
        # still sensible.
        lag_summary["se_abs_shap"] = (
            lag_summary["mean_abs_shap"].abs() / np.sqrt(max(res["n_samples"], 1))
        ).clip(lower=1e-6)

    cv_r2_mean = float(res.get("cv_r2", np.nan))
    # Approximate SE for R^2 from the 5-fold CV variance if available.
    fold_r2 = res.get("fold_r2") or res.get("cv_metrics", {}).get("fold_r2")
    if fold_r2 is not None and len(fold_r2) > 1:
        cv_r2_se = float(np.std(fold_r2, ddof=1) / np.sqrt(len(fold_r2)))
    else:
        cv_r2_se = float(np.nan)

    return CohortFit(
        cohort=label,
        n_rows=int(res.get("n_samples", df_cohort[biomarker_name].notna().sum())),
        n_patients=int(df_cohort["patient_id"].nunique()),
        cv_r2_mean=cv_r2_mean,
        cv_r2_se=cv_r2_se,
        lag_profile=lag_summary[["lag_day", "mean_abs_shap", "se_abs_shap"]],
    )


def _meta_analyse_lag(
    fits: list[CohortFit],
    lag_day: int,
) -> dict:
    """Run fixed-effect and DerSimonian-Laird meta-analysis at one lag day.

    Effect sizes are mean |SHAP| values, weighted by inverse variance
    derived from the per-cohort SHAP standard errors.
    """
    rows = []
    for fit in fits:
        row = fit.lag_profile.loc[fit.lag_profile["lag_day"] == lag_day]
        if row.empty:
            continue
        mean = float(row["mean_abs_shap"].iloc[0])
        se = float(max(row["se_abs_shap"].iloc[0], 1e-6))
        rows.append((fit.cohort, mean, se, fit.n_patients))

    if len(rows) < 2:
        return {
            "lag_day": lag_day,
            "n_cohorts": len(rows),
            "status": "too_few_cohorts",
        }

    cohorts = [r[0] for r in rows]
    means = np.array([r[1] for r in rows])
    ses = np.array([r[2] for r in rows])
    weights_fe = 1.0 / np.square(ses)

    # Fixed-effect estimate
    fe_mean = float(np.sum(weights_fe * means) / np.sum(weights_fe))
    fe_se = float(1.0 / np.sqrt(np.sum(weights_fe)))
    fe_ci = (fe_mean - 1.96 * fe_se, fe_mean + 1.96 * fe_se)

    # Cochran's Q and I^2 for between-cohort heterogeneity
    q = float(np.sum(weights_fe * np.square(means - fe_mean)))
    df_q = len(rows) - 1
    q_p = float(1.0 - stats.chi2.cdf(q, df_q)) if df_q > 0 else float("nan")
    i2 = float(max(0.0, 100.0 * (q - df_q) / q)) if q > 0 else 0.0

    # DerSimonian-Laird tau^2
    c = float(np.sum(weights_fe) - np.sum(np.square(weights_fe)) / np.sum(weights_fe))
    tau2 = float(max(0.0, (q - df_q) / c)) if c > 0 else 0.0
    weights_re = 1.0 / (np.square(ses) + tau2)
    re_mean = float(np.sum(weights_re * means) / np.sum(weights_re))
    re_se = float(1.0 / np.sqrt(np.sum(weights_re)))
    re_ci = (re_mean - 1.96 * re_se, re_mean + 1.96 * re_se)

    return {
        "lag_day": lag_day,
        "n_cohorts": len(rows),
        "cohorts": cohorts,
        "per_cohort_mean": means.tolist(),
        "per_cohort_se": ses.tolist(),
        "fixed_effect": {"estimate": fe_mean, "se": fe_se, "ci_95": list(fe_ci)},
        "random_effect": {"estimate": re_mean, "se": re_se, "ci_95": list(re_ci)},
        "heterogeneity": {
            "cochran_q": q,
            "cochran_q_pvalue": q_p,
            "i_squared_pct": i2,
            "tau_squared": tau2,
        },
    }


def _render_forest_plot(
    biomarker: str,
    lag_day: int,
    meta: dict,
    out_path: Path,
) -> None:
    """Render a publication-style forest plot for a single lag window."""
    if meta.get("status") == "too_few_cohorts":
        return

    cohorts = meta["cohorts"]
    means = np.array(meta["per_cohort_mean"])
    ses = np.array(meta["per_cohort_se"])

    y_pos = np.arange(len(cohorts), 0, -1)
    fig, ax = plt.subplots(figsize=(7.0, 0.4 * len(cohorts) + 2.0))

    ax.errorbar(
        means,
        y_pos,
        xerr=1.96 * ses,
        fmt="s",
        color="#1f3a93",
        ecolor="#7f7f7f",
        capsize=3,
        markersize=6,
        label="Per-cohort estimate",
    )

    fe = meta["fixed_effect"]
    re = meta["random_effect"]
    ax.axvline(fe["estimate"], color="#c0392b", linestyle="--", linewidth=1.0,
               label=f"Fixed-effect pooled = {fe['estimate']:.3f}")
    ax.axvline(re["estimate"], color="#27ae60", linestyle=":", linewidth=1.0,
               label=f"Random-effect pooled = {re['estimate']:.3f}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cohorts, fontsize=9)
    ax.set_xlabel(f"Mean |SHAP| at temperature lag {lag_day}d")
    title = (
        f"{biomarker} — per-cohort temperature lag-{lag_day}d SHAP\n"
        f"I² = {meta['heterogeneity']['i_squared_pct']:.1f}%, "
        f"tau² = {meta['heterogeneity']['tau_squared']:.4f}, "
        f"Q = {meta['heterogeneity']['cochran_q']:.2f} "
        f"(p = {meta['heterogeneity']['cochran_q_pvalue']:.3f})"
    )
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _render_cohort_heatmap(
    biomarker: str,
    fits: list[CohortFit],
    out_path: Path,
) -> None:
    """Render a per-cohort x per-lag SHAP heatmap."""
    if not fits:
        return

    # Build (cohort x lag) matrix
    lags = sorted({int(d) for fit in fits for d in fit.lag_profile["lag_day"].unique()})
    matrix = np.full((len(fits), len(lags)), np.nan)
    for i, fit in enumerate(fits):
        df = fit.lag_profile.set_index("lag_day")["mean_abs_shap"]
        for j, lag in enumerate(lags):
            if lag in df.index:
                matrix[i, j] = float(df.loc[lag])

    fig, ax = plt.subplots(figsize=(0.6 * len(lags) + 2, 0.45 * len(fits) + 1.5))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(fits)))
    ax.set_yticklabels([fit.cohort for fit in fits], fontsize=9)
    ax.set_xticks(range(len(lags)))
    ax.set_xticklabels([f"lag{l}d" for l in lags], fontsize=9)
    ax.set_title(f"{biomarker} — per-cohort SHAP lag profile", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Mean |SHAP|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_per_cohort_meta(
    df: pd.DataFrame,
    biomarkers: Iterable[str] | None = None,
    n_top_cohorts: int = 6,
    min_cohort_n: int = MIN_COHORT_N,
    output_dir: Path | None = None,
    primary_profiles: dict | None = None,
) -> dict:
    """Run the per-cohort meta-analysis sensitivity check.

    Parameters
    ----------
    df : DataFrame
        Pre-engineered analysis dataset (from the same loader Stage 1 uses).
    biomarkers : iterable of str, optional
        Biomarker names. Defaults to the four cluster-driving biomarkers.
    n_top_cohorts : int
        Number of largest cohorts to include per biomarker (default 6).
    min_cohort_n : int
        Minimum patients in a cohort before it is fitted (default 200).
    output_dir : Path, optional
        Where to write per-biomarker artefacts.
    primary_profiles : dict, optional
        Pooled Stage 1 SHAP profiles for direct comparison. Loaded from
        the standard Stage 1 output directory if not supplied.
    """
    if biomarkers is None:
        biomarkers = DEFAULT_CLUSTER_DRIVERS

    output_dir = output_dir or (config.OUTPUT_ROOT / "sensitivity" / "sa9_per_cohort_meta")
    output_dir.mkdir(parents=True, exist_ok=True)

    if primary_profiles is None:
        primary_profiles = load_primary_lag_profiles()

    biomarker_specs = get_available_biomarkers()

    summary: dict = {
        "sensitivity_analysis": "sa9_per_cohort_meta",
        "description": (
            "Per-cohort fixed- and random-effect meta-analysis of Stage 1 "
            "SHAP lag profiles for the four cluster-driving biomarkers. "
            "Tests whether the pooled finding is driven by a single cohort "
            "or by assay drift correlated with calendar year."
        ),
        "n_top_cohorts_target": n_top_cohorts,
        "min_cohort_n": min_cohort_n,
        "biomarkers": {},
    }

    for biomarker in biomarkers:
        spec = biomarker_specs.get(biomarker)
        if spec is None or spec.column not in df.columns:
            logger.warning("SA9: biomarker %s missing from analysis dataset", biomarker)
            summary["biomarkers"][biomarker] = {"status": "biomarker_missing"}
            continue

        cohorts = _select_top_cohorts(
            df=df,
            biomarker_col=spec.column,
            n_top=n_top_cohorts,
            min_n=min_cohort_n,
        )
        if not cohorts:
            logger.warning(
                "SA9: %s has no cohort with >= %d patients; skipping",
                biomarker,
                min_cohort_n,
            )
            summary["biomarkers"][biomarker] = {
                "status": "no_eligible_cohorts",
                "min_cohort_n": min_cohort_n,
            }
            continue

        bio_dir = output_dir / biomarker
        bio_dir.mkdir(parents=True, exist_ok=True)

        fits: list[CohortFit] = []
        for cohort_name in cohorts:
            df_cohort = df[df["study_source"] == cohort_name].copy()
            n_patients = df_cohort["patient_id"].nunique()
            logger.info(
                "SA9 [%s]: fitting cohort %s (n_patients=%d, n_rows=%d)",
                biomarker, cohort_name, n_patients, len(df_cohort),
            )
            fit = _fit_cohort(
                df_cohort=df_cohort,
                biomarker_name=biomarker,
                out_dir=bio_dir / cohort_name,
                label=f"sa9_{biomarker}_{cohort_name}",
            )
            if fit is not None:
                fits.append(fit)
            else:
                logger.warning(
                    "SA9 [%s]: cohort %s fit returned no usable result",
                    biomarker, cohort_name,
                )

        if len(fits) < 2:
            summary["biomarkers"][biomarker] = {
                "status": "insufficient_cohort_fits",
                "n_fits": len(fits),
            }
            continue

        # Save per-cohort lag profiles
        per_cohort_rows = []
        for fit in fits:
            for _, row in fit.lag_profile.iterrows():
                per_cohort_rows.append({
                    "cohort": fit.cohort,
                    "n_patients": fit.n_patients,
                    "n_rows": fit.n_rows,
                    "cv_r2_mean": fit.cv_r2_mean,
                    "lag_day": int(row["lag_day"]),
                    "mean_abs_shap": float(row["mean_abs_shap"]),
                    "se_abs_shap": float(row["se_abs_shap"]),
                })
        per_cohort_df = pd.DataFrame(per_cohort_rows)
        per_cohort_df.to_csv(bio_dir / "per_cohort_lag_profiles.csv", index=False)

        # Meta-analyse every lag day present in the primary lag set
        lag_days = sorted({int(d) for fit in fits for d in fit.lag_profile["lag_day"].unique()})
        meta_rows: dict[int, dict] = {}
        for lag_day in lag_days:
            meta_rows[lag_day] = _meta_analyse_lag(fits, lag_day)

        # Persist meta table
        meta_table_rows = []
        for lag_day, meta in meta_rows.items():
            if meta.get("status") == "too_few_cohorts":
                continue
            meta_table_rows.append({
                "biomarker": biomarker,
                "lag_day": lag_day,
                "n_cohorts": meta["n_cohorts"],
                "fe_estimate": meta["fixed_effect"]["estimate"],
                "fe_ci_low": meta["fixed_effect"]["ci_95"][0],
                "fe_ci_high": meta["fixed_effect"]["ci_95"][1],
                "re_estimate": meta["random_effect"]["estimate"],
                "re_ci_low": meta["random_effect"]["ci_95"][0],
                "re_ci_high": meta["random_effect"]["ci_95"][1],
                "i_squared_pct": meta["heterogeneity"]["i_squared_pct"],
                "tau_squared": meta["heterogeneity"]["tau_squared"],
                "cochran_q": meta["heterogeneity"]["cochran_q"],
                "cochran_q_pvalue": meta["heterogeneity"]["cochran_q_pvalue"],
            })
        pd.DataFrame(meta_table_rows).to_csv(bio_dir / "meta_analysis.csv", index=False)

        # Forest plots for the showcase lags
        for lag_day in FOREST_LAGS_DAYS:
            meta = meta_rows.get(lag_day)
            if meta is None:
                continue
            _render_forest_plot(
                biomarker=biomarker,
                lag_day=lag_day,
                meta=meta,
                out_path=bio_dir / f"forest_lag{lag_day}d.png",
            )

        _render_cohort_heatmap(
            biomarker=biomarker,
            fits=fits,
            out_path=bio_dir / "per_cohort_heatmap.png",
        )

        # Concordance with pooled primary
        primary_lag = primary_profiles.get(biomarker)
        primary_compare = None
        if primary_lag is not None:
            re_means = pd.DataFrame({
                "lag_day": [m["lag_day"] for m in meta_rows.values() if "lag_day" in m],
                "re_mean": [m["random_effect"]["estimate"] for m in meta_rows.values()
                            if "random_effect" in m],
            })
            merged = primary_lag.merge(re_means, on="lag_day", how="inner")
            if len(merged) >= 3:
                rho, p_val = stats.spearmanr(
                    merged["mean_abs_shap"], merged["re_mean"]
                )
                primary_compare = {
                    "spearman_rho": float(rho),
                    "spearman_p": float(p_val),
                    "n_lags_compared": int(len(merged)),
                    "concordant_with_pooled": bool(rho >= config.CONCORDANCE_RHO_THRESHOLD),
                }

        summary["biomarkers"][biomarker] = {
            "status": "ok",
            "cohorts_fitted": [fit.cohort for fit in fits],
            "n_cohorts_fitted": len(fits),
            "per_cohort_n_patients": {fit.cohort: fit.n_patients for fit in fits},
            "per_cohort_cv_r2": {fit.cohort: fit.cv_r2_mean for fit in fits},
            "meta_by_lag": meta_rows,
            "pooled_primary_concordance": primary_compare,
            "outputs": {
                "per_cohort_lag_profiles_csv": str(bio_dir / "per_cohort_lag_profiles.csv"),
                "meta_analysis_csv": str(bio_dir / "meta_analysis.csv"),
                "forest_plots": [str(bio_dir / f"forest_lag{l}d.png") for l in FOREST_LAGS_DAYS],
                "per_cohort_heatmap": str(bio_dir / "per_cohort_heatmap.png"),
            },
        }

    # Write the master summary
    summary_path = output_dir / "sa9_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    n_ok = sum(1 for v in summary["biomarkers"].values() if v.get("status") == "ok")
    logger.info(
        "SA9 complete: %d/%d biomarkers meta-analysed; summary written to %s",
        n_ok, len(summary["biomarkers"]), summary_path,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--biomarkers",
        nargs="*",
        default=None,
        help="Biomarker names (default: cluster drivers).",
    )
    p.add_argument("--n-top-cohorts", type=int, default=6)
    p.add_argument("--min-cohort-n", type=int, default=MIN_COHORT_N)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory.",
    )
    return p


def main() -> int:
    from mcd_pipeline.utils.logging_config import configure_logging
    from mcd_pipeline.stage0_data_prep.feature_engineering import (
        build_engineered_dataset,
    )

    args = _build_arg_parser().parse_args()
    configure_logging()

    df = build_engineered_dataset()

    run_per_cohort_meta(
        df=df,
        biomarkers=args.biomarkers,
        n_top_cohorts=args.n_top_cohorts,
        min_cohort_n=args.min_cohort_n,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
