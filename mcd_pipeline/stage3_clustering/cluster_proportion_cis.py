"""
Stage 3 — Bootstrap 95% CIs for Cluster Proportions and Characteristics
========================================================================

Addresses the Reviewer 2 (Lancet Planetary Health) request to report
95% CIs for cluster proportions and cluster-level characteristics in
Table 3, so point estimates are not over-interpreted.

For each GMM cluster we compute non-parametric bootstrap CIs for:
  - Cluster prevalence (% of cohort)
  - Mean age
  - % female
  - HIV prevalence (%)
  - % informal dwelling
  - % unemployed
  - Cluster composite heat-sensitivity score (mean of PC1-3 weights)

Resampling is at the patient level (independent rows), so the bootstrap
respects the unit of analysis used in Stage 3 clustering.

Outputs
-------
mcd_outputs/stage3/cluster_characteristics_with_ci.csv
mcd_outputs/stage3/cluster_characteristics_with_ci.json
mcd_outputs/stage3/cluster_proportions_bootstrap.npy

Run with:
    python -m mcd_pipeline.stage3_clustering.cluster_proportion_cis \
        --n-bootstrap 2000 --output-format both
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from mcd_pipeline import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default characterisation columns
# ---------------------------------------------------------------------------

CONTINUOUS_FEATURES: tuple[str, ...] = (
    "age_years",
    "composite_heat_sensitivity",
)
CATEGORICAL_FEATURES_RULES: dict[str, callable] = {
    "pct_female": lambda s: 100.0 * (s.astype(str).str.lower() == "female").mean(),
    "pct_hiv_positive": lambda s: 100.0 * (s.astype(str).str.lower() == "positive").mean(),
    "pct_informal_dwelling": lambda s: 100.0 * (
        s.astype(str).str.lower().isin({"informal", "informal_settlement"})
    ).mean(),
    "pct_unemployed": lambda s: 100.0 * (
        s.astype(str).str.lower().isin({"unemployed", "not_employed", "not employed"})
    ).mean(),
}
CATEGORICAL_FEATURE_SOURCES: dict[str, str] = {
    "pct_female": "sex",
    "pct_hiv_positive": "hiv_status",
    "pct_informal_dwelling": "gcro_dwelling_type",
    "pct_unemployed": "gcro_employment_status",
}


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a placeholder composite_heat_sensitivity column if missing.

    The column is created by run_stage3.py; its absence here typically
    means we are running on a partially completed Stage 3.
    """
    if "composite_heat_sensitivity" not in df.columns:
        logger.warning(
            "composite_heat_sensitivity column missing — substituting NaN; "
            "summary will show NaN for this metric."
        )
        df = df.copy()
        df["composite_heat_sensitivity"] = np.nan
    return df


def _summarise_cluster(
    df_cluster: pd.DataFrame,
    n_total: int,
) -> dict[str, float]:
    """Compute the headline statistics for a single cluster on one resample."""
    out: dict[str, float] = {
        "prevalence_pct": 100.0 * len(df_cluster) / max(n_total, 1),
    }
    for col in CONTINUOUS_FEATURES:
        if col in df_cluster.columns:
            out[f"mean_{col}"] = float(df_cluster[col].mean())
        else:
            out[f"mean_{col}"] = float("nan")

    for metric, source_col in CATEGORICAL_FEATURE_SOURCES.items():
        if source_col in df_cluster.columns:
            rule = CATEGORICAL_FEATURES_RULES[metric]
            out[metric] = float(rule(df_cluster[source_col]))
        else:
            out[metric] = float("nan")

    return out


def bootstrap_cluster_characteristics(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    n_bootstrap: int = 2000,
    seed: int = config.MASTER_SEED,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute bootstrap 95% CIs for each cluster's headline statistics.

    Patient-level non-parametric bootstrap with replacement; the resampled
    DataFrame preserves the cluster labels carried by each patient.

    Parameters
    ----------
    df : DataFrame
        Per-patient frame containing cluster_col plus the demographic /
        SES columns referenced in CATEGORICAL_FEATURE_SOURCES, plus any
        of CONTINUOUS_FEATURES that are available.
    cluster_col : str
        Column with cluster assignments (integer or category).
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    DataFrame
        Tidy table with columns
            cluster, metric, point_estimate, mean, ci_lower, ci_upper, se
    ndarray
        Cluster x metric x n_bootstrap array of bootstrap statistics.
    """
    df = _ensure_features(df)
    rng = np.random.default_rng(seed)

    df = df.dropna(subset=[cluster_col]).copy()
    df[cluster_col] = df[cluster_col].astype(int)
    cluster_ids = sorted(df[cluster_col].unique())

    # Pre-compute point estimates from the unresampled data so reviewers
    # have the table they were already shown plus the new CIs.
    point_summary: dict[int, dict[str, float]] = {
        cid: _summarise_cluster(df[df[cluster_col] == cid], n_total=len(df))
        for cid in cluster_ids
    }
    metric_names = list(next(iter(point_summary.values())).keys())

    # Bootstrap loop
    boot_stats = np.full((len(cluster_ids), len(metric_names), n_bootstrap), np.nan)
    n_total = len(df)
    indices = np.arange(n_total)

    for b in range(n_bootstrap):
        sample_idx = rng.integers(0, n_total, size=n_total)
        df_boot = df.iloc[sample_idx]
        cluster_groups = df_boot.groupby(cluster_col)
        for ci, cid in enumerate(cluster_ids):
            if cid not in cluster_groups.groups:
                continue
            g = cluster_groups.get_group(cid)
            stats = _summarise_cluster(g, n_total=len(df_boot))
            for mi, metric in enumerate(metric_names):
                boot_stats[ci, mi, b] = stats.get(metric, np.nan)

        if (b + 1) % 200 == 0:
            logger.info(
                "  Bootstrap %d/%d clusters x %d metrics", b + 1, n_bootstrap, len(metric_names)
            )

    rows: list[dict] = []
    for ci, cid in enumerate(cluster_ids):
        for mi, metric in enumerate(metric_names):
            samples = boot_stats[ci, mi, :]
            samples = samples[~np.isnan(samples)]
            if samples.size == 0:
                rows.append({
                    "cluster": cid,
                    "metric": metric,
                    "point_estimate": point_summary[cid].get(metric, float("nan")),
                    "mean": float("nan"),
                    "ci_lower_2.5": float("nan"),
                    "ci_upper_97.5": float("nan"),
                    "se": float("nan"),
                    "n_bootstrap_used": 0,
                })
                continue
            rows.append({
                "cluster": cid,
                "metric": metric,
                "point_estimate": point_summary[cid].get(metric, float("nan")),
                "mean": float(np.mean(samples)),
                "ci_lower_2.5": float(np.percentile(samples, 2.5)),
                "ci_upper_97.5": float(np.percentile(samples, 97.5)),
                "se": float(np.std(samples, ddof=1)),
                "n_bootstrap_used": int(samples.size),
            })

    return pd.DataFrame(rows), boot_stats


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def run_cluster_proportion_cis(
    stage3_dir: Path | None = None,
    n_bootstrap: int = 2000,
    seed: int = config.MASTER_SEED,
) -> dict:
    """Load the Stage 3 outputs and write the bootstrap CI table.

    Reads cluster_labels.csv, vulnerability_scores.csv, and the patient-
    level characterisation table produced by characterise_clusters().
    """
    stage3_dir = stage3_dir or (config.OUTPUT_ROOT / "stage3")

    cluster_path = stage3_dir / "cluster_labels.csv"
    vuln_path = stage3_dir / "vulnerability_scores.csv"
    characterisation_path = stage3_dir / "patient_cluster_table.csv"

    if not cluster_path.exists():
        raise FileNotFoundError(
            f"cluster_labels.csv not found at {cluster_path}; run Stage 3 first"
        )

    cluster_labels = pd.read_csv(cluster_path)
    cluster_labels = cluster_labels.set_index(cluster_labels.columns[0])
    cluster_labels.columns = ["cluster"]

    if characterisation_path.exists():
        _char_raw = pd.read_csv(characterisation_path)
        characterisation = _char_raw.set_index("patient_id").drop(
            columns=["cluster"], errors="ignore"
        )
    else:
        # Reconstruct patient_cluster_table.csv from the analysis dataset so
        # that categorical bootstrap CIs (HIV %, sex, dwelling) are available.
        logger.warning(
            "patient_cluster_table.csv not found at %s — building from "
            "TIDY inputs and saving for future use.",
            characterisation_path,
        )
        from mcd_pipeline.stage0_data_prep.build_analysis_dataset import (
            build_analysis_dataset,
        )
        _raw = build_analysis_dataset()
        _demo_cols = (
            ["age_years", "sex", "hiv_status"]
            + [c for c in _raw.columns if c.startswith("gcro_")]
        )
        _per_patient = (
            _raw[["patient_id"] + [c for c in _demo_cols if c in _raw.columns]]
            .drop_duplicates(subset="patient_id", keep="first")
            .set_index("patient_id")
        )
        _per_patient = _per_patient.reindex(cluster_labels.index)
        _per_patient["cluster"] = cluster_labels["cluster"].values
        _per_patient.to_csv(characterisation_path)
        characterisation = _per_patient.drop(columns=["cluster"], errors="ignore")
        logger.info(
            "Built and saved patient_cluster_table.csv for %d patients",
            len(characterisation),
        )

    if vuln_path.exists():
        _vuln = pd.read_csv(vuln_path)
        vuln = _vuln.set_index(_vuln.columns[0])
    else:
        logger.warning("vulnerability_scores.csv not found at %s", vuln_path)
        vuln = pd.DataFrame(index=cluster_labels.index)

    df = cluster_labels.join(characterisation, how="left").join(vuln, how="left")

    logger.info(
        "Bootstrapping cluster characteristics on %d patients (n_bootstrap=%d)",
        len(df), n_bootstrap,
    )
    table, raw_samples = bootstrap_cluster_characteristics(
        df=df,
        cluster_col="cluster",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    csv_path = stage3_dir / "cluster_characteristics_with_ci.csv"
    json_path = stage3_dir / "cluster_characteristics_with_ci.json"
    npy_path = stage3_dir / "cluster_proportions_bootstrap.npy"

    table.to_csv(csv_path, index=False)

    grouped = {
        int(cid): table[table["cluster"] == cid].to_dict(orient="records")
        for cid in sorted(table["cluster"].unique())
    }
    with open(json_path, "w") as f:
        json.dump(
            {
                "n_bootstrap": n_bootstrap,
                "seed": seed,
                "by_cluster": grouped,
            },
            f,
            indent=2,
            default=str,
        )

    np.save(npy_path, raw_samples)

    logger.info(
        "Cluster CI table written to %s (%d rows); raw bootstrap saved to %s",
        csv_path, len(table), npy_path,
    )
    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "npy_path": str(npy_path),
        "n_clusters": int(table["cluster"].nunique()),
        "n_bootstrap": n_bootstrap,
        "n_patients": int(len(df)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stage3-dir", type=Path, default=None)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=config.MASTER_SEED)
    return p


def main() -> int:
    from mcd_pipeline.utils.logging_config import configure_logging

    configure_logging()
    args = _build_arg_parser().parse_args()
    run_cluster_proportion_cis(
        stage3_dir=args.stage3_dir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
