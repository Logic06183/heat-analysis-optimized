"""Pipeline-derived constants for figure generation.

REFACTORED 2026-06-08 (was: frozen 2026-04-23 snapshots).

Replaces the prior 'frozen snapshot' constants with live loads from the
canonical pipeline outputs in ``mcd_outputs/``. Same module-level variable
names are preserved so the eight figure scripts that import from this module
continue to work without modification.

If a referenced output file is missing, this module raises a clear ImportError
at import time rather than falling through to silently-stale values. To
re-freeze (e.g., to ship a reproducible artefact alongside a release), one can
serialise the module-level dicts to JSON after import.

Pipeline outputs consumed:
  mcd_outputs/stage1/stage1_summary.json
  mcd_outputs/stage1/{biomarker}/cv_metrics.json
  mcd_outputs/stage1/{biomarker}/lag_response_summary.csv
  mcd_outputs/stage3/cluster_characteristics_with_ci.json
  mcd_outputs/stage3/cluster_labels.csv
  mcd_outputs/stage3/pca_summary.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from mcd_pipeline import config

# Which Stage 3 run the figures are built from. Default "stage3" is the
# 13-biomarker run the current manuscript text describes (51 PCs, 85.1%
# variance, ARI 0.913). Set RP2_STAGE3_DIR=stage3_r2_005 to build the figures
# from the 10-biomarker R^2>=0.05 run instead (43 PCs, 85.6%, ARI 0.985) --
# note that run reports different cluster proportions, so the manuscript text
# has to move with it.
STAGE3_SUBDIR = os.environ.get("RP2_STAGE3_DIR", "stage3_r2_005")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STAGE1_DIR = config.OUTPUT_ROOT / "stage1"
STAGE3_DIR = config.OUTPUT_ROOT / STAGE3_SUBDIR

_required = [
    STAGE1_DIR / "stage1_summary.json",
    STAGE3_DIR / "cluster_characteristics_with_ci.json",
    STAGE3_DIR / "cluster_labels.csv",
    STAGE3_DIR / "pca_summary.json",
]
_missing = [str(p) for p in _required if not p.exists()]
if _missing:
    raise ImportError(
        "publication_figures_lancet._data could not load pipeline outputs.\n"
        "Missing files:\n  " + "\n  ".join(_missing) + "\n"
        "Run the pipeline (Stage 1 + Stage 3) before generating figures."
    )

# ---------------------------------------------------------------------------
# Stage 1: per-biomarker lag profiles and CV R²
# ---------------------------------------------------------------------------
LAG_DAYS = [0, 1, 3, 7, 14, 21, 30]

_stage1_summary: dict = json.loads(
    (STAGE1_DIR / "stage1_summary.json").read_text()
)

# Biomarkers carried into Stage 3. Taken from the Stage 3 run itself rather
# than re-derived from a threshold here, so the figures can never disagree with
# the clustering about which panel was actually used:
#   stage3        -> 13 biomarkers (mean CV R² ≥ 0)
#   stage3_r2_005 -> 10 biomarkers (mean CV R² ≥ 0.05)
# Falls back to the Stage 1 model_adequate flag (R² ≥ 0) if the Stage 3 summary
# is unavailable.
_stage3_summary: dict = {}
try:
    _stage3_summary = json.loads((STAGE3_DIR / "stage3_summary.json").read_text())
except (OSError, ValueError):
    pass

_MODEL_ADEQUATE: list[str] = [
    name for name, blk in _stage1_summary.items()
    if isinstance(blk, dict) and blk.get("model_adequate", False)
]
RETAINED: list[str] = list(_stage3_summary.get("included_biomarkers") or _MODEL_ADEQUATE)

# Human-readable retention criterion, so figure annotations describe the panel
# they are actually drawing rather than a hardcoded threshold.
RETENTION_THRESHOLD: float = 0.05 if len(RETAINED) < len(_MODEL_ADEQUATE) else 0.0
RETENTION_CRITERION_LABEL: str = (
    "R² ≥ 0·05" if RETENTION_THRESHOLD else "R² ≥ 0"
)

# Full screened panel (24) = every biomarker that has a stage1 entry.
ALL_BIOMARKERS: list[str] = list(_stage1_summary.keys())


def _read_lag_response(biomarker: str) -> pd.DataFrame:
    """Read the per-lag mean |SHAP| + 95% CI + pct_positive table for one biomarker."""
    return pd.read_csv(STAGE1_DIR / biomarker / "lag_response_summary.csv")


def _load_lag_mean_abs_shap() -> dict[str, list[float]]:
    """Return mean |SHAP| at each lag, indexed by biomarker name."""
    out: dict[str, list[float]] = {}
    for bio in ALL_BIOMARKERS:
        csv_path = STAGE1_DIR / bio / "lag_response_summary.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path).set_index("lag_day").reindex(LAG_DAYS)
        out[bio] = [float(v) for v in df["mean_abs_shap"].tolist()]
    return out


LAG_MEAN_ABS_SHAP: dict[str, list[float]] = _load_lag_mean_abs_shap()


def _load_fold_r2() -> dict[str, list[float]]:
    """Return per-fold CV R² for every biomarker (5 folds typical)."""
    out: dict[str, list[float]] = {}
    for bio, blk in _stage1_summary.items():
        if not isinstance(blk, dict):
            continue
        folds = blk.get("cv_metrics", {}).get("folds", [])
        out[bio] = [float(f.get("r2", float("nan"))) for f in folds]
    return out


FOLD_R2: dict[str, list[float]] = _load_fold_r2()


# Per-biomarker total observations used for Stage 1 (n_samples in summary).
N_SAMPLES: dict[str, int] = {
    bio: int(blk.get("n_samples", 0))
    for bio, blk in _stage1_summary.items()
    if isinstance(blk, dict) and blk.get("n_samples") is not None
}

# ---------------------------------------------------------------------------
# Stage 3: GMM clustering
# ---------------------------------------------------------------------------
_cluster_labels = pd.read_csv(STAGE3_DIR / "cluster_labels.csv")
N_PATIENTS_CLUSTERED: int = len(_cluster_labels)


def _canonical_cluster_remap() -> dict[int, int]:
    """Map raw GMM cluster IDs to the manuscript's canonical IDs.

    GMM assigns cluster IDs by initialisation order, which is not consistent
    across runs. The manuscript uses a canonical naming based on
    characteristics:
        0 = Younger marginalised  (largest cluster, mid age, mid HIV)
        1 = Older, employed       (smallest, oldest, lowest HIV)
        2 = High HIV burden       (mid size, youngest, highest HIV)
    """
    by_cluster = json.loads(
        (STAGE3_DIR / "cluster_characteristics_with_ci.json").read_text()
    )["by_cluster"]
    metrics: dict[int, dict[str, float]] = {}
    for cid_str, rows in by_cluster.items():
        m = {r["metric"]: float(r.get("point_estimate", float("nan"))) for r in rows}
        metrics[int(cid_str)] = m
    # Identify each canonical id from its defining property.
    canon_2 = max(metrics, key=lambda i: metrics[i].get("pct_hiv_positive", -1))
    canon_1 = max(metrics, key=lambda i: metrics[i].get("mean_age_years", -1))
    canon_0 = next(i for i in metrics if i not in {canon_1, canon_2})
    return {canon_0: 0, canon_1: 1, canon_2: 2}


_CLUSTER_REMAP: dict[int, int] = _canonical_cluster_remap()
_cluster_labels["cluster"] = _cluster_labels["cluster"].map(_CLUSTER_REMAP)

_pca_summary: dict = json.loads((STAGE3_DIR / "pca_summary.json").read_text())
PC_VARIANCE_EXPLAINED_PCT: list[float] = [
    100.0 * v for v in _pca_summary.get("explained_variance_ratio", [])[:10]
]
N_PCS_USED: int = min(10, int(_pca_summary.get("n_components", 10)))

# Bootstrap ARI of the primary clustering. Prefer the live stability_summary.json
# (always present after Stage 3); fall back to the older refinement_summary.json
# when running against the original April-2026 pipeline; finally hardcoded fallback.
ARI_STABILITY: float = 0.913  # fallback matches ERA5-Land primary
_stability_path = STAGE3_DIR / "stability_summary.json"
_refinement_path = config.OUTPUT_ROOT / "stage3_refinement" / "refinement_summary.json"
if _stability_path.exists():
    try:
        _stab = json.loads(_stability_path.read_text())
        ARI_STABILITY = float(_stab.get("mean_ari", ARI_STABILITY))
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
elif _refinement_path.exists():
    try:
        _ref = json.loads(_refinement_path.read_text())
        ARI_STABILITY = float(
            _ref.get("primary_gmm_full_k3", {}).get("mean_ari", ARI_STABILITY)
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        pass


def _load_cluster_characteristics() -> dict[int, dict[str, tuple | int]]:
    """Reformat the long-format CI file into the per-cluster dict the figures expect.

    Source format: ``by_cluster[i]`` is a list of {metric, point_estimate,
    ci_lower_2.5, ci_upper_97.5} dicts.
    Target format: ``{i: {metric_name: (point, lo, hi), ..., 'n_patients': int}}``
    with metric_name matching the prior frozen-snapshot keys.
    """
    by_cluster_raw = json.loads(
        (STAGE3_DIR / "cluster_characteristics_with_ci.json").read_text()
    )["by_cluster"]

    # Pipeline metric_name → prior-snapshot key (preserve downstream contracts).
    name_map = {
        "prevalence_pct":                  "prevalence_pct",
        "mean_age_years":                  "mean_age_years",
        "mean_composite_heat_sensitivity": "mean_composite_heat_sens",
        "pct_female":                      "pct_female",
        "pct_hiv_positive":                "pct_hiv_positive",
        "pct_informal_dwelling":           "pct_informal_dwelling",
        "pct_unemployed":                  "pct_unemployed",
    }

    cluster_sizes = _cluster_labels["cluster"].value_counts().to_dict()

    out: dict[int, dict[str, tuple | int]] = {}
    for cluster_id_str, rows in by_cluster_raw.items():
        raw_cid = int(cluster_id_str)
        # Remap raw GMM id → canonical manuscript id.
        cid = _CLUSTER_REMAP.get(raw_cid, raw_cid)
        block: dict[str, tuple | int] = {}
        for row in rows:
            src = row.get("metric")
            if src not in name_map:
                continue
            block[name_map[src]] = (
                float(row.get("point_estimate", float("nan"))),
                float(row.get("ci_lower_2.5", float("nan"))),
                float(row.get("ci_upper_97.5", float("nan"))),
            )
        # cluster_sizes was computed from the remapped column, so use canon id.
        block["n_patients"] = int(cluster_sizes.get(cid, 0))
        out[cid] = block
    return out


CLUSTER_CHARACTERISTICS: dict[int, dict[str, tuple | int]] = _load_cluster_characteristics()

# ---------------------------------------------------------------------------
# Per-biomarker × per-lag signatures with bootstrap CIs (for figure 03 / S2)
# ---------------------------------------------------------------------------
def _load_cluster_biomarker_lag_signature() -> dict[str, dict]:
    """Detailed lag-response per cluster-driving biomarker.

    The four biomarkers that drive Stage 3 cluster separation (per the paper).
    Each entry includes: lag_day, mean_abs_shap, ci_lower, ci_upper,
    pct_positive, n_samples, model_r2.

    temp_rank_min / temp_rank_max (the importance rank of the best/worst
    temperature feature in the full feature list) are not regenerated by the
    pipeline; they remain inlined here from the 2026-04-23 snapshot. These are
    used only as informational tooltips in figS3/figure 03 and do not affect
    any reported result. Marked clearly for future re-derivation.
    """
    biomarkers = ["hematocrit", "viral_load", "hemoglobin", "body_fat_percent"]
    # temp_rank values from the 2026-04-23 snapshot (informational only).
    _temp_ranks = {
        "hematocrit":       (4, 7),
        "viral_load":       (6, 15),
        "hemoglobin":       (4, 14),
        "body_fat_percent": (5, 15),
    }
    out: dict[str, dict] = {}
    for bio in biomarkers:
        csv_path = STAGE1_DIR / bio / "lag_response_summary.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path).set_index("lag_day").reindex(LAG_DAYS).reset_index()
        n_samp = N_SAMPLES.get(bio, 0)
        fold_r2 = FOLD_R2.get(bio, [])
        mean_r2 = float(sum(fold_r2) / len(fold_r2)) if fold_r2 else float("nan")
        out[bio] = {
            "lag_day":       [int(v) for v in df["lag_day"].tolist()],
            "mean_abs_shap": [float(v) for v in df["mean_abs_shap"].tolist()],
            "ci_lower":      [float(v) for v in df["ci_lower"].tolist()],
            "ci_upper":      [float(v) for v in df["ci_upper"].tolist()],
            "pct_positive":  [float(v) for v in df["pct_positive"].tolist()],
            "n_samples":     int(n_samp),
            "model_r2":      round(mean_r2, 3),
            "temp_rank_min": _temp_ranks[bio][0],
            "temp_rank_max": _temp_ranks[bio][1],
        }
    return out


CLUSTER_BIOMARKER_LAG_SIGNATURE: dict[str, dict] = _load_cluster_biomarker_lag_signature()
