"""
Run Stage 3 — Orchestrate Vulnerability Clustering
===================================================

1. Aggregate person-level SHAP values across biomarkers
2. PCA (85% variance retained)
3. k-means (k=3..8, select by silhouette)
4. Bootstrap stability (500 iterations, threshold 0.80)
5. Characterise clusters by demographics/SES/geography

Usage:
    python -m mcd_pipeline.stage3_clustering.run_stage3
"""

import logging
from pathlib import Path

from mcd_pipeline import config

logger = logging.getLogger(__name__)


def run_stage3(
    stage1_dir: Path = config.OUTPUT_ROOT / "stage1",
    output_dir: Path = config.OUTPUT_ROOT / "stage3",
) -> dict:
    """Run Stage 3 vulnerability clustering.

    Requires Stage 1 outputs (SHAP values for all biomarkers).

    Returns
    -------
    dict
        optimal_k, silhouette_score, stability_ari, cluster_profiles,
        output_paths.
    """
    raise NotImplementedError("Stage 3: run_stage3 orchestrator")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_stage3()
