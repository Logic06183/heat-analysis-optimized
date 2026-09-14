"""
Run All Sensitivity Analyses
============================

Orchestrates all 8 sensitivity analyses. Skips blocked analyses
(SA5 until imputation complete).

Usage:
    python -m mcd_pipeline.sensitivity.run_all_sensitivity
"""

import logging

from mcd_pipeline import config

logger = logging.getLogger(__name__)


def run_all_sensitivity(analysis_df=None) -> dict:
    """Run all enabled sensitivity analyses.

    Checks config.SENSITIVITY_ANALYSES flags. Skips disabled analyses
    with a logged warning.

    Returns
    -------
    dict
        Keyed by SA name with results or skip reason.
    """
    raise NotImplementedError("Sensitivity: run_all_sensitivity orchestrator")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_all_sensitivity()
