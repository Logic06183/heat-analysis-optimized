"""
SA8: Heat Wave Threshold Comparison — 95th vs 90th Percentile
==============================================================

Compares heat wave definitions using 90th vs 95th percentile thresholds
to assess sensitivity of results to heat wave classification.

MCD reference: "heat wave threshold 95th vs 90th percentile"
"""

from mcd_pipeline import config


def run_heatwave_threshold_comparison(analysis_df=None) -> dict:
    """Compare Stage 1 results using 90th vs 95th percentile heat wave flags.

    Returns dict with comparison metrics.
    """
    raise NotImplementedError("SA8: Heatwave threshold comparison")
