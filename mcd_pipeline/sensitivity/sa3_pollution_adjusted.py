"""
SA3: Pollution-Adjusted Subset — PM2.5/NO2 Confounding
======================================================

Restricts analysis to 2007-2021 (when air quality data available)
and re-runs Stage 1 to assess whether pollution confounds
temperature-biomarker relationships.

MCD reference: "pollution-adjusted subset (2007–2021, PM₂.₅/NO₂)"
"""

from mcd_pipeline import config


def run_pollution_adjusted(analysis_df=None) -> dict:
    """Subset to pollution data years and re-run Stage 1.

    Returns dict with comparison metrics.
    """
    raise NotImplementedError("SA3: Pollution-adjusted subset")
