"""
SA4: COVID-19 Period Exclusion
==============================

Excludes 2020-2021 observations and re-runs Stage 1 to assess whether
COVID-era disruptions (lockdowns, healthcare access changes) affect results.

MCD reference: "COVID-19 period exclusion (2020–2021)"
"""

from mcd_pipeline import config


def run_covid_exclusion(analysis_df=None) -> dict:
    """Exclude COVID years and re-run Stage 1.

    Returns dict with comparison metrics.
    """
    raise NotImplementedError("SA4: COVID exclusion")
