"""
SA4: COVID-19 Period Exclusion
==============================

Excludes 2020-2021 observations and re-runs Stage 1 to assess whether
COVID-era disruptions (lockdowns, healthcare access changes) affect results.

MCD reference: "COVID-19 period exclusion (2020–2021)"
"""

import logging
from pathlib import Path

import pandas as pd

from mcd_pipeline import config
from mcd_pipeline.sensitivity._sensitivity_core import run_subset_sensitivity

logger = logging.getLogger(__name__)


def run_covid_exclusion(
    df: pd.DataFrame,
    output_dir: Path = None,
    primary_profiles: dict = None,
) -> dict:
    """Exclude COVID years (2020–2021) and re-run Stage 1.

    South Africa's hard lockdowns (March 2020 onwards) disrupted
    healthcare access patterns. This tests whether COVID-era data
    changes temperature-biomarker relationships.
    """
    output_dir = output_dir or (config.OUTPUT_ROOT / "sensitivity")

    mask = ~df["year"].isin(config.COVID_EXCLUSION_YEARS)

    return run_subset_sensitivity(
        df_full=df,
        subset_mask=mask,
        sa_name="sa4_covid",
        sa_label=f"COVID exclusion (drop {config.COVID_EXCLUSION_YEARS})",
        output_dir=output_dir,
        primary_profiles=primary_profiles,
    )
