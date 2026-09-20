"""
SA5 / SA6: Within-person case-crossover — CD4 count and viral load
==================================================================

Implements a time-stratified, within-person case-crossover analysis for
CD4 count and viral load, with each patient stratified by calendar month
so that they serve as their own control. This conditioning eliminates
all time-invariant confounding (patient, season, study source) by design.

The estimator is a conditional logistic regression of a within-patient
case indicator (biomarker reading at or above the patient's own 75th
percentile) on same-day temperature (per 1 degC), conditioned on the
patient-month stratum. Only informative strata (those containing both a
case and a control) contribute to the conditional likelihood, as in any
matched / conditional design.

Honest limitation: control "days" here are the patient's OTHER VISIT days
within the month, because daily temperature on non-visit days is not in
the analysis dataset. This is a within-person visit-based case-crossover,
not a referent-day design. The available data leave the design
underpowered for both immunological markers; the analysis is reported as
inconclusive in the manuscript supplement (SA5).

History
-------
An earlier version of this module (pre 2026-05) reported a pooled
case-vs-control temperature difference and a per-observation Spearman
correlation. Those statistics did NOT condition on the patient-month
stratum, so they were confounded by exactly the between-stratum
variation a case-crossover is supposed to remove. The conditional
logistic estimator below is the correct estimator for this design and is
what the manuscript supplement now reports.

MCD reference: "case-crossover for CD4 / viral load".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mcd_pipeline import config

logger = logging.getLogger(__name__)

# Biomarkers analysed under the case-crossover design.
CASE_CROSSOVER_BIOMARKERS = ["cd4_count", "viral_load"]

# Same-day temperature column (per 1 degC), matching the primary Stage 1 lag.
TEMP_COL = "temp_lag_0d"

# "Case" = visit on which the biomarker is at or above the patient's own
# 75th percentile. Within-patient case definition removes between-patient
# differences in baseline biomarker level.
THRESHOLD_QUANTILE = 0.75


def _create_case_crossover_strata(
    df: pd.DataFrame,
    temp_col: str = TEMP_COL,
) -> pd.DataFrame:
    """Build the time-stratified case-crossover structure.

    Each patient-month is one stratum, so that conditioning on stratum
    in the downstream model removes between-patient and seasonal
    confounding. Imported by ``sa6b_conditional_logistic_or.py``; keep
    the signature stable.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``visit_date``, ``patient_id`` and the temperature
        column.
    temp_col : str
        Same-day temperature column. Unused inside this helper but kept
        for compatibility with the standalone script.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` augmented with ``year_month``, ``day_of_week``
        and ``stratum`` (patient x year-month).
    """
    df = df.copy()
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    df["year_month"] = df["visit_date"].dt.to_period("M")
    df["day_of_week"] = df["visit_date"].dt.dayofweek
    df["stratum"] = df["patient_id"].astype(str) + "_" + df["year_month"].astype(str)
    return df


def _conditional_logistic_or(
    df: pd.DataFrame,
    biomarker_col: str,
    temp_col: str = TEMP_COL,
    threshold_quantile: float = THRESHOLD_QUANTILE,
) -> dict:
    """Fit a conditional logistic regression and return the OR per degC.

    Defines a within-patient binary case (biomarker at or above the
    patient's own ``threshold_quantile``) and estimates the temperature
    effect via ``statsmodels.discrete.conditional_models.ConditionalLogit``
    conditioned on the patient-month stratum. Only strata containing at
    least one case AND at least one control contribute to the
    conditional likelihood; uninformative strata are dropped (this is a
    property of the design, not a filtering choice).

    Returns
    -------
    dict
        ``status`` and, on success, ``or_per_degree``, ``or_ci_lower``,
        ``or_ci_upper``, ``p_value``, ``n_observations``,
        ``n_informative_strata``, ``n_cases``, ``n_controls``,
        ``threshold_quantile``.
    """
    # Lazy import keeps statsmodels optional for users only running other SAs.
    from statsmodels.discrete.conditional_models import ConditionalLogit

    d = df[df[biomarker_col].notna()].copy()
    if len(d) < 100:
        return {"status": "insufficient_data", "n": int(len(d))}

    # Within-patient case definition: case = >= patient's own threshold.
    thr = d.groupby("patient_id")[biomarker_col].transform(
        lambda x: x.quantile(threshold_quantile)
    )
    d["case"] = (d[biomarker_col] >= thr).astype(int)

    cc = d[["case", temp_col, "stratum"]].dropna()
    # Informative strata only: must contain both a case and a control.
    nun = cc.groupby("stratum")["case"].transform("nunique")
    cc = cc[nun > 1]

    if cc["stratum"].nunique() < 20 or cc["case"].nunique() < 2:
        return {
            "status": "insufficient_informative_strata",
            "n_informative_strata": int(cc["stratum"].nunique()),
        }

    model = ConditionalLogit(
        cc["case"].to_numpy(),
        cc[[temp_col]].to_numpy(),
        groups=cc["stratum"].to_numpy(),
    )
    res = model.fit(disp=0)
    beta = float(res.params[0])
    lo, hi = (float(x) for x in res.conf_int()[0])
    return {
        "status": "success",
        "exposure": f"{temp_col} (per 1 degC)",
        "or_per_degree": float(np.exp(beta)),
        "or_ci_lower": float(np.exp(lo)),
        "or_ci_upper": float(np.exp(hi)),
        "p_value": float(res.pvalues[0]),
        "n_observations": int(len(cc)),
        "n_informative_strata": int(cc["stratum"].nunique()),
        "n_cases": int(cc["case"].sum()),
        "n_controls": int((cc["case"] == 0).sum()),
        "threshold_quantile": threshold_quantile,
    }


def run_case_crossover(
    df: pd.DataFrame,
    output_dir: Path = None,
) -> dict:
    """Run the within-person case-crossover for CD4 count and viral load.

    Workflow per biomarker:
    1. Build patient-month strata.
    2. Define within-patient cases (>= patient's own 75th percentile).
    3. Fit conditional logistic regression conditioned on stratum.
    4. Report OR per degC, 95% CI, p-value, and informative-strata counts.

    Returns
    -------
    dict
        Per-biomarker results plus methodological notes, written to
        ``mcd_outputs/sensitivity/sa6_case_crossover/sa6_summary.json``.
    """
    output_dir = output_dir or (config.OUTPUT_ROOT / "sensitivity")
    sa_output = output_dir / "sa6_case_crossover"
    sa_output.mkdir(parents=True, exist_ok=True)

    logger.info("=== SA5 / SA6: Within-person case-crossover ===")

    df_cc = _create_case_crossover_strata(df)

    if TEMP_COL not in df_cc.columns:
        logger.warning("  %s not in dataset — cannot run case-crossover", TEMP_COL)
        return {
            "sensitivity_analysis": "sa6_case_crossover",
            "status": "error",
            "error": f"{TEMP_COL} not in dataset",
        }

    results: dict = {}
    for bio_name in CASE_CROSSOVER_BIOMARKERS:
        if bio_name not in df_cc.columns:
            logger.warning("  %s not in dataset — skipping", bio_name)
            results[bio_name] = {"status": "column_not_found"}
            continue

        n_valid = df_cc[bio_name].notna().sum()
        if n_valid < 100:
            logger.warning("  %s: only %d valid rows — skipping", bio_name, n_valid)
            results[bio_name] = {"status": "insufficient_data", "n_valid": int(n_valid)}
            continue

        logger.info("  Processing %s (%d valid observations)...", bio_name, n_valid)
        results[bio_name] = _conditional_logistic_or(df_cc, bio_name)

        r = results[bio_name]
        if r.get("status") == "success":
            logger.info(
                "    OR %.3f per degC (95%% CI %.3f-%.3f), p=%.4g  "
                "[%d informative strata, %d cases / %d controls]",
                r["or_per_degree"], r["or_ci_lower"], r["or_ci_upper"],
                r["p_value"], r["n_informative_strata"],
                r["n_cases"], r["n_controls"],
            )
        else:
            logger.info("    %s: %s", bio_name, r.get("status", "unknown"))

    summary = {
        "sensitivity_analysis": "sa6_case_crossover",
        "design": (
            "Within-person, time-stratified case-crossover. "
            "Stratum = patient x calendar month; case = biomarker >= patient's "
            "own 75th percentile; exposure = same-day temperature per 1 degC; "
            "estimator = conditional logistic regression conditioned on stratum. "
            "Only informative strata (containing both a case and a control) "
            "contribute to the conditional likelihood."
        ),
        "limitation": (
            "Visit-based control structure: because daily temperature on "
            "non-visit days is not in the analysis dataset, control days are "
            "restricted to the patient's other clinic visits within the same "
            "month. This leaves the design underpowered for both immunological "
            "markers; the analysis is reported as inconclusive in the "
            "manuscript supplement (SA5)."
        ),
        "biomarkers_tested": CASE_CROSSOVER_BIOMARKERS,
        "per_biomarker": results,
    }

    with open(sa_output / "sa6_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("  SA5 / SA6 complete; written to %s", sa_output / "sa6_summary.json")
    return summary
