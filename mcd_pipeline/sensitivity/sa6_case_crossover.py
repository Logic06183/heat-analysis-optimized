"""
SA6: Case-Crossover Design — CD4 and Viral Load
================================================

Implements a time-stratified case-crossover design for CD4 count and
viral load, where each patient serves as their own control on different
days. This eliminates all time-invariant confounding.

Compares results with the primary XGBoost-SHAP approach.

MCD reference: "case-crossover for CD4/viral load"
"""


def run_case_crossover(analysis_df=None) -> dict:
    """Run case-crossover design for CD4 and viral load.

    Returns dict with odds ratios and comparison to SHAP results.
    """
    raise NotImplementedError("SA6: Case-crossover design")
