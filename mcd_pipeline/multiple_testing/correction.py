"""
Multiple Testing Correction — Bonferroni + FDR
===============================================

Two-level correction strategy from the MCD:

1. **Primary (system-level)**: Bonferroni across 8 biomarker systems.
   alpha = 0.05 / 8 = 0.00625. A system is "significant" if any biomarker
   within it passes this threshold.

2. **Secondary (within-system)**: FDR q=0.10 for individual biomarkers
   within significant systems, and for interaction effects.

MCD reference: "Multiple testing: Bonferroni alpha=0.00625 for 8 biomarker
systems (primary); FDR q=0.10 for individual biomarkers within significant
systems and for interaction effects"
"""

import logging

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import (
    BIOMARKER_SYSTEMS,
    BIOMARKERS,
)

logger = logging.getLogger(__name__)


def bonferroni_system_level(
    system_pvalues: dict[str, float],
    alpha: float = config.BONFERRONI_ALPHA,
) -> dict[str, bool]:
    """Apply Bonferroni correction at the system level.

    Each system's representative p-value (typically the minimum across
    biomarkers in that system) is compared against the Bonferroni-
    corrected threshold.

    Parameters
    ----------
    system_pvalues : dict
        System name -> minimum p-value across biomarkers in that system.
    alpha : float
        Bonferroni-corrected threshold (default: 0.05/8 = 0.00625).

    Returns
    -------
    dict
        System name -> bool (significant or not).
    """
    results = {}
    for system, pval in system_pvalues.items():
        results[system] = pval < alpha

    n_sig = sum(results.values())
    logger.info(
        f"Bonferroni system-level: {n_sig}/{len(results)} systems significant "
        f"(alpha={alpha:.4f})"
    )

    return results


def fdr_within_system(
    biomarker_pvalues: dict[str, float],
    q: float = config.FDR_Q_WITHIN_SYSTEMS,
) -> dict[str, bool]:
    """Apply Benjamini-Hochberg FDR within a significant system.

    Parameters
    ----------
    biomarker_pvalues : dict
        Biomarker name -> p-value (within one system).
    q : float
        FDR threshold (default: 0.10).

    Returns
    -------
    dict
        Biomarker name -> bool (significant after FDR).
    """
    if not biomarker_pvalues:
        return {}

    names = list(biomarker_pvalues.keys())
    pvals = np.array([biomarker_pvalues[n] for n in names])

    rejected, adjusted, _, _ = multipletests(pvals, alpha=q, method="fdr_bh")

    results = dict(zip(names, rejected.tolist()))

    n_sig = sum(results.values())
    logger.info(f"FDR within system: {n_sig}/{len(results)} biomarkers significant (q={q})")

    return results


def apply_full_correction(
    biomarker_pvalues: dict[str, float],
) -> pd.DataFrame:
    """Apply the complete two-level correction to biomarker p-values.

    Step 1: Group biomarkers by system, take minimum p-value per system.
    Step 2: Apply Bonferroni at system level.
    Step 3: For significant systems, apply FDR within that system.

    Parameters
    ----------
    biomarker_pvalues : dict
        Biomarker name -> raw p-value (from permutation testing or
        SHAP significance analysis).

    Returns
    -------
    pd.DataFrame
        Columns: biomarker, system, raw_pvalue, bonferroni_significant,
        fdr_significant, final_significant.
    """
    rows = []

    # Group biomarkers by system
    system_groups: dict[str, dict[str, float]] = {}
    for bm_name, pval in biomarker_pvalues.items():
        if bm_name in BIOMARKERS:
            system = BIOMARKERS[bm_name].system
        else:
            system = "unknown"
        system_groups.setdefault(system, {})[bm_name] = pval

    # Step 1: System-level minimum p-values
    system_min_pvals = {
        system: min(pvals.values())
        for system, pvals in system_groups.items()
    }

    # Step 2: Bonferroni at system level
    system_significant = bonferroni_system_level(system_min_pvals)

    # Step 3: FDR within significant systems
    for system, bm_pvals in system_groups.items():
        is_system_sig = system_significant.get(system, False)

        if is_system_sig:
            fdr_results = fdr_within_system(bm_pvals)
        else:
            fdr_results = {name: False for name in bm_pvals}

        for bm_name, raw_p in bm_pvals.items():
            rows.append({
                "biomarker": bm_name,
                "system": system,
                "raw_pvalue": raw_p,
                "bonferroni_significant": is_system_sig,
                "fdr_significant": fdr_results.get(bm_name, False),
                "final_significant": is_system_sig and fdr_results.get(bm_name, False),
            })

    result = pd.DataFrame(rows).sort_values("raw_pvalue").reset_index(drop=True)

    n_final = result["final_significant"].sum()
    logger.info(f"Full correction: {n_final}/{len(result)} biomarkers significant")

    return result
