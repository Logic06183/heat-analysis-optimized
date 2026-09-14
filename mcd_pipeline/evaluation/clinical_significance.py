"""
Clinical Significance — Domain-Expert Thresholds
=================================================

Defines what constitutes a clinically meaningful change for each biomarker.
Used to contextualize ML model predictions: a statistically significant
SHAP effect that falls below the clinical threshold may not be actionable.

Thresholds sourced from clinical laboratory standards and existing
climate-health literature.

Adapted from archived reference: _archive/sep2025_scripts/advanced_evaluation_metrics.py
"""

from mcd_pipeline.stage0_data_prep.biomarker_definitions import BIOMARKERS


def get_clinical_thresholds() -> dict[str, float]:
    """Return clinical significance thresholds for all biomarkers.

    Returns dict of biomarker_name -> threshold in native units.
    """
    return {name: spec.clinical_threshold for name, spec in BIOMARKERS.items()
            if spec.clinical_threshold is not None}


def classify_effect(biomarker_name: str, effect_size: float) -> str:
    """Classify an effect as 'clinically significant', 'marginal', or 'negligible'.

    Returns one of: 'significant', 'marginal', 'negligible'.
    """
    raise NotImplementedError("Clinical significance: classify_effect")
