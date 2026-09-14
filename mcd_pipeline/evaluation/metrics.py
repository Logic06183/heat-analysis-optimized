"""
Evaluation Metrics — ML Performance and Clinical Relevance
==========================================================

Standard ML metrics (R², RMSE, MAE) plus domain-specific evaluation:
- Clinical significance thresholds per biomarker
- Epidemiological benchmarks (R² 0.05-0.15 typical for climate-health)
- Residual diagnostics

Adapted from archived reference: _archive/sep2025_scripts/climate_health_evaluation.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mcd_pipeline.stage0_data_prep.biomarker_definitions import BiomarkerSpec


def compute_cv_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard regression metrics.

    Returns dict with r2, rmse, mae, n_samples.
    """
    raise NotImplementedError("Evaluation: compute_cv_metrics")


def assess_clinical_significance(
    effect_size: float,
    biomarker: BiomarkerSpec,
) -> dict:
    """Assess whether the predicted temperature effect is clinically meaningful.

    Compares the effect size against the biomarker's clinical threshold.

    Returns dict with is_significant, threshold, effect_to_threshold_ratio.
    """
    raise NotImplementedError("Evaluation: assess_clinical_significance")
