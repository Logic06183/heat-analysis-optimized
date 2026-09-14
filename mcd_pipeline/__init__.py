"""
MCD Pipeline — HE²AT Center Multi-System Biomarker Analysis
============================================================

Three-stage explainable ML pipeline for characterising temperature-biomarker
relationships in Johannesburg, as specified in the approved Manuscript Concept
Document (MCD), February 2026.

Pipeline stages:
    Stage 0: Data preparation & feature engineering
    Stage 1: XGBoost–SHAP lag-response profiling (per biomarker)
    Stage 2: SHAP interaction detection with permutation null
    Stage 3: Vulnerability clustering via PCA → k-means on SHAP vectors

Plus 8 sensitivity analyses and multiple testing correction.

Target journal: Environmental Data Science (Cambridge University Press)
Lead author: Craig Parker
"""

__version__ = "0.1.0"
