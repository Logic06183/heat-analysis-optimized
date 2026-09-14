"""
SA2: MODIS LST vs ERA5 — Urban Heat Island Sensitivity
======================================================

Re-runs Stage 1 using MODIS Land Surface Temperature (1 km resolution)
instead of ERA5 air temperature (28 km resolution). Differences indicate
urban heat island effects not captured at coarser resolution.

MCD reference: "MODIS LST vs ERA5 for urban heat island effects"
"""


def run_modis_comparison(analysis_df=None) -> dict:
    """Re-run Stage 1 with MODIS LST features and compare SHAP profiles.

    Returns
    -------
    dict
        Per-biomarker comparison of SHAP rankings: ERA5 vs MODIS.
    """
    raise NotImplementedError("SA2: MODIS vs ERA5")
