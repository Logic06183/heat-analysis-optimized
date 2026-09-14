"""
Climate Indices — Derived Heat Stress Metrics
==============================================

Utility functions for computing heat stress indices from raw climate data.
These are already computed in the climate-linked datasets but may be needed
for sensitivity analyses using alternative climate sources.

Adapted from archived reference: _archive/sep2025_scripts/scientific_climate_indices.py
and FINAL_DATASETS/CLIMATE_HEALTH_LINKAGE/scripts/climate_extraction_utils.py
"""

import numpy as np


def apparent_temperature(t_air: float, rh: float = 50.0, ws: float = 1.0) -> float:
    """Steadman (1984) apparent temperature.

    Reference: Steadman, R.G. (1984) J.Clim.Appl.Meteorol. 23:1674-1687
    """
    raise NotImplementedError("Utils: apparent_temperature")


def heat_index(t_air: float, rh: float = 50.0) -> float:
    """Rothfusz (1990) / NOAA heat index.

    Reference: Rothfusz, L.P. (1990) NWS Tech.Attach. SR 90-23
    """
    raise NotImplementedError("Utils: heat_index")


def wet_bulb_temperature(t_air: float, rh: float = 50.0) -> float:
    """Stull (2011) wet bulb temperature approximation.

    Reference: Stull, R. (2011) J.Appl.Meteorol.Clim. 50:2267-2269
    """
    raise NotImplementedError("Utils: wet_bulb_temperature")
