"""08. Download the Universal Thermal Climate Index (UTCI) for the study area (appendix SA16)

Question    Reviewer 2 asked for a heat-stress index that reflects humidity,
            wind and radiation as well as air temperature.
Data in     ERA5-HEAT hourly UTCI from the Copernicus Climate Data Store,
            dataset `derived-utci-historical-timeseries` (v1.1 consolidated),
            accessed 10 November 2026. Needs a CDS account (~/.cdsapirc).
Steps       1. Find every 0.25-degree grid cell that holds an exposure location.
            2. Download the hourly series for each cell, March 2005 to December 2022.
            3. Convert Kelvin to degrees Celsius and reduce to daily mean,
               maximum and minimum (UTC days, as in the ERA5-Land linkage).
File out    utci/utci_daily_cells.csv (in UTCI_DIR; climate data only, not committed)
Next        09_utci_sensitivity.py
"""

import os
from pathlib import Path

import cdsapi
import pandas as pd

from r1_common import REPO

UTCI_DIR = Path(os.environ.get("R1_UTCI_DIR", Path(__file__).resolve().parents[1] / "utci"))
UTCI_DIR.mkdir(exist_ok=True)
GRID = 0.25
PERIOD = "2005-03-01/2022-12-31"
KELVIN = 273.15

# ---- 1. Grid cells that hold an exposure location --------------------------

locations = pd.read_csv(
    REPO / "FINAL_DATASETS/CLIMATE_HEALTH_LINKAGE/ERA5_LAND_LINKAGE.csv", usecols=["latitude", "longitude"]
).drop_duplicates()
cells = ((locations / GRID).round() * GRID).drop_duplicates().reset_index(drop=True)
print("grid cells:", len(cells), flush=True)

# ---- 2. Download one hourly series per cell --------------------------------

client = cdsapi.Client(quiet=True)
for cell in cells.itertuples():
    target = UTCI_DIR / f"utci_{cell.latitude:.2f}_{cell.longitude:.2f}.csv"
    already_downloaded = target.exists() and target.stat().st_size > 1e6
    if already_downloaded:
        continue
    request = {
        "variable": ["universal_thermal_climate_index"],
        "location": {"longitude": float(cell.longitude), "latitude": float(cell.latitude)},
        "date": [PERIOD],
        "data_format": "csv",
    }
    client.retrieve("derived-utci-historical-timeseries", request, str(target))
    print("downloaded", target.name, flush=True)

# ---- 3. Hourly Kelvin to daily degrees Celsius -----------------------------

daily_by_cell = []
# One file per grid cell, named utci_<latitude>_<longitude>.csv. The daily file written below is skipped.
cell_files = [p for p in sorted(UTCI_DIR.glob("utci_*.csv")) if p.name != "utci_daily_cells.csv"]
for path in cell_files:
    latitude, longitude = [float(part) for part in path.stem.split("_")[1:3]]
    hourly = pd.read_csv(path, parse_dates=["valid_time"])
    hourly["utci_c"] = hourly["utci"] - KELVIN
    hourly["date"] = hourly["valid_time"].dt.floor("D")
    daily = (
        hourly.groupby("date")["utci_c"].agg(utci_mean="mean", utci_max="max", utci_min="min").reset_index()
    )
    daily["cell_lat"] = latitude
    daily["cell_lon"] = longitude
    daily_by_cell.append(daily)

pd.concat(daily_by_cell).to_csv(UTCI_DIR / "utci_daily_cells.csv", index=False)
print("daily series written", flush=True)
