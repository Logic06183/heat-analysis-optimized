"""
ERA5-Land Extraction for SA-resolution Sensitivity Analysis
============================================================

Extracts ECMWF/ERA5_LAND/DAILY_AGGR daily 2-m air temperature (~9 km / 0.1°)
at the 102 unique ward centroid locations from the TIDY climate dataset.

Produces ERA5_LAND_LINKAGE.csv with the same lag structure as the primary
ERA5 (lags 0, 1, 3, 7, 14, 21, 30 days), linked by (study_source, patient_id,
visit_date). Intended as a higher-resolution-reanalysis sensitivity analysis
bracketing the primary ERA5 (~31 km) and MODIS LST (1 km):

    ERA5 (31 km, air temp, hourly→daily)    — primary, SA0
    ERA5-Land (9 km, air temp, daily)       — NEW, this script
    MODIS LST (1 km, land surface temp)     — SA2

Usage:
    python -m mcd_pipeline.utils.extract_era5_land

Output:
    FINAL_DATASETS/CLIMATE_HEALTH_LINKAGE/ERA5_LAND_LINKAGE.csv

Bands extracted (all in Kelvin, converted to °C inline):
    temperature_2m      → era5land_temp_mean_c
    temperature_2m_max  → era5land_temp_max_c
    temperature_2m_min  → era5land_temp_min_c
    (DTR derived as max − min)

Notes
-----
* DAILY_AGGR aggregates the hourly ERA5-Land product. Daily mean/max/min are
  produced upstream by ECMWF, so we avoid the noise of doing it ourselves.
* ERA5-Land coverage extends 1950–present (gap-free, no cloud masking
  required — unlike MODIS).
* Many of the 102 ward centroids will snap to the same ~9 km pixel; we
  preserve the mapping back to original ward coordinates for joining.
* Extraction is chunked by calendar year to keep `getRegion` payloads under
  GEE's 5,000-element limit.
"""

import logging
from pathlib import Path

import ee
import numpy as np
import pandas as pd

from mcd_pipeline import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ERA5_LAND_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
TEMP_BANDS = ["temperature_2m", "temperature_2m_max", "temperature_2m_min"]
TEMP_BAND_LABELS = {
    "temperature_2m":     "era5land_temp_mean_c",
    "temperature_2m_max": "era5land_temp_max_c",
    "temperature_2m_min": "era5land_temp_min_c",
}
KELVIN_OFFSET = 273.15
NATIVE_SCALE_M = 11132  # ~0.1° at equator; GEE-recommended scale for ERA5-Land

# Lag windows must mirror the primary ERA5 schedule.
LAG_DAYS = [0, 1, 3, 7, 14, 21, 30]

OUTPUT_PATH = (
    config.FINAL_DATASETS
    / "CLIMATE_HEALTH_LINKAGE"
    / "ERA5_LAND_LINKAGE.csv"
)


# ---------------------------------------------------------------------------
# Step 1 — load unique ward centroids (mirrors MODIS extraction)
# ---------------------------------------------------------------------------
def load_ward_locations() -> pd.DataFrame:
    clim = pd.read_csv(config.TIDY_CLIMATE_CSV, low_memory=False)
    locs = (
        clim[["latitude", "longitude"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .assign(ward_id=lambda df: df.index)
    )
    logger.info("Loaded %d unique ward locations", len(locs))
    return locs


# ---------------------------------------------------------------------------
# Step 2 — extract ERA5-Land daily time series via GEE
# ---------------------------------------------------------------------------
def _make_feature_collection(locs: pd.DataFrame) -> ee.FeatureCollection:
    features = []
    for _, row in locs.iterrows():
        features.append(
            ee.Feature(
                ee.Geometry.Point([row["longitude"], row["latitude"]]),
                {"ward_id": int(row["ward_id"])},
            )
        )
    return ee.FeatureCollection(features)


def _to_celsius(image: ee.Image) -> ee.Image:
    out = image.select(TEMP_BANDS).subtract(KELVIN_OFFSET)
    return out.copyProperties(image, ["system:time_start"])


def _extract_year(
    fc: ee.FeatureCollection,
    year: int,
) -> pd.DataFrame:
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    collection = (
        ee.ImageCollection(ERA5_LAND_COLLECTION)
        .filterDate(start, end)
        .map(_to_celsius)
    )
    region = collection.getRegion(fc, scale=NATIVE_SCALE_M).getInfo()
    if not region or len(region) < 2:
        return pd.DataFrame(
            columns=["feature_id", "longitude", "latitude", "time"] + TEMP_BANDS
        )
    return pd.DataFrame(region[1:], columns=region[0]).rename(columns={"id": "feature_id"})


def extract_era5_land_timeseries(
    locs: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Extract ERA5-Land daily 2-m temperature at all ward centroids.

    Returns long-format DataFrame:
        latitude | longitude | date | era5land_temp_mean_c |
                                       era5land_temp_max_c |
                                       era5land_temp_min_c
    Latitude/longitude are the ORIGINAL ward coordinates (not the GEE pixel
    centres) — coordinates returned by GEE are mapped back to ward coords
    via nearest-neighbour matching, mirroring the MODIS extractor.
    """
    logger.info("Initialising GEE (project=joburg-hvi)...")
    ee.Initialize(project="joburg-hvi")

    fc = _make_feature_collection(locs)

    chunks: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        logger.info("  Extracting %d ...", year)
        chunk = _extract_year(fc, year)
        if len(chunk) == 0:
            logger.warning("    no data returned for %d", year)
            continue
        chunks.append(chunk)
        logger.info("    %d obs", len(chunk))

    if not chunks:
        raise RuntimeError("No ERA5-Land data returned for any year")

    df = pd.concat(chunks, ignore_index=True)
    logger.info("Total raw ERA5-Land observations: %d", len(df))

    # Map every ward → its nearest GEE pixel, then expand the daily table so
    # that every (ward, date) row carries the temperature of its pixel.
    # (The MODIS extractor's inverse mapping — pixel → nearest ward — only
    # works at 1 km where each pixel matches at most one ward; at ERA5-Land's
    # ~9 km, each pixel covers many wards, so we must broadcast.)
    df["date"] = pd.to_datetime(df["time"], unit="ms").dt.date.astype(str)
    pixel_coords = df[["longitude", "latitude"]].drop_duplicates().reset_index(drop=True)
    pixel_coords = pixel_coords.assign(_pix_idx=range(len(pixel_coords)))
    logger.info(
        "  GEE returned %d unique pixel centres for %d wards "
        "(many wards share pixels at 9 km resolution)",
        len(pixel_coords), len(locs),
    )
    # For each ward, find the index of its nearest GEE pixel.
    ward_to_pix: list[int] = []
    for _, row in locs.iterrows():
        d = (
            (pixel_coords["latitude"] - row["latitude"]) ** 2
            + (pixel_coords["longitude"] - row["longitude"]) ** 2
        )
        ward_to_pix.append(int(d.idxmin()))
    ward_with_pix = locs.assign(_pix_idx=ward_to_pix)

    # Daily df keyed by pixel index.
    df = df.merge(pixel_coords, on=["longitude", "latitude"], how="left")
    # Broadcast: every (ward, date) pair joins its pixel's row.
    broadcast = ward_with_pix.merge(df, on="_pix_idx", how="left", suffixes=("", "_pix"))
    # Output uses the WARD coordinates (so visit-level joins succeed).
    rename = TEMP_BAND_LABELS
    out = broadcast[["latitude", "longitude", "date", *TEMP_BANDS]].rename(columns=rename)
    return out.dropna(subset=[TEMP_BAND_LABELS["temperature_2m"]])


# ---------------------------------------------------------------------------
# Step 3 — compute lag features matching the ERA5 schedule
# ---------------------------------------------------------------------------
def compute_lag_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ward rolling lag features at LAG_DAYS, plus DTR."""
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["latitude", "longitude", "date"])
    daily["era5land_temp_range_c"] = (
        daily["era5land_temp_max_c"] - daily["era5land_temp_min_c"]
    )

    # Lag semantics MUST mirror the primary ERA5 lag features defined in
    # mcd_pipeline/config.py LAG_FEATURE_COLUMNS:
    #   * lags 0–21 = point-in-time daily mean N days before visit (`dlnm_lag{N}_c`)
    #   * lag 30    = 30-day rolling mean ending on visit day (`era5_temp_lag30d_c`)
    # Mixing point-in-time and rolling for lag 30 was a deliberate choice in
    # the primary pipeline (the DLNM cross-basis only covers lags 0–21).
    POINT_IN_TIME_LAGS = [lag for lag in LAG_DAYS if lag <= 21]
    ROLLING_LAG_30_WINDOW = 30  # days
    rows: list[pd.DataFrame] = []
    grouped = daily.groupby(["latitude", "longitude"], sort=False)
    for (lat, lon), g in grouped:
        g = g.set_index("date").sort_index()
        lag_block: dict[str, pd.Series] = {"latitude": lat, "longitude": lon}
        for lag in POINT_IN_TIME_LAGS:
            lag_block[f"era5land_temp_lag{lag}d_c"] = g["era5land_temp_mean_c"].shift(lag)
        # Lag 30 is the rolling 30-day mean (matches primary `era5_temp_lag30d_c`).
        lag_block["era5land_temp_lag30d_c"] = (
            g["era5land_temp_mean_c"]
            .rolling(window=ROLLING_LAG_30_WINDOW, min_periods=ROLLING_LAG_30_WINDOW)
            .mean()
        )
        # also surface today's mean/min/max/DTR for downstream joins
        lag_block["era5land_temp_mean_c"] = g["era5land_temp_mean_c"]
        lag_block["era5land_temp_max_c"] = g["era5land_temp_max_c"]
        lag_block["era5land_temp_min_c"] = g["era5land_temp_min_c"]
        lag_block["era5land_temp_range_c"] = g["era5land_temp_range_c"]
        out = pd.DataFrame(lag_block).reset_index()
        rows.append(out)

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 4 — join to clinical visits
# ---------------------------------------------------------------------------
def link_to_visits(lagged: pd.DataFrame) -> pd.DataFrame:
    """Join the per-ward lag-feature table to the (study, patient, visit_date) frame."""
    clim = pd.read_csv(config.TIDY_CLIMATE_CSV, low_memory=False)
    keep_cols = ["study_source", "patient_id", "visit_date", "latitude", "longitude"]
    clim = clim[keep_cols].drop_duplicates()
    clim["visit_date"] = pd.to_datetime(clim["visit_date"])
    lagged = lagged.rename(columns={"date": "visit_date"})
    merged = clim.merge(
        lagged,
        on=["latitude", "longitude", "visit_date"],
        how="left",
    )
    return merged


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    locs = load_ward_locations()
    daily = extract_era5_land_timeseries(
        locs,
        start_year=2005,
        end_year=2022,
    )
    logger.info("Daily ERA5-Land table: %d rows", len(daily))

    logger.info("Computing lag features (lags %s + DTR)...", LAG_DAYS)
    lagged = compute_lag_features(daily)
    logger.info("Lagged table: %d rows", len(lagged))

    logger.info("Linking to clinical visits ...")
    linked = link_to_visits(lagged)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    linked.to_csv(OUTPUT_PATH, index=False)
    n_with_temp = linked["era5land_temp_mean_c"].notna().sum()
    logger.info(
        "Wrote %s (%d rows, %d non-null temperature)",
        OUTPUT_PATH, len(linked), n_with_temp,
    )


if __name__ == "__main__":
    main()
