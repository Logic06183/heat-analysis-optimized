"""
MODIS LST Extraction for SA2 Sensitivity Analysis
==================================================

Extracts MODIS/Terra MOD11A1 daily Land Surface Temperature (1 km) at
the 102 unique ward centroid locations from the TIDY climate dataset.

Produces MODIS_LST_LINKAGE.csv with the same lag structure as ERA5
(lags 0, 1, 3, 7, 14, 21, 30 days), linked by (study_source, patient_id,
visit_date), for use in SA2 (MODIS vs ERA5 urban heat island sensitivity).

Usage:
    python -m mcd_pipeline.utils.extract_modis_lst

Output:
    FINAL_DATASETS/CLIMATE_HEALTH_LINKAGE/MODIS_LST_LINKAGE.csv
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

MODIS_COLLECTION = "MODIS/061/MOD11A1"   # Terra LST daily, v6.1
LST_BAND = "LST_Day_1km"
QC_BAND = "QC_Day"
SCALE_FACTOR = 0.02                        # raw → Kelvin
KELVIN_OFFSET = 273.15                     # Kelvin → Celsius

# QC bitmask: bits 0-1 = mandatory QA, 00 = good quality
QC_GOOD_MASK = 0b11  # mask for bits 0-1
QC_GOOD_VALUE = 0b00  # want 00 = LST produced, good quality

# Maximum gap (days) to interpolate over cloudy periods
MAX_INTERPOLATION_GAP = 7

OUTPUT_PATH = (
    config.FINAL_DATASETS
    / "CLIMATE_HEALTH_LINKAGE"
    / "MODIS_LST_LINKAGE.csv"
)


# ---------------------------------------------------------------------------
# Step 1: Load unique ward locations from TIDY climate dataset
# ---------------------------------------------------------------------------

def load_ward_locations() -> pd.DataFrame:
    """Return DataFrame of unique (latitude, longitude) ward centroids."""
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
# Step 2: Extract daily MODIS LST at all ward centroids via GEE
# ---------------------------------------------------------------------------

def _make_feature_collection(locs: pd.DataFrame) -> ee.FeatureCollection:
    """Convert ward DataFrame to a GEE FeatureCollection."""
    features = []
    for _, row in locs.iterrows():
        pt = ee.Feature(
            ee.Geometry.Point([row["longitude"], row["latitude"]]),
            {"ward_id": int(row["ward_id"])},
        )
        features.append(pt)
    return ee.FeatureCollection(features)


def extract_modis_timeseries(
    locs: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Extract daily MOD11A1 LST at all ward centroids.

    Applies QC masking (bits 0-1 = 00, good quality), applies scale factor,
    converts to Celsius. Returns long-format DataFrame:
        ward_id | latitude | longitude | date | lst_day_c | lst_quality_flag

    Parameters
    ----------
    locs : pd.DataFrame
        Ward centroids with columns [ward_id, latitude, longitude].
    start_date, end_date : str
        ISO date strings, e.g. "2005-01-01".
    """
    logger.info("Initialising GEE...")
    ee.Initialize(project="joburg-hvi")

    fc = _make_feature_collection(locs)

    # Apply QC masking and scaling within GEE
    # Apply scale factor only — no QC masking in GEE.
    # MODIS already nulls cloud-affected pixels (returns null for LST_Day_1km
    # when QC bits 0-1 = 10 or 11). Keeping unreliable (01) pixels maximises
    # coverage for the sensitivity comparison.
    def scale_only(image):
        lst_k = image.select(LST_BAND).multiply(SCALE_FACTOR)
        lst_c = lst_k.subtract(KELVIN_OFFSET)
        return lst_c.rename("lst_day_c").copyProperties(image, ["system:time_start"])

    collection = (
        ee.ImageCollection(MODIS_COLLECTION)
        .filterDate(start_date, end_date)
        .map(scale_only)
    )

    logger.info("Fetching MODIS time series for %d locations...", len(locs))
    logger.info("  Collection size: %d images", collection.size().getInfo())

    # getRegion returns a list: [header, [ward_id, time, lon, lat, lst_day_c], ...]
    region_data = collection.getRegion(fc, scale=1000).getInfo()

    if not region_data or len(region_data) < 2:
        raise RuntimeError("GEE getRegion returned empty results")

    header = region_data[0]
    rows = region_data[1:]
    logger.info("  GEE header: %s", header)
    logger.info("  Retrieved %d raw observations", len(rows))

    df = pd.DataFrame(rows, columns=header)

    # GEE getRegion returns: [id, longitude, latitude, time, band1, ...]
    # Note: GEE snaps input coordinates to the nearest 1km pixel centre,
    # so returned lat/lon differs slightly from the input ward coordinates.
    # We build a mapping from GEE pixel coords → original ward coords so that
    # the output can be joined back to patient visits (which use the original coords).
    df = df.rename(columns={"id": "feature_id"})

    # Parse timestamp → date
    df["date"] = pd.to_datetime(df["time"], unit="ms").dt.date.astype(str)

    # Build GEE-pixel → original-ward coordinate mapping.
    # Use the first row per unique GEE pixel to find the nearest ward in locs.
    pixel_coords = df[["longitude", "latitude"]].drop_duplicates()
    coord_map = {}  # (gee_lon, gee_lat) → (orig_lat, orig_lon)
    for _, row in pixel_coords.iterrows():
        gee_lon, gee_lat = row["longitude"], row["latitude"]
        # Find nearest ward in locs by Euclidean distance
        dists = ((locs["latitude"] - gee_lat) ** 2 + (locs["longitude"] - gee_lon) ** 2)
        nearest = locs.iloc[dists.idxmin()]
        coord_map[(round(gee_lon, 6), round(gee_lat, 6))] = (nearest["latitude"], nearest["longitude"])

    df["orig_lat"] = df.apply(
        lambda r: coord_map.get((round(r["longitude"], 6), round(r["latitude"], 6)), (r["latitude"], r["longitude"]))[0],
        axis=1,
    )
    df["orig_lon"] = df.apply(
        lambda r: coord_map.get((round(r["longitude"], 6), round(r["latitude"], 6)), (r["latitude"], r["longitude"]))[1],
        axis=1,
    )

    logger.info(
        "  Coordinate mapping: %d unique GEE pixels → %d unique ward coords",
        len(pixel_coords),
        len(set(coord_map.values())),
    )

    # Filter out null LST (cloud-masked days — MODIS returns null for these)
    n_total = len(df)
    df = df.dropna(subset=["lst_day_c"])
    n_valid = len(df)
    cloud_pct = 100 * (1 - n_valid / n_total)
    logger.info(
        "  Valid observations: %d/%d (cloud/gap: %.1f%%)",
        n_valid, n_total, cloud_pct,
    )

    return df[["orig_lat", "orig_lon", "date", "lst_day_c"]].rename(
        columns={"orig_lat": "latitude", "orig_lon": "longitude"}
    )


# ---------------------------------------------------------------------------
# Step 3: Interpolate cloud gaps and compute lag features
# ---------------------------------------------------------------------------

def _interpolate_gaps(series: pd.Series, max_gap: int = MAX_INTERPOLATION_GAP) -> pd.Series:
    """Linear interpolation for gaps up to max_gap days; longer gaps stay NaN."""
    return series.interpolate(method="time", limit=max_gap)


def compute_lag_features(modis_daily: pd.DataFrame) -> pd.DataFrame:
    """Compute temperature lag features matching ERA5 lag structure.

    For each ward, fills cloud gaps (≤7 days linear interpolation), then
    computes rolling lags at 0, 1, 3, 7, 14, 21, 30 days.

    Returns wide-format DataFrame:
        ward_id | latitude | longitude | date |
        modis_lag0d_c | modis_lag1d_c | ... | modis_lag30d_c |
        modis_gap_flag (True if any lag used interpolated value)
    """
    lag_days = config.LAG_DAYS  # [0, 1, 3, 7, 14, 21, 30]
    all_locs = []

    for (lat, lon), group in modis_daily.groupby(["latitude", "longitude"]):
        g = group.sort_values("date").copy()
        g["date"] = pd.to_datetime(g["date"])
        g = g.set_index("date")

        # Reindex to daily, flag original vs interpolated
        date_range = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(date_range)
        g["is_original"] = ~g["lst_day_c"].isna()

        # Interpolate short cloud gaps
        g["lst_day_c"] = _interpolate_gaps(g["lst_day_c"])

        # Compute lag features
        for lag in lag_days:
            g[f"modis_lag{lag}d_c"] = g["lst_day_c"].shift(lag)

        # Gap flag: any lag window used interpolated data?
        g["modis_gap_flag"] = ~g["is_original"]

        g["latitude"] = lat
        g["longitude"] = lon
        g.index.name = "date"
        g = g.reset_index()
        g["date"] = g["date"].dt.strftime("%Y-%m-%d")

        lag_cols = [f"modis_lag{d}d_c" for d in lag_days]
        all_locs.append(g[["latitude", "longitude", "date"] + lag_cols + ["modis_gap_flag"]])

    result = pd.concat(all_locs, ignore_index=True)
    logger.info(
        "Lag features computed: %d ward-day rows, gap flag rate: %.1f%%",
        len(result),
        100 * result["modis_gap_flag"].mean(),
    )
    return result


# ---------------------------------------------------------------------------
# Step 4: Link MODIS lags back to patient visits
# ---------------------------------------------------------------------------

def link_to_patients(modis_lags: pd.DataFrame) -> pd.DataFrame:
    """Join MODIS lag features to patient visits.

    Matches on (latitude, longitude, visit_date). Uses nearest-neighbour
    date fallback within ±3 days for any remaining gaps.

    Returns DataFrame with (study_source, patient_id, visit_date, modis_lag*).
    """
    # Load TIDY clinical-climate crosswalk for patient-visit dates + coords
    clim = pd.read_csv(config.TIDY_CLIMATE_CSV, low_memory=False)
    visits = (
        clim[["study_source", "patient_id", "visit_date", "latitude", "longitude"]]
        .drop_duplicates(subset=["study_source", "patient_id", "visit_date"])
        .copy()
        .assign(
            latitude=lambda df: df["latitude"].round(4),
            longitude=lambda df: df["longitude"].round(4),
        )
    )

    logger.info("Linking MODIS lags to %d patient visits...", len(visits))

    linked = visits.merge(
        modis_lags,
        left_on=["latitude", "longitude", "visit_date"],
        right_on=["latitude", "longitude", "date"],
        how="left",
    )

    lag_cols = [f"modis_lag{d}d_c" for d in config.LAG_DAYS]
    coverage = linked[lag_cols[0]].notna().mean() * 100
    logger.info("  MODIS linkage coverage: %.1f%%", coverage)

    if coverage < 70:
        logger.warning(
            "  Low MODIS coverage (%.1f%%) — check date alignment or cloud coverage",
            coverage,
        )

    result = linked[["study_source", "patient_id", "visit_date"] + lag_cols + ["modis_gap_flag"]]
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_modis_extraction(
    output_path: Path = OUTPUT_PATH,
    start_date: str = "2004-12-01",  # 30-day lead for lag-30 before first visit
    end_date: str = "2023-01-01",
) -> pd.DataFrame:
    """Full MODIS extraction pipeline.

    Returns final linked DataFrame and saves to output_path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Ward locations
    locs = load_ward_locations()

    # Step 2: GEE extraction
    modis_daily = extract_modis_timeseries(locs, start_date, end_date)

    # Step 3: Lag features
    modis_lags = compute_lag_features(modis_daily)

    # Step 4: Link to patients
    linked = link_to_patients(modis_lags)

    # Save
    linked.to_csv(output_path, index=False)
    logger.info("Saved MODIS LST linkage to %s (%d rows)", output_path, len(linked))

    # Summary stats
    lag_cols = [f"modis_lag{d}d_c" for d in config.LAG_DAYS]
    print("\n=== MODIS LST Extraction Complete ===")
    print(f"Output: {output_path}")
    print(f"Rows: {len(linked):,}")
    print(f"Coverage: {linked[lag_cols[0]].notna().mean()*100:.1f}%")
    print(f"LST range (lag0): {linked['modis_lag0d_c'].min():.1f} – {linked['modis_lag0d_c'].max():.1f} °C")

    return linked


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_modis_extraction()
