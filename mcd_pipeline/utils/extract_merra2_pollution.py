"""
MERRA-2 Air Pollution Extraction for SA3 Sensitivity Analysis
=============================================================

Extracts daily PM2.5 surface mass concentration from NASA MERRA-2 aerosol
reanalysis (NASA/GSFC/MERRA/aer/2) via Google Earth Engine at the unique
ward centroid locations from the TIDY climate dataset.

PM2.5 is computed as the sum of five aerosol surface mass concentration
components that together constitute the PM2.5 size fraction:
    DUSMASS25 — Dust aerosol mass (PM2.5 fraction)
    BCSMASS   — Black carbon aerosol mass
    OCSMASS   — Organic carbon aerosol mass
    SO4SMASS  — Sulphate aerosol mass
    SS25SMASS — Sea salt aerosol mass (PM2.5 fraction)

Units: μg/m³ (input bands are kg/m³ × 1e9).

Resolution: 0.5° × 0.625° (~55 km). All Johannesburg ward centroids will
map to a small number of MERRA-2 pixels, which is expected and documented.

MERRA-2 in GEE is stored as 1-hourly images. This script creates daily
means server-side before calling getRegion(), to stay within GEE memory
limits. Extraction proceeds in annual chunks for robustness.

MCD reference: SA3 genuine pollution-adjusted sensitivity analysis.
Defensibility: MERRA-2 is the standard NASA global atmospheric reanalysis;
  widely used in climate–health epidemiology where ground PM2.5 monitoring
  is absent (Reddington et al. 2017 ACP; WHO 2021 global burden estimates).
  Known to underestimate PM2.5 in sub-Saharan African cities (Hammer et al.
  2020 Env. Sci. Technol.) — a limitation discussed in Methods.

Usage:
    python -m mcd_pipeline.utils.extract_merra2_pollution

Output:
    FINAL_DATASETS/CLIMATE_HEALTH_LINKAGE/MERRA2_POLLUTION_LINKAGE.csv
"""

import logging
from pathlib import Path
from datetime import date, timedelta

import ee
import numpy as np
import pandas as pd

from mcd_pipeline import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MERRA2_AER_COLLECTION = "NASA/GSFC/MERRA/aer/2"

# PM2.5 aerosol surface mass concentration components (kg m^-3)
# Band names verified against GEE catalog (NASA/GSFC/MERRA/aer/2):
#   DUSMASS25 — Dust PM2.5 surface mass concentration
#   BCSMASS   — Black Carbon surface mass concentration
#   OCSMASS   — Organic Carbon surface mass concentration
#   SO4SMASS  — Sulphate surface mass concentration
#   SSSMASS25 — Sea Salt PM2.5 surface mass concentration (note: SSSMASS25, not SS25SMASS)
PM25_BANDS = ["DUSMASS25", "BCSMASS", "OCSMASS", "SO4SMASS", "SSSMASS25"]

# Scale from kg/m³ to μg/m³
KG_TO_UGM3 = 1e9

# GEE extraction scale (meters) — matches MERRA-2 native grid ~55 km
MERRA2_SCALE_M = 55_000

# Maximum interpolation gap (days) for missing MERRA-2 values
MAX_INTERPOLATION_GAP = 7

OUTPUT_PATH = (
    config.FINAL_DATASETS
    / "CLIMATE_HEALTH_LINKAGE"
    / "MERRA2_POLLUTION_LINKAGE.csv"
)


# ---------------------------------------------------------------------------
# Step 1: Ward locations (reuse same logic as MODIS extraction)
# ---------------------------------------------------------------------------

def load_ward_locations() -> pd.DataFrame:
    """Return DataFrame of unique (latitude, longitude) ward centroids."""
    clim = pd.read_csv(config.TIDY_CLIMATE_CSV, low_memory=False)
    locs = (
        clim[["latitude", "longitude"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    logger.info("Loaded %d unique ward locations", len(locs))
    return locs


# ---------------------------------------------------------------------------
# Step 2: GEE extraction — annual chunks of daily MERRA-2 means
# ---------------------------------------------------------------------------

def _make_feature_collection(locs: pd.DataFrame) -> ee.FeatureCollection:
    """Convert ward DataFrame to GEE FeatureCollection."""
    features = [
        ee.Feature(
            ee.Geometry.Point([row["longitude"], row["latitude"]]),
        )
        for _, row in locs.iterrows()
    ]
    return ee.FeatureCollection(features)


def _extract_one_year(
    fc: ee.FeatureCollection,
    year: int,
) -> pd.DataFrame:
    """Extract daily mean PM2.5 for one calendar year.

    MERRA-2 is hourly, so we create daily averages via server-side GEE
    computation: for each day in the year, filter the hourly collection to
    that day and compute the mean image. This avoids returning 24× as many
    rows via getRegion.

    Parameters
    ----------
    fc : ee.FeatureCollection
        Ward centroid locations.
    year : int
        Calendar year to extract.

    Returns
    -------
    pd.DataFrame
        Long-format with columns: latitude, longitude, date, pm25_ugm3.
    """
    start = ee.Date(f"{year}-01-01")
    end = ee.Date(f"{year + 1}-01-01")

    aer_coll = (
        ee.ImageCollection(MERRA2_AER_COLLECTION)
        .filterDate(start, end)
        .select(PM25_BANDS)
    )

    # Count how many hourly images exist for sanity check
    n_images = aer_coll.size().getInfo()
    logger.info("  %d: %d hourly MERRA-2 images", year, n_images)

    if n_images == 0:
        logger.warning("  %d: No MERRA-2 data — skipping", year)
        return pd.DataFrame()

    # Build daily composite ImageCollection server-side.
    # GEE server-side: create one image per day by filtering+averaging.
    # Days of the year 0..364 (or 365 for leap years)
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    n_days = 366 if is_leap else 365

    offsets = ee.List.sequence(0, n_days - 1)

    def make_daily_image(offset):
        day_start = start.advance(offset, "day")
        day_end = day_start.advance(1, "day")
        daily_mean = aer_coll.filterDate(day_start, day_end).mean()
        # Sum PM2.5 components → total PM2.5 (still in kg/m³)
        pm25_kgm3 = daily_mean.expression(
            "DUSMASS25 + BCSMASS + OCSMASS + SO4SMASS + SSSMASS25",
            {
                "DUSMASS25":  daily_mean.select("DUSMASS25"),
                "BCSMASS":    daily_mean.select("BCSMASS"),
                "OCSMASS":    daily_mean.select("OCSMASS"),
                "SO4SMASS":   daily_mean.select("SO4SMASS"),
                "SSSMASS25":  daily_mean.select("SSSMASS25"),
            },
        )
        # Convert to μg/m³
        pm25_ugm3 = pm25_kgm3.multiply(KG_TO_UGM3).rename("pm25_ugm3")
        return pm25_ugm3.set("system:time_start", day_start.millis())

    daily_coll = ee.ImageCollection(offsets.map(make_daily_image))

    # getRegion: returns [header, row, row, ...]
    # Columns: [id, longitude, latitude, time, pm25_ugm3]
    region_data = daily_coll.getRegion(fc, scale=MERRA2_SCALE_M).getInfo()

    if not region_data or len(region_data) < 2:
        logger.warning("  %d: getRegion returned empty results", year)
        return pd.DataFrame()

    header = region_data[0]
    rows = region_data[1:]
    logger.info("  %d: %d raw getRegion rows", year, len(rows))

    df = pd.DataFrame(rows, columns=header)
    df["date"] = pd.to_datetime(df["time"], unit="ms").dt.strftime("%Y-%m-%d")
    df = df.rename(columns={"longitude": "gee_lon", "latitude": "gee_lat"})
    df = df[["gee_lon", "gee_lat", "date", "pm25_ugm3"]].dropna(subset=["pm25_ugm3"])
    df["pm25_ugm3"] = pd.to_numeric(df["pm25_ugm3"], errors="coerce")

    return df


def extract_merra2_timeseries(
    locs: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Extract daily PM2.5 for all ward locations, one year at a time.

    Parameters
    ----------
    locs : pd.DataFrame
        Ward centroids [latitude, longitude].
    start_year, end_year : int
        Inclusive year range (e.g. 2005, 2022).

    Returns
    -------
    pd.DataFrame
        Long-format: gee_lon, gee_lat, date, pm25_ugm3.
    """
    logger.info("Initialising GEE...")
    ee.Initialize(project="joburg-hvi")

    fc = _make_feature_collection(locs)

    all_years = []
    for year in range(start_year, end_year + 1):
        logger.info("Extracting MERRA-2 year %d...", year)
        df_year = _extract_one_year(fc, year)
        if not df_year.empty:
            all_years.append(df_year)

    if not all_years:
        raise RuntimeError("No MERRA-2 data extracted for any year")

    combined = pd.concat(all_years, ignore_index=True)

    # Build GEE pixel → original ward coordinate mapping (nearest neighbour)
    # MERRA-2 at ~55km: most JHB wards map to the same 1-2 pixels
    pixel_coords = combined[["gee_lon", "gee_lat"]].drop_duplicates()
    logger.info(
        "Unique MERRA-2 pixels covering study area: %d", len(pixel_coords)
    )

    coord_map = {}
    for _, row in pixel_coords.iterrows():
        gee_lon, gee_lat = row["gee_lon"], row["gee_lat"]
        dists = (
            (locs["latitude"] - gee_lat) ** 2
            + (locs["longitude"] - gee_lon) ** 2
        )
        nearest = locs.iloc[dists.idxmin()]
        coord_map[(round(gee_lon, 6), round(gee_lat, 6))] = (
            nearest["latitude"],
            nearest["longitude"],
        )

    combined["latitude"] = combined.apply(
        lambda r: coord_map.get(
            (round(r["gee_lon"], 6), round(r["gee_lat"], 6)),
            (r["gee_lat"], r["gee_lon"]),
        )[0],
        axis=1,
    )
    combined["longitude"] = combined.apply(
        lambda r: coord_map.get(
            (round(r["gee_lon"], 6), round(r["gee_lat"], 6)),
            (r["gee_lat"], r["gee_lon"]),
        )[1],
        axis=1,
    )

    n_total = len(combined)
    n_valid = combined["pm25_ugm3"].notna().sum()
    logger.info(
        "MERRA-2 extraction complete: %d/%d valid daily observations",
        n_valid, n_total,
    )
    logger.info(
        "PM2.5 range: %.2f – %.2f μg/m³ (mean: %.2f)",
        combined["pm25_ugm3"].min(),
        combined["pm25_ugm3"].max(),
        combined["pm25_ugm3"].mean(),
    )

    return combined[["latitude", "longitude", "date", "pm25_ugm3"]]


# ---------------------------------------------------------------------------
# Step 3: Interpolate gaps and compute lag features
# ---------------------------------------------------------------------------

def compute_lag_features(merra2_daily: pd.DataFrame) -> pd.DataFrame:
    """Compute PM2.5 lag features matching the ERA5 lag structure.

    For each unique ward-pixel location: fills gaps ≤7 days by linear
    interpolation, then computes rolling lags at 0, 1, 3, 7, 14, 21, 30 days.

    Returns wide-format DataFrame:
        latitude | longitude | date |
        pm25_lag0d_ugm3 | pm25_lag1d_ugm3 | ... | pm25_lag30d_ugm3 |
        pm25_gap_flag
    """
    lag_days = config.LAG_DAYS  # [0, 1, 3, 7, 14, 21, 30]
    all_locs = []

    unique_locs = merra2_daily[["latitude", "longitude"]].drop_duplicates()
    logger.info(
        "Computing lag features for %d unique ward-pixel locations...",
        len(unique_locs),
    )

    for (lat, lon), group in merra2_daily.groupby(["latitude", "longitude"]):
        g = group.sort_values("date").copy()
        g["date"] = pd.to_datetime(g["date"])
        g = g.set_index("date")

        # Reindex to daily full range
        date_range = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(date_range)
        g["is_original"] = ~g["pm25_ugm3"].isna()

        # Interpolate gaps ≤7 days
        g["pm25_ugm3"] = g["pm25_ugm3"].interpolate(
            method="time", limit=MAX_INTERPOLATION_GAP
        )

        # Lag features
        for lag in lag_days:
            g[f"pm25_lag{lag}d_ugm3"] = g["pm25_ugm3"].shift(lag)

        g["pm25_gap_flag"] = ~g["is_original"]
        g["latitude"] = lat
        g["longitude"] = lon
        g.index.name = "date"
        g = g.reset_index()
        g["date"] = g["date"].dt.strftime("%Y-%m-%d")

        lag_cols = [f"pm25_lag{d}d_ugm3" for d in lag_days]
        all_locs.append(
            g[["latitude", "longitude", "date"] + lag_cols + ["pm25_gap_flag"]]
        )

    result = pd.concat(all_locs, ignore_index=True)
    logger.info(
        "Lag features computed: %d ward-day rows, gap flag rate: %.1f%%",
        len(result),
        100 * result["pm25_gap_flag"].mean(),
    )
    return result


# ---------------------------------------------------------------------------
# Step 4: Link pollution lags to patient visits
# ---------------------------------------------------------------------------

def _expand_pixels_to_all_wards(
    pm25_lags: pd.DataFrame,
    locs: pd.DataFrame,
) -> pd.DataFrame:
    """Expand the 4-pixel lag table to cover all 102 ward centroids.

    MERRA-2's coarse resolution (~55 km) means all 102 Johannesburg ward
    centroids fall within just 4 grid pixels. The extraction returns data
    for those 4 pixel centres. This function maps every ward centroid to its
    nearest pixel and duplicates the pixel's time series, so the merge with
    patient visits (which uses original ward coordinates) succeeds.

    Parameters
    ----------
    pm25_lags : pd.DataFrame
        Lag table with 4 unique (latitude, longitude) pixel coordinates.
    locs : pd.DataFrame
        All 102 unique ward centroids from TIDY_climate_variables.csv.

    Returns
    -------
    pd.DataFrame
        Lag table with 102 unique (latitude, longitude) ward coordinates.
    """
    pixel_coords = pm25_lags[["latitude", "longitude"]].drop_duplicates()
    lag_cols = [c for c in pm25_lags.columns if c not in ["latitude", "longitude", "date"]]

    all_ward_rows = []
    for _, ward in locs.iterrows():
        # Find nearest MERRA-2 pixel
        dists = (
            (pixel_coords["latitude"] - ward["latitude"]) ** 2
            + (pixel_coords["longitude"] - ward["longitude"]) ** 2
        )
        nearest = pixel_coords.iloc[dists.values.argmin()]

        # Get this pixel's time series and relabel with ward coordinates
        pixel_ts = pm25_lags[
            (pm25_lags["latitude"] == nearest["latitude"])
            & (pm25_lags["longitude"] == nearest["longitude"])
        ].copy()
        pixel_ts["latitude"] = ward["latitude"]
        pixel_ts["longitude"] = ward["longitude"]
        all_ward_rows.append(pixel_ts)

    expanded = pd.concat(all_ward_rows, ignore_index=True)
    logger.info(
        "Expanded lag table: %d pixels → %d ward-day rows (%d unique wards)",
        len(pixel_coords),
        len(expanded),
        expanded[["latitude", "longitude"]].drop_duplicates().__len__(),
    )
    return expanded


def link_to_patients(pm25_lags: pd.DataFrame) -> pd.DataFrame:
    """Join PM2.5 lag features to patient visits.

    First expands the 4-pixel MERRA-2 lag table to all 102 ward centroids
    (nearest-neighbour pixel assignment), then merges on
    (latitude, longitude, visit_date). Because MERRA-2 is at ~55 km
    resolution, wards sharing a pixel get identical PM2.5 estimates —
    documented in the Methods section as an expected limitation.

    Returns DataFrame with (study_source, patient_id, visit_date, pm25_lag*).
    """
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
    logger.info("Linking PM2.5 lags to %d patient visits...", len(visits))

    # Load all unique ward locations
    locs = (
        clim[["latitude", "longitude"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Expand 4-pixel table to all 102 wards before merging
    pm25_expanded = _expand_pixels_to_all_wards(pm25_lags, locs)
    pm25_expanded["latitude"] = pm25_expanded["latitude"].round(4)
    pm25_expanded["longitude"] = pm25_expanded["longitude"].round(4)

    linked = visits.merge(
        pm25_expanded,
        left_on=["latitude", "longitude", "visit_date"],
        right_on=["latitude", "longitude", "date"],
        how="left",
    )

    lag_cols = [f"pm25_lag{d}d_ugm3" for d in config.LAG_DAYS]
    coverage = linked[lag_cols[0]].notna().mean() * 100
    logger.info("  PM2.5 linkage coverage: %.1f%%", coverage)

    if coverage < 70:
        logger.warning(
            "  Low MERRA-2 coverage (%.1f%%) — check year range alignment",
            coverage,
        )

    out_cols = ["study_source", "patient_id", "visit_date"] + lag_cols + ["pm25_gap_flag"]
    return linked[out_cols]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_merra2_extraction(
    output_path: Path = OUTPUT_PATH,
    start_year: int = 2004,  # 1 extra year for lag-30 lead-in before study start
    end_year: int = 2022,
) -> pd.DataFrame:
    """Full MERRA-2 PM2.5 extraction pipeline.

    Returns final linked DataFrame and saves to output_path.

    Methodology note
    ----------------
    MERRA-2 PM2.5 is a modelled reanalysis product at ~55 km resolution.
    It is the standard data source for climate–health studies in regions
    without comprehensive ground-level PM2.5 monitoring networks (such as
    most of sub-Saharan Africa). MERRA-2 is known to underestimate absolute
    PM2.5 concentrations in African cities (Hammer et al. 2020), so results
    should be interpreted as relative rather than absolute exposure estimates.
    The purpose of this extraction is to enable covariate adjustment in SA3
    to assess whether temperature–biomarker relationships are confounded by
    air pollution, not to provide surveillance-grade PM2.5 estimates.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Ward locations
    locs = load_ward_locations()

    # Step 2: GEE extraction (annual chunks)
    merra2_daily = extract_merra2_timeseries(locs, start_year, end_year)

    # Step 3: Lag features
    pm25_lags = compute_lag_features(merra2_daily)

    # Step 4: Link to patients
    linked = link_to_patients(pm25_lags)

    # Save
    linked.to_csv(output_path, index=False)

    lag_cols = [f"pm25_lag{d}d_ugm3" for d in config.LAG_DAYS]
    print("\n=== MERRA-2 PM2.5 Extraction Complete ===")
    print(f"Output: {output_path}")
    print(f"Rows: {len(linked):,}")
    print(f"Coverage: {linked[lag_cols[0]].notna().mean()*100:.1f}%")
    print(f"PM2.5 range (lag0): {linked['pm25_lag0d_ugm3'].min():.2f} – {linked['pm25_lag0d_ugm3'].max():.2f} μg/m³")
    print(f"PM2.5 mean (lag0): {linked['pm25_lag0d_ugm3'].mean():.2f} μg/m³")
    print("\nNote: MERRA-2 at ~55 km resolution underestimates urban PM2.5.")
    print("      Use for covariate adjustment, not absolute exposure estimates.")

    logger.info("Saved MERRA-2 PM2.5 linkage to %s (%d rows)", output_path, len(linked))
    return linked


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_merra2_extraction()
