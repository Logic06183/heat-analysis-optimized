"""Run SA2 (MODIS vs new primary), SA3 (PM2.5), SA4 (COVID), SA9 (per-cohort) under ERA5-Land."""
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers
from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features

print(f"Primary exposure: {config.PRIMARY_EXPOSURE_SOURCE}")
df = build_analysis_dataset()
biomarkers = get_available_biomarkers()
bio_cols = [b.column for b in biomarkers.values() if b.column and b.column in df.columns]
df = engineer_features(df, bio_cols)
print(f"Dataset: {len(df)} rows")

for label, fn_path in [
    ("SA2 MODIS", "mcd_pipeline.sensitivity.sa2_modis_vs_era5.run_modis_comparison"),
    ("SA4 COVID exclusion", "mcd_pipeline.sensitivity.sa4_covid_exclusion.run_covid_exclusion"),
    ("SA3 PM2.5", "mcd_pipeline.sensitivity.sa3_pollution_adjusted.run_pollution_adjusted"),
    ("SA9 per-cohort meta", "mcd_pipeline.sensitivity.sa9_per_cohort_meta.run_per_cohort_meta"),
]:
    print(f"\n=== {label} ===")
    try:
        module_path, fn_name = fn_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[fn_name])
        fn = getattr(module, fn_name)
        fn(df)
        print(f"{label} DONE")
    except Exception as e:
        print(f"{label} FAILED: {type(e).__name__}: {e}")

print("\n=== ALL REMAINING SAs DONE ===")
