"""Run SA1, SA7, SA8 under ERA5-Land primary."""
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers
from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features

print(f"Primary exposure: {config.PRIMARY_EXPOSURE_SOURCE}")
print(f"Output root: {config.OUTPUT_ROOT}")

print("Building analysis dataset...")
df = build_analysis_dataset()
biomarkers = get_available_biomarkers()
bio_cols = [b.column for b in biomarkers.values() if b.column and b.column in df.columns]
df = engineer_features(df, bio_cols)
print(f"Dataset: {len(df)} rows")

# SA1 (DLNM)
print("\n=== SA1: DLNM validation ===")
try:
    from mcd_pipeline.sensitivity.sa1_dlnm_validation import run_dlnm_validation
    run_dlnm_validation(df)
    print("SA1 DONE")
except Exception as e:
    print(f"SA1 FAILED: {e}")

# SA7 (HIV stratification)
print("\n=== SA7: HIV stratification ===")
try:
    from mcd_pipeline.sensitivity.sa7_hiv_vs_general import run_hiv_stratified
    run_hiv_stratified(df)
    print("SA7 DONE")
except Exception as e:
    print(f"SA7 FAILED: {e}")

# SA8 (heatwave threshold)
print("\n=== SA8: Heatwave threshold ===")
try:
    from mcd_pipeline.sensitivity.sa8_heatwave_threshold import run_heatwave_threshold_sensitivity
    run_heatwave_threshold_sensitivity(df)
    print("SA8 DONE")
except Exception as e:
    print(f"SA8 FAILED: {e}")

print("\n=== ALL SAs COMPLETE ===")
