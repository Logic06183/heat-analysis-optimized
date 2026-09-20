"""Re-run SA8 only — under ERA5-Land primary."""
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers
from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features
from mcd_pipeline.sensitivity.sa8_heatwave_threshold import run_heatwave_threshold_comparison

print(f"Primary exposure: {config.PRIMARY_EXPOSURE_SOURCE}")
df = build_analysis_dataset()
biomarkers = get_available_biomarkers()
bio_cols = [b.column for b in biomarkers.values() if b.column and b.column in df.columns]
df = engineer_features(df, bio_cols)
print(f"Dataset: {len(df)} rows")
print("\n=== SA8: Heatwave threshold ===")
run_heatwave_threshold_comparison(df)
print("SA8 DONE")
