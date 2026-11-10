"""S.11: full Stage 2 table for the 10 retained biomarkers (340 tested pairs) with raw p, BH q and, for every tested pair,
a descriptive direction: Spearman rho between the modifier's value and the temperature feature's SHAP value
(positive = temperature attribution larger where the modifier is higher, e.g. in men for sex_Male). Aggregate output."""
import os, sys, json, logging
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
os.environ.setdefault("RP2_PRIMARY_EXPOSURE", "era5_land")
REPO = Path(__file__).resolve().parents[2]  # repository root; sys.path.insert(0, str(REPO))
logging.basicConfig(level=logging.ERROR)
from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers
from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features
from mcd_pipeline.stage1_lag_profiling.xgboost_trainer import prepare_features_and_target
OUT = Path(__file__).resolve().parents[1] / "outputs"; S1 = config.OUTPUT_ROOT / 'stage1'
RET = ['creatinine', 'diastolic_bp', 'heart_rate', 'cd4_count', 'viral_load', 'ldl_cholesterol', 'waist_hip_ratio', 'body_fat_percent', 'hemoglobin', 'hematocrit']
t = pd.read_csv(OUT / 'stage2_full_interaction_table.csv'); t = t[t.Permutation_p.notna() & t.Biomarker.isin(RET)].copy()
df = build_analysis_dataset(); bios = get_available_biomarkers()
df = engineer_features(df, [b.column for b in bios.values() if b.column and b.column in df.columns])
rho = []
cache = {}
for _, r in t.iterrows():
    b = r.Biomarker
    if b not in cache:
        X, _, _ = prepare_features_and_target(df, bios[b]); sv = np.load(S1 / b / 'shap_values.npy'); f = json.load(open(S1 / b / 'feature_names.json'))
        cache[b] = (X.reset_index(drop=True), pd.DataFrame(sv, columns=f))
    X, S = cache[b]
    if r.Modifier in X.columns and r.Temperature_feature in S.columns:
        rr = spearmanr(X[r.Modifier], S[r.Temperature_feature]).correlation
    else: rr = np.nan
    rho.append(rr)
t['direction_rho'] = rho
t = t.rename(columns={'Observed_interaction': 'observed_mean_abs_interaction', 'Permutation_p': 'p_raw', 'FDR_q_value': 'q_BH'})
t.to_csv(OUT / 'r1_stage2_all_340_pairs.csv', index=False); print(len(t), t['Significant_FDR'].sum(), t.direction_rho.notna().sum())
