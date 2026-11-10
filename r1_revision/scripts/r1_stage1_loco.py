"""7g / S.5: leave-one-cohort-out for the Stage 1 incremental R2 of the 10 retained biomarkers. For each cohort that
contributes to a biomarker, drop it, refit full and covariate-only models on the same GroupKFold folds, record the
incremental R2. Aggregate output."""
import os, sys, json, logging
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
os.environ.setdefault("RP2_PRIMARY_EXPOSURE", "era5_land")
REPO = Path(__file__).resolve().parents[2]  # repository root; sys.path.insert(0, str(REPO))
logging.basicConfig(level=logging.ERROR)
from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers
from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features
from mcd_pipeline.stage1_lag_profiling.xgboost_trainer import prepare_features_and_target
OUT = Path(__file__).resolve().parents[1] / "outputs"
RET = ['creatinine', 'diastolic_bp', 'heart_rate', 'cd4_count', 'viral_load', 'ldl_cholesterol', 'waist_hip_ratio', 'body_fat_percent', 'hemoglobin', 'hematocrit']
TEMP = [f"temp_lag_{d}d" for d in config.LAG_DAYS] + [config.DTR_COLUMN]
df = build_analysis_dataset(); bios = get_available_biomarkers()
df = engineer_features(df, [b.column for b in bios.values() if b.column and b.column in df.columns])
def cv(X, y, g):
    s = []
    for tr, va in GroupKFold(n_splits=config.CV_FOLDS).split(X, y, g):
        m = xgb.XGBRegressor(**config.XGBOOST_PARAMS); m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
        s.append(r2_score(y.iloc[va], m.predict(X.iloc[va])))
    return np.array(s)
rows = []
for b in RET:
    sub0 = df[df[b].notna()]
    cohorts = sorted(sub0['study_source'].astype(str).unique())
    if len(cohorts) < 2:
        rows.append(dict(biomarker=b, dropped='(single cohort)', n_obs=len(sub0))); continue
    for c in cohorts:
        d = df[df['study_source'].astype(str) != c]
        X, y, g = prepare_features_and_target(d, bios[b])
        if g.nunique() < 20: continue
        tc = [x for x in TEMP if x in X.columns]
        f = cv(X, y, g); r = cv(X.drop(columns=tc), y, g)
        rows.append(dict(biomarker=b, dropped=c, n_obs=len(y), r2_full=float(f.mean()), r2_cov=float(r.mean()), delta=float((f - r).mean()), folds_better=int((f > r).sum())))
        print(json.dumps(rows[-1]), flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'r1_stage1_loco.csv', index=False)
print('done')
