"""SA16: UTCI (ERA5-HEAT v1.1, 0.25 deg, accessed 10 Nov 2026) in place of air temperature for the 10 retained biomarkers.
Same lag structure (single-day lags 0,1,3,7,14,21; 30-day mean) and daily UTCI range in place of DTR; same folds and settings.
Reports cross-validated R2 with UTCI, incremental R2 over the covariate-only model, and Spearman rho between the UTCI and
air-temperature lag-importance profiles (mean |SHAP| per lag, XGBoost pred_contribs on the full-data fit). Aggregate only."""
import os, sys, json, logging, time
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
os.environ.setdefault("RP2_PRIMARY_EXPOSURE", "era5_land")
REPO = Path(__file__).resolve().parents[2]  # repository root; sys.path.insert(0, str(REPO))
logging.basicConfig(level=logging.ERROR)
from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers
from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features
from mcd_pipeline.stage1_lag_profiling.xgboost_trainer import prepare_features_and_target
W = Path(__file__).resolve().parents[1]; OUT = W / 'outputs'
RET = ['creatinine', 'diastolic_bp', 'heart_rate', 'cd4_count', 'viral_load', 'ldl_cholesterol', 'waist_hip_ratio', 'body_fat_percent', 'hemoglobin', 'hematocrit']
TEMP = [f"temp_lag_{d}d" for d in config.LAG_DAYS] + [config.DTR_COLUMN]
rng = np.random.default_rng(42)
df = build_analysis_dataset(); bios = get_available_biomarkers()
df = engineer_features(df, [b.column for b in bios.values() if b.column and b.column in df.columns])
link = pd.read_csv(REPO / 'FINAL_DATASETS/CLIMATE_HEALTH_LINKAGE/ERA5_LAND_LINKAGE.csv', usecols=['study_source', 'patient_id', 'visit_date', 'latitude', 'longitude'])
link['cell_lat'] = (link['latitude'] / 0.25).round() * 0.25; link['cell_lon'] = (link['longitude'] / 0.25).round() * 0.25
link['visit_date'] = pd.to_datetime(link['visit_date']).dt.floor('D')
u = pd.read_csv(W / 'utci' / 'utci_daily_cells.csv', parse_dates=['date']).sort_values(['cell_lat', 'cell_lon', 'date'])
u['utci_range'] = u['utci_max'] - u['utci_min']
u['utci_30'] = u.groupby(['cell_lat', 'cell_lon'])['utci_mean'].transform(lambda s: s.rolling(30, min_periods=25).mean())
U = u.set_index(['cell_lat', 'cell_lon', 'date'])
feat = link[['study_source', 'patient_id', 'visit_date', 'cell_lat', 'cell_lon']].copy()
for d in [0, 1, 3, 7, 14, 21]:
    key = pd.MultiIndex.from_arrays([feat['cell_lat'], feat['cell_lon'], feat['visit_date'] - pd.Timedelta(days=d)])
    feat[f'u_lag_{d}d'] = U['utci_mean'].reindex(key).values
key0 = pd.MultiIndex.from_arrays([feat['cell_lat'], feat['cell_lon'], feat['visit_date']])
feat['u_lag_30d'] = U['utci_30'].reindex(key0).values; feat['u_range'] = U['utci_range'].reindex(key0).values
feat = feat.drop_duplicates(['study_source', 'patient_id', 'visit_date'])
df['vd'] = pd.to_datetime(df['visit_date']).dt.floor('D')
m = df[['study_source', 'patient_id', 'vd']].merge(feat.rename(columns={'visit_date': 'vd'}), on=['study_source', 'patient_id', 'vd'], how='left')
cov = float(m['u_lag_0d'].notna().mean()); print('UTCI coverage of visits', round(cov, 4), flush=True)
air = df[TEMP].copy()
corr = float(pd.Series(m['u_lag_7d'].values).corr(pd.Series(df['temp_lag_7d'].values)))
def oof(X, y, g):
    pred = np.full(len(y), np.nan); sc = []
    for tr, va in GroupKFold(n_splits=config.CV_FOLDS).split(X, y, g):
        mdl = xgb.XGBRegressor(**config.XGBOOST_PARAMS); mdl.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
        p = mdl.predict(X.iloc[va]); pred[va] = p; sc.append(r2_score(y.iloc[va], p))
    return pred, sc
def lagimp(X, y):
    mdl = xgb.XGBRegressor(**config.XGBOOST_PARAMS); mdl.fit(X, y, verbose=False)
    c = mdl.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    names = list(X.columns); return np.array([np.abs(c[:, names.index(f'temp_lag_{d}d')]).mean() for d in config.LAG_DAYS])
def boot(y, a, b, g):
    gs = pd.Series(g.values); idx = gs.groupby(gs).indices; un = gs.unique(); yv = np.asarray(y); out = []
    for _ in range(1000):
        ii = np.concatenate([idx[k] for k in rng.choice(un, len(un))]); out.append(r2_score(yv[ii], a[ii]) - r2_score(yv[ii], b[ii]))
    return np.percentile(out, [2.5, 97.5]).tolist()
rows = []
for name in RET:
    b = bios[name]
    Xa, y, g = prepare_features_and_target(df, b)
    idx = y.index
    dfu = df.copy()
    for d in config.LAG_DAYS: dfu[f'temp_lag_{d}d'] = m[f'u_lag_{d}d'].values
    dfu[config.DTR_COLUMN] = m['u_range'].values
    Xu, yu, gu = prepare_features_and_target(dfu, b)
    tc = [c for c in TEMP if c in Xa.columns]
    pa, sa = oof(Xa, y, g); pu, su = oof(Xu, yu, gu); pc, scv = oof(Xa.drop(columns=tc), y, g)
    rho = spearmanr(lagimp(Xa, y), lagimp(Xu, yu)).correlation
    rows.append(dict(biomarker=name, n_obs=len(y), r2_air=float(np.mean(sa)), r2_utci=float(np.mean(su)), r2_cov=float(np.mean(scv)),
        delta_air=float(np.mean(np.array(sa) - np.array(scv))), delta_utci=float(np.mean(np.array(su) - np.array(scv))),
        delta_utci_ci=boot(y, pu, pc, g), lag_profile_rho=float(rho)))
    print(json.dumps(rows[-1]), flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'r1_sa16_utci.csv', index=False)
json.dump(dict(coverage=cov, corr_utci7_air7=corr, source='CDS derived-utci-historical-timeseries v1.1 consolidated, accessed 2026-11-10', finished=time.ctime()), open(OUT / 'r1_sa16_utci_info.json', 'w'), indent=2)
print('done')
