"""Within-panel clustering checks (aggregate output only).
P1 mixed-HIV cohorts only, each on its own measured panel: SCHARP_004 (creatinine, Hb, Hct) and DPHRU_013 (DBP, HR, WHR, body fat):
   does a heat-response cluster concentrate HIV within a cohort?
P2 HIV-positive participants with Hct+Hb+HIV VL measured (ACTG 015/016/019, Ezin_002, WRHI_001): cluster on those three
   biomarkers' temperature-lag SHAP; describe clusters by viraemia share, sex, age.
"""
import json, os, sys, logging
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
os.environ.setdefault("RP2_PRIMARY_EXPOSURE", "era5_land")
REPO = Path(__file__).resolve().parents[2]  # repository root; sys.path.insert(0, str(REPO))
logging.basicConfig(level=logging.ERROR)
from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers
from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features
from mcd_pipeline.stage1_lag_profiling.xgboost_trainer import prepare_features_and_target
OUT = Path(__file__).resolve().parents[1] / "outputs"
S1 = config.OUTPUT_ROOT / "stage1"; S3 = config.OUTPUT_ROOT / "stage3_r2_005"
TEMP = ['temp_lag_%dd' % d for d in config.LAG_DAYS]
df = build_analysis_dataset(); bios = get_available_biomarkers()
df = engineer_features(df, [b.column for b in bios.values() if b.column and b.column in df.columns])
g = df.groupby('patient_id')
study = g['study_source'].agg(lambda s: s.mode().iat[0]).astype(str)
sex = g['sex'].agg(lambda s: s.mode().iat[0] if s.notna().any() else np.nan).astype(str)
age = g['age_years'].mean()
vlmax = g['viral_load'].max()
meta = pd.read_csv(S3 / 'patient_cluster_table.csv').set_index('patient_id')
hiv = meta['hiv_status'].astype(str).str.lower().eq('positive')
def person(name):
    sv = np.load(S1 / name / 'shap_values.npy'); feats = json.load(open(S1 / name / 'feature_names.json'))
    X, _, grp = prepare_features_and_target(df, bios[name])
    idx = [i for i, f in enumerate(feats) if f in TEMP]
    s = pd.DataFrame(sv[:, idx], columns=['%s__%s' % (name, feats[i]) for i in idx]); s.index = grp.reset_index(drop=True).values
    return s.groupby(level=0).mean()
def clus(mat, ks=(2, 3)):
    X = StandardScaler().fit_transform(mat); p = PCA(n_components=0.85, svd_solver='full', random_state=42).fit_transform(X)[:, :10]
    best = min(ks, key=lambda k: GaussianMixture(k, covariance_type='full', random_state=42, n_init=5).fit(p).bic(p))
    gm = GaussianMixture(best, covariance_type='full', random_state=42, n_init=5).fit(p)
    return pd.Series(gm.predict(p), index=mat.index), best
res = {}
rng = np.random.default_rng(42)
def perm(lab, y, n=10000):
    codes = pd.factorize(lab.values)[0]; cnt = np.bincount(codes); obs = (np.bincount(codes, weights=y) / cnt).max()
    mx = np.array([(np.bincount(codes, weights=rng.permutation(y)) / cnt).max() for _ in range(n)]); return float(obs), float((mx >= obs).mean())
for coh, panel in [('JHB_SCHARP_004', ['creatinine', 'hemoglobin', 'hematocrit']), ('JHB_DPHRU_013', ['diastolic_bp', 'heart_rate', 'waist_hip_ratio', 'body_fat_percent'])]:
    m = pd.concat([person(b) for b in panel], axis=1, join='inner'); m = m.loc[m.index.intersection(study.index[study == coh])]
    lab, k = clus(m); h = hiv.reindex(m.index).fillna(False).astype(int).values
    obs, p = perm(lab, h)
    res[coh] = dict(n=int(len(m)), k=int(k), sizes=np.bincount(pd.factorize(lab.values)[0]).tolist(),
        hiv_prev_by_cluster=pd.Series(h, index=m.index).groupby(lab).mean().round(3).tolist(), overall_hiv=float(h.mean()), max_prev=obs, perm_p=p)
    print(coh, res[coh], flush=True)
panel = ['hematocrit', 'hemoglobin', 'viral_load']
m = pd.concat([person(b) for b in panel], axis=1, join='inner'); m = m.loc[m.index.intersection(hiv.index[hiv])]
lab, k = clus(m, ks=(2, 3, 4))
vir = (vlmax.reindex(m.index) >= 1000).astype(int)
d = pd.DataFrame({'c': lab, 'viraemic_ever': vir, 'female': sex.reindex(m.index).str.lower().str.startswith('f').astype(int), 'age': age.reindex(m.index), 'study': study.reindex(m.index)})
summ = d.groupby('c').agg(n=('age', 'size'), viraemic_ever=('viraemic_ever', 'mean'), female=('female', 'mean'), age=('age', 'mean')).round(3)
o_v, p_v = perm(lab, vir.values); o_f, p_f = perm(lab, d['female'].values)
res['hivpos_hct_hb_vl'] = dict(n=int(len(m)), k=int(k), clusters=json.loads(summ.to_json(orient='index')),
    cohort_mix=pd.crosstab(d['c'], d['study']).to_dict(), viraemia_max=o_v, viraemia_perm_p=p_v, female_max=o_f, female_perm_p=p_f)
print(json.dumps(res['hivpos_hct_hb_vl'], default=str), flush=True)
json.dump(res, open(OUT / 'r1_within_panel.json', 'w'), indent=2, default=str)
