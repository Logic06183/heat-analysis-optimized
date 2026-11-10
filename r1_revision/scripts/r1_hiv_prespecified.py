"""Pre-specified HIV tests H1-H5 (GOAL: hiv-goal/GOAL.md, fixed 10 Nov 2026 before running).
Aggregate outputs only."""
import os, sys, json, logging, warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
warnings.filterwarnings("ignore")
os.environ.setdefault("RP2_PRIMARY_EXPOSURE", "era5_land")
REPO = Path(__file__).resolve().parents[2]  # repository root; sys.path.insert(0, str(REPO))
logging.basicConfig(level=logging.ERROR)
from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers
from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features
OUT = Path(__file__).resolve().parents[1] / "outputs"
S3 = config.OUTPUT_ROOT / "stage3_r2_005"
rng = np.random.default_rng(42)

df = build_analysis_dataset(); bios = get_available_biomarkers()
BIO = [b.column for b in bios.values() if b.column and b.column in df.columns]
df = engineer_features(df, BIO)
log = {}
if 'era5_temp_lag7d_c' in df.columns and df['era5_temp_lag7d_c'].notna().mean() > 0.5:
    df['T_primary'] = df['era5_temp_lag7d_c']; log['primary_exposure'] = 'era5_temp_lag7d_c (rolling 7-day mean)'
else:
    df['T_primary'] = df[['temp_lag_0d', 'temp_lag_1d', 'temp_lag_3d', 'temp_lag_7d']].mean(axis=1); log['primary_exposure'] = 'mean of ERA5-Land lags 0,1,3,7'
df['T_secondary'] = df[['temp_lag_0d', 'temp_lag_1d', 'temp_lag_3d']].mean(axis=1)
log['T_primary_nonmissing'] = float(df['T_primary'].notna().mean())
h = df['hiv_status'].astype(str).str.lower()
df['HIV'] = np.where(h.isin(['positive', '1', '1.0', 'true', 'pos']), 1, np.where(h.isin(['negative', '0', '0.0', 'false', 'neg']), 0, np.nan))
df['cohort'] = df['study_source'].astype(str)
df['female'] = df['sex'].astype(str).str.lower().str.startswith('f').astype(float)
df['month'] = pd.to_datetime(df['visit_date']).dt.month; df['year'] = pd.to_datetime(df['visit_date']).dt.year
pid = 'patient_id'

def fe_ols(d, y, xs, cluster=pid):
    """participant FE via within transformation; month dummies and year demeaned too; cluster SE by participant."""
    d = d.dropna(subset=[y] + xs).copy()
    d = d[d.groupby(pid)[y].transform('size') >= 2]
    if d[pid].nunique() < 30 or len(d) < 100: return None
    M = pd.get_dummies(d['month'], prefix='m', drop_first=True, dtype=float)
    Z = pd.concat([d[xs].astype(float), M, d[['year']].astype(float)], axis=1)
    Z = Z.loc[:, Z.std() > 0]
    cols = [c for c in xs if c in Z.columns]
    if len(cols) < len(xs): return None
    Yd = d[y] - d.groupby(pid)[y].transform('mean')
    Zd = Z - Z.groupby(d[pid]).transform('mean')
    Zd = Zd.loc[:, Zd.abs().sum() > 1e-9]
    r = sm.OLS(Yd.values, Zd.values).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d[pid])[0]})
    names = list(Zd.columns)
    return {c: (float(r.params[names.index(c)]), float(r.bse[names.index(c)]), float(r.pvalues[names.index(c)])) for c in cols if c in names} | {'n_obs': int(len(d)), 'n_part': int(d[pid].nunique()), 'cohorts': sorted(d['cohort'].unique().tolist())}

def pool(ests):
    ests = [e for e in ests if e and np.isfinite(e[1]) and e[1] > 0]
    if not ests: return None
    w = np.array([1 / s**2 for _, s in ests]); b = np.array([e for e, _ in ests])
    est = float((w * b).sum() / w.sum()); se = float(np.sqrt(1 / w.sum())); p = float(2 * stats.norm.sf(abs(est / se)))
    return dict(est=est, se=se, lo=est - 1.96 * se, hi=est + 1.96 * se, p=p, k=len(ests))

R = {'log': log}
# ---------------- H1 ----------------
MIXED = ['JHB_DPHRU_013', 'JHB_DPHRU_053', 'JHB_SCHARP_004']
def h1(T, cohorts=MIXED):
    d0 = df[df['cohort'].isin(cohorts)].copy(); per = {}; ests = []
    for b in BIO:
        dd = d0.dropna(subset=[b, 'HIV'])
        if dd.empty: continue
        cnt = dd.groupby(['cohort', pid])['HIV'].first().groupby('cohort').agg(['sum', 'count'])
        cnt['neg'] = cnt['count'] - cnt['sum']
        eligible = (len(cnt) >= 2) or ((cnt['sum'] >= 100) & (cnt['neg'] >= 100)).any()
        if not eligible: continue
        sd = df[b].std(); dd = dd.assign(Y=dd[b] / sd, TxH=dd[T] * dd['HIV'])
        r = fe_ols(dd, 'Y', [T, 'TxH'])
        if r is None or 'TxH' not in r: per[b] = 'not estimable'; continue
        per[b] = dict(interaction=r['TxH'], main_T=r[T], n_obs=r['n_obs'], n_part=r['n_part'], cohorts=r['cohorts'])
        ests.append((r['TxH'][0], r['TxH'][1]))
    return pool(ests), per
p1, per1 = h1('T_primary')
R['H1'] = dict(primary=p1, per_biomarker=per1, secondary=h1('T_secondary')[0],
               loco={c: h1('T_primary', [m for m in MIXED if m != c])[0] for c in MIXED})
print('H1', json.dumps(R['H1']['primary']), flush=True)
# ---------------- H2 / H3 ----------------
pos = df[df['HIV'] == 1].copy()
pos['log10vl'] = np.log10(pos['viral_load'].clip(lower=1))
pos['viraemic'] = (pos['viral_load'] >= 1000).astype(float).where(pos['viral_load'].notna())
pos['logcd4'] = np.log(pos['cd4_count'].clip(lower=1))
def condlogit(d, T):
    from statsmodels.discrete.conditional_models import ConditionalLogit
    d = d.dropna(subset=['viraemic', T]).copy()
    v = d.groupby(pid)['viraemic'].agg(['min', 'max']); keep = v.index[v['min'] != v['max']]
    d = d[d[pid].isin(keep)]
    M = pd.get_dummies(d['month'], prefix='m', drop_first=True, dtype=float)
    X = pd.concat([d[[T]].astype(float), M, d[['year']].astype(float)], axis=1)
    try:
        r = ConditionalLogit(d['viraemic'].values, X.values, groups=pd.factorize(d[pid])[0]).fit(disp=0)
        return dict(model='conditional logit', est=float(r.params[0]), se=float(r.bse[0]), p=float(r.pvalues[0]), OR_per_C=float(np.exp(r.params[0])), n_obs=int(len(d)), n_part=int(d[pid].nunique()), cohorts=sorted(d['cohort'].unique().tolist()))
    except Exception as e:
        r = fe_ols(d.assign(Y=d['viraemic']), 'Y', [T])
        return dict(model='LPM FE (clogit failed: %s)' % e, est=r[T][0], se=r[T][1], p=r[T][2], n_obs=r['n_obs'], n_part=r['n_part'], cohorts=r['cohorts'])
def h2(T, d=pos): return condlogit(d, T)
def h_lin(y, T, d=pos):
    r = fe_ols(d, y, [T]);
    return None if r is None else dict(est=r[T][0], se=r[T][1], p=r[T][2], n_obs=r['n_obs'], n_part=r['n_part'], cohorts=r['cohorts'])
coh_vl = sorted(pos.dropna(subset=['viral_load'])['cohort'].unique()); coh_cd4 = sorted(pos.dropna(subset=['cd4_count'])['cohort'].unique())
R['H2'] = dict(primary=h2('T_primary'), log10vl=h_lin('log10vl', 'T_primary'), secondary=h2('T_secondary'),
               loco={c: h2('T_primary', pos[pos['cohort'] != c]) for c in coh_vl})
print('H2', json.dumps(R['H2']['primary']), flush=True)
R['H3'] = dict(primary=h_lin('logcd4', 'T_primary'), secondary=h_lin('logcd4', 'T_secondary'),
               loco={c: h_lin('logcd4', 'T_primary', pos[pos['cohort'] != c]) for c in coh_cd4})
print('H3', json.dumps(R['H3']['primary']), flush=True)
# ---------------- H4 ----------------
lab = pd.read_csv(S3 / 'cluster_labels.csv').set_index(pid)['cluster']
meta = pd.read_csv(S3 / 'patient_cluster_table.csv').set_index(pid)
hv = meta['hiv_status'].astype(str).str.lower().eq('positive').astype(int)
coh = df.groupby(pid)['cohort'].agg(lambda s: s.mode().iat[0])
c3 = int(pd.DataFrame({'c': lab, 'h': hv.reindex(lab.index)}).groupby('c')['h'].mean().idxmax())
P = pd.DataFrame({'c3': (lab == c3).astype(int), 'hiv': hv.reindex(lab.index), 'coh': coh.reindex(lab.index)}).dropna()
from statsmodels.stats.contingency_tables import StratifiedTable
tabs = []
for c, g in P.groupby('coh'):
    t = pd.crosstab(g['hiv'], g['c3']).reindex(index=[0, 1], columns=[0, 1], fill_value=0).values
    if (t.sum(axis=1) > 0).all() and (t.sum(axis=0) > 0).all(): tabs.append(t.astype(float) + 0.0)
st = StratifiedTable(tabs); cmh = st.test_null_odds(correction=True)
obs = P.loc[P['c3'] == 1, 'hiv'].mean()
hv_arr = P['hiv'].values.astype(float); grp = P['coh'].values; c3m = P['c3'].values == 1
idx = {g: np.where(grp == g)[0] for g in np.unique(grp)}; null = np.empty(10000)
for i in range(10000):
    hp = hv_arr.copy()
    for g, ii in idx.items(): hp[ii] = rng.permutation(hv_arr[ii])
    null[i] = hp[c3m].mean()
R['H4'] = dict(cmh_stat=float(cmh.statistic), p=float(cmh.pvalue), pooled_OR=float(st.oddsratio_pooled), n_strata_informative=len(tabs),
               within_cohort_perm_obs=float(obs), within_cohort_perm_null_max=float(null.max()), within_cohort_perm_p=float((null >= obs).mean()))
print('H4', json.dumps(R['H4']), flush=True)
# ---------------- H5 ----------------
pos['lowcd4'] = (pos['cd4_count'] < 350).astype(float).where(pos['cd4_count'].notna())
def h5(T, modifier='lowcd4', d=pos):
    ests, per = [], {}
    for b in ['hematocrit', 'hemoglobin']:
        sd = df[b].std(); dd = d.dropna(subset=[b, modifier, T]).copy()
        dd = dd.assign(Y=dd[b] / sd, TxM=dd[T] * dd[modifier], M=dd[modifier])
        r = fe_ols(dd, 'Y', [T, 'M', 'TxM'])
        if r is None or 'TxM' not in r: per[b] = 'not estimable'; continue
        per[b] = dict(interaction=r['TxM'], n_obs=r['n_obs'], n_part=r['n_part'], cohorts=r['cohorts']); ests.append((r['TxM'][0], r['TxM'][1]))
    return pool(ests), per
p5, per5 = h5('T_primary')
coh_h5 = sorted(pos.dropna(subset=['lowcd4']).dropna(subset=['hematocrit', 'hemoglobin'], how='all')['cohort'].unique())
R['H5'] = dict(primary=p5, per_biomarker=per5, secondary=h5('T_secondary')[0], viraemic_modifier=h5('T_primary', 'viraemic')[0],
               loco={c: h5('T_primary', d=pos[pos['cohort'] != c])[0] for c in coh_h5})
print('H5', json.dumps(R['H5']['primary']), flush=True)
# ---------------- BH ----------------
ps = {'H1': (R['H1']['primary'] or {}).get('p'), 'H2': R['H2']['primary'].get('p'), 'H3': (R['H3']['primary'] or {}).get('p'),
      'H4': R['H4']['p'], 'H5': (R['H5']['primary'] or {}).get('p')}
keys = [k for k, v in ps.items() if v is not None]; pv = np.array([ps[k] for k in keys]); m = len(pv)
order = np.argsort(pv); q = np.empty(m); prev = 1.0
for rank, i in reversed(list(enumerate(order, start=1))):
    prev = min(prev, pv[i] * m / rank); q[i] = prev
R['BH'] = {k: dict(p=float(ps[k]), q=float(q[keys.index(k)])) for k in keys}
print('BH', json.dumps(R['BH']), flush=True)
json.dump(R, open(OUT / 'r1_hiv_prespecified.json', 'w'), indent=2, default=str)
print('written')
