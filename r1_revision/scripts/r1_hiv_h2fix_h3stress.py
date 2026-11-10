"""(1) H2 rerun: conditional logit with centred covariates + convergence check; LPM FE fallback per pre-specification.
(2) POST HOC stress tests of H3 (CD4, within-person), labelled post hoc: can only weaken, not promote.
   s1 time-since-first-visit cubic spline (ART-recovery trend); s2 person-specific linear trends;
   s3 month-of-year x cohort FE; s4 drop visits within 90 days of first visit; s5 restrict to visits with Hct measured, adjusting for Hct
   (does CD4 move beyond haemoconcentration?)."""
import os, sys, json, logging, warnings
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
warnings.filterwarnings("ignore")
os.environ.setdefault("RP2_PRIMARY_EXPOSURE", "era5_land")
REPO = Path(__file__).resolve().parents[2]  # repository root; sys.path.insert(0, str(REPO))
logging.basicConfig(level=logging.ERROR)
from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers
from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features
from statsmodels.discrete.conditional_models import ConditionalLogit
OUT = Path(__file__).resolve().parents[1] / "outputs"; pid = 'patient_id'
df = build_analysis_dataset(); bios = get_available_biomarkers()
df = engineer_features(df, [b.column for b in bios.values() if b.column and b.column in df.columns])
df['T'] = df['era5_temp_lag7d_c']; df['T2'] = df[['temp_lag_0d', 'temp_lag_1d', 'temp_lag_3d']].mean(axis=1)
h = df['hiv_status'].astype(str).str.lower(); df['HIV'] = np.where(h.isin(['positive','1','1.0','true','pos']),1,np.where(h.isin(['negative','0','0.0','false','neg']),0,np.nan))
df['cohort'] = df['study_source'].astype(str); vd = pd.to_datetime(df['visit_date'])
df['month'] = vd.dt.month; df['yearc'] = vd.dt.year - 2014
df['t_days'] = (vd - vd.groupby(df[pid]).transform('min')).dt.days
pos = df[df['HIV'] == 1].copy()
pos['viraemic'] = (pos['viral_load'] >= 1000).astype(float).where(pos['viral_load'].notna())
pos['logcd4'] = np.log(pos['cd4_count'].clip(lower=1))
R = {}
def clogit(d, T):
    d = d.dropna(subset=['viraemic', T]).copy()
    v = d.groupby(pid)['viraemic'].agg(['min', 'max']); d = d[d[pid].isin(v.index[v['min'] != v['max']])]
    M = pd.get_dummies(d['month'], prefix='m', drop_first=True, dtype=float)
    X = pd.concat([(d[[T]] - d[T].mean()).astype(float), M, d[['yearc']].astype(float)], axis=1)
    mod = ConditionalLogit(d['viraemic'].values, X.values, groups=pd.factorize(d[pid])[0])
    r = mod.fit(method='bfgs', maxiter=2000, disp=0)
    g = np.abs(mod.score(r.params)).max()
    conv = bool(np.isfinite(r.params).all() and g < 1e-2 and abs(r.params[0]) > 0)
    return dict(max_abs_score=float(g), converged=conv, est=float(r.params[0]), se=float(r.bse[0]), p=float(r.pvalues[0]), OR=float(np.exp(r.params[0])), n_obs=int(len(d)), n_part=int(d[pid].nunique()))
def fe(d, y, xs, extra=None, trend=False):
    d = d.dropna(subset=[y] + xs).copy(); d = d[d.groupby(pid)[y].transform('size') >= 2]
    parts = [d[xs].astype(float), pd.get_dummies(d['month'], prefix='m', drop_first=True, dtype=float), d[['yearc']].astype(float)]
    if extra is not None: parts.append(extra(d))
    Z = pd.concat(parts, axis=1); Z = Z.loc[:, Z.std() > 0]
    Yd = d[y] - d.groupby(pid)[y].transform('mean'); Zd = Z - Z.groupby(d[pid]).transform('mean')
    if trend:  # person-specific linear trends: residualise on within-person time
        tt = d['t_days'] - d.groupby(pid)['t_days'].transform('mean')
        def detr(s):
            b = (s * tt).groupby(d[pid]).sum() / (tt**2).groupby(d[pid]).sum().replace(0, np.nan)
            return s - tt * b.reindex(d[pid]).fillna(0).values
        Yd = detr(Yd); Zd = Zd.apply(detr)
    Zd = Zd.loc[:, Zd.abs().sum() > 1e-9]
    r = sm.OLS(Yd.values, Zd.values).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d[pid])[0]})
    i = list(Zd.columns).index(xs[0])
    return dict(est=float(r.params[i]), se=float(r.bse[i]), p=float(r.pvalues[i]), n_obs=int(len(d)), n_part=int(d[pid].nunique()))
c = clogit(pos, 'T'); R['H2_clogit'] = c
lpm = fe(pos.assign(Y=pos['viraemic']), 'Y', ['T']); R['H2_lpm'] = lpm
R['H2_primary_used'] = 'conditional logit' if c['converged'] else 'LPM FE (clogit not converged)'
R['H2_secondary_clogit'] = clogit(pos, 'T2'); R['H2_secondary_lpm'] = fe(pos.assign(Y=pos['viraemic']), 'Y', ['T2'])
R['H2_loco'] = {k: (clogit(pos[pos['cohort'] != k], 'T') if c['converged'] else fe(pos[pos['cohort'] != k].assign(Y=lambda x: x['viraemic']), 'Y', ['T'])) for k in sorted(pos.dropna(subset=['viraemic'])['cohort'].unique())}
print('H2', json.dumps(R['H2_clogit']), json.dumps(R['H2_lpm']), flush=True)
spline = lambda d: pd.DataFrame({'t1': d['t_days'] / 365, 't2': (d['t_days'] / 365)**2, 't3': (d['t_days'] / 365)**3}, index=d.index)
R['H3_posthoc'] = {
  'base': fe(pos, 'logcd4', ['T']),
  's1_time_since_first_visit_cubic': fe(pos, 'logcd4', ['T'], extra=spline),
  's2_person_linear_trend': fe(pos, 'logcd4', ['T'], trend=True),
  's3_cohort_x_month': fe(pos, 'logcd4', ['T'], extra=lambda d: pd.get_dummies(d['cohort'] + '_' + d['month'].astype(str), drop_first=True, dtype=float)),
  's4_drop_first_90_days': fe(pos[pos['t_days'] > 90], 'logcd4', ['T']),
  's5_adjust_hct_same_visit': fe(pos, 'logcd4', ['T', 'hematocrit']),
  's5_same_sample_no_hct': fe(pos.dropna(subset=['hematocrit']), 'logcd4', ['T']),
}
for k, v in R['H3_posthoc'].items(): print('H3', k, json.dumps(v), flush=True)
json.dump(R, open(OUT / 'r1_hiv_h2fix_h3posthoc.json', 'w'), indent=2)
print('written')
