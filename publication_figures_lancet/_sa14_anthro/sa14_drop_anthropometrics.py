# -*- coding: utf-8 -*-
"""SA14: Stage 3 clustering with the two anthropometric biomarkers removed.

Rebuilds the person-level temperature-lag SHAP matrix for the ten retained
biomarkers exactly as mcd_pipeline.stage3_clustering.shap_pca does, reproduces
the primary GMM k=3 solution as a validity check, then refits on the eight
non-anthropometric biomarkers (body fat percentage and waist-hip ratio dropped
together). Reports agreement with the primary grouping, the HIV-burden
cluster's size and prevalence, demographics, and bootstrap stability.

Run from the repository root:
  PYTHONPATH=. RP2_PRIMARY_EXPOSURE=era5_land /usr/local/bin/python3 \
      publication_figures_lancet/_sa14_anthro/sa14_drop_anthropometrics.py
"""
import json, sys, logging
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

logging.basicConfig(level=logging.WARNING)
from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers
from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features
from mcd_pipeline.stage1_lag_profiling.xgboost_trainer import prepare_features_and_target

S1 = config.OUTPUT_ROOT / 'stage1'
S3 = config.OUTPUT_ROOT / 'stage3_r2_005'
OUT = Path(__file__).resolve().parent
RETAINED = ['creatinine', 'diastolic_bp', 'heart_rate', 'cd4_count', 'viral_load',
            'ldl_cholesterol', 'waist_hip_ratio', 'body_fat_percent', 'hemoglobin', 'hematocrit']
ANTHRO = ['waist_hip_ratio', 'body_fat_percent']
TEMP_LAG = ['temp_lag_%dd' % d for d in config.LAG_DAYS]
VAR_THRESH, CLUST_DIMS = 0.85, 10
K, COV, N_INIT, MAX_ITER, SEED = 3, 'full', 5, 300, 42
N_BOOT = 500

print('primary exposure:', config.PRIMARY_EXPOSURE_SOURCE, '| stage1:', S1)
df = build_analysis_dataset()
bios = get_available_biomarkers()
bio_cols = [b.column for b in bios.values() if b.column and b.column in df.columns]
df = engineer_features(df, bio_cols)
print('dataset rows:', len(df))

per = {}
for name in RETAINED:
    sv = np.load(S1 / name / 'shap_values.npy')
    feats = json.load(open(S1 / name / 'feature_names.json'))
    X, _, groups = prepare_features_and_target(df, bios[name])
    assert sv.shape[0] == len(X), (name, sv.shape, len(X))
    idx = [i for i, f in enumerate(feats) if f in TEMP_LAG]
    sub = pd.DataFrame(sv[:, idx], columns=[feats[i] for i in idx])
    sub.index = groups.reset_index(drop=True).values
    sub.index.name = 'patient_id'
    p = sub.groupby('patient_id').mean()
    p.columns = ['%s__%s' % (name, c) for c in p.columns]
    per[name] = p
    print('  %-18s obs=%7d participants=%5d' % (name, sv.shape[0], len(p)))

labels = pd.read_csv(S3 / 'cluster_labels.csv').set_index('patient_id')['cluster']
meta = pd.read_csv(S3 / 'patient_cluster_table.csv').set_index('patient_id')
hiv = meta['hiv_status'].astype(str).str.lower().eq('positive').astype(int)

def fit(mat, seed=SEED):
    X = StandardScaler().fit_transform(mat)
    pca = PCA(n_components=VAR_THRESH, svd_solver='full', random_state=SEED)
    pcs = pca.fit_transform(X)
    n = min(CLUST_DIMS, pcs.shape[1])
    gmm = GaussianMixture(n_components=K, covariance_type=COV, random_state=seed,
                          n_init=N_INIT, max_iter=MAX_ITER).fit(pcs[:, :n])
    lab = pd.Series(gmm.predict(pcs[:, :n]), index=mat.index)
    return lab, pca.n_components_, pcs[:, :n], gmm

def describe(lab):
    g = pd.DataFrame({'c': lab})
    g['hiv'] = hiv.reindex(lab.index).fillna(0).astype(int)
    g['age'] = meta['age_years'].reindex(lab.index)
    g['unemp'] = meta['gcro_employment_status'].reindex(lab.index).astype(str).str.lower().str.startswith('not').astype(int)
    g['informal'] = meta['gcro_dwelling_type'].reindex(lab.index).astype(str).str.lower().eq('informal').astype(int)
    s = g.groupby('c').agg(n=('hiv', 'size'), hiv=('hiv', 'mean'), age=('age', 'mean'),
                           unemp=('unemp', 'mean'), informal=('informal', 'mean'))
    s['share'] = s['n'] / s['n'].sum()
    return s

def bootstrap_ari(mat, pcs_fit, gmm, n_boot=N_BOOT):
    """Same procedure as mcd_pipeline.stage3_clustering.run_stage3._gmm_bootstrap_stability:
    resample with replacement, fit GMM on the bootstrap sample, predict the full sample,
    ARI against the reference labels."""
    rng = np.random.default_rng(SEED)
    ref = gmm.predict(pcs_fit)
    n = len(pcs_fit)
    aris = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        g = GaussianMixture(n_components=K, covariance_type=COV, random_state=int(rng.integers(0, 2**31)),
                            n_init=3, max_iter=MAX_ITER).fit(pcs_fit[idx])
        aris[i] = adjusted_rand_score(ref, g.predict(pcs_fit))
    return float(aris.mean()), float(np.percentile(aris, 2.5)), float(np.percentile(aris, 97.5)), aris

# --- rebuild the primary ten-biomarker solution (validity check) ---
full = pd.concat(list(per.values()), axis=1, join='outer').fillna(0.0)
lab10, nc10, pcs10, gmm10 = fit(full)
common = lab10.index.intersection(labels.index)
ari_pub = adjusted_rand_score(labels.loc[common].values, lab10.loc[common].values)
d10 = describe(lab10)
print('\nTEN-biomarker rebuild: %d participants, %d PCs, ARI vs published labels %.4f' % (len(lab10), nc10, ari_pub))
print(d10.round(3))

# --- eight biomarkers: anthropometrics dropped ---
eight = pd.concat([v for k, v in per.items() if k not in ANTHRO], axis=1, join='outer').fillna(0.0)
lab8, nc8, pcs8, gmm8 = fit(eight)
common = lab8.index.intersection(lab10.index)
ari_8v10 = adjusted_rand_score(lab10.loc[common].values, lab8.loc[common].values)
ari_8vpub = adjusted_rand_score(labels.loc[lab8.index.intersection(labels.index)].values,
                                lab8.loc[lab8.index.intersection(labels.index)].values)
d8 = describe(lab8)
print('\nEIGHT-biomarker (no anthropometrics): %d participants, %d PCs, ARI vs ten-biomarker %.4f, vs published %.4f'
      % (len(lab8), nc8, ari_8v10, ari_8vpub))
print(d8.round(3))

# cross-tabulation: where do the ten-biomarker HIV-burden cluster's members go?
b10 = int(d10['hiv'].idxmax()); b8 = int(d8['hiv'].idxmax())
ct = pd.crosstab(lab10.loc[common], lab8.loc[common])
print('\ncross-tab (rows: ten-biomarker cluster, cols: eight-biomarker cluster)')
print(ct)
kept = float(ct.loc[b10, b8] / ct.loc[b10].sum())
print('share of ten-biomarker HIV-burden cluster (id %d) landing in eight-biomarker HIV-burden cluster (id %d): %.3f' % (b10, b8, kept))

# participants who lose all retained measurements without the anthropometrics
lost = len(lab10) - len(lab8)
print('participants clustered on ten but not eight (only anthropometrics measured): %d' % lost)

# bootstrap stability of the eight-biomarker solution
bm10, blo10, bhi10, a10 = bootstrap_ari(full, pcs10, gmm10)
print('ten-biomarker bootstrap ARI (pipeline method): mean %.3f (2.5-97.5%%: %.3f-%.3f) over %d resamples' % (bm10, blo10, bhi10, N_BOOT))
bm, blo, bhi, a8 = bootstrap_ari(eight, pcs8, gmm8)
print('eight-biomarker bootstrap ARI: mean %.3f (2.5-97.5%%: %.3f-%.3f) over %d resamples' % (bm, blo, bhi, N_BOOT))
print('eight-biomarker bootstrap ARI: share of resamples with ARI >= 0.8: %.3f; median %.3f' % (float((a8 >= 0.8).mean()), float(np.median(a8))))
# where do the HIV-negative members of the ten-biomarker HIV-burden cluster go?
neg10 = lab10.index[(lab10 == b10) & (hiv.reindex(lab10.index).fillna(0) == 0)]
neg10 = neg10.intersection(lab8.index)
print('HIV-negative members of ten-biomarker cluster %d: %d; their eight-biomarker destinations: %s' % (b10, len(neg10), dict(lab8.loc[neg10].value_counts().sort_index())))
pos10 = lab10.index[(lab10 == b10) & (hiv.reindex(lab10.index).fillna(0) == 1)].intersection(lab8.index)
print('HIV-positive members of ten-biomarker cluster %d: %d; destinations: %s' % (b10, len(pos10), dict(lab8.loc[pos10].value_counts().sort_index())))
np.save(OUT / 'sa14_bootstrap_ari_eight.npy', a8); np.save(OUT / 'sa14_bootstrap_ari_ten.npy', a10)

# label permutation for the eight-biomarker HIV-burden cluster
rng = np.random.RandomState(SEED)
h = hiv.reindex(lab8.index).fillna(0).astype(int).values
obs = d8.loc[b8, 'hiv']
maxes = []
for i in range(10000):
    hp = rng.permutation(h)
    maxes.append(pd.Series(hp).groupby(lab8.values).mean().max())
maxes = np.array(maxes)
print('permutation: observed %.3f, max under 10,000 reshuffles %.3f, p = %.5f' % (obs, maxes.max(), float((maxes >= obs).mean())))

out = {
  'ten_biomarker_rebuild': {'n': int(len(lab10)), 'n_pcs': int(nc10), 'ari_vs_published': float(ari_pub),
                            'clusters': json.loads(d10.to_json(orient='index'))},
  'eight_biomarker': {'n': int(len(lab8)), 'n_pcs': int(nc8), 'ari_vs_ten': float(ari_8v10), 'ari_vs_published': float(ari_8vpub),
                      'clusters': json.loads(d8.to_json(orient='index')),
                      'hiv_burden_cluster_id': b8, 'ten_burden_members_retained_share': kept,
                      'participants_lost': int(lost),
                      'bootstrap_ari_mean': bm, 'bootstrap_ari_2p5': blo, 'bootstrap_ari_97p5': bhi, 'bootstrap_ari_median': float(np.median(a8)), 'bootstrap_share_ge_0p8': float((a8 >= 0.8).mean()), 'n_boot': N_BOOT,
                      'ten_rebuild_bootstrap_ari_mean': bm10, 'ten_rebuild_bootstrap_2p5': blo10, 'ten_rebuild_bootstrap_97p5': bhi10,
                      'permutation_max_prev': float(maxes.max()), 'permutation_p': float((maxes >= obs).mean())},
  'crosstab': json.loads(ct.to_json()),
}
json.dump(out, open(OUT / 'sa14_results.json', 'w'), indent=2)
lab8.rename('cluster_eight').to_csv(OUT / 'sa14_cluster_labels_eight.csv')
print('\nwritten', OUT / 'sa14_results.json')
