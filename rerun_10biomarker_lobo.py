# -*- coding: utf-8 -*-
"""Leave-one-biomarker-out on the TEN-biomarker panel.

The repository's leave_one_biomarker_out.json has 13 runs on the 13-biomarker
panel.  The paper claims 10 single-biomarker-drop refits.  This rebuilds the
person-level temperature-lag SHAP matrix for the ten retained biomarkers,
reproduces the primary clustering from it as a validity check, then drops each
biomarker in turn and refits.

Pipeline replicated from mcd_pipeline/stage3_clustering/shap_pca.py:
  per biomarker -> keep temp_lag_* SHAP columns -> mean over each patient's rows
  -> concat across biomarkers, outer join, missing filled with 0
  -> StandardScaler -> PCA(n_components=0.85) -> GMM(k=3, full, n_init=5, seed 42)
     on the first 10 PCs
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

UP = '/mnt/user-data/uploads'
S1 = f'{UP}/heat-analysis-optimized/mcd_outputs_era5_land/stage1'
MASK = f'{UP}/Downloads/rerun10_masks'
S3 = f'{UP}/heat-analysis-optimized/mcd_outputs_era5_land/stage3_r2_005'
OUT = '/home/claude/lee/rerun'

RETAINED = ['creatinine', 'diastolic_bp', 'heart_rate', 'cd4_count', 'viral_load',
            'ldl_cholesterol', 'waist_hip_ratio', 'body_fat_percent',
            'hemoglobin', 'hematocrit']
LAG_DAYS = [0, 1, 3, 7, 14, 21, 30]
TEMP_LAG = ['temp_lag_%dd' % d for d in LAG_DAYS]
VAR_THRESH, CLUST_DIMS = 0.85, 10
K, COV, N_INIT, MAX_ITER, SEED = 3, 'full', 5, 300, 42

per = {}
for name in RETAINED:
    sv = np.load(f'{S1}/{name}/shap_values.npy')
    feats = json.load(open(f'{S1}/{name}/feature_names.json'))
    pid = pd.read_csv(f'{MASK}/{name}_patient_ids.csv')['patient_id']
    assert sv.shape[0] == len(pid), (name, sv.shape[0], len(pid))
    idx = [i for i, f in enumerate(feats) if f in TEMP_LAG]
    sub = pd.DataFrame(sv[:, idx], columns=[feats[i] for i in idx])
    sub.index = pid.values
    sub.index.name = 'patient_id'
    p = sub.groupby('patient_id').mean()
    p.columns = ['%s__%s' % (name, c) for c in p.columns]
    per[name] = p
    print('  %-18s obs=%7d patients=%5d lagfeats=%d' % (name, sv.shape[0], len(p), len(idx)))

labels = pd.read_csv(f'{S3}/cluster_labels.csv').set_index('patient_id')['cluster']
meta = pd.read_csv(f'{S3}/patient_cluster_table.csv').set_index('patient_id')


def cluster(mat, seed=SEED):
    X = StandardScaler().fit_transform(mat)
    pca = PCA(n_components=VAR_THRESH, svd_solver='full', random_state=SEED)
    pcs = pca.fit_transform(X)
    n = min(CLUST_DIMS, pcs.shape[1])
    lab = GaussianMixture(n_components=K, covariance_type=COV, random_state=seed,
                          n_init=N_INIT, max_iter=MAX_ITER).fit_predict(pcs[:, :n])
    return pd.Series(lab, index=mat.index), pca.n_components_


full = pd.concat(list(per.values()), axis=1, join='outer').fillna(0.0)
print('\ncombined matrix: %d patients x %d features' % full.shape)

base_lab, ncomp = cluster(full)
common = base_lab.index.intersection(labels.index)
ari_base = adjusted_rand_score(labels.loc[common].values, base_lab.loc[common].values)
print('rebuilt primary: %d PCA components, ARI vs published labels = %.4f' % (ncomp, ari_base))
print('rebuilt cluster sizes:', dict(base_lab.value_counts().sort_index()))
print('published cluster sizes:', dict(labels.value_counts().sort_index()))

hiv = meta['hiv_status'].astype(str).str.lower().eq('positive').astype(int)


def burden(lab):
    g = pd.DataFrame({'c': lab, 'h': hiv.reindex(lab.index).fillna(0).astype(int)})
    s = g.groupby('c')['h'].agg(['count', 'mean'])
    b = s['mean'].idxmax()
    return int(b), int(s.loc[b, 'count']), float(s.loc[b, 'mean'])

bid, bn, bp = burden(base_lab)
print('rebuilt HIV-burden cluster: id=%d n=%d prev=%.4f' % (bid, bn, bp))

runs = []
for drop in RETAINED:
    mat = pd.concat([v for k, v in per.items() if k != drop], axis=1, join='outer').fillna(0.0)
    lab, nc = cluster(mat)
    common = lab.index.intersection(base_lab.index)
    ari = adjusted_rand_score(base_lab.loc[common].values, lab.loc[common].values)
    i, n, p = burden(lab)
    runs.append({'biomarker_dropped': drop, 'n_biomarkers_kept': len(RETAINED) - 1,
                 'n_patients_clustered': int(len(lab)), 'n_pca_components': int(nc),
                 'ari_vs_primary': float(ari), 'hiv_burden_cluster_size': n,
                 'hiv_burden_cluster_prevalence': p})
    print('  drop %-18s n=%5d ARI=%.3f burden n=%5d HIV=%.1f%%' % (drop, len(lab), ari, n, 100 * p))

pv = [r['hiv_burden_cluster_prevalence'] for r in runs]
av = [r['ari_vs_primary'] for r in runs]
stats = {'n_runs': len(runs), 'hiv_prev_min': float(min(pv)), 'hiv_prev_max': float(max(pv)),
         'hiv_prev_mean': float(np.mean(pv)), 'ari_min': float(min(av)),
         'ari_max': float(max(av)), 'ari_mean': float(np.mean(av)),
         'ari_median': float(np.median(av))}
print('\nLOBO: %d runs, HIV %.1f-%.1f%% mean %.1f%%, ARI median %.3f range %.3f-%.3f'
      % (stats['n_runs'], 100 * stats['hiv_prev_min'], 100 * stats['hiv_prev_max'],
         100 * stats['hiv_prev_mean'], stats['ari_median'], stats['ari_min'], stats['ari_max']))

top2 = sorted(runs, key=lambda r: r['ari_vs_primary'])[:2]
print('two most influential (lowest ARI): %s' % ', '.join(
    '%s (ARI %.3f, HIV %.1f%%)' % (r['biomarker_dropped'], r['ari_vs_primary'],
                                   100 * r['hiv_burden_cluster_prevalence']) for r in top2))

json.dump({'panel': 'ten-biomarker (stage3_r2_005)',
           'rebuild_check': {'ari_vs_published_labels': float(ari_base),
                             'n_pca_components': int(ncomp),
                             'rebuilt_cluster_sizes': {int(k): int(v) for k, v in base_lab.value_counts().items()},
                             'rebuilt_burden_prevalence': bp},
           'summary': stats, 'per_biomarker': runs},
          open(f'{OUT}/lobo_10biomarker.json', 'w'), indent=2)
print('\nwritten %s/lobo_10biomarker.json' % OUT)
