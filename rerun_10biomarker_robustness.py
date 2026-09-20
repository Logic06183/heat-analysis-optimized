# -*- coding: utf-8 -*-
"""Re-run of the Stage 3 robustness checks on the TEN-biomarker panel.

The published clustering is mcd_outputs_era5_land/stage3_r2_005 (10 biomarkers,
9,293 patients, cluster sizes 4,377 / 1,247 / 3,669).  The versions of these
checks currently in the repository were computed on stage3/ (13 biomarkers,
cluster sizes 4,439 / 1,264 / 3,590), which is why the main text and the
appendix disagree.

Everything here operates on the published PC scores and cluster labels, so it
reproduces the primary solution exactly by construction.  Hyperparameters are
taken from mcd_pipeline/config.py and run_stage3.py:

    GMM k=3, covariance 'full', n_init=5, max_iter=300, seed 42
    clustering uses the first PCA_CLUSTERING_DIMS=10 principal components

Covers three of the five claims:
  1. leave-one-cohort-out          (main text: 69.0-90.3%, mean 78.4%)
  3. multi-seed stability          (main text: "reproduced it exactly, ARI 1.000")
  4. HIV permutation test          (main text: null max 55.7%, p<0.0001)

Leave-one-biomarker-out needs the pre-PCA SHAP matrix and is handled separately.
"""
import json
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

SRC = '/mnt/user-data/uploads/heat-analysis-optimized/mcd_outputs_era5_land/stage3_r2_005'
OUT = '/home/claude/lee/rerun'

GMM_K, GMM_COV, GMM_N_INIT, GMM_MAX_ITER = 3, 'full', 5, 300
PCA_CLUSTERING_DIMS = 10
MASTER_SEED = 42
N_PERM = 10000

pc = pd.read_csv(f'{SRC}/pc_scores.csv').set_index('patient_id')
pc10 = pc.iloc[:, :PCA_CLUSTERING_DIMS]
labels = pd.read_csv(f'{SRC}/cluster_labels.csv').set_index('patient_id')['cluster']
meta = pd.read_csv(f'{SRC}/patient_cluster_table.csv').set_index('patient_id')

hiv = meta['hiv_status'].astype(str).str.lower().eq('positive').astype(int)
cohort = meta['study_source']
labels = labels.reindex(pc10.index)
hiv = hiv.reindex(pc10.index).fillna(0).astype(int)
cohort = cohort.reindex(pc10.index)

print('patients %d, PCs used %d' % (len(pc10), pc10.shape[1]))
print('cluster sizes', dict(labels.value_counts().sort_index()))

prim = pd.DataFrame({'cluster': labels, 'hiv': hiv}).groupby('cluster')['hiv'].agg(['count', 'sum'])
prim['prev'] = prim['sum'] / prim['count']
print(prim)
BURDEN = int(prim['prev'].idxmax())
print('HIV-burden cluster: %d  n=%d  prev=%.4f' % (BURDEN, prim.loc[BURDEN, 'count'], prim.loc[BURDEN, 'prev']))


def fit(X, seed=MASTER_SEED):
    return GaussianMixture(n_components=GMM_K, covariance_type=GMM_COV,
                           random_state=seed, n_init=GMM_N_INIT,
                           max_iter=GMM_MAX_ITER).fit_predict(X)


# ---------------------------------------------------------------- sanity check
# refitting on the full PC10 matrix must reproduce the stored labels
check = fit(pc10.values)
ari_self = adjusted_rand_score(labels.values, check)
print('\nself-consistency ARI (refit vs stored labels): %.6f' % ari_self)

# ------------------------------------------------------- 1. leave-one-cohort-out
loco = []
for c in sorted(cohort.dropna().unique()):
    keep = cohort != c
    if int(keep.sum()) < 100:
        print('  skip %s (only %d remain)' % (c, keep.sum())); continue
    idx = pc10.index[keep]
    lab = pd.Series(fit(pc10.loc[idx].values), index=idx)
    ari = adjusted_rand_score(labels.loc[idx].values, lab.values)
    g = pd.DataFrame({'cluster': lab, 'hiv': hiv.loc[idx]}).groupby('cluster')['hiv'].agg(['count', 'sum'])
    g['prev'] = g['sum'] / g['count']
    bid = int(g['prev'].idxmax())
    loco.append({'cohort_dropped': c, 'n_kept': int(keep.sum()), 'n_dropped': int((~keep).sum()),
                 'ari_vs_primary': float(ari), 'hiv_burden_cluster_id': bid,
                 'hiv_burden_cluster_size': int(g.loc[bid, 'count']),
                 'hiv_burden_cluster_prevalence': float(g.loc[bid, 'prev'])})
    print('  drop %-18s n=%5d ARI=%.3f burden n=%5d HIV=%.1f%%'
          % (c, keep.sum(), ari, g.loc[bid, 'count'], 100 * g.loc[bid, 'prev']))

lp = [r['hiv_burden_cluster_prevalence'] for r in loco]
la = [r['ari_vs_primary'] for r in loco]
loco_stats = {'n_runs': len(loco),
              'hiv_prev_min': float(min(lp)), 'hiv_prev_max': float(max(lp)),
              'hiv_prev_mean': float(np.mean(lp)),
              'ari_min': float(min(la)), 'ari_max': float(max(la)),
              'ari_mean': float(np.mean(la)), 'ari_median': float(np.median(la))}
print('\nLOCO: %d runs, HIV %.1f-%.1f%% mean %.1f%%, ARI median %.3f range %.3f-%.3f'
      % (loco_stats['n_runs'], 100 * loco_stats['hiv_prev_min'], 100 * loco_stats['hiv_prev_max'],
         100 * loco_stats['hiv_prev_mean'], loco_stats['ari_median'],
         loco_stats['ari_min'], loco_stats['ari_max']))

# --------------------------------------------------------- 3. multi-seed stability
SEEDS = [42, 7, 123, 2024, 17, 314, 99, 1234, 9999, 11]
seed_labels = {s: fit(pc10.values, seed=s) for s in SEEDS}
pair = []
for i in range(len(SEEDS)):
    for j in range(i + 1, len(SEEDS)):
        pair.append(adjusted_rand_score(seed_labels[SEEDS[i]], seed_labels[SEEDS[j]]))
vs_primary = [adjusted_rand_score(labels.values, seed_labels[s]) for s in SEEDS]
burden_prev_by_seed = []
for s in SEEDS:
    g = pd.DataFrame({'cluster': seed_labels[s], 'hiv': hiv.values}).groupby('cluster')['hiv'].mean()
    burden_prev_by_seed.append(float(g.max()))
seed_stats = {'n_seeds': len(SEEDS), 'n_pairwise': len(pair),
              'pairwise_ari_mean': float(np.mean(pair)), 'pairwise_ari_min': float(min(pair)),
              'pairwise_ari_max': float(max(pair)), 'pairwise_ari_median': float(np.median(pair)),
              'ari_vs_primary_min': float(min(vs_primary)), 'ari_vs_primary_mean': float(np.mean(vs_primary)),
              'burden_prevalence_min': float(min(burden_prev_by_seed)),
              'burden_prevalence_max': float(max(burden_prev_by_seed))}
print('\nmulti-seed: pairwise ARI mean %.3f min %.3f median %.3f; vs primary min %.3f; burden HIV %.1f-%.1f%%'
      % (seed_stats['pairwise_ari_mean'], seed_stats['pairwise_ari_min'], seed_stats['pairwise_ari_median'],
         seed_stats['ari_vs_primary_min'], 100 * seed_stats['burden_prevalence_min'],
         100 * seed_stats['burden_prevalence_max']))

# ------------------------------------------------------- 4. HIV permutation test
rng = np.random.default_rng(MASTER_SEED)
hiv_arr = hiv.values.copy()
lab_arr = labels.values
observed = float(pd.DataFrame({'c': lab_arr, 'h': hiv_arr}).groupby('c')['h'].mean().max())
null_max = np.empty(N_PERM)
for k in range(N_PERM):
    perm = rng.permutation(hiv_arr)
    sums = np.bincount(lab_arr, weights=perm, minlength=GMM_K)
    cnts = np.bincount(lab_arr, minlength=GMM_K)
    null_max[k] = (sums / cnts).max()
n_ge = int((null_max >= observed).sum())
p_emp = (n_ge + 1) / (N_PERM + 1)
perm_stats = {'n_iterations': N_PERM, 'observed_max_cluster_hiv_pct': 100 * observed,
              'null_mean_pct': float(100 * null_max.mean()), 'null_sd_pct': float(100 * null_max.std(ddof=1)),
              'null_p95_pct': float(100 * np.percentile(null_max, 95)),
              'null_max_observed_pct': float(100 * null_max.max()),
              'n_null_ge_observed': n_ge,
              'empirical_p_value': p_emp,
              'reportable_p': 'p<0.0001' if p_emp < 0.0001 else 'p=%.4f' % p_emp}
print('\npermutation (%d iters): observed %.1f%%, null mean %.1f%% max %.1f%%, %d >= observed, p=%.5f -> %s'
      % (N_PERM, perm_stats['observed_max_cluster_hiv_pct'], perm_stats['null_mean_pct'],
         perm_stats['null_max_observed_pct'], n_ge, p_emp, perm_stats['reportable_p']))

summary = {
    'panel': 'ten-biomarker (stage3_r2_005)',
    'n_patients': int(len(pc10)),
    'primary_cluster_sizes': {int(k): int(v) for k, v in labels.value_counts().sort_index().items()},
    'primary_hiv_prevalence_by_cluster': {int(k): float(v) for k, v in prim['prev'].items()},
    'hiv_burden_cluster_id': BURDEN,
    'hiv_burden_cluster_prevalence': float(prim.loc[BURDEN, 'prev']),
    'self_consistency_ari': float(ari_self),
    'leave_one_cohort_out': {'summary': loco_stats, 'per_cohort': loco},
    'multi_seed_stability': seed_stats,
    'hiv_permutation_test': perm_stats,
}
with open(f'{OUT}/robustness_10biomarker.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('\nwritten %s/robustness_10biomarker.json' % OUT)
