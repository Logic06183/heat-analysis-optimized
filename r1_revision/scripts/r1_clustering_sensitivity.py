"""R1 analyses 7b, 7c, 7f, 7g: Stage 3 clustering sensitivity.

Rebuilds the person-level temperature-lag SHAP matrix exactly as SA11
(publication_figures_lancet/_sa14_anthro) and mcd_pipeline.stage3_clustering
do, checks it reproduces the published labels, then runs:

7b missingness-aware variants
   b1 availability-only: cluster on which biomarkers were measured (0/1), no SHAP
   b2 measured-only standardisation: each SHAP feature standardised on measured
      participants only, unmeasured set to the measured mean (0 after scaling),
      so an unmeasured biomarker counts as an average response, not a zero one
   b3 availability-pattern centred: SHAP features centred within each
      availability pattern before clustering (removes between-pattern shifts)
   b4 common-panel subset: participants with haematocrit, haemoglobin and HIV
      viral load all measured, clustered on those three biomarkers only
7c number of components k = 2..5 (BIC, AIC, silhouette, bootstrap ARI,
   HIV-enriched cluster)
7f excluding cohorts with HIV prevalence 100% or <1%
7g cohort-centred: SHAP features centred within cohort (measured values only)

For every variant: max cluster HIV prevalence, 10,000-reshuffle permutation
max and p, share of published Cluster 3 HIV-positive members landing in the
variant's HIV-enriched cluster, ARI vs published labels.
Aggregate outputs only.
"""
import json, os, sys, logging, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score

os.environ.setdefault("RP2_PRIMARY_EXPOSURE", "era5_land")
REPO = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(REPO))
logging.basicConfig(level=logging.WARNING)
from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers
from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features
from mcd_pipeline.stage1_lag_profiling.xgboost_trainer import prepare_features_and_target

OUT = Path(__file__).resolve().parents[1] / "outputs"; OUT.mkdir(exist_ok=True)
S1 = config.OUTPUT_ROOT / "stage1"; S3 = config.OUTPUT_ROOT / "stage3_r2_005"
RETAINED = ['creatinine', 'diastolic_bp', 'heart_rate', 'cd4_count', 'viral_load',
            'ldl_cholesterol', 'waist_hip_ratio', 'body_fat_percent', 'hemoglobin', 'hematocrit']
TEMP_LAG = ['temp_lag_%dd' % d for d in config.LAG_DAYS]
VAR_THRESH, CLUST_DIMS, COV, N_INIT, MAX_ITER, SEED = 0.85, 10, 'full', 5, 300, 42
N_BOOT = int(os.environ.get("R1_NBOOT_CLUST", "200")); N_PERM = 10000
t0 = time.time()

df = build_analysis_dataset()
bios = get_available_biomarkers()
df = engineer_features(df, [b.column for b in bios.values() if b.column and b.column in df.columns])
study = df.groupby('patient_id')['study_source'].agg(lambda s: s.mode().iat[0]).astype(str)
visits = {}
per = {}
for name in RETAINED:
    sv = np.load(S1 / name / 'shap_values.npy')
    feats = json.load(open(S1 / name / 'feature_names.json'))
    X, _, groups = prepare_features_and_target(df, bios[name])
    assert sv.shape[0] == len(X)
    idx = [i for i, f in enumerate(feats) if f in TEMP_LAG]
    sub = pd.DataFrame(sv[:, idx], columns=[feats[i] for i in idx])
    sub.index = groups.reset_index(drop=True).values; sub.index.name = 'patient_id'
    visits[name] = sub.groupby('patient_id').size().describe().to_dict()
    p = sub.groupby('patient_id').mean(); p.columns = ['%s__%s' % (name, c) for c in p.columns]
    per[name] = p
raw = pd.concat(list(per.values()), axis=1, join='outer')          # NaN = unmeasured
avail = pd.DataFrame({n: raw[[c for c in raw.columns if c.startswith(n + '__')]].notna().any(axis=1).astype(int) for n in RETAINED})
pattern = avail.astype(str).agg(''.join, axis=1)

pub = pd.read_csv(S3 / 'cluster_labels.csv').set_index('patient_id')['cluster']
meta = pd.read_csv(S3 / 'patient_cluster_table.csv').set_index('patient_id')
hiv_all = meta['hiv_status'].astype(str).str.lower().eq('positive').astype(int)
pub_c3 = int(pd.DataFrame({'c': pub, 'h': hiv_all.reindex(pub.index)}).groupby('c')['h'].mean().idxmax())
c3_pos = pub.index[(pub == pub_c3) & (hiv_all.reindex(pub.index) == 1)]

def gmm_fit(mat, k=3, seed=SEED, std=True):
    Xs = StandardScaler().fit_transform(mat) if std else np.asarray(mat, float)
    pca = PCA(n_components=VAR_THRESH, svd_solver='full', random_state=SEED)
    pcs = pca.fit_transform(Xs); n = min(CLUST_DIMS, pcs.shape[1])
    g = GaussianMixture(n_components=k, covariance_type=COV, random_state=seed, n_init=N_INIT, max_iter=MAX_ITER).fit(pcs[:, :n])
    return pd.Series(g.predict(pcs[:, :n]), index=mat.index), pcs[:, :n], g, int(pca.n_components_), float(pca.explained_variance_ratio_[:n].sum())

def summarise(lab, name, extra=None):
    h = hiv_all.reindex(lab.index).fillna(0).astype(int).values
    codes = pd.factorize(lab.values)[0]; k = codes.max() + 1
    cnt = np.bincount(codes, minlength=k)
    prev = np.bincount(codes, weights=h, minlength=k) / cnt
    top = int(np.argmax(prev)); obs = float(prev[top])
    rng = np.random.default_rng(SEED); mx = np.empty(N_PERM)
    for i in range(N_PERM):
        hp = rng.permutation(h); mx[i] = (np.bincount(codes, weights=hp, minlength=k) / cnt).max()
    top_label = pd.unique(lab.values)[top] if False else lab.values[codes == top][0]
    members = lab.index[lab == top_label]
    inter = c3_pos.intersection(lab.index)
    common = lab.index.intersection(pub.index)
    r = dict(variant=name, n=int(len(lab)), k=int(k), sizes=cnt.tolist(), hiv_prev=[round(float(x), 4) for x in prev],
             hiv_enriched_prev=obs, hiv_enriched_n=int(cnt[top]), perm_max=float(mx.max()), perm_p=float((mx >= obs).mean()),
             c3_hivpos_retained=float(len(inter.intersection(members)) / max(len(inter), 1)), c3_hivpos_in_sample=int(len(inter)),
             ari_vs_published=float(adjusted_rand_score(pub.loc[common], lab.loc[common])))
    if extra: r.update(extra)
    print(json.dumps(r), flush=True)
    return r

def boot_ari(pcs, g, k, nb=N_BOOT):
    rng = np.random.default_rng(SEED); ref = g.predict(pcs); a = []
    for _ in range(nb):
        idx = rng.integers(0, len(pcs), len(pcs))
        gb = GaussianMixture(n_components=k, covariance_type=COV, random_state=int(rng.integers(0, 2**31)), n_init=3, max_iter=MAX_ITER).fit(pcs[idx])
        a.append(adjusted_rand_score(ref, gb.predict(pcs)))
    return float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

res = {'visits_per_participant': visits, 'n_availability_patterns': int(pattern.nunique()),
       'top_patterns': pattern.value_counts().head(10).to_dict()}
out = []
primary = raw.fillna(0.0)
lab0, pcs0, g0, nc0, ve0 = gmm_fit(primary)
out.append(summarise(lab0, 'primary_rebuild', dict(n_pcs_85=nc0, var_explained_10pcs=ve0)))

# 7b1 availability only
labA, *_ = gmm_fit(avail.loc[primary.index].astype(float), std=False)
out.append(summarise(labA, 'b1_availability_only', dict(ari_vs_primary_rebuild=float(adjusted_rand_score(lab0, labA.loc[lab0.index])))))
# 7b2 measured-only standardisation
z = (raw - raw.mean()) / raw.std(ddof=0)
labB, *_ = gmm_fit(z.fillna(0.0), std=False)
out.append(summarise(labB, 'b2_measured_only_standardised'))
# 7b3 centred within availability pattern
cen = raw.groupby(pattern).transform(lambda s: s - s.mean())
labC, *_ = gmm_fit(cen.fillna(0.0))
out.append(summarise(labC, 'b3_centred_within_availability_pattern'))
# 7b4 common panel
cp = ['hematocrit', 'hemoglobin', 'viral_load']
keep = avail.index[avail[cp].all(axis=1)]
sub = raw.loc[keep, [c for c in raw.columns if c.split('__')[0] in cp]]
labD, *_ = gmm_fit(sub.fillna(0.0))
out.append(summarise(labD, 'b4_common_panel_hct_hb_vl', dict(cohorts=int(study.reindex(keep).nunique()))))
# 7g cohort-centred
cc = raw.groupby(study.reindex(raw.index)).transform(lambda s: s - s.mean())
labE, *_ = gmm_fit(cc.fillna(0.0))
out.append(summarise(labE, 'g_cohort_centred'))
# 7f exclude 100% / <1% HIV prevalence cohorts
cprev = hiv_all.reindex(study.index).groupby(study).mean()
excl = cprev[(cprev >= 0.999) | (cprev < 0.01)].index.tolist()
keepF = primary.index[~study.reindex(primary.index).isin(excl)]
labF, *_ = gmm_fit(primary.loc[keepF])
out.append(summarise(labF, 'f_excluding_100pct_and_lt1pct_hiv_cohorts', dict(excluded_cohorts=excl, cohort_hiv_prev={k: round(float(v), 4) for k, v in cprev.items()})))
# 7c k comparison
kc = []
for k in [2, 3, 4, 5]:
    labk, pcsk, gk, _, _ = gmm_fit(primary, k=k)
    rs = np.random.default_rng(SEED).choice(len(pcsk), 3000, replace=False)
    bm, blo, bhi = boot_ari(pcsk, gk, k)
    kc.append(summarise(labk, f'c_k{k}', dict(bic=float(gk.bic(pcsk)), aic=float(gk.aic(pcsk)),
        silhouette_3000=float(silhouette_score(pcsk[rs], labk.values[rs])), boot_ari_mean=bm, boot_ari_lo=blo, boot_ari_hi=bhi)))
res['variants'] = out; res['k_comparison'] = kc
res['run'] = dict(n_perm=N_PERM, n_boot=N_BOOT, seed=SEED, seconds=time.time() - t0, finished=time.ctime())
json.dump(res, open(OUT / 'r1_7bcfg_clustering.json', 'w'), indent=2, default=str)
print('done', time.time() - t0)
