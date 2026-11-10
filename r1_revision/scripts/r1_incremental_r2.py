"""R1 analysis 7a + 7d + 7h(n per model).

7a  Incremental cross-validated R2 of temperature: for every Stage 1 biomarker,
    the primary model (temperature lags + DTR + covariates) is compared with an
    otherwise identical covariate-only model on the same participant-grouped
    folds (GroupKFold, 5 folds, primary XGBoost settings). Uncertainty: 1,000
    participant-level bootstrap resamples of the pooled out-of-fold predictions.
7d  HIV viral load as a binary outcome: viraemic (>=1000 copies/mL, WHO
    threshold for virological failure) vs not, same features, AUC full vs
    covariate-only, same folds.
7h  n participants and observations per biomarker model.

Run from the repo root with RP2_PRIMARY_EXPOSURE=era5_land.
Outputs (aggregate only, no participant rows) go to R1_revision_work/outputs.
"""
import json, os, sys, time, logging
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, roc_auc_score

os.environ.setdefault("RP2_PRIMARY_EXPOSURE", "era5_land")
REPO = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(REPO))
from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import MIN_SAMPLE_SIZE, get_available_biomarkers
from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
from mcd_pipeline.stage0_data_prep.feature_engineering import engineer_features
from mcd_pipeline.stage1_lag_profiling.xgboost_trainer import prepare_features_and_target

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)
logging.basicConfig(level=logging.WARNING)
N_BOOT = int(os.environ.get("R1_NBOOT", "1000"))
rng = np.random.default_rng(config.BOOTSTRAP_RANDOM_SEED)
TEMP_COLS = [f"temp_lag_{d}d" for d in config.LAG_DAYS] + [config.DTR_COLUMN]

def oof(X, y, groups, clf=False):
    gkf = GroupKFold(n_splits=config.CV_FOLDS)
    pred = np.full(len(y), np.nan); folds = np.full(len(y), -1); fold_scores = []
    for k, (tr, va) in enumerate(gkf.split(X, y, groups)):
        if clf:
            m = xgb.XGBClassifier(**config.XGBOOST_PARAMS, eval_metric="logloss")
            m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
            p = m.predict_proba(X.iloc[va])[:, 1]
            fold_scores.append(roc_auc_score(y.iloc[va], p) if y.iloc[va].nunique() > 1 else np.nan)
        else:
            m = xgb.XGBRegressor(**config.XGBOOST_PARAMS)
            m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
            p = m.predict(X.iloc[va])
            fold_scores.append(r2_score(y.iloc[va], p))
        pred[va] = p; folds[va] = k
    return pred, folds, fold_scores

def boot_delta(y, pf, pr, groups, score):
    g = pd.Series(groups.values); uniq = g.unique()
    idx_by = g.groupby(g).indices
    out = []
    yv = np.asarray(y)
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by[p] for p in pick])
        try:
            out.append(score(yv[ii], pf[ii]) - score(yv[ii], pr[ii]))
        except ValueError:
            pass
    return np.percentile(out, [2.5, 97.5]).tolist()

t0 = time.time()
df = build_analysis_dataset()
specs = get_available_biomarkers()
bio_cols = [b.column for b in specs.values() if b.column and b.column in df.columns]
df = engineer_features(df, bio_cols)
pub = {}
for b in specs.values():
    p = config.OUTPUT_ROOT / "stage1" / str(b.column) / "cv_metrics.json"
    if b.column and p.exists():
        pub[b.column] = json.load(open(p))["mean_r2"]

rows = []
for name, b in specs.items():
    if not b.column or b.column not in df.columns or df[b.column].notna().sum() < MIN_SAMPLE_SIZE:
        continue
    X, y, groups = prepare_features_and_target(df, b)
    tcols = [c for c in TEMP_COLS if c in X.columns]
    pf, folds, sf = oof(X, y, groups)
    pr, _, sr = oof(X.drop(columns=tcols), y, groups)
    ci = boot_delta(y, pf, pr, groups, r2_score)
    rows.append(dict(biomarker=b.column, n_obs=len(y), n_participants=groups.nunique(),
        n_temp_features=len(tcols), r2_published=pub.get(b.column), r2_full=float(np.mean(sf)),
        r2_covariates_only=float(np.mean(sr)), delta_r2=float(np.mean(np.array(sf) - np.array(sr))),
        delta_r2_lo=ci[0], delta_r2_hi=ci[1],
        folds_temperature_better=int(sum(a > c for a, c in zip(sf, sr))),
        fold_delta=";".join(f"{a-c:.4f}" for a, c in zip(sf, sr))))
    pd.DataFrame(rows).to_csv(OUT / "r1_7a_incremental_r2.csv", index=False)
    print(f"{b.column}: full {np.mean(sf):.4f} (pub {pub.get(b.column)}) cov {np.mean(sr):.4f} d {rows[-1]['delta_r2']:.4f} [{ci[0]:.4f},{ci[1]:.4f}] {time.time()-t0:.0f}s", flush=True)

# 7d viral suppression
vl = specs["viral_load"]
X, y, groups = prepare_features_and_target(df, vl)
raw = df.loc[df[vl.column].notna(), vl.column].astype(float).values
yb = pd.Series((raw >= 1000).astype(int), index=y.index)
tcols = [c for c in TEMP_COLS if c in X.columns]
pf, _, sf = oof(X, yb, groups, clf=True)
pr, _, sr = oof(X.drop(columns=tcols), yb, groups, clf=True)
ci = boot_delta(yb, pf, pr, groups, roc_auc_score)
json.dump(dict(n_obs=int(len(yb)), n_participants=int(groups.nunique()), prevalence_viraemic=float(yb.mean()),
    auc_full=float(np.nanmean(sf)), auc_covariates_only=float(np.nanmean(sr)),
    delta_auc=float(np.nanmean(np.array(sf) - np.array(sr))), delta_auc_ci=ci,
    fold_auc_full=sf, fold_auc_cov=sr), open(OUT / "r1_7d_viral_suppression.json", "w"), indent=2)
print("7d done", time.time() - t0, flush=True)
json.dump(dict(xgboost=xgb.__version__, pandas=pd.__version__, numpy=np.__version__, n_boot=N_BOOT,
    seed=config.BOOTSTRAP_RANDOM_SEED, exposure=config.PRIMARY_EXPOSURE_SOURCE, finished=time.ctime()),
    open(OUT / "r1_7a_7d_run_info.json", "w"), indent=2)
