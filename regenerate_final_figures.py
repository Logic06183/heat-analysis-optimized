#!/usr/bin/env python3
"""
regenerate_final_figures.py  —  Parker et al. 2026

Rebuilds the figure-script caches from a chosen Stage 3 run and re-runs the
cluster-dependent figure scripts.

WHICH RUN? (read this before using --stage3-dir)
-----------------------------------------------
Two Stage 3 runs exist under mcd_outputs_era5_land/ and they are NOT
interchangeable presentations of one analysis -- they are different analyses:

  stage3          13 biomarkers (R^2 >= 0 retention), 51 PCs, 85.1% variance,
                  PC1 9.6% / PC2 5.1%, PC1xPC2 = 14.7%, ARI 0.913,
                  cluster proportions 47.8 / 13.6 / 38.6
  stage3_r2_005   10 biomarkers (R^2 >= 0.05 retention), 43 PCs, 85.6% variance,
                  PC1 8.0% / PC2 6.2%, PC1xPC2 = 14.3%, ARI 0.985,
                  cluster proportions 47.1 / 13.4 / 39.5

The default is **stage3_r2_005**: the figures are to show the ten biomarkers
that pass through to clustering (decision of 2026-07-31).

OUTSTANDING: as of that date the manuscript (v51) and appendix (v34) still
describe **stage3** throughout -- "Thirteen of 24 biomarkers", "51 components
(85.1% variance)", "14.7% of the variance", ARI 0.913, 47.8/38.6 -- and contain
none of the stage3_r2_005 numbers. The figures this script produces are
therefore ahead of the prose until those documents are updated.

Usage (from the repo root, analysis env active):
    python regenerate_final_figures.py --dry-run
    python regenerate_final_figures.py                        # 10-biomarker run (default)
    python regenerate_final_figures.py --stage3-dir stage3    # the 13-biomarker run
    python regenerate_final_figures.py --no-run               # caches only
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent
FIG  = REPO / "publication_figures_lancet"
OUTPUT_ROOT = REPO / "mcd_outputs_era5_land"   # RP2_PRIMARY_EXPOSURE=era5_land

DEFAULT_STAGE3 = "stage3_r2_005"

# figS7 is intentionally excluded: it is the model-SELECTION panel, whose ARI
# values are measured across candidate configurations. It is not a view of any
# single run and must not be patched row-by-row.
RERUN = [
    "fig02_cluster_profiles.py",
    "fig03_cluster_biomarker_signatures.py",
    "figS1_pca_diagnostic.py",
    "figS2_lag_signatures_all_biomarkers.py",
    "figS4_interaction_heatmap.py",
    "fig01_lag_signatures_and_r2.py",
]

SILH_SAMPLE_N = 3000
SILH_SEED = 42          # mcd_pipeline.config.MASTER_SEED


def log(msg): print(f"[regen] {msg}")


def _subprocess_env(stage3: str) -> dict:
    """Environment the figure scripts need to resolve the intended run.

    RP2_PRIMARY_EXPOSURE selects config.OUTPUT_ROOT and defaults to "era5"
    (-> mcd_outputs/); the results live under "era5_land". RP2_STAGE3_DIR
    selects the Stage 3 subdirectory. PYTHONPATH is required because Python
    puts the *script's* directory on sys.path[0], not cwd, so the repo-root
    `mcd_pipeline` package is otherwise invisible.
    """
    env = os.environ.copy()
    env["RP2_PRIMARY_EXPOSURE"] = "era5_land"
    env["RP2_STAGE3_DIR"] = stage3
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.setdefault("MPLBACKEND", "Agg")
    return env


def _read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def _pc_columns(df: pd.DataFrame):
    """PC column names in order (handles 'PC1..', 'pc_1', or numeric names)."""
    import re
    cands = [c for c in df.columns if re.fullmatch(r"(?i)pc[_ ]?\d+", str(c))]
    if cands:
        return sorted(cands, key=lambda c: int(re.findall(r"\d+", c)[0]))
    num = [c for c in df.columns if str(c).isdigit()]
    return sorted(num, key=lambda c: int(c))


def canonical_remap(stage_dir: Path) -> dict[int, int]:
    """Raw GMM cluster id -> the manuscript's canonical id.

    Mirrors _data._canonical_cluster_remap exactly. GMM numbers clusters by
    initialisation order, so raw ids are not stable across runs; the figures
    address clusters by canonical id (0 younger marginalised, 1 older employed,
    2 high HIV burden). The caches this script writes are consumed alongside
    _data.py's canonical ids, so they must be remapped the same way -- under
    stage3 the mapping is NOT the identity (raw 2 is canonical 0).
    """
    by_cluster = json.loads(
        (stage_dir / "cluster_characteristics_with_ci.json").read_text()
    )["by_cluster"]
    metrics = {
        int(cid): {r["metric"]: float(r.get("point_estimate", float("nan"))) for r in rows}
        for cid, rows in by_cluster.items()
    }
    canon_2 = max(metrics, key=lambda i: metrics[i].get("pct_hiv_positive", -1))
    canon_1 = max(metrics, key=lambda i: metrics[i].get("mean_age_years", -1))
    canon_0 = next(i for i in metrics if i not in {canon_1, canon_2})
    return {canon_0: 0, canon_1: 1, canon_2: 2}


def rebuild_caches(stage_dir: Path, dry: bool):
    pc = _read_csv(stage_dir / "pc_scores.csv")
    lab = _read_csv(stage_dir / "cluster_labels.csv")
    id_col = next((c for c in ("patient_id","patient","id","pid") if c in pc.columns), None)
    lab_id = next((c for c in ("patient_id","patient","id","pid") if c in lab.columns), None)
    clus_col = next((c for c in ("cluster","label","cluster_id","gmm_cluster") if c in lab.columns), None)
    if clus_col is None:
        others = [c for c in lab.columns if c != lab_id]
        clus_col = others[0] if others else None
    if id_col is None or lab_id is None or clus_col is None:
        raise RuntimeError(f"Could not resolve id/cluster columns. "
                           f"pc cols={list(pc.columns)}  lab cols={list(lab.columns)}")
    pcs = _pc_columns(pc)
    if len(pcs) < 10:
        raise RuntimeError(f"Expected >=10 PC columns, found {pcs}")

    merged = pc.merge(lab[[lab_id, clus_col]], left_on=id_col, right_on=lab_id, how="inner")
    remap = canonical_remap(stage_dir)
    merged[clus_col] = merged[clus_col].map(remap)
    if merged[clus_col].isna().any():
        raise RuntimeError(f"Cluster remap left NaNs; remap={remap}")
    sizes = merged[clus_col].value_counts().sort_index().to_dict()
    log(f"merged {len(merged)} patients; canonical remap {remap}; sizes {sizes}")

    def write_cache(name, k):
        out = merged[[id_col] + pcs[:k] + [clus_col]].copy()
        out = out.rename(columns={id_col: "patient_id", clus_col: "cluster"})
        out = out.rename(columns={pcs[i]: f"PC{i+1}" for i in range(k)})
        dest = FIG / name
        if dry:
            log(f"DRY would write {dest} ({len(out)} rows, cols={list(out.columns)})")
        else:
            out.to_csv(dest, index=False)
            log(f"wrote {dest} ({len(out)} rows)")

    write_cache("_pc_cluster.csv", 3)
    write_cache("_pc10_cluster.csv", 10)
    rebuild_silhouette_sample(merged, pcs, clus_col, dry)


def rebuild_silhouette_sample(merged, pcs, clus_col, dry: bool):
    """Rebuild _silh_sample.npy, the (n,2) [cluster, silhouette] cache for figS1 panel c.

    Nothing in the repo produces this file, so it survives reruns untouched and
    silently keeps whatever run's geometry it was first built from. n and seed
    reproduce the sampling the figure's own axis label documents.
    """
    import numpy as np
    from sklearn.metrics import silhouette_samples

    dest = FIG / "_silh_sample.npy"
    X_all = merged[pcs[:10]].to_numpy(dtype=float)
    y_all = merged[clus_col].to_numpy()
    n = min(SILH_SAMPLE_N, len(merged))
    idx = np.random.default_rng(SILH_SEED).choice(len(merged), size=n, replace=False)
    X, y = X_all[idx], y_all[idx]
    sil = silhouette_samples(X, y, metric="euclidean")
    out = np.column_stack([y.astype(float), sil])
    meds = {int(c): round(float(np.median(sil[y == c])), 3) for c in np.unique(y)}
    if dry:
        log(f"DRY would write {dest} (n={n}, per-cluster medians={meds})")
    else:
        np.save(dest, out)
        log(f"wrote {dest} (n={n}, per-cluster medians={meds})")


def run_scripts(stage3: str, dry: bool):
    env = _subprocess_env(stage3)
    for name in RERUN:
        p = FIG / name
        if not p.exists():
            log(f"skip (missing): {name}"); continue
        if dry:
            log(f"DRY would run: python {p.name}"); continue
        log(f"running {name} ...")
        r = subprocess.run([sys.executable, str(p)], cwd=str(FIG), env=env)
        log(f"  {'OK' if r.returncode==0 else 'FAILED (rc=%d)'%r.returncode}: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3-dir", default=DEFAULT_STAGE3,
                    help=f"Stage 3 subdirectory under {OUTPUT_ROOT.name}/ "
                         f"(default {DEFAULT_STAGE3}, the 10-biomarker run)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-run", action="store_true", help="rebuild caches only")
    a = ap.parse_args()

    stage_dir = OUTPUT_ROOT / a.stage3_dir
    for p in (FIG, stage_dir):
        if not p.exists():
            sys.exit(f"[regen] path not found: {p}")
    log(f"repo={REPO}")
    log(f"stage3 run = {stage_dir}")
    if a.stage3_dir == "stage3_r2_005":
        log("NOTE: manuscript v51 / appendix v34 still describe the 13-biomarker "
            "stage3 run; their prose and captions have to follow these figures.")

    rebuild_caches(stage_dir, a.dry_run)
    if a.no_run:
        log("caches rebuilt; skipping re-run (--no-run)."); return
    run_scripts(a.stage3_dir, a.dry_run)
    log("done.")


if __name__ == "__main__":
    main()
