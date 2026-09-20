#!/usr/bin/env python3
"""Re-run the Figure S7 clustering-method selection panel on the ten-biomarker run.

The published panel mixed sources: its four comparison rows came from the original
ERA5 refinement (n=9,736) while the primary row carried the 13-biomarker ERA5-Land
stability (ARI 0.913). This re-runs all five candidates on one profile --
mcd_outputs_era5_land/stage3_r2_005 -- so the panel compares like with like.

Methodology is copied from mcd_pipeline/stage3_clustering/stage3_refinement.py
(_bootstrap_ari, _kmeans_fit_fn, _gmm_fit_fn) so the numbers are produced the same
way as the ones they replace.

The "all components" baseline tracks the run: it was PC56 under ERA5 and is PC43
here, being every component retained at the 85% variance threshold.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

REPO = Path(__file__).resolve().parent
STAGE = REPO / "mcd_outputs_era5_land" / "stage3_r2_005"
OUT = REPO / "mcd_outputs_era5_land" / "stage3_refinement_r2_005"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42            # mcd_pipeline.config.MASTER_SEED
N_ITER = 500         # figure caption: "500-iteration bootstrap stability"
ARI_THRESHOLD = 0.80


def log(m): print(f"[figS7] {m}", flush=True)


def bootstrap_ari(X, labels_ref, fit_fn, n_iterations=N_ITER, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(X)
    aris = np.empty(n_iterations)
    t0 = time.time()
    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        boot_labels = fit_fn(X[idx], X, rng)
        aris[i] = adjusted_rand_score(labels_ref, boot_labels)
        if (i + 1) % 100 == 0:
            log(f"    {i+1}/{n_iterations} iters, running mean ARI={aris[:i+1].mean():.3f} "
                f"({time.time()-t0:.0f}s)")
    return {
        "mean_ari": float(aris.mean()),
        "std_ari": float(aris.std()),
        "ci_lower": float(np.percentile(aris, 2.5)),
        "ci_upper": float(np.percentile(aris, 97.5)),
    }, aris


def kmeans_fit_fn(k):
    def fn(X_boot, X_full, rng):
        km = KMeans(n_clusters=k, random_state=int(rng.integers(0, 2**31)), n_init=10)
        km.fit(X_boot)
        return km.predict(X_full)
    return fn


def gmm_fit_fn(k):
    def fn(X_boot, X_full, rng):
        g = GaussianMixture(n_components=k, covariance_type="full",
                            random_state=int(rng.integers(0, 2**31)),
                            n_init=3, max_iter=300)
        g.fit(X_boot)
        return g.predict(X_full)
    return fn


def main():
    pcs = pd.read_csv(STAGE / "pc_scores.csv")
    pc_cols = [c for c in pcs.columns if c.upper().startswith("PC")]
    pc_cols = sorted(pc_cols, key=lambda c: int(c[2:]))
    X_all = pcs[pc_cols].to_numpy(dtype=float)
    ev = json.load(open(STAGE / "pca_summary.json"))["explained_variance_ratio"]
    n_total = len(pc_cols)
    log(f"pc_scores: {X_all.shape[0]} patients x {n_total} components")

    CONFIGS = [
        {"name": f"PC{n_total}_kmeans_k4", "n_dims": n_total, "method": "kmeans", "k": 4,
         "is_original": True},
        {"name": "PC5_gmm_k3",    "n_dims": 5,  "method": "gmm",    "k": 3},
        {"name": "PC5_kmeans_k4", "n_dims": 5,  "method": "kmeans", "k": 4},
        {"name": "PC5_kmeans_k3", "n_dims": 5,  "method": "kmeans", "k": 3},
        {"name": "PC10_gmm_k3",   "n_dims": 10, "method": "gmm",    "k": 3,
         "is_primary": True},
    ]

    results = []
    for cfg in CONFIGS:
        d, k, meth = cfg["n_dims"], cfg["k"], cfg["method"]
        X = X_all[:, :d]
        log(f"--- {cfg['name']}: {meth} k={k} on {d} PCs ---")
        if meth == "kmeans":
            ref = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(X)
            fit_fn = kmeans_fit_fn(k)
        else:
            ref = GaussianMixture(n_components=k, covariance_type="full",
                                  random_state=SEED, n_init=3,
                                  max_iter=300).fit(X).predict(X)
            fit_fn = gmm_fit_fn(k)
        sil = float(silhouette_score(X, ref))
        sizes = [int((ref == c).sum()) for c in range(k)]
        log(f"    silhouette={sil:.4f} sizes={sizes}")
        # Resume support: a completed bootstrap is cached as ari_<name>.npy, so an
        # interrupted run does not repeat the expensive part.
        cache = OUT / f"ari_{cfg['name']}.npy"
        if cache.exists():
            aris = np.load(cache)
            if len(aris) == N_ITER:
                stab = {"mean_ari": float(aris.mean()), "std_ari": float(aris.std()),
                        "ci_lower": float(np.percentile(aris, 2.5)),
                        "ci_upper": float(np.percentile(aris, 97.5))}
                log(f"    reusing cached bootstrap ({N_ITER} iters)")
            else:
                stab, aris = bootstrap_ari(X, ref, fit_fn); np.save(cache, aris)
        else:
            stab, aris = bootstrap_ari(X, ref, fit_fn)
            np.save(cache, aris)
        rec = {**cfg,
               "cum_var": float(np.sum(ev[:d])),
               "silhouette": sil,
               "sizes": sizes,
               **stab,
               "meets_stability_threshold": stab["mean_ari"] >= ARI_THRESHOLD}
        log(f"    ARI={stab['mean_ari']:.4f} +/- {stab['std_ari']:.4f} "
            f"[{stab['ci_lower']:.3f}, {stab['ci_upper']:.3f}]")
        results.append(rec)

    best = max(results, key=lambda r: r["mean_ari"])
    summary = {
        "source_run": str(STAGE.relative_to(REPO)),
        "n_patients": int(X_all.shape[0]),
        "n_biomarkers": 10,
        "n_components_total": n_total,
        "n_bootstrap_iterations": N_ITER,
        "seed": SEED,
        "ari_threshold": ARI_THRESHOLD,
        "best_approach": best["name"],
        "all_results": results,
    }
    (OUT / "refinement_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote {OUT/'refinement_summary.json'}")
    log(f"best = {best['name']} (ARI {best['mean_ari']:.4f})")


if __name__ == "__main__":
    sys.exit(main())
