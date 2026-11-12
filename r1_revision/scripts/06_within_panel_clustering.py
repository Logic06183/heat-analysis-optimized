"""06. Clustering inside one cohort or one biomarker panel (appendix SA14)

Question    When everyone has the same biomarkers measured, do clusters still
            separate people by HIV status or viraemia?
Data in     Published Stage 1 SHAP values; the visit-level analysis table.
Steps       A. Two cohorts that enrolled people with and without HIV, each
               clustered on its own biomarker panel (2 or 3 clusters, chosen
               by BIC). Permutation test for the highest HIV prevalence.
            B. People living with HIV who had haematocrit, haemoglobin and HIV
               viral load measured, clustered on those three (2 to 4 clusters).
               Permutation tests for the share ever viraemic and the share female.
File out    r1_within_panel.json
Reported    SA14: HPTN 075 (SCHARP_004) 32%, 24%, 4% HIV-positive, p=0.03;
            Women's Body Study (DPHRU_013); viraemia across groups p=0.86.
"""

import numpy as np
import pandas as pd

from r1_common import (
    N_PERMUTATIONS,
    VIRAEMIA_THRESHOLD,
    PARTICIPANT,
    SEED,
    cluster_profiles,
    cohort_of_each_participant,
    highest_prevalence_permutation_test,
    load_visits,
    person_level_temperature_shap,
    published_clusters,
    write_json,
)

MIXED_COHORT_PANELS = {
    "JHB_SCHARP_004": ["creatinine", "hemoglobin", "hematocrit"],
    "JHB_DPHRU_013": ["diastolic_bp", "heart_rate", "waist_hip_ratio", "body_fat_percent"],
}
HIV_PANEL = ["hematocrit", "hemoglobin", "viral_load"]

# One generator for all four permutation tests, drawn in the order they appear below.
rng = np.random.default_rng(SEED)

visits, biomarkers = load_visits()
cohort = cohort_of_each_participant(visits)
_, hiv_positive, _ = published_clusters()
hiv_positive = hiv_positive.astype(bool)

by_participant = visits.groupby(PARTICIPANT)
sex = by_participant["sex"].agg(lambda s: s.mode().iat[0] if s.notna().any() else np.nan).astype(str)
mean_age = by_participant["age_years"].mean()
highest_viral_load = by_participant["viral_load"].max()


def profile_for(panel):
    """Participants with every biomarker in the panel measured, and their SHAP profile."""
    parts = [person_level_temperature_shap(visits, biomarkers, name)[0] for name in panel]
    return pd.concat(parts, axis=1, join="inner")


def cluster_with_best_k(profile, candidates):
    """Cluster with each candidate number of clusters and keep the lowest BIC."""
    # max_iter=100 is scikit-learn's default, used when these small panels were first clustered.
    fits = {k: cluster_profiles(profile, k=k, max_iter=100) for k in candidates}
    best_k = min(candidates, key=lambda k: fits[k]["mixture"].bic(fits[k]["components"]))
    return fits[best_k]["labels"], best_k


results = {}

# ---- A. Mixed-HIV cohorts, each on its own panel --------------------------

for cohort_name, panel in MIXED_COHORT_PANELS.items():
    profile = profile_for(panel)
    in_cohort = cohort.index[cohort == cohort_name]
    profile = profile.loc[profile.index.intersection(in_cohort)]

    labels, k = cluster_with_best_k(profile, candidates=(2, 3))
    hiv = hiv_positive.reindex(profile.index).fillna(False).astype(int)
    highest, p_value, _ = highest_prevalence_permutation_test(labels.values, hiv.values, N_PERMUTATIONS, rng)

    results[cohort_name] = {
        "n": int(len(profile)),
        "k": int(k),
        # sizes and prevalences are both in cluster-label order, so they line up
        "sizes": labels.value_counts().sort_index().tolist(),
        "hiv_prev_by_cluster": hiv.groupby(labels).mean().round(3).tolist(),
        "overall_hiv": float(hiv.mean()),
        "max_prev": highest,
        "perm_p": p_value,
    }
    print(cohort_name, results[cohort_name], flush=True)

# ---- B. People living with HIV, common panel ------------------------------

profile = profile_for(HIV_PANEL)
living_with_hiv = hiv_positive.index[hiv_positive]
profile = profile.loc[profile.index.intersection(living_with_hiv)]

labels, k = cluster_with_best_k(profile, candidates=(2, 3, 4))
people = pd.DataFrame(
    {
        "cluster": labels,
        "viraemic_ever": (highest_viral_load.reindex(profile.index) >= VIRAEMIA_THRESHOLD).astype(int),
        "female": sex.reindex(profile.index).str.lower().str.startswith("f").astype(int),
        "age": mean_age.reindex(profile.index),
        "study": cohort.reindex(profile.index),
    }
)
summary = (
    people.groupby("cluster")
    .agg(
        n=("age", "size"),
        viraemic_ever=("viraemic_ever", "mean"),
        female=("female", "mean"),
        age=("age", "mean"),
    )
    .round(3)
)

viraemia_highest, viraemia_p, _ = highest_prevalence_permutation_test(
    labels.values, people["viraemic_ever"].values, N_PERMUTATIONS, rng
)
female_highest, female_p, _ = highest_prevalence_permutation_test(
    labels.values, people["female"].values, N_PERMUTATIONS, rng
)

results["hivpos_hct_hb_vl"] = {
    "n": int(len(profile)),
    "k": int(k),
    "clusters": summary.to_dict(orient="index"),
    "cohort_mix": pd.crosstab(people["cluster"], people["study"]).to_dict(),
    "viraemia_max": viraemia_highest,
    "viraemia_perm_p": viraemia_p,
    "female_max": female_highest,
    "female_perm_p": female_p,
}
print(results["hivpos_hct_hb_vl"], flush=True)

write_json(results, "r1_within_panel.json")
