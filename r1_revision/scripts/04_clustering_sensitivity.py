"""04. What do the participant clusters reflect? (appendix SA13 and SA14)

Question    Reviewer 3 and the statistical reviewer asked whether the
            HIV-enriched cluster survives when missing biomarkers are handled
            differently, how many clusters the data support, and (reviewer 5)
            what happens without the all-HIV and no-HIV cohorts.
Data in     Published Stage 1 SHAP values and Stage 3 cluster labels.
Steps       1. Rebuild the person-level temperature SHAP profile and confirm
               it reproduces the published clusters (adjusted Rand index 1).
            2. Re-cluster under each variant below.
            3. For each variant report the cluster sizes, HIV prevalence per
               cluster, a permutation p value for the highest prevalence
               (HIV labels shuffled across all participants, 10,000 times),
               and agreement with the published clusters.
            4. Compare two to five clusters on fit (BIC, AIC), separation
               (silhouette) and stability (bootstrap adjusted Rand index).
File out    r1_7bcfg_clustering.json
Reported    SA13 (number of clusters) and SA14 (cohort design and missingness).

Variants
  primary_rebuild   as published: an unmeasured biomarker counts as zero
  b1  cluster only on WHICH biomarkers were measured (0/1), no SHAP values
  b2  standardise each SHAP feature on the people who had it measured, so an
      unmeasured biomarker counts as an average response
  b3  centre SHAP features within each availability pattern
  b4  only people with haematocrit, haemoglobin and HIV viral load measured,
      clustered on those three
  g   centre SHAP features within cohort
  f   drop cohorts where everyone or almost no one (<1%) is living with HIV
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from r1_common import (
    N_PERMUTATIONS,
    RETAINED,
    SEED,
    cluster_profiles,
    cohort_of_each_participant,
    highest_prevalence_permutation_test,
    load_visits,
    person_level_temperature_shap,
    published_clusters,
    write_json,
)

N_BOOT = int(os.environ.get("R1_NBOOT_CLUST", "200"))
SILHOUETTE_SAMPLE = 3000
COMMON_PANEL = ["hematocrit", "hemoglobin", "viral_load"]

started = time.time()
visits, biomarkers = load_visits()
cohort = cohort_of_each_participant(visits)
published_labels, hiv_positive, cluster3 = published_clusters()

# HIV-positive members of the published Cluster 3. We track where they land.
in_cluster3 = published_labels == cluster3
is_positive = hiv_positive.reindex(published_labels.index) == 1
cluster3_hiv_positive = published_labels.index[in_cluster3 & is_positive]


# ---- Step 1: person-level profile -----------------------------------------

profiles_by_biomarker = []
visits_per_participant = {}
for name in RETAINED:
    per_person, n_visits = person_level_temperature_shap(visits, biomarkers, name)
    profiles_by_biomarker.append(per_person)
    visits_per_participant[name] = n_visits.describe().to_dict()

# One row per participant. NaN means that biomarker was never measured for them.
profile = pd.concat(profiles_by_biomarker, axis=1, join="outer")


def columns_of(name):
    return [c for c in profile.columns if c.startswith(name + "__")]


measured = pd.DataFrame(
    {name: profile[columns_of(name)].notna().any(axis=1).astype(int) for name in RETAINED}
)
availability_pattern = measured.astype(str).agg("".join, axis=1)


# ---- Describing one clustering --------------------------------------------


def describe(labels, variant, extra=None):
    """Sizes, HIV prevalence by cluster, permutation test and agreement with the published clusters."""
    hiv = hiv_positive.reindex(labels.index).fillna(0).astype(int).values
    codes = pd.factorize(labels.values)[0]
    sizes = np.bincount(codes)
    prevalence = np.bincount(codes, weights=hiv) / sizes
    enriched = int(np.argmax(prevalence))
    enriched_members = labels.index[codes == enriched]

    observed, p_value, largest_shuffled = highest_prevalence_permutation_test(
        labels.values, hiv, N_PERMUTATIONS, np.random.default_rng(SEED)
    )

    tracked = cluster3_hiv_positive.intersection(labels.index)
    still_together = len(tracked.intersection(enriched_members)) / max(len(tracked), 1)
    shared = labels.index.intersection(published_labels.index)

    result = {
        "variant": variant,
        "n": int(len(labels)),
        "k": int(len(sizes)),
        "sizes": sizes.tolist(),
        "hiv_prev": [round(float(p), 4) for p in prevalence],
        "hiv_enriched_prev": observed,
        "hiv_enriched_n": int(sizes[enriched]),
        "perm_max": largest_shuffled,
        "perm_p": p_value,
        "c3_hivpos_retained": float(still_together),
        "c3_hivpos_in_sample": int(len(tracked)),
        "ari_vs_published": float(adjusted_rand_score(published_labels.loc[shared], labels.loc[shared])),
    }
    if extra:
        result.update(extra)
    print(result, flush=True)
    return result


def bootstrap_stability(components, mixture, k):
    """Refit on resampled participants and compare with the original labels (adjusted Rand index).

    The standardisation and PCA are not refitted, only the mixture, so this measures the stability
    of the grouping given the components.
    """
    rng = np.random.default_rng(SEED)
    reference = mixture.predict(components)
    agreement = []
    for _ in range(N_BOOT):
        rows = rng.integers(0, len(components), len(components))
        refit = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=int(rng.integers(0, 2**31)),
            n_init=3,
            max_iter=300,
        ).fit(components[rows])
        agreement.append(adjusted_rand_score(reference, refit.predict(components)))
    agreement = np.array(agreement)
    return {
        "boot_ari_mean": float(agreement.mean()),
        "boot_ari_median": float(np.median(agreement)),
        "boot_ari_lo": float(np.percentile(agreement, 2.5)),
        "boot_ari_hi": float(np.percentile(agreement, 97.5)),
        # A mean can sit below the 2.5th percentile when a few refits land on a different solution.
        "boot_ari_share_below_0_9": float((agreement < 0.9).mean()),
    }


# ---- Step 2 and 3: the variants -------------------------------------------

variants = []

unmeasured_as_zero = profile.fillna(0.0)
primary = cluster_profiles(unmeasured_as_zero)
variants.append(
    describe(
        primary["labels"],
        "primary_rebuild",
        {
            "n_pcs_85": primary["n_components_at_85pct"],
            "var_explained_10pcs": primary["variance_in_clustered_components"],
        },
    )
)

# b1: availability only
availability_only = cluster_profiles(measured.loc[unmeasured_as_zero.index].astype(float), standardise=False)
agreement_with_primary = adjusted_rand_score(
    primary["labels"], availability_only["labels"].loc[primary["labels"].index]
)
variants.append(
    describe(
        availability_only["labels"],
        "b1_availability_only",
        {
            "ari_vs_primary_rebuild": float(agreement_with_primary),
        },
    )
)

# b2: standardise on measured participants only (pandas skips NaN), then fill with 0 = the measured mean
standardised_on_measured = (profile - profile.mean()) / profile.std(ddof=0)
result = cluster_profiles(standardised_on_measured.fillna(0.0), standardise=False)
variants.append(describe(result["labels"], "b2_measured_only_standardised"))

# b3: centre within availability pattern
centred_within_pattern = profile.groupby(availability_pattern).transform(
    lambda column: column - column.mean()
)
result = cluster_profiles(centred_within_pattern.fillna(0.0))
variants.append(describe(result["labels"], "b3_centred_within_availability_pattern"))

# b4: common panel
has_common_panel = measured.index[measured[COMMON_PANEL].all(axis=1)]
common_panel_columns = [c for c in profile.columns if c.split("__")[0] in COMMON_PANEL]
result = cluster_profiles(profile.loc[has_common_panel, common_panel_columns].fillna(0.0))
variants.append(
    describe(
        result["labels"],
        "b4_common_panel_hct_hb_vl",
        {
            "cohorts": int(cohort.reindex(has_common_panel).nunique()),
        },
    )
)

# g: centre within cohort
centred_within_cohort = profile.groupby(cohort.reindex(profile.index)).transform(
    lambda column: column - column.mean()
)
result = cluster_profiles(centred_within_cohort.fillna(0.0))
variants.append(describe(result["labels"], "g_cohort_centred"))

# f: drop cohorts where HIV prevalence is 100% or under 1%
cohort_hiv_prevalence = hiv_positive.reindex(cohort.index).groupby(cohort).mean()
all_or_none = (cohort_hiv_prevalence >= 0.999) | (cohort_hiv_prevalence < 0.01)
excluded_cohorts = cohort_hiv_prevalence[all_or_none].index.tolist()
in_mixed_cohort = ~cohort.reindex(unmeasured_as_zero.index).isin(excluded_cohorts)
result = cluster_profiles(unmeasured_as_zero.loc[in_mixed_cohort.values])
variants.append(
    describe(
        result["labels"],
        "f_excluding_100pct_and_lt1pct_hiv_cohorts",
        {
            "excluded_cohorts": excluded_cohorts,
            "cohort_hiv_prev": {name: round(float(p), 4) for name, p in cohort_hiv_prevalence.items()},
        },
    )
)


# ---- Step 4: two to five clusters -----------------------------------------

k_comparison = []
for k in [2, 3, 4, 5]:
    fit = cluster_profiles(unmeasured_as_zero, k=k)
    components = fit["components"]
    sample = np.random.default_rng(SEED).choice(len(components), SILHOUETTE_SAMPLE, replace=False)
    stability = bootstrap_stability(components, fit["mixture"], k)
    k_comparison.append(
        describe(
            fit["labels"],
            f"c_k{k}",
            {
                "bic": float(fit["mixture"].bic(components)),
                "aic": float(fit["mixture"].aic(components)),
                "silhouette_3000": float(silhouette_score(components[sample], fit["labels"].values[sample])),
                **stability,
            },
        )
    )


write_json(
    {
        "visits_per_participant": visits_per_participant,
        "n_availability_patterns": int(availability_pattern.nunique()),
        "pattern_column_order": RETAINED,  # each pattern is one 0/1 digit per biomarker, in this order
        "top_patterns": availability_pattern.value_counts().head(10).to_dict(),
        "n_hiv_status_unknown_counted_as_negative": int(hiv_positive.reindex(profile.index).isna().sum()),
        "variants": variants,
        "k_comparison": k_comparison,
        "run": {
            "n_perm": N_PERMUTATIONS,
            "n_boot": N_BOOT,
            "seed": SEED,
            "seconds": time.time() - started,
            "finished": time.ctime(),
        },
    },
    "r1_7bcfg_clustering.json",
)
