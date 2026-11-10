# Review guide: from each reported number to the code

For a reviewer checking the revision analyses. Each block gives the claim as the paper makes it,
the script, and the field in `outputs/` that holds the number. Appendix numbers are those of the
revised appendix. Decimal points are written as full stops here; the paper uses the Lancet midline dot.

Read `scripts/r1_common.py` first (about 250 lines). It holds the five steps every analysis reuses:

1. `load_visits` builds the visit-level table with the main pipeline's own functions.
2. `cross_validate` fits the published XGBoost model in five folds grouped by participant.
3. `bootstrap_intervals` resamples participants to put an interval on one model's score and on the difference between two models.
4. `person_level_temperature_shap` and `cluster_profiles` rebuild the Stage 3 clustering.
5. `highest_prevalence_permutation_test` shuffles a label across participants.

## 1. Incremental R-squared (appendix SA12, table S10; Results)

Script `02_incremental_r2.py`, output `r1_7a_incremental_r2.csv`. One row per biomarker.
`delta_r2` is the increment, `delta_r2_lo` and `delta_r2_hi` its 95% interval, `r2_full` the
cross-validated R-squared with `r2_full_lo` and `r2_full_hi` its 95% interval, `n_obs` and
`n_participants` the sample. Both intervals come from the same 1,000 bootstrap resamples.

| Biomarker | Increment (95% interval) | Row `biomarker` |
|---|---|---|
| Creatinine | 0.072 (0.005 to 0.154) | `creatinine` |
| Body fat percentage | 0.045 (0.007 to 0.083) | `body_fat_percent` |
| Haematocrit | 0.042 (0.030 to 0.055) | `hematocrit` |
| HIV viral load | 0.025 (0.014 to 0.036) | `viral_load` |
| Haemoglobin | 0.020 (0.009 to 0.031) | `hemoglobin` |
| Heart rate | 0.017 (0.010 to 0.025) | `heart_rate` |
| Diastolic blood pressure | 0.015 (0.008 to 0.022) | `diastolic_bp` |
| CD4 count | 0.012 (-0.003 to 0.024) | `cd4_count` |
| LDL cholesterol | 0.012 (-0.016 to 0.039) | `ldl_cholesterol` |
| Waist-hip ratio | 0.009 (-0.039 to 0.053) | `waist_hip_ratio` |

"A clean rerun reproduced the published cross-validated R-squared for all 24 biomarkers":
compare columns `r2_full` and `r2_published` in the same file.

Viraemia (1,000 copies per mL or more): AUC 0.818 with temperature, 0.810 without, difference
0.008 (0.003 to 0.012). Output `r1_7d_viral_suppression.json`, fields `auc_full`,
`auc_covariates_only`, `delta_auc`, `delta_auc_ci`. 19.3% of 22,036 observations were viraemic
(`prevalence_viraemic`, `n_obs`).

Leave one cohort out: script `03_incremental_r2_leave_one_cohort_out.py`, output
`r1_stage1_loco.csv`, column `delta` by `biomarker` and `dropped`. The increment stays above zero
after every exclusion for eight biomarkers. The two exceptions are creatinine (-0.031 without
JHB_WRHI_001, the PEARLS cohort) and waist-hip ratio (-0.002; only two cohorts measured it).
Column `r2_full` in the same file shows that overall R-squared is less stable: after at least one
exclusion it falls below 0.05 for heart rate, diastolic blood pressure, LDL cholesterol, waist-hip
ratio and body fat percentage.

## 2. Number of clusters (appendix SA13, table S11)

Script `04_clustering_sensitivity.py`, output `r1_7bcfg_clustering.json`, list `k_comparison`
(entries `c_k2` to `c_k5`).

| Clusters | BIC (`bic`) | Silhouette (`silhouette_3000`) | Bootstrap stability (`boot_ari_mean`) | Highest HIV prevalence (`hiv_enriched_prev`) |
|---|---|---|---|---|
| 2 | 168,629 | 0.116 | 0.983 | 64.9% |
| 3 | 116,096 | 0.095 | 0.987 | 77.8% |
| 4 | 95,815 | 0.061 | 0.807 | 95.6% |
| 5 | 76,544 | 0.060 | 0.872 | 99.1% |

## 3. Cohort design and missing biomarkers (appendix SA14, table S12; Results)

Same script and file, list `variants`.

| Claim | Entry `variant` | Field |
|---|---|---|
| The rebuild reproduces the published clusters | `primary_rebuild` | `ari_vs_published` = 1.0; `n_pcs_85` = 43; `var_explained_10pcs` = 0.432 |
| Clustering on availability alone gives an HIV-enriched group with different members | `b1_availability_only` | `hiv_enriched_prev` 0.835, `ari_vs_published` 0.36, `c3_hivpos_retained` 0.20 |
| Same when standardising on measured participants only | `b2_measured_only_standardised` | 0.845, 0.39, 0.20 |
| Centring within availability pattern keeps Cluster 3 | `b3_centred_within_availability_pattern` | 0.781, `ari_vs_published` 0.96, `c3_hivpos_retained` 0.98 |
| Centring within cohort keeps Cluster 3 | `g_cohort_centred` | 0.778, 0.96, 0.98 |
| Common panel: everyone is HIV-positive, so not testable | `b4_common_panel_hct_hb_vl` | `n` 1,896, `hiv_enriched_prev` 1.0 |
| Without the all-HIV and no-HIV cohorts no HIV-enriched group forms | `f_excluding_100pct_and_lt1pct_hiv_cohorts` | `n` 2,213, `hiv_enriched_prev` 0.202, `perm_p` 0.82 |

Membership by cohort: script `05_availability_by_cohort.py`, output
`r1_availability_by_cohort.csv`. "In all six cohorts that measured haematocrit, 95% to 100% of
participants fell into Cluster 3": rows where column `hematocrit` is above zero, column
`cluster3_share` (0.954 to 1.000). Two rows in the same file bear on how to read this. JHB_SCHARP_006
measured creatinine only, enrolled no one with HIV, and 99.8% of its participants are in Cluster 3.
JHB_Aurum_009 enrolled only people living with HIV, did not measure haematocrit, and 21.9% are in Cluster 3.

Within one cohort: script `06_within_panel_clustering.py`, output `r1_within_panel.json`.
HPTN 075 (`JHB_SCHARP_004`): `hiv_prev_by_cluster` 0.316, 0.242, 0.040, `perm_p` 0.028.
Women's Body Study (`JHB_DPHRU_013`): 0.519, 0.649, 1.000 (the last group has 7 people), `perm_p` 0.027.
People living with HIV on the common panel (`hivpos_hct_hb_vl`): share ever viraemic does not
differ between groups, `viraemia_perm_p` 0.86.

Shuffling HIV status within cohorts, p = 0.28: script `07`, output `r1_hiv_tests.json`,
`H4.within_cohort_perm_p`.

## 4. Pre-specified HIV tests (appendix SA15, table S13)

Script `07_hiv_prespecified_tests.py`, output `r1_hiv_tests.json`.

| Test | Estimate | p | q | Where |
|---|---|---|---|---|
| H1 HIV x temperature, mixed cohorts | 0.0043 SD per degree (-0.0008 to 0.0093) | 0.100 | 0.166 | `H1.primary`, `BH.H1` |
| H2 viraemia within person | odds ratio 0.963 per degree (0.935 to 0.992) | 0.013 | 0.034 | `H2.primary`, `BH.H2` |
| H3 log CD4 within person | 0.0053 per degree | 0.014 | 0.034 | `H3.primary`, `BH.H3` |
| H4 Cluster 3 by HIV within cohort | pooled odds ratio 1.86 | 0.578 | 0.722 | `H4`, `BH.H4` |
| H5 low CD4 x temperature | 0.0007 SD per degree | 0.753 | 0.753 | `H5.primary`, `BH.H5` |

Also in `H2`: within-person log10 HIV viral load, -0.0053 per degree, p = 0.22 (`H2.log10_viral_load`).
Three cohorts enrolled people with and without HIV, but only JHB_DPHRU_013 had enough of both
with repeated biomarkers to estimate the interaction, so H1 rests on that cohort alone. It pooled
13 biomarkers by inverse-variance weighting, which treats them as independent; biomarkers from the
same people are correlated, so the pooled interval is if anything too narrow. The per-biomarker
results are in `H1.per_biomarker`
(hip circumference 0.036 SD per degree, p = 0.0004, uncorrected; the other 12 p from 0.05 to 0.99).

H2 model choice. The pre-specified model was a conditional logit with a linear probability
fallback. The conditional logit converges (`H2.conditional_logit.converged` true, `max_abs_score`
zero to machine precision) and is the reported result. The fallback is kept for comparison in
`H2.linear_probability` (-0.0018 per degree, p = 0.233).

Correction to H2, 10 November 2026. The first version of script 07 fitted this model with BFGS,
which stopped just short of the maximum (largest score 0.13 against a tolerance of 0.01), so the
script reported the fallback and a draft of the appendix said the model had not converged. An
independent code review caught it. Newton's method reaches the maximum at the estimate BFGS had
stopped at, so the estimate did not move; what changed is that it is now the reported result, and
the q values with it (H1 0.249 to 0.166, H2 0.388 to 0.034, H3 0.068 to 0.034). The outputs before
the correction are in the history of this branch.

H2 against the decision rule: sign under the secondary window `H2.secondary` (odds ratio 0.994,
p = 0.64); sign with each cohort left out `H2.loco` (all below 1; weakest without JHB_WRHI_001,
0.981, p = 0.30). H2 post hoc checks (`H2_posthoc`): cubic in time since first visit 0.964
(p = 0.050); first 90 days excluded 0.925 (p = 0.034; 227 participants); first 180 days excluded
0.926 (p = 0.083); cohort-by-month terms not estimable; `by_cohort` below 1 in three cohorts and
above 1 in two, with only JHB_WRHI_001 excluding 1 (0.932, p = 0.019). 49% of results in the first
90 days were viraemic against 5% afterwards (`share_viraemic_first_90_days`, `share_viraemic_after_90_days`).

H3 post hoc checks (`H3_posthoc`): with visits in the first 90 days excluded the estimate is
-0.0009, p = 0.59 (`s4_drop_first_90_days`). The other entries are the cubic time trend
(`s1_time_since_first_visit_cubic`), person-specific trends (`s2_person_linear_trend`),
cohort-by-month adjustment (`s3_cohort_x_month`) and adjustment for haematocrit at the same visit
(`s5_adjust_hct_same_visit`, with `s5_same_sample_no_hct` the unadjusted estimate in the same visits).

The Benjamini-Hochberg correction is across the five primary tests. A q value below 0.10 was the
threshold in the analysis plan written before the tests were run (appendix SA15).

"Mixed-HIV cohort" is defined per analysis, because each needs something different. Script 07 uses
the three cohorts that enrolled both groups. Script 06 uses the two of those with enough people
for a within-cohort clustering. Variant `f` in script 04 drops only cohorts that are entirely or
almost entirely one status (100% or under 1%), which leaves five.

## 5. UTCI (appendix SA16, table S14)

Scripts `08_utci_download.py` and `09_utci_sensitivity.py`, outputs `r1_sa16_utci.csv` and
`r1_sa16_utci_info.json`. UTCI was available for every visit (`coverage` 1.0) and correlated 0.94
with air temperature at the 7-day lag (`corr_utci7_air7`). Per biomarker: `delta_air` and
`delta_utci` are the two increments, `delta_utci_ci` the interval for UTCI, `lag_profile_rho` the
agreement between lag profiles (median 0.59 across the ten).

## 6. Table 1 and supplementary data file 1

Table 1 columns Female, Age and Informal: script `01_table1_demographics.py`, output
`r1_table1_demographics.csv`.

Supplementary data file 1: script `10_stage2_full_table.py`, output
`r1_stage2_all_340_pairs.csv` (340 rows; `p_raw`, `q_BH`, `direction_rho`).

## 7. History of the code, for transparency

The analyses were first run on 10 November 2026 with denser versions of these scripts. They were
then rewritten so that each step is named and each script reads top to bottom, and rerun. The
rerun reproduced every value in the first outputs: 5,674 values across 13 files, none differing
(relative tolerance 1e-6). After an independent code review the scripts were tidied again and rerun
into the files now in `outputs/`; the values again matched. Two things differ by design: permutation
p values are now (1 + count) / (1 + permutations), so a p of 0 becomes 0.0001, and cluster sizes in
`r1_within_panel.json` are listed in the same order as the prevalences. Afterwards one thing was added at a reviewer's request: the bootstrap in
script 02 now also reports an interval for each model's own R-squared (`r2_full_lo`, `r2_full_hi`).
It uses the same resamples, so the increments and their intervals are unchanged.

The HIV tests were first run as two scripts and are now the single script 07, which applies the
model choice for H2 and the Benjamini-Hochberg correction in one place, so `r1_hiv_tests.json` holds
the reported q values directly. The correction to H2 is described in section 4.
