# Multi-system biomarker signatures of ambient heat exposure in Johannesburg

Analysis code for:

Parker C, Brink N, Radebe L, et al. *Multi-system biomarker signatures of ambient heat exposure in Johannesburg: a multi-cohort study with a hypothesis-generating vulnerability cluster in people living with HIV.* Submitted to The Lancet Regional Health – Africa, 2026.

HE2AT Center (HEat and HEalth African Transdisciplinary Center), NIH award U54 TW 012083, Fogarty International Center. Study protocol: Jack C, Parker C, et al. BMJ Open 2024; 14: e077529.

Corresponding author: Craig Parker, Wits Planetary Health Research, University of the Witwatersrand.

## What the study does

Pooled secondary analysis of harmonised individual-level data from 14 adult Johannesburg cohorts (2005 to 2022; 9,745 participants; 598,433 biomarker observations). Each clinic visit is linked to ERA5-Land daily mean 2 m air temperature over seven lag windows (day of visit, 1, 3, 7, 14, 21 days, and a 30-day cumulative mean), taken at the participant's residential location for the three cohorts that recorded it (Thol'Impilo, MASC, PEARLS) and at the study clinic for the other eleven, and to ward-level socioeconomic covariates from the Gauteng City-Region Observatory Quality of Life Survey.

Three stages:

1. **Stage 1, screening.** One gradient-boosted model (XGBoost) per biomarker, judged by 5-fold cross-validation grouped by participant, with SHAP attribution for lag timing. Biomarkers with mean cross-validated R² of 0·05 or more are retained (10 of 24).
2. **Stage 2, effect modification.** SHAP interaction values for each retained biomarker against sex, HIV status, age, socioeconomic covariates and other temperature lags, tested against a permutation null with Benjamini–Hochberg control at q=0·10.
3. **Stage 3, grouping.** Participant-level SHAP profiles across the retained biomarkers, standardised, reduced by principal components to 85% of variance, and grouped by a Gaussian mixture model (k=3, full covariance, first ten components, seed 42). Stability by 500-resample bootstrap and adjusted Rand index.

Eleven sensitivity analyses are reported in the paper's appendix (SA1 to SA11). Five further analyses were run, or drafted, and are not reported; they are listed below so the code is accounted for.

## Layout

```
run_pipeline.py                 entry point; --stage 0 1 2 3, plus --revision flags (below)
mcd_pipeline/
  config.py                     paths, exposure product (RP2_PRIMARY_EXPOSURE=era5_land), seeds
  stage0_data_prep/             harmonised dataset assembly and feature engineering
  stage1_lag_profiling/         per-biomarker XGBoost, SHAP, lag-response summaries
  stage2_interactions/          SHAP interactions, permutation null, FDR
  stage3_clustering/            participant-level SHAP matrix, PCA, GMM, bootstrap stability,
                                cluster proportion CIs, parsimony diagnostics
  multiple_testing/             Benjamini-Hochberg helpers
  sensitivity/                  alternative specifications (see table below)
  utils/                        ERA5-Land, MODIS and MERRA-2 extraction, reproducibility helpers
dlnm_r/                         distributed lag non-linear model cross-check (R, dlnm)
publication_figures_lancet/     figure scripts, captions, final figure files, and SA11
tests/                          pytest suite
RP2_METADATA/                   variable-level codebook (no participant records)
DOCUMENTATION/, INTEGRATION_SCRIPTS/, mcd_publication/   supporting notes
environment.yml, requirements.txt
```

Several analyses were run from standalone scripts at the repository root rather than through `run_pipeline.py`. Each script's docstring says what it does and how it was invoked; the table below maps them to the appendix.

## Sensitivity analyses: appendix numbering against script names

The appendix numbers the analyses SA1 to SA11 in the order they appear in the paper. Script names predate that numbering and do not match it: a script called `sa9_*` is not appendix SA9. Read this table, not the file names. Mappings were confirmed against each script's docstring on 21 September 2026.

Five of the reported analyses were pre-specified in the analysis plan (SA1, SA3, SA4, SA9, SA10); the other six were added after the plan was fixed. SA2 compares the two reanalysis grids and is not the plan's exposure-product item, which was the MODIS substitution listed below.

| Appendix | Analysis | Script(s) |
|---|---|---|
| SA1 | DLNM cross-check | `mcd_pipeline/sensitivity/sa1_dlnm_validation.py`; `export_dlnm_r_input_era5land.py`; `dlnm_r/sa1_dlnm_validation_era5land.R` |
| SA2 | 31 km ERA5 against the 9 km ERA5-Land primary, at the clustering stage | `mcd_pipeline/sensitivity/sa12_era5land.py` and `sa12_recover.py` (Stage 1 under ERA5-Land, written when ERA5 was the primary exposure); cluster agreement in `mcd_outputs_era5_land/sensitivity/cluster_consistency_primary_vs_era5land/` |
| SA3 | HIV-stratified Stage 1 | `mcd_pipeline/sensitivity/sa7_hiv_vs_general.py` |
| SA4 | Heatwave-day thresholds | `mcd_pipeline/sensitivity/sa8_heatwave_threshold.py`; runner `run_sa8.py` |
| SA5 | Humidity-aware exposure metrics | `mcd_pipeline/sensitivity/sa10_humidity.py`; `run_pipeline.py --revision-sa10` or `run_sa10.py` |
| SA6 | Leave-one-cohort-out, leave-one-biomarker-out, label permutation, multi-seed refit | `rerun_10biomarker_robustness.py`; `rerun_10biomarker_lobo.py`; `cement_findings.py`; `cement_findings_lobo.py` |
| SA7 | Nested cross-validation of hyperparameters | `run_nested_cv_sensitivity.py` |
| SA8 | Sex-stratified Stage 1 | `run_sa13_sex_stratified.py` |
| SA9 | Pollution adjustment (MERRA-2 PM2.5) | `mcd_pipeline/sensitivity/sa3_pollution_adjusted.py`; `mcd_pipeline/utils/extract_merra2_pollution.py` |
| SA10 | COVID-19 period exclusion | `mcd_pipeline/sensitivity/sa4_covid_exclusion.py` |
| SA11 | Clustering without body-composition measures | `publication_figures_lancet/_sa14_anthro/sa14_drop_anthropometrics.py`, with `sa14_results.json`, bootstrap arrays and run log alongside |

Run and not reported in the paper:

| Analysis | Script(s) | Why not reported |
|---|---|---|
| MODIS land surface temperature in place of ERA5-Land (pre-specified) | `mcd_pipeline/sensitivity/sa2_modis_vs_era5.py` | Surface skin temperature at a fixed overpass under cloud gaps measures a different quantity from 2 m air temperature; mean Spearman ρ 0·18 across the retained panel reflects that difference rather than fragility of the finding. |
| Within-person case-crossover for CD4 count and viral load (pre-specified) | `mcd_pipeline/sensitivity/sa6_case_crossover.py`; `sa6b_conditional_logistic_or.py` | Temperature was available only on clinic-visit days, leaving 154 (viral load) and 66 (CD4) informative strata; odds ratios per °C were null with wide intervals (0·985, 0·909 to 1·067; 0·976, 0·856 to 1·112). Underpowered. |
| Per-cohort random-effects meta-analysis of Stage 1 lag profiles | `mcd_pipeline/sensitivity/sa9_per_cohort_meta.py`; `run_pipeline.py --revision-sa9` | Two to five cohorts per biomarker, I² near 100% at most lags; the cohort question is answered by the leave-one-cohort-out refit (SA6). Fixed-effect and DerSimonian-Laird estimates only. |
| Imputation comparison (KNN, MICE, complete case; pre-specified) | `mcd_pipeline/sensitivity/sa5_imputation_compare.py` | Never run. Socioeconomic covariates are assigned to every participant by statistical matching and the pipeline imputes nothing, so there was nothing to compare. |
| Viral-load censoring check (refits restricted to detectable results) | No script retained. | Not reported. The censoring proportion itself (72·6% of viral-load results at a study-specific substituted value) is reported in the appendix from the harmonised data. |

Provenance notes:

- SA6 numbers in the paper come from `rerun_10biomarker_robustness.py` and `rerun_10biomarker_lobo.py`, written to `mcd_outputs_era5_land/diagnostics_10biomarker/`. `sa11_leave_one_cohort_out.py` is the earlier implementation and its output under `sensitivity/sa11_leave_one_cohort_out/` is from the thirteen-biomarker fit.
- SA9 adjusts for MERRA-2 PM2.5 at the seven lag windows, not for boundary-layer height.
- SA10 (`sa4_covid_exclusion.py`) drops calendar years 2020 and 2021 and re-runs Stage 1, reporting lag-profile concordance; it does not refit Stage 3.

Runners `run_sas_era5land.py` and `run_remaining_sas.py` batch several of the `mcd_pipeline/sensitivity/` scripts under the ERA5-Land exposure. Cluster proportion confidence intervals and the k-selection parsimony diagnostics reported in the appendix run through `run_pipeline.py --revision-cluster-cis` and `--revision-parsimony`.

## Figures

`publication_figures_lancet/` holds the figure scripts (`fig01_*.py` to `figS7_*.py`, `figS_DAG.py`, and `figS8_study_flow.py`, which draws the study flow from counts reported in the paper and reads no data), their caption text, and the figure files as submitted under `artwork_for_submission/` (SVG, PNG and PDF; main figures regenerated 4 and 18 September 2026, with Figures 1 and 3 and supplementary figures S1 to S8 re-rendered on 21 September, all from the ten-biomarker Stage 3 run). The supplementary artwork as submitted (figures S1 to S7) comes from `_render_clean.py`, which runs a figure script as plain artwork in the journal's house style: header sentences and interpretive text panels removed, midline decimal points, en rules and true minus signs; the captions in the appendix carry the explanatory text. `_render_clean.py` must be run from inside `publication_figures_lancet/`, because Figures S1 and S6 locate their inputs relative to the script. Figure S5 is drawn on the Table 1 denominators (participants contributing biomarker observations). Earlier supplementary renders went through `_save_artwork_plain.py` (plain tight-bbox save, the form the supplementary set was produced in); main-text figures save at Lancet column width via `_lancet_style.save_at_width`. The scripts read the retained-biomarker panel and Stage 3 outputs from `mcd_outputs_era5_land/`, which is generated from the clinical data and is not distributed; a few scripts also read small participant-level cache files written by `regenerate_final_figures.py`, which are likewise not in the repository.

## Data

The harmonised clinical dataset is not in this repository and cannot be made public: participant consent was given to the contributing studies, not to a pooled secondary analysis, and the combination of demographic, clinical and ward-level variables carries re-identification risk. Access for qualified researchers is controlled through the HE2AT Center Data Access Committee under a data use agreement; requests to the corresponding author. See the paper's data availability statement.

A metadata catalogue describing every variable, its definition and its cohort coverage is at https://doi.org/10.5281/zenodo.19482613. `RP2_METADATA/RP2_Data_Dictionary_Codebook.csv` in this repository is a variable-level dictionary with no participant records.

ERA5-Land is openly available from the Copernicus Climate Data Store. GCRO Quality of Life Survey data are available from https://www.gcro.ac.za.

## Running

The quickest route is the pinned core requirements, which install with pip on Python 3.11 to 3.13:

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-core.txt
export RP2_PRIMARY_EXPOSURE=era5_land
PYTHONPATH=. pytest
```

`environment.yml` and `requirements.txt` are full snapshots of the analysis machine (September 2025) and include conda and Jupyter tooling the analysis does not need; use them only to reproduce that environment exactly. The DLNM cross-check (SA1) runs in R with `dlnm` 2.4.7, `lme4`, `mgcv`, `splines`, `dplyr`, `tidyr`, `lubridate`, `jsonlite` and the plotting packages loaded at the top of `dlnm_r/sa1_dlnm_validation_era5land.R`.

Full pipeline:

```
conda env create -f environment.yml
conda activate heat-analysis
export RP2_PRIMARY_EXPOSURE=era5_land
python run_pipeline.py --stage 0 1 2 3
python run_pipeline.py --revision          # SA5, cluster CIs, parsimony diagnostics
pytest
```

Standalone scripts at the root are run from the repository root with `PYTHONPATH=.` and the same environment variable. Every stage requires the harmonised dataset under `FINAL_DATASETS/`, which is not distributed; without it the code can be read but the analyses cannot be re-executed. Of the 272 tests, 216 pass without the dataset (checked on 24 September 2026 in a clean environment built from `requirements-core.txt`); the other 56 check for it or load it and will fail or error in a checkout without it.

## Ethics

University of the Witwatersrand Human Research Ethics Committee (Medical), reference 220606. Each contributing study held its own ethics approval under its original protocol.

## Licence

MIT. See `LICENSE`.
