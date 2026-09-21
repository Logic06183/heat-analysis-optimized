# Multi-system biomarker signatures of ambient heat exposure in Johannesburg

Analysis code for:

Parker C, Brink N, Radebe L, et al. *Multi-system biomarker signatures of ambient heat exposure in Johannesburg: a multi-cohort study with a hypothesis-generating vulnerability cluster in people living with HIV.* Submitted to The Lancet Regional Health – Africa, 2026.

HE2AT Center (HEat and HEalth African Transdisciplinary Center), NIH award U54 TW 012083, Fogarty International Center. Study protocol: Jack C, Parker C, et al. BMJ Open 2024; 14: e077529.

Corresponding author: Craig Parker, Wits Planetary Health Research, University of the Witwatersrand.

## What the study does

Pooled secondary analysis of harmonised individual-level data from 14 adult Johannesburg cohorts (2005 to 2022; 9,745 participants; 598,433 biomarker observations). Each clinic visit is linked to ERA5-Land daily mean 2 m air temperature at the clinic over seven lag windows (day of visit, 1, 3, 7, 14, 21 days, and a 30-day cumulative mean) and to ward-level socioeconomic covariates from the Gauteng City-Region Observatory Quality of Life Survey.

Three stages:

1. **Stage 1, screening.** One gradient-boosted model (XGBoost) per biomarker, judged by 5-fold cross-validation grouped by participant, with SHAP attribution for lag timing. Biomarkers with mean cross-validated R² of 0·05 or more are retained (10 of 24).
2. **Stage 2, effect modification.** SHAP interaction values for each retained biomarker against sex, HIV status, age, socioeconomic covariates and other temperature lags, tested against a permutation null with Benjamini–Hochberg control at q=0·10.
3. **Stage 3, grouping.** Participant-level SHAP profiles across the retained biomarkers, standardised, reduced by principal components to 85% of variance, and grouped by a Gaussian mixture model (k=3, full covariance, first ten components, seed 42). Stability by 500-resample bootstrap and adjusted Rand index.

Fourteen sensitivity analyses are reported in the paper's appendix (SA1 to SA14).

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
publication_figures_lancet/     figure scripts, captions, final figure files, and SA14
tests/                          pytest suite
RP2_METADATA/                   variable-level codebook (no participant records)
DOCUMENTATION/, INTEGRATION_SCRIPTS/, mcd_publication/   supporting notes
environment.yml, requirements.txt
```

Several analyses were run from standalone scripts at the repository root rather than through `run_pipeline.py`. Each script's docstring says what it does and how it was invoked; the table below maps them to the appendix.

## Sensitivity analyses: appendix numbering against script names

The appendix numbers the analyses SA1 to SA14 in the order they appear in the paper. Script names predate that numbering and do not match it: a script called `sa9_*` is not appendix SA9. Read this table, not the file names. Mappings were confirmed against each script's docstring on 20 September 2026.

| Appendix | Analysis | Script(s) |
|---|---|---|
| SA1 | DLNM cross-check | `mcd_pipeline/sensitivity/sa1_dlnm_validation.py`; `export_dlnm_r_input_era5land.py`; `dlnm_r/sa1_dlnm_validation_era5land.R` |
| SA2 | MODIS land surface temperature and 31 km ERA5 | `mcd_pipeline/sensitivity/sa2_modis_vs_era5.py`; `mcd_pipeline/sensitivity/sa12_era5land.py` and `sa12_recover.py` (ERA5 against ERA5-Land, written when ERA5 was the primary exposure) |
| SA3 | Within-person case-crossover | `mcd_pipeline/sensitivity/sa6_case_crossover.py`; `sa6b_conditional_logistic_or.py` |
| SA4 | HIV-stratified Stage 1 | `mcd_pipeline/sensitivity/sa7_hiv_vs_general.py` |
| SA5 | Heatwave-day thresholds | `mcd_pipeline/sensitivity/sa8_heatwave_threshold.py`; runner `run_sa8.py` |
| SA6 | Per-cohort meta-analysis | `mcd_pipeline/sensitivity/sa9_per_cohort_meta.py`; `run_pipeline.py --revision-sa9` |
| SA7 | Humidity-aware exposure metrics | `mcd_pipeline/sensitivity/sa10_humidity.py`; `run_pipeline.py --revision-sa10` or `run_sa10.py` |
| SA8 | Leave-one-cohort-out and cluster stability | `sa11_leave_one_cohort_out.py`; `rerun_10biomarker_lobo.py`; `rerun_10biomarker_robustness.py`; `cement_findings.py`; `cement_findings_lobo.py` |
| SA9 | Nested cross-validation of hyperparameters | `run_nested_cv_sensitivity.py` |
| SA10 | Sex-stratified Stage 1 | `run_sa13_sex_stratified.py` |
| SA11 | Viral load censoring | No standalone script in this repository. |
| SA12 | Pollution-dispersion proxy | `mcd_pipeline/sensitivity/sa3_pollution_adjusted.py`; `mcd_pipeline/utils/extract_merra2_pollution.py` |
| SA13 | COVID-19 period exclusion | `mcd_pipeline/sensitivity/sa4_covid_exclusion.py` |
| SA14 | Clustering without body-composition measures | `publication_figures_lancet/_sa14_anthro/sa14_drop_anthropometrics.py`, with `sa14_results.json`, bootstrap arrays and run log alongside |

Provenance notes, from an audit of outputs against the appendix on 21 September 2026:

- SA6 pools with fixed-effect and DerSimonian-Laird random-effect meta-analysis (`meta_analysis.csv` per biomarker); no Hartung-Knapp adjustment is implemented.
- SA8 numbers reported in the paper (leave-one-cohort-out, leave-one-biomarker-out, label permutation, multi-seed) come from `rerun_10biomarker_robustness.py` and `rerun_10biomarker_lobo.py`, written to `mcd_outputs_era5_land/diagnostics_10biomarker/`. `sa11_leave_one_cohort_out.py` is the earlier implementation and its output under `sensitivity/sa11_leave_one_cohort_out/` is from the thirteen-biomarker fit.
- SA12 adjusts for MERRA-2 PM2.5 at the seven lag windows (`extract_merra2_pollution.py`), not for boundary-layer height.
- SA13 (`sa4_covid_exclusion.py`) drops calendar years 2020 and 2021 and re-runs Stage 1, reporting lag-profile concordance; it does not refit Stage 3.
- SA3 (`sa6_case_crossover.py`, `sa6b_conditional_logistic_or.py`) reports conditional-logistic odds ratios per degree Celsius within patient-by-month strata.

Runners `run_sas_era5land.py` and `run_remaining_sas.py` batch several of the `mcd_pipeline/sensitivity/` scripts under the ERA5-Land exposure. `sa5_imputation_compare.py` is not reported in the paper. Cluster proportion confidence intervals and the k-selection parsimony diagnostics reported in the appendix run through `run_pipeline.py --revision-cluster-cis` and `--revision-parsimony`.

## Figures

`publication_figures_lancet/` holds the figure scripts (`fig01_*.py` to `figS7_*.py`, `figS_DAG.py`), their caption text, and the figure files as submitted under `artwork_for_submission/` (SVG, PNG and PDF; regenerated 4 and 18 September 2026 from the ten-biomarker Stage 3 run). Supplementary figures are rendered through `_save_artwork_plain.py` (plain tight-bbox save, the form the supplementary set was produced in); main-text figures save at Lancet column width via `_lancet_style.save_at_width`. The scripts read the retained-biomarker panel and Stage 3 outputs from `mcd_outputs_era5_land/`, which is generated from the clinical data and is not distributed; a few scripts also read small participant-level cache files written by `regenerate_final_figures.py`, which are likewise not in the repository.

## Data

The harmonised clinical dataset is not in this repository and cannot be made public: participant consent was given to the contributing studies, not to a pooled secondary analysis, and the combination of demographic, clinical and ward-level variables carries re-identification risk. Access for qualified researchers is controlled through the HE2AT Center Data Access Committee under a data use agreement; requests to the corresponding author. See the paper's data availability statement.

A metadata catalogue describing every variable, its definition and its cohort coverage is at https://doi.org/10.5281/zenodo.19482613. `RP2_METADATA/RP2_Data_Dictionary_Codebook.csv` in this repository is a variable-level dictionary with no participant records.

ERA5-Land is openly available from the Copernicus Climate Data Store. GCRO Quality of Life Survey data are available from https://www.gcro.ac.za.

## Running

```
conda env create -f environment.yml
conda activate heat-analysis
export RP2_PRIMARY_EXPOSURE=era5_land
python run_pipeline.py --stage 0 1 2 3
python run_pipeline.py --revision          # SA6, SA7, cluster CIs, parsimony diagnostics
pytest
```

Standalone scripts at the root are run from the repository root with `PYTHONPATH=.` and the same environment variable. Every stage requires the harmonised dataset under `FINAL_DATASETS/`, which is not distributed; without it the code can be read but the analyses cannot be re-executed. Of the 272 tests, 216 run without the dataset; the other 56 check for it or load it and will fail or error in a checkout without it.

## Ethics

University of the Witwatersrand Human Research Ethics Committee (Medical), reference 220606. Each contributing study held its own ethics approval under its original protocol.

## Licence

MIT. See `LICENSE`.
