# R1 revision analyses

Analyses added for the first-round revision of "Multi-system biomarker signatures of ambient heat
exposure in Johannesburg" (The Lancet Regional Health - Africa, TLRHAFRICA-D-26-01053).

If you are reviewing the code, start with `REVIEW_GUIDE.md`. It lists every number the revision
reports, the script that produced it and where to find it in the output files.

## Layout

```
r1_revision/
  REVIEW_GUIDE.md     number-by-number map from the paper to the code
  scripts/
    r1_common.py      shared steps, written once: load data, cross-validate, bootstrap, cluster, permute
    01 ... 10         one script per analysis, numbered in the order the paper uses them
  outputs/            aggregate results only (no participant-level rows)
```

Each numbered script opens with the same five-part header: the question it answers, the data it
reads, the steps, the files it writes and where the result is reported. The scripts run top to
bottom with no hidden state.

| Script | Appendix | Answers |
|---|---|---|
| `01_table1_demographics.py` | Table 1 | Sex, age and dwelling type by cohort |
| `02_incremental_r2.py` | SA12 | How much temperature adds to each model; HIV viral load as viraemic or not |
| `03_incremental_r2_leave_one_cohort_out.py` | SA12 | Whether one cohort carries the temperature signal |
| `04_clustering_sensitivity.py` | SA13, SA14 | Number of clusters; missing-biomarker treatments; cohort exclusions |
| `05_availability_by_cohort.py` | SA14 | Which cohorts measured which biomarkers, and Cluster 3 share |
| `06_within_panel_clustering.py` | SA14 | Clustering inside one cohort or one biomarker panel |
| `07_hiv_prespecified_tests.py` | SA15 | Five pre-specified HIV tests and the post hoc checks of H3 |
| `08_utci_download.py` | SA16 | Downloads UTCI from the Copernicus Climate Data Store |
| `09_utci_sensitivity.py` | SA16 | The screen repeated with UTCI |
| `10_stage2_full_table.py` | Supplementary data file 1 | All 340 Stage 2 pairs with direction |

## How to run

The scripts import the main pipeline (`mcd_pipeline`) from the repository root and read the
harmonised dataset from `FINAL_DATASETS/`. That dataset holds participant-level clinical data and
is not public; access is through the HE2AT Center. They also read the published Stage 1 to 3
outputs in `mcd_outputs_era5_land/`.

```
cd r1_revision/scripts
python3 02_incremental_r2.py
```

Script 08 needs a Copernicus Climate Data Store account. Script 09 needs the file script 08 writes.
Everything else is independent. Run times on a 2023 laptop: script 02 about 10 minutes, 03 about
11, 04 about 4, 09 about 6, the rest under a minute.

Settings (model hyperparameters, folds, lags, seeds) come from `mcd_pipeline/config.py`, the same
file the published analysis used. Software used for the reported run: Python 3.11, xgboost 3.0.2,
scikit-learn 1.7.0, pandas 2.3.0, numpy 2.2.6, statsmodels 0.14.5, scipy 1.16.0.

## What was checked

- Script 02 refits the published model for all 24 screened biomarkers. Its `r2_full` column equals
  the published cross-validated R-squared (`r2_published`) for every one.
- Script 04 rebuilds the published clusters from the Stage 1 SHAP values before running any
  variant (adjusted Rand index 1.000 against the published labels).
- The scripts were rewritten for readability on 10 November 2026. The rewritten versions were
  rerun and reproduce the earlier outputs value for value (see `REVIEW_GUIDE.md`, last section).
