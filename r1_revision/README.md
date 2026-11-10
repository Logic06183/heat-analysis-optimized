# R1 revision analyses (Lancet Regional Health – Africa, TLRHAFRICA-D-26-01053)

Scripts added for the first-round revision. They import the main pipeline from the repository root
(`mcd_pipeline`), read the harmonised dataset from `FINAL_DATASETS/` (not public; held by the HE²AT Center)
and write aggregate results only to `outputs/`. Run with `RP2_PRIMARY_EXPOSURE=era5_land`.

| Script | Appendix | What it does |
|---|---|---|
| `scripts/r1_incremental_r2.py` | SA12, table S10 | Cross-validated R-squared of the full model minus a covariate-only model on the same participant-grouped folds, with 1,000 participant-bootstrap intervals; HIV viraemia (≥1,000 copies/mL) classifier AUC with and without temperature |
| `scripts/r1_stage1_loco.py` | SA12 | Leave-one-cohort-out version of the incremental R-squared |
| `scripts/r1_clustering_sensitivity.py` | SA13, SA14, table S11 | Two to five groups; missing-biomarker treatments; within-cohort HIV permutation; exclusion of all-HIV and no-HIV cohorts |
| `scripts/r1_availability_by_cohort.py` | SA14, table S12 | Retained biomarkers measured by cohort and share of each cohort in Cluster 3 |
| `scripts/r1_within_panel.py` | SA14 | Clustering within a single biomarker panel and within-cohort HIV gradients |
| `scripts/r1_hiv_prespecified.py` | SA15, table S13 | Five pre-specified within-cohort and within-person tests of HIV-related differences (H1 to H5) |
| `scripts/r1_hiv_h2fix_h3stress.py` | SA15 | Linear-probability fallback for H2 after the conditional logit failed to converge; H3 stress check |
| `scripts/get_utci.py`, `scripts/r1_utci_sensitivity.py` | SA16, table S14 | UTCI (CDS derived-utci-historical-timeseries v1.1) in place of air temperature |
| `scripts/r1_stage2_table.py` | Supplementary data file 1 | All 340 Stage 2 pairs with raw p, q and a descriptive direction |
| `scripts/r1_table1_demographics.py` | Table 1 | Sex, age and informal dwelling by cohort |

`outputs/` holds the aggregate results reported in the paper (no participant-level rows).
The figure labels were revised for R1 in `publication_figures_lancet/` (cluster names, HIV viral load,
non-causal SHAP-direction keys); `artwork_for_submission/` holds the R1 artwork.
