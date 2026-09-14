# HE2AT Center — Multi-System Biomarker Heat Analysis

**Paper**: "Heat, Vulnerability, and the Body: Multi-System Biomarker Signatures
of Temperature Exposure in Johannesburg Revealed Through Explainable Machine Learning"

**Lead author**: Craig Parker | **Target journal**: Environmental Data Science (CUP)

**Funding**: NIH U54 TW 012083 (HE2AT Center)

## Repository Structure

```
heat_analysis_optimized/
│
├── run_pipeline.py                # Top-level orchestrator (--dry-run, --validate, --stage N)
│
├── mcd_pipeline/                  # Analysis pipeline (MCD-aligned)
│   ├── config.py                  # Central configuration (all paths + MCD parameters)
│   ├── stage0_data_prep/          # Data merging, feature engineering, validation
│   ├── stage1_lag_profiling/      # XGBoost-SHAP per biomarker (50 bootstrap replicates)
│   ├── stage2_interactions/       # SHAP interaction detection + permutation null
│   ├── stage3_clustering/         # PCA → k-means vulnerability clustering
│   ├── sensitivity/               # 8 sensitivity analyses
│   ├── multiple_testing/          # Bonferroni (system) + FDR (within-system)
│   ├── evaluation/                # ML metrics + clinical significance
│   └── utils/                     # Reproducibility, logging, climate indices
│
├── FINAL_DATASETS/                # All data (see detailed structure below)
│   ├── TIDY_DATASETS/             # ★ Long-format tidy CSVs — canonical pipeline inputs
│   ├── INDIVIDUAL_STUDIES/        # 22 per-study harmonised datasets (14 JHB + 8 ABJ)
│   ├── CLIMATE_HEALTH_LINKAGE/    # ERA5 climate extraction + linked datasets
│   ├── SOCIOECONOMIC_LINKAGE/     # Ward-based SES matching (v4.0, all 14 JHB studies)
│   ├── HARMONIZATION_NOTEBOOKS/   # Jupyter notebooks producing the above
│   ├── ABIDJAN_RAW_DATA/          # Raw Abidjan PAC-CI study data (8 studies)
│   └── metadata/                  # QA reports, harmonization JSONs, traceability
│
├── tests/                         # pytest suite
├── mcd_outputs/                   # Pipeline outputs (gitignored, regenerable)
├── mcd_publication/               # Publication figures, tables, literature defence
├── dlnm_r/                        # R-based DLNM sensitivity analysis
├── INTEGRATION_SCRIPTS/           # Data creation scripts
├── RP2_METADATA/                  # Study metadata, REDCap codebook
│
├── _archive/                      # Archived Sep 2025 code (reference only, gitignored)
└── _legacy_data_archive/          # Pre-reorganisation data (gitignored)
```

## Quick Start

```bash
# Validate config (no data needed)
python run_pipeline.py --dry-run

# Validate data integrity
python run_pipeline.py --validate

# Run tests
pytest tests/ -v
```

```python
import pandas as pd
from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.biomarker_definitions import get_available_biomarkers

# Wide-format analysis dataset
df = pd.read_csv(config.ANALYSIS_READY_CSV, low_memory=False)

# Tidy (long-format) datasets
biomarkers = pd.read_csv(config.TIDY_BIOMARKERS_CSV)
climate    = pd.read_csv(config.TIDY_CLIMATE_CSV)
ses        = pd.read_csv(config.TIDY_SOCIOECONOMIC_CSV)

biomarkers_available = get_available_biomarkers()
print(f"{len(biomarkers_available)} biomarkers across 8 organ systems")
```

## Data

| Dataset | Records | Description |
|---------|---------|-------------|
| Johannesburg clinical | 11,114 | 14 cohorts (2003-2021), 27 biomarkers, 8 organ systems |
| Abidjan clinical | 76,000+ | 8 PAC-CI studies (harmonization in progress) |
| ERA5 climate | linked | Temperature lags 0-30 days, heat indices, variability |
| GCRO socioeconomic | 58,616 | Quality of Life surveys (2011-2021), 6 waves |
| SES linkage | all 14 JHB | Ward-based NND matching (v4.0) |

## MCD Analysis Plan (3 Stages)

**Stage 1 — Lag-response profiling**: One XGBoost model per biomarker with
temperature at lags 0, 1, 3, 7, 14, 21, 30 days. Interventional TreeSHAP.
50 bootstrap replicates for SHAP stability.

**Stage 2 — Interaction detection**: SHAP interaction values for
temperature x dwelling type, income, HIV status, age. 500-permutation null.
Benjamini-Hochberg FDR q=0.10.

**Stage 3 — Vulnerability clustering**: PCA (85% variance) on individual
SHAP vectors, then k-means (k=3-8, silhouette selection). Bootstrap stability
>0.80 over 500 iterations.

**Multiple testing**: Bonferroni alpha=0.00625 for 8 biomarker systems;
FDR q=0.10 within significant systems.

## Status

- [x] Data harmonisation and QC (ALCOA+ framework) — 14 JHB studies
- [x] Abidjan harmonisation — 8 PAC-CI studies (76K records)
- [x] Climate linkage (v2.0 pipeline, ERA5 + DLNM lags)
- [x] Socioeconomic linkage (v4.0, all 14 JHB studies, ward-based NND)
- [x] Tidy dataset generation (5 long-format CSVs)
- [x] Heat Vulnerability Index (HVI) — 3-component framework
- [ ] Stage 1: Lag-response profiling
- [ ] Stage 2: Interaction detection
- [ ] Stage 3: Vulnerability clustering
- [ ] Sensitivity analyses
- [ ] Manuscript draft
