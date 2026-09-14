# RP2 Dataset Metadata for eLwazi REDCap Catalogue

Record ID: 169-1 | Event: Dataset (Instance #5)

---

## DATASET DESCRIPTORS

| Field | Value |
|-------|-------|
| **Consortium or Network** | HE2AT Center |
| **Data Provider** | HE2AT Center / CSAG, University of Cape Town |
| **Project Title** | Heat, Vulnerability, and the Body: Multi-System Biomarker Signatures of Temperature Exposure in Johannesburg |
| **Dataset Name** | Johannesburg Clinical Analysis Ready Dataset (RP2) |
| **Dataset Description** | Harmonised multi-system clinical biomarker dataset from 14 Johannesburg clinical studies (17 study arms) linked with ERA5 reanalysis temperature data and GCRO socioeconomic indicators, comprising 62,154 visit records from 12,804 unique participants for heat-health analysis. |
| **Dataset Provenance** | Harmonised data |
| **Description of Source Datasets** | Harmonised from 14 independent Johannesburg clinical studies: DPHRU (053, 013), EZIN (025, 002), WRHI (001, 003), Aurum (009), VIDA (007, 008), ACTG (015, 016, 017, 018, 019, 021), and SCHARP (004, 006). Clinical data linked with ERA5 reanalysis climate data (lags 0-30 days) and Gauteng City-Region Observatory (GCRO) Quality of Life survey socioeconomic indicators via statistical matching. |
| **Dataset Subject/s** | Adult (age range: 13-76 years, mean 35 years) |
| **Dataset Design** | Cohort (Longitudinal) |
| **Data Domains** | Demographic & Health, Geospatial |
| **Total Subjects** | 12,804 unique participants (62,154 visit records) |
| **African Countries** | South Africa |

---

## VERSION, FORMAT & STORAGE

| Field | Value |
|-------|-------|
| **Dataset Version** | 3.0 |
| **Version History** | v1.0: Initial harmonisation of 14 studies. v2.0: Added climate linkage (ERA5 reanalysis, lags 0-30d). v3.0: Added GCRO socioeconomic statistical matching and quality-controlled analysis-ready format. |
| **Dataset Status** | Final dataset |
| **Dated From** | 2003-10-20 |
| **Dated To** | 2021-12-09 |
| **Dataset Storage Location** | CSAG GitLab (gitlab.csag.uct.ac.za) |
| **URL, if available** | http://gitlab.csag.uct.ac.za/he2at/heat-analysis-optimized |
| **Dataset Workspace** | CSAG GitLab |
| **Dataset Format** | CSV |
| **Last Update** | 10-03-2026 |
| **Catalogue Ready** | Yes |

---

## DETAILED DATA SUMMARY

### File Inventory

| File | Records | Columns | Description |
|------|---------|---------|-------------|
| JOHANNESBURG_CLINICAL_ANALYSIS_READY.csv | 62,154 | 96 | Primary analysis-ready dataset (QC'd, ALCOA+ compliant) |
| JOHANNESBURG_CLINICAL_FULL.csv | 11,114 | 205 | Full columns including all original study-specific variables |
| GCRO_SOCIOECONOMIC_CLIMATE_ENHANCED_LABELED.csv | 58,616 | 90 | GCRO socioeconomic + climate indicators |
| INDIVIDUAL_STUDIES/ (14 files) | varies | varies | Per-study harmonised datasets |
| CLIMATE_HEALTH_LINKAGE/ (14 notebooks) | varies | varies | Per-study climate extraction and linkage |

### Source Studies (17 study arms from 14 studies)

| Study | Records | Primary Focus |
|-------|---------|---------------|
| JHB_VIDA_007 | 15,581 | Clinical trial |
| JHB_Aurum_009 | 12,575 | Clinical trial |
| JHB_Ezin_002 | 11,059 | Clinical trial |
| JHB_WRHI_001 | 10,044 | Clinical trial |
| JHB_ACTG_019 | 2,594 | Clinical trial |
| JHB_ACTG_015 | 2,364 | Clinical trial |
| JHB_WRHI_003 | 2,235 | Clinical trial |
| JHB_SCHARP_006 | 1,359 | Clinical trial |
| JHB_DPHRU_053 | 1,013 | Cohort |
| JHB_DPHRU_013 | 784 | Cohort |
| JHB_ACTG_016 | 766 | Clinical trial |
| JHB_VIDA_008 | 550 | Clinical trial |
| JHB_EZIN_025 | 489 | Clinical trial |
| JHB_SCHARP_004 | 403 | Clinical trial |
| JHB_ACTG_018 | 240 | Clinical trial |
| JHB_ACTG_021 | 78 | Clinical trial |
| JHB_ACTG_017 | 20 | Clinical trial |

### Participant Demographics

| Characteristic | Value |
|----------------|-------|
| Total participants | 12,804 |
| Total visit records | 62,154 |
| Age range | 13 - 76 years |
| Age mean / median | 35.0 / 34.0 years |
| Female | 28,565 (46.0%) |
| Male | 14,927 (24.0%) |
| HIV Positive | 42,174 (67.9%) |
| HIV Negative | 1,359 (2.2%) |
| HIV Unknown | 16,386 (26.4%) |

### Biomarker Categories (8 organ systems, 30 registered biomarkers)

1. **Renal**: creatinine, creatinine_clearance, bun
2. **Metabolic**: fasting_glucose, fasting_insulin, hba1c
3. **Cardiovascular**: systolic_bp, diastolic_bp, heart_rate
4. **Inflammatory**: hs_crp, wbc_count
5. **Immunological**: cd4_count, viral_load
6. **Hepatic**: alt, ast, alp, total_bilirubin, albumin, total_protein
7. **Lipid**: total_cholesterol, hdl_cholesterol, ldl_cholesterol, triglycerides
8. **Body Composition**: bmi, total_fat_mass, total_lean_mass, body_fat_percent, grip_strength

### Climate Variables

- ERA5 reanalysis temperature: mean, max, min, range
- Temperature lags: 1-day through 30-day
- Derived indices: apparent temperature, heat wave day, heat stress category
- Temperature variability: 7-day and 30-day windows
- Temperature anomaly

### Socioeconomic Variables (GCRO)

- Dwelling type, employment status, education level, income bracket
- Heat vulnerability index, ward-level spatial identifiers
- Source: Gauteng City-Region Observatory Quality of Life Surveys

---

## FUNDING

NIH U54 TW 012083 (HE2AT Center)

---

## COPY-PASTE VALUES FOR REDCAP FORM

Below are the exact values to enter in each REDCap field:

**Consortium or Network**: HE2AT Center
**Data Provider**: HE2AT Center / CSAG, University of Cape Town
**Project Title**: Heat, Vulnerability, and the Body: Multi-System Biomarker Signatures of Temperature Exposure in Johannesburg
**Dataset Name**: Johannesburg Clinical Analysis Ready Dataset (RP2)
**Dataset Description**: Harmonised multi-system clinical biomarker dataset from 14 Johannesburg clinical studies linked with ERA5 climate data and GCRO socioeconomic indicators. 62,154 visit records from 12,804 participants for heat-health analysis.
**Dataset Provenance**: Harmonised data
**Description of Source Datasets**: Harmonised from 14 Johannesburg clinical studies (DPHRU, EZIN, WRHI, Aurum, VIDA, ACTG, SCHARP) linked with ERA5 reanalysis climate data (lags 0-30 days) and GCRO Quality of Life survey socioeconomic indicators via statistical matching.
**Dataset Subject/s**: Adult
**Dataset Design**: Cohort (Longitudinal)
**Data Domains**: Demographic & Health, Geospatial
**Total Subjects**: 12,804
**African Countries**: South Africa
**Dataset Version**: 3.0
**Version History**: v1.0 Initial harmonisation. v2.0 Added ERA5 climate linkage. v3.0 Added GCRO socioeconomic matching and analysis-ready QC.
**Dataset Status**: Final dataset
**Dated From**: 2003-10-20
**Dated To**: 2021-12-09
**Dataset Storage Location**: CSAG GitLab (gitlab.csag.uct.ac.za)
**URL**: http://gitlab.csag.uct.ac.za/he2at/heat-analysis-optimized
**Dataset Workspace**: CSAG GitLab
**Dataset Format**: CSV
**Last Update**: 10-03-2026
**Catalogue Ready**: Yes
