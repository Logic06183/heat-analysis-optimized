# RP2 eLwazi REDCap — All Remaining Forms

Record ID: 169-1 | Event: Dataset (Instance #5)
Dataset: Johannesburg Clinical Analysis Ready Dataset (RP2)

---

## FORM 2: Data Access Pathways

| Field | Value |
|-------|-------|
| **Main Publication DOI** | *(pending — Parker et al. 2026, target: Environmental Data Science)* |
| **Other Publications** | *(none yet)* |
| **Access Paths** | Directly contact primary dataset owner; Directly contact Data Access Committee |
| **Data Access Committee Email or URL** | cparker@csag.uct.ac.za |

> Once Zenodo DOI is minted, add: Access via 3rd Party Platform, and enter DOI.

---

## FORM 3: Use Conditions

**Suggested values** (adapt to your institutional data governance):

| Field | Suggested Value |
|-------|-----------------|
| **Data Use Ontology (DUO) Code** | DUO:0000007 — Disease Specific Research (DS) |
| **Primary Use** | Health research related to heat exposure and climate-health impacts |
| **Secondary Use Allowed** | Yes — with approval from data access committee |
| **Restrictions** | No commercial use; no re-identification attempts; results must be published open-access |
| **Consent Type** | Broad consent for health research (per original study protocols) |
| **Ethical Approval** | Original studies had individual IRB/ethics approvals; harmonised dataset falls under HE2AT Center ethics (UCT HREC) |
| **IRB/Ethics Reference** | UCT Human Research Ethics Committee — HE2AT Center umbrella protocol |
| **Geographic Restriction** | None |
| **Time Limit on Use** | None |
| **Publication Moratorium** | None |
| **User Authentication Required** | Yes |
| **Collaboration Required** | Yes — co-authorship or acknowledgement expected |
| **DUOS Consent Code** | DS-HIV (Disease-Specific: HIV-related research) |
| **Data Use Agreement Required** | Yes |

---

## FORM 4: Demographics and Health Data

### Dataset Overview

| Field | Value |
|-------|-------|
| **Total Participants** | 12,804 unique individuals |
| **Total Visit Records** | 62,154 |
| **Geographic Location** | Johannesburg, Gauteng, South Africa |
| **Data Collection Period** | October 2003 – December 2021 |
| **Number of Source Studies** | 14 studies (17 study arms) |

### Age

| Field | Value |
|-------|-------|
| **Age Range** | 13 – 76 years |
| **Mean Age** | 35.0 years |
| **Median Age** | 34.0 years |
| **Age Available** | 39,129 records (63.0%) |
| **Age Groups** | <18: 155; 18-24: 5,541; 25-34: 14,824; 35-44: 11,802; 45-54: 5,204; 55-64: 1,483; 65+: 120 |

### Sex

| Category | Count | % of available |
|----------|-------|----------------|
| Female | 28,565 | 65.3% |
| Male | 14,927 | 34.1% |
| Unknown | 240 | 0.5% |
| Not available | 1 | <0.1% |
| *(Missing)* | 18,421 | 29.6% of total |

### Race / Ethnicity

| Category | Count |
|----------|-------|
| African Heritage | 10,693 |
| Black | 2,634 |
| Asian | 1,173 |
| Caucasian | 301 |
| White | 60 |
| Coloured | 58 |
| Other | 38 |
| *(Missing)* | 47,197 (75.9%) |

### HIV Status

| Category | Count | % of available |
|----------|-------|----------------|
| Positive | 42,174 | 70.4% |
| Unknown | 16,386 | 27.3% |
| Negative | 1,359 | 2.3% |
| *(Missing)* | 2,235 | 3.6% of total |

### Antiretroviral Therapy (ART)

| Category | Count |
|----------|-------|
| On ART (1) | 32,218 |
| Not on ART (0) | 9,837 |
| *(Missing)* | 20,091 (32.3%) |

### WHO Glycemic Classification (subset, n=981)

| Category | Count |
|----------|-------|
| Normal Glucose Tolerance (NGT) | 721 |
| Type 2 Diabetes (T2D) | 120 |
| Impaired Glucose Tolerance (IGT) | 112 |
| Impaired Fasting Glucose (IFG) | 28 |

### Health Data Domains

| Domain | Variables | Total Observations | Key Variables |
|--------|-----------|-------------------|---------------|
| Cardiovascular / Vitals | 5 | 157,100 | systolic_bp, diastolic_bp, heart_rate, respiratory_rate, oxygen_saturation |
| HIV / Immunological | 4 | 89,462 | cd4_count, viral_load, log10_viral_load, viral_load_undetectable |
| Hematology | 6 | 73,866 | hemoglobin, hematocrit, mcv, mch, mchc, rdw |
| Renal / Electrolytes | 6 | 43,380 | creatinine, creatinine_clearance, bun, sodium, potassium, calcium |
| Hepatic | 4 | 38,818 | alt, ast, total_bilirubin, albumin |
| Body Composition | 9 | 38,387 | bmi, weight_kg, height_m, waist/hip, DXA (fat/lean mass), grip_strength |
| Lipid Panel | 4 | 29,093 | total_cholesterol, hdl, ldl, triglycerides |
| Metabolic | 4 | 10,358 | fasting_glucose, fasting_insulin, hba1c, vitamin_d |
| Inflammatory | 1 | 955 | hs_crp |

### Disease / Condition Focus

| Condition | Relevance |
|-----------|-----------|
| HIV/AIDS | Primary — 70.4% HIV-positive; CD4, viral load, ART status tracked |
| Heat-related illness | Primary — temperature exposure linked to biomarker changes |
| Cardiometabolic disease | Secondary — BP, glucose, lipids, BMI measured |
| Renal disease | Secondary — creatinine, eGFR derivable |
| Hepatic disease | Secondary — ALT, AST, bilirubin measured |
| Diabetes / Pre-diabetes | Secondary — fasting glucose, HbA1c, WHO glycemic status |
| Sarcopenia / Body composition | Secondary — DXA measures, grip strength |

### Source Studies

| Study | Period | Records | Patients | Primary Focus |
|-------|--------|---------|----------|---------------|
| JHB_VIDA_007 | Jun 2020 – Dec 2021 | 15,581 | 4,254 | Vaccine/infectious disease trial |
| JHB_Aurum_009 | Jan 2013 – Dec 2015 | 12,575 | 2,550 | HIV treatment trial |
| JHB_Ezin_002 | Jan 2017 – Mar 2020 | 11,059 | 1,053 | HIV treatment trial |
| JHB_WRHI_001 | Jul 2012 – Nov 2015 | 10,044 | 1,072 | Women's reproductive health / HIV |
| JHB_ACTG_019 | May 2005 – May 2010 | 2,594 | 100 | AIDS Clinical Trials Group |
| JHB_ACTG_015 | Nov 2005 – Jan 2011 | 2,364 | 153 | AIDS Clinical Trials Group |
| JHB_WRHI_003 | Jun 2016 – May 2018 | 2,235 | 300 | Women's reproductive health / HIV |
| JHB_SCHARP_006 | (dates unavailable) | 1,359 | 451 | Statistical HIV analysis |
| JHB_DPHRU_053 | Jan 2017 – Jul 2018 | 1,013 | 1,013 | Developmental pathways cohort |
| JHB_DPHRU_013 | Feb 2011 – Jun 2013 | 784 | 247 | Developmental pathways cohort |
| JHB_ACTG_016 | Oct 2007 – Jul 2010 | 766 | 108 | AIDS Clinical Trials Group |
| JHB_VIDA_008 | Apr 2020 – Aug 2021 | 550 | 550 | Vaccine/infectious disease trial |
| JHB_EZIN_025 | Sep 2020 – Jul 2021 | 489 | 489 | HIV treatment trial |
| JHB_SCHARP_004 | (dates unavailable) | 403 | 401 | Statistical HIV analysis |
| JHB_ACTG_018 | May 2011 – May 2012 | 240 | 240 | AIDS Clinical Trials Group |
| JHB_ACTG_021 | Mar 2006 – Jun 2007 | 78 | 78 | AIDS Clinical Trials Group |
| JHB_ACTG_017 | Oct 2003 – Aug 2004 | 20 | 20 | AIDS Clinical Trials Group |

---

## FORM 5: Omics Dataset

**Status: Not Applicable**

This dataset does not contain omics data (no genomic, transcriptomic, proteomic, or metabolomic data). Mark as N/A or leave incomplete.

---

## FORM 6: Geospatial Dataset

| Field | Value |
|-------|-------|
| **Geospatial Data Included** | Yes |
| **Spatial Resolution** | Point locations (study site coordinates) |
| **Coordinate Reference System** | WGS84 (EPSG:4326) |
| **Latitude** | -26.2041 |
| **Longitude Range** | 27.9394 – 28.0473 (two Johannesburg study site clusters) |
| **Geographic Coverage** | Johannesburg, Gauteng Province, South Africa |
| **City** | Johannesburg |
| **Country** | South Africa |
| **Admin Level** | City-level (all sites within Greater Johannesburg) |
| **Spatial Variables in Dataset** | latitude, longitude, city |
| **Climate Data Source** | ERA5 reanalysis (ECMWF), 0.25° grid resolution |
| **Climate Grid Cell** | Johannesburg (-26.2°S, 28.0°E) |
| **Climate Variables** | 16 variables: daily mean/max/min temperature, diurnal range, lagged temperature averages (1-30 days), 7-day and 30-day variability, apparent temperature, temperature anomaly, heat wave indicator, heat stress category |
| **Climate Temporal Resolution** | Daily |
| **Climate Temporal Coverage** | Matched to visit dates within 2003–2021 |
| **Socioeconomic Spatial Data** | GCRO Quality of Life survey ward-level indicators statistically matched to clinical records |
| **Mapping/Visualization Tools** | Compatible with QGIS, R (sf/terra), Python (geopandas) |

---

## FORM 7: Image Dataset

**Status: Not Applicable**

This dataset does not contain image or video data. Mark as N/A or leave incomplete.

---

## FORM 8: Pathogen Dataset

**Status: Not Applicable**

This dataset does not contain pathogen/microbiology data (no culture, PCR, or sequencing results). HIV status is recorded as a clinical covariate, not from pathogen characterisation. Mark as N/A or leave incomplete.

---

## ZENODO PUBLICATION — Ready-to-Use Fields

### Title
Johannesburg Clinical Analysis Ready Dataset (RP2): Harmonised Multi-System Biomarker and Climate Data for Heat-Health Analysis

### Creators
1. Craig Parker — Climate System Analysis Group (CSAG), University of Cape Town, South Africa (ORCID: add yours)
2. *(Add co-investigators)*

### Description
Harmonised clinical biomarker dataset comprising 62,154 visit records from 12,804 unique participants across 14 independent Johannesburg, South Africa clinical studies (2003–2021). The dataset integrates multi-system biomarkers spanning 8 organ systems (renal, metabolic, cardiovascular, inflammatory, immunological, hepatic, lipid, body composition — 96 harmonised variables) with ERA5 reanalysis temperature data (daily values and lagged moving averages from 1 to 30 days) and Gauteng City-Region Observatory (GCRO) Quality of Life survey socioeconomic indicators linked via statistical matching.

Created for the HE2AT Center study: "Heat, Vulnerability, and the Body: Multi-System Biomarker Signatures of Temperature Exposure in Johannesburg Revealed Through Explainable Machine Learning" (Parker et al., target journal: Environmental Data Science).

Participants are predominantly HIV-positive (70.4%), aged 13–76 years (mean 35), from clinical trials and cohort studies conducted by ACTG, Aurum, DPHRU, Ezintsha, SCHARP, VIDA, and WRHI at Johannesburg sites. The dataset is pseudonymised, quality-controlled, and ALCOA+ compliant (Attributable, Legible, Contemporaneous, Original, Accurate).

Dataset version 3.0 (final). Available as CSV format with accompanying data dictionary/codebook.

### Keywords
heat-health; biomarkers; climate-health; Johannesburg; South Africa; ERA5; clinical data; harmonised dataset; HE2AT Center; temperature exposure; explainable machine learning; SHAP; XGBoost; multi-system biomarkers; HIV; urban health; environmental epidemiology; DLNM

### Resource Type
Dataset

### License
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
*(or Restricted Access if data cannot be shared publicly)*

### Funding
- NIH U54 TW 012083 — HE2AT Center (Funder: National Institutes of Health)

### Related Identifiers
| Identifier | Relation |
|-----------|----------|
| http://gitlab.csag.uct.ac.za/he2at/heat-analysis-optimized | Is supplemented by (this code) |
| *(Parker et al. DOI when published)* | Is supplement to |

### Communities
- HE2AT Center
- DS-I Africa

### Version
3.0

### Language
English

### Dates
- Collected: 2003-10-20 / 2021-12-09
- Created: 2026-03-10

### Suggested Files to Upload
1. `RP2_Data_Dictionary_Codebook.csv` — Variable-level codebook (96 variables, 18 metadata fields)
2. `JOHANNESBURG_CLINICAL_ANALYSIS_READY.csv` — Primary dataset (if data sharing permitted)
3. `eLwazi_REDCap_all_forms.md` — This metadata document

### Suggested Citation
Parker, C. (2026). Johannesburg Clinical Analysis Ready Dataset (RP2): Harmonised Multi-System Biomarker and Climate Data for Heat-Health Analysis (Version 3.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
