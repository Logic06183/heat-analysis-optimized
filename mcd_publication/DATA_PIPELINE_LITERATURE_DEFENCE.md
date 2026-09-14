# Data Pipeline Literature Defence

**For**: Parker et al. (2026), "Heat, Vulnerability, and the Body: Multi-System
Biomarker Signatures of Temperature Exposure in Johannesburg Revealed Through
Explainable Machine Learning" — target: *Environmental Data Science*

**Purpose**: Citation-backed defence of every methodological choice in the
data pipeline: harmonization, biomarker selection, climate variables, demographics,
and ML methodology.

---

## 1. Multi-Study Clinical Data Harmonization

### The Challenge
14 clinical trial datasets (HIV/NCD studies, 2003–2021) with heterogeneous
variable names, coding schemes, and biomarker panels → harmonized into a single
analysis-ready dataset of 9,745 patients × 98 features.

### Literature Support

- **Fortier I et al. (2017)**. "Maelstrom Research guidelines for rigorous
  retrospective data harmonization." *Int J Epidemiol* 46(1):103-105. — The
  definitive reference for retrospective multi-study harmonization. Introduces
  the "DataSchema" concept for target harmonized variables. Our approach follows
  these guidelines: define target schema → map study variables → validate.
  https://academic.oup.com/ije/article/46/1/103/2617181

- **Doiron D et al. (2021)**. "Data harmonization and data pooling from cohort
  studies: a practical approach." *BMC Med Res Methodol*. PMC8631396. — Practical
  guidance covering variable-level documentation, processing algorithms, and
  quality checks. Our per-study harmonization notebooks follow this framework.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8631396/

- **Ventura-Cots M et al. (2023)**. "Large-Scale Data Harmonization Across
  Prospective Studies." *Am J Epidemiol* 192(12):2033. — Demonstrates large-scale
  harmonization with variable mapping and quality assurance steps.
  https://academic.oup.com/aje/article/192/12/2033/7219191

- **Gaye A et al. (2015)**. "Harmonising and linking biomedical and clinical data
  across disparate data archives." *Eur J Hum Genet*. — Addresses cross-biobank
  harmonization with different data structures.
  https://www.nature.com/articles/ejhg2015165

### ALCOA+ Data Integrity
Our harmonization follows ALCOA+ principles (Attributable, Legible,
Contemporaneous, Original, Accurate + Complete, Consistent, Enduring,
Available), originating from FDA GxP/GCP regulatory guidance. Each
harmonization step is documented in per-study Jupyter notebooks with
checksums for reproducibility.

---

## 2. Biomarker Selection — Multi-Organ Heat-Health Pathways

### Conceptual Framework
27 biomarkers across 8 organ systems, selected based on established
heat-health pathways. Each biomarker has a known physiological mechanism
linking temperature exposure to organ-specific effects.

### Comprehensive References

**Multi-system overview**:
- **Ding R et al. (2025)**. "Heat Stress-Mediated Multi-Organ Injury:
  Pathophysiology and Treatment Strategies." *Comprehensive Physiology* (Wiley).
  — Covers CNS, intestinal, hepatic, cardiovascular, and skeletal muscle heat
  injury. HSP pathways, inflammatory cascades, oxidative stress.
  https://onlinelibrary.wiley.com/doi/full/10.1002/cph4.70012

- **Bouchama A et al. (2022)**. "Biomarkers of heatstroke-induced organ injury
  and repair." PMC9529995. — Identifies specific biomarkers (eHSP72, SDC-1,
  vWF) as prognostic indicators of multi-organ heat damage.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9529995/

### Organ-Specific Evidence

| System | Biomarkers | Pathway | Key Citation |
|--------|-----------|---------|-------------|
| **Renal** | Creatinine, eGFR | Heat → dehydration → AKI → CKD | Wen et al. (2024) *Lancet Planet Health* |
| **Metabolic** | Glucose, HbA1c, insulin | Heat → HSP72 → insulin sensitivity | Pallubinsky et al. (2020) *Acta Physiol* |
| **Cardiovascular** | SBP, DBP, heart rate | Heat → cardiac load → arrhythmia | Cheng et al. (2022) *Lancet Planet Health* |
| **Inflammatory** | WBC, neutrophils | Heat → systemic inflammation | Bouchama et al. (2022) |
| **Immunological** | CD4, viral load | Heat → immune modulation (HIV) | Berger et al. (2021) PMC7810285 |
| **Hepatic** | ALT, AST, bilirubin | Heat → hepatocyte injury | Dong et al. (2022) *Crit Care* |
| **Lipid** | Cholesterol, HDL, LDL | Cold → lipid increase; heat → HDL decrease | Halonen et al. (2011) *Environ Res* |
| **Body composition** | BMI, waist circ | Obesity → impaired thermoregulation | Folkerts et al. (2024) |

**Renal**:
- **Wen B et al. (2024)**. "Ambient heat exposure and kidney function in patients
  with CKD." *Lancet Planetary Health*. — Each additional year with heat index
  ≥30°C → eGFR decline of -0.62%/year.
  https://www.thelancet.com/journals/lanplh/article/PIIS2542-5196(24)00026-3/fulltext
- **Glaser J et al. (2016)**. "Occupational Heat Stress and Kidney Health."
  PMC5733743. — Heat-dehydration-renal stress pathway.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5733743/

**Cardiovascular**:
- **Cheng J et al. (2022)**. "Heat exposure and cardiovascular health outcomes."
  *Lancet Planetary Health*. — 1°C increase → 2.1% increase in CV mortality.
  https://www.thelancet.com/journals/lanplh/article/PIIS2542-5196(22)00117-6/fulltext
- **Ebi KL et al. (2024)**. "Heat and Cardiovascular Mortality." *Circulation
  Research*. — Mechanisms: cardiac load, BP changes, prothrombotic conditions.
  https://www.ahajournals.org/doi/10.1161/CIRCRESAHA.123.323615

**Metabolic**:
- **Pallubinsky H et al. (2020)**. "Passive exposure to heat improves glucose
  metabolism." *Acta Physiologica*. PMC7379279. — HSP72 pathway in insulin
  sensitivity enhancement.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7379279/
- **Blauw LL et al. (2023)**. "Diabetes and climate change." PMC10039694. —
  Seasonal HbA1c patterns; heat-related hypoglycemia risk.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10039694/

**Hepatic**:
- **Dong Z et al. (2022)**. "Pathogenesis and therapeutic strategies of heat
  stroke-induced liver injury." *Critical Care*. — Direct thermal hepatocyte
  injury; AST/ALT elevation as heat stroke complication.
  https://ccforum.biomedcentral.com/articles/10.1186/s13054-022-04273-w

**Lipid**:
- **Halonen JI et al. (2011)**. "Outdoor temperature associated with serum HDL
  and LDL." *Environ Res*. PMC4437587. — HDL decreased -1.76% per 5°C increase;
  LDL increased 1.74% per 5°C increase.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4437587/

**HIV/Immune**:
- **Berger E et al. (2021)**. "The Synergistic Relationship Between Climate
  Change and the HIV/AIDS Epidemic." PMC7810285.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7810285/
- **Lancet HIV (2024)**. "Effect of climate change on the HIV response."
  https://www.thelancet.com/journals/lanhiv/article/PIIS2352-3018(24)00009-2/fulltext
- **Lancet Countdown Africa (2025)**. Climate-health in Africa, including
  disproportionate burden on immunocompromised populations.
  https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(25)02174-9/abstract

---

## 3. Climate Variable Selection (ERA5 + DLNM Lags + Heat Indices)

### ERA5 Reanalysis
- **Hersbach H et al. (2020)**. "The ERA5 global reanalysis." *QJRMS*
  146(730):1999-2049. — Definitive ERA5 reference. Hourly, 31km resolution.
  https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3803

- **Mistry MN et al. (2022)**. "Comparison of weather station and climate
  reanalysis data for modelling temperature-related mortality." *Sci Rep*. —
  ERA5 temperature compares well to station observations with similar risk
  estimates.
  https://www.nature.com/articles/s41598-022-09049-4

### DLNM Lag Structure (0–21 days)
- **Gasparrini A (2010)**. "Distributed lag non-linear models." *Stat Med*. —
  Foundational DLNM paper introducing cross-basis framework.
  https://onlinelibrary.wiley.com/doi/10.1002/sim.3940

- **Gasparrini A et al. (2015)**. "Mortality risk attributable to high and low
  ambient temperature." *The Lancet* 386:369-375. — Landmark study: 74 million
  deaths, 384 locations, 13 countries. Uses 0–21 day lag window (same as ours).
  7.71% of mortality attributable to non-optimum temperature.
  https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(14)62114-0/fulltext

- **Gasparrini A et al. (2011)**. "Distributed Lag Linear and Non-Linear Models
  in R: The Package dlnm." *J Stat Softw*. PMC3191524.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3191524/

### Heat Stress Indices
- **Jendritzky G et al. (2012)**. "UTCI — why another thermal index?" *Int J
  Biometeorol* 56(3):421-428. — UTCI methodology.
  https://pubmed.ncbi.nlm.nih.gov/22187087/

- **Nairn JR, Fawcett RJB (2015)**. "The Excess Heat Factor: A Metric for
  Heatwave Intensity." *Int J Environ Res Public Health* 12(1):227. PMC4306859.
  — Defines EHF methodology. Supports our EHF and 90th percentile threshold.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4306859/

- **Di Napoli C et al. (2021)**. "Evaluation of ERA5-based UTCI on mortality
  data in Europe." *Environ Res*. — Validates ERA5-derived thermal indices
  against mortality.
  https://www.sciencedirect.com/science/article/pii/S0013935121005211

---

## 4. Demographics as Confounders/Effect Modifiers

### Age and Sex as Effect Modifiers
- **Basu R (2009)**. "Is the association between temperature and mortality
  modified by age, gender and socio-economic status?" *Environ Res*. — Directly
  examines age, sex, SES as effect modifiers. Clear increasing harmful effect
  with age; females more susceptible.
  https://pubmed.ncbi.nlm.nih.gov/20569969/

- **Folkerts MA et al. (2024)**. "Sex differences in mortality associated with
  heatwaves: a systematic review and meta-analysis." PMC11516269. — Most studies
  find higher heat-related mortality risk for women.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11516269/

### GroupKFold Cross-Validation (Preventing Data Leakage)
- **Rokem A et al. (2024)**. "The effects of data leakage on connectome-based
  machine learning models." PMC10793416. — Demonstrates quantitatively how
  failure to group repeated measures leads to inflated performance. Directly
  motivates GroupKFold for our repeated clinical measurements.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10793416/

---

## 5. XGBoost-SHAP Methodology

- **Chen T, Guestrin C (2016)**. "XGBoost: A Scalable Tree Boosting System."
  *KDD 2016*, 785-794. — Foundational XGBoost paper.
  https://dl.acm.org/doi/10.1145/2939672.2939785

- **Lundberg SM, Lee SI (2017)**. "A Unified Approach to Interpreting Model
  Predictions." *NeurIPS 30*, 4766-4777. — Foundational SHAP paper.
  https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions

- **Janzing D et al. (2020)**. "Feature relevance quantification in explainable
  AI: A causal problem." *AISTATS*. — Causal justification for interventional
  (not observational) SHAP values with correlated features. Supports our use of
  interventional TreeSHAP for correlated temperature lag features.
  https://proceedings.mlr.press/v108/janzing20a.html

- **Kim S et al. (2025)**. "Explainable Machine Learning for Heat-Related
  Illness Prediction: An XGBoost-SHAP Approach." *Bioengineering* 12(11):1276.
  — Direct precedent: XGBoost-SHAP for heat-health prediction.
  https://www.mdpi.com/2306-5354/12/11/1276

---

## Quick Reference: Highest-Impact Citations by Section

| Section | Must-Cite | Journal | Impact |
|---------|-----------|---------|--------|
| Harmonization | Fortier et al. (2017) | *Int J Epidemiol* | Definitive guidelines |
| Renal biomarkers | Wen et al. (2024) | *Lancet Planet Health* | Heat → eGFR decline |
| CV biomarkers | Cheng et al. (2022) | *Lancet Planet Health* | 2.1% CV mortality per °C |
| Climate/ERA5 | Hersbach et al. (2020) | *QJRMS* | ERA5 reference paper |
| DLNM lags | Gasparrini et al. (2015) | *The Lancet* | 74M deaths, 0-21 day lag |
| Demographics | Basu (2009) | *Environ Res* | Age/sex/SES as modifiers |
| XGBoost | Chen & Guestrin (2016) | *KDD* | Foundational |
| SHAP | Lundberg & Lee (2017) | *NeurIPS* | Foundational |
| Interventional SHAP | Janzing et al. (2020) | *AISTATS* | Correlated features |
| HIV-heat | Lancet HIV (2024) | *Lancet HIV* | Climate-HIV interaction |
| GroupKFold | Rokem et al. (2024) | PMC | Data leakage prevention |
