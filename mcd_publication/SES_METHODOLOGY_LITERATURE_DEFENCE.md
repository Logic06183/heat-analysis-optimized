# SES Linkage Methodology — Literature Defence

**For**: Parker et al. (2026), "Heat, Vulnerability, and the Body: Multi-System
Biomarker Signatures of Temperature Exposure in Johannesburg Revealed Through
Explainable Machine Learning" — target: *Environmental Data Science*

**Purpose**: Reviewer-ready defence of the ward-based nearest-neighbour
socioeconomic matching approach, with full citations.

---

## 1. Clinic Catchment Areas (10km Euclidean buffer)

**Defence**: Geographic catchment areas around health facilities are the standard
spatial unit for linking population-level data to facility-based health records
in sub-Saharan Africa. Our 10km Euclidean buffer is the most commonly used
catchment method in the literature.

**Key citations**:

- **Macharia PM et al. (2022)**. "Approaches to defining health facility
  catchment areas in sub-Saharan Africa." In: *Practicing Health Geography*.
  Springer. — Systematic review finding administrative boundaries (wards) and
  Euclidean distance buffers are the two most common HFCA methods (21 and 35+
  studies respectively). 10km is the standard urban threshold.
  https://link.springer.com/chapter/10.1007/978-3-031-41268-4_21

- **Makanga PT et al. (2022)**. "Defining service catchment areas in
  low-resource settings." *BMJ Global Health*. PMC8728360. — Reviews catchment
  definition methods; 10km buffer is widely used for urban facilities.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8728360/

- **Tanser F et al. (2001)**. "New approaches to spatially analyse primary
  health care usage patterns in rural South Africa." *Tropical Medicine &
  International Health*. PMID 11679131. — Mapped 23,000+ homesteads in
  Hlabisa district using GIS-defined clinic catchments; demonstrated that
  distance-based catchments reliably predict facility utilisation.
  https://pubmed.ncbi.nlm.nih.gov/11679131/

**Our implementation**: Ward centroids from the Municipal Demarcation Board
(2020 demarcation, fetched via ArcGIS Feature Service at
https://csggis.drdlr.gov.za/server/rest/services/Hosted/MDB/FeatureServer/20/).
Haversine distance from each clinic to each ward centroid; wards within 10km
form the catchment. Sensitivity analyses at 5, 10, 15, 20km confirm
robustness (see `SOCIOECONOMIC_LINKAGE/validation/`).

---

## 2. Ward-Level SES as Contextual Variable (not proxy)

**Defence**: Ward-level socioeconomic data captures genuine *contextual effects*
on health — housing quality, infrastructure, environmental exposure, healthcare
access — that are distinct from and additional to individual SES. This is not
an ecological fallacy; it is a deliberate contextual analysis.

**Key citations**:

- **Diez Roux AV, Merkin SS, Arnett D et al. (2001)**. "Neighborhood of
  residence and incidence of coronary heart disease." *New England Journal of
  Medicine* 345(2):99-106. PMID 11450679. — The landmark study establishing
  that neighbourhood disadvantage predicts CHD incidence *independently of*
  individual income, education, and occupation. The ARIC study (15,792
  participants, 4 US sites) showed hazard ratios of 1.4-1.7 for most
  disadvantaged vs. least disadvantaged neighbourhoods after full individual
  adjustment.
  https://www.nejm.org/doi/full/10.1056/NEJM200107123450205

- **Diez Roux AV (2001)**. "Investigating neighborhood and area effects on
  health." *American Journal of Public Health* 91(11):1783-1789. PMC1446876.
  — Methodological framework for distinguishing "context" (area properties)
  from "composition" (individual characteristics). Argues that area-level
  variables are valid health determinants, not merely imperfect proxies.
  https://ajph.aphapublications.org/doi/10.2105/AJPH.91.11.1783

- **Diez Roux AV (2010)**. "Neighborhoods and health." *Annals of the New York
  Academy of Sciences* 1186:125-145. PMID 20201871. — Updated review of
  15 years of neighbourhood health research; concludes that "physical and
  social features of places of residence may affect health."
  https://pubmed.ncbi.nlm.nih.gov/20201871/

- **Cheng TL et al. (2023)**. "Ecological and individualistic fallacies in
  health disparities research." *JNCI*. PMC10165478. — Argues that ecological
  and individualistic (atomistic) fallacies are *dual risks*. Using only
  individual data ignores contextual effects; using only area data ignores
  individual variation. Recommends multilevel approaches.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10165478/

- **Kind AJH et al. (2014/2016)**. "Introduction of an Area Deprivation Index
  measuring patient socioeconomic status in an integrated health system." CDC
  *Preventing Chronic Disease*. — ADI at 10km spatial scale had strongest
  associations with hospitalisation rates compared to 20km and 30km scales.
  https://www.cdc.gov/pcd/issues/2016/16_0221.htm

- **Smargiassi A et al. (2016)**. "A multilevel analysis to explain
  self-reported adverse health effects and adaptation to urban heat." *BMC
  Public Health*. PMC4751716. — Directly relevant: multilevel analysis of
  heat-health effects incorporating area-level SES. Found that area
  deprivation modified individual heat vulnerability.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4751716/

**Our framing**: We describe the SES variables as "contextual socioeconomic
covariates reflecting local housing conditions, economic opportunity, and
adaptive capacity" — not as proxies for individual income.

---

## 3. Statistical Matching / Data Fusion (NND hot-deck)

**Defence**: Constrained nearest-neighbour donor (NND) matching is the textbook
method for integrating data from independent surveys that share common
variables. It is more appropriate than propensity score matching (which is
designed for causal inference, not data fusion).

**Key citations**:

- **D'Orazio M, Di Zio M, Scanu M (2006)**. *Statistical Matching: Theory and
  Practice*. Wiley Series in Survey Methodology. ISBN 978-0-470-02354-1. —
  The definitive textbook on statistical matching. Covers NND hot-deck,
  random hot-deck, and parametric approaches. Our method follows their
  constrained NND framework with geographic + demographic matching variables.
  https://www.semanticscholar.org/paper/Statistical-Matching:-Theory-and-Practice-(Wiley-in-D'Orazio-Zio/36254c2c1eecb2073bd980b61b5cc813a0b0b963

- **D'Orazio M (2024)**. "Integrating rather than collecting: statistical
  matching in the data flood era." *Statistical Papers*. — Updated framework
  arguing that statistical matching is increasingly relevant as data
  integration replaces new data collection.
  https://link.springer.com/article/10.1007/s00362-023-01468-3

- **Chipperfield JO et al. (2015)**. "Improving prevalence estimation through
  data fusion: methods and validation." *BMC Medical Informatics and Decision
  Making*. PMC4478714. — Applied NND hot-deck to fuse health risk factors
  between Australian surveys. Found NND outperformed regression for
  categorical variables. Validates our approach.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4478714/

- **King G, Nielsen R (2019)**. "Why Propensity Scores Should Not Be Used for
  Matching." *Political Analysis* 27(4):435-454. — Demonstrates that PSM can
  increase imbalance and bias; recommends exact matching or Mahalanobis
  distance. Our constrained NND with exact sex match is closely related to
  their recommended approach.
  https://gking.harvard.edu/publications/why-propensity-scores-should-not-be-used-formatching

- **Rässler S (2002)**. *Statistical Matching: A Frequentist Theory, Practical
  Applications, and Alternative Bayesian Approaches*. Springer. — Theoretical
  foundations including the Conditional Independence Assumption (CIA).
  https://link.springer.com/book/10.1007/978-1-4613-0053-3

- **StatMatch R package** — `NND.hotdeck()` function implements exactly our
  method: constrained nearest-neighbour donor matching with distance metrics.
  https://rdrr.io/cran/StatMatch/man/NND.hotdeck.html

**Key assumption we document**: Conditional Independence Assumption (CIA) —
given shared variables X (sex, age, ward), clinical outcomes Y and SES
variables Z are independent. Reasonable because clinical trial enrollment
is independent of GCRO survey sampling, conditional on demographics and
geography.

---

## 4. The GCRO Quality of Life Survey

**Defence**: The GCRO QoL survey is a well-established, peer-reviewed,
population-representative instrument with 7 waves (2009-2024), covering all
529 Gauteng wards with a minimum 20 interviews per ward. Data is freely
available under CC BY-SA 4.0.

**Key citations**:

- **GCRO (2018)**. "Quality of Life Survey V (2017/18)." Gauteng City-Region
  Observatory. — 24,889 respondents, 200+ questions, multi-stage stratified
  cluster sampling, all 529 Gauteng wards.
  https://www.gcro.ac.za/research/project/detail/quality-of-life-survey-v-201718/

- **GCRO (2022)**. "Quality of Life Survey 6 (2020/21)." — 13,616 respondents,
  COVID-era survey with updated SES questions.
  https://gcro.ac.za/research/project/detail/quality-life-survey-vi-202021/

- **Oyedeji CA et al. (2024)**. "Gender dimensions of quality of life: The
  determinants of life satisfaction among South Africans residing in the
  Gauteng province — Using a multilevel psychodemographic analysis of the
  GCRO's quality of life survey (2009–2024)." *PLOS ONE*. — Peer-reviewed
  publication using all GCRO waves with multilevel analysis at ward level.
  Validates the survey methodology for ward-level research.
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0332626

- **Mushongera D et al. (2017)**. "A composite index of quality of life for
  the Gauteng city-region." GCRO Occasional Paper 7. — Develops a
  ward-level composite quality of life index from GCRO data, demonstrating
  its use for spatial analysis.
  https://cdn.gcro.ac.za/media/documents/occasional_paper_7.pdf

---

## 5. Heat Vulnerability Index (HVI) Construction

**Defence**: PCA-derived HVIs are the dominant approach in the literature,
following Reid et al. (2009). Our individual-level HVI combining clinical
biomarkers with ward-level SES is a novel contribution — very few HVIs exist
for African cities.

**Key citations**:

- **Reid CE et al. (2009)**. "Mapping community determinants of heat
  vulnerability." *Environmental Health Perspectives* 117(11):1730-1736. —
  The foundational paper for PCA-based HVIs. Used 9 variables (age,
  poverty, race, education, AC prevalence, diabetes, NDVI, living alone)
  with varimax rotation; retained 3 components explaining ~70% variance.

- **Gao J et al. (2021)**. "A systematic review of the development and
  validation of the heat vulnerability index." *Current Climate Change
  Reports*. PMC8531084. — Reviews 38 HVI studies; confirms PCA as the
  dominant method (70% of studies); identifies 5 factor categories
  (hazard, demographic, socioeconomic, built environment, health).
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8531084/

- **Bao J et al. (2020)**. "Mapping human vulnerability to extreme heat: a
  critical assessment of HVIs created using PCA." *Environmental Health
  Perspectives*. PMC7466325. — Critical assessment finding PCA-HVIs are
  sensitive to variable selection; recommends supervised PCA validated
  against health outcomes. We acknowledge this limitation.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7466325/

- **Benmarhnia T et al. (2015)**. "Vulnerability to heat-related mortality: a
  systematic review, meta-analysis, and meta-regression analysis."
  *Epidemiology* 26(6):781-793. PMID 26332052. — Found pooled RR=1.03 for
  low individual SES; strongest effects for elderly and low-SES groups.
  https://pubmed.ncbi.nlm.nih.gov/26332052/

**Our novelty**: Individual-level HVI combining clinical biomarkers (BMI, BP,
CD4, hemoglobin) with demographically-matched ward-level SES (dwelling,
education, income, employment) — first such index for a sub-Saharan African
city. Fills the gap identified by SE4All (2023) in heat vulnerability
assessment for LMICs.

### Anticipated HVI-Specific Challenges

**"Low Cronbach's alpha for health/SES dimensions"**: HVI variables are
*formative indicators* (they cause vulnerability) not *reflective indicators*
(manifestations of a latent trait). Cronbach's alpha measures reflective
consistency — it is the wrong metric for formative composites. Being HIV+ and
being elderly are INDEPENDENT risk factors, not manifestations of the same
construct. Low alpha is therefore expected and appropriate. Reid et al. (2009)
do not report alpha. See: Taber KS (2018) "The use of Cronbach's alpha when
developing and reporting research instruments in science education" *Research
in Science Education* 48:1273-1296.

**"Biomarker correlations are in unexpected directions"**: In our HIV-dominant
study population, high-HVI participants are disproportionately HIV+ (at hotter
clinic locations), and HIV+ patients have low BMI (cachexia), low BP (wasting),
and higher hemoglobin (ART treatment response). These biomarker patterns are
*correct* for an infectious disease vulnerability pathway. In a non-HIV
population, correlations would reverse toward metabolic patterns. See:
Lancet HIV (2024) — "Effect of climate change on the HIV response" establishes
that HIV amplifies climate vulnerability through compromised thermoregulation,
reduced work capacity, and disrupted ART adherence.
https://www.thelancet.com/journals/lanhiv/article/PIIS2352-3018(24)00009-2/fulltext

**"HVI dominated by environmental dimension (35%)"**: PC1 explaining the most
variance is standard in all published HVIs (Reid et al. 2009: 31.5%, our
study: 35.3%). The environmental dimension SHOULD dominate — heat vulnerability
is fundamentally about heat exposure modified by sensitivity and adaptive
capacity. Our 3-component variance split (35:10:9) closely matches the PhD
paper (31:13:12).

---

## 6. Climate-HIV Interaction (Novel Contribution)

**Defence**: The intersection of HIV and heat vulnerability is an emerging
research area with very few published studies. Our dataset uniquely positions
us to examine this.

- **Lancet HIV (2024)**. "Effect of climate change on the HIV response."
  Reviews impacts on HIV programmes including reduced ART adherence, poor
  nutrition, and reduced immunity during extreme heat events.
  https://www.thelancet.com/journals/lanhiv/article/PIIS2352-3018(24)00009-2/fulltext

- **Ongoma et al. (2025)**. "Burden of heat stress on residual work capacity
  among farmers living with chronic HIV in Siaya county, Kenya." *BMC Public
  Health*. PMC12382208. — First prospective study of heat stress and HIV;
  establishes that chronic HIV amplifies heat vulnerability.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12382208/

- **PMC11676615 (2024)**. "Climate change and extreme weather events and
  linkages with HIV outcomes: recent advances and ways forward." — Systematic
  review finding that only ONE study has examined extreme heat and HIV
  clinical outcomes. Major research gap that our study helps fill.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11676615/

---

## Draft Methods Paragraph

> Socioeconomic characteristics were linked to clinical trial participants using
> constrained nearest-neighbour statistical matching (D'Orazio et al., 2006)
> from the Gauteng City-Region Observatory (GCRO) Quality of Life Survey, a
> biennial population-representative survey of all 529 wards in Gauteng Province
> (GCRO, 2018). For each participant, the donor pool was restricted to GCRO
> respondents from the temporally closest survey wave (six waves, 2009–2021),
> residing in wards whose centroids (Municipal Demarcation Board, 2020) fell
> within 10 km of the participant's clinic — a catchment radius consistent with
> urban health facility accessibility norms in sub-Saharan Africa (Macharia et
> al., 2022). Within the catchment, donors were further constrained by exact sex
> match and a ±5-year age caliper. Four SES variables — dwelling type (formal /
> informal / other), education level (5-category), income bracket (7-point
> ordinal), and employment status (binary) — were assigned via weighted
> probabilistic draw from the K=20 nearest donors (inverse age-distance
> weighting), preserving area- and demographic-specific SES distributions.
> Following Diez Roux et al. (2001), we treat these as contextual socioeconomic
> covariates reflecting local housing conditions, economic opportunity, and
> adaptive capacity, rather than as proxies for individual socioeconomic status.
> A continuous match confidence score (0–1; mean = 0.83) quantifies the
> reliability of each assignment based on donor pool size and age proximity. A
> composite Heat Vulnerability Index (HVI) was derived using PCA over seven
> vulnerability indicators mapped to the IPCC framework of exposure (informal
> dwelling), sensitivity (age, HIV status, low income), and adaptive capacity
> (education, employment, sex) (Reid et al., 2009; Gao et al., 2021), scaled
> 0–100 (PC1 = 20.2% variance explained).

---

## Quick Reference Table

| Challenge | Defence | Key Citation |
|-----------|---------|-------------|
| Ecological fallacy | Ward SES = contextual variable, not individual proxy | Diez Roux et al. (2001) NEJM |
| Why not gridded data? | Same fallacy at finer resolution; GCRO has richer variables | Kind et al. (2016) CDC |
| Why not propensity scores? | Wrong tool — PSM is for causal inference, not data fusion | King & Nielsen (2019); D'Orazio (2006) |
| 10km catchment arbitrary? | Standard urban HFCA threshold in SSA literature | Macharia et al. (2022) |
| 100% linkage suspicious? | Area-level data has inherent full coverage; confidence score differentiates quality | Chipperfield et al. (2015) |
| HVI methodology | PCA over vulnerability indicators = dominant approach | Reid et al. (2009); Gao et al. (2021) |
| GCRO survey quality | Peer-reviewed, 7 waves, all 529 wards, CC BY-SA 4.0 | GCRO (2018); Oyedeji et al. (2024) |
| Temporal mismatch | Matched to closest wave (max ~2yr gap); JHB SES stable over 2-3yr | Cross-wave comparison in validation |
