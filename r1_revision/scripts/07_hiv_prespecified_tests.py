"""07. Five pre-specified tests for an HIV-related temperature association (appendix SA15)

Question    Once cohort differences are removed, is there any sign that
            temperature relates to biomarkers differently in people living
            with HIV? The five tests and the decision rule were written down
            before any of them was run.
Data in     The visit-level analysis table and the published cluster labels.
Exposure    Primary: mean temperature over the 7 days before the visit.
            Secondary: mean of the lag 0, 1 and 3 day temperatures.
Design      H1, H2, H3 and H5 compare each person with themselves across
            visits (participant fixed effects), adjusting for calendar month
            and year, with standard errors clustered by participant. People
            with one visit do not contribute.
Tests
  H1  HIV x temperature interaction on each biomarker, in the three cohorts
      that enrolled people with and without HIV. Pooled across biomarkers.
  H2  Among people living with HIV: is a visit more likely to be viraemic
      (1,000 copies per mL or more) after warmer weeks?
      Pre-specified model: conditional logistic regression. Pre-specified
      fallback if it does not converge: linear probability model with
      participant fixed effects. It did not converge (see `H2` in the output),
      so the fallback is the reported result.
  H3  Among people living with HIV: log CD4 count against temperature.
  H4  Is Cluster 3 membership associated with HIV within cohorts?
      Cochran-Mantel-Haenszel test across cohorts, plus a permutation that
      shuffles HIV status within each cohort.
  H5  Among people living with HIV: does low CD4 (<350) modify the
      temperature association for haematocrit and haemoglobin?
Multiplicity Benjamini-Hochberg across the five primary p values.
Post hoc    Stress tests of H3, run after seeing it pass. They are labelled
            post hoc and can only weaken the result.
File out    r1_hiv_tests.json
Reported    SA15 and its table. H3 passed the threshold (q=0.068) but
            vanished when the first 90 days of follow-up were excluded.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.stats.contingency_tables import StratifiedTable

from r1_common import PARTICIPANT, SEED, load_visits, published_clusters, write_json

PRIMARY = "T_primary"
SECONDARY = "T_secondary"
MIXED_HIV_COHORTS = ["JHB_DPHRU_013", "JHB_DPHRU_053", "JHB_SCHARP_004"]
VIRAEMIA_THRESHOLD = 1000
LOW_CD4 = 350
N_PERMUTATIONS = 10000
CONVERGENCE_TOLERANCE = 1e-2  # largest allowed absolute score (gradient) at the solution
MIN_PARTICIPANTS = 30
MIN_OBSERVATIONS = 100


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

visits, biomarkers = load_visits()
BIOMARKER_COLUMNS = [b.column for b in biomarkers.values() if b.column and b.column in visits.columns]

assert visits["era5_temp_lag7d_c"].notna().mean() > 0.5, "7-day mean temperature must be available"
visits[PRIMARY] = visits["era5_temp_lag7d_c"]
visits[SECONDARY] = visits[["temp_lag_0d", "temp_lag_1d", "temp_lag_3d"]].mean(axis=1)

hiv_text = visits["hiv_status"].astype(str).str.lower()
visits["HIV"] = np.nan
visits.loc[hiv_text.isin(["positive", "1", "1.0", "true", "pos"]), "HIV"] = 1
visits.loc[hiv_text.isin(["negative", "0", "0.0", "false", "neg"]), "HIV"] = 0

visit_date = pd.to_datetime(visits["visit_date"])
visits["cohort"] = visits["study_source"].astype(str)
visits["month"] = visit_date.dt.month
visits["year"] = visit_date.dt.year
first_visit = visit_date.groupby(visits[PARTICIPANT]).transform("min")
visits["days_since_first_visit"] = (visit_date - first_visit).dt.days

# People living with HIV, with the three outcomes used in H2, H3 and H5.
plwh = visits[visits["HIV"] == 1].copy()
has_viral_load = plwh["viral_load"].notna()
has_cd4 = plwh["cd4_count"].notna()
plwh["viraemic"] = (plwh["viral_load"] >= VIRAEMIA_THRESHOLD).astype(float).where(has_viral_load)
plwh["log10_viral_load"] = np.log10(plwh["viral_load"].clip(lower=1))
plwh["log_cd4"] = np.log(plwh["cd4_count"].clip(lower=1))
plwh["low_cd4"] = (plwh["cd4_count"] < LOW_CD4).astype(float).where(has_cd4)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def month_dummies(data):
    return pd.get_dummies(data["month"], prefix="m", drop_first=True, dtype=float)


def within_person_regression(data, outcome, predictors, extra_covariates=None, person_trends=False):
    """Linear regression with participant fixed effects.

    Every variable has each participant's own mean subtracted, so only
    change within a person is used. Adjusts for calendar month and year.
    Standard errors are clustered by participant.

    extra_covariates  optional function: data -> DataFrame of more adjusters
    person_trends     also remove each participant's own linear time trend

    Returns None when there is too little data or a predictor has no
    within-person variation. Otherwise a dict with (estimate, SE, p) for each
    predictor and the sample size.
    """
    data = data.dropna(subset=[outcome] + predictors).copy()
    visits_per_person = data.groupby(PARTICIPANT)[outcome].transform("size")
    data = data[visits_per_person >= 2]
    if data[PARTICIPANT].nunique() < MIN_PARTICIPANTS or len(data) < MIN_OBSERVATIONS:
        return None

    blocks = [data[predictors].astype(float), month_dummies(data), data[["year"]].astype(float)]
    if extra_covariates is not None:
        blocks.append(extra_covariates(data))
    design = pd.concat(blocks, axis=1)
    design = design.loc[:, design.std() > 0]
    if any(p not in design.columns for p in predictors):
        return None

    person = data[PARTICIPANT]
    y = data[outcome] - data.groupby(PARTICIPANT)[outcome].transform("mean")
    design = design - design.groupby(person).transform("mean")

    if person_trends:
        time = data["days_since_first_visit"]
        time = time - time.groupby(person).transform("mean")
        time_squared_sum = (time**2).groupby(person).sum().replace(0, np.nan)

        def remove_person_trend(column):
            slope = (column * time).groupby(person).sum() / time_squared_sum
            return column - time * slope.reindex(person).fillna(0).values

        y = remove_person_trend(y)
        design = design.apply(remove_person_trend)

    design = design.loc[:, design.abs().sum() > 1e-9]
    fit = sm.OLS(y.values, design.values).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(person)[0]}
    )

    columns = list(design.columns)
    estimates = {}
    for p in predictors:
        if p in columns:
            i = columns.index(p)
            estimates[p] = (float(fit.params[i]), float(fit.bse[i]), float(fit.pvalues[i]))
    return {
        "estimates": estimates,
        "n_obs": int(len(data)),
        "n_part": int(data[PARTICIPANT].nunique()),
        "cohorts": sorted(data["cohort"].unique().tolist()),
    }


def one_coefficient(data, outcome, predictor, adjust_for=(), **options):
    """within_person_regression for one predictor of interest, plus optional adjusters."""
    fit = within_person_regression(data, outcome, [predictor] + list(adjust_for), **options)
    if fit is None:
        return None
    estimate, se, p = fit["estimates"][predictor]
    return {
        "est": estimate,
        "se": se,
        "p": p,
        "n_obs": fit["n_obs"],
        "n_part": fit["n_part"],
        "cohorts": fit["cohorts"],
    }


def pool_inverse_variance(estimates):
    """Fixed-effect (inverse-variance) pooled estimate from (estimate, SE) pairs."""
    usable = [(e, s) for e, s in estimates if np.isfinite(s) and s > 0]
    if not usable:
        return None
    values = np.array([e for e, _ in usable])
    weights = np.array([1 / s**2 for _, s in usable])
    pooled = float((weights * values).sum() / weights.sum())
    se = float(np.sqrt(1 / weights.sum()))
    return {
        "est": pooled,
        "se": se,
        "lo": pooled - 1.96 * se,
        "hi": pooled + 1.96 * se,
        "p": float(2 * stats.norm.sf(abs(pooled / se))),
        "k": len(usable),
    }


def conditional_logit_viraemia(data, temperature):
    """Conditional logistic regression of viraemia on temperature, matched on participant.

    Only participants with both viraemic and non-viraemic visits inform it.
    Temperature and year are centred to help the optimiser. `converged`
    checks the gradient at the reported solution.
    """
    data = data.dropna(subset=["viraemic", temperature]).copy()
    per_person = data.groupby(PARTICIPANT)["viraemic"].agg(["min", "max"])
    discordant = per_person.index[per_person["min"] != per_person["max"]]
    data = data[data[PARTICIPANT].isin(discordant)]

    design = pd.concat(
        [
            (data[[temperature]] - data[temperature].mean()).astype(float),
            month_dummies(data),
            (data[["year"]] - 2014).astype(float),
        ],
        axis=1,
    )
    model = ConditionalLogit(
        data["viraemic"].values, design.values, groups=pd.factorize(data[PARTICIPANT])[0]
    )
    fit = model.fit(method="bfgs", maxiter=2000, disp=0)

    largest_score = float(np.abs(model.score(fit.params)).max())
    converged = bool(
        np.isfinite(fit.params).all() and largest_score < CONVERGENCE_TOLERANCE and abs(fit.params[0]) > 0
    )
    return {
        "max_abs_score": largest_score,
        "converged": converged,
        "est": float(fit.params[0]),
        "se": float(fit.bse[0]),
        "p": float(fit.pvalues[0]),
        "OR": float(np.exp(fit.params[0])),
        "n_obs": int(len(data)),
        "n_part": int(data[PARTICIPANT].nunique()),
    }


results = {
    "exposure": {"primary": "era5_temp_lag7d_c (rolling 7-day mean)", "secondary": "mean of lags 0, 1, 3"}
}


# --------------------------------------------------------------------------
# H1  HIV x temperature within mixed cohorts
# --------------------------------------------------------------------------


def h1(temperature, cohorts):
    data = visits[visits["cohort"].isin(cohorts)]
    per_biomarker = {}
    interaction_estimates = []
    for column in BIOMARKER_COLUMNS:
        rows = data.dropna(subset=[column, "HIV"])
        if rows.empty:
            continue

        # Eligible if two or more cohorts measured it,
        # or one cohort has at least 100 people in each HIV group.
        hiv_by_person = rows.groupby(["cohort", PARTICIPANT])["HIV"].first()
        n_positive = hiv_by_person.groupby("cohort").sum()
        n_negative = hiv_by_person.groupby("cohort").count() - n_positive
        enough_in_one_cohort = ((n_positive >= 100) & (n_negative >= 100)).any()
        if len(n_positive) < 2 and not enough_in_one_cohort:
            continue

        # Outcome in SD units so biomarkers can be pooled.
        rows = rows.assign(Y=rows[column] / visits[column].std(), TxHIV=rows[temperature] * rows["HIV"])
        fit = within_person_regression(rows, "Y", [temperature, "TxHIV"])
        if fit is None or "TxHIV" not in fit["estimates"]:
            per_biomarker[column] = "not estimable"
            continue

        per_biomarker[column] = {
            "interaction": fit["estimates"]["TxHIV"],
            "main_T": fit["estimates"][temperature],
            "n_obs": fit["n_obs"],
            "n_part": fit["n_part"],
            "cohorts": fit["cohorts"],
        }
        estimate, se, _ = fit["estimates"]["TxHIV"]
        interaction_estimates.append((estimate, se))
    return pool_inverse_variance(interaction_estimates), per_biomarker


h1_pooled, h1_per_biomarker = h1(PRIMARY, MIXED_HIV_COHORTS)
results["H1"] = {
    "primary": h1_pooled,
    "per_biomarker": h1_per_biomarker,
    "secondary": h1(SECONDARY, MIXED_HIV_COHORTS)[0],
    "loco": {
        dropped: h1(PRIMARY, [c for c in MIXED_HIV_COHORTS if c != dropped])[0]
        for dropped in MIXED_HIV_COHORTS
    },
}
print("H1", results["H1"]["primary"], flush=True)


# --------------------------------------------------------------------------
# H2  Viraemia within person
# --------------------------------------------------------------------------


def without_cohort(data, cohort):
    return data[data["cohort"] != cohort]


conditional_logit = conditional_logit_viraemia(plwh, PRIMARY)
linear_probability = one_coefficient(plwh, "viraemic", PRIMARY)

USE_CONDITIONAL_LOGIT = conditional_logit["converged"]
if USE_CONDITIONAL_LOGIT:
    h2_primary = conditional_logit
    h2_model = "conditional logit"
else:
    h2_primary = linear_probability
    h2_model = "linear probability model with participant fixed effects (conditional logit did not converge)"


def h2_fit(data, temperature):
    """H2 with whichever model the convergence check selected."""
    if USE_CONDITIONAL_LOGIT:
        return conditional_logit_viraemia(data, temperature)
    return one_coefficient(data, "viraemic", temperature)


cohorts_with_viraemia = sorted(plwh.dropna(subset=["viraemic"])["cohort"].unique())
results["H2"] = {
    "model_used": h2_model,
    "primary": h2_primary,
    "conditional_logit": conditional_logit,
    "linear_probability": linear_probability,
    "log10_viral_load": one_coefficient(plwh, "log10_viral_load", PRIMARY),
    "secondary": h2_fit(plwh, SECONDARY),
    "loco": {dropped: h2_fit(without_cohort(plwh, dropped), PRIMARY) for dropped in cohorts_with_viraemia},
}
print("H2", h2_model, h2_primary, flush=True)


# --------------------------------------------------------------------------
# H3  CD4 within person, then post hoc stress tests
# --------------------------------------------------------------------------

cohorts_with_cd4 = sorted(plwh.dropna(subset=["cd4_count"])["cohort"].unique())
results["H3"] = {
    "primary": one_coefficient(plwh, "log_cd4", PRIMARY),
    "secondary": one_coefficient(plwh, "log_cd4", SECONDARY),
    "loco": {
        dropped: one_coefficient(without_cohort(plwh, dropped), "log_cd4", PRIMARY)
        for dropped in cohorts_with_cd4
    },
}
print("H3", results["H3"]["primary"], flush=True)


def cubic_in_years_since_first_visit(data):
    years = data["days_since_first_visit"] / 365
    return pd.DataFrame({"t1": years, "t2": years**2, "t3": years**3}, index=data.index)


def cohort_by_month_dummies(data):
    return pd.get_dummies(data["cohort"] + "_" + data["month"].astype(str), drop_first=True, dtype=float)


after_first_90_days = plwh[plwh["days_since_first_visit"] > 90]
with_haematocrit = plwh.dropna(subset=["hematocrit"])

results["H3_posthoc"] = {
    "base": one_coefficient(plwh, "log_cd4", PRIMARY),
    # CD4 rises fastest in the first months on treatment. These three allow for that.
    "s1_time_since_first_visit_cubic": one_coefficient(
        plwh, "log_cd4", PRIMARY, extra_covariates=cubic_in_years_since_first_visit
    ),
    "s2_person_linear_trend": one_coefficient(plwh, "log_cd4", PRIMARY, person_trends=True),
    "s4_drop_first_90_days": one_coefficient(after_first_90_days, "log_cd4", PRIMARY),
    # Season could differ by cohort.
    "s3_cohort_x_month": one_coefficient(plwh, "log_cd4", PRIMARY, extra_covariates=cohort_by_month_dummies),
    # Does CD4 move beyond what haematocrit at the same visit explains?
    "s5_adjust_hct_same_visit": one_coefficient(plwh, "log_cd4", PRIMARY, adjust_for=["hematocrit"]),
    "s5_same_sample_no_hct": one_coefficient(with_haematocrit, "log_cd4", PRIMARY),
}
for label, estimate in results["H3_posthoc"].items():
    print("H3 post hoc", label, estimate, flush=True)


# --------------------------------------------------------------------------
# H4  Cluster 3 and HIV within cohort
# --------------------------------------------------------------------------

labels, hiv_positive, cluster3 = published_clusters()
cohort_of_person = visits.groupby(PARTICIPANT)["cohort"].agg(lambda s: s.mode().iat[0])
people = pd.DataFrame(
    {
        "in_cluster3": (labels == cluster3).astype(int),
        "hiv": hiv_positive.reindex(labels.index),
        "cohort": cohort_of_person.reindex(labels.index),
    }
).dropna()

# One 2x2 table (HIV by Cluster 3) per cohort. Cohorts with an empty row or column carry no information.
tables = []
for _, cohort_people in people.groupby("cohort"):
    table = pd.crosstab(cohort_people["hiv"], cohort_people["in_cluster3"])
    table = table.reindex(index=[0, 1], columns=[0, 1], fill_value=0).values.astype(float)
    if (table.sum(axis=1) > 0).all() and (table.sum(axis=0) > 0).all():
        tables.append(table)
stratified = StratifiedTable(tables)
cmh = stratified.test_null_odds(correction=True)

# Permutation: shuffle HIV status within each cohort and recompute HIV prevalence in Cluster 3.
rng = np.random.default_rng(SEED)
hiv_values = people["hiv"].values.astype(float)
cluster3_members = people["in_cluster3"].values == 1
rows_of_cohort = {c: np.where(people["cohort"].values == c)[0] for c in np.unique(people["cohort"].values)}
observed_prevalence = hiv_values[cluster3_members].mean()
shuffled_prevalence = np.empty(N_PERMUTATIONS)
for i in range(N_PERMUTATIONS):
    shuffled = hiv_values.copy()
    for rows in rows_of_cohort.values():
        shuffled[rows] = rng.permutation(hiv_values[rows])
    shuffled_prevalence[i] = shuffled[cluster3_members].mean()

results["H4"] = {
    "cmh_stat": float(cmh.statistic),
    "p": float(cmh.pvalue),
    "pooled_OR": float(stratified.oddsratio_pooled),
    "n_strata_informative": len(tables),
    "within_cohort_perm_obs": float(observed_prevalence),
    "within_cohort_perm_null_max": float(shuffled_prevalence.max()),
    "within_cohort_perm_p": float((shuffled_prevalence >= observed_prevalence).mean()),
}
print("H4", results["H4"], flush=True)


# --------------------------------------------------------------------------
# H5  Low CD4 x temperature on haematocrit and haemoglobin
# --------------------------------------------------------------------------


def h5(temperature, modifier, data):
    per_biomarker = {}
    interaction_estimates = []
    for column in ["hematocrit", "hemoglobin"]:
        rows = data.dropna(subset=[column, modifier, temperature])
        rows = rows.assign(
            Y=rows[column] / visits[column].std(),
            M=rows[modifier],
            TxM=rows[temperature] * rows[modifier],
        )
        fit = within_person_regression(rows, "Y", [temperature, "M", "TxM"])
        if fit is None or "TxM" not in fit["estimates"]:
            per_biomarker[column] = "not estimable"
            continue
        per_biomarker[column] = {
            "interaction": fit["estimates"]["TxM"],
            "n_obs": fit["n_obs"],
            "n_part": fit["n_part"],
            "cohorts": fit["cohorts"],
        }
        estimate, se, _ = fit["estimates"]["TxM"]
        interaction_estimates.append((estimate, se))
    return pool_inverse_variance(interaction_estimates), per_biomarker


has_low_cd4_and_blood = plwh.dropna(subset=["low_cd4"]).dropna(subset=["hematocrit", "hemoglobin"], how="all")
cohorts_for_h5 = sorted(has_low_cd4_and_blood["cohort"].unique())

h5_pooled, h5_per_biomarker = h5(PRIMARY, "low_cd4", plwh)
results["H5"] = {
    "primary": h5_pooled,
    "per_biomarker": h5_per_biomarker,
    "secondary": h5(SECONDARY, "low_cd4", plwh)[0],
    "viraemic_modifier": h5(PRIMARY, "viraemic", plwh)[0],
    "loco": {dropped: h5(PRIMARY, "low_cd4", without_cohort(plwh, dropped))[0] for dropped in cohorts_for_h5},
}
print("H5", results["H5"]["primary"], flush=True)


# --------------------------------------------------------------------------
# Benjamini-Hochberg across the five primary tests
# --------------------------------------------------------------------------

primary_p = {
    "H1": results["H1"]["primary"]["p"],
    "H2": results["H2"]["primary"]["p"],
    "H3": results["H3"]["primary"]["p"],
    "H4": results["H4"]["p"],
    "H5": results["H5"]["primary"]["p"],
}
ordered = sorted(primary_p, key=primary_p.get)
n_tests = len(ordered)
q_values = {}
smallest_so_far = 1.0
# Work from the largest p down, so each q is the smallest adjusted value at or above its rank.
for rank in range(n_tests, 0, -1):
    test = ordered[rank - 1]
    smallest_so_far = min(smallest_so_far, primary_p[test] * n_tests / rank)
    q_values[test] = smallest_so_far
results["BH"] = {test: {"p": float(primary_p[test]), "q": float(q_values[test])} for test in primary_p}
print("BH", results["BH"], flush=True)

write_json(results, "r1_hiv_tests.json")
