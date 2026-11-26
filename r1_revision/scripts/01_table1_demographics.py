"""01. Sex, age and dwelling type by cohort (manuscript table 1).

Question    Reviewer 5 asked for more demographic detail on each cohort.
Data in     The visit-level analysis table.
Steps       1. Reduce to one row per participant.
            2. Summarise by cohort, then for everyone.
File out    r1_table1_demographics.csv (one row per cohort plus TOTAL).
Reported    Table 1: Female (%), Age (mean, SD), Informal (%).

Unemployment is computed here but is not reported. It is a ward-level matched
value and came out at 0% in three cohorts, which looks like an artefact of
the matching.
"""

import numpy as np
import pandas as pd

from r1_common import OUTPUT_DIR, PARTICIPANT, build_analysis_dataset


def first_recorded(values):
    """A participant's first non-missing value, or NaN."""
    recorded = values.dropna()
    return recorded.iloc[0] if len(recorded) else np.nan


def is_female(sex_values):
    sex = first_recorded(sex_values)
    return np.nan if pd.isna(sex) else str(sex).lower().startswith("f")


def lives_in_informal_dwelling(dwelling_values):
    dwelling = first_recorded(dwelling_values)
    return np.nan if pd.isna(dwelling) else str(dwelling).lower() == "informal"


def is_unemployed(employment_values):
    employment = first_recorded(employment_values)
    return np.nan if pd.isna(employment) else str(employment).lower().startswith("not")


def percent(flags):
    return round(100 * flags.astype(float).mean(), 1)


def summarise(people):
    return pd.Series(
        {
            "n": len(people),
            "female_pct": percent(people["female"]),
            "age_mean": people["age"].mean(),
            "age_sd": people["age"].std(),
            "age_missing": people["age"].isna().mean(),
            "visits_median": people["visits"].median(),
            "informal_pct": percent(people["informal"]),
            "unemployed_pct": percent(people["unemployed"]),
        }
    )


visits = build_analysis_dataset()
by_participant = visits.groupby(PARTICIPANT)

people = pd.DataFrame(
    {
        "study": by_participant["study_source"].agg(lambda s: s.mode().iat[0]),
        "female": by_participant["sex"].agg(is_female),
        "age": by_participant["age_years"].min(),
        "visits": by_participant.size(),
        "informal": by_participant["gcro_dwelling_type"].agg(lives_in_informal_dwelling),
        "unemployed": by_participant["gcro_employment_status"].agg(is_unemployed),
    }
)

table = people.groupby("study").apply(summarise)
table.loc["TOTAL"] = summarise(people)
share_without_age = table["age_missing"].round(3)
table = table.round(1)
table["age_missing"] = share_without_age

table.to_csv(OUTPUT_DIR / "r1_table1_demographics.csv")
print(table.to_string())
