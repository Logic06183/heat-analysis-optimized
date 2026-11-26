"""05. Which cohorts measured which biomarkers, and who ends up in Cluster 3? (appendix SA14)

Question    Does Cluster 3 membership follow the biomarkers a cohort measured?
Data in     The visit-level analysis table and the published cluster labels.
Steps       1. For each participant, note which retained biomarkers they ever
               had measured, their cohort, HIV status and whether they are in
               Cluster 3.
            2. Summarise by cohort.
File out    r1_availability_by_cohort.csv  (one row per cohort)
Reported    SA14 and its table; Results ("In all six cohorts that measured
            haematocrit, 95% to 100% of participants fell into Cluster 3").
"""

import pandas as pd

from r1_common import OUTPUT_DIR, PARTICIPANT, RETAINED, build_analysis_dataset, published_clusters

visits = build_analysis_dataset()
labels, hiv_positive, cluster3 = published_clusters()
by_participant = visits.groupby(PARTICIPANT)

people = pd.DataFrame(
    {name: by_participant[name].apply(lambda values: values.notna().any()) for name in RETAINED}
)
people["study"] = by_participant["study_source"].agg(lambda s: s.mode().iat[0])
people = people.loc[people.index.intersection(labels.index)]
people["hiv"] = hiv_positive.reindex(people.index)
people["in_cluster3"] = labels.reindex(people.index) == cluster3

table = people.groupby("study").agg(
    n=("hiv", "size"),
    hiv_prev=("hiv", "mean"),
    cluster3_share=("in_cluster3", "mean"),
    **{name: (name, "sum") for name in RETAINED},
)
table = table.round(3)

table.to_csv(OUTPUT_DIR / "r1_availability_by_cohort.csv")
print(table.to_string())
