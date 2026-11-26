"""03. Does any single cohort carry the temperature signal? (appendix SA12)

Question    The statistical reviewer asked whether cohort-level differences
            influence the main findings.
Data in     The visit-level analysis table. The ten retained biomarkers.
Steps       For each biomarker and each cohort that measured it:
            1. Drop that cohort's visits.
            2. Refit the full and the covariate-only model (script 02's method).
            3. Record the incremental R-squared.
File out    r1_stage1_loco.csv  (one row per biomarker and dropped cohort)
Reported    SA12 and one Results sentence: the increment stayed positive after
            every single-cohort exclusion for eight of ten biomarkers.
"""

import pandas as pd

from r1_common import (
    OUTPUT_DIR,
    RETAINED,
    cross_validate,
    load_visits,
    prepare_features_and_target,
    without_temperature,
)

MIN_PARTICIPANTS_LEFT = 20

visits, biomarkers = load_visits()
cohort_of_visit = visits["study_source"].astype(str)

rows = []
for name in RETAINED:
    measured = visits[name].notna()
    cohorts_with_data = sorted(cohort_of_visit[measured].unique())

    if len(cohorts_with_data) < 2:
        rows.append({"biomarker": name, "dropped": "(single cohort)", "n_obs": int(measured.sum())})
        continue

    for dropped in cohorts_with_data:
        remaining = visits[cohort_of_visit != dropped]
        X, y, participants = prepare_features_and_target(remaining, biomarkers[name])
        if participants.nunique() < MIN_PARTICIPANTS_LEFT:
            continue

        _, r2_full = cross_validate(X, y, participants)
        _, r2_cov = cross_validate(without_temperature(X), y, participants)

        rows.append(
            {
                "biomarker": name,
                "dropped": dropped,
                "n_obs": len(y),
                "r2_full": float(r2_full.mean()),
                "r2_cov": float(r2_cov.mean()),
                "delta": float((r2_full - r2_cov).mean()),
                "folds_better": int((r2_full > r2_cov).sum()),
            }
        )
        print(rows[-1], flush=True)

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "r1_stage1_loco.csv", index=False)
