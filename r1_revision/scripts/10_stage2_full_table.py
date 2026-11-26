"""10. Every Stage 2 interaction that was tested, with its direction (supplementary data file 1)

Question    The statistical reviewer asked for all interaction estimates, raw
            p values, adjusted q values and effect directions.
Data in     stage2_full_interaction_table.csv in the pipeline output folder:
            the published Stage 2 results for every biomarker (observed
            interaction, permutation null, raw p, Benjamini-Hochberg q).
            Published Stage 1 SHAP values.
Steps       1. Keep the pairs that were tested for the ten retained biomarkers (340).
            2. Add a descriptive direction for each pair: the Spearman
               correlation between the modifier's value and the temperature
               feature's SHAP value across visits. Positive means the
               temperature attribution is larger where the modifier is higher
               (for sex_Male, larger in men).
File out    r1_stage2_all_340_pairs.csv
Reported    Supplementary data file 1; table 2 footnote.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from r1_common import OUTPUT_DIR, RETAINED, STAGE1_DIR, config, load_visits, prepare_features_and_target

visits, biomarkers = load_visits()

pairs = pd.read_csv(config.OUTPUT_ROOT / "stage2_full_interaction_table.csv")
was_tested = pairs["Permutation_p"].notna()
pairs = pairs[was_tested & pairs["Biomarker"].isin(RETAINED)].copy()


def features_and_shap(name):
    """The model's feature table and its SHAP values for one biomarker, row for row."""
    X, _, _ = prepare_features_and_target(visits, biomarkers[name])
    shap_values = np.load(STAGE1_DIR / name / "shap_values.npy")
    with open(STAGE1_DIR / name / "feature_names.json") as handle:
        feature_names = json.load(handle)
    assert shap_values.shape[0] == len(X), "SHAP rows must match the model's rows"
    return X.reset_index(drop=True), pd.DataFrame(shap_values, columns=feature_names)


loaded = {}
directions = []
for pair in pairs.itertuples():
    if pair.Biomarker not in loaded:
        loaded[pair.Biomarker] = features_and_shap(pair.Biomarker)
    X, shap = loaded[pair.Biomarker]

    if pair.Modifier in X.columns and pair.Temperature_feature in shap.columns:
        direction = spearmanr(X[pair.Modifier], shap[pair.Temperature_feature]).correlation
    else:
        direction = np.nan
    directions.append(direction)

pairs["direction_rho"] = directions
pairs = pairs.rename(
    columns={
        "Observed_interaction": "observed_mean_abs_interaction",
        "Permutation_p": "p_raw",
        "FDR_q_value": "q_BH",
    }
)
pairs.to_csv(OUTPUT_DIR / "r1_stage2_all_340_pairs.csv", index=False)
print("pairs:", len(pairs), "| significant:", pairs["Significant_FDR"].sum())
