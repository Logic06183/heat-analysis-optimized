"""Export Table S4 - the FULL list of FDR-significant temperature x modifier
interactions (Stage 2), i.e. every pair the Figure S4 heatmap marks significant.

This reuses the *exact* classification + FDR logic from
figS4_interaction_heatmap.py (imported, not re-implemented) so the table is
guaranteed to match the figure. The figure only plots the top-10 triplets in
panel d; this script emits all of them.

Output:
    table_s4_interaction_pairs.csv   (machine-readable, ready to paste/build)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make both the figures dir (for _data/_lancet_style) and the repo root
# (for mcd_pipeline) importable, regardless of where this is launched from.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for p in (str(_HERE), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import json

import numpy as np
import pandas as pd

# Reuse the figure's own logic - identical classification + thresholds.
from figS4_interaction_heatmap import (
    BIOMARKER_ORDER, FDR_THRESHOLD, classify_modifier,
)
from _lancet_style import BIOMARKER_PRETTY, BIOMARKER_SYSTEM

# Paper primary spec = ERA5-Land. The figure currently (incorrectly) reads the
# non-ERA5-Land mcd_outputs/ path; point the table at the ERA5-Land summary so it
# reflects the 32-pair primary result. Override with TABLE_S4_SOURCE env var.
import os
STAGE2_SOURCE = Path(os.environ.get(
    "TABLE_S4_SOURCE",
    str(_REPO / "mcd_outputs_era5_land" / "stage2" / "stage2_summary.json"),
))


def load_stage2() -> dict:
    with open(STAGE2_SOURCE) as fh:
        return json.load(fh)


def clean_modifier(mod: str) -> str:
    return (mod.replace("gcro_", "")
               .replace("_", " ")
               .replace("sex Male", "Male (vs F)")
               .replace("hiv status Positive", "HIV+"))


def clean_temp(tf: str) -> str:
    return tf.replace("temp_lag_", "")


def main() -> None:
    stage2 = load_stage2()
    rows = []
    for bm in BIOMARKER_ORDER:
        rec = stage2.get(bm)
        if rec is None:
            continue
        bm_peak = None  # within-biomarker max |interaction| for scaling
        mags = [float(e["mean_abs_interaction"])
                for e in rec.get("top_temperature_modifiers", [])
                if classify_modifier(e["modifier"]) is not None]
        if mags:
            bm_peak = max(mags)
        for entry in rec.get("top_temperature_modifiers", []):
            mod = entry["modifier"]
            cat = classify_modifier(mod)
            if cat is None:
                continue
            p_adj = float(entry["p_adjusted"])
            if p_adj > FDR_THRESHOLD:
                continue
            mag = float(entry["mean_abs_interaction"])
            rows.append({
                "biomarker": BIOMARKER_PRETTY.get(bm, bm),
                "organ_system": BIOMARKER_SYSTEM.get(bm, ""),
                "modifier": clean_modifier(mod),
                "modifier_category": cat,
                "temperature_feature": clean_temp(entry["temp_feature"]),
                "mean_abs_interaction": round(mag, 4),
                "scaled_within_biomarker": round(mag / bm_peak, 4) if bm_peak else np.nan,
                "q_value_BH": round(p_adj, 4),
            })

    df = pd.DataFrame(rows)
    # Most significant first; ties broken by stronger interaction.
    df = df.sort_values(
        ["q_value_BH", "mean_abs_interaction"],
        ascending=[True, False],
    ).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    out = _HERE / "table_s4_interaction_pairs.csv"
    df.to_csv(out, index=False)

    # ---- Console summary for verification against the figure ----
    print(f"Total FDR-significant interaction pairs (q <= {FDR_THRESHOLD}): {len(df)}")
    print("\nBy modifier category (compare to panel c legend):")
    print(df["modifier_category"].value_counts().to_string())
    print("\nBy biomarker (compare to panel b counts):")
    print(df["biomarker"].value_counts().to_string())
    print(f"\nSaved: {out}")
    print("\n--- First 12 rows preview ---")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
