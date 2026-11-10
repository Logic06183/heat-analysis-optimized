"""Per-cohort demographics for revised Table 1 (aggregate only)."""
import os, sys, logging
from pathlib import Path
import numpy as np, pandas as pd
os.environ.setdefault("RP2_PRIMARY_EXPOSURE", "era5_land")
REPO = Path(__file__).resolve().parents[2]  # repository root; sys.path.insert(0, str(REPO))
logging.basicConfig(level=logging.ERROR)
from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
OUT = Path(__file__).resolve().parents[1] / "outputs"
df = build_analysis_dataset()
g = df.groupby('patient_id')
p = pd.DataFrame({'study': g['study_source'].agg(lambda s: s.mode().iat[0]),
    'female': g['sex'].agg(lambda s: str(s.dropna().iloc[0]).lower().startswith('f') if s.notna().any() else np.nan),
    'age': g['age_years'].min(),
    'visits': g.size(),
    'informal': g['gcro_dwelling_type'].agg(lambda s: str(s.dropna().iloc[0]).lower() == 'informal' if s.notna().any() else np.nan),
    'unemployed': g['gcro_employment_status'].agg(lambda s: str(s.dropna().iloc[0]).lower().startswith('not') if s.notna().any() else np.nan)})
t = p.groupby('study').agg(n=('age', 'size'), female_pct=('female', lambda s: s.astype(float).mean()), age_mean=('age', 'mean'), age_sd=('age', 'std'), age_missing=('age', lambda s: s.isna().mean()),
    visits_median=('visits', 'median'), informal_pct=('informal', lambda s: s.astype(float).mean()), unemployed_pct=('unemployed', lambda s: s.astype(float).mean()))
for c in ['female_pct', 'informal_pct', 'unemployed_pct']: t[c] = (100 * t[c].astype(float)).round(1)
tot = pd.DataFrame({'n': [len(p)], 'female_pct': [round(100 * p.female.astype(float).mean(), 1)], 'age_mean': [p.age.mean()], 'age_sd': [p.age.std()], 'age_missing': [p.age.isna().mean()], 'visits_median': [p.visits.median()], 'informal_pct': [round(100 * p.informal.astype(float).mean(), 1)], 'unemployed_pct': [round(100 * p.unemployed.astype(float).mean(), 1)]}, index=['TOTAL'])
t = pd.concat([t, tot]).round(1); t.to_csv(OUT / 'r1_table1_demographics.csv'); print(t.to_string())
print(df['gcro_dwelling_type'].astype(str).value_counts().head(), df['gcro_employment_status'].astype(str).value_counts().head())
