"""Aggregate table: participants with each retained biomarker measured, by cohort, plus HIV prevalence and Cluster 3 share. No participant rows."""
import os, sys, logging
from pathlib import Path
import pandas as pd
os.environ.setdefault("RP2_PRIMARY_EXPOSURE", "era5_land")
REPO = Path(__file__).resolve().parents[2]  # repository root; sys.path.insert(0, str(REPO))
logging.basicConfig(level=logging.ERROR)
from mcd_pipeline import config
from mcd_pipeline.stage0_data_prep.build_analysis_dataset import build_analysis_dataset
OUT = Path(__file__).resolve().parents[1] / "outputs"
RET = ['creatinine','diastolic_bp','heart_rate','cd4_count','viral_load','ldl_cholesterol','waist_hip_ratio','body_fat_percent','hemoglobin','hematocrit']
df = build_analysis_dataset()
S3 = config.OUTPUT_ROOT / "stage3_r2_005"
lab = pd.read_csv(S3 / 'cluster_labels.csv').set_index('patient_id')['cluster']
meta = pd.read_csv(S3 / 'patient_cluster_table.csv').set_index('patient_id')
hiv = meta['hiv_status'].astype(str).str.lower().eq('positive')
c3 = int(pd.DataFrame({'c': lab, 'h': hiv.reindex(lab.index)}).groupby('c')['h'].mean().idxmax())
g = df.groupby('patient_id')
per = pd.DataFrame({b: g[b].apply(lambda s: s.notna().any()) for b in RET})
per['study'] = g['study_source'].agg(lambda s: s.mode().iat[0])
per = per.loc[per.index.intersection(lab.index)]
per['hiv'] = hiv.reindex(per.index); per['c3'] = (lab.reindex(per.index) == c3)
t = per.groupby('study').agg(n=('hiv', 'size'), hiv_prev=('hiv', 'mean'), cluster3_share=('c3', 'mean'), **{b: (b, 'sum') for b in RET})
t.round(3).to_csv(OUT / 'r1_availability_by_cohort.csv'); print(t.round(3).to_string())
