"""Download ERA5-HEAT UTCI hourly time series (CDS derived-utci-historical-timeseries, v1.1 consolidated) for every
0.25-degree cell that holds an exposure location; build daily mean and daily range (UTC days, as the ERA5-Land
linkage), in degC. Accessed 10 Nov 2026."""
import cdsapi, pandas as pd, numpy as np, os, time
from pathlib import Path
H = Path(__file__).resolve().parent
U = H.parent / 'utci'; U.mkdir(exist_ok=True)  # downloads and daily cells (not committed)
link = pd.read_csv(H.parents[1] / 'FINAL_DATASETS/CLIMATE_HEALTH_LINKAGE/ERA5_LAND_LINKAGE.csv', usecols=['latitude', 'longitude'])
cells = (link[['latitude', 'longitude']].drop_duplicates() / 0.25).round() * 0.25
cells = cells.drop_duplicates().reset_index(drop=True)
print('cells', len(cells), flush=True)
c = cdsapi.Client(quiet=True)
for _, r in cells.iterrows():
    fn = U / f'utci_{r.latitude:.2f}_{r.longitude:.2f}.csv'
    if fn.exists() and fn.stat().st_size > 1e6: continue
    c.retrieve('derived-utci-historical-timeseries', {'variable': ['universal_thermal_climate_index'], 'location': {'longitude': float(r.longitude), 'latitude': float(r.latitude)}, 'date': ['2005-03-01/2022-12-31'], 'data_format': 'csv'}, str(fn))
    print('got', fn.name, flush=True)
out = []
for f in H.glob('utci_-*.csv'):
    d = pd.read_csv(f, parse_dates=['valid_time'])
    lat, lon = [float(x) for x in f.stem.split('_')[1:3]]
    d['utci_c'] = d['utci'] - 273.15; d['date'] = d['valid_time'].dt.floor('D')
    g = d.groupby('date')['utci_c'].agg(utci_mean='mean', utci_max='max', utci_min='min').reset_index()
    g['cell_lat'], g['cell_lon'] = lat, lon; out.append(g)
pd.concat(out).to_csv(U / 'utci_daily_cells.csv', index=False)
print('daily written', flush=True)
