# Integration Scripts

## Active Script

### `CREATE_SEPARATE_DATASETS.py`
The canonical data integration script. Reads from the harmonised source at
`/home/cparker/incoming/RP2/00_FINAL_DATASETS/01_PRIMARY_DATASETS/JOHANNESBURG_COMPLETE_BIOMARKERS_FULLY_HARMONIZED.csv`,
deduplicates on Patient ID + primary_date, merges climate features, and
produces the datasets in `FINAL_DATASETS/`.

## Archived Scripts

Nine earlier integration script variants (FINAL_INTEGRATION_SCRIPT.py,
FULL_DATA_INTEGRATION.py, etc.) were archived to `_archive/sep2025_integration_scripts/`
on 2026-02-25. They all referenced the older `_ENHANCED.csv` source file
(12,033 records) rather than the current `_FULLY_HARMONIZED.csv` (11,114
records after QC).
