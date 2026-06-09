# Pipeline Script Map

This directory is intentionally layered. Read this file first before running or editing scripts.

## Current working rule

Current cohort denominator is fixed at **C4 = 6,583 HADM / 7,507 ICU stays** (`../cohort/mimiciv_trauma_cohort_los48.csv`). C5 is retained only as an optional ventilated subset and is not the current sample-construction denominator.

Current sample-design work should use raw official MIMIC-IV rows first. Do **not** build hourly or daily summaries until the hour/day summary schema is explicitly fixed.

```text
current active path:
  02_sample_mtou_raw/
```

## Directory layout

```text
pipeline/
├── README.md
├── 01_cohort_extraction/
│   ├── extract_mimiciv_trauma_cohort.py
│   └── mimiciv_trauma_cohort_config.json
├── 02_sample_mtou_raw/
│   ├── build_sample_MtoU_raw.py
│   └── sample_MtoU_raw_config.json
└── 90_hold_not_active/
    ├── build_mimiciv_trauma_uw_state_v1.py
    └── uw_state_v1_config.json
```

## 01_cohort_extraction

Purpose: build the packaged trauma cohort from official MIMIC-IV raw tables and the clean E-code workbook.

Primary outputs already generated:

```text
../cohort/mimiciv_trauma_cohort_los48.csv          # C4 primary trauma cohort
../cohort/mimiciv_trauma_cohort_vent3_subset.csv   # C5 ventilated subset
../cohort/mimiciv_trauma_cohort_membership.csv     # HADM-level membership index
../metadata/cohort_layers.csv
../metadata/cohort_exclusion_summary.csv
```

Current cohort interpretation:

```text
CURRENT DENOMINATOR FOR SAMPLE CONSTRUCTION:
  C4 primary trauma cohort: 6,583 HADM / 7,507 ICU stays

RETAINED REFERENCE SUBSET ONLY:
  C5 ventilated subset:     1,738 HADM / 2,232 ICU stays
```

Do not use C5 as the default cohort unless the task is explicitly sepsis/ventilation-specific or requests the ventilated subset.

Rerun command:

```bash
python3 "/mnt/d/Data/Trauma cohort MIMICIV/pipeline/01_cohort_extraction/extract_mimiciv_trauma_cohort.py" \
  --config "/mnt/d/Data/Trauma cohort MIMICIV/pipeline/01_cohort_extraction/mimiciv_trauma_cohort_config.json"
```

## 02_sample_mtou_raw — active now

Purpose: for one HADM, extract UW G1-G4-relevant **raw MIMIC-IV rows** only.

This does not perform:

```text
hourly aggregation
daily summary
forward-fill
lab memory construction
clinical binning
model tokenization
```

Current sample output:

```text
../sample_MtoU_raw/hadm_20002252/
```

Run one HADM:

```bash
python3 "/mnt/d/Data/Trauma cohort MIMICIV/pipeline/02_sample_mtou_raw/build_sample_MtoU_raw.py" \
  --hadm-id 20002252
```

Default behavior if `--hadm-id` is omitted: use the first HADM in `../cohort/mimiciv_trauma_cohort_los48.csv`.

Output contract for each HADM:

```text
sample_MtoU_raw/hadm_<hadm_id>/
├── 00_cohort_rows.csv
├── 00_field_map.csv
├── 01_admissions_raw.csv
├── 02_patients_raw.csv
├── 03_icustays_raw.csv
├── 04_diagnoses_icd_raw.csv
├── 10_chartevents_G1_G3_raw.csv
├── 20_labevents_G4_raw.csv
├── 30_inputevents_G3_raw.csv
├── 40_outputevents_G4_uop_raw.csv
├── 50_procedureevents_G3_raw.csv
├── 60_prescriptions_antibiotics_raw.csv
└── manifest.json
```

Batch extension later should add only a thin loop over `hadm_id` values and reuse the same per-HADM folder contract.

## 90_hold_not_active

Purpose: parking area for the earlier hourly-state attempt.

Where it came from:

```text
build_mimiciv_trauma_uw_state_v1.py
uw_state_v1_config.json
```

These were created during the first UW G1-G4 mapping pass, before we narrowed the task back to `sample_MtoU_raw`. The script attempted to build `static_profile.csv`, `hourly_state.csv`, `field_registry.csv`, and `build_report.json` directly from the C4 cohort.

Why it is still useful:

```text
1. It records an early itemid registry for UW-like fields.
2. It shows one possible future implementation for hourly aggregation.
3. It preserves design choices that may be useful later: relative_hour, observed flags, latest lab memory, cumulative interventions.
```

Why it is not active now:

```text
hour-level summary is not fixed
daily summary is not fixed
aggregation rules are not approved
current task is raw-row sample extraction only
```

How to handle later:

```text
1. After raw samples are inspected, decide the hour/day summary schema.
2. Extract any useful itemid registry pieces into the active raw/sample config if needed.
3. Either promote a revised hourly builder into a new active folder, e.g. 03_hourly_summary/, or delete this hold folder after its useful parts are absorbed.
4. Do not run this script as-is for dataset construction.
```

## Cleanup rule

Do not leave `__pycache__`, temporary outputs, or exploratory scripts in `pipeline/`. Put new durable scripts into a numbered subfolder and update this README in the same change.
