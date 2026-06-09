# EHR-Predict

Trauma ICU patient-state modeling pipeline.  
MIMIC-IV v2.2 + UW trauma field schema → canonical hourly state → next-state prediction + 12h report design.

## Project Structure

```text
EHR-Predict/
├── README.md
├── AGENTS.md
│
├── data dicision/
│   ├── trauma cohort/          # Step 1: Cohort extraction (done)
│   ├── trauma MIMICIV sample/  # Step 1b: one-HADM raw MIMIC-IV source-row sample
│   └── field adapter/          # Step 2: UW↔MIMIC field alignment (pending)
│
├── Input design/
│   ├── input/               # Step 3a: Model input schema
│   └── summary design/      # Step 3b: daily/report summary schema
│
├── sample builder/          # Step 4: final task sample construction; NOT started
│   ├── single/              # empty until input + summary schema are fixed
│   └── mixed/               # empty until single-sample contract is fixed
│
├── audit/                   # audit scripts, tests, bug reports
│   ├── scripts/
│   └── bug_reports.md
│
├── reference/               # reference only — not active pipeline
│   ├── hf_ehr/
│   ├── pipeline design/
│   ├── papers/
│   ├── ehr2path-rich-docs/
│   ├── pipeline_hold/       # inactive pipeline attempts / parked scripts
│   └── legacy/
│
└── agent-artifact/          # agent-readable project state/wiki
    ├── project_state.yaml
    ├── path_registry.yaml
    ├── active_pipeline.yaml
    ├── MANIFEST.yaml
    ├── INDEX.jsonl
    └── archive/
```

## Pipeline Order

```text
trauma cohort → trauma MIMICIV sample → field adapter → Input design → sample builder
```

## Current State

| Stage | Status |
|---|---|
| trauma cohort | done: C4 = 6,583 HADM / 7,507 ICU stays |
| trauma MIMICIV sample | done: one-HADM raw source-row sample under `data dicision/trauma MIMICIV sample/samples/hadm_20002252/` |
| field adapter | pending: build current UW↔MIMIC field registry |
| input design | in progress: `Input design/` |
| summary design | in progress: `Input design/summary design/report_schema_v1.md` |
| sample builder | not started; active folders must stay empty until input+summary schema are fixed |

## Key Paths

- Cohort CSV: `data dicision/trauma cohort/cohort/mimiciv_trauma_cohort_los48.csv`
- Cohort config: `data dicision/trauma cohort/pipeline/01_cohort_extraction/mimiciv_trauma_cohort_config.json`
- Raw MIMIC-IV sample: `data dicision/trauma MIMICIV sample/samples/hadm_20002252/`
- Field registry reference: `agent-artifact/archive/compiled_202605xx/uw_mimic_field_registry_v0_20260523.yaml`
- Raw-row extraction reference, not active sample builder: `reference/pipeline_hold/02_sample_mtou_raw_not_active/raw_mimic_rows/`
- Data root: `/mnt/d/Data`
- MIMIC-IV: `/mnt/d/Data/mimic-iv-2.2`

## Rules

1. Do not put partial/raw-row extraction scripts into `sample builder/`.
2. `sample builder/` starts only after field adapter + input schema + summary schema are fixed.
3. Old scripts and parked attempts belong under `reference/pipeline_hold/` or `reference/legacy/`.
4. Active stage folders contain only files for the current stage boundary.
5. Desktop = display/reference; D:\Data = data package; workspace = active project structure.
