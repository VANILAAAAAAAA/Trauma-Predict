# EHR-Predict — Agent Entry Point

## Load Order

1. `agent-artifact/project_state.yaml` — current cohort and pipeline status
2. `agent-artifact/path_registry.yaml` — authoritative paths
3. `agent-artifact/active_pipeline.yaml` — active stage contracts
4. `README.md` — human-facing structure

## Pipeline Stages

```text
trauma cohort (done) → trauma MIMICIV sample (done) → field adapter (pending) → Input design (in progress) → sample builder (not started)
```

**Current denominator:** C4 cohort: 6,583 HADM / 7,507 ICU stays  
**Cohort CSV:** `data dicision/trauma cohort/cohort/mimiciv_trauma_cohort_los48.csv`  
**Raw MIMIC-IV sample:** `data dicision/trauma MIMICIV sample/samples/hadm_20002252/`  
**C5 ventilated subset:** 1,738 HADM — optional, not current denominator.

## Current Work

`Input design/summary design/` — summary/report design based on available fields.

## Hard Boundaries

- Do **not** create or place files under `sample builder/single/` or `sample builder/mixed/` until field adapter + input schema + summary schema are fixed.
- Raw-row extraction is not a training sample builder. The parked raw-row extraction code is reference only:
  `reference/pipeline_hold/02_sample_mtou_raw_not_active/raw_mimic_rows/`
- Do **not** propose "minimal necessary files" or conservative partial structure for EHR-Predict project organization. Follow the full planned pipeline structure.
- Active code lives only in its correct pipeline stage folder.
- `reference/` = read-only reference / parked scripts.
- `agent-artifact/` = agent wiki, machine-readable.
- `audit/` = audit scripts + bug_reports.md only.
- No shared `scripts/` dump.

## Modeling Constraints

- V1 model direction: self-supervised next-hour state prediction.
- 12h report: summary/report design and evaluation target; not a raw-row sample builder.
- daily/hour inputs = multi-resolution views of the same patient trajectory.
- Labs: intermittent memory features, not dense hourly.
- All hourly fields: event_time <= current_hour_end only.
- Patient-level split mandatory.
- No leakage: future data never enters input.
