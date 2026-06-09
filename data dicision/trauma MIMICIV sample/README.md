# Trauma MIMICIV Sample

Purpose: raw MIMIC-IV/ED source-row samples for inspecting official source rows before UW field adaptation.

Status: active data-inspection sample, not final model sample builder.

Boundary:
- raw filtered MIMIC-IV rows only
- ED rows included when ED linkage exists
- no hourly aggregation
- no daily summary
- no forward fill
- no tokenization
- no model-ready task sample

Pipeline:
```bash
python3 pipeline/extract_one_mimiciv_raw_sample.py --config pipeline/sample_MtoU_raw_config.json --hadm-id <hadm_id>
```

Input cohort:
`../trauma cohort/cohort/mimiciv_trauma_cohort_los48.csv`

Current samples:

```text
samples/hadm_20002252/  # first C4 HADM; no ED linkage rows
samples/hadm_20021110/  # ED-linked sample; recommended for G2/ED field discussion
```

ED-linked sample `hadm_20021110` contains:

```text
05_ed_linkage_raw.csv
06_edstays_raw.csv
07_ed_triage_raw.csv
08_ed_vitalsign_raw.csv
09_ed_diagnosis_raw.csv
```

Next stage:
Use `data dicision/field adapter/` to align these raw MIMIC-IV/ED source rows to the UW-style field registry.
