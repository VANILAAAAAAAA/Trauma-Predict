# Sample Builder

Status: **not started**.

This stage starts only after all upstream contracts are fixed:

1. `data dicision/field adapter/` — current UW↔MIMIC field registry
2. `Input design/input/` — model input schema
3. `Input design/summary design/` — daily/report summary schema

## Hard boundary

Do not place raw-row extraction scripts, partial builders, or one-off examples under:

```text
sample builder/single/
sample builder/mixed/
```

Those folders are intentionally empty until the full task-sample contract is approved.

## Parked reference

The previous raw-row extraction script is parked here as reference only:

```text
reference/pipeline_hold/02_sample_mtou_raw_not_active/raw_mimic_rows/
```

It extracts official MIMIC-IV rows for one HADM. It does not build a complete model input sample.
