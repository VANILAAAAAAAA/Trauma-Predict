# Processed Current-Field Sample — HADM 23492347

Selected because hospital LOS ≈ 14.05 days and ICU LOS ≈ 14.05 days.

Source raw sample:

```text
/home/vanila/code/EHR-Predict/data dicision/trauma MIMICIV sample/samples/hadm_23492347
```

Field boundary:

```text
/home/vanila/code/EHR-Predict/data dicision/field adapter/field.md
```

## Files

| File | Content |
|---|---|
| `00_static_fields.csv` | STATIC include fields: age, male, mechanism_cat, transfer, initial_ed_sbp |
| `01_hourly_current_fields_first312h.csv` | 312 hourly rows using include-only current fields |
| `01_hourly_observed_recency_metadata.csv` | observed/recency metadata for hourly/lab fields |
| `02_first48h_fields.csv` | G2* first-48h conditional fields |
| `03_daily_summary_preview.csv` | simple 13-day daily summary preview |
| `04_field_provenance.csv` | provenance notes for static fields |
| `manifest.json` | generation metadata and caveats |

## Caveats

- This is a field-adapter demonstration sample, not a final training sample builder.
- `initial_ed_sbp` is missing because this HADM has no ED linkage rows.
- `lymphocytes` and `neutrophils` are converted to K/uL from same-time WBC and differential percentage when direct absolute counts are unavailable.
- Inputevent amounts are assigned to the start hour in this preview; production code should split intervals by hour overlap.
- Post-312h data are not included in the processed sample.
