# Processed Current-Field Sample — HADM 26798055

Selected for ED linkage + head injury + ventilation + ICU ~9d.

## STATIC fields (all non-missing)

```
age-67 male-M mech-B tr-D ed_sbp-128 rsi-1.78 head-Y
```

## STATIC details

| Field | Value | Source |
|---|---|---|
| age | 67 | cohort age_at_admit |
| male | M | patients.gender |
| mechanism_cat | B (blunt) | trauma_mechanisms=MACHINE |
| transfer | D (direct) | admission_location=EMERGENCY ROOM |
| initial_ed_sbp | 128 mmHg | ED vitalsign (triage was empty, vitalsign fallback) |
| rsi | 1.78 | 128 / 72 (SBP/HR) |
| head_injury | Y | ICD-9: 80121, 85101, 80001, 80225 |

## First-48h

```
base_def_48=5.0  lactate_48=1.1  rbc_48=350 mL  crys_48=3050 mL
```

## File structure

```
Input design/patient sample_processed/hadm_26798055/
├── 00_static_fields.csv
├── 01_hourly_current_fields_first312h.csv      (312h x 20 cols)
├── 01_hourly_observed_recency_metadata.csv
├── 02_first48h_fields.csv
├── 03_daily_summary_preview.csv                (13 days)
├── 04_field_provenance.csv
├── manifest.json
└── README.md
```
