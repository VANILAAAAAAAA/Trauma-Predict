# MIMIC-IV Trauma Cohort

## Cohort definition

| Layer | Role | Rule | HADM | ICU stays | Excluded |
|---|---:|---:|---:|---:|---:|
| C0 | source universe | MIMIC-IV hospital admissions | 431,231 | 73,181 | — |
| C1 | valid ICU data | patient row + ICU stay + CHARTEVENTS | 66,239 | 73,181 | 364,992 |
| C2 | trauma candidate | C1 + clean ICD-9/10 E-code trauma match | 8,141 | 9,135 | 58,098 |
| C3 | adult trauma | C2 + age 18–89 | 7,456 | 8,381 | 685 |
| **C4** | **primary cohort** | C3 + hospital LOS ≥ 48h | **6,583** | **7,507** | 873 |
| C5 | ventilated subset | C4 + ventilator days ≥ 3 | 1,738 | 2,232 | 4,845 |

**Current research cohort is fixed at C4: 6,583 HADM / 7,507 ICU stays.**

C5 is retained only as an optional ventilated subset for sensitivity or future sepsis/ventilation-specific analysis. It is **not** the current cohort denominator and should not be used for current sample construction unless explicitly requested.

## Exclusion breakdown

| Transition | Reason | Count |
|---|---|---:|
| C0→C1 | no ICU stay | 364,992 |
| C1→C2 | non-trauma admission | 58,098 |
| C2→C3 | age > 89 | 685 |
| C3→C4 | LOS < 48h died | 199 |
| C3→C4 | LOS < 48h alive | 674 |
| C4→C5 | not intubated | 3,467 |
| C4→C5 | intubated < 3 days | 1,378 |

## Cohort tables

| File | Content | Rows | Unique HADM |
|---|---|---|---|
| `cohort/mimiciv_trauma_cohort_los48.csv` | C4 primary cohort, one row per ICU stay | 7,507 | 6,583 |
| `cohort/mimiciv_trauma_cohort_vent3_subset.csv` | C5 ventilated subset | 2,232 | 1,738 |
| `cohort/mimiciv_trauma_cohort_membership.csv` | HADM-level membership index | 6,583 | 6,583 |

Primary cohort key columns:

```
subject_id    hadm_id    stay_id
admittime    dischtime    hospital_los_hours    hospital_expire_flag
age_at_admit    gender    anchor_age    anchor_year
intime    outtime    icu_los_days
vent_day_count    is_vent3_subset    has_chartevents_data
trauma_icd_codes    trauma_icd_versions    trauma_seq_nums
trauma_mechanisms    trauma_intents    trauma_types
```

Membership key columns:

```
subject_id    hadm_id    n_icu_stays    in_c4_los48    in_c5_vent3
vent_day_count    hospital_los_hours    age_at_admit
trauma_icd_codes    trauma_icd_versions    trauma_mechanisms
```

## E-code dictionary

Clean pre-normalized workbook: `dictionary/qualified_traumatic_Ecodes_clean.xlsx`

- ICD-9: E-prefixed 5-char no-dot (e.g. `E8120`, `E8160`)
- ICD-10: no-dot uppercase (e.g. `W19XXXA`, `V4352XA`)
- 111 ICD-10 neglect codes pre-removed
- 2,870 unique codes: 740 ICD-9 + 2,130 ICD-10

## Matching contract

**Workbook side:** clean exact allowlist.

**MIMIC-IV ICD-9:** require E-prefix, pad short codes to 5 chars, exact match.

**MIMIC-IV ICD-10:** strip non-alphanumeric, exact match.

## Known correction

Previous extractor produced shorter cohorts (1,511 / 1,580 HADM) due to missing
ICD-9 E-code trailing-zero padding. Some Excel entries were read as `E816` instead
of `E8160`, causing exact-match failures against MIMIC-IV's `E8160`. ~158 trauma
admissions were recovered after fixing.

## Files

```
├── README.md
├── cohort/
│   ├── mimiciv_trauma_cohort_los48.csv       ← C4 primary cohort
│   ├── mimiciv_trauma_cohort_vent3_subset.csv  ← C5 ventilated subset
│   └── mimiciv_trauma_cohort_membership.csv  ← HADM-level index
├── metadata/
│   ├── cohort_layers.md
│   ├── cohort_layers.csv
│   └── cohort_exclusion_summary.csv
├── dictionary/
│   └── qualified_traumatic_Ecodes_clean.xlsx
├── pipeline/
│   ├── README.md                              ← read first; script map
│   └── 01_cohort_extraction/
│       ├── extract_mimiciv_trauma_cohort.py
│       └── mimiciv_trauma_cohort_config.json
```

Raw one-HADM MIMIC-IV source-row sample is stored separately:

```text
../trauma MIMICIV sample/
├── README.md
├── pipeline/
│   ├── extract_one_mimiciv_raw_sample.py
│   └── sample_MtoU_raw_config.json
└── samples/
    └── hadm_20002252/
```

## Re-run

Cohort extraction from the project workspace:

```bash
cd "/home/vanila/code/EHR-Predict/data dicision/trauma cohort/pipeline/01_cohort_extraction"
python3 extract_mimiciv_trauma_cohort.py --config mimiciv_trauma_cohort_config.json
```

Expected layer result:

```text
C4_LOS: hadm=6583 stays=7507
C5_FINAL: hadm=1738 stays=2232
```

The extractor writes its C5 CSV to the D: package path `mimiciv_trauma_cohort_clean.csv`; the active project cohort files are the standard C4/C5 files listed above.

One-HADM raw MIMIC-IV source-row sample extraction:

```bash
cd "/home/vanila/code/EHR-Predict/data dicision/trauma MIMICIV sample"
python3 pipeline/extract_one_mimiciv_raw_sample.py --config pipeline/sample_MtoU_raw_config.json
```

Read first:

```text
pipeline/README.md
```

## Source data

MIMIC-IV v2.2 local paths (not included):

```
/mnt/d/Data/mimic-iv-2.2/hosp/admissions.csv.gz
/mnt/d/Data/mimic-iv-2.2/hosp/patients.csv.gz
/mnt/d/Data/mimic-iv-2.2/hosp/diagnoses_icd.csv.gz
/mnt/d/Data/mimic-iv-2.2/icu/icustays.csv.gz
/mnt/d/Data/mimic-iv-2.2/icu/chartevents.csv.gz
```

## Limitations

- ISS/AIS severity scoring not applied.
- Primary key `hadm_id`; CSV rows are per ICU stay.
- Age >89 excluded per MIMIC policy.
- Mechanism labels from E-code workbook, not clinical adjudication.
