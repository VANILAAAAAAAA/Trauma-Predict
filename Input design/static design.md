# STATIC Design

One row per HADM. All fields known at or before ICU admission.

---

## Format

Single line, `field-value` pairs separated by `-`, spaced between pairs.

```
age-50 male-M mech-B tr-D ed_sbp-120 rsi-2.3 head-Y
```

Categorical values: single uppercase letter.
Continuous values: number.
Missing: `NA`.

---

## Fields

| Field | Type | Values | Rule |
|---|---|---|---|
| `age` | int | 18–89+ | cohort `age_at_admit`; >89 truncated |
| `male` | cat | `M` / `F` | patients.gender |
| `mech` | cat | `B` / `P` / `O` | B=blunt, P=penetrating, O=other |
| `tr` | cat | `D` / `T` | D=direct, T=transfer |
| `ed_sbp` | float | mmHg | ED triage SBP; `NA` if no ED linkage |
| `rsi` | float | ratio | SBP / HR from ED; `NA` if no ED linkage |
| `head` | cat | `Y` / `N` | ICD-10 S00-S09 or ICD-9 800-804,850-854 |

---

## Example

```
age-50 male-M mech-B tr-D ed_sbp-120 rsi-2.3 head-Y
age-35 male-F mech-P tr-T ed_sbp-NA rsi-NA head-N
```

---

## Provenance

| Field | Source |
|---|---|
| `age` | `00_cohort_rows.age_at_admit` |
| `male` | `00_cohort_rows.gender` → M/F |
| `mech` | `00_cohort_rows.trauma_mechanisms` + workbook |
| `tr` | `01_admissions_raw.admission_location` |
| `ed_sbp` | ED `triage.sbp`; `NA` if no linkage |
| `rsi` | ED SBP / ED HR; `NA` if no linkage |
| `head` | `04_diagnoses_icd_raw`; broad rule |

## Leakage

None. All values are admission-known or pre-ICU.
