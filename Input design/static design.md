# STATIC Design

One row per HADM. All fields known at or before ICU admission.

---

## Format

Single line, `field-value` pairs separated by `-`, spaced between pairs.

```
age-50 male-M mech-B tr-D ed_sbp-120
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
| `mech` | cat | `B` / `P` / `O` | B=blunt, P=penetrating, O=other; from trauma mechanisms |
| `tr` | cat | `D` / `T` | D=direct, T=transfer; from admission location/type |
| `ed_sbp` | float | mmHg | ED triage SBP; `NA` if no ED linkage |

---

## Example

```
age-50 male-M mech-B tr-D ed_sbp-120
```

```
age-35 male-F mech-P tr-T ed_sbp-NA
```

---

## Provenance

| Field | Source |
|---|---|
| `age` | `00_cohort_rows.age_at_admit` |
| `male` | `00_cohort_rows.gender` → M/F |
| `mech` | `00_cohort_rows.trauma_mechanisms` + workbook mapping |
| `tr` | `01_admissions_raw.admission_location` / `admission_type` |
| `ed_sbp` | ED `triage.sbp`; `NA` if no linkage |

---

## Leakage

None. All values are admission-known or pre-ICU.
