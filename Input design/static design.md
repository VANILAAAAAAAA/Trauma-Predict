# STATIC Design

One row per HADM. All fields known at or before ICU admission.

---

## Format

Single line, `field-value` pairs separated by `-`, spaced between pairs.

```
age-67 male-M mech-B tr-D edsbp-128 rsi-1.78 head-Y
```

Categorical values: single uppercase letter.
Continuous values: number, rounded to 1 decimal.
Missing: `NA`.

---

## Fields

| Field | Type | Values | Rule |
|---|---|---|---|
| `age` | int | 18–89+ | cohort `age_at_admit`; >89 truncated |
| `male` | cat | `M` / `F` | patients.gender |
| `mech` | cat | `B` / `P` / `O` | B=blunt, P=penetrating, O=other; from `trauma_mechanisms` + workbook |
| `tr` | cat | `D` / `T` | D=direct admit, T=transfer; from `admissions.admission_location` |
| `edsbp` | float | mmHg | ED triage SBP; fallback to ED vitalsign; `NA` if no ED linkage |
| `rsi` | float | ratio | SBP / HR from ED; `NA` if no ED linkage |
| `head` | cat | `Y` / `N` | ICD-10 S00-S09 or ICD-9 800-804,850-854; ED source preferred, hosp ICD fallback |

---

## Examples (real extracted, HADM redacted)

Complete (all ED fields present):

```
age-67 male-M mech-B tr-D edsbp-128 rsi-1.78 head-Y
```

Partial (no ED linkage, ED fields NA):

```
age-50 male-M mech-B tr-D edsbp-NA rsi-NA head-N
```

---

## Provenance

| Field | Source |
|---|---|
| `age` | `00_static_fields.age` from cohort `age_at_admit` |
| `male` | `00_static_fields.male` from `patients.gender` → M/F |
| `mech` | `00_static_fields.mechanism_cat` from `trauma_mechanisms` → B/P/O |
| `tr` | `00_static_fields.transfer` from `admissions.admission_location` → D/T |
| `edsbp` | `00_static_fields.initial_ed_sbp` from ED `vitalsign.sbp`; `NA` if no linkage |
| `rsi` | `00_static_fields.rsi` from ED SBP / ED HR; `NA` if no linkage |
| `head` | `00_static_fields.head_injury` from ICD diagnoses; broad rule |

## First-48h Fields (not in STATIC line, gated by anchor_hour >= 48)

```
base_def_48=5.0  lactate_48=1.1  rbc_48=350 mL  crys_48=3050 mL
```

## Leakage

None. All values are admission-known or pre-ICU.
