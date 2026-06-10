# Field Summary

| Group | Fields                                                       | Count |
| :---- | :----------------------------------------------------------- | :---- |
| G1    | `hr, sbp, dbp, map, rr, temp, fio2`                          | 7     |
| G2    | `age, male, mechanism_cat, transfer, initial_ed_sbp, rsi, head_injury` | 7     |
| G2*   | `base_def_48, lactate_48, rbc_48, crys_48`                   | 4     |
| G3    | `bolus_sum_until_h, rbc_sum_until_h, vent_h, vent_day_sum_until_h` | 4     |
| G4    | `bicarb, strong_ion, bun, creatinine, wbc, lymphocytes, neutrophils, uop` | 8     |

## G1 — Vital Signs

| Field | Unit | Description |
|---|---:|---|
| `hr` | bpm | Heart rate |
| `sbp` | mmHg | Systolic blood pressure |
| `dbp` | mmHg | Diastolic blood pressure |
| `map` | mmHg | Mean arterial pressure |
| `rr` | breaths/min | Respiratory rate |
| `temp` | °C | Temperature |
| `fio2` | fraction | Fraction of inspired oxygen |

## G2 — Static Profile

| Field | Unit | Description |
|---|---:|---|
| `age` | years | Age at admission |
| `male` | binary | Sex indicator; 1 = male, 0 = female |
| `mechanism_cat` | category | Injury mechanism category |
| `transfer` | category | Transfer or arrival context before ICU |
| `initial_ed_sbp` | mmHg | Initial ED systolic blood pressure |
| `rsi` | ratio | Reverse shock index = SBP / HR |
| `head_injury` | binary | Head injury from diagnoses_icd (S00-S09 / 800-854) |

## G2* — First 48h Summary

| Field | Unit | Description |
|---|---:|---|
| `base_def_48` | mEq/L | First-48h base deficit summary |
| `lactate_48` | mmol/L | First-48h lactate summary |
| `rbc_48` | mL | First-48h RBC transfusion volume |
| `crys_48` | mL | First-48h crystalloid volume |

## G3 — Cumulative Exposures

| Field | Unit | Description |
|---|---:|---|
| `bolus_sum_until_h` | mL | Cumulative crystalloid exposure until current hour |
| `rbc_sum_until_h` | mL | Cumulative RBC transfusion exposure until current hour |
| `vent_h` | binary | Current ventilation status |
| `vent_day_sum_until_h` | days | Cumulative ventilation days until current hour |

## G4 — Laboratory Result

| Field | Unit | Description |
|---|---:|---|
| `bicarb` | mEq/L | Bicarbonate |
| `strong_ion` | mEq/L | Strong ion difference proxy |
| `bun` | mg/dL | Blood urea nitrogen |
| `creatinine` | mg/dL | Creatinine |
| `wbc` | K/uL | White blood cell count |
| `lymphocytes` | K/uL | Absolute lymphocyte count |
| `neutrophils` | K/uL | Absolute neutrophil count |
| `uop` | mL/h | Urine output |
