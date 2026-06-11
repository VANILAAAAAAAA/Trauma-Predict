# HOUR Token Vocabulary

## Sequence

```text
[HOUR_h]
G1 field tokens + value tensor + recency tensor
G3 field tokens + value tensor
G4 field tokens + value tensor + recency tensor
[SEP]
```

`h = 0..24` within the current ICU day.

## G1 Vital Tokens

| Field | Token | Value path | Recency | Bucket |
|---|---|---|---|---|
| `hr` | `[heart_rate]` | value tensor | yes | pending evidence |
| `sbp` | `[systolic_bp]` | value tensor | yes | pending evidence |
| `dbp` | `[diastolic_bp]` | value tensor | yes | pending evidence |
| `map` | `[mean_arterial_pressure]` | value tensor | yes | pending evidence |
| `rr` | `[respiratory_rate]` | value tensor | yes | pending evidence |
| `temp` | `[temperature]` | value tensor | yes | pending evidence |
| `fio2` | `[fio2]` | value tensor | yes | pending evidence |

## G3 Treatment Tokens

| Field | Token | Value path | Recency | Bucket |
|---|---|---|---|---|
| `bolus_sum_until_h` | `[crystalloid_cumulative]` | value tensor | no | pending evidence |
| `rbc_sum_until_h` | `[rbc_cumulative]` | value tensor | no | pending evidence |
| `vent_h` | `[ventilation_status]` | categorical/value | no | no |
| `vent_day_sum_until_h` | `[ventilation_days_cumulative]` | value tensor | no | pending evidence |

## G4 Lab / Output Tokens

| Field | Token | Value path | Recency | Bucket |
|---|---|---|---|---|
| `bicarb` | `[bicarbonate]` | value tensor | yes | pending evidence |
| `strong_ion` | `[strong_ion_difference]` | value tensor | yes | pending evidence |
| `bun` | `[bun]` | value tensor | yes | pending evidence |
| `creatinine` | `[creatinine]` | value tensor | yes | pending evidence |
| `wbc` | `[wbc]` | value tensor | yes | pending evidence |
| `lymphocytes` | `[lymphocytes]` | value tensor | yes | pending evidence |
| `neutrophils` | `[neutrophils]` | value tensor | yes | pending evidence |
| `uop` | `[urine_output]` | value tensor | no | pending evidence |

## Rule

No HOUR bucket is frozen without reference evidence in `tokenizer/reference/`.
