# DAY Token Vocabulary

## Sequence

```text
[DAY_d]
G1 aggregate tokens + value tensor
G3 daily exposure tokens + value tensor
G4 summary tokens + value tensor
optional [FIRST48] G2* tokens + value tensor + bucket token
[SEP]
```

`d = 1..32`; first 12 completed days are expected, 20 extra days reserved.

## G1 Aggregate Tokens

| Pattern | Meaning |
|---|---|
| `[heart_rate_last]`, `[heart_rate_max]`, `[heart_rate_mean]`, `[heart_rate_observed_hours]` | HR daily summary |
| `[systolic_bp_last]`, `[systolic_bp_min]`, `[systolic_bp_mean]`, `[systolic_bp_observed_hours]` | SBP daily summary |
| `[diastolic_bp_last]`, `[diastolic_bp_min]`, `[diastolic_bp_mean]`, `[diastolic_bp_observed_hours]` | DBP daily summary |
| `[mean_arterial_pressure_last]`, `[mean_arterial_pressure_min]`, `[mean_arterial_pressure_mean]`, `[map_lt65_hours]`, `[map_lt70_hours]` | MAP daily summary |
| `[respiratory_rate_last]`, `[respiratory_rate_max]`, `[respiratory_rate_mean]` | RR daily summary |
| `[temperature_last]`, `[temperature_max]`, `[temperature_mean]` | temperature daily summary |
| `[fio2_last]`, `[fio2_max]`, `[fio2_mean]` | FiO2 daily summary |

## G3 Daily Exposure Tokens

| Token | Source field |
|---|---|
| `[crystalloid_daily_total]` | `iv_d` |
| `[rbc_daily_total]` | `rbc_d` |
| `[ventilation_hours]` | `vent_h` |
| `[ventilation_days_cumulative]` | `vday` |

## G4 Summary Tokens

| Pattern | Source field |
|---|---|
| `[bicarbonate_last|min|max]` | `bicarb` |
| `[strong_ion_difference_last|min|max]` | `si` |
| `[creatinine_last|min|max]` | `cr` |
| `[wbc_last|min|max]` | `wbc` |
| `[bun_last]` | `bun` |
| `[lymphocytes_last|min|max]` | `lymph` |
| `[neutrophils_last|min|max]` | `neut` |
| `[urine_output_daily_total]`, `[urine_output_low_hours]` | `uop` |

## FIRST48 Tokens

Appear only when history ≥48h.

| Field | Token | Bucket evidence |
|---|---|---|
| `base_def_48` | `[base_deficit_48h]` | `baseDef48Cat`: 0–2.9 / 3–5.9 / 6–9.9 / ≥10 |
| `lactate_48` | `[lactate_48h]` | `lactate48Cat`: ≤2.9 / 3.0–5.0 / ≥5.1 |
| `rbc_48` | `[rbc_48h]` | no Cat threshold in UW table |
| `crys_48` | `[crystalloid_48h]` | `crys48Cat`: 0–1999 / 2000–4992 / 5000–9984 / ≥10000 |

## Evidence-backed FIRST48 Buckets

| Field | Tokens |
|---|---|
| base deficit 48h | `[base_deficit_48h_bin_normal]`, `[base_deficit_48h_bin_mild]`, `[base_deficit_48h_bin_moderate]`, `[base_deficit_48h_bin_severe]` |
| lactate 48h | `[lactate_48h_bin_normal]`, `[lactate_48h_bin_mild]`, `[lactate_48h_bin_severe]` |
| crystalloid 48h | `[crystalloid_48h_bin_low_volume]`, `[crystalloid_48h_bin_moderate_volume]`, `[crystalloid_48h_bin_high_volume]`, `[crystalloid_48h_bin_very_high_volume]` |

## Rule

DAY aggregate buckets are not frozen unless backed by `tokenizer/reference/`.
