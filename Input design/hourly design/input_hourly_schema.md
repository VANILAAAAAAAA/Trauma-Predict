# Hourly Input Schema

Source field boundary: `data dicision/field adapter/field.md`  
Temporal contract: first 13 ICU days = 312h; input at hour h ≤ h end only.

---

## Block Layout

Each training sample at landmark hour `t`:

```text
[STATIC]       — one row, admission/pre-ICU fields
[HOURLY 1..H]  — H = min(t, last_observed_hour), each row = one hour
```

No daily summary in the hourly input schema. Daily design is in `summary design/`.

---

## 1. STATIC Block

One row per admission. Values do not change with t.

| Field | Type | Unit | Description |
|---|---|---|---|
| `age` | float | years | Age at admission; >89 truncated |
| `male` | int {0,1} | — | 1 = male |
| `mechanism_cat` | int {1,2,3} | — | 1=blunt, 2=penetrating, 3=other |
| `transfer` | int {0,1} | — | Transfer/arrival context |
| `initial_ed_sbp` | float | mmHg | Initial ED SBP; NA if missing |

---

## 2. HOURLY Row Structure

Each row = one hour `h` (0-indexed from ICU admission).

Every field carries:

```text
{var}_value        — value at hour h (LOCF if no new measurement)
{var}_observed     — 1 if new measurement in hour h, else 0
{var}_recency_h    — hours since last measurement; 0 if observed this hour
```

LOCF rule:

```text
if observed in hour h:
    value_h = aggregate(valid raw rows in hour h)
    observed_h = 1
    recency_h = 0
else:
    value_h = latest observed value before h
    observed_h = 0
    recency_h = h - last_observed_hour
```

If no prior value exists for this variable in this ICU stay:

```text
value_h = NA
observed_h = 0
recency_h = NA (or sentinel value)
```

---

## 3. G1 — Vital Signs (7 fields)

Hourly dense physiology. Aggregation: mean of valid raw rows in hour h.

| Field | Unit | MIMIC source |
|---|---|---|
| `hr` | bpm | chartevents 220045 |
| `sbp` | mmHg | chartevents 220050/225309/220179/224167/227243 |
| `dbp` | mmHg | chartevents 220051/225310/220180/224643/227242 |
| `map` | mmHg | chartevents 220052/220181 |
| `rr` | breaths/min | chartevents 220210 |
| `temp` | °C | chartevents 223761/223762/226329 |
| `fio2` | fraction | chartevents 223835 |

Unit conversions:

```text
temp: if Fahrenheit → temp_c = (temp_f - 32) * 5 / 9
fio2: if raw > 1 and raw <= 100 → fio2 = raw / 100
```

Columns per field: `{var}_value`, `{var}_observed`, `{var}_recency_h`

Total G1 columns: `7 × 3 = 21`

---

## 4. G3 — Treatment / Exposure (4 final, 2 intermediate)

G3 fields at hour h are cumulative-until-h. Intermediate hourly delta fields are also exposed for the model.

### Final fields

| Field | Unit | Formula |
|---|---|---|
| `bolus_sum_until_h` | mL | `sum_{τ≤h}(iv_fluid_ml_1h_τ)` |
| `rbc_sum_until_h` | mL | `sum_{τ≤h}(rbc_ml_1h_τ)` |
| `vent_h` | int {0,1} | 1 if invasive ventilation interval overlaps hour h |
| `vent_day_sum_until_h` | days | count distinct ICU days d ≤ h with any vent_h=1 |

Ventilation analysis marker:

```text
vent_ge3d_by_h = 1 if vent_day_sum_until_h ≥ 3 else 0
```

Computed from history up to h only. Final C5 membership for stratified evaluation only.

### Intermediate fields (hourly delta)

| Field | Unit | Formula |
|---|---|---|
| `iv_fluid_ml_1h` | mL/h | crystalloid volume overlapping hour h |
| `rbc_ml_1h` | mL/h | RBC volume overlapping hour h |

G3 fields do not use LOCF. Unobserved hours = 0 for delta fields.

Columns:

```text
Final: bolus_sum_until_h, rbc_sum_until_h, vent_h, vent_day_sum_until_h
Intermediate: iv_fluid_ml_1h, rbc_ml_1h
Total G3: 6 columns
```

---

## 5. G4 — Labs / Output (8 final, 4 intermediate)

### Final lab-memory fields

Lab values use latest-observation-before-h with recency metadata.

| Field | Unit | MIMIC source |
|---|---|---|
| `bicarb` | mEq/L | labevents 50803/50882/51739 |
| `bun` | mg/dL | labevents 51006/52647 |
| `creatinine` | mg/dL | labevents 50912/52546 |
| `wbc` | K/uL | labevents 51300/51301 |
| `lymphocytes` | K/uL | labevents 51133/52769/53132 (absolute count) |
| `neutrophils` | K/uL | labevents 52075/53133 (absolute count) |
| `strong_ion` | mEq/L | derived: `(Na + K) - (Cl + bicarb)` |
| `uop` | mL/h | outputevents urine itemids |

### Intermediate lab fields

Needed to compute `strong_ion` and `base_def_48` (G2* conditional field in daily design).

| Field | Unit | MIMIC source |
|---|---|---|
| `na` | mEq/L | labevents 50983/52623/50824 |
| `k` | mEq/L | labevents 50971/52610/50822 |
| `cl` | mEq/L | labevents 50902/52535/50806 |
| `base_excess` | mEq/L | labevents 50802 |

Columns per field: `{var}_value`, `{var}_observed`, `{var}_recency_h`

```text
Final: 8 × 3 = 24 columns
Intermediate: 4 × 3 = 12 columns
uop is hourly output delta — no recency needed (use value only, or value + observed)
```

Total G4: 36 columns (or 35 if uop has value only)

---

## 6. Special Hour Markers

| Field | Type | Description |
|---|---|---|
| `hour_index` | int | ICU-relative hour, 0-based |
| `is_current_hour` | int {0,1} | 1 only at h=t (current landmark) |

Use `hour_index` for positional encoding, not as raw model input.

---

## 7. Column Count Summary

| Block | Columns | Notes |
|---|---|---|
| STATIC | 5 | G2 static fields |
| HOURLY (per hour) | 63 | 21 (G1) + 6 (G3) + 36 (G4) |

H hours of history → `5 + 63 × H` columns for flat input, or `5` static tokens + `H × ~63` structured tokens for sequence input.

---

## 8. Not in Hourly Input

Fields deliberately excluded from hourly input:

| Field | Reason |
|---|---|
| All G2* fields | Conditional; appear only in daily/summary blocks when history ≥ 48h |
| `rsi`, `er_disp_cat`, `head_injury` | Hold; constructibility/formula unconfirmed |
| All `*Cat` fields | Hold; thresholds/codebook unconfirmed |
| `apache` | Exclude; no native MIMIC-IV field |
| `surgSum`, `surgHours` | Audit pending; OR interval construction unconfirmed |
| `abx_48`, `surg_48` | Audit pending |
| `Sepsis`, `infectionDay`, `infectionHour` | Labels; never input |

---

## 9. Temporal Contract

```text
max_landmark_h = 312h (13 days)
input window: history ≤ t
target window: t+1 (next-hour) or (t, t+12] (12h report)
post-312h: administrative censoring
```

For next-hour target at landmark t:

```text
if t + 1 > last_observed_hour_this_stay:
    skip this training sample
```
