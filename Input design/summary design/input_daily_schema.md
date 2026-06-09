# Daily Input Schema (Summary Design)

Source field boundary: `data dicision/field adapter/field.md`  
Temporal contract: 13 days = 312h. Daily blocks cover completed 24h windows.

---

## Block Layout

Each training sample at landmark hour `t`:

```text
[STATIC]              — same as hourly schema
[DAILY 1..D]          — D = floor(t / 24), each block = one completed 24h period
[HOURLY 1..H]         — H = t mod 24, current-day hourly detail (see hourly design)
```

The daily blocks summarize completed full days. The current partial day is handled by the hourly block.

---

## 1. Daily Block Definition

Block index `d` (1-based) covers ICU hours:

```text
day d = hours [(d-1)*24, d*24 - 1]
```

Example:

```text
day 1 = hours [0, 23]
day 2 = hours [24, 47]
day 3 = hours [48, 71]
```

A daily block is produced only when the full 24h window is complete at landmark t:

```text
if t >= d * 24:
    produce DAY_d block
```

---

## 2. Daily Block Metadata

| Field | Type | Description |
|---|---|---|
| `day_index` | int | 1-based day index |
| `start_hour` | int | block start hour (0-based) |
| `end_hour` | int | block end hour (exclusive) |
| `n_observed_hours` | int | hours in block with ≥1 chart event |

---

## 3. G1 — Physiology (7 fields)

For each vital, aggregate over the 24h block.

| Aggregation | Suffix | Applies to |
|---|---|---|
| last value in block | `_last` | all G1 |
| minimum in block | `_min` | sbp, dbp, map |
| maximum in block | `_max` | hr, rr, temp, fio2 |
| mean in block | `_mean` | all G1 |
| observed hours in block | `_n_obs` | all G1 |

### Burden indicators

| Indicator | Formula | Applies to |
|---|---|---|
| `map_lt65_hours` | count(h in block where map < 65) | map |
| `map_lt70_hours` | count(h in block where map < 70) | map |
| `fio2_ge05_hours` | count(h in block where fio2 ≥ 0.5) | fio2 |

### G1 columns

```text
hr:     last, max, mean, n_obs
sbp:    last, min, mean, n_obs
dbp:    last, min, mean, n_obs
map:    last, min, mean, n_obs, map_lt65_hours, map_lt70_hours
rr:     last, max, mean, n_obs
temp:   last, max, mean, n_obs
fio2:   last, max, mean, n_obs, fio2_ge05_hours
```

Total G1 per daily block: ~32 columns

---

## 4. G3 — Treatment / Exposure (4 final)

### Block totals

| Field | Formula |
|---|---|
| `iv_fluid_total_day` | `sum(h in block, iv_fluid_ml_1h)` |
| `rbc_total_day` | `sum(h in block, rbc_ml_1h)` |
| `vent_hours_day` | `count(h in block with vent_h = 1)` |
| `vent_any_day` | `1 if vent_hours_day > 0 else 0` |

### Cumulative at block end

| Field | Formula |
|---|---|
| `bolus_sum_until_block_end` | cumulative crystalloid at end of this block |
| `rbc_sum_until_block_end` | cumulative RBC at end of this block |
| `vent_day_sum_until_block_end` | cumulative vent days at end of this block |

Note: `vent_day_sum_until_block_end` should equal `d` if ventilated every day through block d.

### G3 columns

```text
iv_fluid_total_day, rbc_total_day, vent_hours_day, vent_any_day,
bolus_sum_until_block_end, rbc_sum_until_block_end, vent_day_sum_until_block_end
```

Total G3 per daily block: 7 columns

---

## 5. G4 — Labs / Output (8 final)

### Lab summary per block

For each lab variable, summarize over the 24h block.

| Aggregation | Suffix |
|---|---|
| last observed value | `_last` |
| minimum in block | `_min` |
| maximum in block | `_max` |
| measurement count | `_n` |
| hours since last measurement at block end | `_recency` |

### Lab variables

```text
bicarb, bun, creatinine, wbc, lymphocytes, neutrophils, strong_ion
```

7 labs × 5 aggregations = 35 columns

### Output

| Field | Aggregation |
|---|---|
| `uop_total_day` | sum of uop over block |
| `uop_low_hours` | count(h where uop < threshold) |

Threshold pending; default proxy: 30 mL/h if no patient weight.

Total G4 per daily block: ~37 columns

---

## 6. Intermediate Lab Summary (for G2* computation)

These support base_def_48 and strong_ion in the first-48h conditional fields.

| Field | Aggregations |
|---|---|
| `base_excess` | last, min, n |
| `na` | last, n |
| `k` | last, n |
| `cl` | last, n |

4 intermediates × ~3 aggregations = ~12 columns

---

## 7. G2* — First-48h Conditional Fields

Appear **only when the completed block history covers the first 48h window**.

Rule:

```text
if latest completed day_index >= 2 (i.e., hours [0, 47] complete):
    include for that day and all subsequent days
```

These are fixed-first-48h summaries, not per-block.

| Field | Source | Formula |
|---|---|---|
| `base_def_48` | base_excess hours [0-47] | `max(0, -min(base_excess))` within first 48h |
| `lactate_48` | lactate hours [0-47] | `max(lactate)` within first 48h |
| `rbc_48` | rbc_ml_1h hours [0-47] | `sum(rbc_ml_1h)` within first 48h |
| `crys_48` | iv_fluid_ml_1h hours [0-47] | `sum(iv_fluid_ml_1h)` within first 48h |

When available, these 4 fields are included as static-like values in daily blocks.

---

## 8. Column Count Summary

| Section | Columns |
|---|---|
| Block metadata | 4 |
| G1 vitals | ~32 |
| G3 treatment | 7 |
| G4 labs | ~37 |
| G4 intermediates | ~12 |
| G2* conditional | 4 (if ≥ day 3) |

Per daily block: ~92–96 columns

For D completed days:

```text
STATIC: 5 columns
DAILY: D × ~94 columns
HOURLY: H × ~63 columns (see hourly schema)
```

---

## 9. Not in Daily Summary

Daily summary excludes constructs that duplicate hourly detail or belong to target/label side:

| Excluded | Reason |
|---|---|
| All `*Cat` fields | Hold; thresholds unconfirmed |
| `abx_48`, `surg_48` | Audit pending |
| `surgSum`, `surgHours` | OR interval audit pending |
| Hourly LOCF metadata | Belongs to hourly schema, not daily |
| Future summary/labels | Target side only |
| Vent daily trend/delta | Keep cumulative only in daily; delta in hourly |
