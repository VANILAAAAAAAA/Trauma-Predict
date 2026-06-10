# DAY Design

One block per completed 24h ICU period. `day-d` covers hours `[(d-1)*24, d*24-1]`. Block produced only when `h >= d*24`.

---

## When Daily Blocks Appear

At landmark `t`, only completed days are included:

```
day-1 ... day-D    where D = floor(t / 24)
```

For `t=50` → `day-1` and `day-2`.

---

## Format

Per day, 2–3 lines grouped by variable type. Values are aggregates over the 24h window.

```
day-d  G1_aggs...
       G3_aggs...
       G4_aggs...
```

No section markers between G1/G3/G4 lines.

---

## G1 — Physiology Aggregates

| Suffix | Meaning |
|---|---|
| `_min` | minimum in block |
| `_max` | maximum in block |
| `_last` | last value in block |
| `_mean` | mean in block |
| `_N` | observed hours in block |

Plus burden indicators for MAP:

| Field | Meaning |
|---|---|
| `map_lt65h` | hours with MAP < 65 |
| `map_lt70h` | hours with MAP < 70 |

Not all suffices apply to every variable. Core set per variable:

| Variable | Aggregates |
|---|---|
| `hr` | last, max, mean, N |
| `sbp` | last, min, mean, N |
| `dbp` | last, min, mean, N |
| `map` | last, min, mean, N, lt65h, lt70h |
| `rr` | last, max, mean, N |
| `temp` | last, max, mean, N |
| `fio2` | last, max, mean, N |

---

## G3 — Treatment Block Totals

| Field | Meaning |
|---|---|
| `iv_d` | crystalloid total this day (mL) |
| `rbc_d` | RBC total this day (mL) |
| `vent_h` | hours on ventilator this day |
| `vday` | cumulative vent days at end of block |

No per-minute deltas; daily summary only.

---

## G4 — Lab / Output Block Summaries

| Field | Aggregates |
|---|---|
| `bicarb` | last, min, max |
| `si` | last, min, max |
| `cr` | last, min, max |
| `wbc` | last, min, max |
| `bun` | last |
| `lymph` | last, min, max |
| `neut` | last, min, max |
| `uop_d` | total UOP this day (mL) |
| `uop_lowh` | hours with UOP < threshold |

All labs use `_last` as the end-of-day snapshot.

---

## G2* — First-48h Fields

Appear when day-2 is complete (history ≥ 48h). Fixed first-48h window, not per-day.

| Field | Meaning | Source |
|---|---|---|
| `base_def` | max(0, -min base_excess) in first 48h | intermediate `base_excess` |
| `lac48` | max lactate in first 48h | intermediate `lactate` |
| `rbc48` | sum RBC volume in first 48h | from hourly deltas |
| `crs48` | sum crystalloid volume in first 48h | from hourly deltas |

When available, appended to the day-2 block:

```
day-2  ... G1 G3 G4 lines ...
       first48 base_def-1.0 lac48-1.9 rbc48-278 crs48-700
```

---

## Example

```
day-1  hr_last-72 hr_max-128 map_min-62 map_max-85 map_lt65h-3 map_lt70h-6
       fio2_max-0.55 temp_max-38.4 rr_max-28 rr_last-16
       iv_d-1200 rbc_d-0 vent_h-24 vday-1
       cr_last-0.7 cr_max-0.9 wbc_last-13.2 bicarb_last-27
       si_last-8.1 si_min-7.5 lymph_last-0.7 uop_d-1200 uop_lowh-2
day-2  hr_last-68 hr_max-112 map_min-65 map_max-82 map_lt65h-1
       fio2_max-0.45 temp_max-37.8
       iv_d-800 rbc_d-278 vent_h-24 vday-2
       cr_last-0.8 wbc_last-11.0 bicarb_last-26
       si_last-8.0 uop_d-980 uop_lowh-0
       first48 base_def-1.0 lac48-1.9 rbc48-278 crs48-700
```

---

## Not Included

Hourly recency metadata (belongs to hour design), raw counts, intermediate lab variables.
