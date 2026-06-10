# HOUR Design

One row per ICU hour `h`, indexed from intime. Each value carries recency: `|N` = hours since last measurement (`|0` = observed this hour).

---

## Format

```
hour-h G1... G3... G4...
```

G1 and G4 values with `|recency`. G3 cumulative/status without recency.

---

## G1 — Vital Signs (hourly, with recency)

| Field | Unit | Aggregation | Recency |
|---|---|---|---|
| `hr` | bpm | mean of valid chart rows in hour h | `|N` |
| `sbp` | mmHg | mean | `|N` |
| `dbp` | mmHg | mean | `|N` |
| `map` | mmHg | mean | `|N` |
| `rr` | /min | mean | `|N` |
| `temp` | °C | mean; convert F→C | `|N` |
| `fio2` | fraction | mean; convert %>1→fraction | `|N` |

LOCF: if no measurement this hour, carry last value forward, recency increases.

---

## G3 — Treatment / Exposure (no recency)

| Field | Unit | Rule |
|---|---|---|
| `bolus` | mL | cumulative crystalloid at end of hour h |
| `rbc` | mL | cumulative RBC at end of hour h |
| `vent` | 0/1 | ventilation active this hour |
| `vday` | days | cumulative vent days up to hour h |

No LOCF for G3. Unobserved delta hours = 0.

Intermediate deltas (`iv_ml_1h`, `rbc_ml_1h`) are used to compute cumulative values but not serialized.

---

## G4 — Labs / Output (with recency, n≥0)

| Field | Unit | Recency |
|---|---|---|
| `bicarb` | mEq/L | `|N` |
| `si` | mEq/L | `|N` |
| `bun` | mg/dL | `|N` |
| `cr` | mg/dL | `|N` |
| `wbc` | K/uL | `|N` |
| `lymph` | K/uL | `|N` |
| `neut` | K/uL | `|N` |
| `uop` | mL/h | no recency |

`si` = `(Na + K) - (Cl + bicarb)`; requires all four components available.

`lymph` / `neut`: prefer absolute count (K/uL); if only differential % available, convert via same-time WBC.

Lab LOCF: carry last value forward. Recency can exceed 24h for infrequent labs.

---

## Intermediate Labs (computation only)

Needed to derive `si`, G2* fields, and metabolic labels. Stored in canonical state but not serialized as text tokens.

`na`, `k`, `cl`, `base_excess`, `lactate`

---

## Hour Index

`hour-h` where `h` is ICU-relative (0 = first hour). Up to 312 (13 days).

---

## Full Example

Hour 0 (first ICU hour):

```
hour-0 hr-72|0 sbp-118|0 dbp-64|0 map-85|0 rr-16|0 temp-37.1|0 fio2-0.4|0 bolus-0 rbc-278 vent-1 vday-1 bicarb-27|0 si-8.1|0 cr-0.7|0 wbc-13.2|0 lymph-0.7|0 neut-4.5|0 uop-230
```

Hour 48 (day 3 start, no new labs):

```
hour-48 hr-69|0 sbp-131|0 dbp-69|0 map-73|0 rr-17|0 temp-36.5|0 fio2-0.4|0 bolus-900 rbc-278 vent-1 vday-3 bicarb-27|48 cr-0.7|48 wbc-13.2|48 uop-40
```

---

## Not Serialized

Intermediate variables, G2* first-48h fields (→ daily design), labels.
