# Report Label Schema (12h Future Window)

Source fields: `data dicision/field adapter/field.md` (include-only)  
Target: deterministic labels from future canonical hours `(t, t+12]`.

---

## Design Rules

1. Labels are constructed from `S_{t+1:t+12}` (future canonical hourly state).
2. Labels are **multi-label binary**, with a few multi-class.
3. Every label carries a **validity mask** for censoring/missingness.
4. Labels depend only on fields confirmed in the include-only field.md.
5. Window truncation: if `t+12 > 312h` or `t+12 > last_observed_hour`, mask affected labels.
6. Labels never enter encoder input — they exist only as training targets.

---

## 1. Window Metadata

Every sample carries these metadata labels about the future window:

| Label | Type | Description |
|---|---|---|
| `meta_observed_hours` | int [0,12] | Hours in window with ≥1 chart event |
| `meta_window_truncated` | int {0,1} | 1 if window < 12h (censored/discharged) |
| `meta_censored_by_death` | int {0,1} | 1 if patient died inside window |
| `meta_censored_by_discharge` | int {0,1} | 1 if patient discharged inside window |

---

## 2. Hemodynamic Labels

### `hemo_map_lt65_ge2h` (binary)

```python
map_vals = [s.map_value for s in S_future if s.map_observed == 1]
hours_lt65 = count(map_vals < 65)
label = 1 if hours_lt65 >= 2 else 0
mask = 1 if count(map_vals not NA) >= 6 else 0
```

### `hemo_map_lt65_ge6h` (binary)

```python
label = 1 if hours_lt65 >= 6 else 0
mask = 1 if count(map_vals not NA) >= 8 else 0
```

### `hemo_map_worst_bin` (multi-class: 0/1/2)

```python
map_min = min(map_vals where observed)
label: 0=normal(map_min≥70), 1=low(65≤map_min<70), 2=severe(map_min<65)
mask = 1 if map_min is not None else 0
```

### `hemo_sbp_lt90_any` (binary)

```python
sbp_vals = [s.sbp_value for s in S_future if s.sbp_observed == 1]
label = 1 if any(sbp < 90 for sbp in sbp_vals) else 0
mask = 1 if count(sbp_vals not NA) >= 4 else 0
```

---

## 3. Respiratory Labels

### `resp_fio2_increase_ge015` (binary)

```python
fio2_now = S_t.fio2_value
fio2_max = max([s.fio2_value for s in S_future if s.fio2_observed == 1])
label = 1 if fio2_max - fio2_now > 0.15 else 0
mask = 1 if fio2_now is not NA and any fio2 observed in window else 0
```

### `resp_fio2_ge05_any` (binary)

```python
label = 1 if any(fio2 ≥ 0.5 for fio2 in fio2_vals) else 0
mask = 1 if any fio2 observed in window else 0
```

### `resp_vent_started` (binary)

```python
vent_now = S_t.vent_h
vent_future = [s.vent_h for s in S_future]
label = 1 if vent_now == 0 and any(v == 1 for v in vent_future) else 0
mask = 1  # vent is always observable
```

### `resp_vent_continued` (binary)

```python
label = 1 if vent_now == 1 and all(v == 1 for v in vent_future) else 0
mask = 1
```

---

## 4. Renal Labels

### `renal_creatinine_rise_ge03` (binary)

```python
cr_now = S_t.creatinine_value
cr_future = [s.creatinine_value for s in S_future if s.creatinine_observed == 1]
label = 1 if cr_future[-1] - cr_now ≥ 0.3 else 0
mask = 1 if cr_now is not NA and cr_future is not empty else 0
```

Clinical basis: KDIGO AKI Stage 1 — Cr rise ≥ 0.3 mg/dL within 48h; 12h window is conservative subset.

### `renal_uop_low_ge6h` (binary)

```python
uop_threshold = 30  # mL/h proxy; use 0.5 × weight_kg if available
uop_vals = [s.uop_value for s in S_future if s.uop_observed == 1]
low_hours = count(uop < uop_threshold for uop in uop_vals)
label = 1 if low_hours ≥ 6 else 0
mask = 1 if count(uop_vals not NA) ≥ 8 else 0
```

Clinical basis: SOFA Renal — UOP < 0.5 mL/kg/h.

---

## 5. Metabolic / Perfusion Labels

### `meta_lactate_gt2` (binary)

```python
lac_vals = [s.lactate_value for s in S_future if s.lactate_observed == 1]
label = 1 if max(lac_vals) > 2.0 else 0
mask = 1 if len(lac_vals) ≥ 1 else 0
```

Note: `lactate` is an intermediate variable in the hourly schema, not a UW final field. It is needed to compute `lactate_48` (G2*) and these metabolic labels.

Clinical basis: Sepsis-3 — lactate > 2 mmol/L = tissue hypoperfusion.

### `meta_lactate_gt4` (binary)

```python
label = 1 if max(lac_vals) > 4.0 else 0
mask = 1 if len(lac_vals) ≥ 1 else 0
```

Clinical basis: SSC guideline — lactate > 4 = severe.

### `meta_lactate_rising_gt05` (binary)

```python
if len(lac_vals) ≥ 2:
    label = 1 if lac_vals[-1] - lac_vals[0] > 0.5 else 0
mask = 1 if len(lac_vals) ≥ 2 else 0
```

### `meta_be_worse` (binary)

```python
be_now = S_t.base_excess_value
be_future = [s.base_excess_value for s in S_future if s.base_excess_observed == 1]
if be_now is not NA and be_future is not empty:
    label = 1 if be_future[-1] < be_now - 2 else 0  # BE drops 2+ mEq/L
mask = 1 if be_now is not NA and be_future is not empty else 0
```

Note: `base_excess` is an intermediate variable for `base_def_48` computation.

---

## 6. Critical Events

### `event_death_12h` (binary)

```python
label = 1 if patient died within (t, t+12]
mask = 1  # always available from cohort
```

### `event_discharge_12h` (binary)

```python
label = 1 if patient discharged within (t, t+12]
mask = 1
```

---

## 7. Label Summary

| Category | Labels | Count |
|---|---|---|
| Window metadata | `meta_observed_hours`, `meta_window_truncated`, `meta_censored_by_death`, `meta_censored_by_discharge` | 4 |
| Hemodynamic | `hemo_map_lt65_ge2h`, `hemo_map_lt65_ge6h`, `hemo_map_worst_bin`, `hemo_sbp_lt90_any` | 4 |
| Respiratory | `resp_fio2_increase_ge015`, `resp_fio2_ge05_any`, `resp_vent_started`, `resp_vent_continued` | 4 |
| Renal | `renal_creatinine_rise_ge03`, `renal_uop_low_ge6h` | 2 |
| Metabolic | `meta_lactate_gt2`, `meta_lactate_gt4`, `meta_lactate_rising_gt05`, `meta_be_worse` | 4 |
| Events | `event_death_12h`, `event_discharge_12h` | 2 |

**Total: 20 labels per landmark sample.**

---

## 8. Labels NOT Included

Labels deliberately excluded because their construction depends on fields not confirmed in include-only boundary:

| Excluded Label | Depends On | Reason |
|---|---|---|
| `hemo_si_gt1_any` | HR/SBP ratio (Shock Index) | `rsi` field is hold — formula direction unconfirmed |
| `infx_new_abx` | Antibiotic prescriptions | `abx_48` audit pending; prescription table not in registry |
| `infx_new_culture` | Microbiology cultures | Culture table not yet mapped |
| `infx_suspected` | abx + culture | Derivative of excluded labels |
| `event_sepsis_onset` | Sepsis label | `Sepsis` is label-only; sepsis construction pending |
| `comp_any_deterioration` | Composite | Derivative; compute from individual labels post-hoc |
| `comp_multi_system` | Composite | Derivative; compute from individual labels post-hoc |

---

## 9. Training Mask Rule

For each label:

```python
if meta_window_truncated == 1:
    # window is incomplete → mask labels that require full 12h
    mask = 0

if meta_censored_by_death == 1:
    # death is an absorbing state → mask physiology labels after death
    mask = 0

# For individual labels, also mask if required observations are missing
# (e.g., no MAP observations in window → mask hemo labels)
```

The model head should produce predictions for all labels, but only compute loss on unmasked labels.
