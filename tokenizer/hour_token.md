# HOUR Token Vocabulary

> Updated 2026-06-16. Engineering contract: `vital_values[T,7]` + `vital_mask[T,7]`.
> Implementation: `/home/vanila/code/EHR-Predict/projector/model.py`.

## Scope

HOUR is the recent-window high-resolution ICU state. The engineering input is fixed dense tensors, not sparse text tokens.

**Engineering contract** (authoritative):

```python
vital_values: [B, T, 7]  # hr, sbp, dbp, map, rr, temp, fio2
vital_mask:   [B, T, 7]  # 1 = observed, 0 = missing
```

`T` defaults to `recent_h=24`. Implementation: `/home/vanila/code/EHR-Predict/projector/model.py` (VitalValueProjector).

Every hour has exactly 7 vital slots. Missing vitals are represented by `mask=0`, not by deleting the slot. FiO2 is the main sparse vital (MIMIC ~26% observed) but stays in the fixed 7-slot layout.

Numeric values only enter the value MLP when `mask=1`. Out-of-range values (e.g. SBP 95119, MAP 780 found in raw MIMIC chartevents) are filtered to `mask=0` before standardization. Physiological range filter is mandatory; see `projector/build_vital_dataset.py` `VALID_RANGE`.

## Design Principle

- HOUR vitals are fixed `[T,7]` tensors — not conditional, not text
- Missingness is `vital_mask[t,f]=0`, not LOCF, not zero-fill
- `[HOUR_REL_k]` encodes time position
- Sparse event channels (vent, bolus, RBC) are separate from the vital slots
- No cumulative values in HOUR; cumulative burden lives in DAY
- Labs and urine output are not HOUR tokens; they belong to DAY

## Per-slot encoding

Each of the `T * 7` slots carries, before projector pooling:

```text
field embedding     — which vital (hr, sbp, …, fio2)
time embedding      — which hour (-T+1 … 0)
mask embedding      — observed / missing
value MLP output    — standardized numeric value (masked)
```

After projection, slots enter the Transformer encoder as a flat `[T*7, D]` sequence. Two-dimensional attention (cross-vital + temporal) is handled by combined field+time positional embeddings.

## Sparse event channels

Separate from the fixed vital slots. One event type remains in HOUR:

| Source | Token | When |
|---|---|---|
| MIMIC procedureevents | `[vent_on]` | invasive ventilation active in this hour |

Vent is a conditional context signal, not a prediction target. It is repeated every active hour. Bolus and RBC hourly events have been moved to DAY summary: `[rbc_transfusion_event_present]` for RBC, and bolus/crystalloid remains hold until the MIMIC fluid registry is frozen.

## Unit Normalization

```text
FiO2: fraction [0, 1]; raw percent (>1) divided by 100
Temperature: °C; raw °F converted
Blood pressure: mmHg
MIMIC bolus/crystalloid: mL per hour after interval splitting
MIMIC RBC transfusion: mL per hour after interval splitting
```

Do not reinterpret UW source-coded values as mL unless a conversion is documented.

## Review rendering (not training input)

The compact text rendering below is for human review only. The training collator emits `vital_values[T,7]` and `vital_mask[T,7]` tensors.

```text
[HOUR_REL_k]
[heart_rate] <numeric> [systolic_bp] <numeric> [diastolic_bp] <numeric>
[mean_arterial_pressure] <numeric> [respiratory_rate] <numeric>
[temperature] <numeric> [fio2] <numeric>
[vent_on]                          # if active
[CUR]                              # only when k=0
[SEP]
```

## Event construction (MIMIC-IV)

Inputevents are interval events. Split amount over [starttime, endtime) by hour-level overlap. RBC itemids: 225168, 226368, 227070. Crystalloid itemid registry is not frozen; maintenance fluid, carriers, and diluents are not yet separated from resuscitation fluids.

For UW `bolusSum`/`RBCsum`: detect events by differencing successive cumulative values. Only positive-delta hours get event tokens. Source-scale deltas are not mL unless a conversion is documented.

## Physiological range filter

Mandatory before standardization. From `projector/build_vital_dataset.py`:

| Vital | Range |
|---|---|
| hr | 20–250 |
| sbp | 40–300 |
| dbp | 20–200 |
| map | 30–200 |
| rr | 4–80 |
| temp | 25–45 |
| fio2 | 0.21–1.0 |

Values outside range are set to `mask=0`. Do not clip into range.

## Related

- Implementation: `/home/vanila/code/EHR-Predict/projector/model.py`
- Dataset builder: `/home/vanila/code/EHR-Predict/projector/build_vital_dataset.py`
- DAY token dictionary: `tokenizer/day_token.md`
- STATIC token vocabulary: `tokenizer/static_token.md`
