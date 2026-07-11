# Stage A HOUR 24 vs 168 Token Ablation

## Decision under test

Does preserving every vital-hour pair as an encoder token improve NEXT_HOUR vital
forecasting over the archived Stage A v1 representation that aggregates all seven
vitals and ventilation into one token per hour?

This branch is the 168-token candidate only:

```text
branch: codex/stage-a-v3-field-hour-168-20260710
tag: stage-a-v3-field-hour-168-run-20260711
base commit: 5ce25c1
control archive: kaggle_stage_a_v1_hour24_numeric_projection_20260709
control run: t4x2_stage_a_hour
candidate run: t4x2_stage_a_field_hour_168
```

## Frozen experiment contract

| Contract item | 24-token control | 168-token candidate |
| --- | --- | --- |
| Dataset | `main_route_first_train_8h_v2` | Same artifact |
| Train/val/test | 31,980 / 4,378 / 3,895 | Same records and splits |
| Sample JSON | Existing Stage A v1 record | Unchanged |
| HOUR side tensor | `[L,7]` values, `[L,7]` masks, `[L,1]` vent | Same tensors |
| NEXT_HOUR target | Existing latest-observation target | Unchanged |
| Normalization | Stage A v1 fixed means/stds | Identical |
| Base model | `answerdotai/ModernBERT-base` | Identical |
| Seed | 20260708 | Identical |
| Steps | 4000 | Identical |
| LR/warmup/weight decay | `3e-5` / 500 / 0.01 | Identical |
| Loss | Masked SmoothL1, vital values only | Identical |
| VENT | Input only | Input only |
| Warm start | No | No |
| Adapter parameters | 333,824 | 333,824 |
| Adapter initialization | Seed 20260708 | Same module shapes/order and seed |
| Varied factor | One token per hour | Seven field tokens per hour |

`validate_field_hour_ablation_config` fails preflight if the candidate drifts
outside `model.hour_tokenization`, run naming, or output paths.

## Sample handling

No new Dataset is required. The serialized sample remains:

```text
input_text: STATIC + DAY + <H-23> ... <H0> + <STATE>
hour_values: [L,7]
hour_mask:   [L,7]
hour_vent:   [L,1]
target:      unchanged NEXT_HOUR values/mask
```

The difference is model-side only.

### Control path

```text
[value + mask] for seven vitals + VENT
  -> concatenate eight field features
  -> shared 512 -> 256 -> 768 hour network
  -> one encoder token per hour
```

### Candidate path

```text
for each vital in an hour:
  activate that vital's field slot + the same hour's VENT slot
  -> the same-shape shared 512 -> 256 -> 768 hour network
  -> one encoder token per vital-hour
```

Each `<H-k>` placeholder is expanded after tokenization and before ModernBERT:

```text
serialized: <H-01> <H0>
effective:  7 tokens for H-01 + 7 tokens for H0
```

A full 24-hour window therefore contributes exactly 168 HOUR tokens. VENT is
broadcast as context to the seven tokens for its hour; there is no VENT target,
VENT query, or extra 24-token trajectory.

## Kaggle execution

Upload and run:

```text
notebooks/kaggle/train_stage_a_field_hour_168.ipynb
```

Required runtime:

```text
GPU: T4 x2
Kaggle Secret for private GitHub: GITHUB_TOKEN
Kaggle Dataset or API access: vanilaaaa/trauma-predict-main-route-first-train-8h-v2
```

The launcher performs dependency checks, reconstructs the same dataset, checks
all split counts, validates the frozen config diff, hashes the runtime dataset,
scans effective expanded token lengths, runs a two-step DDP smoke test, streams
filtered training progress, trains 4000 steps, evaluates all validation records,
and creates a compressed output archive.

During formal training the page prints rank-zero `TRAIN_LOSS=` and `EVAL_LOSS=`
lines at every configured logging/evaluation interval, plus bounded status
milestones and a heartbeat every 300 seconds. Full subprocess output remains in
the run-local log. This is an observability-only launcher repair and does not
change model computation.

## Interpretation boundary

This is an ablation of **hour-level aggregation vs field-hour tokenization**, not
only raw token count. The representations necessarily differ in which field
features are presented together before the shared hour network, while adapter
parameter count and all data/training contracts are held fixed.

One paired seed can identify a large effect but cannot establish seed-robust
superiority. If the candidate improvement is near the expected run-to-run noise,
both 24-token and 168-token variants require additional matched seeds before the
architecture is selected.
