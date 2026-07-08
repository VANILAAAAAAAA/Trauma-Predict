# Training Stages

This repository follows the frozen textual V1 staged training contract from `/home/vanila/code/EHR-Predict/llmwiki/input_textual_design_framework_v1.md`.

## Stage A: NEXT_HOUR

Stage A trains only the auxiliary next-hour task:

```text
training_stage: stage_a_next_hour
active targets: NEXT_HOUR values + ventilation
inactive targets: NEXT_24H
loss: L_hour only
checkpoint label: Stage A HOUR adapter pretraining
```

Allowed active losses:

| Loss key | Active | Required weight |
| --- | --- | --- |
| `next_hour_values` | yes | positive |
| `next_hour_vent` | yes | positive |
| `next24_domain` | no | `0.0` |
| `next24_binary` | no | `0.0` |
| `next24_multiclass` | no | `0.0` |

The Stage A collator does not emit `NEXT_24H` labels to the model. The model also gates loss computation with `active_losses`, so `NEXT_24H` loss cannot enter Stage A through a non-zero weight or an accidentally retained label tensor.

Primary metrics are MAE/RMSE for next-hour numeric vitals and AUROC/F1 for next-hour ventilation.

## Stage B: NEXT_24H

Stage B starts from a Stage A checkpoint and trains the main future-summary target:

```text
training_stage: stage_b_next24
checkpoint source: Stage A checkpoint
active targets: NEXT_24H
inactive targets: NEXT_HOUR
loss: L_summary only
```

The config validator requires `training.stage_a_checkpoint` before Stage B can run.

## Stage C: Alternating

Stage C is optional and explicit:

```text
training_stage: stage_c_alternating
active targets: NEXT_HOUR + NEXT_24H
loss schedule: every k summary steps inserts one hour step
```

The config validator requires `training.alternating_summary_steps >= 1`. The current Stage A branch does not launch Stage C.

## Joint Baseline

The previously repaired full-route run is labeled:

```text
training_stage: joint_baseline
active targets: NEXT_HOUR + NEXT_24H
status: not Stage A
```

It must not be reported as HOUR-only pretraining or as the first step of the staged V1 contract.

## Required Run Declaration

Before launching any Kaggle or server training run, record these fields:

| Field | Required meaning |
| --- | --- |
| `run label` | Human-readable run name. |
| `training stage` | One of `stage_a_next_hour`, `stage_b_next24`, `stage_c_alternating`, `joint_baseline`. |
| `input surfaces` | STATIC/DAY/HOUR surfaces passed as input. |
| `active heads` | Heads whose labels are emitted and losses are active. |
| `inactive heads` | Heads present in the module but excluded from loss. |
| `loss weights` | Exact weights, including `0.0` for inactive losses. |
| `frozen/unfrozen modules` | Backbone, HOUR adapter, and heads status. |
| `checkpoint source` | Empty for Stage A from scratch; Stage A checkpoint for Stage B. |
| `expected output checkpoint label` | Stage-specific checkpoint identity. |
| `design match` | Why the run matches the frozen contract. |

## Kaggle Stage A Entry

Use `notebooks/kaggle/train_stage_a_hour.ipynb` with:

```text
full config: configs/train/t4x2_stage_a_hour.yaml
single-GPU fallback config: configs/train/p100_stage_a_hour.yaml
smoke config: configs/train/t4x2_stage_a_hour_smoke.yaml
dataset: vanilaaaa/trauma-predict-main-route-first-train-8h-v2
```
