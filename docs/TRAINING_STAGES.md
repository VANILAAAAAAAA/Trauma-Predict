# Training Stages

This file documents the retained textual V1 staged contract from `/home/vanila/code/EHR-Predict/llmwiki/input_textual_design_framework_v1.md`. The active structured Relation V2 route is a separate single model, not Stage A/B/C and not an ablation; its frozen contract is defined by `configs/train/p100_multires_event_v2_relation_v2.yaml` and the repository README.

## Stage A: NEXT_HOUR Values

Stage A trains only the auxiliary next-hour vital-value task:

```text
training_stage: stage_a_next_hour
input: hour_values + hour_mask + hour_vent
active targets: NEXT_HOUR values
inactive NEXT_HOUR target: ventilation
inactive targets: NEXT_24H
loss: L_next_hour_values only
checkpoint label: Stage A HOUR adapter pretraining
implementation status: runnable in this branch
```

`hour_vent` remains an input covariate inside the HOUR side tensor. It is not a Stage A target, label, loss term, or decision criterion.

The HOUR adapter is field-aware. For each hour, each vital is encoded as:

```text
field embedding + per-field scalar value projection + per-field mask embedding
```

Ventilation is encoded as:

```text
vent field embedding + vent state embedding
```

The seven vital field embeddings plus the ventilation input embedding are concatenated and projected into the encoder hidden size before injection at the matching `<H*>` placeholder. The Stage A preferred encoder config uses `answerdotai/ModernBERT-base`; the older Longformer config remains only in historical joint-baseline launchers.

Allowed active losses:

| Loss key | Active | Required weight |
| --- | --- | --- |
| `next_hour_values` | yes | positive |
| `next_hour_vent` | no | `0.0` |
| `next24_domain` | no | `0.0` |
| `next24_binary` | no | `0.0` |
| `next24_multiclass` | no | `0.0` |

The Stage A collator emits only `next_hour_values` and `next_hour_mask`. It does not emit `next_hour_vent` or any `NEXT_24H` labels. The model also gates loss computation with `active_losses`, so inactive losses cannot enter Stage A through a non-zero weight or an accidentally retained label tensor.

Primary metrics are normalized MAE/RMSE, normalized signed bias, raw MAE/RMSE/bias, and Pearson r for next-hour numeric vitals, always compared with the H0 carry-forward persistence baseline. Ventilation persistence can be reported as a diagnostic baseline only. Full Stage A configs export all validation predictions and include input-side `input.hour.h0` so the evaluator can compute H0 carry-forward from the same records.

Stage A full-run configs use `resume: true`. Resume is accepted only when the discovered checkpoint contains `training_stage_metadata.json` matching the current `training_stage`, `active_losses`, and `loss_weights`.

Kaggle Stage A configs disable tqdm and log training loss every 250 steps. Full command output is kept in run-local `logs/` files so Kaggle notebook stdout stays readable.

## Stage B: NEXT_24H

Stage B starts from a Stage A checkpoint and trains the main future-summary target:

```text
training_stage: stage_b_next24
checkpoint source: Stage A checkpoint
active targets: NEXT_24H
inactive targets: NEXT_HOUR
loss: L_summary only
implementation status: contract reserved; runner intentionally blocked until checkpoint loading is implemented
```

The contract validator requires `training.stage_a_checkpoint`. The training runner must load that Stage A checkpoint before Stage B is enabled. Until that loader exists, `train_kaggle.py` rejects Stage B before writing a run snapshot.

## Stage C: Alternating

Stage C is optional and explicit:

```text
training_stage: stage_c_alternating
active targets: NEXT_HOUR values + NEXT_24H
loss schedule: every k summary steps inserts one hour step
implementation status: contract reserved; runner intentionally blocked until alternating scheduling is implemented
```

The contract validator requires `training.alternating_summary_steps >= 1`. The training runner must implement the alternating schedule before Stage C is enabled. Until that scheduler exists, `train_kaggle.py` rejects Stage C before writing a run snapshot.

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
