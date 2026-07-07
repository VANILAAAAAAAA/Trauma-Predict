# Kaggle Runbook

First target: GPU T4 x2 with fp16. P100 is the fallback path. TPU is not in the first training path because it adds XLA-specific debugging and checkpoint friction.

## Expected Inputs

Kaggle should receive a prebuilt data artifact mounted outside Git:

```text
dataset_manifest.json
sample_manifest.csv
train/*.jsonl.zst
val/*.jsonl.zst
test/*.jsonl.zst
```

The source MIMIC extraction, field adapter, sample builder, and patient split generation are performed before Kaggle.

## Launch

```bash
pip install -r requirements-kaggle.txt
accelerate launch \
  --config_file configs/accelerate/t4x2.yaml \
  notebooks/kaggle/train_kaggle.py \
  --config configs/train/t4x2_first_run.yaml
```

Fallback:

```bash
accelerate launch \
  --config_file configs/accelerate/single_gpu.yaml \
  notebooks/kaggle/train_kaggle.py \
  --config configs/train/t4x2_first_run.yaml
```

## Required Outputs

Write outputs under `/kaggle/working` or the configured output root:

- `metrics.jsonl`
- resumable checkpoints
- validation predictions
- run config snapshot
- environment snapshot

Do not commit these outputs to Git.
