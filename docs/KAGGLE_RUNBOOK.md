# Kaggle Runbook

First target: GPU T4 x2 with fp16. P100 is the fallback path. TPU is not in the first training path because it adds XLA-specific debugging and checkpoint friction.

## Expected Inputs

Kaggle should receive a prebuilt data artifact mounted outside Git:

```text
dataset_manifest.json
sample_manifest.csv
train/*.jsonl.gz
val/*.jsonl.gz
test/*.jsonl.gz
```

The source MIMIC extraction, field adapter, sample builder, and patient split generation are performed before Kaggle.

## Code and Data Wiring

Do not store training samples in GitHub. Use GitHub only for code.

Recommended first run:

1. Create or upload a private Kaggle Dataset that contains the prebuilt artifact above.
2. Create a Kaggle Notebook with Internet enabled.
3. Add the private Dataset through the notebook's `Add Data` panel.
4. Clone this repository in the notebook.
5. Point `TRAUMA_PREDICT_DATA_ROOT` to the mounted dataset path under `/kaggle/input/...`.

Notebook setup cell:

```bash
git clone https://github.com/VANILAAAAAAAA/Trauma-Predict.git
cd Trauma-Predict
pip install -r requirements-kaggle.txt
```

Environment setup:

```python
import os

os.environ["TRAUMA_PREDICT_DATA_ROOT"] = "/kaggle/input/<your-private-dataset-name>"
os.environ["TRAUMA_PREDICT_OUTPUT_ROOT"] = "/kaggle/working/trauma-predict-runs"
```

Linking a Kaggle Notebook to GitHub is optional. For this project, cloning a pinned commit is more reproducible than relying on notebook sync state.

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
