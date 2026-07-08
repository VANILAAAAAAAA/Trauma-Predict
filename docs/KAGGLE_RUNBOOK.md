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

Notebook setup cell if the repository is public:

```bash
git clone https://github.com/VANILAAAAAAAA/Trauma-Predict.git
cd Trauma-Predict
git fetch origin --tags
git checkout --detach main-route-v1-training-20260708
pip install -r requirements-kaggle.txt
pip check
```

If the repository remains private, create a Kaggle Secret named `GITHUB_TOKEN`
with a read-only GitHub token, then clone without printing the token:

```python
from kaggle_secrets import UserSecretsClient
import subprocess

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
repo_url = f"https://x-access-token:{token}@github.com/VANILAAAAAAAA/Trauma-Predict.git"
subprocess.run(["git", "clone", repo_url], check=True)
subprocess.run(
    ["git", "-C", "Trauma-Predict", "remote", "set-url", "origin", "https://github.com/VANILAAAAAAAA/Trauma-Predict.git"],
    check=True,
)
```

Then:

```bash
cd Trauma-Predict
git fetch origin --tags
git checkout --detach main-route-v1-training-20260708
pip install -r requirements-kaggle.txt
pip check
```

Environment setup in the Kaggle notebook:

```bash
export TRAUMA_PREDICT_DATA_ROOT="/kaggle/input/<your-private-dataset-name>"
export TRAUMA_PREDICT_OUTPUT_ROOT="/kaggle/working/trauma-predict-runs"
test -f "$TRAUMA_PREDICT_DATA_ROOT/dataset_manifest.json"
test -f "$TRAUMA_PREDICT_DATA_ROOT/sample_manifest.csv"
find "$TRAUMA_PREDICT_DATA_ROOT" -maxdepth 2 -type f | sort | sed -n '1,40p'
```

Linking a Kaggle Notebook to GitHub is optional. For this project, cloning a pinned tag is more reproducible than relying on notebook sync state. Update the tag name only when the committed training contract changes.

The Kaggle requirements intentionally do not install `torch`. Use Kaggle's preinstalled CUDA PyTorch, then pin the Hugging Face stack from `requirements-kaggle.txt`. The notebook runtime guard fails if CUDA is unavailable, if `torch` was upgraded to an incompatible wheel, or if the required imports cannot load.

## Private Dataset Upload Pattern

The clean data path is:

```text
local EHR-Predict sample artifact -> private Kaggle Dataset -> Kaggle Notebook /kaggle/input mount
```

Google Drive can be used as a transfer or backup location, but the training
notebook should read from a Kaggle private Dataset whenever possible. Pulling
from Drive on every notebook run is slower, less reproducible, and requires
extra credential handling.

Local upload with Kaggle CLI:

```bash
python3 -m pip install kaggle
kaggle auth login

cp -r /tmp/trauma_predict_first_train_8h /tmp/kaggle_trauma_predict_first_train_8h
cat > /tmp/kaggle_trauma_predict_first_train_8h/dataset-metadata.json <<'JSON'
{
  "title": "trauma-predict-main-route-first-train-8h-v2",
  "id": "vanilaaaa/trauma-predict-main-route-first-train-8h-v2",
  "licenses": [{"name": "other"}]
}
JSON

kaggle datasets create -p /tmp/kaggle_trauma_predict_first_train_8h --dir-mode zip
```

For later refreshes:

```bash
kaggle datasets version \
  -p /tmp/kaggle_trauma_predict_first_train_8h \
  -m "Refresh first training artifact"
```

## Launch

Run `notebooks/kaggle/verify_private_dataset.ipynb` first. It handles both attached private Datasets under `/kaggle/input` and Kaggle API downloads into `/kaggle/working`, reconstructs `train/val/test`, and normalizes Kaggle-expanded `.jsonl` files back to the manifest-declared `.jsonl.gz` shard names.

The direct preflight command expects the reconstructed artifact root, not the raw Kaggle upload folder:

```bash
export TRAUMA_PREDICT_DATA_ROOT="/kaggle/working/trauma-predict-main-route-first-train-8h-v2"
export TRAUMA_PREDICT_OUTPUT_ROOT="/kaggle/working/trauma-predict-runs"

python notebooks/kaggle/train_kaggle.py \
  --config configs/train/t4x2_first_run.yaml \
  --dry-run
```

Run the token-length scan before training. It verifies that no sample exceeds the configured encoder window and writes a JSON summary into the run folder:

```bash
python notebooks/kaggle/scan_token_lengths.py \
  --dataset-config configs/dataset/first_train.yaml \
  --train-config configs/train/t4x2_first_run.yaml \
  --output-json "$TRAUMA_PREDICT_OUTPUT_ROOT/t4x2_first_run/token_length_summary.json"
```

Run the smoke config before the full run when the notebook session or dependency image has changed:

```bash
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  notebooks/kaggle/train_kaggle.py \
  --config configs/train/t4x2_smoke.yaml
```

Training entry after dry run, token scan, and smoke pass. Use `torchrun` for the first Kaggle
run; it avoids the `accelerate` CLI import path that can pull in incompatible
vision packages on Kaggle images.

```bash
export TRAUMA_PREDICT_DATA_ROOT="/kaggle/working/trauma-predict-main-route-first-train-8h-v2"
export TRAUMA_PREDICT_OUTPUT_ROOT="/kaggle/working/trauma-predict-runs"
export PYTHONPATH="/kaggle/working/Trauma-Predict/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

pip install -r requirements-kaggle.txt
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  notebooks/kaggle/train_kaggle.py \
  --config configs/train/t4x2_first_run.yaml
```

Fallback:

```bash
export TRAUMA_PREDICT_DATA_ROOT="/kaggle/working/trauma-predict-main-route-first-train-8h-v2"
export TRAUMA_PREDICT_OUTPUT_ROOT="/kaggle/working/trauma-predict-runs"
export PYTHONPATH="/kaggle/working/Trauma-Predict/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

python notebooks/kaggle/train_kaggle.py \
  --config configs/train/t4x2_first_run.yaml
```

Alternative single-GPU distributed launch:

```bash
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=1 \
  notebooks/kaggle/train_kaggle.py \
  --config configs/train/t4x2_first_run.yaml
```

## Required Outputs

Write outputs under `/kaggle/working` or the configured output root:

- `metrics.jsonl`
- resumable checkpoints
- `final_model/`
- validation predictions
- run config snapshot
- environment snapshot
- `training_result.json`

Do not commit these outputs to Git.
