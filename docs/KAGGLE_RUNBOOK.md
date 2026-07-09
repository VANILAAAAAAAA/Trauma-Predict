# Kaggle Runbook

First formal target: Stage A `NEXT_HOUR` vital-value training on GPU T4 x2 with fp16. `hour_vent` remains an input covariate but is not a Stage A target or loss. P100 is the fallback path. TPU is not in the first training path because it adds XLA-specific debugging and checkpoint friction.

## Expected Inputs

Kaggle should receive a prebuilt data artifact mounted outside Git:

```text
dataset_manifest.json
sample_manifest.csv
train/*.jsonl.gz
val/*.jsonl.gz
test/*.jsonl.gz
patient_split.csv      optional run metadata
anchor_plan.csv        optional run metadata
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
git checkout --detach stage-a-hour-field-aware-20260709
pip install -r requirements-kaggle.txt
python -m pip check || true
```

If the repository remains private, create a Kaggle Secret named `GITHUB_TOKEN`
with a read-only GitHub token, then clone without printing the token:

```python
from kaggle_secrets import UserSecretsClient
import subprocess

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
repo_url = f"https://x-access-token:{token}@github.com/VANILAAAAAAAA/Trauma-Predict.git"
result = subprocess.run(
    ["git", "clone", repo_url],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=False,
)
if result.returncode != 0:
    raise RuntimeError("Private GitHub clone failed; token was not printed.")
subprocess.run(
    ["git", "-C", "Trauma-Predict", "remote", "set-url", "origin", "https://github.com/VANILAAAAAAAA/Trauma-Predict.git"],
    check=True,
)
```

Then:

```bash
cd Trauma-Predict
git fetch origin --tags
git checkout --detach stage-a-hour-field-aware-20260709
pip install -r requirements-kaggle.txt
python -m pip check || true
```

Environment setup in the Kaggle notebook:

```bash
export TRAUMA_PREDICT_DATA_ROOT="/kaggle/input/<your-private-dataset-name>"
export TRAUMA_PREDICT_OUTPUT_ROOT="/kaggle/working/trauma-predict-runs"
test -f "$TRAUMA_PREDICT_DATA_ROOT/dataset_manifest.json"
test -f "$TRAUMA_PREDICT_DATA_ROOT/sample_manifest.csv"
find "$TRAUMA_PREDICT_DATA_ROOT" -maxdepth 2 -type f | sort | sed -n '1,40p'
```

Linking a Kaggle Notebook to GitHub is optional. For this project, cloning a pinned tag is more reproducible than relying on notebook sync state. Use `stage-a-hour-field-aware-20260709` for the Stage A run after that tag is pushed.

The Kaggle requirements intentionally do not install `torch`. Use Kaggle's preinstalled CUDA PyTorch, then pin the Hugging Face stack from `requirements-kaggle.txt`. Kaggle base images often have unrelated global `pip check` conflicts from preinstalled packages, so the notebook treats `pip check` as diagnostic only. The scoped runtime guard is the blocking check: it verifies CUDA, the PyTorch wheel, and the exact Hugging Face package versions used by this repository.

Manual runtime guard after `pip install -r requirements-kaggle.txt`:

```bash
python - <<'PY'
import sys
sys.path.insert(0, "/kaggle/working/Trauma-Predict/src")
import torch, transformers, accelerate, tokenizers, huggingface_hub
expected = {
    "transformers": "4.44.2",
    "accelerate": "0.34.2",
    "tokenizers": "0.19.1",
    "huggingface_hub": "0.36.2",
}
actual = {
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "tokenizers": tokenizers.__version__,
    "huggingface_hub": huggingface_hub.__version__,
}
assert actual == expected, (expected, actual)
assert torch.cuda.is_available()
assert not torch.__version__.startswith("2.12.1+cu130")
print("runtime_guard OK")
PY
```

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

Run `notebooks/kaggle/verify_private_dataset.ipynb` first. It verifies the formal v2 Dataset `vanilaaaa/trauma-predict-main-route-first-train-8h-v2`, handles both attached private Datasets under `/kaggle/input` and Kaggle API downloads into `/kaggle/working`, reconstructs `train/val/test`, and normalizes Kaggle-expanded `.jsonl` files back to the manifest-declared `.jsonl.gz` shard names. It also asserts the expected split counts: train 31,980, val 4,378, test 3,895.

For the formal Stage A route, use `notebooks/kaggle/train_stage_a_hour.ipynb`. The older `train_full_first_run.ipynb` is a `joint_baseline` launcher and is not Stage A.

The direct preflight command expects the reconstructed artifact root, not the raw Kaggle upload folder:

```bash
export TRAUMA_PREDICT_DATA_ROOT="/kaggle/working/trauma-predict-main-route-first-train-8h-v2"
export TRAUMA_PREDICT_OUTPUT_ROOT="/kaggle/working/trauma-predict-runs"

python notebooks/kaggle/train_kaggle.py \
  --config configs/train/t4x2_stage_a_hour.yaml \
  --dry-run
```

Run the token-length scan before training. It verifies that no sample exceeds the configured encoder window and writes a JSON summary into the run folder:

```bash
python notebooks/kaggle/scan_token_lengths.py \
  --dataset-config configs/dataset/first_train.yaml \
  --train-config configs/train/t4x2_stage_a_hour.yaml \
  --output-json "$TRAUMA_PREDICT_OUTPUT_ROOT/t4x2_stage_a_hour/token_length_summary.json"
```

Run the smoke config before the full run when the notebook session or dependency image has changed:

```bash
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  notebooks/kaggle/train_kaggle.py \
  --config configs/train/t4x2_stage_a_hour_smoke.yaml
```

Training entry after dry run, token scan, and smoke pass. Use `torchrun` for the first Kaggle
run; it avoids the `accelerate` CLI import path that can pull in incompatible
vision packages on Kaggle images. Full Stage A configs are resumable. If a checkpoint is present, the runner checks `training_stage_metadata.json` before resuming and rejects mismatched stages or loss weights. A valid Stage A metadata file has `active_loss_names=["next_hour_values"]`.

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
  --config configs/train/t4x2_stage_a_hour.yaml
```

Fallback:

```bash
export TRAUMA_PREDICT_DATA_ROOT="/kaggle/working/trauma-predict-main-route-first-train-8h-v2"
export TRAUMA_PREDICT_OUTPUT_ROOT="/kaggle/working/trauma-predict-runs"
export PYTHONPATH="/kaggle/working/Trauma-Predict/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

python notebooks/kaggle/train_kaggle.py \
  --config configs/train/p100_stage_a_hour.yaml
```

Alternative single-GPU distributed launch:

```bash
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=1 \
  notebooks/kaggle/train_kaggle.py \
  --config configs/train/p100_stage_a_hour.yaml
```

## Required Outputs

Write outputs under `/kaggle/working` or the configured output root:

- `metrics.jsonl`
- resumable checkpoints
- `final_model/`
- validation predictions with input H0 context for persistence-baseline evaluation
- run config snapshot
- environment snapshot
- `training_result.json`

Do not commit these outputs to Git.

## Stage Boundary

This branch keeps Stage B and Stage C contracts in code so Stage A checkpoints have a defined continuation path. The training runner intentionally rejects Stage B until `training.stage_a_checkpoint` is actually loaded, and rejects Stage C until alternating scheduling is implemented. Do not bypass those guards by relabeling a joint run.
