# Stage A.1 Residual Runbook

Stage A.1 uses the Stage A checkpoint as a warm start, removes the vent loss, keeps `hour_vent` as input only, and trains `NEXT_HOUR` numeric values as H0 residuals:

```text
predicted_next_hour_norm = h0_norm + predicted_delta_norm
loss = SmoothL1(predicted_next_hour_norm, target_next_hour_norm)
     + SmoothL1(predicted_delta_norm, target_next_hour_norm - h0_norm)
```

## Local dry run

Use this on WSL after placing or unpacking the main-route dataset and the Stage A checkpoint locally.

```bash
cd /home/vanila/code/Trauma-Predict

export TRAUMA_PREDICT_DATA_ROOT=/path/to/trauma-predict-main-route-first-train-8h-v2
export TRAUMA_PREDICT_OUTPUT_ROOT=/tmp/trauma-predict-runs
export STAGE_A_CHECKPOINT_DIR=/home/vanila/code/EHR-Predict/reference/training_runs/kaggle_stage_a_hour_modernbert_4000_20260709/runs/t4x2_stage_a_hour/checkpoint-4000
export PYTHONPATH=/home/vanila/code/Trauma-Predict/src:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

python notebooks/kaggle/train_kaggle.py \
  --config configs/train/t4x2_stage_a1_residual.yaml \
  --dry-run
```

Expected signal:

```text
dry_run_snapshot=/tmp/trauma-predict-runs/t4x2_stage_a1_residual/run_config_snapshot.json
data_preflight_summary=/tmp/trauma-predict-runs/t4x2_stage_a1_residual/data_preflight_summary.json
```

## Local syntax check

```bash
cd /home/vanila/code/Trauma-Predict

python3 -m py_compile \
  src/trauma_predict/modeling/main_route.py \
  src/trauma_predict/training/stages.py \
  src/trauma_predict/training/main_route.py \
  notebooks/kaggle/train_kaggle.py \
  notebooks/kaggle/run_stage_a1_residual.py
```

## Package Stage A checkpoint for Kaggle

Use the actual Stage A checkpoint folder downloaded from Kaggle. The package should contain one checkpoint directory, not the whole previous run folder.

```bash
cd /home/vanila/code/EHR-Predict

CKPT_SRC=/home/vanila/code/EHR-Predict/reference/training_runs/kaggle_stage_a_hour_modernbert_4000_20260709/runs/t4x2_stage_a_hour/checkpoint-4000
PKG=/tmp/trauma-predict-stage-a-hour-ckpt-4000-20260709

rm -rf "$PKG"
mkdir -p "$PKG/checkpoint-4000"

cp "$CKPT_SRC/model.safetensors" "$PKG/checkpoint-4000/"
cp "$CKPT_SRC/training_stage_metadata.json" "$PKG/checkpoint-4000/" 2>/dev/null || true
cp "$CKPT_SRC/tokenizer.json" "$PKG/checkpoint-4000/" 2>/dev/null || true
cp "$CKPT_SRC/tokenizer_config.json" "$PKG/checkpoint-4000/" 2>/dev/null || true
cp "$CKPT_SRC/special_tokens_map.json" "$PKG/checkpoint-4000/" 2>/dev/null || true

cat > "$PKG/dataset-metadata.json" <<'JSON'
{
  "title": "trauma-predict-stage-a-hour-ckpt-4000-20260709",
  "id": "vanilaaaa/trauma-predict-stage-a-hour-ckpt-4000-20260709",
  "licenses": [{"name": "other"}]
}
JSON

kaggle datasets create -p "$PKG" -r zip
```

For a refresh:

```bash
kaggle datasets version \
  -p "$PKG" \
  -m "Refresh Stage A checkpoint for Stage A.1 residual warm start"
```

## Kaggle notebook

Use:

```text
notebooks/kaggle/train_stage_a1_residual.ipynb
```

Attach both Kaggle Datasets:

```text
vanilaaaa/trauma-predict-main-route-first-train-8h-v2
vanilaaaa/trauma-predict-stage-a-hour-ckpt-4000-20260709
```

If automatic checkpoint discovery fails, set this in the notebook before the launcher cell:

```python
import os
os.environ["STAGE_A_CHECKPOINT_DIR"] = "/kaggle/input/trauma-predict-stage-a-hour-ckpt-4000-20260709/checkpoint-4000"
```

The clear pass signal before background hosting is:

```text
STAGE_A1_CONFIG_OK
TOKEN_LENGTH_SCAN_OK
STAGE_A1_SMOKE_RUN_OK
```

After that, full training starts and writes the long stream to:

```text
/kaggle/working/trauma-predict-runs/t4x2_stage_a1_residual/logs/torchrun_train.log
```

## Kaggle shell checks

Find the checkpoint mounted by Kaggle:

```bash
find /kaggle/input -maxdepth 6 \( -name model.safetensors -o -name main_route_model.pt \) -print
```

Run only preflight from Kaggle:

```bash
cd /kaggle/working/Trauma-Predict

export TRAUMA_PREDICT_OUTPUT_ROOT=/kaggle/working/trauma-predict-runs
export STAGE_A_CHECKPOINT_DIR=/kaggle/input/trauma-predict-stage-a-hour-ckpt-4000-20260709/checkpoint-4000

python notebooks/kaggle/train_kaggle.py \
  --config configs/train/t4x2_stage_a1_residual.yaml \
  --dry-run
```

Run smoke manually:

```bash
cd /kaggle/working/Trauma-Predict

export TRAUMA_PREDICT_DATA_ROOT=/kaggle/working/trauma-predict-main-route-first-train-8h-v2
export TRAUMA_PREDICT_OUTPUT_ROOT=/kaggle/working/trauma-predict-runs
export STAGE_A_CHECKPOINT_DIR=/kaggle/input/trauma-predict-stage-a-hour-ckpt-4000-20260709/checkpoint-4000
export PYTHONPATH=/kaggle/working/Trauma-Predict/src:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  notebooks/kaggle/train_kaggle.py \
  --config configs/train/t4x2_stage_a1_residual_smoke.yaml
```

Run full manually:

```bash
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  notebooks/kaggle/train_kaggle.py \
  --config configs/train/t4x2_stage_a1_residual.yaml
```
