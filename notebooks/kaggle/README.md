# Kaggle Entry Points

This folder contains Kaggle-specific launch wrappers only. Core model, data, and evaluation logic belongs under `src/trauma_predict/`.

Kaggle inputs must be mounted datasets or working-directory files outside Git. Do not place MIMIC-derived data in this repository.

The active entrypoint is `train_multires_event_v1.ipynb`, pinned to tag `multires-event-v1-baseline-run-20260712-r2`. It delegates to `run_multires_event_v1.py`, requires T4 x2, prefers an attached exact `multires_event_v1_c4_full_20260712` private dataset, and otherwise uses an authenticated owner-private Kaggle CLI download fallback. The launcher downloads the outer Dataset ZIP without Kaggle's recursive `--unzip`, safely extracts exactly one package layer, verifies the manifest, and normalizes either the preserved `shards.zip` or a preserved `.jsonl.gz` split tree. It then runs exact preflight, a two-step DDP smoke, resumable 4,000-step DDP training, full validation, and export. The route is PyTorch-only: it installs `requirements-multires-kaggle.txt` and does not install Transformers or Accelerate. Attempt logs are retained under `t4x2_multires_event_v1_full/logs/attempt-NNNN/`.

Legacy Stage A uses `train_stage_a_hour.ipynb` and `run_stage_a_hour.py`; it remains available only for historical reproduction of the textual ModernBERT experiments.
