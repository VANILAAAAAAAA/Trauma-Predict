# Kaggle Entry Points

This folder contains Kaggle-specific launch wrappers only. Core model, data, and evaluation logic belongs under `src/trauma_predict/`.

Kaggle inputs must be mounted datasets or working-directory files outside Git. Do not place MIMIC-derived data in this repository.

Stage A must use `train_stage_a_hour.ipynb`, `configs/train/t4x2_stage_a_hour_smoke.yaml`, and either `configs/train/t4x2_stage_a_hour.yaml` or `configs/train/p100_stage_a_hour.yaml` depending on visible GPU count. The active Stage A loss is `next_hour_values` only; `hour_vent` is input-only. The full-route notebook is a `joint_baseline` launcher and is not Stage A.
