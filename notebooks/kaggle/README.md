# Kaggle Entry Points

This folder contains Kaggle-specific launch wrappers only. Core model, data, and evaluation logic belongs under `src/trauma_predict/`.

Kaggle inputs must be mounted datasets or working-directory files outside Git. Do not place MIMIC-derived data in this repository.

Stage A must use `train_stage_a_hour.ipynb`, which bootstraps the pinned Git ref and then delegates to `run_stage_a_hour.py`. The launcher uses `configs/train/t4x2_stage_a_hour_smoke.yaml` and either `configs/train/t4x2_stage_a_hour.yaml` or `configs/train/p100_stage_a_hour.yaml` depending on visible GPU count. The active Stage A loss is `next_hour_values` only; `hour_vent` is input-only. Detailed subprocess output is kept in run-local `logs/` files to keep Kaggle's web notebook responsive. The full-route notebook is a `joint_baseline` launcher and is not Stage A.

The 168-token paired ablation has a separate branch, tag, notebook, launcher,
configs, run name, and output archive. Use
`train_stage_a_field_hour_168.ipynb`; it requires T4 x2 and validates that every
training/data setting still matches the archived Stage A v1 control before it
starts.

Formal Stage A launchers keep complete subprocess logs on disk. During formal
training the notebook prints stable `TRAIN_LOSS=` and `EVAL_LOSS=` lines from
rank zero, plus bounded status milestones and a 300-second heartbeat.
