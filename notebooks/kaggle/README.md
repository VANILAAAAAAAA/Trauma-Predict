# Kaggle Entry Points

This folder contains Kaggle-specific launch wrappers only. Core model, data, and evaluation logic belongs under `src/trauma_predict/`.

Kaggle inputs must be mounted datasets or working-directory files outside Git. Do not place MIMIC-derived data in this repository.

The completed baseline entrypoint is `train_multires_event_v1.ipynb`, pinned to tag `multires-event-v1-baseline-run-20260712-r3`. It delegates to `run_multires_event_v1.py`, requires T4 x2, prefers an attached exact `multires_event_v1_c4_full_20260712` private dataset, and otherwise uses an authenticated owner-private Kaggle CLI download fallback. The launcher downloads the outer Dataset ZIP without `--unzip`, safely extracts exactly one package layer, verifies the manifest, and normalizes the preserved `shards.zip`, `.jsonl.gz` split tree, or Kaggle-hosted `shards/<split>/*.jsonl` tree back to canonical gzip shards. It then runs exact preflight, a two-step DDP smoke, resumable 4,000-step DDP training, full validation, and export. The route is PyTorch-only: it installs `requirements-multires-kaggle.txt` and does not install Transformers or Accelerate. Attempt logs are retained under `t4x2_multires_event_v1_full/logs/attempt-NNNN/`.

The active V2 delivery Notebook is `trauma_predict_relation_v2_p100_r9.ipynb`. Its exact filename produces the frozen `trauma-predict-relation-v2-p100-r9` slug for UI upload. Select one P100, keep Internet off, and use Save & Run All. The Notebook prefers the bound private Dataset and otherwise uses Kaggle's authenticated `kagglehub` path for the same `vanila111/trauma-predict-relation-v2-p100-r9-bundle`; no user token is required.

`run_relation_v2_p100_bundle.py` validates one P100, every Dataset/source/normalization identity, every restored prior-output file, and the 48,728,439-parameter 52+39 Relation V2 route. It advances through step 250, 1500, 2750, and 4000, then resumes exactly 2,048 new free-running anchors per full evaluation Save Run, except the final remainder. The 2,048 boundary preserves complete B32 loader batches and 128-anchor atomic chunks. `kernel-metadata-relation-v2-p100.template.json` is the optional CLI-push template.

`train_multires_event_v2_relational_primary.ipynb`, `train_relational_primary.py`, `run_relational_primary_bundle.py`, and `tools/build_relational_primary_bundle.py` remain v8 historical evidence and fail closed. Older V2 Notebooks and `run_multires_event_v2.py` are also fail-closed historical surfaces. `historical_v8_dataset_evidence.py` retains only Dataset byte-audit helpers and exposes no training route.

Legacy Stage A uses `train_stage_a_hour.ipynb` and `run_stage_a_hour.py`; it remains available only for historical reproduction of the textual ModernBERT experiments.
