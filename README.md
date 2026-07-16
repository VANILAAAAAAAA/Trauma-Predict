# Trauma-Predict

Trauma-Predict is the code-only repository for training and evaluating trauma ICU prediction models. Restricted MIMIC-derived data, generated samples, checkpoints, run outputs, and agent artifacts are not stored here.

The upstream research workspace remains:

```text
/home/vanila/code/EHR-Predict
```

Use that workspace for cohort construction, field adapter development, sample-builder evidence, and historical project artifacts. Use this repository for reproducible training code, configs, schemas, tests, and Kaggle launchers.

## Current Scope

- Active implementation: `multires_event_v2_m4_relation_v2`, a 48,728,439-parameter structured encoder-decoder Transformer over the immutable V1 three-resolution history and one-to-one `full_r9` target sidecar. It predicts one joint trajectory of six ordered M4 blocks over 29 field processes and 414 stochastic factors, with mandatory 52-row target-target and 39-row input-target relation paths, block-geometry temporal fusion for the eight input-only fields, and no H1/F24 head or loss. New training has not started.
- Frozen data authorities: `multires_event_v1_c4_full_20260712` supplies 50,350 input anchors and the persisted patient split; `multires_event_m4_target_v2_c4_full_20260714_r9` supplies the exactly aligned V2 targets. r9 repairs eight binary64 respiratory-simplex residuals and otherwise preserves r8 targets exactly under the audited reverse mapping.
- Model route: this version has one joint causal Relation V2 route. Relation-off, block, trajectory, mode selection, and promotion logic are outside this contract; any later ablation requires a separately frozen experiment.
- Backbone boundary: the Relation V2 model uses neither ModernBERT, a tokenizer, nor another text backbone. The inputs are structured continuous/categorical event tuples and the outputs are mixed-measure clinical process distributions; pretrained text is a separate later matched factor, not a substitute for this decoder contract.
- Optimizer contract: one-group AdamW over the raw 414-factor joint-NLL batch mean, with no hidden factor normalization or global gradient clipping. Every step must carry complete gradient, scaler, Adam-state, LR, and resume-schedule health evidence.
- Hosted contract: the new single-P100 route uses `trauma_predict_relation_v2_p100_r9.ipynb`, an offline source archive, the private `vanila111/trauma-predict-relation-v2-p100-r9-bundle` Dataset, and hash-validated prior-Notebook-output restore. Training advances through 250/1500/2750/4000-step boundaries before resumable free-running evaluation. The v8 Notebook, launcher, entrypoint, and bundle builder remain historical fail-closed files.
- Historical V1 scratch Transformer and GRU-D runs remain retained evidence, not the active prediction task.
- Legacy textual routes remain for experiment traceability and are not the active multi-resolution baseline.
- Sample unit: one ICU stay plus one prediction anchor.
- Primary key: `(subject_id, hadm_id, stay_id, prediction_hour)`.
- Split key: `subject_id`.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/trauma_predict/` | Importable Python package for data manifests, split handling, training, and evaluation code. |
| `configs/` | Versioned dataset, training, and accelerator configs. |
| `schemas/` | JSON Schema contracts for generated dataset and sample manifests. |
| `notebooks/kaggle/` | Kaggle entrypoints and runbook-only notebooks/scripts. |
| `docs/` | Repository structure, data policy, Kaggle runbook, and file index. |
| `tests/` | Contract and hygiene tests that run without restricted data. |
| `tools/` | Repository maintenance tools, including file-index validation. |

## Data Policy

This repository must not contain:

- MIMIC-IV, MIMIC-CXR, or derived patient-level records.
- Generated train/validation/test sample shards.
- Model checkpoints or predictions derived from restricted data.
- `agent-artifact/` or historical workspace state copied from EHR-Predict.

Local or Kaggle training should receive data through paths outside Git, mounted datasets, or private storage controlled according to the applicable data-use agreement.

## Basic Checks

```bash
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m unittest discover -s tests -p 'test_multires_event_v2*.py'
python tools/update_file_index.py --check
```

## Kaggle Direction

The active hosted delivery surface is `notebooks/kaggle/trauma_predict_relation_v2_p100_r9.ipynb`. Upload it under its filename-derived slug, select one P100, keep Internet off, and use Save & Run All. It resolves the exact private bundle from an attached Input or Kaggle's authenticated `kagglehub` Dataset path, then delegates to `run_relation_v2_p100_bundle.py`.

`notebooks/kaggle/train_multires_event_v2_relational_primary.ipynb`, `train_relational_primary.py`, `run_relational_primary_bundle.py`, and `tools/build_relational_primary_bundle.py` remain v8 historical evidence and fail closed. They cannot train, resume, or package Relation V2.
