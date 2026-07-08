# Trauma-Predict

Trauma-Predict is the clean machine-learning repository for training and evaluating the textual V1 trauma ICU prediction model. It is a code-only repository: restricted MIMIC-derived data, generated samples, checkpoints, run outputs, and agent artifacts are not stored here.

The upstream research workspace remains:

```text
/home/vanila/code/EHR-Predict
```

Use that workspace for cohort construction, field adapter development, sample-builder evidence, and historical project artifacts. Use this repository for reproducible training code, configs, schemas, tests, and Kaggle launchers.

## Current Scope

- Sample unit: one ICU stay plus one prediction anchor.
- Primary key: `(subject_id, hadm_id, stay_id, prediction_hour)`.
- Split key: `subject_id`.
- First training target: HOUR placeholder adapter with dynamic `HOUR len=L` side tensors and structured `NEXT_HOUR` / `NEXT_24H` heads.
- First compute target: Kaggle GPU T4 x2, with single-GPU fallback.

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
python tools/update_file_index.py --check
```

## Kaggle Direction

Use `notebooks/kaggle/train_full_first_run.ipynb` or `notebooks/kaggle/train_kaggle.py` as the first training launch surface. Kaggle should run from a pinned Git tag, pass artifact preflight, pass token-length scan, and run the smoke config before the full T4 x2 config. Source MIMIC extraction and field-ready sample generation stay outside this repository.
