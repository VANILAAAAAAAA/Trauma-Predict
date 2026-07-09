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
- Formal staged route: Stage A trains `NEXT_HOUR` vital values only with `hour_vent` retained as an input covariate, Stage B trains `NEXT_24H` from the Stage A checkpoint, and Stage C is optional alternating joint training. This branch keeps Stage B/C contracts reserved but blocks their training entry until checkpoint loading and alternating scheduling are implemented.
- Current baseline route: a joint `NEXT_HOUR` + `NEXT_24H` run is allowed only when labeled `joint_baseline`, not Stage A.
- First compute target: Kaggle GPU T4 x2 with `answerdotai/ModernBERT-base` for Stage A, with single-GPU fallback.

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

Use `notebooks/kaggle/train_stage_a_hour.ipynb` for the Stage A HOUR values-only run. It is a thin Kaggle bootstrap; the versioned Python launcher performs clone verification, runtime guard, artifact reconstruction, preflight, token-length scan, smoke training, full training, and output archiving. Detailed command output is written to run-local `logs/` files instead of flooding notebook stdout. `notebooks/kaggle/train_full_first_run.ipynb` is a joint-baseline launcher and must not be used or reported as Stage A. Source MIMIC extraction and field-ready sample generation stay outside this repository.
