# Trauma-Predict

Trauma-Predict is the code-only repository for training and evaluating trauma ICU prediction models. Restricted MIMIC-derived data, generated samples, checkpoints, run outputs, and agent artifacts are not stored here.

The upstream research workspace remains:

```text
/home/vanila/code/EHR-Predict
```

Use that workspace for cohort construction, field adapter development, sample-builder evidence, and historical project artifacts. Use this repository for reproducible training code, configs, schemas, tests, and Kaggle launchers.

## Current Scope

- Active route: `multires_event_v1_baseline`, trained from scratch with a hierarchical event Transformer and fixed typed H1/M4 queries. It does not use ModernBERT or a tokenizer.
- Frozen artifact: `multires_event_v1_c4_full_20260712`, 50,350 samples with persisted patient split; the baseline learns 986 direct primary queries and derives F24 only for evaluation.
- Hosted contract: Kaggle T4 x2, PyTorch `torchrun`/DDP, fp16, 2-step smoke followed by 4,000 optimizer steps. Interval validation uses one fixed anchor for each of the 505 eligible validation subjects with persisted anchors; final validation uses all 6,309 anchors and subject-macro aggregation.
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
python tools/update_file_index.py --check
```

## Kaggle Direction

Use `notebooks/kaggle/train_multires_event_v1.ipynb` for the active baseline and select T4 x2. An attached exact private C4 dataset is preferred; otherwise the owner-private Kaggle dataset is downloaded by CLI and verified before use. The pinned notebook delegates exact identity preflight, DDP smoke, resumable full training, final evaluation, and export to `run_multires_event_v1.py`. Logs are append-only by attempt, while rank zero surfaces 250-step `TRAIN_LOSS`/`EVAL_LOSS` and a five-minute heartbeat. The older Stage A notebooks remain versioned historical entrypoints. Source MIMIC extraction and sample generation stay outside this repository.
