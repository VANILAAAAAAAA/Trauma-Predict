# Trauma-Predict

Trauma-Predict is the code-only repository for training and evaluating trauma ICU prediction models. Restricted MIMIC-derived data, generated samples, checkpoints, run outputs, and agent artifacts are not stored here.

The upstream research workspace remains:

```text
/home/vanila/code/EHR-Predict
```

Use that workspace for cohort construction, field adapter development, sample-builder evidence, and historical project artifacts. Use this repository for reproducible training code, configs, schemas, tests, and Kaggle launchers.

## Current Scope

- Active implementation: `multires_event_v2_m4_trajectory`, trained from scratch as a structured encoder-decoder Transformer over the immutable V1 three-resolution history and one-to-one `full_r8` target sidecar. It predicts one joint six-block M4 trajectory over 29 field processes and 414 stochastic factors, with no H1/F24 head or loss.
- Frozen data authorities: `multires_event_v1_c4_full_20260712` supplies 50,350 input anchors and the persisted patient split; `multires_event_m4_target_v2_c4_full_20260713_r8` supplies the exactly aligned V2 targets.
- Matched modes: `block`, `trajectory`, and `relational` have identical 30,684,479-parameter structures and differ only in the declared attention/relation access rule.
- Optimizer contract: one-group AdamW over the raw 414-factor joint-NLL batch mean, with no hidden factor normalization or global gradient clipping. Every step must carry complete gradient, scaler, Adam-state, LR, and resume-schedule health evidence.
- Hosted contract: the first formal action is source-authorized only for `block`, pinned to tag `multires-event-v2-block-run-20260713-r2`, an offline exact Git bundle, and private target Dataset `vanilaaaa/trauma-predict-multires-event-v2-c4-r8-20260713`; standalone smoke, trajectory, and relational training remain source-blocked. The Notebook sets `TRAUMA_PREDICT_DRY_RUN_ONLY=0`, but the same-process T4 x2 capacity gate must still pass two B32/GPU FP16 updates and 100 validation anchors × 100 trajectories before the unchanged 4,000-step block run can begin.
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

Open the prepared private Kaggle upload shell, replace its source with `notebooks/kaggle/train_multires_event_v2.ipynb`, select T4 x2, and Save & Run without changing frozen identities. The shell already binds the exact V1 base, r8 target, and offline Git bundle, so the run does not depend on Kaggle Secrets or Internet access. The Notebook checks out `multires-event-v2-block-run-20260713-r2`, selects only `block`, and sets `TRAUMA_PREDICT_DRY_RUN_ONLY=0`. The launcher verifies source and data identities before entering one `torchrun`; its same-process capacity gate must pass before formal optimization starts. If Kaggle exposes uploaded target shards as plain JSONL, the launcher reconstructs gzip with the r8 builder's line-wise `TextIOWrapper` procedure and verifies every reconstructed shard against the manifest byte hash. Promotion remains a separate action over three distinct completed run roots. The V1 and Stage A notebooks remain versioned historical entrypoints. Source MIMIC extraction and sample generation stay outside this repository.
