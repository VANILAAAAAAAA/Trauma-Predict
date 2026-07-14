# Trauma-Predict

Trauma-Predict is the code-only repository for training and evaluating trauma ICU prediction models. Restricted MIMIC-derived data, generated samples, checkpoints, run outputs, and agent artifacts are not stored here.

The upstream research workspace remains:

```text
/home/vanila/code/EHR-Predict
```

Use that workspace for cohort construction, field adapter development, sample-builder evidence, and historical project artifacts. Use this repository for reproducible training code, configs, schemas, tests, and Kaggle launchers.

## Current Scope

- Active implementation: `multires_event_v2_m4_trajectory`, trained from scratch as a structured encoder-decoder Transformer over the immutable V1 three-resolution history and one-to-one `full_r8` target sidecar. It predicts six M4 blocks over 29 field processes and 414 stochastic factors, with no H1/F24 head or loss. `trajectory` is the intended causally connected six-block primary model; `block` is only its independent-block attention control.
- Frozen data authorities: `multires_event_v1_c4_full_20260712` supplies 50,350 input anchors and the persisted patient split; `multires_event_m4_target_v2_c4_full_20260713_r8` supplies the exactly aligned V2 targets.
- Matched modes: `block`, `trajectory`, and `relational` have identical structures at a chosen capacity and differ only in the declared attention/relation access rule. The implemented control size is 30,684,479 parameters; a width-only 47,801,855-parameter candidate is diagnostic-only until the predeclared capacity comparison closes.
- Backbone boundary: neither capacity uses ModernBERT, a tokenizer, or another text backbone. The inputs are structured continuous/categorical event tuples and the outputs are mixed-measure clinical process distributions; pretrained text is a separate later matched factor, not a substitute for this decoder contract.
- Optimizer contract: one-group AdamW over the raw 414-factor joint-NLL batch mean, with no hidden factor normalization or global gradient clipping. Every step must carry complete gradient, scaler, Adam-state, LR, and resume-schedule health evidence.
- Hosted contract: r7 passed the private zero-Input dual-T4 verification path, including the complete 50,350-anchor preflight, two exact B64 FP16 optimizer updates, checkpoint/resume, and 100 validation anchors with 100 free-running trajectories each. The final r8 source only replaces the irrelevant global Kaggle-environment `pip check` with fail-closed imports of this route's three installed dependencies. It authorizes only the capacity-gated `block` control; `trajectory`, `relational`, standalone smoke, and direct non-gated formal entry remain blocked.
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

The release Notebook is `notebooks/kaggle/train_multires_event_v2.ipynb`, pinned to immutable tag `multires-event-v2-block-run-20260714-r8`. It is a two-cell, zero-Input route: select T4 x2, keep Internet enabled, and choose Save & Run All. It automatically clones the pinned source and downloads both frozen private Datasets. The same-process capacity gate must pass before formal optimizer step one. The V1 and Stage A notebooks remain versioned historical entrypoints. Source MIMIC extraction and sample generation stay outside this repository.
