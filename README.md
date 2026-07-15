# Trauma-Predict

Trauma-Predict is the code-only repository for training and evaluating trauma ICU prediction models. Restricted MIMIC-derived data, generated samples, checkpoints, run outputs, and agent artifacts are not stored here.

The upstream research workspace remains:

```text
/home/vanila/code/EHR-Predict
```

Use that workspace for cohort construction, field adapter development, sample-builder evidence, and historical project artifacts. Use this repository for reproducible training code, configs, schemas, tests, and Kaggle launchers.

## Current Scope

- Active implementation: `multires_event_v2_m4_relational_primary`, trained from scratch as a 47,801,855-parameter structured encoder-decoder Transformer over the immutable V1 three-resolution history and one-to-one `full_r9` target sidecar. It predicts one joint trajectory of six ordered M4 blocks over 29 field processes and 414 stochastic factors, with causal cross-block attention, typed relation bias, and no H1/F24 head or loss.
- Frozen data authorities: `multires_event_v1_c4_full_20260712` supplies 50,350 input anchors and the persisted patient split; `multires_event_m4_target_v2_c4_full_20260714_r9` supplies the exactly aligned V2 targets. r9 repairs eight binary64 respiratory-simplex residuals and otherwise preserves r8 targets exactly under the audited reverse mapping.
- Experiment order: `relational` is the primary model. `trajectory` and `block` are optional later matched ablations, not prerequisites, capacity gates, or authorization gates for the primary run. They must retain the same 47,801,855-parameter structure, rows, optimizer/runtime identity, and evaluation code; only the declared attention/relation access rule may differ.
- Backbone boundary: neither capacity uses ModernBERT, a tokenizer, or another text backbone. The inputs are structured continuous/categorical event tuples and the outputs are mixed-measure clinical process distributions; pretrained text is a separate later matched factor, not a substitute for this decoder contract.
- Optimizer contract: one-group AdamW over the raw 414-factor joint-NLL batch mean, with no hidden factor normalization or global gradient clipping. Every step must carry complete gradient, scaler, Adam-state, LR, and resume-schedule health evidence.
- Hosted contract: the release Notebook may read only one attached immutable bundle and must launch the exact relational primary directly on T4 x2. Before Dataset loading, a two-rank canary exercises both rank-local artifact handling and the exact production best-checkpoint save/load collective boundary with a zero-parameter state dictionary; this is an I/O synchronization check, not a capacity model or optimization attempt. Its mounted-input and canary markers are not training evidence. Readiness requires the same formal run to complete optimizer steps 1 and 2 and write a hash-validated step-2 checkpoint containing the model, AdamW, scheduler, GradScaler, sampler, per-rank RNG, metrics, and run identity. No clone, network download, data rebuild, full-dataset rollout, disposable capacity model, or silent fallback is allowed before formal training.
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

The primary release Notebook is `notebooks/kaggle/train_multires_event_v2_relational_primary.ipynb`. Its private Dataset attachment contains the pinned source release, input normalization, and V1/r9 payload inventories. Select T4 x2 and choose Save & Run All; Internet is not required. The Notebook verifies the mounted identities, extracts only the explicitly reported hash-bound payload packs for files no larger than 64 KiB, symlinks the bulk patient shards without copying or extraction, and invokes the no-argument `train_relational_primary.py` entrypoint, which can launch only the exact 47,801,855-parameter relational configuration. `RELATIONAL_PRIMARY_MOUNTED_PREFLIGHT_OK` means only that mounted inputs and hardware passed. `MULTIRES_EVENT_V2_BEST_CHECKPOINT_COLLECTIVE_CANARY_OK` proves that both ranks completed the exact best-checkpoint save/load synchronization path before Dataset loading; it is still not a training marker. `MULTIRES_EVENT_V2_FORMAL_STEP2_CHECKPOINT_OK` is the first training-valid readiness marker. Historical V1 and V2 launchers remain evidence, not the active entrypoint.

If a hosted session ends before step 4,000, download its complete run output and package the selected checkpoint archive with `tools/build_relational_primary_bundle.py --resume-archive ... --resume-checkpoint-dir checkpoint-XXXXXXXX`. The launcher restores it before torchrun; the core resume path rejects any mismatch in checkpoint hashes, world size, run identity, optimizer/scheduler step, sampler, scaler, or per-rank RNG state.
