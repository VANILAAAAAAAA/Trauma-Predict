# File Index

Every tracked file in this repository must appear in this table. Run `python tools/update_file_index.py --check` before committing.

| Path | Area | Purpose | Data policy |
| --- | --- | --- | --- |
| `.env.example` | config | Environment variable template for local/Kaggle paths. | No secrets or real data paths. |
| `.gitignore` | repo | Blocks data, checkpoints, caches, and secrets from Git. | Protects restricted artifacts. |
| `README.md` | docs | Repository entry point and scope boundary. | Documents code-only policy. |
| `configs/accelerate/single_gpu.yaml` | config | Single-GPU fallback accelerator config. | No data. |
| `configs/accelerate/t4x2.yaml` | config | Kaggle T4 x2 accelerator config. | No data. |
| `configs/dataset/first_train.yaml` | config | First training dataset artifact paths and required sample fields. | Uses environment-variable paths only. |
| `configs/train/p100_stage_a_hour.yaml` | config | Stage A single-GPU/P100 HOUR values-only training config. | Uses environment-variable paths only. |
| `configs/train/t4x2_first_run.yaml` | config | Joint-baseline T4 x2 training config; not Stage A. | Uses environment-variable paths only. |
| `configs/train/t4x2_smoke.yaml` | config | Joint-baseline smoke config; not Stage A. | Uses environment-variable paths only. |
| `configs/train/t4x2_stage_a_hour.yaml` | config | Stage A T4 x2 HOUR values-only training config with ventilation and `NEXT_24H` losses inactive. | Uses environment-variable paths only. |
| `configs/train/t4x2_stage_a_hour_smoke.yaml` | config | Stage A smoke config that proves HOUR values-only model/data/runtime wiring. | Uses environment-variable paths only. |
| `docs/DATA_POLICY.md` | docs | Allowed and forbidden repository content policy. | No data. |
| `docs/FILE_INDEX.md` | docs | Tracked-file index. | No data. |
| `docs/KAGGLE_RUNBOOK.md` | docs | Kaggle launch and output policy. | No data. |
| `docs/REPO_STRUCTURE.md` | docs | Directory structure and design rules. | No data. |
| `docs/TRAINING_STAGES.md` | docs | Stage A/B/C and joint-baseline training contract. | No data. |
| `notebooks/kaggle/README.md` | kaggle | Explains Kaggle launcher folder boundary. | No data. |
| `notebooks/kaggle/run_stage_a_hour.py` | kaggle | Automated Stage A Kaggle launcher for preferred-encoder HOUR-only training. | Reads mounted private data only at runtime. |
| `notebooks/kaggle/scan_token_lengths.py` | kaggle | Scans shard input token lengths against the configured encoder window before training. | Reads mounted private data only at runtime. |
| `notebooks/kaggle/train_kaggle.py` | kaggle | Kaggle-compatible training entrypoint wrapper. | No data. |
| `notebooks/kaggle/train_full_first_run.ipynb` | kaggle | End-to-end Kaggle notebook for the joint-baseline run; not Stage A. | No data. |
| `notebooks/kaggle/train_stage_a_hour.ipynb` | kaggle | End-to-end Kaggle notebook for Stage A HOUR-only training. | No data. |
| `notebooks/kaggle/verify_private_dataset.ipynb` | kaggle | Kaggle notebook that verifies private Dataset mounting or API download before preflight. | No data. |
| `pyproject.toml` | packaging | Python package, optional dependencies, and test config. | No data. |
| `requirements-kaggle.txt` | packaging | Kaggle install requirements. | No data. |
| `schemas/dataset_manifest.schema.json` | schema | Contract for generated dataset manifests. | Schema only. |
| `schemas/sample_manifest.schema.json` | schema | Contract for generated sample manifests. | Schema only. |
| `src/trauma_predict/__init__.py` | package | Package version. | No data. |
| `src/trauma_predict/cli.py` | package | CLI entry point for repository checks. | No data. |
| `src/trauma_predict/data/__init__.py` | package | Data utility namespace. | No data. |
| `src/trauma_predict/data/main_route.py` | package | Loads main-route records and batches HOUR side tensors for training. | No data. |
| `src/trauma_predict/data/main_route_contract.py` | package | Validates the standard textual V1 main-route record, HOUR tensors, and structured targets. | No data. |
| `src/trauma_predict/data/manifest.py` | package | Dataset manifest loading and validation helpers. | No data. |
| `src/trauma_predict/data/preflight.py` | package | Validates generated training artifacts before Kaggle execution. | No data. |
| `src/trauma_predict/data/records.py` | package | Reads generated JSONL shard records for training. | No data. |
| `src/trauma_predict/data/splits.py` | package | Patient-level split invariant helpers. | No data. |
| `src/trauma_predict/eval/__init__.py` | package | Evaluation namespace. | No data. |
| `src/trauma_predict/eval/metrics.py` | package | Basic metric aggregation helpers. | No data. |
| `src/trauma_predict/modeling/__init__.py` | package | Modeling namespace. | No data. |
| `src/trauma_predict/modeling/main_route.py` | package | Encoder model with HourStateAdapter injection and structured prediction heads. | No data. |
| `src/trauma_predict/training/__init__.py` | package | Training namespace. | No data. |
| `src/trauma_predict/training/checkpoints.py` | package | Checkpoint retention helpers. | No data. |
| `src/trauma_predict/training/config.py` | package | YAML config loading and environment expansion. | No data. |
| `src/trauma_predict/training/main_route.py` | package | Hugging Face Trainer loop for main-route structured prediction. | No data. |
| `src/trauma_predict/training/runtime.py` | package | Shared training runtime helpers for logging, checkpoints, and snapshots. | No data. |
| `src/trauma_predict/training/stages.py` | package | Explicit Stage A/B/C and joint-baseline active-loss contracts. | No data. |
| `tests/test_data_preflight.py` | tests | Tests generated artifact preflight checks with synthetic rows. | Synthetic records only. |
| `tests/test_manifest_contracts.py` | tests | Tests schema and manifest helper behavior. | Synthetic records only. |
| `tests/test_repo_hygiene.py` | tests | Tests file index and forbidden repository paths. | No data. |
| `tests/test_training_main_route.py` | tests | Tests main-route config, label encoding, collator alignment, adapter shape, and checkpoint helpers. | Synthetic records only. |
| `tools/update_file_index.py` | tools | Validates that all tracked files appear in this index. | No data. |
