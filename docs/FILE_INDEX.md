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
| `configs/train/t4x2_first_run.yaml` | config | First T4 x2 training run config. | Uses environment-variable paths only. |
| `docs/DATA_POLICY.md` | docs | Allowed and forbidden repository content policy. | No data. |
| `docs/FILE_INDEX.md` | docs | Tracked-file index. | No data. |
| `docs/KAGGLE_RUNBOOK.md` | docs | Kaggle launch and output policy. | No data. |
| `docs/REPO_STRUCTURE.md` | docs | Directory structure and design rules. | No data. |
| `notebooks/kaggle/README.md` | kaggle | Explains Kaggle launcher folder boundary. | No data. |
| `notebooks/kaggle/train_kaggle.py` | kaggle | Kaggle-compatible training entrypoint wrapper. | No data. |
| `pyproject.toml` | packaging | Python package, optional dependencies, and test config. | No data. |
| `requirements-kaggle.txt` | packaging | Kaggle install requirements. | No data. |
| `schemas/dataset_manifest.schema.json` | schema | Contract for generated dataset manifests. | Schema only. |
| `schemas/sample_manifest.schema.json` | schema | Contract for generated sample manifests. | Schema only. |
| `src/trauma_predict/__init__.py` | package | Package version. | No data. |
| `src/trauma_predict/cli.py` | package | CLI entry point for repository checks. | No data. |
| `src/trauma_predict/data/__init__.py` | package | Data utility namespace. | No data. |
| `src/trauma_predict/data/manifest.py` | package | Dataset manifest loading and validation helpers. | No data. |
| `src/trauma_predict/data/preflight.py` | package | Validates generated training artifacts before Kaggle execution. | No data. |
| `src/trauma_predict/data/splits.py` | package | Patient-level split invariant helpers. | No data. |
| `src/trauma_predict/eval/__init__.py` | package | Evaluation namespace. | No data. |
| `src/trauma_predict/eval/metrics.py` | package | Basic metric aggregation helpers. | No data. |
| `src/trauma_predict/training/__init__.py` | package | Training namespace. | No data. |
| `src/trauma_predict/training/checkpoints.py` | package | Checkpoint retention helpers. | No data. |
| `src/trauma_predict/training/config.py` | package | YAML config loading and environment expansion. | No data. |
| `tests/test_data_preflight.py` | tests | Tests generated artifact preflight checks with synthetic rows. | Synthetic records only. |
| `tests/test_manifest_contracts.py` | tests | Tests schema and manifest helper behavior. | Synthetic records only. |
| `tests/test_repo_hygiene.py` | tests | Tests file index and forbidden repository paths. | No data. |
| `tools/update_file_index.py` | tools | Validates that all tracked files appear in this index. | No data. |
