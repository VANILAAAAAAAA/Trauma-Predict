from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import run_stage_a_hour as base

SRC_ROOT = base.REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trauma_predict.training.config import load_yaml_config
from trauma_predict.training.hour_token_ablation import validate_field_hour_ablation_config


FULL_CONFIG = "configs/train/t4x2_stage_a_field_hour_168.yaml"
SMOKE_CONFIG = "configs/train/t4x2_stage_a_field_hour_168_smoke.yaml"
CONTROL_CONFIG = "configs/train/t4x2_stage_a_hour.yaml"
RUN_NAME = "t4x2_stage_a_field_hour_168"


def select_training_route(gpu_count: int) -> tuple[str, str, int]:
    if gpu_count != 2:
        raise RuntimeError(
            "The canonical 24-token control used Kaggle T4 x2. "
            f"This paired 168-token run requires exactly 2 visible GPUs; got {gpu_count}."
        )
    return FULL_CONFIG, RUN_NAME, 2


def run_ablation_preflight(train_config: str, run_name: str, log_dir: Path) -> None:
    original_preflight(train_config, run_name, log_dir)
    control = load_yaml_config(base.REPO_ROOT / CONTROL_CONFIG)
    candidate = load_yaml_config(base.REPO_ROOT / FULL_CONFIG)
    contract = validate_field_hour_ablation_config(control, candidate)
    contract["dataset"] = dataset_fingerprint(base.DATA_ROOT)
    contract["effective_full_window_hour_tokens"] = 168
    contract["ventilation_input"] = "broadcast to the seven vital tokens for each hour"
    contract["target_contract"] = "unchanged Stage A v1 NEXT_HOUR latest-observation values"
    contract_path = base.OUTPUT_ROOT / run_name / "ablation_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "STAGE_A_168_ABLATION_CONTRACT_OK",
        json.dumps({
            "ablation_id": contract["ablation_id"],
            "control_commit": contract["control"]["git_commit"],
            "dataset_id": contract["dataset"]["dataset_id"],
            "dataset_files": len(contract["dataset"]["files"]),
            "sample_contract_changed": contract["sample_contract_changed"],
            "warm_start": contract["warm_start"],
        }, sort_keys=True),
    )


def dataset_fingerprint(root: Path) -> dict[str, object]:
    manifest_path = root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [manifest_path, root / "sample_manifest.csv"]
    paths.extend(sorted(root.glob("*/shard-*.jsonl.gz")))
    files = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        files.append({
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    return {
        "dataset_ref": base.DATASET_REF,
        "dataset_id": manifest.get("dataset_id"),
        "counts": manifest.get("counts"),
        "files": files,
    }


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


original_preflight = base.run_stage_a_preflight
base.T4X2_TRAIN_CONFIG = FULL_CONFIG
base.P100_TRAIN_CONFIG = FULL_CONFIG
base.SMOKE_CONFIG = SMOKE_CONFIG
base.select_training_route = select_training_route
base.run_stage_a_preflight = run_ablation_preflight


if __name__ == "__main__":
    base.main()
