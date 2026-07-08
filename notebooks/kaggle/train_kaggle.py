from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trauma_predict.data.preflight import preflight_training_artifact
from trauma_predict.training.config import load_yaml_config
from trauma_predict.training.main_route import run_main_route_training, validate_main_route_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle training launcher for Trauma-Predict.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate config, dataset artifact, and write snapshots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    is_main = is_main_process()
    config = load_yaml_config(args.config)
    validate_main_route_config(config)
    output_dir = Path(config.get("outputs", {}).get("output_dir", "outputs/kaggle_run"))
    _reject_unexpanded_path(output_dir, "outputs.output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        snapshot = {
            "created_at": utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "config_path": str(args.config),
            "config": config,
            "dry_run": args.dry_run,
        }
        write_json(output_dir / "run_config_snapshot.json", snapshot)

    dataset_config_path = _resolve_config_path(str(config["data"]["config_path"]), args.config)
    dataset_config = load_yaml_config(dataset_config_path)
    preflight = preflight_training_artifact(dataset_config)
    if is_main:
        write_json(output_dir / "data_preflight_summary.json", preflight.to_dict())

    if args.dry_run:
        if is_main:
            print(f"dry_run_snapshot={output_dir / 'run_config_snapshot.json'}")
            print(f"data_preflight_summary={output_dir / 'data_preflight_summary.json'}")
        return

    metrics_path = Path(config.get("outputs", {}).get("metrics_jsonl", output_dir / "metrics.jsonl"))
    _reject_unexpanded_path(metrics_path, "outputs.metrics_jsonl")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if is_main:
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "created_at": utc_now(),
                "event": "data_preflight_pass",
                **preflight.to_dict(),
            }, sort_keys=True) + "\n")

    result = run_main_route_training(config, dataset_config, output_dir, preflight)
    if is_main:
        write_json(output_dir / "training_result.json", result.to_dict())

        print(f"run_config_snapshot={output_dir / 'run_config_snapshot.json'}")
        print(f"data_preflight_summary={output_dir / 'data_preflight_summary.json'}")
        print(f"metrics_jsonl={metrics_path}")
        print(f"training_result={output_dir / 'training_result.json'}")
        print("training_status=complete")


def _resolve_config_path(value: str, parent_config: Path) -> Path:
    if "${" in value:
        raise ValueError(f"data.config_path has unexpanded environment variable: {value}")
    path = Path(value)
    if path.exists():
        return path
    repo_root_candidate = parent_config.resolve().parents[2] / value
    if repo_root_candidate.exists():
        return repo_root_candidate
    raise FileNotFoundError(path)


def _reject_unexpanded_path(path: Path, label: str) -> None:
    if "${" in str(path):
        raise ValueError(f"{label} has unexpanded environment variable: {path}")


def is_main_process() -> bool:
    return os.environ.get("RANK", "0") in ("", "-1", "0")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    main()
