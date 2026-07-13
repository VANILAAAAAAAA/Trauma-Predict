from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trauma_predict.training.multires_event import (  # noqa: E402
    load_and_preflight,
    resolve_repo_path,
    run_multires_event_training,
)
from trauma_predict.training.observability import atomic_write_json, utc_now  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch/DDP entry point for multires_event_v1 baseline training."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    if args.dry_run:
        config, model_config, dataset_root, supervision_path, model_path, preflight = load_and_preflight(
            config_path, repo_root=REPO_ROOT
        )
        output_dir = resolve_repo_path(config["outputs"]["output_dir"], REPO_ROOT)
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "schema_version": "trauma_predict.multires_dry_run.v1",
            "created_at": utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "config_path": str(config_path.resolve()),
            "model_path": str(model_path),
            "supervision_path": str(supervision_path),
            "dataset_root": str(dataset_root),
            "config": config,
            "model_config": model_config,
            "dataset_preflight": preflight,
        }
        atomic_write_json(output_dir / "dry_run_preflight.json", snapshot)
        print("MULTIRES_EVENT_PREFLIGHT_OK", json.dumps({
            "dataset_id": preflight["dataset_id"],
            "dataset_fingerprint": preflight["dataset_fingerprint"],
            "counts": preflight["counts"],
            "validation_subjects": preflight["validation_subjects"],
        }, sort_keys=True), flush=True)
        return
    run_multires_event_training(config_path, repo_root=REPO_ROOT)


if __name__ == "__main__":
    main()
