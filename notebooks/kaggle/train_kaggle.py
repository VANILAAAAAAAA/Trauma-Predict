from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

from trauma_predict.training.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle training launcher for Trauma-Predict.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate config and write a run snapshot only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = Path(config.get("outputs", {}).get("output_dir", "outputs/kaggle_run"))
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(args.config),
        "config": config,
        "dry_run": args.dry_run,
    }
    (output_dir / "run_config_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.dry_run:
        print(f"dry_run_snapshot={output_dir / 'run_config_snapshot.json'}")
        return

    raise NotImplementedError(
        "Training loop wiring comes next. The Kaggle launcher and config snapshot are ready."
    )


if __name__ == "__main__":
    main()
