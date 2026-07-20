from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue the recovered GRU-D baseline from global step 2500 to 4000."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    source_root = repo_root / "src"
    if not source_root.is_dir():
        raise FileNotFoundError(f"repository source root is absent: {source_root}")
    sys.path.insert(0, str(source_root))

    from trauma_predict.training.grud_h1_v2 import run_grud_h1_v2_training

    print(
        f"GRUD_V2_NOTEBOOK_START config={args.config.resolve()} "
        "mode=resume restored_step=2500 target_step=4000 new_optimizer_steps=1500",
        flush=True,
    )
    run_grud_h1_v2_training(args.config.resolve(), repo_root=repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
