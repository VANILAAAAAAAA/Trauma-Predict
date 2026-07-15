from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trauma_predict.training.multires_event_v2 import (  # noqa: E402
    _load_v2_best_model,
    _materialize_v2_best_model,
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: worker OUTPUT_DIR")
    output_dir = Path(sys.argv[1]).resolve()
    completed = False
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        timeout=timedelta(seconds=10),
    )
    try:
        identity = {"test": "two-rank-best-checkpoint-collective"}
        model = torch.nn.Identity()
        _materialize_v2_best_model(
            output_dir=output_dir,
            model=model,
            identity_hashes=identity,
            step=250,
            metric=1.25,
        )
        selected = _load_v2_best_model(
            output_dir,
            model,
            torch.device("cpu"),
            expected_identity_hashes=identity,
            expected_best_step=250,
        )
        if dist.get_rank() == 0:
            print(json.dumps(selected, sort_keys=True), flush=True)
        completed = True
    finally:
        if completed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
