from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from torch.distributed.elastic.multiprocessing.errors import record  # noqa: E402
from trauma_predict.training.multires_event_v2 import (  # noqa: E402
    run_multires_event_v2_training,
)


PRIMARY_CONFIG = REPO_ROOT / "configs/train/t4x2_multires_event_v2_relational.yaml"


@record
def main() -> None:
    if not PRIMARY_CONFIG.is_file():
        raise FileNotFoundError(PRIMARY_CONFIG)
    run_multires_event_v2_training(PRIMARY_CONFIG, repo_root=REPO_ROOT)


if __name__ == "__main__":
    main()
