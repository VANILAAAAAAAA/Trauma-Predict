from __future__ import annotations

from pathlib import Path


def sorted_checkpoints(output_dir: Path) -> list[Path]:
    checkpoints = [path for path in output_dir.glob("checkpoint-*") if path.is_dir()]
    return sorted(checkpoints, key=_checkpoint_step)


def checkpoints_to_prune(output_dir: Path, keep_last: int) -> list[Path]:
    if keep_last < 1:
        raise ValueError("keep_last must be >= 1")
    checkpoints = sorted_checkpoints(output_dir)
    return checkpoints[:-keep_last]


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1
