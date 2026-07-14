from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import trauma_predict.eval.multires_event_v2_free_running as free_running  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--inject-rank-one-writer-noop", action="store_true")
    parser.add_argument(
        "--inject-stage",
        choices=(
            "write",
            "hash",
            "gather",
            "scoring",
            "report",
            "optimizer",
            "checkpoint",
            "finalization",
        ),
    )
    parser.add_argument("--inject-rank", type=int, choices=(0, 1), default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    completed = False
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        timeout=timedelta(seconds=15),
    )
    try:
        rank = dist.get_rank()
        stage = args.inject_stage
        if args.inject_rank_one_writer_noop:
            stage = "write"
            args.inject_rank = 1
        if stage == "write" and rank == args.inject_rank:
            free_running.append_rank_local_jsonl = lambda *_args, **_kwargs: None
        if stage == "hash" and rank == args.inject_rank:
            def fail_hash(*_args, **_kwargs):
                raise OSError("injected hash failure")

            free_running.sha256_file = fail_hash
        if stage == "gather" and rank == args.inject_rank:
            def fail_gather(*_args, **_kwargs):
                raise RuntimeError("injected gather failure")

            free_running._gather_objects = fail_gather
        if stage in {
            "scoring",
            "report",
            "optimizer",
            "checkpoint",
            "finalization",
        }:
            def injected_boundary():
                if rank == args.inject_rank:
                    raise RuntimeError(f"injected {stage} failure")
                return {"rank": rank, "stage": stage}

            free_running._collect_distributed_phase(
                f"test {stage}",
                injected_boundary,
            )
            raise AssertionError(f"injected {stage} failure did not propagate")
        result = free_running.verify_rank_local_artifact_preflight(
            output_dir=args.output_dir,
            mode="block",
        )
        if dist.get_rank() == 0:
            print(json.dumps(result, sort_keys=True), flush=True)
        completed = True
    finally:
        if completed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
