from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import quantiles
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trauma_predict.data.main_route_contract import HOUR_SPECIAL_TOKENS, STATE_TOKEN
from trauma_predict.data.records import read_jsonl, resolve_shard_paths
from trauma_predict.training.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan main-route sample token lengths.")
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_config = load_yaml_config(args.dataset_config)
    train_config = load_yaml_config(args.train_config)
    model_config = train_config["model"]
    max_input_tokens = int(model_config["max_input_tokens"])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_config["base_model"]))
    tokenizer.add_special_tokens({"additional_special_tokens": [*HOUR_SPECIAL_TOKENS, STATE_TOKEN]})

    by_split: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for split in args.splits:
        lengths: list[int] = []
        worst: list[dict[str, Any]] = []
        for path in resolve_shard_paths(dataset_config, split):
            for row in read_jsonl(path):
                token_count = len(tokenizer(str(row["input_text"]), add_special_tokens=True, truncation=False)["input_ids"])
                lengths.append(token_count)
                item = {
                    "sample_id": row.get("sample_id"),
                    "split": split,
                    "tokens": token_count,
                    "shard": str(path),
                }
                worst.append(item)
                if token_count > max_input_tokens:
                    failures.append(item)
        worst = sorted(worst, key=lambda item: int(item["tokens"]), reverse=True)[:10]
        by_split[split] = _summarize_lengths(lengths, worst)

    payload = {
        "base_model": model_config["base_model"],
        "max_input_tokens": max_input_tokens,
        "by_split": by_split,
        "failures": failures[:50],
        "failure_count": len(failures),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(f"token length scan failed: {len(failures)} samples exceed {max_input_tokens}")
    return 0


def _summarize_lengths(lengths: list[int], worst: list[dict[str, Any]]) -> dict[str, Any]:
    if not lengths:
        raise ValueError("cannot summarize empty split")
    ordered = sorted(lengths)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "max": ordered[-1],
        "worst": worst,
    }


def _percentile(ordered: list[int], percentile: int) -> int:
    if len(ordered) == 1:
        return ordered[0]
    cut_points = quantiles(ordered, n=100, method="inclusive")
    return int(round(cut_points[percentile - 1]))


if __name__ == "__main__":
    raise SystemExit(main())
