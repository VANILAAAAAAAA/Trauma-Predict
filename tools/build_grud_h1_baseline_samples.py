#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trauma_predict.data.grud_h1_sample import BuildContract, build_grud_h1_dataset


DEFAULT_CONFIG = ROOT / "configs" / "dataset" / "grud_h1_baseline_c4.yaml"
DEFAULT_REGISTRY = (
    ROOT / "configs" / "contracts" / "grud_h1_baseline" / "registry_manifest_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the matched GRU-D H1 input sidecar over frozen C4 anchors."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--field-ready-root", type=Path)
    parser.add_argument("--base-dataset-root", type=Path)
    parser.add_argument("--target-dataset-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--dataset-id")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-shards",
        type=int,
        help="Build the first N authority shards for an explicitly named gate artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    _validate_config(config)
    paths = config["paths"]
    field_ready_root = _resolve_path(
        args.field_ready_root, paths["field_ready_root"], "TRAUMA_PREDICT_FIELD_READY_ROOT"
    )
    base_dataset_root = _resolve_path(
        args.base_dataset_root, paths["base_dataset_root"], "TRAUMA_PREDICT_DATA_ROOT"
    )
    target_dataset_root = _resolve_path(
        args.target_dataset_root, paths["target_dataset_root"], "TRAUMA_PREDICT_V2_TARGET_ROOT"
    )
    output_root = _resolve_path(
        args.output_root, paths["output_root"], "TRAUMA_PREDICT_GRUD_H1_OUTPUT_ROOT"
    )
    expected = config["expected_counts"]
    authority = config["authority"]
    input_contract = config["input"]
    contract = BuildContract(
        dataset_id=args.dataset_id or str(config["dataset_id"]),
        expected_base_manifest_sha256=str(authority["base_sample_manifest_sha256"]),
        expected_target_manifest_sha256=str(authority["target_sample_manifest_sha256"]),
        expected_samples=int(expected["samples"]),
        expected_split_counts={split: int(expected[split]) for split in ("train", "val", "test")},
        expected_shards=int(expected["shards"]),
        max_history_hours=int(input_contract["max_history_hours"]),
        h1_template_count=int(input_contract["registered_channels"]),
    )
    workers = args.workers or int(config["builder"]["workers"])
    print(
        "GRUD_H1_SAMPLE_BUILD_START "
        f"dataset={contract.dataset_id} workers={workers} "
        f"mode={'resume' if args.resume else 'fresh'}",
        flush=True,
    )
    build_grud_h1_dataset(
        field_ready_root=field_ready_root,
        base_dataset_root=base_dataset_root,
        target_dataset_root=target_dataset_root,
        output_root=output_root,
        registry_path=args.registry,
        contract=contract,
        workers=workers,
        resume=args.resume,
        max_shards=args.max_shards,
    )


def _resolve_path(argument: Path | None, config_value: object, environment_name: str) -> Path:
    if argument is not None:
        return argument
    text = str(config_value)
    marker = "${" + environment_name + "}"
    value = os.environ.get(environment_name) if text == marker else os.path.expandvars(text)
    if not value or "$" in value:
        raise ValueError(f"provide --{environment_name.lower().replace('_', '-')} or set {environment_name}")
    return Path(value)


def _validate_config(config: Any) -> None:
    if not isinstance(config, dict):
        raise ValueError("dataset config must be a mapping")
    if config.get("schema_version") != "trauma_predict.grud_h1_baseline_dataset_config.v1":
        raise ValueError("unsupported GRU-D H1 dataset config")
    required = {"dataset_id", "paths", "authority", "expected_counts", "input", "builder"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"dataset config lacks keys: {sorted(missing)}")


if __name__ == "__main__":
    main()
