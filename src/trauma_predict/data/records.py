from __future__ import annotations

import gzip
import json
from glob import glob
from pathlib import Path
from typing import Any, Iterator


def resolve_shard_paths(dataset_config: dict[str, Any], split: str) -> list[Path]:
    pattern = str(dataset_config.get(f"{split}_shards") or "")
    if not pattern:
        raise ValueError(f"dataset config missing {split}_shards")
    if "${" in pattern:
        raise ValueError(f"dataset config {split}_shards has unexpanded environment variable: {pattern}")
    paths = sorted(Path(path).resolve() for path in glob(pattern) if Path(path).is_file())
    if not paths:
        raise ValueError(f"dataset config {split}_shards matched no files: {pattern}")
    return paths


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} JSONL row must be an object")
            yield payload


def load_text_records(paths: list[Path], required_fields: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            for field in required_fields:
                if row.get(field) in ("", None):
                    raise ValueError(f"{path} row missing required field {field}")
            records.append({
                "sample_id": str(row["sample_id"]),
                "subject_id": str(row["subject_id"]),
                "hadm_id": str(row["hadm_id"]),
                "stay_id": str(row["stay_id"]),
                "prediction_hour": float(row["prediction_hour"]),
                "input_text": str(row["input_text"]),
                "target_text": str(row["target_text"]),
            })
    if not records:
        raise ValueError("no text records loaded")
    return records


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")
