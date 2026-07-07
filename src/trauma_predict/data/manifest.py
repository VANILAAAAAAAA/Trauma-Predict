from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    sample_unit: str
    split_key: str
    counts: dict[str, int]
    shards: dict[str, list[str]]
    source: dict[str, Any]

    @classmethod
    def from_json(cls, path: Path) -> "DatasetManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_dataset_manifest(payload)
        return cls(
            dataset_id=str(payload["dataset_id"]),
            sample_unit=str(payload["sample_unit"]),
            split_key=str(payload["split_key"]),
            counts={key: int(payload["counts"][key]) for key in ("subjects", "hadm", "stays", "samples")},
            shards={key: list(value) for key, value in payload["shards"].items()},
            source=dict(payload["source"]),
        )


def validate_dataset_manifest(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "trauma_predict.dataset_manifest.v1":
        raise ValueError("dataset manifest schema_version mismatch")
    if payload.get("sample_unit") != "icu_stay_anchor":
        raise ValueError("dataset sample_unit must be icu_stay_anchor")
    if payload.get("split_key") != "subject_id":
        raise ValueError("dataset split_key must be subject_id")
    counts = payload.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("dataset counts must be an object")
    for key in ("subjects", "hadm", "stays", "samples"):
        value = counts.get(key)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"dataset counts.{key} must be a positive integer")
    shards = payload.get("shards")
    if not isinstance(shards, dict):
        raise ValueError("dataset shards must be an object")
    for split in ("train", "val", "test"):
        if not isinstance(shards.get(split), list):
            raise ValueError(f"dataset shards.{split} must be a list")
