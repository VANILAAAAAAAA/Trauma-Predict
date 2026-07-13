from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contract import EventTemplateRegistry, SupervisionContract, TargetLayout


@dataclass(frozen=True)
class MultiresPreflightResult:
    dataset_id: str
    dataset_fingerprint: str
    source_fingerprint: str
    sample_count: int
    split_counts: Mapping[str, int]
    subject_counts: Mapping[str, int]
    shard_count: int
    target_layout: TargetLayout
    supervision_sha256: str
    normalization_fallback_key_count: int = 0
    normalization_fallback_level_counts: Mapping[str, int] | None = None


def preflight_multires_event(
    config: Mapping[str, Any],
    dataset_root: str | Path,
    supervision_path: str | Path,
) -> MultiresPreflightResult:
    root = Path(dataset_root).resolve()
    data_config = _data_config(config)
    expected = _expected_config(data_config)
    manifest_path = root / str(data_config.get("dataset_manifest") or "dataset_manifest.json")
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != "multires_event_dataset_manifest_v2":
        raise ValueError("multires event dataset_manifest schema mismatch")
    if manifest.get("status") != "SUCCEEDED" or manifest.get("plan_only") is not False:
        raise ValueError("multires event dataset is not the completed formal artifact")

    dataset_id = str(manifest.get("dataset_id") or "")
    dataset_fingerprint = str(manifest.get("fingerprint") or "")
    source = _mapping(manifest.get("source"), "dataset_manifest.source")
    source_fingerprint = str(source.get("source_fingerprint") or "")
    _assert_expected("dataset_id", dataset_id, data_config.get("dataset_id"))
    _assert_expected(
        "dataset_fingerprint",
        dataset_fingerprint,
        expected.get("dataset_fingerprint") or data_config.get("dataset_fingerprint"),
    )
    _assert_expected(
        "source_fingerprint",
        source_fingerprint,
        expected.get("source_fingerprint") or data_config.get("source_fingerprint"),
    )

    supervision = SupervisionContract.from_json(supervision_path)
    registry_hashes = _mapping(source.get("registry_sha256"), "source.registry_sha256")
    _assert_expected(
        "registry manifest sha256",
        str(registry_hashes.get("manifest") or ""),
        str(supervision.payload["base_registry"]["registry_manifest_sha256"]),
    )
    files = _mapping(manifest.get("files"), "dataset_manifest.files")
    templates_path = root / str(files.get("event_templates") or "event_templates.json")
    observed_template_hash = _sha256_file(templates_path)
    _assert_expected(
        "event_templates sha256",
        observed_template_hash,
        str(supervision.payload["base_registry"]["event_templates_sha256"]),
    )
    templates = EventTemplateRegistry.from_json(templates_path)

    sample_manifest_path = root / str(files.get("sample_manifest") or "sample_manifest.csv")
    rows = _read_sample_manifest(sample_manifest_path)
    split_counts = Counter(str(row["split"]) for row in rows)
    sample_count = len(rows)
    subjects_by_split: dict[str, set[str]] = defaultdict(set)
    subject_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row["split"])
        subject_id = str(row["subject_id"])
        if split not in {"train", "val", "test"}:
            raise ValueError(f"sample_manifest has invalid split={split}")
        subjects_by_split[split].add(subject_id)
        subject_splits[subject_id].add(split)
    leaked = sorted(subject for subject, splits in subject_splits.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"persisted sample_manifest crosses patient split: {leaked[:5]}")

    counts = _mapping(manifest.get("counts"), "dataset_manifest.counts")
    declared_split = _mapping(counts.get("built_by_split"), "counts.built_by_split")
    if sample_count != int(counts.get("samples", -1)):
        raise ValueError("sample_manifest count does not match dataset_manifest")
    for split in ("train", "val", "test"):
        if split_counts[split] != int(declared_split[split]):
            raise ValueError(f"sample_manifest {split} count mismatch")

    expected_samples = expected.get("samples")
    if expected_samples is None and isinstance(data_config.get("expected_counts"), Mapping):
        expected_samples = data_config["expected_counts"].get("samples")
    _assert_expected("sample count", sample_count, expected_samples)
    expected_split = expected.get("split_samples")
    if isinstance(expected_split, Mapping):
        for split in ("train", "val", "test"):
            _assert_expected(f"{split} count", split_counts[split], expected_split.get(split))
    elif isinstance(data_config.get("expected_counts"), Mapping):
        for split in ("train", "val", "test"):
            _assert_expected(
                f"{split} count", split_counts[split], data_config["expected_counts"].get(split)
            )

    shard_payload = _mapping(files.get("shards"), "files.shards")
    expected_shards = expected.get("shards")
    if expected_shards is None and isinstance(data_config.get("expected_counts"), Mapping):
        expected_shards = data_config["expected_counts"].get("shards")
    _assert_expected("shard count", len(shard_payload), expected_shards)
    if len(shard_payload) != int(counts.get("completed_shards", -1)):
        raise ValueError("dataset_manifest completed_shards mismatch")

    rows_by_shard = Counter(str(row["shard_key"]) for row in rows)
    target_layout: TargetLayout | None = None
    preflight_config = _mapping_or_empty(config.get("preflight"))
    if not preflight_config and isinstance(data_config.get("preflight"), Mapping):
        preflight_config = data_config["preflight"]
    verify_headers = bool(preflight_config.get("verify_all_shard_headers", True))
    verify_hashes = bool(preflight_config.get("verify_shard_sha256", False))
    for shard_key, raw in sorted(shard_payload.items()):
        item = _mapping(raw, f"files.shards.{shard_key}")
        path = root / str(item["sample_path"])
        if not path.is_file():
            raise FileNotFoundError(f"missing canonical shard {path}")
        if rows_by_shard[str(shard_key)] != int(item["sample_count"]):
            raise ValueError(f"sample_manifest count mismatch for shard {shard_key}")
        if verify_hashes:
            _assert_expected(
                f"{shard_key} sha256", _sha256_file(path), str(item["sample_sha256"])
            )
        if verify_headers or target_layout is None:
            first = _read_first_gzip_row(path)
            if str(first.get("split")) != str(item["split"]):
                raise ValueError(f"shard {shard_key} first row split mismatch")
            if target_layout is None:
                target_layout = supervision.compile_target_layout(first, templates, strict=True)
            else:
                supervision.assert_record_layout(first, target_layout)
            filtered = supervision.filter_input_record(first)
            if any(int(event[0]) == 9 for event in filtered["input_events"]):
                raise AssertionError("preflight input filter retained gcs_verbal")
    if target_layout is None:
        raise ValueError("no target layout could be compiled from canonical shards")

    return MultiresPreflightResult(
        dataset_id=dataset_id,
        dataset_fingerprint=dataset_fingerprint,
        source_fingerprint=source_fingerprint,
        sample_count=sample_count,
        split_counts={split: split_counts[split] for split in ("train", "val", "test")},
        subject_counts={split: len(subjects_by_split[split]) for split in ("train", "val", "test")},
        shard_count=len(shard_payload),
        target_layout=target_layout,
        supervision_sha256=supervision.source_sha256,
    )


def _data_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("data")
    return value if isinstance(value, Mapping) else config


def _expected_config(data_config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = data_config.get("expected")
    return value if isinstance(value, Mapping) else {}


def _read_sample_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "subject_id", "split", "shard_key"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("sample_manifest.csv header mismatch")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_manifest.csv contains duplicate sample_id values")
    return rows


def _read_first_gzip_row(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path} first row is not an object")
                return value
    raise ValueError(f"gzip shard is empty: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_expected(label: str, observed: Any, expected: Any) -> None:
    if expected in (None, ""):
        return
    if str(observed) != str(expected):
        raise ValueError(f"{label} mismatch: {observed} != {expected}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
