from __future__ import annotations

import csv
import gzip
import json
from glob import glob
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from trauma_predict.data.main_route_contract import validate_main_route_record
from trauma_predict.data.manifest import DatasetManifest
from trauma_predict.data.splits import assert_patient_level_split


SPLITS = ("train", "val", "test")
SAMPLE_MANIFEST_FIELDS = ("sample_id", "subject_id", "hadm_id", "stay_id", "prediction_hour", "split", "shard_path")


@dataclass(frozen=True)
class ArtifactPreflightResult:
    dataset_id: str
    dataset_root: str
    manifest_samples: int
    sample_manifest_rows: int
    shard_rows: int
    split_counts: dict[str, int]
    shard_counts: dict[str, int]
    shard_files: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_root": self.dataset_root,
            "manifest_samples": self.manifest_samples,
            "sample_manifest_rows": self.sample_manifest_rows,
            "shard_rows": self.shard_rows,
            "split_counts": self.split_counts,
            "shard_counts": self.shard_counts,
            "shard_files": self.shard_files,
        }


def preflight_training_artifact(dataset_config: dict[str, Any]) -> ArtifactPreflightResult:
    required_fields = tuple(dataset_config.get("required_sample_fields") or ())
    if not required_fields:
        raise ValueError("dataset config required_sample_fields must be non-empty")

    dataset_manifest_path = _resolved_path(dataset_config, "dataset_manifest")
    sample_manifest_path = _resolved_path(dataset_config, "sample_manifest")
    dataset_root = dataset_manifest_path.parent

    manifest_payload = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    manifest = DatasetManifest.from_json(dataset_manifest_path)
    if manifest_payload.get("plan_only"):
        raise ValueError("dataset_manifest.json is a plan-only artifact, not a training artifact")

    declared_shards = _declared_shards(dataset_root, manifest_payload)
    configured_shards = _configured_shards(dataset_config)
    if configured_shards != declared_shards:
        raise ValueError(
            "configured shard globs do not match dataset_manifest shards: "
            f"configured={_relative_shards(dataset_root, configured_shards)} "
            f"declared={_relative_shards(dataset_root, declared_shards)}"
        )

    manifest_rows = _read_csv_rows(sample_manifest_path)
    if not manifest_rows:
        raise ValueError("sample_manifest.csv has no rows")
    _validate_manifest_rows(manifest_rows)
    assert_patient_level_split(manifest_rows)

    shard_rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        for path in configured_shards[split]:
            shard_rows = list(_read_jsonl(path))
            if not shard_rows:
                raise ValueError(f"shard has no rows: {path}")
            for row in shard_rows:
                _validate_record(row, required_fields, split, path)
            shard_rows_by_split[split].extend(shard_rows)

    shard_rows = [row for split in SPLITS for row in shard_rows_by_split[split]]
    _validate_unique_sample_ids(shard_rows, "shards")
    _validate_unique_sample_ids(manifest_rows, "sample_manifest")

    manifest_ids = {str(row["sample_id"]) for row in manifest_rows}
    shard_ids = {str(row["sample_id"]) for row in shard_rows}
    if manifest_ids != shard_ids:
        missing = sorted(manifest_ids - shard_ids)[:5]
        extra = sorted(shard_ids - manifest_ids)[:5]
        raise ValueError(f"sample_manifest and shards disagree on sample_id set: missing={missing} extra={extra}")

    expected_samples = int(manifest.counts["samples"])
    if expected_samples != len(manifest_rows) or expected_samples != len(shard_rows):
        raise ValueError(
            "dataset sample count mismatch: "
            f"manifest={expected_samples} sample_manifest={len(manifest_rows)} shards={len(shard_rows)}"
        )

    split_counts = Counter(str(row["split"]) for row in manifest_rows)
    shard_counts = Counter(str(row["split"]) for row in shard_rows)
    if split_counts != shard_counts:
        raise ValueError(f"sample_manifest and shard split counts differ: {split_counts} != {shard_counts}")

    declared_by_split = manifest_payload.get("counts", {}).get("by_split")
    if isinstance(declared_by_split, dict):
        observed = {split: int(split_counts.get(split, 0)) for split in SPLITS}
        expected = {split: int(declared_by_split.get(split, 0)) for split in SPLITS}
        if observed != expected:
            raise ValueError(f"counts.by_split mismatch: manifest={expected} observed={observed}")

    return ArtifactPreflightResult(
        dataset_id=manifest.dataset_id,
        dataset_root=str(dataset_root),
        manifest_samples=expected_samples,
        sample_manifest_rows=len(manifest_rows),
        shard_rows=len(shard_rows),
        split_counts={split: int(split_counts.get(split, 0)) for split in SPLITS},
        shard_counts={split: int(shard_counts.get(split, 0)) for split in SPLITS},
        shard_files=_relative_shards(dataset_root, configured_shards),
    )


def _resolved_path(config: dict[str, Any], key: str) -> Path:
    value = str(config.get(key) or "")
    if not value:
        raise ValueError(f"dataset config missing {key}")
    if "${" in value:
        raise ValueError(f"dataset config {key} has unexpanded environment variable: {value}")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _configured_shards(config: dict[str, Any]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for split in SPLITS:
        key = f"{split}_shards"
        pattern = str(config.get(key) or "")
        if not pattern:
            raise ValueError(f"dataset config missing {key}")
        if "${" in pattern:
            raise ValueError(f"dataset config {key} has unexpanded environment variable: {pattern}")
        paths = sorted(Path(path).resolve() for path in glob(pattern) if Path(path).is_file())
        if not paths:
            raise ValueError(f"dataset config {key} matched no shard files: {pattern}")
        result[split] = paths
    return result


def _declared_shards(dataset_root: Path, manifest: dict[str, Any]) -> dict[str, list[Path]]:
    shards = manifest.get("shards")
    if not isinstance(shards, dict):
        raise ValueError("dataset_manifest shards must be an object")
    result: dict[str, list[Path]] = {}
    for split in SPLITS:
        values = shards.get(split)
        if not isinstance(values, list):
            raise ValueError(f"dataset_manifest shards.{split} must be a list")
        paths = [(dataset_root / str(value)).resolve() for value in values]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"dataset_manifest shards.{split} missing files: {missing[:5]}")
        if not paths:
            raise ValueError(f"dataset_manifest shards.{split} is empty")
        result[split] = paths
    return result


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_manifest_rows(rows: list[dict[str, str]]) -> None:
    for index, row in enumerate(rows, start=2):
        _validate_common_fields(row, SAMPLE_MANIFEST_FIELDS, f"sample_manifest.csv row {index}")


def _validate_record(row: dict[str, Any], required_fields: Iterable[str], split: str, path: Path) -> None:
    validate_main_route_record(row, required_fields, split=split, label=f"{path} row {row.get('sample_id')}")


def _validate_common_fields(row: dict[str, Any], required_fields: Iterable[str], label: str) -> None:
    for field in required_fields:
        if row.get(field) in ("", None):
            raise ValueError(f"{label} missing required field {field}")
    split = str(row.get("split") or "")
    if split not in SPLITS:
        raise ValueError(f"{label} has invalid split: {split}")
    hour = float(row.get("prediction_hour", -1))
    if not 0 <= hour < 336:
        raise ValueError(f"{label} prediction_hour out of range: {hour}")


def _validate_unique_sample_ids(rows: list[dict[str, Any]], label: str) -> None:
    counts = Counter(str(row.get("sample_id") or "") for row in rows)
    duplicates = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} has duplicate sample_id values: {duplicates[:5]}")


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} JSONL row must be an object")
            yield payload


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    if path.suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ValueError(f"{path} requires zstandard to read .zst shards") from exc
        raw = path.open("rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        import io

        return io.TextIOWrapper(reader, encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _relative_shards(root: Path, shards: dict[str, list[Path]]) -> dict[str, list[str]]:
    return {
        split: [str(path.resolve().relative_to(root.resolve())) for path in paths]
        for split, paths in shards.items()
    }
