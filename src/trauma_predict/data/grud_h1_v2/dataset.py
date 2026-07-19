from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from trauma_predict.data.grud_h1_sample.builder import content_hash
from trauma_predict.data.multires_event_v2.contract import (
    MultiresEventV2Contract,
    sha256_file,
)


H1_DATASET_SCHEMA = "grud_h1_baseline_dataset_manifest_v1"
H1_SAMPLE_SCHEMA = "grud_h1_baseline_input_sample_v1"
H1_DATASET_ID = "grud_h1_baseline_c4_20260717_v1"
TARGET_DATASET_ID = "multires_event_m4_target_v2_c4_full_20260714_r9"

H1_MANIFEST_HEADER = (
    "sample_id",
    "subject_id",
    "hadm_id",
    "stay_id",
    "prediction_hour",
    "split",
    "base_content_hash",
    "target_content_hash",
    "h1_content_hash",
    "h1_shard_key",
    "h1_line_index",
    "target_shard_key",
    "target_line_index",
)
TARGET_MANIFEST_HEADER = (
    "sample_id",
    "subject_id",
    "hadm_id",
    "stay_id",
    "prediction_hour",
    "split",
    "base_content_hash",
    "target_content_hash",
    "target_shard_key",
    "target_line_index",
)


@dataclass(frozen=True)
class ShardSpec:
    shard_key: str
    split: str
    path: Path
    sample_count: int
    sha256: str


@dataclass(frozen=True)
class GRUDH1V2ManifestEntry:
    sample_id: str
    subject_id: str
    hadm_id: str
    stay_id: str
    prediction_hour: int
    split: str
    base_content_hash: str
    target_content_hash: str
    h1_content_hash: str
    h1_shard_key: str
    h1_line_index: int
    target_shard_key: str
    target_line_index: int


class GRUDH1V2Dataset:
    """Hash-bound H1-input/full-r9-target view for the matched GRU-D route.

    The H1 manifest is the sample and split authority.  Every row is joined to
    the independently persisted full-r9 manifest by ``sample_id`` and both
    original content hashes before either gzip shard is opened.  Gzip shards
    are cached independently because H1 and target rows have different sizes.
    """

    def __init__(
        self,
        h1_root: str | Path,
        target_root: str | Path,
        *,
        split: str,
        contract: MultiresEventV2Contract | Any | None = None,
        cache_shards: int = 1,
        strict: bool = True,
        verify_shard_hashes: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"invalid dataset split: {split!r}")
        if cache_shards < 1:
            raise ValueError("cache_shards must be at least one")
        self.root = Path(h1_root).resolve()
        self.h1_root = self.root
        self.target_root = Path(target_root).resolve()
        self.split = split
        self.cache_shards = int(cache_shards)
        self.strict = bool(strict)
        self.verify_shard_hashes = bool(verify_shard_hashes)
        self.contract = contract or MultiresEventV2Contract.from_dataset_root(self.target_root)
        contract_root = Path(getattr(self.contract, "dataset_root", self.target_root)).resolve()
        if contract_root != self.target_root:
            raise ValueError("V2 target contract and target_root differ")

        self.manifest_payload = _read_json(self.root / "dataset_manifest.json")
        self.dataset_id = str(self.manifest_payload.get("dataset_id") or "")
        self.dataset_fingerprint = str(self.manifest_payload.get("fingerprint") or "")
        self.target_manifest_payload = _mapping(
            getattr(self.contract, "manifest", None), "target contract manifest"
        )
        self.target_dataset_id = str(self.target_manifest_payload.get("dataset_id") or "")
        self.contract_bundle_hash = str(getattr(self.contract, "contract_bundle_hash", ""))

        h1_manifest_path, target_manifest_path = self._validate_dataset_manifests()
        self.h1_shards = _load_h1_shards(self.root, self.manifest_payload)
        self.target_shards = _load_target_shards(
            self.target_root, self.target_manifest_payload
        )
        all_entries = _load_joined_entries(
            h1_manifest_path,
            target_manifest_path,
            self.h1_shards,
            self.target_shards,
        )
        self._validate_counts(all_entries)
        self.entries = tuple(entry for entry in all_entries if entry.split == self.split)
        if not self.entries:
            raise ValueError(f"H1 manifest contains no {self.split} samples")

        self.sample_ids = tuple(entry.sample_id for entry in self.entries)
        self.subject_ids = tuple(entry.subject_id for entry in self.entries)
        self.shard_keys = tuple(entry.h1_shard_key for entry in self.entries)
        self._h1_cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._target_cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        h1_record = self._record(
            entry.h1_shard_key,
            entry.h1_line_index,
            specs=self.h1_shards,
            cache=self._h1_cache,
            label="H1",
        )
        target_record = self._record(
            entry.target_shard_key,
            entry.target_line_index,
            specs=self.target_shards,
            cache=self._target_cache,
            label="target",
        )
        self._validate_join(entry, h1_record, target_record)
        return {
            "sample_id": entry.sample_id,
            "subject_id": entry.subject_id,
            "hadm_id": entry.hadm_id,
            "stay_id": entry.stay_id,
            "prediction_hour": entry.prediction_hour,
            "split": entry.split,
            "base_content_hash": entry.base_content_hash,
            "target_content_hash": entry.target_content_hash,
            "h1_content_hash": entry.h1_content_hash,
            "input_record": h1_record,
            "target_record": target_record,
        }

    def iter_indices(self, indices: Sequence[int]) -> Iterator[dict[str, Any]]:
        for index in indices:
            yield self[index]

    def indices_by_subject(self) -> Mapping[str, tuple[int, ...]]:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, entry in enumerate(self.entries):
            grouped[entry.subject_id].append(index)
        return {subject: tuple(indices) for subject, indices in grouped.items()}

    def _validate_dataset_manifests(self) -> tuple[Path, Path]:
        if self.manifest_payload.get("schema") != H1_DATASET_SCHEMA:
            raise ValueError("H1 dataset manifest schema mismatch")
        if self.dataset_id != H1_DATASET_ID:
            raise ValueError(f"H1 dataset_id mismatch: {self.dataset_id!r}")
        if self.manifest_payload.get("status") != "SUCCEEDED":
            raise ValueError("H1 dataset is not marked SUCCEEDED")
        if self.manifest_payload.get("full_authority_build") is not True:
            raise ValueError("GRU-D training requires the complete H1 authority build")
        input_contract = _mapping(
            self.manifest_payload.get("input_contract"), "H1 input_contract"
        )
        if (
            input_contract.get("resolution") != "H1"
            or int(input_contract.get("registered_channels", -1)) != 118
            or int(input_contract.get("max_history_hours", -1)) != 312
        ):
            raise ValueError("H1 input contract differs from the frozen 312x118 view")
        target_contract = _mapping(
            self.manifest_payload.get("target_contract"), "H1 target_contract"
        )
        if str(target_contract.get("dataset_id") or "") != self.target_dataset_id:
            raise ValueError("H1 target dataset identity differs from mounted full_r9")
        if self.target_dataset_id != TARGET_DATASET_ID:
            raise ValueError(f"target dataset_id mismatch: {self.target_dataset_id!r}")

        h1_files = _mapping(self.manifest_payload.get("files"), "H1 files")
        h1_manifest_spec = _mapping(h1_files.get("sample_manifest"), "H1 sample_manifest")
        h1_manifest_path = self.root / str(h1_manifest_spec.get("path") or "")
        target_files = _mapping(self.target_manifest_payload.get("files"), "target files")
        target_manifest_spec = _mapping(
            target_files.get("sample_manifest"), "target sample_manifest"
        )
        target_manifest_path = self.target_root / str(
            target_manifest_spec.get("path") or ""
        )
        if self.strict:
            _assert_hash(
                h1_manifest_path,
                str(h1_manifest_spec.get("sha256") or ""),
                "H1 sample manifest",
            )
            _assert_hash(
                target_manifest_path,
                str(target_manifest_spec.get("sha256") or ""),
                "full_r9 sample manifest",
            )
            templates_spec = _mapping(
                h1_files.get("h1_event_templates"), "H1 event templates"
            )
            _assert_hash(
                self.root / str(templates_spec.get("path") or ""),
                str(templates_spec.get("sha256") or ""),
                "H1 channel registry",
            )
        authority = _mapping(self.manifest_payload.get("authority"), "H1 authority")
        if str(authority.get("target_sample_manifest_sha256") or "") != str(
            target_manifest_spec.get("sha256") or ""
        ):
            raise ValueError("H1 authority is not bound to the mounted full_r9 manifest")
        return h1_manifest_path, target_manifest_path

    def _validate_counts(self, entries: Sequence[GRUDH1V2ManifestEntry]) -> None:
        counts = _mapping(self.manifest_payload.get("counts"), "H1 counts")
        if len(entries) != int(counts.get("samples", -1)):
            raise ValueError("H1 manifest row count differs from dataset manifest")
        observed_split = Counter(entry.split for entry in entries)
        expected_split = {
            str(key): int(value)
            for key, value in _mapping(counts.get("by_split"), "H1 counts.by_split").items()
        }
        if dict(observed_split) != expected_split:
            raise ValueError("H1 manifest split counts differ from dataset manifest")
        if len(self.h1_shards) != int(counts.get("shards", -1)):
            raise ValueError("H1 shard count differs from dataset manifest")

    def _record(
        self,
        shard_key: str,
        line_index: int,
        *,
        specs: Mapping[str, ShardSpec],
        cache: OrderedDict[str, list[dict[str, Any]]],
        label: str,
    ) -> dict[str, Any]:
        rows = cache.pop(shard_key, None)
        if rows is None:
            spec = specs.get(shard_key)
            if spec is None:
                raise ValueError(f"unknown {label} shard: {shard_key}")
            if self.verify_shard_hashes:
                _assert_hash(spec.path, spec.sha256, f"{label} shard {shard_key}")
            rows = _read_jsonl_gzip(spec.path)
            if len(rows) != spec.sample_count:
                raise ValueError(
                    f"{label} shard row count mismatch for {shard_key}: "
                    f"{len(rows)} != {spec.sample_count}"
                )
        cache[shard_key] = rows
        while len(cache) > self.cache_shards:
            cache.popitem(last=False)
        if not 0 <= line_index < len(rows):
            raise ValueError(f"{label} line index exceeds {shard_key}: {line_index}")
        return rows[line_index]

    def _validate_join(
        self,
        entry: GRUDH1V2ManifestEntry,
        h1_record: Mapping[str, Any],
        target_record: Mapping[str, Any],
    ) -> None:
        if h1_record.get("schema") != H1_SAMPLE_SCHEMA:
            raise ValueError(f"H1 sample schema mismatch for {entry.sample_id}")
        self.contract.validate_target_record(target_record, verify_content_hash=self.strict)
        identity = {
            "sample_id": entry.sample_id,
            "subject_id": entry.subject_id,
            "hadm_id": entry.hadm_id,
            "stay_id": entry.stay_id,
            "prediction_hour": entry.prediction_hour,
            "split": entry.split,
        }
        for key, expected in identity.items():
            if str(h1_record.get(key)) != str(expected):
                raise ValueError(f"H1 {key} mismatch for {entry.sample_id}")
            if str(target_record.get(key)) != str(expected):
                raise ValueError(f"full_r9 {key} mismatch for {entry.sample_id}")
        observed_h1_hash = str(h1_record.get("content_hash") or "")
        if observed_h1_hash != entry.h1_content_hash:
            raise ValueError(f"H1 content hash join mismatch for {entry.sample_id}")
        if self.strict and content_hash(dict(h1_record)) != observed_h1_hash:
            raise ValueError(f"H1 payload content hash mismatch for {entry.sample_id}")
        source_reference = _mapping(h1_record.get("source_reference"), "H1 source_reference")
        if str(source_reference.get("base_content_hash") or "") != entry.base_content_hash:
            raise ValueError(f"base content hash join mismatch for {entry.sample_id}")
        if str(source_reference.get("target_content_hash") or "") != entry.target_content_hash:
            raise ValueError(f"H1 target content hash join mismatch for {entry.sample_id}")
        if str(target_record.get("base_content_hash") or "") != entry.base_content_hash:
            raise ValueError(f"full_r9 base content hash mismatch for {entry.sample_id}")
        if str(target_record.get("target_content_hash") or "") != entry.target_content_hash:
            raise ValueError(f"full_r9 target content hash mismatch for {entry.sample_id}")
        target_reference = _mapping(h1_record.get("target_reference"), "H1 target_reference")
        expected_reference = {
            "sample_id": entry.sample_id,
            "target_content_hash": entry.target_content_hash,
            "target_shard_key": entry.target_shard_key,
            "target_line_index": entry.target_line_index,
        }
        for key, expected in expected_reference.items():
            if str(target_reference.get(key)) != str(expected):
                raise ValueError(f"H1 target reference {key} mismatch for {entry.sample_id}")


def _load_joined_entries(
    h1_manifest_path: Path,
    target_manifest_path: Path,
    h1_shards: Mapping[str, ShardSpec],
    target_shards: Mapping[str, ShardSpec],
) -> tuple[GRUDH1V2ManifestEntry, ...]:
    h1_rows = _read_csv_exact(h1_manifest_path, H1_MANIFEST_HEADER)
    target_rows = _read_csv_exact(target_manifest_path, TARGET_MANIFEST_HEADER)
    target_by_id: dict[str, Mapping[str, str]] = {}
    for row in target_rows:
        sample_id = str(row["sample_id"])
        if not sample_id or sample_id in target_by_id:
            raise ValueError(f"duplicate/empty sample_id in full_r9 manifest: {sample_id!r}")
        target_by_id[sample_id] = row
    if [row["sample_id"] for row in h1_rows] != [row["sample_id"] for row in target_rows]:
        raise ValueError("H1 and full_r9 sample manifests differ in sample identity/order")

    entries: list[GRUDH1V2ManifestEntry] = []
    seen: set[str] = set()
    identity_keys = (
        "subject_id", "hadm_id", "stay_id", "prediction_hour", "split",
        "base_content_hash", "target_content_hash", "target_shard_key",
        "target_line_index",
    )
    for row in h1_rows:
        sample_id = str(row["sample_id"])
        if not sample_id or sample_id in seen:
            raise ValueError(f"duplicate/empty sample_id in H1 manifest: {sample_id!r}")
        seen.add(sample_id)
        target = target_by_id.get(sample_id)
        if target is None:
            raise ValueError(f"H1 sample is absent from full_r9 manifest: {sample_id}")
        for key in identity_keys:
            if str(row[key]) != str(target[key]):
                raise ValueError(f"H1/full_r9 manifest mismatch for {sample_id}: {key}")
        split = str(row["split"])
        if split not in {"train", "val", "test"}:
            raise ValueError(f"invalid split in H1 manifest: {split!r}")
        h1_shard_key = str(row["h1_shard_key"])
        target_shard_key = str(row["target_shard_key"])
        h1_line_index = _nonnegative_int(row["h1_line_index"], "h1_line_index")
        target_line_index = _nonnegative_int(
            row["target_line_index"], "target_line_index"
        )
        if h1_shard_key not in h1_shards or target_shard_key not in target_shards:
            raise ValueError(f"manifest references an unknown shard for {sample_id}")
        if h1_line_index >= h1_shards[h1_shard_key].sample_count:
            raise ValueError(f"H1 line index exceeds declared shard for {sample_id}")
        if target_line_index >= target_shards[target_shard_key].sample_count:
            raise ValueError(f"target line index exceeds declared shard for {sample_id}")
        entries.append(
            GRUDH1V2ManifestEntry(
                sample_id=sample_id,
                subject_id=str(row["subject_id"]),
                hadm_id=str(row["hadm_id"]),
                stay_id=str(row["stay_id"]),
                prediction_hour=int(row["prediction_hour"]),
                split=split,
                base_content_hash=_sha256(row["base_content_hash"], "base_content_hash"),
                target_content_hash=_sha256(
                    row["target_content_hash"], "target_content_hash"
                ),
                h1_content_hash=_sha256(row["h1_content_hash"], "h1_content_hash"),
                h1_shard_key=h1_shard_key,
                h1_line_index=h1_line_index,
                target_shard_key=target_shard_key,
                target_line_index=target_line_index,
            )
        )
    _validate_locator_coverage(entries, h1_shards, target_shards)
    return tuple(entries)


def _validate_locator_coverage(
    entries: Sequence[GRUDH1V2ManifestEntry],
    h1_shards: Mapping[str, ShardSpec],
    target_shards: Mapping[str, ShardSpec],
) -> None:
    h1_indices: dict[str, list[int]] = defaultdict(list)
    target_indices: dict[str, list[int]] = defaultdict(list)
    for entry in entries:
        h1_indices[entry.h1_shard_key].append(entry.h1_line_index)
        target_indices[entry.target_shard_key].append(entry.target_line_index)
    for specs, observed, label in (
        (h1_shards, h1_indices, "H1"),
        (target_shards, target_indices, "target"),
    ):
        for shard_key, spec in specs.items():
            if sorted(observed.get(shard_key, ())) != list(range(spec.sample_count)):
                raise ValueError(f"{label} manifest does not cover {shard_key} exactly once")


def _load_h1_shards(root: Path, manifest: Mapping[str, Any]) -> dict[str, ShardSpec]:
    files = _mapping(manifest.get("files"), "H1 files")
    values = _mapping(files.get("h1_shards"), "H1 shards")
    result: dict[str, ShardSpec] = {}
    for key, raw in values.items():
        item = _mapping(raw, f"H1 shard {key}")
        shard_key = str(item.get("shard_key") or key)
        if shard_key != str(key) or shard_key in result:
            raise ValueError(f"invalid/duplicate H1 shard key: {key}")
        result[shard_key] = ShardSpec(
            shard_key=shard_key,
            split=str(item.get("split") or ""),
            path=root / str(item.get("sample_path") or ""),
            sample_count=_positive_int(item.get("samples"), "H1 shard samples"),
            sha256=_sha256(item.get("sample_sha256"), "H1 shard sha256"),
        )
    if not result:
        raise ValueError("H1 dataset manifest contains no shards")
    return result


def _load_target_shards(root: Path, manifest: Mapping[str, Any]) -> dict[str, ShardSpec]:
    files = _mapping(manifest.get("files"), "target files")
    values = _mapping(files.get("target_shards"), "target shards")
    result: dict[str, ShardSpec] = {}
    for key, raw in values.items():
        item = _mapping(raw, f"target shard {key}")
        shard_key = str(key)
        if shard_key in result:
            raise ValueError(f"duplicate target shard key: {shard_key}")
        result[shard_key] = ShardSpec(
            shard_key=shard_key,
            split=shard_key.split("-", 1)[0],
            path=root / str(item.get("path") or ""),
            sample_count=_positive_int(item.get("samples"), "target shard samples"),
            sha256=_sha256(item.get("sha256"), "target shard sha256"),
        )
    if not result:
        raise ValueError("target dataset manifest contains no shards")
    return result


def _read_csv_exact(path: Path, expected_header: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ValueError(f"manifest header drift: {path}")
        return list(reader)


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(payload)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != _sha256(expected, f"{label} hash"):
        raise ValueError(f"{label} hash mismatch")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sha256(value: Any, label: str) -> str:
    result = str(value or "")
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if result < 0 or str(result) != str(value):
        raise ValueError(f"{label} must be a non-negative integer")
    return result


def _positive_int(value: Any, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


__all__ = [
    "GRUDH1V2Dataset",
    "GRUDH1V2ManifestEntry",
    "H1_DATASET_ID",
    "TARGET_DATASET_ID",
]
