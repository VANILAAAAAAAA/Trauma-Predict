from __future__ import annotations

import copy
import csv
import gzip
import json
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .contract import MultiresEventV2Contract, sha256_file


SIDECAR_MANIFEST_HEADER = (
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

BASE_INPUT_KEYS = (
    "schema",
    "sample_id",
    "subject_id",
    "hadm_id",
    "stay_id",
    "sample_key",
    "split",
    "prediction_hour",
    "static",
    "input_allocation",
    "block_table",
    "input_events",
    "input_source_count",
    "registry",
    "content_hash",
)


@dataclass(frozen=True)
class TargetShardSpec:
    shard_key: str
    path: Path
    sample_count: int
    sha256: str


@dataclass(frozen=True)
class TargetManifestEntry:
    sample_id: str
    subject_id: str
    hadm_id: str
    stay_id: str
    prediction_hour: int
    split: str
    base_content_hash: str
    target_content_hash: str
    target_shard_key: str
    target_line_index: int


class MultiresEventV2Dataset:
    """Strict sample-id/hash join of immutable V1 inputs and V2 M4 targets.

    ``base_dataset`` is the existing map-style V1 loader for one persisted split.
    Its V1 target arrays are used by that loader for its own shape checks, but this
    wrapper never exposes or batches them as supervision.
    """

    def __init__(
        self,
        base_dataset: Any,
        target_root: str | Path,
        *,
        contract: MultiresEventV2Contract | None = None,
        cache_shards: int = 1,
        strict: bool = True,
        verify_shard_hashes: bool = False,
    ) -> None:
        if cache_shards < 1:
            raise ValueError("cache_shards must be at least one")
        self.base_dataset = base_dataset
        self.root = Path(target_root).resolve()
        self.contract = contract or MultiresEventV2Contract.from_dataset_root(self.root)
        if self.contract.dataset_root != self.root:
            raise ValueError("V2 contract and target dataset roots differ")
        self.split = str(getattr(base_dataset, "split", ""))
        if self.split not in {"train", "val", "test"}:
            raise ValueError("base V1 dataset must expose a train/val/test split")
        self.strict = bool(strict)
        self.cache_shards = int(cache_shards)
        self.manifest_payload = self.contract.manifest
        self.dataset_id = str(self.manifest_payload["dataset_id"])
        self.contract_bundle_hash = self.contract.contract_bundle_hash
        self._validate_base_authority()

        files = _mapping(self.manifest_payload.get("files"), "dataset_manifest.files")
        sample_manifest_spec = _mapping(files.get("sample_manifest"), "files.sample_manifest")
        sample_manifest_path = self.root / str(sample_manifest_spec.get("path") or "")
        if self.strict:
            _assert_file_hash(
                sample_manifest_path,
                str(sample_manifest_spec.get("sha256") or ""),
                "V2 sample manifest",
            )
            subject_split_spec = _mapping(files.get("subject_split"), "files.subject_split")
            _assert_file_hash(
                self.root / str(subject_split_spec.get("path") or ""),
                str(subject_split_spec.get("sha256") or ""),
                "V2 subject split",
            )
        self.shards = _load_target_shards(
            self.root,
            files,
            verify_hashes=verify_shard_hashes,
        )
        all_entries = _load_target_manifest_entries(sample_manifest_path, self.shards)
        self._validate_sidecar_counts(all_entries)
        self.entries = tuple(entry for entry in all_entries if entry.split == self.split)
        if not self.entries:
            raise ValueError(f"V2 target manifest contains no {self.split} samples")

        base_ids = tuple(str(value) for value in getattr(base_dataset, "sample_ids", ()))
        if len(base_ids) != len(base_dataset):
            raise ValueError("base V1 dataset sample_ids do not align with its length")
        if len(base_ids) != len(set(base_ids)):
            raise ValueError("base V1 dataset contains duplicate sample_id values")
        self._base_index = {sample_id: index for index, sample_id in enumerate(base_ids)}
        missing_base = [entry.sample_id for entry in self.entries if entry.sample_id not in self._base_index]
        if missing_base:
            raise ValueError(
                f"V2 sidecar sample_id is absent from base V1 split: {missing_base[:5]}"
            )
        self.sample_ids = tuple(entry.sample_id for entry in self.entries)
        self.subject_ids = tuple(entry.subject_id for entry in self.entries)
        self.shard_keys = tuple(entry.target_shard_key for entry in self.entries)
        self._cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        base_record = self.base_dataset[self._base_index[entry.sample_id]]
        target_record = self._target_record(entry)
        self._validate_join(entry, base_record, target_record)
        return {
            "sample_id": entry.sample_id,
            "subject_id": entry.subject_id,
            "hadm_id": entry.hadm_id,
            "stay_id": entry.stay_id,
            "prediction_hour": entry.prediction_hour,
            "split": entry.split,
            "base_content_hash": entry.base_content_hash,
            "target_content_hash": entry.target_content_hash,
            "input_record": _input_only_record(base_record),
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

    def _target_record(self, entry: TargetManifestEntry) -> dict[str, Any]:
        rows = self._load_shard(entry.target_shard_key)
        if not 0 <= entry.target_line_index < len(rows):
            raise ValueError(
                f"V2 target line index exceeds {entry.target_shard_key}: "
                f"{entry.target_line_index}"
            )
        return rows[entry.target_line_index]

    def _load_shard(self, shard_key: str) -> list[dict[str, Any]]:
        cached = self._cache.pop(shard_key, None)
        if cached is not None:
            self._cache[shard_key] = cached
            return cached
        spec = self.shards.get(shard_key)
        if spec is None:
            raise ValueError(f"V2 sample manifest references unknown shard={shard_key}")
        rows: list[dict[str, Any]] = []
        with gzip.open(spec.path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{spec.path}:{line_number} must contain an object")
                rows.append(payload)
        if len(rows) != spec.sample_count:
            raise ValueError(
                f"V2 target shard count mismatch for {shard_key}: "
                f"{len(rows)} != {spec.sample_count}"
            )
        self._cache[shard_key] = rows
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return rows

    def _validate_join(
        self,
        entry: TargetManifestEntry,
        base_record: Mapping[str, Any],
        target_record: Mapping[str, Any],
    ) -> None:
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
            if str(base_record.get(key)) != str(expected):
                raise ValueError(
                    f"base V1 {key} mismatch for {entry.sample_id}: "
                    f"{base_record.get(key)} != {expected}"
                )
            if str(target_record.get(key)) != str(expected):
                raise ValueError(
                    f"V2 target {key} mismatch for {entry.sample_id}: "
                    f"{target_record.get(key)} != {expected}"
                )
        base_hash = str(base_record.get("content_hash") or "")
        target_base_hash = str(target_record.get("base_content_hash") or "")
        if not (base_hash == entry.base_content_hash == target_base_hash):
            raise ValueError(
                f"base_content_hash join mismatch for {entry.sample_id}: "
                f"base={base_hash}, manifest={entry.base_content_hash}, "
                f"target={target_base_hash}"
            )
        target_hash = str(target_record.get("target_content_hash") or "")
        if target_hash != entry.target_content_hash:
            raise ValueError(
                f"target_content_hash manifest mismatch for {entry.sample_id}: "
                f"{target_hash} != {entry.target_content_hash}"
            )

    def _validate_base_authority(self) -> None:
        base = _mapping(self.manifest_payload.get("base_dataset"), "base_dataset")
        observed_id = str(getattr(self.base_dataset, "dataset_id", ""))
        observed_fingerprint = str(getattr(self.base_dataset, "dataset_fingerprint", ""))
        if observed_id != str(base.get("dataset_id")):
            raise ValueError(f"base V1 dataset_id mismatch: {observed_id} != {base.get('dataset_id')}")
        if observed_fingerprint != str(base.get("fingerprint")):
            raise ValueError("base V1 dataset fingerprint mismatch")
        base_root = Path(getattr(self.base_dataset, "root", "")).resolve()
        if not base_root.is_dir():
            raise FileNotFoundError(base_root)
        # base_dataset.root in the sidecar is build provenance, not a runtime
        # location. Dataset identity and relocation safety are established by the
        # immutable ID/fingerprint and the three persisted authority hashes below.
        if self.strict:
            _assert_file_hash(
                base_root / "dataset_manifest.json",
                str(base.get("dataset_manifest_sha256") or ""),
                "base V1 dataset manifest",
            )
            manifest = _mapping(
                getattr(self.base_dataset, "manifest_payload", None),
                "base_dataset.manifest_payload",
            )
            files = _mapping(manifest.get("files"), "base_dataset.files")
            _assert_file_hash(
                base_root / str(files.get("sample_manifest") or "sample_manifest.csv"),
                str(base.get("sample_manifest_sha256") or ""),
                "base V1 sample manifest",
            )
            _assert_file_hash(
                base_root / str(files.get("subject_split") or "subject_split.csv"),
                str(base.get("subject_split_sha256") or ""),
                "base V1 subject split",
            )

    def _validate_sidecar_counts(self, entries: Sequence[TargetManifestEntry]) -> None:
        counts = _mapping(self.manifest_payload.get("counts"), "counts")
        if len(entries) != int(counts.get("samples", -1)):
            raise ValueError("V2 sample manifest row count differs from dataset manifest")
        observed_split = Counter(entry.split for entry in entries)
        declared_split = _mapping(counts.get("by_split"), "counts.by_split")
        if dict(observed_split) != {str(key): int(value) for key, value in declared_split.items()}:
            raise ValueError("V2 sample manifest split counts differ from dataset manifest")
        observed_shards = Counter(entry.target_shard_key for entry in entries)
        if len(observed_shards) != int(counts.get("shards", -1)):
            raise ValueError("V2 sample manifest shard count differs from dataset manifest")
        for shard_key, count in observed_shards.items():
            if count != self.shards[shard_key].sample_count:
                raise ValueError(f"V2 sample manifest count mismatch for shard {shard_key}")


def _input_only_record(record: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in BASE_INPUT_KEYS if key not in record]
    if missing:
        raise ValueError(f"base V1 record is missing input keys: {missing}")
    result = {key: copy.deepcopy(record[key]) for key in BASE_INPUT_KEYS}
    block_table = result["block_table"]
    if not isinstance(block_table, list):
        raise ValueError("base V1 block_table must be an array")
    result["block_table"] = [block for block in block_table if block.get("side") == "input"]
    if not result["block_table"]:
        raise ValueError("base V1 input contains no input blocks")
    if len(result["input_events"]) != len(result["input_source_count"]):
        raise ValueError("base V1 input events and source counts are misaligned")
    forbidden = {"target_events", "target_mask", "target_source_count", "target_contract"}
    if forbidden.intersection(result):
        raise AssertionError("V1 target supervision leaked into the V2 input record")
    return result


def _load_target_shards(
    root: Path,
    files: Mapping[str, Any],
    *,
    verify_hashes: bool,
) -> dict[str, TargetShardSpec]:
    raw_shards = _mapping(files.get("target_shards"), "files.target_shards")
    result: dict[str, TargetShardSpec] = {}
    for raw_key, raw_value in raw_shards.items():
        shard_key = str(raw_key)
        item = _mapping(raw_value, f"files.target_shards.{shard_key}")
        path = root / str(item.get("path") or "")
        if not path.is_file():
            raise FileNotFoundError(path)
        declared_hash = str(item.get("sha256") or "")
        if verify_hashes:
            _assert_file_hash(path, declared_hash, f"V2 target shard {shard_key}")
        result[shard_key] = TargetShardSpec(
            shard_key=shard_key,
            path=path,
            sample_count=int(item.get("samples", -1)),
            sha256=declared_hash,
        )
        if result[shard_key].sample_count < 1:
            raise ValueError(f"V2 target shard {shard_key} has an invalid sample count")
    if not result:
        raise ValueError("V2 dataset manifest declares no target shards")
    return result


def _load_target_manifest_entries(
    path: Path,
    shards: Mapping[str, TargetShardSpec],
) -> tuple[TargetManifestEntry, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    entries: list[TargetManifestEntry] = []
    seen_ids: set[str] = set()
    indices_by_shard: dict[str, list[int]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SIDECAR_MANIFEST_HEADER:
            raise ValueError("V2 sample_manifest.csv header mismatch")
        for line_number, row in enumerate(reader, start=2):
            sample_id = str(row["sample_id"])
            if not sample_id or sample_id in seen_ids:
                raise ValueError(f"duplicate/empty V2 sample_id at line {line_number}: {sample_id}")
            seen_ids.add(sample_id)
            split = str(row["split"])
            if split not in {"train", "val", "test"}:
                raise ValueError(f"invalid V2 split at line {line_number}: {split}")
            shard_key = str(row["target_shard_key"])
            if shard_key not in shards:
                raise ValueError(f"unknown V2 target shard at line {line_number}: {shard_key}")
            line_index = int(row["target_line_index"])
            if line_index < 0:
                raise ValueError(f"negative V2 target_line_index at line {line_number}")
            indices_by_shard[shard_key].append(line_index)
            entries.append(
                TargetManifestEntry(
                    sample_id=sample_id,
                    subject_id=str(row["subject_id"]),
                    hadm_id=str(row["hadm_id"]),
                    stay_id=str(row["stay_id"]),
                    prediction_hour=int(row["prediction_hour"]),
                    split=split,
                    base_content_hash=str(row["base_content_hash"]),
                    target_content_hash=str(row["target_content_hash"]),
                    target_shard_key=shard_key,
                    target_line_index=line_index,
                )
            )
    for shard_key, indices in indices_by_shard.items():
        if sorted(indices) != list(range(shards[shard_key].sample_count)):
            raise ValueError(f"V2 target line indices are not a complete range for {shard_key}")
    return tuple(entries)


def _assert_file_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} sha256 mismatch: {observed} != {expected}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value
