from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .contract import EventTemplateRegistry, SupervisionContract


@dataclass(frozen=True)
class ShardSpec:
    shard_key: str
    split: str
    path: Path
    sample_count: int
    sha256: str


@dataclass(frozen=True)
class ManifestEntry:
    sample_id: str
    subject_id: str
    hadm_id: str
    stay_id: str
    prediction_hour: int
    split: str
    shard_key: str
    row_index: int


class MultiresEventDataset:
    """Map-style view over persisted gzip shards without rebuilding splits.

    Gzip does not support cheap random seeks. The sampler paired with this dataset
    groups yielded indices by shard, and the small LRU cache therefore decompresses
    each selected shard at most once per epoch.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        supervision: SupervisionContract,
        *,
        cache_shards: int = 1,
        strict: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"invalid dataset split: {split}")
        if cache_shards < 1:
            raise ValueError("cache_shards must be at least one")
        self.root = Path(dataset_root).resolve()
        self.split = split
        self.supervision = supervision
        self.cache_shards = int(cache_shards)
        self.manifest_payload = _read_json(self.root / "dataset_manifest.json")
        self.dataset_id = str(self.manifest_payload.get("dataset_id") or "")
        self.dataset_fingerprint = str(self.manifest_payload.get("fingerprint") or "")
        self.source_fingerprint = str(
            _mapping(self.manifest_payload.get("source"), "dataset_manifest.source").get(
                "source_fingerprint"
            )
            or ""
        )
        self.shards = _load_shard_specs(self.root, self.manifest_payload)
        self.entries = _load_manifest_entries(
            self.root / str(_mapping(self.manifest_payload["files"], "files")["sample_manifest"]),
            split,
            self.shards,
        )
        if not self.entries:
            raise ValueError(f"sample_manifest contains no {split} samples")
        self.subject_ids = tuple(entry.subject_id for entry in self.entries)
        self.sample_ids = tuple(entry.sample_id for entry in self.entries)
        self.shard_keys = tuple(entry.shard_key for entry in self.entries)
        self.templates = EventTemplateRegistry.from_json(
            self.root / str(_mapping(self.manifest_payload["files"], "files")["event_templates"])
        )
        self._cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        first = self._raw_record(self.entries[0])
        self.target_layout = supervision.compile_target_layout(first, self.templates, strict=strict)
        if strict:
            self._validate_manifest_contract()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        record = self._raw_record(entry)
        if str(record.get("sample_id")) != entry.sample_id:
            raise ValueError(
                f"shard row mismatch at {entry.shard_key}:{entry.row_index}: "
                f"{record.get('sample_id')} != {entry.sample_id}"
            )
        if str(record.get("split")) != self.split:
            raise ValueError(f"persisted shard row crossed split boundary for {entry.sample_id}")
        return self.supervision.filter_input_record(record)

    def iter_indices(self, indices: Sequence[int]) -> Iterator[dict[str, Any]]:
        for index in indices:
            yield self[index]

    def indices_by_subject(self) -> Mapping[str, tuple[int, ...]]:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, entry in enumerate(self.entries):
            grouped[entry.subject_id].append(index)
        return {key: tuple(values) for key, values in grouped.items()}

    def _raw_record(self, entry: ManifestEntry) -> dict[str, Any]:
        rows = self._load_shard(entry.shard_key)
        if not 0 <= entry.row_index < len(rows):
            raise ValueError(
                f"sample_manifest row index exceeds {entry.shard_key}: {entry.row_index}"
            )
        return rows[entry.row_index]

    def _load_shard(self, shard_key: str) -> list[dict[str, Any]]:
        cached = self._cache.pop(shard_key, None)
        if cached is not None:
            self._cache[shard_key] = cached
            return cached
        spec = self.shards.get(shard_key)
        if spec is None:
            raise ValueError(f"sample_manifest references unknown shard_key={shard_key}")
        rows: list[dict[str, Any]] = []
        with gzip.open(spec.path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{spec.path}:{line_number} row must be an object")
                rows.append(payload)
        if len(rows) != spec.sample_count:
            raise ValueError(
                f"shard sample count mismatch for {shard_key}: {len(rows)} != {spec.sample_count}"
            )
        self._cache[shard_key] = rows
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return rows

    def _validate_manifest_contract(self) -> None:
        if self.manifest_payload.get("schema") != "multires_event_dataset_manifest_v2":
            raise ValueError("dataset manifest schema mismatch")
        if self.manifest_payload.get("status") != "SUCCEEDED" or self.manifest_payload.get("plan_only"):
            raise ValueError("dataset manifest is not a completed formal artifact")
        declared = _mapping(self.manifest_payload.get("counts"), "dataset_manifest.counts")
        split_counts = _mapping(declared.get("built_by_split"), "counts.built_by_split")
        if len(self.entries) != int(split_counts[self.split]):
            raise ValueError(
                f"sample_manifest {self.split} count mismatch: "
                f"{len(self.entries)} != {split_counts[self.split]}"
            )
        observed_per_shard = Counter(entry.shard_key for entry in self.entries)
        for shard_key, observed in observed_per_shard.items():
            expected = self.shards[shard_key].sample_count
            if observed != expected:
                raise ValueError(
                    f"sample_manifest count for {shard_key} is {observed}, expected {expected}"
                )


def _load_shard_specs(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, ShardSpec]:
    files = _mapping(manifest.get("files"), "dataset_manifest.files")
    values = _mapping(files.get("shards"), "dataset_manifest.files.shards")
    result: dict[str, ShardSpec] = {}
    for shard_key, raw in values.items():
        item = _mapping(raw, f"files.shards.{shard_key}")
        path = (root / str(item["sample_path"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing canonical gzip shard: {path}")
        result[str(shard_key)] = ShardSpec(
            shard_key=str(shard_key),
            split=str(item["split"]),
            path=path,
            sample_count=int(item["sample_count"]),
            sha256=str(item["sample_sha256"]),
        )
    if not result:
        raise ValueError("dataset manifest declares no shards")
    return result


def _load_manifest_entries(
    path: Path,
    split: str,
    shards: Mapping[str, ShardSpec],
) -> list[ManifestEntry]:
    if not path.is_file():
        raise FileNotFoundError(f"missing persisted sample_manifest.csv: {path}")
    row_in_shard: Counter[str] = Counter()
    entries: list[ManifestEntry] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id", "subject_id", "hadm_id", "stay_id", "prediction_hour", "split", "shard_key"
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("sample_manifest.csv header does not match multires_event_v1")
        for line_number, row in enumerate(reader, start=2):
            shard_key = str(row["shard_key"])
            row_index = row_in_shard[shard_key]
            row_in_shard[shard_key] += 1
            row_split = str(row["split"])
            spec = shards.get(shard_key)
            if spec is None:
                raise ValueError(f"sample_manifest.csv:{line_number} unknown shard_key={shard_key}")
            if spec.split != row_split:
                raise ValueError(f"sample_manifest.csv:{line_number} split/shard mismatch")
            sample_id = str(row["sample_id"])
            if sample_id in seen_ids:
                raise ValueError(f"duplicate sample_id in sample_manifest.csv: {sample_id}")
            seen_ids.add(sample_id)
            if row_split != split:
                continue
            entries.append(ManifestEntry(
                sample_id=sample_id,
                subject_id=str(row["subject_id"]),
                hadm_id=str(row["hadm_id"]),
                stay_id=str(row["stay_id"]),
                prediction_hour=int(row["prediction_hour"]),
                split=row_split,
                shard_key=shard_key,
                row_index=row_index,
            ))
    return entries


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value
