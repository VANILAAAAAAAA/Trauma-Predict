from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import time
from collections import Counter, OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO

from .builder import GRUDH1SampleBuilder
from .io import load_stay
from .registry import EventRegistry


SPLITS = ("train", "val", "test")
BASE_MANIFEST_COLUMNS = (
    "sample_id", "subject_id", "hadm_id", "stay_id", "prediction_hour",
    "split", "content_hash", "shard_key", "trajectory_path",
)
TARGET_MANIFEST_COLUMNS = (
    "sample_id", "subject_id", "hadm_id", "stay_id", "prediction_hour",
    "split", "base_content_hash", "target_content_hash", "target_shard_key",
    "target_line_index",
)
OUTPUT_MANIFEST_COLUMNS = (
    "sample_id", "subject_id", "hadm_id", "stay_id", "prediction_hour",
    "split", "base_content_hash", "target_content_hash", "h1_content_hash",
    "h1_shard_key", "h1_line_index", "target_shard_key", "target_line_index",
)


@dataclass(frozen=True)
class AuthorityRow:
    sample_id: str
    subject_id: str
    hadm_id: str
    stay_id: str
    prediction_hour: int
    split: str
    base_content_hash: str
    base_shard_key: str
    target_content_hash: str
    target_shard_key: str
    target_line_index: int


@dataclass(frozen=True)
class BuildContract:
    dataset_id: str
    expected_base_manifest_sha256: str
    expected_target_manifest_sha256: str
    expected_samples: int
    expected_split_counts: dict[str, int]
    expected_shards: int
    max_history_hours: int = 312
    h1_template_count: int = 118


class AtomicGzipJsonlWriter:
    def __init__(self, final_path: Path) -> None:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        self.final_path = final_path
        self.partial_path = final_path.with_name(f"{final_path.name}.partial-{os.getpid()}")
        self.partial_path.unlink(missing_ok=True)
        self.raw = self.partial_path.open("wb")
        compressed = gzip.GzipFile(filename="", mode="wb", fileobj=self.raw, mtime=0)
        self.text: TextIO = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
        self.closed = False

    def write(self, payload: dict[str, Any]) -> None:
        self.text.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    def commit(self) -> None:
        if self.closed:
            return
        self.text.flush()
        self.text.close()
        if not self.raw.closed:
            self.raw.flush()
            os.fsync(self.raw.fileno())
            self.raw.close()
        os.replace(self.partial_path, self.final_path)
        _fsync_directory(self.final_path.parent)
        self.closed = True

    def abort(self) -> None:
        if not self.closed:
            try:
                self.text.close()
            finally:
                if not self.raw.closed:
                    self.raw.close()
                self.closed = True
        self.partial_path.unlink(missing_ok=True)


def build_grud_h1_dataset(
    *,
    field_ready_root: Path,
    base_dataset_root: Path,
    target_dataset_root: Path,
    output_root: Path,
    registry_path: Path,
    contract: BuildContract,
    workers: int,
    resume: bool = False,
    max_shards: int | None = None,
) -> dict[str, Any]:
    """Build an input-only H1 sidecar over the frozen V1/r9 sample authority."""
    roots = {
        "field_ready": field_ready_root.resolve(),
        "base": base_dataset_root.resolve(),
        "target": target_dataset_root.resolve(),
        "output": output_root.resolve(),
    }
    if workers < 1:
        raise ValueError("workers must be positive")
    if max_shards is not None and max_shards < 1:
        raise ValueError("max_shards must be positive")
    for source_name in ("field_ready", "base", "target"):
        if _paths_overlap(roots["output"], roots[source_name]):
            raise ValueError(f"output root overlaps immutable {source_name} root")

    base_manifest_path = roots["base"] / "sample_manifest.csv"
    target_manifest_path = roots["target"] / "sample_manifest.csv"
    base_manifest_sha256 = sha256_file(base_manifest_path)
    target_manifest_sha256 = sha256_file(target_manifest_path)
    if base_manifest_sha256 != contract.expected_base_manifest_sha256:
        raise ValueError("base sample_manifest.csv differs from the frozen contract")
    if target_manifest_sha256 != contract.expected_target_manifest_sha256:
        raise ValueError("r9 target sample_manifest.csv differs from the frozen contract")

    rows = load_joined_authority(base_manifest_path, target_manifest_path)
    _validate_authority(rows, contract)
    shard_plan = _plan_shards(rows)
    if len(shard_plan) != contract.expected_shards:
        raise ValueError(
            f"base authority exposes {len(shard_plan)} shards, expected {contract.expected_shards}"
        )
    selected_plan = shard_plan[:max_shards] if max_shards is not None else shard_plan
    selected_rows = [row for _, shard_rows in selected_plan for row in shard_rows]

    contract_root = registry_path.resolve().parent
    contract_hashes = _tree_hashes(contract_root)
    implementation_hashes = _tree_hashes(Path(__file__).resolve().parent, suffix=".py")
    fingerprint_payload = {
        "schema": "grud_h1_baseline_build_fingerprint_v1",
        "dataset_id": contract.dataset_id,
        "base_sample_manifest_sha256": base_manifest_sha256,
        "target_sample_manifest_sha256": target_manifest_sha256,
        "contract_sha256": contract_hashes,
        "implementation_sha256": implementation_hashes,
        "max_history_hours": contract.max_history_hours,
        "h1_template_count": contract.h1_template_count,
        "selected_shards": [key for key, _ in selected_plan],
        "selected_samples": len(selected_rows),
    }
    fingerprint = _payload_hash(fingerprint_payload)

    state_path = roots["output"] / "build_state.json"
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError("--resume requires an existing build_state.json")
        state = _read_json(state_path)
        if state.get("fingerprint") != fingerprint:
            raise ValueError("resume rejected because the H1 build fingerprint changed")
        _verify_completed_shards(roots["output"], state)
    else:
        if roots["output"].exists() and any(roots["output"].iterdir()):
            raise FileExistsError("output root is non-empty; use a new root or --resume")
        roots["output"].mkdir(parents=True, exist_ok=True)
        _copy_contract_bundle(contract_root, roots["output"] / "contracts")
        registry = EventRegistry.load(registry_path)
        templates = [
            row for row in registry.expanded_contract() if "H1" in row["input_resolutions"]
        ]
        if len(templates) != contract.h1_template_count:
            raise ValueError("H1 template count differs from the build contract")
        _atomic_write_json(
            roots["output"] / "h1_event_templates.json",
            {
                "schema": "grud_h1_channel_registry_v1",
                "registry_version": registry.version,
                "tuple_order": [
                    "field_id", "operator_id", "condition_id", "value", "block_id"
                ],
                "channels": [
                    {"channel_id": index, **template}
                    for index, template in enumerate(templates)
                ],
            },
        )
        state = {
            "schema": "grud_h1_baseline_build_state_v1",
            "dataset_id": contract.dataset_id,
            "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "status": "PLANNED",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "planned_shards": len(selected_plan),
            "selected_samples": len(selected_rows),
            "completed_shards": [],
            "shards": {},
            "built_samples": 0,
            "built_by_split": {split: 0 for split in SPLITS},
            "event_count": 0,
            "h1_block_count": 0,
        }
        _atomic_write_json(state_path, state)

    if state.get("status") == "SUCCEEDED":
        return _read_json(roots["output"] / "dataset_manifest.json")

    completed = set(state.get("completed_shards") or ())
    pending = [(key, shard_rows) for key, shard_rows in selected_plan if key not in completed]
    state["status"] = "RUNNING"
    state["updated_at"] = utc_now()
    state.pop("last_error", None)
    _atomic_write_json(state_path, state)
    start = time.monotonic()

    try:
        if workers == 1:
            for shard_key, shard_rows in pending:
                result = _build_shard(
                    shard_key, shard_rows, roots["field_ready"], roots["output"],
                    registry_path.resolve(), contract.max_history_hours,
                )
                _commit_shard_state(state, state_path, result)
                _print_progress(state, start, shard_key)
        else:
            _build_parallel(
                pending=pending,
                workers=workers,
                roots=roots,
                registry_path=registry_path.resolve(),
                max_history_hours=contract.max_history_hours,
                state=state,
                state_path=state_path,
                start=start,
            )

        _consolidate_sample_manifest(roots["output"], selected_plan, state)
        manifest = _finalize_manifest(
            output_root=roots["output"], state=state, contract=contract,
            fingerprint_payload=fingerprint_payload,
            source_manifest_hashes={"base": base_manifest_sha256, "target": target_manifest_sha256},
            full_build=max_shards is None,
        )
        state["status"] = "SUCCEEDED"
        state["updated_at"] = utc_now()
        _atomic_write_json(state_path, state)
        _atomic_write_json(roots["output"] / "dataset_manifest.json", manifest)
        _atomic_write_text(roots["output"] / "SUCCEEDED", f"{contract.dataset_id}\n")
        print(
            "GRUD_H1_SAMPLE_BUILD_SUCCEEDED "
            f"samples={state['built_samples']} shards={len(state['completed_shards'])} "
            f"output={roots['output']}", flush=True,
        )
        return manifest
    except BaseException as exc:
        state["status"] = "FAILED"
        state["updated_at"] = utc_now()
        state["last_error"] = {"type": type(exc).__name__, "message": str(exc)}
        _atomic_write_json(state_path, state)
        raise


def _build_parallel(
    *,
    pending: list[tuple[str, list[AuthorityRow]]],
    workers: int,
    roots: dict[str, Path],
    registry_path: Path,
    max_history_hours: int,
    state: dict[str, Any],
    state_path: Path,
    start: float,
) -> None:
    executor = ProcessPoolExecutor(max_workers=workers)
    futures: dict[Future[dict[str, Any]], str] = {}
    try:
        for shard_key, shard_rows in pending:
            future = executor.submit(
                _build_shard, shard_key, shard_rows, roots["field_ready"], roots["output"],
                registry_path, max_history_hours,
            )
            futures[future] = shard_key
        for future in as_completed(futures):
            result = future.result()
            _commit_shard_state(state, state_path, result)
            _print_progress(state, start, str(result["shard_key"]))
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


def load_joined_authority(
    base_manifest_path: Path, target_manifest_path: Path,
) -> list[AuthorityRow]:
    base_rows = _read_csv_exact(base_manifest_path, BASE_MANIFEST_COLUMNS)
    target_rows = _read_csv_exact(target_manifest_path, TARGET_MANIFEST_COLUMNS)
    if len(base_rows) != len(target_rows):
        raise ValueError("base and r9 target manifests have different row counts")
    output: list[AuthorityRow] = []
    identity_fields = (
        "sample_id", "subject_id", "hadm_id", "stay_id", "prediction_hour", "split",
    )
    for index, (base, target) in enumerate(zip(base_rows, target_rows)):
        if any(base[field] != target[field] for field in identity_fields):
            raise ValueError(f"base/r9 identity or order drift at manifest row {index + 2}")
        if base["content_hash"] != target["base_content_hash"]:
            raise ValueError(f"base content hash drift at manifest row {index + 2}")
        prediction_hour = int(base["prediction_hour"])
        if not 1 <= prediction_hour <= 312:
            raise ValueError(f"anchor outside H1 contract at manifest row {index + 2}")
        row = AuthorityRow(
            sample_id=base["sample_id"], subject_id=base["subject_id"],
            hadm_id=base["hadm_id"], stay_id=base["stay_id"],
            prediction_hour=prediction_hour, split=base["split"],
            base_content_hash=base["content_hash"], base_shard_key=base["shard_key"],
            target_content_hash=target["target_content_hash"],
            target_shard_key=target["target_shard_key"],
            target_line_index=int(target["target_line_index"]),
        )
        expected_id = f"hadm_{row.hadm_id}_stay_{row.stay_id}_h{row.prediction_hour}"
        if row.sample_id != expected_id:
            raise ValueError(f"sample_id differs from primary key: {row.sample_id}")
        output.append(row)
    return output


def _validate_authority(rows: list[AuthorityRow], contract: BuildContract) -> None:
    if len(rows) != contract.expected_samples:
        raise ValueError(f"authority exposes {len(rows)} samples, expected {contract.expected_samples}")
    if len({row.sample_id for row in rows}) != len(rows):
        raise ValueError("sample authority contains duplicate sample_id values")
    split_counts = dict(Counter(row.split for row in rows))
    if split_counts != contract.expected_split_counts:
        raise ValueError(f"split counts differ from the frozen contract: {split_counts}")
    subject_splits: dict[str, str] = {}
    for row in rows:
        if row.split not in SPLITS:
            raise ValueError(f"unknown split: {row.split}")
        previous = subject_splits.setdefault(row.subject_id, row.split)
        if previous != row.split:
            raise ValueError(f"subject appears in multiple splits: {row.subject_id}")


def _plan_shards(rows: Iterable[AuthorityRow]) -> list[tuple[str, list[AuthorityRow]]]:
    grouped: OrderedDict[str, list[AuthorityRow]] = OrderedDict()
    for row in rows:
        grouped.setdefault(row.base_shard_key, []).append(row)
    return list(grouped.items())


def _build_shard(
    shard_key: str,
    rows: list[AuthorityRow],
    field_ready_root: Path,
    output_root: Path,
    registry_path: Path,
    max_history_hours: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"empty shard plan: {shard_key}")
    split = rows[0].split
    if any(row.split != split or row.base_shard_key != shard_key for row in rows):
        raise ValueError(f"mixed split or shard authority in {shard_key}")
    registry = EventRegistry.load(registry_path)
    builder = GRUDH1SampleBuilder(registry, max_history_hours=max_history_hours)
    sample_path = output_root / "h1_shards" / split / f"{shard_key}.jsonl.gz"
    manifest_path = output_root / "manifests" / split / f"{shard_key}.csv"
    writer = AtomicGzipJsonlWriter(sample_path)
    manifest_buffer = io.StringIO(newline="")
    manifest_writer = csv.DictWriter(manifest_buffer, fieldnames=OUTPUT_MANIFEST_COLUMNS)
    manifest_writer.writeheader()
    cached_key: tuple[str, str] | None = None
    cached_stay: Any = None
    event_count = 0
    block_count = 0
    try:
        for line_index, row in enumerate(rows):
            stay_key = (row.hadm_id, row.stay_id)
            if cached_key != stay_key:
                cached_key = stay_key
                cached_stay = load_stay(
                    field_ready_root / f"hadm_{row.hadm_id}_stay_{row.stay_id}"
                )
            if (
                cached_stay.subject_id != row.subject_id
                or cached_stay.hadm_id != row.hadm_id
                or cached_stay.stay_id != row.stay_id
            ):
                raise ValueError(f"field-ready identity differs for {row.sample_id}")
            sample = builder.build(
                cached_stay, prediction_hour=row.prediction_hour, split=row.split,
                base_content_hash=row.base_content_hash,
                target_content_hash=row.target_content_hash,
                target_shard_key=row.target_shard_key,
                target_line_index=row.target_line_index,
            )
            if sample["sample_id"] != row.sample_id:
                raise ValueError(f"H1 builder changed sample identity: {row.sample_id}")
            writer.write(sample)
            event_count += len(sample["input_events"])
            block_count += int(sample["input_geometry"]["block_count"])
            manifest_writer.writerow({
                "sample_id": row.sample_id, "subject_id": row.subject_id,
                "hadm_id": row.hadm_id, "stay_id": row.stay_id,
                "prediction_hour": row.prediction_hour, "split": row.split,
                "base_content_hash": row.base_content_hash,
                "target_content_hash": row.target_content_hash,
                "h1_content_hash": sample["content_hash"],
                "h1_shard_key": shard_key, "h1_line_index": line_index,
                "target_shard_key": row.target_shard_key,
                "target_line_index": row.target_line_index,
            })
        writer.commit()
        _atomic_write_text(manifest_path, manifest_buffer.getvalue())
    except BaseException:
        writer.abort()
        raise
    return {
        "shard_key": shard_key, "split": split, "samples": len(rows),
        "events": event_count, "h1_blocks": block_count,
        "first_sample_id": rows[0].sample_id, "last_sample_id": rows[-1].sample_id,
        "sample_path": str(sample_path.relative_to(output_root)),
        "sample_sha256": sha256_file(sample_path),
        "manifest_path": str(manifest_path.relative_to(output_root)),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _commit_shard_state(
    state: dict[str, Any], state_path: Path, result: dict[str, Any],
) -> None:
    shard_key = str(result["shard_key"])
    if shard_key in set(state.get("completed_shards") or ()):
        raise ValueError(f"shard committed twice: {shard_key}")
    state.setdefault("shards", {})[shard_key] = result
    state.setdefault("completed_shards", []).append(shard_key)
    state["built_samples"] = int(state.get("built_samples", 0)) + int(result["samples"])
    state["event_count"] = int(state.get("event_count", 0)) + int(result["events"])
    state["h1_block_count"] = int(state.get("h1_block_count", 0)) + int(result["h1_blocks"])
    split = str(result["split"])
    state.setdefault("built_by_split", {})[split] = (
        int(state.get("built_by_split", {}).get(split, 0)) + int(result["samples"])
    )
    state["updated_at"] = utc_now()
    _atomic_write_json(state_path, state)


def _verify_completed_shards(output_root: Path, state: dict[str, Any]) -> None:
    completed = state.get("completed_shards") or []
    if len(completed) != len(set(completed)):
        raise ValueError("resume state contains duplicate completed shards")
    for shard_key in completed:
        metadata = (state.get("shards") or {}).get(shard_key)
        if not isinstance(metadata, dict):
            raise ValueError(f"resume state lacks metadata for {shard_key}")
        for path_key, hash_key in (
            ("sample_path", "sample_sha256"), ("manifest_path", "manifest_sha256"),
        ):
            path = output_root / str(metadata[path_key])
            if not path.is_file() or sha256_file(path) != metadata[hash_key]:
                raise ValueError(f"completed shard changed before resume: {path}")


def _consolidate_sample_manifest(
    output_root: Path,
    shard_plan: list[tuple[str, list[AuthorityRow]]],
    state: dict[str, Any],
) -> None:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(OUTPUT_MANIFEST_COLUMNS)
    observed = 0
    for shard_key, _ in shard_plan:
        metadata = state["shards"][shard_key]
        path = output_root / metadata["manifest_path"]
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            if tuple(next(reader)) != OUTPUT_MANIFEST_COLUMNS:
                raise ValueError(f"per-shard manifest header drift: {path}")
            for row in reader:
                writer.writerow(row)
                observed += 1
    if observed != int(state["built_samples"]):
        raise ValueError("consolidated sample manifest count differs from build state")
    _atomic_write_text(output_root / "sample_manifest.csv", output.getvalue())


def _finalize_manifest(
    *,
    output_root: Path,
    state: dict[str, Any],
    contract: BuildContract,
    fingerprint_payload: dict[str, Any],
    source_manifest_hashes: dict[str, str],
    full_build: bool,
) -> dict[str, Any]:
    if full_build:
        if int(state["built_samples"]) != contract.expected_samples:
            raise ValueError("full H1 build does not contain every frozen anchor")
        if state["built_by_split"] != contract.expected_split_counts:
            raise ValueError("full H1 build split counts differ from the frozen split")
    sample_manifest_path = output_root / "sample_manifest.csv"
    return {
        "schema": "grud_h1_baseline_dataset_manifest_v1",
        "dataset_id": contract.dataset_id,
        "status": "SUCCEEDED",
        "created_at": state["created_at"],
        "completed_at": utc_now(),
        "full_authority_build": full_build,
        "sample_unit": "one ICU stay and one frozen prediction anchor",
        "input_contract": {
            "resolution": "H1", "history": "ICU hour 0 through the prediction anchor",
            "max_history_hours": contract.max_history_hours, "fields": 37,
            "registered_channels": contract.h1_template_count,
            "tuple_order": [
                "field_id", "operator_id", "condition_id", "value", "block_id"
            ],
            "normalization": "none in persisted sample",
            "availability_gate": "available_hour <= prediction_hour",
        },
        "target_contract": {
            "storage": "referenced, not duplicated",
            "dataset_id": "multires_event_m4_target_v2_c4_full_20260714_r9",
            "blocks": 6, "resolution": "M4", "field_processes": 29,
            "stochastic_factors": 414,
        },
        "authority": {
            "same_sample_id_order_as_base": True, "patient_split_reused": True,
            "base_sample_manifest_sha256": source_manifest_hashes["base"],
            "target_sample_manifest_sha256": source_manifest_hashes["target"],
        },
        "fingerprint": state["fingerprint"],
        "fingerprint_payload": fingerprint_payload,
        "counts": {
            "samples": state["built_samples"], "by_split": state["built_by_split"],
            "shards": len(state["completed_shards"]),
            "h1_blocks": state["h1_block_count"], "events": state["event_count"],
        },
        "files": {
            "sample_manifest": {
                "path": "sample_manifest.csv", "sha256": sha256_file(sample_manifest_path),
            },
            "h1_event_templates": {
                "path": "h1_event_templates.json",
                "sha256": sha256_file(output_root / "h1_event_templates.json"),
            },
            "build_state": "build_state.json", "contracts": "contracts/",
            "h1_shards": {
                key: state["shards"][key] for key in fingerprint_payload["selected_shards"]
            },
        },
    }


def _read_csv_exact(path: Path, expected_header: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ValueError(f"manifest header drift: {path}")
        return list(reader)


def _copy_contract_bundle(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"contract target already exists: {target}")
    shutil.copytree(source, target)
    if _tree_hashes(source) != _tree_hashes(target):
        raise ValueError("copied contract bundle hash mismatch")


def _tree_hashes(root: Path, *, suffix: str | None = None) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and (suffix is None or path.suffix == suffix)
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial-{os.getpid()}")
    partial.write_text(text, encoding="utf-8", newline="\n")
    os.replace(partial, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _paths_overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _print_progress(state: dict[str, Any], start: float, shard_key: str) -> None:
    elapsed = max(time.monotonic() - start, 1e-9)
    print(
        "GRUD_H1_SAMPLE_PROGRESS "
        f"shard={shard_key} completed={len(state['completed_shards'])}/{state['planned_shards']} "
        f"samples={state['built_samples']}/{state['selected_samples']} "
        f"rate={state['built_samples'] / elapsed:.2f}_samples_per_second",
        flush=True,
    )


__all__ = [
    "AuthorityRow", "BuildContract", "OUTPUT_MANIFEST_COLUMNS", "SPLITS",
    "build_grud_h1_dataset", "load_joined_authority", "sha256_file",
]
