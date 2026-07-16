from __future__ import annotations

"""Read-only historical Dataset identity and materialization evidence for v8.

This module deliberately has no training entrypoint, mode selector, promotion
logic, torchrun command, or Relation V2 authorization.  It retains only the
old Dataset-byte checks needed to audit historical evidence in local tests.
"""

import gzip
import io
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
KAGGLE_SCRIPT_ROOT = Path(__file__).resolve().parent
for import_root in (SRC_ROOT, KAGGLE_SCRIPT_ROOT):
    if str(import_root) not in os.sys.path:
        os.sys.path.insert(0, str(import_root))

import run_multires_event_v1 as v1_route  # noqa: E402
from trauma_predict.training.observability import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    utc_now,
)


HISTORICAL_ROUTE_ID = "multires_event_v2_m4_v8_modes"
HOSTED_ROUTE_STATUS = "historical_evidence_only"
BASE_DATASET_REF = "vanilaaaa/trauma-predict-multires-event-v1-c4-20260712"
TARGET_DATASET_REF = "vanilaaaa/trauma-predict-multires-event-v2-c4-r8-20260713"
EXPECTED_COUNTS = {
    "samples": 50350,
    "train": 37734,
    "val": 6309,
    "test": 6307,
    "shards": 52,
}
EXPECTED_SHARD_COUNTS = {"train": 38, "val": 7, "test": 7}
TARGET_CONTRACT_FILES = (
    "target_process_registry_v2.json",
    "target_emission_registry_v2.json",
    "target_projection_registry_v2.json",
    "field_category_matrix_v1.csv",
    "field_relation_edges_v1.csv",
    "event_element_extension_v2.json",
    "target_sidecar_schema_v2.json",
)
TARGET_AUTHORITY = {
    "dataset_id": "multires_event_m4_target_v2_c4_full_20260713_r8",
    "manifest_sha256": "fb8748a5d396c5342be143032096acef03af2345bdd80e53dc82f69a7875b8b6",
    "sample_manifest_sha256": "96ce73f2cfb4a2a8af0bd21cbbab9634bd02268d03e7cda68ac4f21229596a4e",
    "contract_bundle_hash": "10e9ed6c2fb94610fa61edc5061b8465e967ef6c222f22455877da583420cd10",
    "process_contract_sha256": "3f90bec35d6473a0e9dc69f3654d1b55eaf1c9d3f9850078df1361e84b2cd7db",
    "emission_contract_sha256": "e926e1a3e6e3e71039a26548ca8d3f35bf2eee5725be3195992d4d47f715e96c",
    "projection_contract_sha256": "7efdf7d3c0415e6aa26d99411f5df66907b5ff74b30f6880e72de72fe4c3d34b",
    "relation_contract_sha256": "65286cd9fb7e1038270de39ea17daafffb160cf9c5ab7bb3beb2556a9aa8eea0",
    "sidecar_schema_sha256": "a2e4018d9dac3c4245ad13852036e6cb3ff9014eea9dc996fa9b0b6235251e8f",
    "process_contract_version": "2026-07-13-r8",
    "emission_contract_version": "2026-07-13-r8",
    "projection_contract_version": "2026-07-13-r8",
}
KAGGLE_INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
TARGET_DOWNLOAD_ROOT = Path(
    os.environ.get(
        "TRAUMA_PREDICT_V2_DOWNLOAD_ROOT",
        "/kaggle/working/kaggle-dataset-multires-event-v2-target",
    )
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def run_to_log(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("historical Dataset evidence cannot execute external commands")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return payload


def _file_hash_or_empty(path: Path) -> str:
    return sha256_file(path) if path.is_file() else ""


def _target_counts(manifest: Mapping[str, Any]) -> dict[str, int]:
    counts = manifest.get("counts") or {}
    by_split = counts.get("by_split") or {}
    return {
        "samples": int(counts.get("samples", -1)),
        "train": int(by_split.get("train", -1)),
        "val": int(by_split.get("val", -1)),
        "test": int(by_split.get("test", -1)),
        "shards": int(counts.get("shards", -1)),
    }


def _matches_target_authority(root: Path, manifest: Mapping[str, Any]) -> bool:
    hashes = manifest.get("contract_hashes") or {}
    return (
        manifest.get("dataset_id") == TARGET_AUTHORITY["dataset_id"]
        and _target_counts(manifest) == EXPECTED_COUNTS
        and sha256_file(root / "dataset_manifest.json")
        == TARGET_AUTHORITY["manifest_sha256"]
        and _file_hash_or_empty(root / "sample_manifest.csv")
        == TARGET_AUTHORITY["sample_manifest_sha256"]
        and manifest.get("contract_bundle_hash")
        == TARGET_AUTHORITY["contract_bundle_hash"]
        and hashes.get("process") == TARGET_AUTHORITY["process_contract_sha256"]
        and hashes.get("emission") == TARGET_AUTHORITY["emission_contract_sha256"]
        and hashes.get("projection")
        == TARGET_AUTHORITY["projection_contract_sha256"]
        and hashes.get("relation") == TARGET_AUTHORITY["relation_contract_sha256"]
        and hashes.get("sidecar_schema")
        == TARGET_AUTHORITY["sidecar_schema_sha256"]
    )


def find_exact_target_dataset(input_root: Path) -> Path:
    if not input_root.is_dir():
        raise FileNotFoundError(f"dataset search root is absent: {input_root}")
    exact: list[Path] = []
    inspected: list[dict[str, Any]] = []
    for manifest_path in sorted(input_root.rglob("dataset_manifest.json")):
        try:
            manifest = _read_json(manifest_path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        inspected.append(
            {
                "root": str(manifest_path.parent),
                "dataset_id": manifest.get("dataset_id"),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
        if _matches_target_authority(manifest_path.parent, manifest):
            exact.append(manifest_path.parent.resolve())
    unique = sorted(set(exact))
    if len(unique) > 1:
        raise RuntimeError(
            f"multiple exact V2 target sidecar datasets are attached; retain one: {unique}"
        )
    if not unique:
        raise FileNotFoundError(
            f"no exact V2 target sidecar dataset found; inspected={inspected}"
        )
    return unique[0]


def resolved_target_dataset_ref() -> str:
    override = os.environ.get("TRAUMA_PREDICT_V2_DATASET_REF")
    if override is not None and override != TARGET_DATASET_REF:
        raise ValueError(
            "TRAUMA_PREDICT_V2_DATASET_REF must exactly equal the frozen source ref "
            f"{TARGET_DATASET_REF!r}; got {override!r}"
        )
    return TARGET_DATASET_REF


def download_exact_dataset(**_kwargs: Any) -> Path:
    raise RuntimeError("historical Dataset evidence cannot download artifacts")


def explicit_or_download_target_root(log_dir: Path) -> Path:
    dataset_ref = resolved_target_dataset_ref()
    explicit = os.environ.get("TRAUMA_PREDICT_V2_TARGET_ROOT")
    if explicit:
        root = Path(explicit).resolve()
        if not root.is_dir() or not _matches_target_authority(
            root, _read_json(root / "dataset_manifest.json")
        ):
            raise ValueError(
                f"TRAUMA_PREDICT_V2_TARGET_ROOT is not the frozen target sidecar: {root}"
            )
        return root
    return download_exact_dataset(
        dataset_ref=dataset_ref,
        download_root=TARGET_DOWNLOAD_ROOT,
        finder=find_exact_target_dataset,
        log_path=log_dir / "target_dataset_download.log",
        label="TARGET_DATASET_DOWNLOAD",
    )


def preflight_dataset_download_access(log_dir: Path) -> None:
    if os.environ.get("TRAUMA_PREDICT_DATA_ROOT") or os.environ.get(
        "TRAUMA_PREDICT_V2_TARGET_ROOT"
    ):
        raise RuntimeError(
            "formal zero-Input hosting forbids explicit data roots; both frozen "
            "Datasets must be downloaded by the Notebook"
        )
    v1_route.configure_kaggle_credentials()
    for label, dataset_ref in (
        ("BASE_DATASET_ACCESS", BASE_DATASET_REF),
        ("TARGET_DATASET_ACCESS", resolved_target_dataset_ref()),
    ):
        run_to_log(
            ["kaggle", "datasets", "files", "-d", dataset_ref, "--page-size", "1"],
            log_dir / f"{label.lower()}.log",
            env=os.environ.copy(),
            label=label,
        )


def prepare_target_root(source_root: Path, destination: Path, log_dir: Path) -> Path:
    if is_prepared_target(destination):
        return destination.resolve()
    if is_prepared_target(source_root):
        return source_root.resolve()
    if not _matches_target_authority(
        source_root,
        _read_json(source_root / "dataset_manifest.json"),
    ):
        raise ValueError("target preparation source is not the frozen V2 authority")

    temporary = destination.with_name(f".{destination.name}.prepare-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for name in (
        "dataset_manifest.json",
        "sample_manifest.csv",
        "subject_split.csv",
        "SUCCEEDED",
    ):
        source = source_root / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, temporary / name)
    _materialize_target_contracts(source_root, temporary / "contracts")
    shard_count = _copy_extracted_target_shards(source_root, temporary)
    if shard_count != EXPECTED_COUNTS["shards"]:
        raise RuntimeError(
            f"V2 target payload materialized {shard_count} shards, expected 52"
        )
    if not is_prepared_target(temporary):
        raise ValueError(
            "prepared V2 target failed exact post-materialization identity checks"
        )
    if destination.exists():
        raise FileExistsError(destination)
    temporary.replace(destination)
    atomic_write_json(
        log_dir / "target_dataset_prepare.json",
        {
            "schema_version": "trauma_predict.multires_event_v2_target_prepare.v1",
            "created_at": utc_now(),
            "source_root": str(source_root),
            "destination": str(destination),
            "contract_layout": "extracted_contract_tree",
            "target_shard_layout": "kaggle_hosted_extracted_target_tree",
            "materialized_target_shards": shard_count,
        },
    )
    return destination.resolve()


def _materialize_target_contracts(source_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in TARGET_CONTRACT_FILES:
        candidates = sorted(path for path in source_root.rglob(name) if path.is_file())
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"historical target contract {name} must have exactly one source"
            )
        shutil.copy2(candidates[0], destination / name)
    _verify_target_contract_files(destination, source_root / "dataset_manifest.json")


def _discover_extracted_target_shards(source_root: Path) -> dict[str, list[Path]]:
    discovered: dict[str, list[Path]] = {split: [] for split in EXPECTED_SHARD_COUNTS}
    candidates = set(source_root.rglob("*.jsonl.gz"))
    candidates.update(source_root.rglob("*.jsonl"))
    for source in sorted(candidates):
        parts = source.relative_to(source_root).parts
        if any(part in {"validation", "manifests", "audit"} for part in parts):
            continue
        split_candidates = [part for part in parts[:-1] if part in discovered]
        split = split_candidates[0] if len(split_candidates) == 1 else None
        if split in discovered and source.name.startswith(f"{split}-"):
            discovered[split].append(source)
    return discovered


def _copy_extracted_target_shards(source_root: Path, destination: Path) -> int:
    discovered = _discover_extracted_target_shards(source_root)
    observed = {split: len(paths) for split, paths in discovered.items()}
    if observed != EXPECTED_SHARD_COUNTS:
        raise FileNotFoundError(
            f"historical target extracted counts are {observed}; "
            f"expected {EXPECTED_SHARD_COUNTS}"
        )
    for split, paths in discovered.items():
        for source in paths:
            target_name = (
                source.name if source.name.endswith(".jsonl.gz") else f"{source.name}.gz"
            )
            target = destination / "target_shards" / split / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == ".jsonl":
                _recompress_target_shard(source, target)
            else:
                shutil.copy2(source, target)
    return sum(observed.values())


def _recompress_target_shard(source: Path, target: Path) -> None:
    with target.open("wb") as raw_output:
        compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0)
        output = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
        try:
            with source.open("r", encoding="utf-8", newline="") as input_handle:
                for line_number, line in enumerate(input_handle, start=1):
                    if not line.endswith("\n"):
                        raise ValueError(
                            f"plain hosted target shard lacks LF at line {line_number}: {source}"
                        )
                    output.write(line)
        finally:
            output.flush()
            output.close()


def is_prepared_target(root: Path) -> bool:
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = _read_json(manifest_path)
        if not _matches_target_authority(root, manifest):
            return False
        _verify_target_contract_files(root / "contracts", manifest_path)
        _verify_target_shard_files(root, manifest)
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
        return False
    return True


def _verify_target_contract_files(contract_root: Path, manifest_path: Path) -> None:
    manifest = _read_json(manifest_path)
    declared = manifest.get("contract_hashes") or {}
    key_by_file = {
        "target_process_registry_v2.json": "process",
        "target_emission_registry_v2.json": "emission",
        "target_projection_registry_v2.json": "projection",
        "field_category_matrix_v1.csv": "category",
        "field_relation_edges_v1.csv": "relation",
        "event_element_extension_v2.json": "element_extension",
        "target_sidecar_schema_v2.json": "sidecar_schema",
    }
    for filename, key in key_by_file.items():
        path = contract_root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != declared.get(key):
            raise ValueError(f"target contract hash mismatch: {filename}")


def _verify_target_shard_files(root: Path, manifest: Mapping[str, Any]) -> None:
    declared = (manifest.get("files") or {}).get("target_shards") or {}
    if not isinstance(declared, Mapping) or len(declared) != EXPECTED_COUNTS["shards"]:
        raise ValueError("target manifest must declare exactly 52 target shard hashes")
    split_counts = {split: 0 for split in EXPECTED_SHARD_COUNTS}
    split_samples = {split: 0 for split in EXPECTED_SHARD_COUNTS}
    for metadata in declared.values():
        if not isinstance(metadata, Mapping):
            raise ValueError("target shard metadata must be a mapping")
        relative = Path(str(metadata.get("path") or ""))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) != 3
            or relative.parts[0] != "target_shards"
        ):
            raise ValueError(f"unsafe declared target shard path: {relative}")
        split = relative.parts[1]
        if split not in split_counts:
            raise ValueError(f"target shard split is invalid: {relative}")
        path = root / relative
        expected_hash = str(metadata.get("sha256") or "")
        if (
            not path.is_file()
            or not SHA256_PATTERN.fullmatch(expected_hash)
            or sha256_file(path) != expected_hash
        ):
            raise ValueError(f"target shard byte hash mismatch: {relative}")
        split_counts[split] += 1
        split_samples[split] += int(metadata.get("samples", -1))
    if split_counts != EXPECTED_SHARD_COUNTS:
        raise ValueError(f"target shard split counts mismatch: {split_counts}")
    if split_samples != {key: EXPECTED_COUNTS[key] for key in split_samples}:
        raise ValueError(f"target shard sample counts mismatch: {split_samples}")
