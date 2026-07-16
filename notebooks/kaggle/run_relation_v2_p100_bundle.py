from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Mapping


MANIFEST_SCHEMA = "trauma_predict.multires_event_v2_relation_v2_p100_bundle.v2"
DATA_INVENTORY_SCHEMA = "trauma_predict.mounted_file_inventory.v3"
SOURCE_INVENTORY_SCHEMA = "trauma_predict.source_release_inventory.v1"
SOURCE_RELEASE_SCHEMA = "trauma_predict.multires_event_v2_source_release.v1"
RUNTIME_WHEELHOUSE_SCHEMA = "trauma_predict.p100_torch_runtime_wheelhouse.v1"
HOSTED_STAGE_SCHEMA = "trauma_predict.multires_event_v2_p100_hosted_stage.v1"
HOSTED_STOP_READINESS_SCHEMA = (
    "trauma_predict.multires_event_v2_hosted_stop_readiness.v1"
)
CHECKPOINT_SCHEMA = "trauma_predict.multires_event_v2_checkpoint.v2"
BEST_CHECKPOINT_SCHEMA = "trauma_predict.multires_event_v2_best_checkpoint.v1"
RUN_NAME = "p100_multires_event_v2_relation_v2"
ROUTE = "multires_event_v2_m4_relation_v2"
EXPECTED_PARAMETERS = 48_728_439
EXPECTED_DATASET_REF = "vanila111/trauma-predict-relation-v2-p100-r9-bundle"
EXPECTED_NOTEBOOK_REF = "vanila111/trauma-predict-relation-v2-p100-r9"
EXPECTED_BASE_DATASET_ID = "multires_event_v1_c4_full_20260712"
EXPECTED_TARGET_DATASET_ID = "multires_event_m4_target_v2_c4_full_20260714_r9"
EXPECTED_NORMALIZATION_SHA256 = (
    "4f54dbeaab4b2becd349d1d8fcaac7b6bdea2567a20874ee7d29338c1f930add"
)
EXPECTED_RELATION_BUNDLE_SHA256 = (
    "0331ec0d552e47790d1dc4f8bae3520062c9e6f5fa62cf62e87c187f6783c033"
)
EXPECTED_TRAINING_STOP_STEPS = (250, 1500, 2750, 4000)
EXPECTED_VALIDATION_ANCHORS = 6309
EXPECTED_RUNTIME_CONTRACT_SHA256 = (
    "aada1dee4ee21e02fd5c81ae97d441c38e72d770eec5398932ee295d08f8f2cc"
)
EXPECTED_RUNTIME_INVENTORY_SHA256 = (
    "8063e83b243589e26c353d335fd5137505bfa90b2d5aa0b1226c15fd810120a1"
)
EXPECTED_RUNTIME_TORCH_VERSION = "2.10.0+cu126"
EXPECTED_RUNTIME_CUDA_VERSION = "12.6"
EXPECTED_RUNTIME_PYTHON_ABI = "cp312"
EXPECTED_RUNTIME_CUDA_ARCH = "sm_60"
EXPECTED_RUNTIME_WHEEL_COUNT = 28
EXPECTED_RUNTIME_WHEEL_BYTES = 3_587_233_664
RUNTIME_ROOT_ENV = "TRAUMA_PREDICT_RUNTIME_SITE_PACKAGES"
RUNTIME_LOCK_ENV = "TRAUMA_PREDICT_RUNTIME_LOCK_SHA256"
STAGE_MANIFEST_NAME = "hosted_stage_manifest.json"

BASE_AUTHORITY = {
    "dataset_id": EXPECTED_BASE_DATASET_ID,
    "dataset_manifest_sha256": (
        "4e7742900907e0e2f774099ba1dd485468210ff3da9ddaef3ec3bf67957000c3"
    ),
    "sample_manifest_sha256": (
        "b3d4305353997320fe310c4df6e15619026db6f229a124b0c9a5e1d89898f05e"
    ),
    "subject_split_sha256": (
        "89deb50c2c6415dff5ce00338a980e25531433e8dee835b004a27d561e7adb6d"
    ),
    "succeeded_sha256": (
        "ac40c796d3bb57d5647be42ae49da20034430e5c2bbfa7458a65422ff64c06c9"
    ),
}
TARGET_AUTHORITY = {
    "dataset_id": EXPECTED_TARGET_DATASET_ID,
    "dataset_manifest_sha256": (
        "6c4e1e300686195fb2c58bfcbd74df6c7cb905d7031985cb7a7624d5c7061f1e"
    ),
    "sample_manifest_sha256": (
        "df5eedcee0abf7d09fea86572db471047bdaa82dc28b14dc8bbf0dac0e32dd0e"
    ),
    "succeeded_sha256": (
        "0c5c7c80eae22fb64c350f90dd3b915c702779d4abf6da362f424bdfbae00cd5"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _safe_relative(value: Any, label: str) -> Path:
    relative = Path(str(value or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a non-empty relative path")
    return relative


def _validate_runtime_dependencies() -> dict[str, str]:
    ranges = {
        "numpy": (1, 3),
        "PyYAML": (6, 7),
        "safetensors": (0, 1),
    }
    versions: dict[str, str] = {}
    for package, (minimum_major, maximum_major) in ranges.items():
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Kaggle image lacks required package: {package}") from exc
        try:
            major = int(version.split(".", 1)[0])
        except ValueError as exc:
            raise RuntimeError(f"cannot parse {package} version: {version}") from exc
        if not minimum_major <= major < maximum_major:
            raise RuntimeError(f"unsupported {package} version: {version}")
        versions[package] = version
    return versions


def _resolve_file(bundle: Path, row: Mapping[str, Any], label: str) -> Path:
    relative = _safe_relative(row.get("path"), f"{label}.path")
    path = bundle / relative
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"missing mounted {label}: {path}")
    expected_size = row.get("size_bytes")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise ValueError(f"mounted {label} byte count mismatch")
    expected = str(row.get("sha256") or "")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"mounted {label} hash mismatch: {observed} != {expected}")
    return path


def _validate_runtime_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _mapping(manifest.get("runtime"), "runtime")
    contract = _mapping(runtime.get("contract"), "runtime.contract")
    if (
        runtime.get("schema") != RUNTIME_WHEELHOUSE_SCHEMA
        or runtime.get("python_abi") != EXPECTED_RUNTIME_PYTHON_ABI
        or runtime.get("torch_version") != EXPECTED_RUNTIME_TORCH_VERSION
        or runtime.get("cuda_version") != EXPECTED_RUNTIME_CUDA_VERSION
        or runtime.get("required_cuda_arch") != EXPECTED_RUNTIME_CUDA_ARCH
        or runtime.get("inventory_sha256")
        != EXPECTED_RUNTIME_INVENTORY_SHA256
        or int(runtime.get("file_count", -1)) != EXPECTED_RUNTIME_WHEEL_COUNT
        or int(runtime.get("total_bytes", -1)) != EXPECTED_RUNTIME_WHEEL_BYTES
        or runtime.get("network_install") is not False
        or runtime.get("pip_requirement")
        != f"torch=={EXPECTED_RUNTIME_TORCH_VERSION}"
        or contract.get("path") != "p100_torch_2_10_cu126_cp312.json"
        or int(contract.get("size_bytes", -1)) != 6144
        or contract.get("sha256") != EXPECTED_RUNTIME_CONTRACT_SHA256
    ):
        raise ValueError("mounted bundle runtime differs from the frozen P100 cu126 lock")
    raw_rows = runtime.get("files")
    if not isinstance(raw_rows, list) or len(raw_rows) != EXPECTED_RUNTIME_WHEEL_COUNT:
        raise ValueError("mounted bundle runtime wheel rows differ from the frozen lock")
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_rows):
        row = _mapping(raw, f"runtime.files[{index}]")
        relative = _safe_relative(row.get("path"), f"runtime.files[{index}].path")
        name = relative.as_posix()
        size = int(row.get("size_bytes", -1))
        digest = str(row.get("sha256") or "")
        if (
            len(relative.parts) != 1
            or not name.endswith(".whl")
            or name in names
            or size < 1
            or not _is_sha256(digest)
        ):
            raise ValueError(f"invalid runtime wheel row: {row!r}")
        names.add(name)
        rows.append({"path": name, "size_bytes": size, "sha256": digest})
    if rows != sorted(rows, key=lambda row: row["path"]):
        raise ValueError("runtime wheel rows are not sorted by filename")
    if sum(int(row["size_bytes"]) for row in rows) != EXPECTED_RUNTIME_WHEEL_BYTES:
        raise ValueError("runtime wheel rows do not match the frozen byte count")
    inventory = {
        "schema": RUNTIME_WHEELHOUSE_SCHEMA,
        "python_abi": EXPECTED_RUNTIME_PYTHON_ABI,
        "torch_version": EXPECTED_RUNTIME_TORCH_VERSION,
        "cuda_version": EXPECTED_RUNTIME_CUDA_VERSION,
        "required_cuda_arch": EXPECTED_RUNTIME_CUDA_ARCH,
        "files": rows,
    }
    if _sha256_payload(inventory) != EXPECTED_RUNTIME_INVENTORY_SHA256:
        raise ValueError("runtime wheel inventory differs from the frozen cu126 lock")
    return {**runtime, "files": rows}


def _validate_isolated_torch_runtime(
    torch: Any,
    manifest: Mapping[str, Any],
) -> str:
    _validate_runtime_manifest(manifest)
    runtime_root_value = os.environ.get(RUNTIME_ROOT_ENV, "").strip()
    if not runtime_root_value:
        raise RuntimeError(f"{RUNTIME_ROOT_ENV} is required")
    runtime_root = Path(runtime_root_value).resolve()
    if not runtime_root.is_dir():
        raise FileNotFoundError(f"isolated cu126 runtime is absent: {runtime_root}")
    torch_file = Path(str(getattr(torch, "__file__", ""))).resolve()
    try:
        torch_file.relative_to(runtime_root)
    except ValueError as exc:
        raise RuntimeError(
            f"PyTorch was not imported from the isolated cu126 runtime: {torch_file}"
        ) from exc
    if os.environ.get(RUNTIME_LOCK_ENV) != EXPECTED_RUNTIME_CONTRACT_SHA256:
        raise RuntimeError("isolated cu126 runtime lock identity is absent or changed")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"P100 cu126 runtime requires Python 3.12, found {sys.version}")
    if str(torch.__version__) != EXPECTED_RUNTIME_TORCH_VERSION:
        raise RuntimeError(f"unexpected PyTorch runtime: {torch.__version__}")
    if str(torch.version.cuda) != EXPECTED_RUNTIME_CUDA_VERSION:
        raise RuntimeError(f"unexpected PyTorch CUDA runtime: {torch.version.cuda}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Relation V2 requires exactly one visible P100; "
            f"found {torch.cuda.device_count()} visible GPUs"
        )
    device_name = torch.cuda.get_device_name(0)
    if "P100" not in device_name.upper():
        raise RuntimeError(f"select Kaggle P100; current device is {device_name!r}")
    if tuple(torch.cuda.get_device_capability(0)) != (6, 0):
        raise RuntimeError("selected GPU is not the frozen Pascal sm_60 device")
    if EXPECTED_RUNTIME_CUDA_ARCH not in set(torch.cuda.get_arch_list()):
        raise RuntimeError("isolated PyTorch wheel does not contain sm_60 kernels")
    probe = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float16, requires_grad=True)
    loss = (probe.square() + 1.0).sum()
    loss.backward()
    torch.cuda.synchronize()
    if probe.grad is None or not bool(torch.isfinite(probe.grad).all().item()):
        raise RuntimeError("P100 cu126 CUDA backward smoke produced a non-finite gradient")
    return str(device_name)


def _find_bundle(explicit: Path | None) -> tuple[Path, dict[str, Any], Path]:
    if explicit is not None:
        candidates = [explicit.resolve() / "run_bundle_manifest.json"]
    else:
        candidates = sorted(Path("/kaggle/input").glob("*/run_bundle_manifest.json"))
    matches: list[tuple[Path, dict[str, Any], Path]] = []
    for path in candidates:
        if path.is_symlink() or not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") == MANIFEST_SCHEMA:
            matches.append((path.parent.resolve(), payload, path.resolve()))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one mounted Relation V2 P100 bundle, found {len(matches)}"
        )
    return matches[0]


def _safe_extract_regular_files(
    archive: Path,
    destination: Path,
    *,
    expected_members: Mapping[str, Mapping[str, Any]],
    label: str,
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with tarfile.open(archive, "r:*") as handle:
        members = handle.getmembers()
        observed_names = {member.name for member in members}
        if len(members) != len(expected_members) or observed_names != set(expected_members):
            raise ValueError(f"{label} members differ from the hash-bound inventory")
        for member in members:
            target = (destination / member.name).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"{label} path escapes destination: {member.name}") from exc
            declared = expected_members[member.name]
            if (
                not member.isfile()
                or member.issym()
                or member.islnk()
                or member.size != int(declared.get("size_bytes", -1))
            ):
                raise ValueError(f"{label} member contract failed: {member.name}")
        handle.extractall(destination, members=members, filter="data")
    for name, row in expected_members.items():
        path = destination / name
        if path.is_symlink() or not path.is_file() or _sha256(path) != row.get("sha256"):
            raise ValueError(f"extracted {label} file/hash failed: {name}")


def _materialize_dataset_view(
    bundle: Path,
    declared: Mapping[str, Any],
    destination: Path,
    packed_root: Path,
    *,
    label: str,
) -> Path:
    inventory_path = _resolve_file(
        bundle,
        _mapping(declared.get("inventory"), f"{label}.inventory"),
        f"{label} inventory",
    )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("schema") != DATA_INVENTORY_SCHEMA:
        raise ValueError(f"{label} inventory schema mismatch")
    files = inventory.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{label} inventory must declare files")
    destination.mkdir(parents=True, exist_ok=False)
    packed_members: dict[str, dict[str, Any]] = {}
    packed_uncompressed_bytes = 0
    for index, raw_row in enumerate(files):
        row = _mapping(raw_row, f"{label}.inventory.files[{index}]")
        if row.get("storage") == "packed":
            member = _safe_relative(row.get("archive_member"), "archive_member").as_posix()
            if member in packed_members:
                raise ValueError(f"{label} inventory contains duplicate packed members")
            packed_members[member] = row
            packed_uncompressed_bytes += int(row.get("size_bytes", -1))
    packed_payload = inventory.get("packed_payload")
    if packed_payload is not None:
        packed_row = _mapping(packed_payload, f"{label}.inventory.packed_payload")
        archive = _resolve_file(bundle, packed_row, f"{label} small payload pack")
        if (
            int(packed_row.get("file_count", -1)) != len(packed_members)
            or int(packed_row.get("uncompressed_bytes", -1))
            != packed_uncompressed_bytes
            or int(packed_row.get("archive_bytes", -1)) != archive.stat().st_size
        ):
            raise ValueError(f"{label} small payload pack summary mismatch")
        _safe_extract_regular_files(
            archive,
            packed_root,
            expected_members=packed_members,
            label=f"{label} small payload pack",
        )
    elif packed_members or int(inventory.get("packed_file_count", -1)) != 0:
        raise ValueError(f"{label} inventory lacks its declared small payload pack")

    seen_destinations: set[str] = set()
    seen_payloads: set[str] = set()
    mounted_rows = 0
    for index, raw_row in enumerate(files):
        row = _mapping(raw_row, f"{label}.inventory.files[{index}]")
        destination_relative = _safe_relative(row.get("destination"), "destination")
        storage = row.get("storage")
        if storage == "mounted":
            payload_relative = _safe_relative(row.get("mounted_path"), "mounted_path")
            source = bundle / payload_relative
            payload_key = f"mounted:{payload_relative.as_posix()}"
            mounted_rows += 1
        elif storage == "packed":
            archive_member = _safe_relative(row.get("archive_member"), "archive_member")
            source = packed_root / archive_member
            payload_key = f"packed:{archive_member.as_posix()}"
        else:
            raise ValueError(f"{label} inventory storage must be mounted or packed")
        destination_key = destination_relative.as_posix()
        if payload_key in seen_payloads or destination_key in seen_destinations:
            raise ValueError(f"{label} inventory contains duplicate paths")
        seen_payloads.add(payload_key)
        seen_destinations.add(destination_key)
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_size != int(row.get("size_bytes", -1))
            or _sha256(source) != row.get("sha256")
        ):
            raise ValueError(f"mounted {label} payload identity failed: {source}")
        target = destination / destination_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    if (
        int(inventory.get("file_count", -1)) != len(files)
        or int(inventory.get("packed_file_count", -1)) != len(packed_members)
        or int(inventory.get("direct_mounted_file_count", -1)) != mounted_rows
    ):
        raise ValueError(f"{label} inventory count mismatch")
    return destination


def _validate_dataset_identity(
    root: Path,
    declared: Mapping[str, Any],
    authority: Mapping[str, str],
    *,
    label: str,
) -> None:
    paths = {
        "dataset_manifest_sha256": root / "dataset_manifest.json",
        "sample_manifest_sha256": root / "sample_manifest.csv",
        "succeeded_sha256": root / "SUCCEEDED",
    }
    if "subject_split_sha256" in authority:
        paths["subject_split_sha256"] = root / "subject_split.csv"
    for key, path in paths.items():
        # The verified runtime view deliberately uses read-only symlinks to the
        # mounted bundle so 600+ MiB of immutable payload is never duplicated.
        # `_materialize_dataset_view` already validates every source byte and
        # creates every link from the hash-bound inventory.
        if not path.is_file():
            raise FileNotFoundError(f"mounted {label} is incomplete: {path}")
        observed = _sha256(path)
        if observed != authority[key] or observed != declared.get(key):
            raise ValueError(f"mounted {label} {key} mismatch")
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "SUCCEEDED"
        or manifest.get("dataset_id") != authority["dataset_id"]
        or declared.get("dataset_id") != authority["dataset_id"]
    ):
        raise ValueError(f"mounted {label} identity/status mismatch")


def _extract_source(bundle: Path, manifest: Mapping[str, Any], scratch: Path) -> Path:
    source = _mapping(manifest.get("source"), "source")
    archive = _resolve_file(
        bundle,
        _mapping(source.get("archive"), "source.archive"),
        "source archive",
    )
    inventory_path = _resolve_file(
        bundle,
        _mapping(source.get("inventory"), "source.inventory"),
        "source inventory",
    )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    files = inventory.get("files")
    if (
        inventory.get("schema") != SOURCE_INVENTORY_SCHEMA
        or inventory.get("archive_root") != "Trauma-Predict"
        or not isinstance(files, list)
        or not files
        or int(inventory.get("file_count", -1)) != len(files)
    ):
        raise ValueError("source release inventory contract failed")
    if any(
        inventory.get(key) != source.get(key)
        for key in ("git_commit", "git_head_tree", "source_tree_sha256")
    ):
        raise ValueError("source release identity differs between manifest and inventory")
    expected_members: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(files):
        row = _mapping(raw_row, f"source.inventory.files[{index}]")
        name = _safe_relative(row.get("path"), "source member").as_posix()
        if name in expected_members or not _is_sha256(row.get("sha256")):
            raise ValueError("source inventory contains a duplicate or invalid hash")
        expected_members[name] = row
    source_parent = scratch / "source"
    _safe_extract_regular_files(
        archive,
        source_parent,
        expected_members=expected_members,
        label="source archive",
    )
    repo_root = source_parent / "Trauma-Predict"
    release_path = repo_root / "SOURCE_RELEASE.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if (
        release.get("schema_version") != SOURCE_RELEASE_SCHEMA
        or release.get("git_commit") != source.get("git_commit")
        or release.get("git_head_tree") != source.get("git_head_tree")
        or release.get("source_tree_sha256") != source.get("source_tree_sha256")
    ):
        raise ValueError("SOURCE_RELEASE.json differs from the bundle source identity")
    required = (
        "notebooks/kaggle/train_relation_v2_p100.py",
        "configs/train/p100_multires_event_v2_relation_v2.yaml",
        "src/trauma_predict/training/multires_event_v2.py",
    )
    for relative in required:
        path = repo_root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"source release lacks active file: {relative}")
    return repo_root


def _checkpoint_steps(run_dir: Path) -> list[int]:
    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.is_dir():
        return []
    if any(checkpoint_root.glob(".checkpoint-*.partial")):
        raise ValueError("prior output contains an incomplete checkpoint partial")
    steps: list[int] = []
    for path in sorted(checkpoint_root.glob("checkpoint-*")):
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"invalid checkpoint entry: {path}")
        try:
            steps.append(int(path.name.removeprefix("checkpoint-")))
        except ValueError as exc:
            raise ValueError(f"invalid checkpoint directory name: {path}") from exc
    return steps


def _validate_checkpoint(
    checkpoint: Path,
    *,
    expected_step: int,
    expected_identity_hashes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_path = checkpoint / "checkpoint_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest is absent: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_files = {
        "model.pt",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
        "trainer_state.json",
        "identity_hashes.json",
        "rng-rank-0000.pt",
        "sampler-rank-0000.pt",
    }
    files = manifest.get("files")
    hashes = manifest.get("sha256")
    if (
        manifest.get("schema_version") != CHECKPOINT_SCHEMA
        or int(manifest.get("global_step", -1)) != expected_step
        or int(manifest.get("world_size", -1)) != 1
        or not isinstance(files, list)
        or set(str(name) for name in files) != expected_files
        or not isinstance(hashes, dict)
        or set(str(name) for name in hashes) != expected_files
    ):
        raise ValueError(f"checkpoint manifest contract failed: {checkpoint}")
    if expected_identity_hashes is not None and manifest.get("identity_hashes") != dict(
        expected_identity_hashes
    ):
        raise ValueError("checkpoint identity hashes differ from the run identity")
    for name in sorted(expected_files):
        path = checkpoint / name
        digest = str(hashes[name])
        if (
            Path(name).name != name
            or path.is_symlink()
            or not path.is_file()
            or not _is_sha256(digest)
            or _sha256(path) != digest
        ):
            raise ValueError(f"checkpoint file/hash failed: {path}")
    return manifest


def _validate_best_checkpoint(
    run_dir: Path,
    *,
    identity_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    pointer_path = run_dir / "best_checkpoint.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    model_path = run_dir / "best_checkpoint/model.pt"
    identity_path = run_dir / "best_checkpoint/identity_hashes.json"
    if (
        pointer.get("schema_version") != BEST_CHECKPOINT_SCHEMA
        or pointer.get("path") != "best_checkpoint"
        or not isinstance(pointer.get("step"), int)
        or int(pointer["step"]) < 1
        or pointer.get("identity_hashes") != dict(identity_hashes)
        or json.loads(identity_path.read_text(encoding="utf-8")) != dict(identity_hashes)
        or model_path.is_symlink()
        or not model_path.is_file()
        or _sha256(model_path) != pointer.get("model_sha256")
    ):
        raise ValueError("best checkpoint identity/hash validation failed")
    return pointer


def _validate_stop_readiness(
    run_dir: Path,
    *,
    expected_step: int,
    identity_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    readiness_path = run_dir / "formal_hosted_stop_readiness.json"
    stage_path = run_dir / f"hosted_stages/step-{expected_step:08d}.json"
    if any(path.is_symlink() or not path.is_file() for path in (readiness_path, stage_path)):
        raise FileNotFoundError("hosted stop readiness or immutable stage record is absent")
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    retained = json.loads(stage_path.read_text(encoding="utf-8"))
    critical = (
        "schema_version",
        "status",
        "stop_step",
        "global_step",
        "checkpoint",
        "checkpoint_manifest_sha256",
        "checkpoint_model_sha256",
        "best_step",
        "best_model_sha256",
        "identity_hashes",
        "interval_evaluation",
    )
    if any(readiness.get(key) != retained.get(key) for key in critical):
        raise ValueError("current and retained hosted-stop readiness records disagree")
    checkpoint = run_dir / _safe_relative(readiness.get("checkpoint"), "checkpoint")
    checkpoint_manifest = _validate_checkpoint(
        checkpoint,
        expected_step=expected_step,
        expected_identity_hashes=identity_hashes,
    )
    best = _validate_best_checkpoint(run_dir, identity_hashes=identity_hashes)
    interval = _mapping(readiness.get("interval_evaluation"), "interval_evaluation")
    if (
        readiness.get("schema_version") != HOSTED_STOP_READINESS_SCHEMA
        or readiness.get("status") != "PASSED"
        or readiness.get("run_name") != RUN_NAME
        or readiness.get("model_contract") != "relation_v2"
        or int(readiness.get("model_parameter_count", -1)) != EXPECTED_PARAMETERS
        or int(readiness.get("stop_step", -1)) != expected_step
        or int(readiness.get("global_step", -1)) != expected_step
        or readiness.get("checkpoint_manifest_sha256")
        != _sha256(checkpoint / "checkpoint_manifest.json")
        or readiness.get("checkpoint_model_sha256") != checkpoint_manifest["sha256"]["model.pt"]
        or int(readiness.get("best_step", -1)) != int(best["step"])
        or readiness.get("best_model_sha256") != best["model_sha256"]
        or readiness.get("identity_hashes") != dict(identity_hashes)
        or int(interval.get("step", -1)) != expected_step
        or int(interval.get("samples", -1)) != EXPECTED_VALIDATION_ANCHORS
        or interval.get("phase") != "interval"
        or interval.get("model_contract") != "relation_v2"
    ):
        raise ValueError("hosted stop readiness failed strict launcher revalidation")
    return readiness


def _validate_run_identity(
    run_dir: Path,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    identity_hashes_path = run_dir / "identity_hashes.json"
    model_identity_path = run_dir / "model_identity.json"
    dataset_identity_path = run_dir / "dataset_identity.json"
    source_identity_path = run_dir / "source_identity.json"
    for path in (
        identity_hashes_path,
        model_identity_path,
        dataset_identity_path,
        source_identity_path,
    ):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"prior run identity is incomplete: {path}")
    identity_hashes = json.loads(identity_hashes_path.read_text(encoding="utf-8"))
    model = json.loads(model_identity_path.read_text(encoding="utf-8"))
    dataset = json.loads(dataset_identity_path.read_text(encoding="utf-8"))
    source = json.loads(source_identity_path.read_text(encoding="utf-8"))
    bundle_source = _mapping(manifest.get("source"), "source")
    if (
        not isinstance(identity_hashes, dict)
        or not identity_hashes
        or any(
            not _is_sha256(value) and key not in {"git_commit", "git_head_tree"}
            for key, value in identity_hashes.items()
        )
        or model.get("model_contract") != "relation_v2"
        or int(model.get("parameter_count", -1)) != EXPECTED_PARAMETERS
        or dataset.get("base_dataset_id") != EXPECTED_BASE_DATASET_ID
        or dataset.get("target_dataset_id") != EXPECTED_TARGET_DATASET_ID
        or dataset.get("base_dataset_manifest_sha256")
        != BASE_AUTHORITY["dataset_manifest_sha256"]
        or dataset.get("target_dataset_manifest_sha256")
        != TARGET_AUTHORITY["dataset_manifest_sha256"]
        or dataset.get("relation_contract_sha256") != EXPECTED_RELATION_BUNDLE_SHA256
        or dataset.get("normalization_artifact_sha256")
        != EXPECTED_NORMALIZATION_SHA256
        or source.get("git_commit") != bundle_source.get("git_commit")
        or source.get("git_head_tree") != bundle_source.get("git_head_tree")
        or source.get("source_tree_sha256") != bundle_source.get("source_tree_sha256")
        or source.get("git_clean") is not True
    ):
        raise ValueError("prior output run/source/data/model identity differs from this bundle")
    if identity_hashes.get("git_commit") != source["git_commit"] or identity_hashes.get(
        "git_head_tree"
    ) != source["git_head_tree"]:
        raise ValueError("prior output identity hashes do not bind its source release")
    return identity_hashes


def _run_file_inventory(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if path == run_dir / STAGE_MANIFEST_NAME:
            continue
        if path.is_symlink():
            raise ValueError(f"hosted run output contains a forbidden symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file() or any(
            ".tmp" in part or ".partial" in part for part in path.parts
        ):
            raise ValueError(f"hosted run output contains an incomplete file: {path}")
        rows.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not rows:
        raise ValueError("hosted run output contains no state-bearing files")
    return rows


def _validate_inventory_rows(run_dir: Path, rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("hosted stage manifest lacks its run-file inventory")
    observed_paths: set[str] = set()
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"run_files[{index}]")
        relative = _safe_relative(row.get("path"), "run file")
        key = relative.as_posix()
        path = run_dir / relative
        if (
            key in observed_paths
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(row.get("size_bytes", -1))
            or _sha256(path) != row.get("sha256")
        ):
            raise ValueError(f"prior output run-file identity failed: {key}")
        observed_paths.add(key)
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path != run_dir / STAGE_MANIFEST_NAME
    }
    if actual != observed_paths:
        raise ValueError("prior output contains unregistered or missing run files")
    return rows


def _validate_stage_manifest(
    run_dir: Path,
    *,
    manifest: Mapping[str, Any],
    bundle_manifest_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = run_dir / STAGE_MANIFEST_NAME
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("prior output lacks the hash-bound hosted stage manifest")
    stage = json.loads(path.read_text(encoding="utf-8"))
    source = _mapping(manifest.get("source"), "source")
    if (
        stage.get("schema_version") != HOSTED_STAGE_SCHEMA
        or stage.get("run_name") != RUN_NAME
        or stage.get("route") != ROUTE
        or int(stage.get("model_parameter_count", -1)) != EXPECTED_PARAMETERS
        or stage.get("bundle_manifest_sha256") != bundle_manifest_sha256
        or stage.get("dataset_ref") != EXPECTED_DATASET_REF
        or stage.get("source_archive_sha256")
        != _mapping(source.get("archive"), "source.archive").get("sha256")
        or stage.get("source_tree_sha256") != source.get("source_tree_sha256")
        or stage.get("normalization_sha256") != EXPECTED_NORMALIZATION_SHA256
        or stage.get("relation_bundle_sha256") != EXPECTED_RELATION_BUNDLE_SHA256
        or stage.get("runtime_contract_sha256")
        != EXPECTED_RUNTIME_CONTRACT_SHA256
        or stage.get("runtime_inventory_sha256")
        != EXPECTED_RUNTIME_INVENTORY_SHA256
        or int(stage.get("run_file_count", -1)) != len(stage.get("run_files") or ())
    ):
        raise ValueError("prior output hosted stage identity differs from this bundle")
    rows = _validate_inventory_rows(run_dir, stage.get("run_files"))
    identity_hashes = _validate_run_identity(run_dir, manifest=manifest)
    steps = _checkpoint_steps(run_dir)
    latest = max(steps, default=0)
    if (
        latest not in EXPECTED_TRAINING_STOP_STEPS
        or int(stage.get("latest_checkpoint_step", -1)) != latest
    ):
        raise ValueError("prior output stage does not bind its latest checkpoint")
    for step in steps:
        _validate_checkpoint(
            run_dir / f"checkpoints/checkpoint-{step:08d}",
            expected_step=step,
            expected_identity_hashes=identity_hashes,
        )
    if latest in EXPECTED_TRAINING_STOP_STEPS:
        _validate_stop_readiness(
            run_dir,
            expected_step=latest,
            identity_hashes=identity_hashes,
        )
    progress_path = run_dir / "free_running/hosted_progress.json"
    if (run_dir / "SUCCESS").is_file():
        if stage.get("status") != "SUCCEEDED":
            raise ValueError("successful run has a non-success hosted stage status")
    elif latest < 4000:
        if stage.get("status") != "TRAINING_STAGED":
            raise ValueError("incomplete training has an invalid hosted stage status")
    elif progress_path.is_file():
        _validate_free_running_progress(run_dir)
        if stage.get("status") != "EVALUATION_IN_PROGRESS":
            raise ValueError("hosted stage status does not match free-running progress")
    elif latest >= 4000 and not (run_dir / "SUCCESS").is_file():
        if stage.get("status") != "TRAINING_COMPLETE_EVALUATION_PENDING":
            raise ValueError("step-4000 stage must declare evaluation pending")
    return stage, rows


def _prior_output_not_found(error: BaseException) -> bool:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if status == 404 or getattr(error, "status_code", None) == 404:
        return True
    text = str(error).lower()
    return "404" in text and ("not found" in text or "no output" in text)


def _download_prior_output(notebook_ref: str, destination: Path) -> Path | None:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Kaggle image lacks kagglehub for strict prior-output restore") from exc
    if destination.exists():
        raise FileExistsError(destination)
    try:
        downloaded = kagglehub.notebook_output_download(
            notebook_ref,
            force_download=True,
        )
    except BaseException as exc:
        if _prior_output_not_found(exc):
            print(
                "RELATION_V2_P100_NO_PRIOR_OUTPUT_404 "
                f"notebook_ref={notebook_ref} fresh_start=true",
                flush=True,
            )
            return None
        raise RuntimeError("prior Kaggle notebook output lookup failed") from exc
    if not downloaded:
        raise RuntimeError("kagglehub returned no prior-output path")
    root = Path(downloaded).resolve()
    if not any(root.rglob("*")):
        raise RuntimeError(
            "kagglehub prior-output lookup returned an empty directory without a 404"
        )
    return root


def _find_prior_run(root: Path) -> Path:
    matches = sorted(
        path.parent
        for path in root.rglob(STAGE_MANIFEST_NAME)
        if path.parent.name == RUN_NAME and path.is_file() and not path.is_symlink()
    )
    unique = sorted(set(path.resolve() for path in matches))
    if len(unique) != 1:
        raise RuntimeError(
            f"expected exactly one hash-bound prior run under {root}, found {len(unique)}"
        )
    return unique[0]


def _validate_success(run_dir: Path) -> None:
    success_path = run_dir / "SUCCESS"
    run_manifest_path = run_dir / "run_manifest.json"
    if success_path.is_symlink() or not success_path.is_file():
        raise FileNotFoundError("final SUCCESS identity is absent")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if (
        success.get("schema_version") != "trauma_predict.multires_event_v2_success.v1"
        or success.get("run_manifest_sha256") != _sha256(run_manifest_path)
        or run_manifest.get("status") != "SUCCEEDED"
        or run_manifest.get("route") != ROUTE
        or run_manifest.get("run_name") != RUN_NAME
        or run_manifest.get("model_contract") != "relation_v2"
        or int(
            _mapping(run_manifest.get("training"), "run_manifest.training").get(
                "training_completed_step", -1
            )
        )
        != 4000
    ):
        raise ValueError("final Relation V2 SUCCESS/run manifest validation failed")


def _select_stop_step(current_step: int) -> int:
    for step in EXPECTED_TRAINING_STOP_STEPS:
        if current_step < step:
            return step
    return 0


def _free_running_rows(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(row["path"]): str(row["sha256"])
        for row in rows
        if str(row["path"]).startswith("free_running/")
    }


def _validate_free_running_progress(run_dir: Path) -> dict[str, Any]:
    free_root = run_dir / "free_running"
    progress_path = free_root / "hosted_progress.json"
    if progress_path.is_symlink() or not progress_path.is_file():
        raise FileNotFoundError("free-running hosted progress is absent")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    identity = _mapping(progress.get("identity"), "free-running progress.identity")
    chunks = progress.get("chunk_manifests")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("free-running hosted progress lacks chunk manifests")
    completed = int(progress.get("completed_anchors", -1))
    expected = int(progress.get("expected_anchors", -1))
    new_anchors = int(progress.get("new_anchors", -1))
    if (
        progress.get("schema_version")
        != "trauma_predict.multires_event_v2_free_running_hosted_progress.v1"
        or progress.get("status") not in {"INCOMPLETE", "COMPLETE"}
        or completed != int(progress.get("completed", -2))
        or expected != int(progress.get("expected", -2))
        or expected != EXPECTED_VALIDATION_ANCHORS
        or not 0 < completed <= expected
        or new_anchors < 1
        or progress.get("identity_sha256") != _sha256_payload(identity)
        or progress.get("chunk_manifest_set_sha256") != _sha256_payload(chunks)
        or sum(int(_mapping(row, "chunk manifest").get("anchors", -1)) for row in chunks)
        != completed
    ):
        raise ValueError("free-running hosted progress contract failed")
    if progress["status"] == "INCOMPLETE" and completed >= expected:
        raise ValueError("incomplete free-running progress already covers every anchor")
    if progress["status"] == "COMPLETE" and completed != expected:
        raise ValueError("complete free-running progress does not cover every anchor")
    seen: set[str] = set()
    for expected_index, raw_row in enumerate(chunks):
        row = _mapping(raw_row, "chunk manifest")
        relative = _safe_relative(row.get("manifest_path"), "chunk manifest path")
        key = relative.as_posix()
        path = free_root / relative
        if (
            key in seen
            or re.fullmatch(
                r"chunks/rank[0-9]{5}/chunk[0-9]{6}/manifest\.json",
                key,
            )
            is None
            or int(row.get("rank", -1)) != 0
            or int(row.get("chunk_index", -1)) != expected_index
            or int(row.get("anchors", -1)) < 1
            or path.is_symlink()
            or not path.is_file()
            or _sha256(path) != row.get("manifest_sha256")
        ):
            raise ValueError(f"free-running chunk progress hash failed: {key}")
        seen.add(key)
    return progress


def _write_stage_manifest(
    run_dir: Path,
    *,
    manifest: Mapping[str, Any],
    bundle_manifest_sha256: str,
    run_files: list[dict[str, Any]],
) -> Path:
    steps = _checkpoint_steps(run_dir)
    source = _mapping(manifest.get("source"), "source")
    if (run_dir / "SUCCESS").is_file():
        status = "SUCCEEDED"
    elif max(steps, default=0) < 4000:
        status = "TRAINING_STAGED"
    elif (run_dir / "free_running/hosted_progress.json").is_file():
        status = "EVALUATION_IN_PROGRESS"
    else:
        status = "TRAINING_COMPLETE_EVALUATION_PENDING"
    payload = {
        "schema_version": HOSTED_STAGE_SCHEMA,
        "status": status,
        "run_name": RUN_NAME,
        "route": ROUTE,
        "model_parameter_count": EXPECTED_PARAMETERS,
        "dataset_ref": EXPECTED_DATASET_REF,
        "bundle_manifest_sha256": bundle_manifest_sha256,
        "source_archive_sha256": _mapping(source.get("archive"), "source.archive")[
            "sha256"
        ],
        "source_tree_sha256": source["source_tree_sha256"],
        "normalization_sha256": EXPECTED_NORMALIZATION_SHA256,
        "relation_bundle_sha256": EXPECTED_RELATION_BUNDLE_SHA256,
        "runtime_contract_sha256": EXPECTED_RUNTIME_CONTRACT_SHA256,
        "runtime_inventory_sha256": EXPECTED_RUNTIME_INVENTORY_SHA256,
        "latest_checkpoint_step": max(steps, default=0),
        "run_file_count": len(run_files),
        "run_files": run_files,
    }
    path = run_dir / STAGE_MANIFEST_NAME
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _validate_manifest_contract(manifest: Mapping[str, Any]) -> None:
    hardware = _mapping(manifest.get("hardware"), "hardware")
    hosted = _mapping(manifest.get("hosted"), "hosted")
    relation = _mapping(manifest.get("relation_contract"), "relation_contract")
    _validate_runtime_manifest(manifest)
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("dataset_ref") != EXPECTED_DATASET_REF
        or manifest.get("notebook_ref") != EXPECTED_NOTEBOOK_REF
        or manifest.get("route") != ROUTE
        or manifest.get("run_name") != RUN_NAME
        or int(manifest.get("model_parameter_count", -1)) != EXPECTED_PARAMETERS
        or hardware.get("accelerator") != "NVIDIA Tesla P100"
        or int(hardware.get("required_cuda_devices", -1)) != 1
        or int(hardware.get("world_size", -1)) != 1
        or int(hardware.get("per_device_train_batch_size", -1)) != 64
        or tuple(hosted.get("training_stop_steps") or ())
        != EXPECTED_TRAINING_STOP_STEPS
        or int(hosted.get("free_running_max_new_anchors", -1)) < 1
        or relation.get("bundle_sha256") != EXPECTED_RELATION_BUNDLE_SHA256
        or int(relation.get("target_target_edges", -1)) != 52
        or int(relation.get("input_target_edges", -1)) != 39
        or int(relation.get("edge_specific_parameters", -1)) != 91
    ):
        raise ValueError("mounted bundle differs from the frozen P100 Relation V2 contract")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch the offline r9 Relation V2 model on one Kaggle P100"
    )
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--scratch-root", type=Path, default=Path("/kaggle/temp/relation_v2_p100"))
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--prior-output-root", type=Path)
    parser.add_argument("--skip-prior-output-download", action="store_true")
    args = parser.parse_args()

    bundle, manifest, manifest_path = _find_bundle(args.bundle_root)
    _validate_manifest_contract(manifest)
    runtime = _validate_runtime_manifest(manifest)
    _resolve_file(
        bundle,
        _mapping(runtime.get("contract"), "runtime.contract"),
        "P100 cu126 runtime contract",
    )
    bundle_manifest_sha256 = _sha256(manifest_path)
    launcher = _resolve_file(
        bundle,
        _mapping(manifest.get("launcher"), "launcher"),
        "bundle launcher",
    )
    if launcher.resolve() != Path(__file__).resolve():
        raise ValueError("executed launcher is not the manifest-bound mounted launcher")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("isolated P100 cu126 PyTorch runtime is absent") from exc
    device_name = _validate_isolated_torch_runtime(torch, manifest)
    dependencies = _validate_runtime_dependencies()

    scratch = args.scratch_root.resolve()
    output_root = args.output_root.resolve()
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        if Path("/kaggle/working").resolve() not in (output_root, *output_root.parents):
            raise ValueError("hosted outputs must remain under /kaggle/working")
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    output_root.mkdir(parents=True, exist_ok=True)

    data = _mapping(manifest.get("data"), "data")
    base_declared = _mapping(data.get("base"), "data.base")
    target_declared = _mapping(data.get("target"), "data.target")
    base_root = _materialize_dataset_view(
        bundle,
        base_declared,
        scratch / "mounted_data/multires_event_v1_c4_full_20260712",
        scratch / "small_payloads/base",
        label="V1 base",
    )
    target_root = _materialize_dataset_view(
        bundle,
        target_declared,
        scratch / "mounted_data/multires_event_m4_target_v2_c4_full_20260714_r9",
        scratch / "small_payloads/target",
        label="r9 target",
    )
    _validate_dataset_identity(
        base_root,
        base_declared,
        BASE_AUTHORITY,
        label="V1 base",
    )
    _validate_dataset_identity(
        target_root,
        target_declared,
        TARGET_AUTHORITY,
        label="r9 target",
    )
    repo_root = _extract_source(bundle, manifest, scratch)
    normalization = _resolve_file(
        bundle,
        _mapping(manifest.get("input_normalization"), "input_normalization"),
        "input normalization",
    )
    if _sha256(normalization) != EXPECTED_NORMALIZATION_SHA256:
        raise ValueError("mounted normalization differs from the frozen train-only artifact")

    run_dir = output_root / RUN_NAME
    prior_stage: dict[str, Any] | None = None
    prior_rows: list[dict[str, Any]] = []
    prior_completed_anchors = 0
    prior_root: Path | None = None
    if args.prior_output_root is not None:
        prior_root = args.prior_output_root.resolve()
    elif run_dir.is_dir():
        prior_root = output_root
    elif not args.skip_prior_output_download:
        prior_root = _download_prior_output(EXPECTED_NOTEBOOK_REF, scratch / "prior_output")
    if prior_root is not None:
        prior_run = _find_prior_run(prior_root)
        prior_stage, prior_rows = _validate_stage_manifest(
            prior_run,
            manifest=manifest,
            bundle_manifest_sha256=bundle_manifest_sha256,
        )
        if prior_run != run_dir:
            if run_dir.exists():
                raise FileExistsError("current output and restored prior output both exist")
            shutil.copytree(prior_run, run_dir, symlinks=False)
            _validate_stage_manifest(
                run_dir,
                manifest=manifest,
                bundle_manifest_sha256=bundle_manifest_sha256,
            )
        prior_progress_path = run_dir / "free_running/hosted_progress.json"
        if prior_progress_path.is_file():
            prior_completed_anchors = int(
                _validate_free_running_progress(run_dir)["completed_anchors"]
            )
        print(
            "RELATION_V2_P100_PRIOR_OUTPUT_VERIFIED "
            f"step={prior_stage['latest_checkpoint_step']} files={len(prior_rows)}",
            flush=True,
        )
    elif run_dir.exists():
        raise RuntimeError("output directory exists without a verified prior stage")

    if run_dir.is_dir() and (run_dir / "SUCCESS").is_file():
        _validate_success(run_dir)
        print("RELATION_V2_P100_ALREADY_SUCCEEDED", flush=True)
        return 0

    current_step = max(_checkpoint_steps(run_dir), default=0) if run_dir.is_dir() else 0
    stop_step = _select_stop_step(current_step)
    if stop_step and stop_step <= current_step:
        raise AssertionError("selected hosted stop step must exceed the restored checkpoint")
    free_running_limit = int(
        _mapping(manifest.get("hosted"), "hosted")["free_running_max_new_anchors"]
    )
    contracts = output_root / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(normalization, contracts / "multires_event_v1_input_normalization.json")

    environment = os.environ.copy()
    runtime_root = str(Path(os.environ[RUNTIME_ROOT_ENV]).resolve())
    environment.update(
        TRAUMA_PREDICT_DATA_ROOT=str(base_root),
        TRAUMA_PREDICT_V2_TARGET_ROOT=str(target_root),
        TRAUMA_PREDICT_OUTPUT_ROOT=str(output_root),
        TRAUMA_PREDICT_V2_HOSTED_STOP_STEP=str(stop_step),
        TRAUMA_PREDICT_V2_FREE_RUNNING_MAX_NEW_ANCHORS=str(free_running_limit),
        PYTHONPATH=os.pathsep.join((runtime_root, str(repo_root / "src"))),
        PYTHONNOUSERSITE="1",
        TRAUMA_PREDICT_RUNTIME_SITE_PACKAGES=runtime_root,
        TRAUMA_PREDICT_RUNTIME_LOCK_SHA256=EXPECTED_RUNTIME_CONTRACT_SHA256,
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
    )
    print(
        "RELATION_V2_P100_MOUNTED_PREFLIGHT_OK "
        f"device={device_name!r} parameters={EXPECTED_PARAMETERS} "
        f"torch={torch.__version__} cuda={torch.version.cuda} arch=sm_60 "
        "relations=52+39 world_size=1 train_batch=64 "
        f"restored_step={current_step} stop_step={stop_step} "
        f"free_running_max_new_anchors={free_running_limit} "
        f"dependencies={json.dumps(dependencies, sort_keys=True)}",
        flush=True,
    )
    command = [sys.executable, str(repo_root / "notebooks/kaggle/train_relation_v2_p100.py")]
    completed = subprocess.run(command, cwd=repo_root, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"formal Relation V2 P100 process exited with code {completed.returncode}"
        )
    if not run_dir.is_dir():
        raise RuntimeError("formal Relation V2 process returned without its run directory")

    identity_hashes = _validate_run_identity(run_dir, manifest=manifest)
    steps = _checkpoint_steps(run_dir)
    latest = max(steps, default=0)
    for step in steps:
        _validate_checkpoint(
            run_dir / f"checkpoints/checkpoint-{step:08d}",
            expected_step=step,
            expected_identity_hashes=identity_hashes,
        )
    if stop_step:
        if latest != stop_step:
            raise RuntimeError(
                f"hosted training stopped at checkpoint {latest}, expected {stop_step}"
            )
        _validate_stop_readiness(
            run_dir,
            expected_step=stop_step,
            identity_hashes=identity_hashes,
        )
    elif (run_dir / "SUCCESS").is_file():
        _validate_success(run_dir)

    run_files = _run_file_inventory(run_dir)
    if stop_step == 0 and not (run_dir / "SUCCESS").is_file():
        progress = _validate_free_running_progress(run_dir)
        before = _free_running_rows(prior_rows)
        after = _free_running_rows(run_files)
        if not after or not any(before.get(path) != digest for path, digest in after.items()):
            raise RuntimeError(
                "post-step4000 invocation returned without a new hash-bound free-running artifact"
            )
        if int(progress["completed_anchors"]) <= prior_completed_anchors:
            raise RuntimeError("free-running hosted progress did not advance")
    stage_path = _write_stage_manifest(
        run_dir,
        manifest=manifest,
        bundle_manifest_sha256=bundle_manifest_sha256,
        run_files=run_files,
    )
    print(
        "RELATION_V2_P100_HOSTED_STAGE_COMMITTED "
        f"step={latest} status={'SUCCEEDED' if (run_dir / 'SUCCESS').is_file() else 'PARTIAL'} "
        f"files={len(run_files)} stage_manifest_sha256={_sha256(stage_path)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
