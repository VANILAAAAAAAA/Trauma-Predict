from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "trauma_predict.multires_event_v2_relational_primary_bundle.v1"
RUN_NAME = "t4x2_multires_event_v2_relational"
EXPECTED_PARAMETERS = 47_801_855
EXPECTED_TARGET_DATASET_ID = "multires_event_m4_target_v2_c4_full_20260714_r9"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


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


def _resolve_file(bundle: Path, row: dict[str, Any], label: str) -> Path:
    relative = Path(str(row.get("path") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path must remain inside the mounted bundle")
    path = bundle / relative
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"missing mounted {label}: {path}")
    expected = str(row.get("sha256") or "")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"mounted {label} hash mismatch: {observed} != {expected}")
    return path


def _find_bundle(explicit: Path | None) -> tuple[Path, dict[str, Any]]:
    if explicit is not None:
        candidates = [explicit.resolve() / "run_bundle_manifest.json"]
    else:
        candidates = sorted(Path("/kaggle/input").glob("*/run_bundle_manifest.json"))
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") == MANIFEST_SCHEMA:
            matches.append((path.parent.resolve(), payload))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one mounted relational-primary bundle, found {len(matches)}"
        )
    return matches[0]


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"source archive path escapes destination: {member.name}") from exc
            if member.issym() or member.islnk():
                raise ValueError(f"source archive links are forbidden: {member.name}")
        handle.extractall(destination, members=members, filter="data")


def _validate_dataset_identity(
    root: Path,
    declared: dict[str, Any],
    *,
    expected_dataset_id: str,
    label: str,
) -> None:
    manifest_path = root / "dataset_manifest.json"
    sample_manifest_path = root / "sample_manifest.csv"
    succeeded = root / "SUCCEEDED"
    for path in (manifest_path, sample_manifest_path, succeeded):
        if not path.is_file():
            raise FileNotFoundError(f"mounted {label} is incomplete: {path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "SUCCEEDED" or payload.get("dataset_id") != expected_dataset_id:
        raise ValueError(f"mounted {label} identity/status mismatch")
    checks = {
        "dataset_manifest_sha256": _sha256(manifest_path),
        "sample_manifest_sha256": _sha256(sample_manifest_path),
        "succeeded_sha256": _sha256(succeeded),
    }
    for key, observed in checks.items():
        if str(declared.get(key) or "") != observed:
            raise ValueError(f"mounted {label} {key} mismatch")


def _safe_relative(value: Any, label: str) -> Path:
    relative = Path(str(value or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a non-empty relative path")
    return relative


def _materialize_dataset_view(
    bundle: Path,
    declared: dict[str, Any],
    destination: Path,
    *,
    label: str,
) -> Path:
    """Create a no-copy filesystem view over flat files mounted by Kaggle."""

    inventory_path = _resolve_file(
        bundle,
        _mapping(declared.get("inventory"), f"{label}.inventory"),
        f"{label} inventory",
    )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("schema") != "trauma_predict.mounted_file_inventory.v1":
        raise ValueError(f"{label} inventory schema mismatch")
    files = inventory.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{label} inventory must declare files")
    destination.mkdir(parents=True, exist_ok=False)
    seen_destinations: set[str] = set()
    seen_payloads: set[str] = set()
    for index, raw_row in enumerate(files):
        row = _mapping(raw_row, f"{label}.inventory.files[{index}]")
        payload_relative = _safe_relative(row.get("mounted_path"), "mounted_path")
        destination_relative = _safe_relative(row.get("destination"), "destination")
        payload_key = payload_relative.as_posix()
        destination_key = destination_relative.as_posix()
        if payload_key in seen_payloads or destination_key in seen_destinations:
            raise ValueError(f"{label} inventory contains duplicate paths")
        seen_payloads.add(payload_key)
        seen_destinations.add(destination_key)
        source = bundle / payload_relative
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"missing mounted {label} payload: {source}")
        if source.stat().st_size != int(row.get("size_bytes", -1)):
            raise ValueError(f"mounted {label} payload size mismatch: {source}")
        expected_sha256 = str(row.get("sha256") or "")
        if _sha256(source) != expected_sha256:
            raise ValueError(f"mounted {label} payload hash mismatch: {source}")
        target = destination / destination_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    if int(inventory.get("file_count", -1)) != len(files):
        raise ValueError(f"{label} inventory file count mismatch")
    return destination


def _restore_optional_checkpoint(
    bundle: Path,
    manifest: dict[str, Any],
    output_root: Path,
) -> None:
    resume = manifest.get("resume")
    if resume is None:
        return
    row = _mapping(resume, "resume")
    archive = _resolve_file(bundle, row, "resume checkpoint archive")
    _safe_extract(archive, output_root)
    expected = output_root / RUN_NAME / "checkpoints" / str(row.get("checkpoint_dir"))
    if not (expected / "checkpoint_manifest.json").is_file():
        raise FileNotFoundError(
            "restored checkpoint archive lacks its declared complete checkpoint"
        )
    print(f"RELATIONAL_PRIMARY_RESUME_RESTORED path={expected}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch the mounted r9 relational primary without network or data rebuilding"
    )
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument(
        "--working-root",
        type=Path,
        default=Path("/kaggle/working/multires_event_v2_relational_primary_r9"),
    )
    args = parser.parse_args()

    bundle, manifest = _find_bundle(args.bundle_root)
    if int(manifest.get("model_parameter_count", -1)) != EXPECTED_PARAMETERS:
        raise ValueError("bundle model parameter count is not the frozen 47,801,855")
    if manifest.get("mode") != "relational" or manifest.get("run_name") != RUN_NAME:
        raise ValueError("bundle is not the authorized relational primary")
    if (
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        and Path("/kaggle/working").resolve()
        not in args.working_root.resolve().parents
    ):
        raise ValueError("hosted output must remain under /kaggle/working")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Kaggle image lacks PyTorch") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(
            "relational primary requires exactly two visible GPUs; "
            f"found {torch.cuda.device_count()}"
        )
    dependency_versions = _validate_runtime_dependencies()

    source = _mapping(manifest.get("source"), "source")
    source_archive = _resolve_file(bundle, source, "source release")
    data = _mapping(manifest.get("data"), "data")
    base_declared = _mapping(data.get("base"), "data.base")
    target_declared = _mapping(data.get("target"), "data.target")
    working_root = args.working_root.resolve()
    if working_root.exists():
        shutil.rmtree(working_root)
    data_views = working_root / "mounted_data"
    base_root = _materialize_dataset_view(
        bundle,
        base_declared,
        data_views / "multires_event_v1_c4_full_20260712",
        label="V1 base",
    )
    target_root = _materialize_dataset_view(
        bundle,
        target_declared,
        data_views / "multires_event_m4_target_v2_c4_full_20260714_r9",
        label="r9 target",
    )
    _validate_dataset_identity(
        base_root,
        base_declared,
        expected_dataset_id="multires_event_v1_c4_full_20260712",
        label="V1 base",
    )
    _validate_dataset_identity(
        target_root,
        target_declared,
        expected_dataset_id=EXPECTED_TARGET_DATASET_ID,
        label="r9 target",
    )
    normalization = _resolve_file(
        bundle,
        _mapping(manifest.get("input_normalization"), "input_normalization"),
        "input normalization",
    )

    source_parent = working_root / "source"
    repo_root = source_parent / "Trauma-Predict"
    output_root = working_root / "output"
    _safe_extract(source_archive, source_parent)
    if not (repo_root / "notebooks/kaggle/train_relational_primary.py").is_file():
        raise FileNotFoundError("source release lacks the primary training entry point")
    output_contracts = output_root / "contracts"
    output_contracts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(normalization, output_contracts / "multires_event_v1_input_normalization.json")
    _restore_optional_checkpoint(bundle, manifest, output_root)

    environment = os.environ.copy()
    environment.update(
        TRAUMA_PREDICT_DATA_ROOT=str(base_root),
        TRAUMA_PREDICT_V2_TARGET_ROOT=str(target_root),
        TRAUMA_PREDICT_OUTPUT_ROOT=str(output_root),
        PYTHONPATH=str(repo_root / "src"),
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
    )
    hosted_verification_stop = manifest.get("hosted_verification_stop_after_formal_step2", False)
    if not isinstance(hosted_verification_stop, bool):
        raise TypeError("hosted_verification_stop_after_formal_step2 must be boolean")
    hosted_resume_verification_stop = manifest.get(
        "hosted_verification_stop_after_resume_step3", False
    )
    if not isinstance(hosted_resume_verification_stop, bool):
        raise TypeError("hosted_verification_stop_after_resume_step3 must be boolean")
    if hosted_verification_stop and hosted_resume_verification_stop:
        raise ValueError("hosted verification stop modes are mutually exclusive")
    if hosted_verification_stop:
        environment["TRAUMA_PREDICT_V2_HOSTED_VERIFY_STOP_AFTER_FORMAL_STEP2"] = "1"
    if hosted_resume_verification_stop:
        environment["TRAUMA_PREDICT_V2_HOSTED_VERIFY_STOP_AFTER_RESUME_STEP3"] = "1"
    print(
        "RELATIONAL_PRIMARY_MOUNTED_PREFLIGHT_OK "
        f"target={EXPECTED_TARGET_DATASET_ID} parameters={EXPECTED_PARAMETERS} "
        f"mode=relational GPUs=2 dependencies={json.dumps(dependency_versions, sort_keys=True)}",
        flush=True,
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(repo_root / "notebooks/kaggle/train_relational_primary.py"),
    ]
    completed = subprocess.run(command, cwd=repo_root, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"formal relational primary exited with code {completed.returncode}")
    if hosted_verification_stop:
        readiness_path = output_root / RUN_NAME / "formal_step2_readiness.json"
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        checkpoint = output_root / RUN_NAME / str(readiness.get("checkpoint") or "")
        checkpoint_manifest = checkpoint / "checkpoint_manifest.json"
        if (
            readiness.get("status") != "PASSED"
            or int(readiness.get("global_step", -1)) != 2
            or readiness.get("mode") != "relational"
            or int(readiness.get("model_parameter_count", -1)) != EXPECTED_PARAMETERS
            or _sha256(checkpoint_manifest)
            != readiness.get("checkpoint_manifest_sha256")
        ):
            raise RuntimeError("hosted formal step-2 evidence failed launcher revalidation")
        print(
            "RELATIONAL_PRIMARY_HOSTED_FORMAL_STEP2_VERIFIED "
            f"checkpoint={checkpoint}",
            flush=True,
        )
    if hosted_resume_verification_stop:
        readiness_path = output_root / RUN_NAME / "formal_resume_step3_readiness.json"
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        checkpoint = output_root / RUN_NAME / str(readiness.get("checkpoint") or "")
        checkpoint_manifest = checkpoint / "checkpoint_manifest.json"
        if (
            readiness.get("status") != "PASSED"
            or int(readiness.get("restored_from_step", -1)) != 2
            or int(readiness.get("global_step", -1)) != 3
            or readiness.get("mode") != "relational"
            or int(readiness.get("model_parameter_count", -1)) != EXPECTED_PARAMETERS
            or _sha256(checkpoint_manifest)
            != readiness.get("checkpoint_manifest_sha256")
        ):
            raise RuntimeError("hosted formal resume step-3 evidence failed launcher revalidation")
        print(
            "RELATIONAL_PRIMARY_HOSTED_FORMAL_RESUME_STEP3_VERIFIED "
            f"checkpoint={checkpoint}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
