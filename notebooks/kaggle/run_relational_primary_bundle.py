from __future__ import annotations

import hashlib
import importlib.metadata
import json
import tarfile
from pathlib import Path
from typing import Any


HISTORICAL_BUNDLE_SCHEMA = "trauma_predict.multires_event_v2_relational_primary_bundle.v2"
MANIFEST_SCHEMA = HISTORICAL_BUNDLE_SCHEMA
RELATION_V2_ROUTE_ID = "multires_event_v2_m4_relation_v2"
HOSTED_ROUTE_STATUS = "pending"
DISABLED_MESSAGE = (
    "HISTORICAL_RELATIONAL_PRIMARY_BUNDLE_DISABLED: this launcher is bound to "
    "the invalid v8 relation implementation and cannot launch Relation Contract "
    "V2. A new Relation V2 hosted bundle and notebook have not been frozen or "
    "published."
)


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


def _safe_extract(
    archive: Path,
    destination: Path,
    *,
    expected_file_members: set[str] | None = None,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:*") as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"source archive path escapes destination: {member.name}"
                ) from exc
            if member.issym() or member.islnk():
                raise ValueError(f"source archive links are forbidden: {member.name}")
        if expected_file_members is not None:
            observed_file_members = {member.name for member in members if member.isfile()}
            if observed_file_members != expected_file_members:
                raise ValueError("small payload pack members do not match its inventory")
            if any(not member.isfile() for member in members):
                raise ValueError("small payload pack may contain regular files only")
        handle.extractall(destination, members=members, filter="data")


def _safe_relative(value: Any, label: str) -> Path:
    relative = Path(str(value or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a non-empty relative path")
    return relative


def _materialize_dataset_view(
    bundle: Path,
    declared: dict[str, Any],
    destination: Path,
    packed_root: Path,
    *,
    label: str,
) -> Path:
    """Retain the audited historical hash/materialization helper for evidence tests."""

    inventory_path = _resolve_file(
        bundle,
        _mapping(declared.get("inventory"), f"{label}.inventory"),
        f"{label} inventory",
    )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("schema") != "trauma_predict.mounted_file_inventory.v2":
        raise ValueError(f"{label} inventory schema mismatch")
    files = inventory.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{label} inventory must declare files")
    destination.mkdir(parents=True, exist_ok=False)
    packed_members: set[str] = set()
    packed_uncompressed_bytes = 0
    for index, raw_row in enumerate(files):
        row = _mapping(raw_row, f"{label}.inventory.files[{index}]")
        if row.get("storage") == "packed":
            archive_member = _safe_relative(row.get("archive_member"), "archive_member")
            member_key = archive_member.as_posix()
            if member_key in packed_members:
                raise ValueError(f"{label} inventory contains duplicate packed members")
            packed_members.add(member_key)
            packed_uncompressed_bytes += int(row.get("size_bytes", -1))
    packed_payload = inventory.get("packed_payload")
    if packed_payload is not None:
        packed_row = _mapping(packed_payload, f"{label}.inventory.packed_payload")
        archive = _resolve_file(bundle, packed_row, f"{label} small payload pack")
        if int(packed_row.get("file_count", -1)) != len(packed_members):
            raise ValueError(f"{label} small payload pack file count mismatch")
        if int(packed_row.get("uncompressed_bytes", -1)) != packed_uncompressed_bytes:
            raise ValueError(f"{label} small payload pack byte count mismatch")
        if int(packed_row.get("archive_bytes", -1)) != archive.stat().st_size:
            raise ValueError(f"{label} small payload pack archive size mismatch")
        _safe_extract(archive, packed_root, expected_file_members=packed_members)
    elif packed_members or int(inventory.get("packed_file_count", -1)) != 0:
        raise ValueError(f"{label} inventory lacks its declared small payload pack")
    seen_destinations: set[str] = set()
    seen_payloads: set[str] = set()
    for index, raw_row in enumerate(files):
        row = _mapping(raw_row, f"{label}.inventory.files[{index}]")
        destination_relative = _safe_relative(row.get("destination"), "destination")
        storage = row.get("storage")
        if storage == "mounted":
            payload_relative = _safe_relative(row.get("mounted_path"), "mounted_path")
            source = bundle / payload_relative
            payload_key = f"mounted:{payload_relative.as_posix()}"
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
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"missing {storage} {label} payload: {source}")
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
    packed_rows = sum(
        row.get("storage") == "packed" for row in files if isinstance(row, dict)
    )
    mounted_rows = sum(
        row.get("storage") == "mounted" for row in files if isinstance(row, dict)
    )
    if int(inventory.get("packed_file_count", -1)) != packed_rows:
        raise ValueError(f"{label} inventory packed file count mismatch")
    if int(inventory.get("direct_mounted_file_count", -1)) != mounted_rows:
        raise ValueError(f"{label} inventory mounted file count mismatch")
    return destination


def main() -> int:
    """Reject the historical bundle before reading inputs or starting torchrun."""

    raise RuntimeError(DISABLED_MESSAGE)


if __name__ == "__main__":
    raise SystemExit(main())
