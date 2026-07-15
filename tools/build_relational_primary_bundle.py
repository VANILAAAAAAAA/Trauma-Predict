from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any


BUNDLE_SCHEMA = "trauma_predict.multires_event_v2_relational_primary_bundle.v2"
INVENTORY_SCHEMA = "trauma_predict.mounted_file_inventory.v2"
MODEL_PARAMETER_COUNT = 47_801_855
RUN_NAME = "t4x2_multires_event_v2_relational"
BASE_DATASET_ID = "multires_event_v1_c4_full_20260712"
TARGET_DATASET_ID = "multires_event_m4_target_v2_c4_full_20260714_r9"
SMALL_PAYLOAD_THRESHOLD_BYTES = 64 * 1024
MAX_KAGGLE_TOP_LEVEL_FILES = 200


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def dataset_identity(root: Path, expected_id: str) -> dict[str, Any]:
    manifest_path = root / "dataset_manifest.json"
    sample_manifest_path = root / "sample_manifest.csv"
    succeeded_path = root / "SUCCEEDED"
    for path in (manifest_path, sample_manifest_path, succeeded_path):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != expected_id or manifest.get("status") != "SUCCEEDED":
        raise ValueError(f"dataset authority mismatch at {root}")
    return {
        "dataset_id": expected_id,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "sample_manifest_sha256": sha256_file(sample_manifest_path),
        "succeeded_sha256": sha256_file(succeeded_path),
    }


def build_inventory(
    root: Path,
    output: Path,
    *,
    prefix: str,
) -> tuple[dict[str, Any], int, int, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    files = sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    if not files:
        raise ValueError(f"dataset contains no regular files: {root}")
    pack_path = output / f"payload_{prefix}_small_files.tar"
    packed_files = 0
    packed_bytes = 0
    direct_files = 0
    with tarfile.open(pack_path, "w") as pack:
        for index, source in enumerate(files):
            relative = source.relative_to(root).as_posix()
            digest = sha256_file(source)
            # A neutral extension prevents Kaggle from converting structured
            # payloads; the inventory restores each original destination name.
            mounted_name = f"payload_{prefix}_{index:04d}_{digest[:16]}.blob"
            size_bytes = source.stat().st_size
            row = {
                "destination": relative,
                "sha256": digest,
                "size_bytes": size_bytes,
            }
            if size_bytes <= SMALL_PAYLOAD_THRESHOLD_BYTES:
                member = tarfile.TarInfo(mounted_name)
                member.size = size_bytes
                member.mode = 0o444
                member.uid = 0
                member.gid = 0
                member.uname = ""
                member.gname = ""
                member.mtime = 0
                with source.open("rb") as handle:
                    pack.addfile(member, handle)
                row.update(storage="packed", archive_member=mounted_name)
                packed_files += 1
                packed_bytes += size_bytes
            else:
                destination = output / mounted_name
                link_or_copy(source, destination)
                row.update(storage="mounted", mounted_path=mounted_name)
                direct_files += 1
            rows.append(row)
            total_bytes += size_bytes
    packed_payload: dict[str, Any] | None = None
    if packed_files:
        packed_payload = {
            "path": pack_path.name,
            "sha256": sha256_file(pack_path),
            "file_count": packed_files,
            "uncompressed_bytes": packed_bytes,
            "archive_bytes": pack_path.stat().st_size,
        }
    else:
        pack_path.unlink()
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "source_root_name": root.name,
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "direct_mounted_file_count": direct_files,
        "packed_file_count": packed_files,
        "files": rows,
    }
    if packed_payload is not None:
        inventory["packed_payload"] = packed_payload
    inventory_path = output / f"{prefix}_inventory.json"
    write_json(inventory_path, inventory)
    return (
        {"path": inventory_path.name, "sha256": sha256_file(inventory_path)},
        total_bytes,
        len(rows),
        {
            "direct_files": direct_files,
            "packed_files": packed_files,
            "packed_uncompressed_bytes": packed_bytes,
            "pack_archives": int(packed_payload is not None),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the immutable no-copy Kaggle bundle for the relational primary"
    )
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-ref", required=True)
    parser.add_argument("--resume-archive", type=Path)
    parser.add_argument("--resume-checkpoint-dir")
    parser.add_argument("--hosted-verification-stop-after-formal-step2", action="store_true")
    parser.add_argument("--hosted-verification-stop-after-resume-step3", action="store_true")
    args = parser.parse_args()
    if (args.resume_archive is None) != (args.resume_checkpoint_dir is None):
        parser.error("--resume-archive and --resume-checkpoint-dir must be paired")
    if args.hosted_verification_stop_after_formal_step2 and (
        args.hosted_verification_stop_after_resume_step3
    ):
        parser.error("hosted verification stop modes are mutually exclusive")
    if args.hosted_verification_stop_after_resume_step3 and args.resume_checkpoint_dir != (
        "checkpoint-00000002"
    ):
        parser.error("resume step-3 verification requires checkpoint-00000002")
    return args


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite bundle: {output}")
    output.mkdir(parents=True)
    try:
        base_root = args.base_root.resolve()
        target_root = args.target_root.resolve()
        base_identity = dataset_identity(base_root, BASE_DATASET_ID)
        target_identity = dataset_identity(target_root, TARGET_DATASET_ID)
        base_inventory, base_bytes, base_files, base_storage = build_inventory(
            base_root, output, prefix="base"
        )
        target_inventory, target_bytes, target_files, target_storage = build_inventory(
            target_root, output, prefix="target"
        )

        source_archive = output / "trauma_predict_relational_primary_source.tar.gz"
        launcher = output / "run_relational_primary_bundle.py"
        normalization = output / "multires_event_v1_input_normalization.json"
        shutil.copy2(args.source_archive.resolve(), source_archive)
        shutil.copy2(args.launcher.resolve(), launcher)
        shutil.copy2(args.normalization.resolve(), normalization)

        manifest = {
            "schema": BUNDLE_SCHEMA,
            "dataset_ref": args.dataset_ref,
            "mode": "relational",
            "run_name": RUN_NAME,
            "model_parameter_count": MODEL_PARAMETER_COUNT,
            "hosted_verification_stop_after_formal_step2": bool(
                args.hosted_verification_stop_after_formal_step2
            ),
            "hosted_verification_stop_after_resume_step3": bool(
                args.hosted_verification_stop_after_resume_step3
            ),
            "data": {
                "base": {**base_identity, "inventory": base_inventory},
                "target": {**target_identity, "inventory": target_inventory},
            },
            "source": {"path": source_archive.name, "sha256": sha256_file(source_archive)},
            "launcher": {"path": launcher.name, "sha256": sha256_file(launcher)},
            "input_normalization": {
                "path": normalization.name,
                "sha256": sha256_file(normalization),
            },
            "payload_summary": {
                "base_bytes": base_bytes,
                "target_bytes": target_bytes,
                "logical_dataset_files": base_files + target_files,
                "direct_mounted_dataset_files": (
                    base_storage["direct_files"] + target_storage["direct_files"]
                ),
                "small_packed_dataset_files": (
                    base_storage["packed_files"] + target_storage["packed_files"]
                ),
                "small_packed_uncompressed_bytes": (
                    base_storage["packed_uncompressed_bytes"]
                    + target_storage["packed_uncompressed_bytes"]
                ),
                "bulk_patient_payload_copy_inside_notebook": False,
                "bulk_patient_payload_extraction_inside_notebook": False,
                "small_payload_pack_extraction_inside_notebook": True,
            },
        }
        if args.resume_archive is not None:
            checkpoint_dir = str(args.resume_checkpoint_dir)
            if not checkpoint_dir.startswith("checkpoint-") or not checkpoint_dir.removeprefix(
                "checkpoint-"
            ).isdigit():
                raise ValueError("resume checkpoint directory must be checkpoint-<integer>")
            resume_archive = output / "relational_primary_resume_checkpoint.tar.gz"
            shutil.copy2(args.resume_archive.resolve(), resume_archive)
            manifest["resume"] = {
                "path": resume_archive.name,
                "sha256": sha256_file(resume_archive),
                "checkpoint_dir": checkpoint_dir,
            }
        write_json(output / "run_bundle_manifest.json", manifest)
        owner, slug = args.dataset_ref.split("/", 1)
        write_json(
            output / "dataset-metadata.json",
            {
                "title": "Trauma Predict Relational Primary R9 Run Bundle",
                "id": f"{owner}/{slug}",
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            },
        )
        top_level_files = len([path for path in output.iterdir() if path.is_file()])
        if top_level_files > MAX_KAGGLE_TOP_LEVEL_FILES:
            raise ValueError(
                "bundle exceeds the frozen Kaggle top-level file budget: "
                f"{top_level_files} > {MAX_KAGGLE_TOP_LEVEL_FILES}"
            )
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    print(
        "RELATIONAL_PRIMARY_BUNDLE_BUILT "
        f"path={output} manifest_sha256={sha256_file(output / 'run_bundle_manifest.json')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
