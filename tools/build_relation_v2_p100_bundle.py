from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Mapping


BUNDLE_SCHEMA = "trauma_predict.multires_event_v2_relation_v2_p100_bundle.v1"
DATA_INVENTORY_SCHEMA = "trauma_predict.mounted_file_inventory.v3"
SOURCE_INVENTORY_SCHEMA = "trauma_predict.source_release_inventory.v1"
SOURCE_RELEASE_SCHEMA = "trauma_predict.multires_event_v2_source_release.v1"
ROUTE = "multires_event_v2_m4_relation_v2"
RUN_NAME = "p100_multires_event_v2_relation_v2"
MODEL_PARAMETER_COUNT = 48_728_439
DATASET_REF = "vanila111/trauma-predict-relation-v2-p100-r9-bundle"
NOTEBOOK_REF = "vanila111/trauma-predict-relation-v2-p100-r9"
BASE_DATASET_ID = "multires_event_v1_c4_full_20260712"
TARGET_DATASET_ID = "multires_event_m4_target_v2_c4_full_20260714_r9"
NORMALIZATION_SHA256 = (
    "4f54dbeaab4b2becd349d1d8fcaac7b6bdea2567a20874ee7d29338c1f930add"
)
RELATION_BUNDLE_SHA256 = (
    "0331ec0d552e47790d1dc4f8bae3520062c9e6f5fa62cf62e87c187f6783c033"
)
SMALL_PAYLOAD_THRESHOLD_BYTES = 64 * 1024
MAX_KAGGLE_TOP_LEVEL_FILES = 200
HOSTED_TRAINING_STOPS = (250, 1500, 2750, 4000)
DEFAULT_FREE_RUNNING_MAX_NEW_ANCHORS = 2048

BASE_AUTHORITY = {
    "dataset_id": BASE_DATASET_ID,
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
    "dataset_id": TARGET_DATASET_ID,
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

REQUIRED_SOURCE_PATHS = {
    "configs/dataset/multires_event_v2_relation_v2_c4.yaml",
    "configs/evaluation/multires_event_v2_relation_v2_metrics.json",
    "configs/model/multires_event_v2_relation_v2.yaml",
    "configs/train/p100_multires_event_v2_relation_v2.yaml",
    "notebooks/kaggle/kernel-metadata-relation-v2-p100.template.json",
    "notebooks/kaggle/run_relation_v2_p100_bundle.py",
    "notebooks/kaggle/trauma_predict_relation_v2_p100_r9.ipynb",
    "notebooks/kaggle/train_relation_v2_p100.py",
    "pyproject.toml",
    "requirements-multires-kaggle.txt",
    "src/trauma_predict/training/multires_event_v2.py",
    "tools/build_relation_v2_p100_bundle.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_payload(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(json_bytes(value))


def _git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout


def _is_git_object(value: str) -> bool:
    return len(value) in {40, 64} and all(character in "0123456789abcdef" for character in value)


def _tar_add_bytes(handle: tarfile.TarFile, name: str, content: bytes) -> None:
    import io

    member = tarfile.TarInfo(name)
    member.size = len(content)
    member.mode = 0o444
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    member.mtime = 0
    handle.addfile(member, io.BytesIO(content))


def _tracked_source_files(repo_root: Path) -> list[Path]:
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "source release requires a completely clean Git worktree; commit the accepted "
            "Relation V2 implementation before building the hosted bundle"
        )
    raw = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        capture_output=True,
        check=True,
    ).stdout
    relative_paths = [Path(item.decode("utf-8")) for item in raw.split(b"\0") if item]
    observed = {path.as_posix() for path in relative_paths}
    missing = sorted(REQUIRED_SOURCE_PATHS - observed)
    if missing:
        raise RuntimeError(f"source release lacks required tracked paths: {missing}")
    files: list[Path] = []
    for relative in relative_paths:
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid Git source path: {relative}")
        path = repo_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source release permits regular tracked files only: {path}")
        files.append(path)
    return files


def build_source_release(
    repo_root: Path,
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    files = _tracked_source_files(repo_root)
    if any(path.relative_to(repo_root).as_posix() == "SOURCE_RELEASE.json" for path in files):
        raise ValueError("SOURCE_RELEASE.json must be generated by the bundle builder, not tracked")
    git_commit = _git(repo_root, "rev-parse", "HEAD").strip()
    git_head_tree = _git(repo_root, "rev-parse", "HEAD^{tree}").strip()
    if not _is_git_object(git_commit) or not _is_git_object(git_head_tree):
        raise ValueError("source Git commit/tree identity is invalid")

    executable_candidates = sorted((repo_root / "src/trauma_predict").rglob("*.py"))
    executable_candidates.extend(
        repo_root / relative
        for relative in (
            "notebooks/kaggle/train_relation_v2_p100.py",
            "notebooks/kaggle/run_relation_v2_p100_bundle.py",
            "requirements-multires-kaggle.txt",
            "pyproject.toml",
        )
    )
    executable_hashes: dict[str, str] = {}
    for path in sorted(set(executable_candidates)):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"executable source identity file is absent: {path}")
        executable_hashes[path.relative_to(repo_root).as_posix()] = sha256_file(path)
    source_tree_sha256 = sha256_payload(
        {
            "schema_version": "trauma_predict.multires_event_v2_source_tree.v1",
            "files": executable_hashes,
        }
    )
    release = {
        "schema_version": SOURCE_RELEASE_SCHEMA,
        "git_commit": git_commit,
        "git_head_tree": git_head_tree,
        "source_tree_sha256": source_tree_sha256,
        "source_file_count": len(executable_hashes),
    }
    release_content = json_bytes(release)

    rows: list[dict[str, Any]] = []
    archive = output / "trauma_predict_relation_v2_p100_source.blob"
    with tarfile.open(archive, "w") as handle:
        for path in files:
            relative = path.relative_to(repo_root).as_posix()
            content = path.read_bytes()
            member_name = f"Trauma-Predict/{relative}"
            _tar_add_bytes(handle, member_name, content)
            rows.append(
                {
                    "path": member_name,
                    "size_bytes": len(content),
                    "sha256": sha256_bytes(content),
                }
            )
        release_member = "Trauma-Predict/SOURCE_RELEASE.json"
        _tar_add_bytes(handle, release_member, release_content)
        rows.append(
            {
                "path": release_member,
                "size_bytes": len(release_content),
                "sha256": sha256_bytes(release_content),
            }
        )

    inventory_payload = {
        "schema": SOURCE_INVENTORY_SCHEMA,
        "archive_root": "Trauma-Predict",
        "git_commit": git_commit,
        "git_head_tree": git_head_tree,
        "source_tree_sha256": source_tree_sha256,
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "files": rows,
    }
    inventory_path = output / "source_inventory.json"
    write_json(inventory_path, inventory_payload)
    source = {
        "archive": {
            "path": archive.name,
            "size_bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        },
        "inventory": {
            "path": inventory_path.name,
            "sha256": sha256_file(inventory_path),
        },
        "git_commit": git_commit,
        "git_head_tree": git_head_tree,
        "source_tree_sha256": source_tree_sha256,
    }
    return source, inventory_payload


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def dataset_identity(root: Path, authority: Mapping[str, str]) -> dict[str, Any]:
    manifest_path = root / "dataset_manifest.json"
    sample_manifest_path = root / "sample_manifest.csv"
    succeeded_path = root / "SUCCEEDED"
    required = {
        "dataset_manifest_sha256": manifest_path,
        "sample_manifest_sha256": sample_manifest_path,
        "succeeded_sha256": succeeded_path,
    }
    if "subject_split_sha256" in authority:
        required["subject_split_sha256"] = root / "subject_split.csv"
    for key, path in required.items():
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != authority[key]:
            raise ValueError(f"dataset authority {key} mismatch at {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("dataset_id") != authority["dataset_id"]
        or manifest.get("status") != "SUCCEEDED"
    ):
        raise ValueError(f"dataset identity/status mismatch at {root}")
    return dict(authority)


def build_data_inventory(
    root: Path,
    output: Path,
    *,
    prefix: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    files = sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    if not files:
        raise ValueError(f"dataset contains no regular files: {root}")
    pack_path = output / f"payload_{prefix}_small_pack.blob"
    packed_files = 0
    packed_bytes = 0
    direct_files = 0
    total_bytes = 0
    with tarfile.open(pack_path, "w") as pack:
        for index, source in enumerate(files):
            relative = source.relative_to(root).as_posix()
            digest = sha256_file(source)
            mounted_name = f"payload_{prefix}_{index:04d}_{digest[:16]}.blob"
            size_bytes = source.stat().st_size
            row: dict[str, Any] = {
                "destination": relative,
                "sha256": digest,
                "size_bytes": size_bytes,
            }
            if size_bytes <= SMALL_PAYLOAD_THRESHOLD_BYTES:
                with source.open("rb") as handle:
                    member = tarfile.TarInfo(mounted_name)
                    member.size = size_bytes
                    member.mode = 0o444
                    member.uid = 0
                    member.gid = 0
                    member.uname = ""
                    member.gname = ""
                    member.mtime = 0
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
    inventory: dict[str, Any] = {
        "schema": DATA_INVENTORY_SCHEMA,
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
        {
            "bytes": total_bytes,
            "files": len(rows),
            "direct_files": direct_files,
            "packed_files": packed_files,
            "packed_bytes": packed_bytes,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the immutable offline Kaggle bundle for Relation V2 on one P100"
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-ref", default=DATASET_REF)
    parser.add_argument("--notebook-ref", default=NOTEBOOK_REF)
    parser.add_argument(
        "--free-running-max-new-anchors",
        type=int,
        default=DEFAULT_FREE_RUNNING_MAX_NEW_ANCHORS,
    )
    args = parser.parse_args()
    if args.dataset_ref != DATASET_REF:
        parser.error(f"--dataset-ref must remain frozen to {DATASET_REF}")
    if args.notebook_ref != NOTEBOOK_REF:
        parser.error(f"--notebook-ref must remain frozen to {NOTEBOOK_REF}")
    if args.free_running_max_new_anchors < 1:
        parser.error("--free-running-max-new-anchors must be positive")
    return args


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output = args.output.resolve()
    try:
        output.relative_to(repo_root)
    except ValueError:
        pass
    else:
        raise ValueError("bundle output must live outside the clean source repository")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite bundle: {output}")
    output.mkdir(parents=True)
    try:
        base_root = args.base_root.resolve()
        target_root = args.target_root.resolve()
        normalization_source = args.normalization.resolve()
        launcher_source = args.launcher.resolve()
        expected_launcher = repo_root / "notebooks/kaggle/run_relation_v2_p100_bundle.py"
        if launcher_source != expected_launcher or not launcher_source.is_file():
            raise ValueError("launcher must be the accepted Relation V2 P100 bundle entrypoint")
        if sha256_file(normalization_source) != NORMALIZATION_SHA256:
            raise ValueError("input normalization differs from the frozen train-only artifact")

        source, source_inventory = build_source_release(repo_root, output)
        base_identity = dataset_identity(base_root, BASE_AUTHORITY)
        target_identity = dataset_identity(target_root, TARGET_AUTHORITY)
        base_inventory, base_storage = build_data_inventory(base_root, output, prefix="base")
        target_inventory, target_storage = build_data_inventory(
            target_root, output, prefix="target"
        )

        launcher = output / "run_relation_v2_p100_bundle.py"
        normalization = output / "multires_event_v1_input_normalization.json"
        shutil.copy2(launcher_source, launcher)
        shutil.copy2(normalization_source, normalization)
        archived_launcher_rows = [
            row
            for row in source_inventory["files"]
            if row["path"]
            == "Trauma-Predict/notebooks/kaggle/run_relation_v2_p100_bundle.py"
        ]
        if (
            len(archived_launcher_rows) != 1
            or archived_launcher_rows[0]["sha256"] != sha256_file(launcher)
        ):
            raise RuntimeError(
                "mounted launcher bytes differ from the launcher inside the source release"
            )
        manifest = {
            "schema": BUNDLE_SCHEMA,
            "dataset_ref": DATASET_REF,
            "notebook_ref": NOTEBOOK_REF,
            "route": ROUTE,
            "run_name": RUN_NAME,
            "model_parameter_count": MODEL_PARAMETER_COUNT,
            "hardware": {
                "accelerator": "NVIDIA Tesla P100",
                "required_cuda_devices": 1,
                "required_device_name_substring": "P100",
                "world_size": 1,
                "precision": "fp16",
                "per_device_train_batch_size": 64,
            },
            "hosted": {
                "training_stop_steps": list(HOSTED_TRAINING_STOPS),
                "free_running_max_new_anchors": int(
                    args.free_running_max_new_anchors
                ),
                "prior_output_restore": "required_if_a_prior_notebook_output_exists",
                "missing_or_invalid_prior_output_policy": "fail_closed",
            },
            "relation_contract": {
                "bundle_sha256": RELATION_BUNDLE_SHA256,
                "target_target_edges": 52,
                "input_target_edges": 39,
                "edge_specific_parameters": 91,
            },
            "data": {
                "base": {**base_identity, "inventory": base_inventory},
                "target": {**target_identity, "inventory": target_inventory},
            },
            "source": source,
            "launcher": {"path": launcher.name, "sha256": sha256_file(launcher)},
            "input_normalization": {
                "path": normalization.name,
                "sha256": sha256_file(normalization),
            },
            "payload_summary": {
                "base_bytes": base_storage["bytes"],
                "target_bytes": target_storage["bytes"],
                "logical_dataset_files": base_storage["files"] + target_storage["files"],
                "direct_mounted_dataset_files": (
                    base_storage["direct_files"] + target_storage["direct_files"]
                ),
                "small_packed_dataset_files": (
                    base_storage["packed_files"] + target_storage["packed_files"]
                ),
                "small_packed_uncompressed_bytes": (
                    base_storage["packed_bytes"] + target_storage["packed_bytes"]
                ),
                "source_files": int(source_inventory["file_count"]),
                "bulk_patient_payload_copy_inside_notebook": False,
                "bulk_patient_payload_extraction_inside_notebook": False,
                "small_payload_pack_extraction_inside_notebook": True,
                "network_source_checkout": False,
            },
        }
        write_json(output / "run_bundle_manifest.json", manifest)
        write_json(
            output / "dataset-metadata.json",
            {
                "title": "Trauma Predict Relation V2 P100 R9 Run Bundle",
                "id": DATASET_REF,
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            },
        )
        top_level_files = sum(path.is_file() for path in output.iterdir())
        if top_level_files > MAX_KAGGLE_TOP_LEVEL_FILES:
            raise ValueError(
                "bundle exceeds the Kaggle top-level file budget: "
                f"{top_level_files} > {MAX_KAGGLE_TOP_LEVEL_FILES}"
            )
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    print(
        "RELATION_V2_P100_BUNDLE_BUILT "
        f"path={output} "
        f"manifest_sha256={sha256_file(output / 'run_bundle_manifest.json')} "
        f"dataset_ref={DATASET_REF}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
