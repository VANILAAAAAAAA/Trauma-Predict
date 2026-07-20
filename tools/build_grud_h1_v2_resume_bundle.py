from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Mapping


BUNDLE_SCHEMA = "trauma_predict.grud_h1_joint_m4_resume_bundle.v1"
SOURCE_SCHEMA = "trauma_predict.grud_h1_joint_m4_resume_source_release.v1"
ROUTE = "grud_h1_to_joint_m4_v2"
RUN_NAME = "p100_grud_h1_joint_m4_v2_resume_2500"
DATASET_REF = "vanila111/trauma-grud-h1-m4-v2-resume-2500-bundle"
NOTEBOOK_REF = "vanila111/trauma-predict-grud-h1-joint-m4-v2-resume-2500"
SCIENCE_DATASET_REF = "vanila111/trauma-predict-grud-h1-joint-m4-v2-bundle"
RUNTIME_DATASET_REF = "vanila111/trauma-predict-relation-v2-p100-r9-bundle"
SOURCE_NOTEBOOK_REF = "vanila111/trauma-predict-grud-h1-joint-m4-v2"
SOURCE_NOTEBOOK_VERSION = 4
SOURCE_SESSION_OUTPUT_ID = 336631710
RESUME_STEP = 2500
TARGET_STEP = 4000
RESUME_RNG_SEED = 979_216_224
CHECKPOINT_SHA256 = "ba5da75fe63808374916fd270f899e45f3a9c0c3452c85fdbe6edc5dfb233054"
CHECKPOINT_MANIFEST_SHA256 = "28dac8094956b407cb8721a6e6b98f171a81867de79569f8ed121bef62021221"
SOURCE_METRICS_SHA256 = "422ba862a026c7c920cab2fcf93327b5365ae33e1decef16f3e9d992f1c9c72e"
SOURCE_TRAINING_MANIFEST_SHA256 = "21ec36a674d897fdbe9a565cec293a3823c0de3dda8fa9f5bec38e51882698cd"
SOURCE_ARCHIVE_NAME = "grud_h1_joint_m4_resume_source.blob"
STATE_ARCHIVE_NAME = "grud_h1_joint_m4_resume_2500_state.blob"

REQUIRED_SOURCE_PATHS = {
    "configs/dataset/grud_h1_joint_m4_v2_c4.yaml",
    "configs/model/grud_h1_joint_m4_v2.yaml",
    "configs/train/p100_grud_h1_joint_m4_v2_resume_2500.yaml",
    "notebooks/kaggle/run_grud_h1_joint_m4_resume_bundle.py",
    "notebooks/kaggle/train_grud_h1_joint_m4_v2_resume.py",
    "notebooks/kaggle/trauma_predict_grud_h1_joint_m4_v2_resume_2500.ipynb",
    "src/trauma_predict/training/grud_h1_v2.py",
    "tools/build_grud_h1_v2_resume_bundle.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_row(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def _add_regular_file(
    handle: tarfile.TarFile,
    source: Path,
    archive_name: str,
) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"archive source must be a regular file: {source}")
    member = tarfile.TarInfo(archive_name)
    member.size = source.stat().st_size
    member.mode = 0o444
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    member.mtime = 0
    with source.open("rb") as source_handle:
        handle.addfile(member, source_handle)


def _add_bytes(handle: tarfile.TarFile, archive_name: str, content: bytes) -> None:
    member = tarfile.TarInfo(archive_name)
    member.size = len(content)
    member.mode = 0o444
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    member.mtime = 0
    handle.addfile(member, io.BytesIO(content))


def build_source_archive(output: Path, repo_root: Path) -> dict[str, Any]:
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError("resume source bundle requires a clean committed worktree")
    commit = _git(repo_root, "rev-parse", "HEAD")
    raw = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo_root, capture_output=True, check=True
    ).stdout
    relatives = [Path(item.decode("utf-8")) for item in raw.split(b"\0") if item]
    tracked = {path.as_posix() for path in relatives}
    missing = sorted(REQUIRED_SOURCE_PATHS - tracked)
    if missing:
        raise RuntimeError(f"resume source release lacks required paths: {missing}")
    archive = output / SOURCE_ARCHIVE_NAME
    inventory: list[dict[str, Any]] = []
    with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as handle:
        for relative in sorted(relatives, key=lambda item: item.as_posix()):
            source = repo_root / relative
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"tracked source is not a regular file: {source}")
            row = {
                "path": relative.as_posix(),
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
            inventory.append(row)
            _add_regular_file(handle, source, relative.as_posix())
        _add_bytes(
            handle,
            "SOURCE_RELEASE.json",
            json_bytes(
                {
                    "schema": SOURCE_SCHEMA,
                    "git_commit": commit,
                    "files": inventory,
                    "inventory_sha256": sha256_payload(inventory),
                }
            ),
        )
    return {
        **file_row(archive),
        "format": "deterministic_ustar",
        "git_commit": commit,
        "file_count": len(inventory),
        "inventory_sha256": sha256_payload(inventory),
    }


def _recovery_paths(recovery_root: Path) -> dict[str, Path]:
    run = recovery_root / "p100_grud_h1_joint_m4_v2"
    return {
        "checkpoint": run / "checkpoint-2500/checkpoint.pt",
        "checkpoint_manifest": run / "checkpoint-2500/manifest.json",
        "source_metrics": run / "metrics.jsonl",
        "source_training_manifest": run / "training_manifest.json",
        "training_log": recovery_root / "logs/grud_v2_training.log",
        "runtime_log": recovery_root / "logs/grud_v2_runtime_install.log",
        "kernel_log": recovery_root / "trauma-predict-grud-h1-joint-m4-v2.log",
    }


def build_state_archive(output: Path, recovery_root: Path) -> dict[str, Any]:
    paths = _recovery_paths(recovery_root)
    required_hashes = {
        "checkpoint": CHECKPOINT_SHA256,
        "checkpoint_manifest": CHECKPOINT_MANIFEST_SHA256,
        "source_metrics": SOURCE_METRICS_SHA256,
        "source_training_manifest": SOURCE_TRAINING_MANIFEST_SHA256,
    }
    for key, expected in required_hashes.items():
        path = paths[key]
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"recovered {key} differs from the cancelled v4 output")
    for key in ("training_log", "runtime_log", "kernel_log"):
        path = paths[key]
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"recovery evidence is absent: {path}")

    members = {
        "checkpoint-2500/checkpoint.pt": paths["checkpoint"],
        "checkpoint-2500/manifest.json": paths["checkpoint_manifest"],
        "source/metrics.jsonl": paths["source_metrics"],
        "source/training_manifest.json": paths["source_training_manifest"],
        "source/logs/grud_v2_training.log": paths["training_log"],
        "source/logs/grud_v2_runtime_install.log": paths["runtime_log"],
        "source/logs/kaggle_session.log": paths["kernel_log"],
    }
    archive = output / STATE_ARCHIVE_NAME
    inventory: list[dict[str, Any]] = []
    with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as handle:
        for name, source in sorted(members.items()):
            inventory.append(
                {
                    "path": name,
                    "size_bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                }
            )
            _add_regular_file(handle, source, name)
    return {
        **file_row(archive),
        "format": "deterministic_ustar",
        "files": inventory,
        "inventory_sha256": sha256_payload(inventory),
    }


def build_bundle(
    *,
    repo_root: Path,
    recovery_root: Path,
    output: Path,
) -> Mapping[str, Any]:
    repo_root = repo_root.resolve()
    recovery_root = recovery_root.resolve()
    output = output.resolve()
    if any(
        output == root or output in root.parents or root in output.parents
        for root in (repo_root, recovery_root)
    ):
        raise ValueError("resume bundle output overlaps a source authority")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"resume bundle output is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source = build_source_archive(output, repo_root)
    state = build_state_archive(output, recovery_root)
    launcher_source = repo_root / "notebooks/kaggle/run_grud_h1_joint_m4_resume_bundle.py"
    launcher = output / launcher_source.name
    shutil.copy2(launcher_source, launcher)
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "route": ROUTE,
        "run_name": RUN_NAME,
        "dataset_ref": DATASET_REF,
        "notebook_ref": NOTEBOOK_REF,
        "science_dataset_ref": SCIENCE_DATASET_REF,
        "runtime_dataset_ref": RUNTIME_DATASET_REF,
        "source_notebook_ref": SOURCE_NOTEBOOK_REF,
        "source_notebook_version": SOURCE_NOTEBOOK_VERSION,
        "source_session_output_id": SOURCE_SESSION_OUTPUT_ID,
        "source_last_observed_step": 2900,
        "discarded_uncheckpointed_steps": [2501, 2900],
        "resume_step": RESUME_STEP,
        "target_step": TARGET_STEP,
        "new_optimizer_steps": TARGET_STEP - RESUME_STEP,
        "sampler_continuity": "reconstructed_epoch_and_microbatch_cursor",
        "rng_continuity": "deterministic_reset_not_bitwise_equivalent",
        "rng_seed": RESUME_RNG_SEED,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        "source_metrics_sha256": SOURCE_METRICS_SHA256,
        "source_training_manifest_sha256": SOURCE_TRAINING_MANIFEST_SHA256,
        "source_release": source,
        "resume_state": state,
        "launcher": file_row(launcher),
    }
    (output / "grud_v2_resume_bundle_manifest.json").write_bytes(json_bytes(manifest))
    (output / "dataset-metadata.json").write_bytes(
        json_bytes(
            {
                "id": DATASET_REF,
                "title": "Trauma Predict GRU-D Resume 2500 Bundle",
                "isPrivate": True,
                "licenses": [{"name": "other"}],
            }
        )
    )
    print(
        "GRUD_V2_RESUME_BUNDLE_OK "
        f"output={output} checkpoint_step={RESUME_STEP} "
        f"target_step={TARGET_STEP} source_commit={source['git_commit']}",
        flush=True,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_bundle(
        repo_root=args.repo_root,
        recovery_root=args.recovery_root,
        output=args.output,
    )


if __name__ == "__main__":
    main()
