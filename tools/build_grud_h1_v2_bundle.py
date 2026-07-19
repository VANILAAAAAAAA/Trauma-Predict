from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


BUNDLE_SCHEMA = "trauma_predict.grud_h1_joint_m4_p100_bundle.v1"
SOURCE_SCHEMA = "trauma_predict.grud_h1_joint_m4_source_release.v1"
ROUTE = "grud_h1_to_joint_m4_v2"
RUN_NAME = "p100_grud_h1_joint_m4_v2"
DATASET_REF = "vanila111/trauma-predict-grud-h1-joint-m4-v2-bundle"
NOTEBOOK_REF = "vanila111/trauma-predict-grud-h1-joint-m4-v2"
H1_DATASET_ID = "grud_h1_baseline_c4_20260717_v1"
TARGET_DATASET_ID = "multires_event_m4_target_v2_c4_full_20260714_r9"

H1_LOCKS = {
    "dataset_manifest.json": "2d30bdd75071f50b1631639087c2338e69ae346ec1facad13c6a8285e70288cf",
    "sample_manifest.csv": "6762897d5f516dc3442a7a206bc3bf19c3e43e32a2444f2807a475d3db61412b",
    "h1_event_templates.json": "de7628958ef80a7ca01d9a5ed7bb590bbbcd3590df3ef79d068711fd01fa554a",
}
TARGET_LOCKS = {
    "dataset_manifest.json": "6c4e1e300686195fb2c58bfcbd74df6c7cb905d7031985cb7a7624d5c7061f1e",
    "sample_manifest.csv": "df5eedcee0abf7d09fea86572db471047bdaa82dc28b14dc8bbf0dac0e32dd0e",
}
EXPECTED_COUNTS = {"samples": 50350, "train": 37734, "val": 6309, "test": 6307}
RUNTIME_DATASET_REF = "vanila111/trauma-predict-relation-v2-p100-r9-bundle"
NORMALIZATION_SHA256 = (
    "80b277662fdcfd8758b85b8ad74aad739bdb7c6a68d7d1e80a9e1c3e684fe03a"
)

REQUIRED_SOURCE_PATHS = {
    "configs/dataset/grud_h1_joint_m4_v2_c4.yaml",
    "configs/model/grud_h1_joint_m4_v2.yaml",
    "configs/train/p100_grud_h1_joint_m4_v2.yaml",
    "notebooks/kaggle/run_grud_h1_joint_m4_bundle.py",
    "notebooks/kaggle/train_grud_h1_joint_m4_v2.py",
    "notebooks/kaggle/trauma_predict_grud_h1_joint_m4_v2.ipynb",
    "src/trauma_predict/training/grud_h1_v2.py",
    "tools/build_grud_h1_v2_bundle.py",
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


def file_row(path: Path, *, relative: str | None = None) -> dict[str, Any]:
    return {
        "path": relative or path.name,
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


def _regular_files(root: Path, prefixes: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for prefix in prefixes:
        path = root / prefix
        if path.is_symlink():
            raise ValueError(f"scientific artifact cannot contain symlink: {path}")
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_symlink():
                    raise ValueError(f"scientific artifact cannot contain symlink: {child}")
                if child.is_file():
                    files.append(child)
        else:
            raise FileNotFoundError(path)
    return sorted(set(files), key=lambda item: item.relative_to(root).as_posix())


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


def _verify_lock(root: Path, locks: Mapping[str, str], label: str) -> None:
    for relative, expected in locks.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"{label} authority is absent: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"{label} {relative} hash differs: {observed} != {expected}")


def _validate_counts(h1_root: Path) -> None:
    counts: Counter[str] = Counter()
    sample_ids: set[str] = set()
    with (h1_root / "sample_manifest.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row.get("sample_id") or "")
            split = str(row.get("split") or "")
            if not sample_id or sample_id in sample_ids or split not in {"train", "val", "test"}:
                raise ValueError("H1 sample manifest identity is invalid")
            sample_ids.add(sample_id)
            counts[split] += 1
    observed = {
        "samples": len(sample_ids),
        "train": counts["train"],
        "val": counts["val"],
        "test": counts["test"],
    }
    if observed != EXPECTED_COUNTS:
        raise ValueError(f"H1 sample counts differ: {observed} != {EXPECTED_COUNTS}")


def build_data_archive(
    output: Path,
    h1_root: Path,
    target_root: Path,
    normalization_path: Path,
) -> dict[str, Any]:
    _verify_lock(h1_root, H1_LOCKS, "H1")
    _verify_lock(target_root, TARGET_LOCKS, "target")
    _validate_counts(h1_root)
    if normalization_path.is_symlink() or not normalization_path.is_file():
        raise FileNotFoundError(normalization_path)
    if sha256_file(normalization_path) != NORMALIZATION_SHA256:
        raise ValueError("GRU-D normalization differs from the frozen train-subject artifact")

    h1_files = _regular_files(
        h1_root,
        (
            "dataset_manifest.json",
            "sample_manifest.csv",
            "h1_event_templates.json",
            "SUCCEEDED",
            "contracts",
            "h1_shards",
        ),
    )
    target_files = _regular_files(
        target_root,
        (
            "dataset_manifest.json",
            "sample_manifest.csv",
            "SUCCEEDED",
            "contracts",
            "target_shards",
        ),
    )
    archive = output / "grud_h1_joint_m4_science_data.tar"
    with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as handle:
        for source in h1_files:
            relative = source.relative_to(h1_root).as_posix()
            _add_regular_file(handle, source, f"h1/{relative}")
        for source in target_files:
            relative = source.relative_to(target_root).as_posix()
            _add_regular_file(handle, source, f"target/{relative}")
        _add_regular_file(
            handle,
            normalization_path,
            "normalization/grud_h1_normalization.json",
        )
    return {
        **file_row(archive),
        "format": "deterministic_ustar",
        "h1_files": len(h1_files),
        "target_files": len(target_files),
        "normalization_sha256": sha256_file(normalization_path),
    }


def build_source_archive(output: Path, repo_root: Path) -> dict[str, Any]:
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError("source bundle requires a clean committed worktree")
    commit = _git(repo_root, "rev-parse", "HEAD")
    raw = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo_root, capture_output=True, check=True
    ).stdout
    relatives = [Path(item.decode("utf-8")) for item in raw.split(b"\0") if item]
    tracked = {path.as_posix() for path in relatives}
    missing = sorted(REQUIRED_SOURCE_PATHS - tracked)
    if missing:
        raise RuntimeError(f"source release lacks required paths: {missing}")
    inventory: list[dict[str, Any]] = []
    archive = output / "grud_h1_joint_m4_source.tar"
    with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as handle:
        for relative in sorted(relatives, key=lambda item: item.as_posix()):
            source = repo_root / relative
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"tracked source is not a regular file: {source}")
            inventory.append(
                {
                    "path": relative.as_posix(),
                    "size_bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                }
            )
            _add_regular_file(handle, source, relative.as_posix())
        source_manifest = {
            "schema": SOURCE_SCHEMA,
            "git_commit": commit,
            "files": inventory,
            "inventory_sha256": sha256_payload(inventory),
        }
        _add_bytes(handle, "SOURCE_RELEASE.json", json_bytes(source_manifest))
    return {
        **file_row(archive),
        "format": "deterministic_ustar",
        "git_commit": commit,
        "file_count": len(inventory),
        "inventory_sha256": sha256_payload(inventory),
    }


def build_bundle(
    *,
    repo_root: Path,
    h1_root: Path,
    target_root: Path,
    normalization_path: Path,
    output: Path,
    dataset_ref: str,
    notebook_ref: str,
) -> dict[str, Any]:
    roots = [repo_root.resolve(), h1_root.resolve(), target_root.resolve(), normalization_path.resolve()]
    output = output.resolve()
    if any(
        output == root or output in root.parents or root in output.parents
        for root in roots
    ):
        raise ValueError("bundle output overlaps an input authority")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"bundle output is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    data = build_data_archive(output, roots[1], roots[2], roots[3])
    source = build_source_archive(output, roots[0])
    launcher_source = roots[0] / "notebooks/kaggle/run_grud_h1_joint_m4_bundle.py"
    launcher = output / launcher_source.name
    shutil.copy2(launcher_source, launcher)
    launcher_row = file_row(launcher)

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "route": ROUTE,
        "run_name": RUN_NAME,
        "dataset_ref": dataset_ref,
        "notebook_ref": notebook_ref,
        "runtime_dataset_ref": RUNTIME_DATASET_REF,
        "fresh_start": True,
        "target_step": 4000,
        "forced_stop": False,
        "authority": {
            "h1_dataset_id": H1_DATASET_ID,
            "target_dataset_id": TARGET_DATASET_ID,
            "counts": EXPECTED_COUNTS,
            "h1_locks": H1_LOCKS,
            "target_locks": TARGET_LOCKS,
        },
        "science_data": data,
        "source_release": source,
        "launcher": launcher_row,
    }
    manifest_path = output / "grud_v2_bundle_manifest.json"
    manifest_path.write_bytes(json_bytes(manifest))
    metadata = {
        "id": dataset_ref,
        "title": "Trauma Predict GRU-D H1 Joint M4 V2 Bundle",
        "isPrivate": True,
        "licenses": [{"name": "other"}],
    }
    (output / "dataset-metadata.json").write_bytes(json_bytes(metadata))
    print(
        "GRUD_V2_BUNDLE_OK "
        f"output={output} data_bytes={data['size_bytes']} "
        f"source_commit={source['git_commit']}",
        flush=True,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--h1-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-ref", default=DATASET_REF)
    parser.add_argument("--notebook-ref", default=NOTEBOOK_REF)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_bundle(
        repo_root=args.repo_root,
        h1_root=args.h1_root,
        target_root=args.target_root,
        normalization_path=args.normalization,
        output=args.output,
        dataset_ref=str(args.dataset_ref),
        notebook_ref=str(args.notebook_ref),
    )


if __name__ == "__main__":
    main()
