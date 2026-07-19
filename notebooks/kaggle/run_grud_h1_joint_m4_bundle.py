from __future__ import annotations

import hashlib
import json
import os
import selectors
import shutil
import subprocess
import sys
import tarfile
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping


SCIENCE_SCHEMA = "trauma_predict.grud_h1_joint_m4_p100_bundle.v1"
SCIENCE_DATASET_REF = "vanila111/trauma-predict-grud-h1-joint-m4-v2-bundle"
RUNTIME_BUNDLE_SCHEMA = "trauma_predict.multires_event_v2_relation_v2_p100_bundle.v3"
RUNTIME_DATASET_REF = "vanila111/trauma-predict-relation-v2-p100-r9-bundle"
RUNTIME_SCHEMA = "trauma_predict.p100_torch_runtime_wheelhouse.v1"
RUNTIME_CONTRACT_SHA256 = "aada1dee4ee21e02fd5c81ae97d441c38e72d770eec5398932ee295d08f8f2cc"
RUNTIME_INVENTORY_SHA256 = "8063e83b243589e26c353d335fd5137505bfa90b2d5aa0b1226c15fd810120a1"
RUNTIME_TORCH_VERSION = "2.10.0+cu126"
RUNTIME_CUDA_VERSION = "12.6"
RUNTIME_CUDA_ARCH = "sm_60"
RUNTIME_ARCHIVE_NAME = "p100_torch_2_10_cu126_cp312_wheelhouse.blob"
RUNTIME_ROOT = Path("/kaggle/temp/grud_v2_p100_runtime_py312_cu126")
WHEELHOUSE_ROOT = Path("/kaggle/temp/grud_v2_p100_wheelhouse_cp312_cu126")
DATA_ROOT = Path("/kaggle/temp/grud_v2_science_data")
SOURCE_ROOT = Path("/kaggle/temp/grud_v2_source")
OUTPUT_ROOT = Path("/kaggle/working/p100_grud_h1_joint_m4_v2")
LOG_ROOT = Path("/kaggle/working/logs")

H1_LOCKS = {
    "dataset_manifest.json": "2d30bdd75071f50b1631639087c2338e69ae346ec1facad13c6a8285e70288cf",
    "sample_manifest.csv": "6762897d5f516dc3442a7a206bc3bf19c3e43e32a2444f2807a475d3db61412b",
    "h1_event_templates.json": "de7628958ef80a7ca01d9a5ed7bb590bbbcd3590df3ef79d068711fd01fa554a",
}
TARGET_LOCKS = {
    "dataset_manifest.json": "6c4e1e300686195fb2c58bfcbd74df6c7cb905d7031985cb7a7624d5c7061f1e",
    "sample_manifest.csv": "df5eedcee0abf7d09fea86572db471047bdaa82dc28b14dc8bbf0dac0e32dd0e",
}
NORMALIZATION_SHA256 = "80b277662fdcfd8758b85b8ad74aad739bdb7c6a68d7d1e80a9e1c3e684fe03a"


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


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _safe_relative(value: Any, label: str) -> Path:
    relative = Path(str(value or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} is not a safe relative path")
    return relative


def _resolve_file(bundle: Path, row: Mapping[str, Any], label: str) -> Path:
    relative = _safe_relative(row.get("path"), f"{label}.path")
    path = bundle / relative
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is absent: {path}")
    if path.stat().st_size != int(row.get("size_bytes", -1)):
        raise ValueError(f"{label} size differs from its manifest")
    if sha256_file(path) != str(row.get("sha256") or ""):
        raise ValueError(f"{label} hash differs from its manifest")
    return path


def _find_manifest(
    root: Path,
    name: str,
    *,
    schema: str,
    dataset_ref: str,
) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.rglob(name)):
        if path.is_symlink() or not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") == schema and payload.get("dataset_ref") == dataset_ref:
            matches.append((path.parent.resolve(), payload))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one pre-bound {dataset_ref} Input, found {len(matches)}"
        )
    return matches[0]


def _extract_regular_ustar(archive: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"extraction destination already exists: {destination}")
    destination.mkdir(parents=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"archive member escapes destination: {member.name}") from exc
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(f"archive permits regular files only: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read archive member: {member.name}")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def bind_science_input(input_root: str | Path = "/kaggle/input") -> dict[str, str]:
    bundle, manifest = _find_manifest(
        Path(input_root),
        "grud_v2_bundle_manifest.json",
        schema=SCIENCE_SCHEMA,
        dataset_ref=SCIENCE_DATASET_REF,
    )
    if (
        manifest.get("route") != "grud_h1_to_joint_m4_v2"
        or manifest.get("run_name") != "p100_grud_h1_joint_m4_v2"
        or manifest.get("fresh_start") is not True
        or int(manifest.get("target_step", -1)) != 4000
        or manifest.get("forced_stop") is not False
    ):
        raise ValueError("science bundle run contract differs from the fresh 4000-step baseline")
    data_archive = _resolve_file(
        bundle,
        _mapping(manifest.get("science_data"), "science_data"),
        "science data archive",
    )
    source_archive = _resolve_file(
        bundle,
        _mapping(manifest.get("source_release"), "source_release"),
        "source release archive",
    )
    launcher = _resolve_file(
        bundle,
        _mapping(manifest.get("launcher"), "launcher"),
        "manifest-bound launcher",
    )
    if launcher.resolve() != Path(__file__).resolve():
        raise ValueError("executed launcher is not the manifest-bound mounted launcher")

    _extract_regular_ustar(data_archive, DATA_ROOT)
    _extract_regular_ustar(source_archive, SOURCE_ROOT)
    for relative, expected in H1_LOCKS.items():
        path = DATA_ROOT / "h1" / relative
        if sha256_file(path) != expected:
            raise ValueError(f"extracted H1 authority differs: {relative}")
    for relative, expected in TARGET_LOCKS.items():
        path = DATA_ROOT / "target" / relative
        if sha256_file(path) != expected:
            raise ValueError(f"extracted target authority differs: {relative}")
    normalization = DATA_ROOT / "normalization/grud_h1_normalization.json"
    if normalization.is_symlink() or not normalization.is_file():
        raise FileNotFoundError("science bundle lacks the frozen train-only normalization")
    if sha256_file(normalization) != NORMALIZATION_SHA256:
        raise ValueError("extracted normalization differs from the train-subject contract")
    print(
        "GRUD_V2_INPUT_OK "
        f"science_bundle={bundle} runtime_input={RUNTIME_DATASET_REF}",
        flush=True,
    )
    print(
        "GRUD_V2_DATA_OK samples=50350 train=37734 val=6309 test=6307 "
        "channels=118 history=H1 target_blocks=6 target_fields=29 factors=414",
        flush=True,
    )
    return {
        "bundle": str(bundle),
        "data_root": str(DATA_ROOT),
        "source_root": str(SOURCE_ROOT),
        "normalization": str(normalization),
    }


def _extract_runtime_wheels(
    archive: Path,
    destination: Path,
    rows: list[dict[str, Any]],
) -> list[Path]:
    expected = {str(row["path"]): row for row in rows}
    if len(expected) != len(rows):
        raise ValueError("runtime inventory has duplicate wheel names")
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:") as handle:
        members = handle.getmembers()
        if {member.name for member in members} != set(expected) or len(members) != len(rows):
            raise ValueError("runtime archive members differ from the frozen inventory")
        for member in members:
            row = expected[member.name]
            target = (destination / member.name).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise ValueError("runtime archive member escapes destination") from exc
            if (
                not member.isfile()
                or Path(member.name).name != member.name
                or member.size != int(row["size_bytes"])
            ):
                raise ValueError(f"invalid runtime member: {member.name}")
            source = handle.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read runtime member: {member.name}")
            digest = hashlib.sha256()
            with source, target.open("xb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != row["sha256"]:
                raise ValueError(f"runtime wheel hash differs: {member.name}")
    return [destination / str(row["path"]) for row in rows]


def install_p100_runtime(input_root: str | Path = "/kaggle/input") -> dict[str, str]:
    bundle, manifest = _find_manifest(
        Path(input_root),
        "run_bundle_manifest.json",
        schema=RUNTIME_BUNDLE_SCHEMA,
        dataset_ref=RUNTIME_DATASET_REF,
    )
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"P100 runtime requires Python 3.12, found {sys.version}")
    runtime = _mapping(manifest.get("runtime"), "runtime")
    contract_row = _mapping(runtime.get("contract"), "runtime.contract")
    if contract_row.get("sha256") != RUNTIME_CONTRACT_SHA256:
        raise ValueError("runtime contract hash differs from the frozen P100 lock")
    contract_path = _resolve_file(bundle, contract_row, "runtime contract")
    contract = _mapping(json.loads(contract_path.read_text(encoding="utf-8")), "contract")
    expected = {
        "schema": RUNTIME_SCHEMA,
        "python_abi": "cp312",
        "torch_version": RUNTIME_TORCH_VERSION,
        "cuda_version": RUNTIME_CUDA_VERSION,
        "required_cuda_arch": RUNTIME_CUDA_ARCH,
        "inventory_sha256": RUNTIME_INVENTORY_SHA256,
    }
    for key, value in expected.items():
        if runtime.get(key) != value or contract.get(key) != value:
            raise ValueError(f"runtime scalar differs for {key}")
    rows = runtime.get("files")
    if not isinstance(rows, list) or len(rows) != 28 or rows != contract.get("files"):
        raise ValueError("runtime wheel inventory differs from the frozen contract")
    inventory = {**expected, "files": rows}
    inventory.pop("inventory_sha256")
    if sha256_payload(inventory) != RUNTIME_INVENTORY_SHA256:
        raise ValueError("runtime inventory digest differs from the frozen contract")
    archive_row = _mapping(runtime.get("archive"), "runtime.archive")
    if archive_row.get("path") != RUNTIME_ARCHIVE_NAME:
        raise ValueError("runtime archive name differs from the frozen contract")
    archive = _resolve_file(bundle, archive_row, "P100 runtime archive")
    print(f"GRUD_V2_P100_RUNTIME_ARCHIVE_OK {archive}", flush=True)

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    install_log = LOG_ROOT / "grud_v2_runtime_install.log"
    if RUNTIME_ROOT.exists():
        raise FileExistsError(f"isolated runtime unexpectedly exists: {RUNTIME_ROOT}")
    wheel_paths = _extract_runtime_wheels(archive, WHEELHOUSE_ROOT, rows)
    temporary = RUNTIME_ROOT.with_name(f".{RUNTIME_ROOT.name}.install-{os.getpid()}")
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--no-input",
        "--no-compile",
        "--no-cache-dir",
        "--disable-pip-version-check",
        "--target",
        str(temporary),
        *[str(path) for path in wheel_paths],
    ]
    with install_log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    marker = {
        "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
        "runtime_inventory_sha256": RUNTIME_INVENTORY_SHA256,
        "torch_version": RUNTIME_TORCH_VERSION,
    }
    (temporary / "RUNTIME_READY.json").write_text(
        json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(RUNTIME_ROOT)
    shutil.rmtree(WHEELHOUSE_ROOT)

    environment = os.environ.copy()
    inherited = environment.get("PYTHONPATH", "").strip()
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(RUNTIME_ROOT), inherited) if item
    )
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PIP_NO_INDEX"] = "1"
    environment["TRAUMA_PREDICT_RUNTIME_SITE_PACKAGES"] = str(RUNTIME_ROOT)
    environment["TRAUMA_PREDICT_RUNTIME_LOCK_SHA256"] = RUNTIME_CONTRACT_SHA256
    smoke = (
        "import torch; "
        "assert str(torch.__version__)=='2.10.0+cu126'; "
        "assert str(torch.version.cuda)=='12.6'; "
        "assert torch.cuda.is_available() and torch.cuda.device_count()==1; "
        "assert 'P100' in torch.cuda.get_device_name(0).upper(); "
        "assert tuple(torch.cuda.get_device_capability(0))==(6,0); "
        "assert 'sm_60' in torch.cuda.get_arch_list(); "
        "x=torch.nn.Linear(8,8,device='cuda',dtype=torch.float16); "
        "y=x(torch.ones(2,8,device='cuda',dtype=torch.float16)).float().square().mean(); "
        "y.backward(); torch.cuda.synchronize()"
    )
    subprocess.run([sys.executable, "-c", smoke], env=environment, check=True)
    print(
        "GRUD_V2_P100_RUNTIME_OK torch=2.10.0+cu126 cuda=12.6 gpu=P100 arch=sm_60",
        flush=True,
    )
    return environment


def _validate_training_output(
    output_root: Path = OUTPUT_ROOT,
) -> tuple[Path, Path, str]:
    manifest_path = output_root / "training_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError("completed training lacks training_manifest.json")
    manifest = _mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")), "training manifest"
    )
    if (
        manifest.get("schema_version")
        != "trauma_predict.grud_h1_v2_training_manifest.v1"
        or manifest.get("route") != "grud_h1_to_joint_m4_v2"
        or manifest.get("status") != "SUCCEEDED"
        or int(manifest.get("completed_step", -1)) != 4000
    ):
        raise ValueError("training manifest does not close the 4000-step contract")

    selected_row = _mapping(
        manifest.get("selected_checkpoint"), "training_manifest.selected_checkpoint"
    )
    selected_step = int(selected_row.get("step", -1))
    selected_relative = _safe_relative(
        selected_row.get("path"), "training_manifest.selected_checkpoint.path"
    )
    if (
        selected_step < 1
        or selected_step > 4000
        or selected_relative.as_posix()
        != f"checkpoint-{selected_step}/checkpoint.pt"
    ):
        raise ValueError("selected checkpoint identity differs from its manifest")
    selected = output_root / selected_relative
    selected_hash = str(selected_row.get("sha256") or "")
    if (
        selected.is_symlink()
        or not selected.is_file()
        or sha256_file(selected) != selected_hash
    ):
        raise ValueError("selected checkpoint is absent or differs from the training manifest")

    final_row = _mapping(
        manifest.get("step_4000_checkpoint"),
        "training_manifest.step_4000_checkpoint",
    )
    final_relative = _safe_relative(
        final_row.get("path"), "training_manifest.step_4000_checkpoint.path"
    )
    if final_relative.as_posix() != "checkpoint-4000/checkpoint.pt":
        raise ValueError("step-4000 checkpoint identity differs from its manifest")
    final_checkpoint = output_root / final_relative
    if (
        final_checkpoint.is_symlink()
        or not final_checkpoint.is_file()
        or sha256_file(final_checkpoint) != str(final_row.get("sha256") or "")
    ):
        raise ValueError("step-4000 checkpoint is absent or differs from the training manifest")
    return selected, manifest_path, selected_hash


def run_training(
    science_state: Mapping[str, str],
    runtime_environment: Mapping[str, str],
) -> None:
    source_root = Path(science_state["source_root"])
    data_root = Path(science_state["data_root"])
    script = source_root / "notebooks/kaggle/train_grud_h1_joint_m4_v2.py"
    config = source_root / "configs/train/p100_grud_h1_joint_m4_v2.yaml"
    if script.is_symlink() or not script.is_file() or not config.is_file():
        raise FileNotFoundError("source release lacks the formal GRU-D training entry")
    environment = dict(runtime_environment)
    inherited = environment.get("PYTHONPATH", "").strip()
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(source_root / "src"), inherited) if item
    )
    environment["PYTHONUNBUFFERED"] = "1"
    environment["TRAUMA_PREDICT_GRUD_H1_ROOT"] = str(data_root / "h1")
    environment["TRAUMA_PREDICT_V2_TARGET_ROOT"] = str(data_root / "target")
    environment["TRAUMA_PREDICT_GRUD_NORMALIZATION_PATH"] = str(
        science_state["normalization"]
    )
    environment["TRAUMA_PREDICT_OUTPUT_ROOT"] = "/kaggle/working"
    environment["TRAUMA_PREDICT_REPO_ROOT"] = str(source_root)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    train_log = LOG_ROOT / "grud_v2_training.log"
    command = [
        sys.executable,
        str(script),
        "--repo-root",
        str(source_root),
        "--config",
        str(config),
    ]
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("training process has no stdout pipe")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    tail: deque[str] = deque(maxlen=80)
    started = time.monotonic()
    last_heartbeat = started
    latest_step = 0
    with train_log.open("w", encoding="utf-8") as log_handle:
        while process.poll() is None:
            ready = selector.select(timeout=5.0)
            for key, _ in ready:
                line = key.fileobj.readline()
                if not line:
                    continue
                log_handle.write(line)
                log_handle.flush()
                stripped = line.rstrip("\n")
                tail.append(stripped)
                if stripped.startswith("GRUD_V2_"):
                    print(stripped, flush=True)
                for token in stripped.split():
                    if token.startswith("step=") and token[5:].isdigit():
                        latest_step = max(latest_step, int(token[5:]))
            now = time.monotonic()
            if now - last_heartbeat >= 300:
                print(
                    "GRUD_V2_HEARTBEAT "
                    f"phase=train elapsed_s={int(now-started)} step={latest_step} "
                    f"log_bytes={train_log.stat().st_size}",
                    flush=True,
                )
                last_heartbeat = now
        for line in process.stdout:
            log_handle.write(line)
            stripped = line.rstrip("\n")
            tail.append(stripped)
            if stripped.startswith("GRUD_V2_"):
                print(stripped, flush=True)
    returncode = process.wait()
    if returncode:
        print("GRUD_V2_FAILED", f"returncode={returncode}", *tail, sep="\n", flush=True)
        raise RuntimeError("GRU-D joint-M4 training failed")

    selected, manifest_path, expected_hash = _validate_training_output()
    print(
        "GRUD_V2_EXPORT_OK "
        f"checkpoint={selected} sha256={expected_hash} manifest={manifest_path}",
        flush=True,
    )
    print("GRUD_V2_NOTEBOOK_FINISHED status=SUCCEEDED step=4000", flush=True)


def main() -> None:
    state = bind_science_input()
    environment = install_p100_runtime()
    run_training(state, environment)


if __name__ == "__main__":
    main()
