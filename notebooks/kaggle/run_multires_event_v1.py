from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trauma_predict.training.observability import (  # noqa: E402
    atomic_write_json,
    heartbeat,
    next_attempt_dir,
    utc_now,
)


EXPECTED_GIT_REF = "multires-event-v1-baseline-run-20260712-r3"
DATASET_REF = "vanilaaaa/trauma-predict-multires-event-v1-c4-20260712"
EXPECTED_DATASET_ID = "multires_event_v1_c4_full_20260712"
EXPECTED_DATASET_FINGERPRINT = "d58d003b6a9b2dd7c1f8d269a1867b534ea475a91118d7d4d44804bee69f9e47"
EXPECTED_COUNTS = {"samples": 50350, "train": 37734, "val": 6309, "test": 6307, "shards": 52}
EXPECTED_SHARD_COUNTS = {"train": 38, "val": 7, "test": 7}
SMOKE_CONFIG = "configs/train/t4x2_multires_event_v1_smoke.yaml"
FULL_CONFIG = "configs/train/t4x2_multires_event_v1_full.yaml"
SMOKE_RUN_NAME = "t4x2_multires_event_v1_smoke"
FULL_RUN_NAME = "t4x2_multires_event_v1_full"
KAGGLE_WORKING = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
KAGGLE_INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
OUTPUT_ROOT = Path(os.environ.get(
    "TRAUMA_PREDICT_OUTPUT_ROOT", KAGGLE_WORKING / "trauma-predict-runs"
))
PREPARED_DATA_ROOT = Path(os.environ.get(
    "TRAUMA_PREDICT_PREPARED_DATA_ROOT",
    KAGGLE_WORKING / "trauma-predict-multires-event-v1-c4-20260712",
))
DOWNLOAD_ROOT = Path(os.environ.get(
    "TRAUMA_PREDICT_DOWNLOAD_ROOT",
    KAGGLE_WORKING / "kaggle-dataset-multires-event-v1-c4-20260712",
))
FAILURE_TAIL_LINES = int(os.environ.get("TRAUMA_PREDICT_FAILURE_TAIL_LINES", "80"))
STREAM_PREFIXES = (
    "TRAIN_LOSS ",
    "EVAL_LOSS ",
    "RESUME_CHECKPOINT ",
    "MULTIRES_EVENT_PREFLIGHT_OK",
    "MULTIRES_EVENT_TRAINING_COMPLETE",
)


def main() -> None:
    print("repo_root", REPO_ROOT, flush=True)
    print("output_root", OUTPUT_ROOT, flush=True)
    verify_source_identity()
    require_t4x2_runtime()
    full_run_dir = OUTPUT_ROOT / FULL_RUN_NAME
    attempt_dir = next_attempt_dir(full_run_dir)
    print("attempt_log_dir", attempt_dir, flush=True)
    dataset_source = explicit_or_attached_dataset_root(attempt_dir)
    print("dataset_source", dataset_source, flush=True)
    print("dataset_ref", DATASET_REF, flush=True)
    dataset_root = prepare_dataset_root(dataset_source, PREPARED_DATA_ROOT, attempt_dir)
    print("prepared_dataset_root", dataset_root, flush=True)
    atomic_write_json(attempt_dir / "attempt_manifest.json", {
        "schema_version": "trauma_predict.multires_kaggle_attempt.v1",
        "started_at": utc_now(),
        "git_ref": EXPECTED_GIT_REF,
        "dataset_ref": DATASET_REF,
        "dataset_source": str(dataset_source),
        "prepared_dataset_root": str(dataset_root),
        "smoke_config": SMOKE_CONFIG,
        "full_config": FULL_CONFIG,
    })
    install_requirements(attempt_dir)
    runtime_guard()

    env = repo_env(dataset_root)
    run_to_log(
        [sys.executable, "notebooks/kaggle/train_multires_event_v1.py", "--config", FULL_CONFIG, "--dry-run"],
        attempt_dir / "preflight.log",
        env=env,
        label="PREFLIGHT",
    )
    if os.environ.get("TRAUMA_PREDICT_DRY_RUN_ONLY") == "1":
        print("MULTIRES_EVENT_DRY_RUN_ONLY_FINISHED", flush=True)
        return

    if os.environ.get("TRAUMA_PREDICT_SKIP_SMOKE") != "1":
        archive_previous_smoke_output()
        print_run_contract(SMOKE_CONFIG, attempt_dir / "smoke.log")
        run_torchrun(SMOKE_CONFIG, attempt_dir / "smoke.log", env=env, label="SMOKE")
        require_success_marker(OUTPUT_ROOT / SMOKE_RUN_NAME)
        print("MULTIRES_EVENT_SMOKE_OK", flush=True)

    print_run_contract(FULL_CONFIG, attempt_dir / "full.log")
    run_torchrun(FULL_CONFIG, attempt_dir / "full.log", env=env, label="FULL")
    require_success_marker(full_run_dir)
    validate_final_outputs(full_run_dir)
    atomic_write_json(attempt_dir / "attempt_complete.json", {
        "schema_version": "trauma_predict.multires_kaggle_attempt_complete.v1",
        "completed_at": utc_now(),
        "run_dir": str(full_run_dir),
        "run_manifest": str(full_run_dir / "run_manifest.json"),
        "success": str(full_run_dir / "SUCCESS"),
    })
    print("MULTIRES_EVENT_KAGGLE_RUN_FINISHED", flush=True)
    print("run_dir", full_run_dir, flush=True)


def find_exact_attached_dataset(input_root: Path) -> Path:
    if not input_root.is_dir():
        raise FileNotFoundError(
            f"Kaggle input root is absent: {input_root}. Attach private dataset {DATASET_REF}."
        )
    exact: list[Path] = []
    inspected: list[dict[str, Any]] = []
    for manifest_path in sorted(input_root.rglob("dataset_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        counts = manifest.get("counts", {})
        observed = {
            "samples": int(counts.get("samples", -1)),
            "train": int(counts.get("selected_by_split", {}).get("train", -1)),
            "val": int(counts.get("selected_by_split", {}).get("val", -1)),
            "test": int(counts.get("selected_by_split", {}).get("test", -1)),
            "shards": int(counts.get("completed_shards", -1)),
        }
        inspected.append({
            "root": str(manifest_path.parent),
            "dataset_id": manifest.get("dataset_id"),
            "fingerprint": manifest.get("fingerprint"),
            "counts": observed,
        })
        if (
            manifest.get("dataset_id") == EXPECTED_DATASET_ID
            and manifest.get("fingerprint") == EXPECTED_DATASET_FINGERPRINT
            and observed == EXPECTED_COUNTS
        ):
            exact.append(manifest_path.parent.resolve())
    unique = sorted(set(exact))
    if len(unique) > 1:
        raise RuntimeError(f"multiple exact multires datasets are attached; detach all but one: {unique}")
    if not unique:
        raise FileNotFoundError(
            f"no exact attached dataset matches id/fingerprint/counts; inspected={inspected}"
        )
    return unique[0]


def explicit_or_attached_dataset_root(log_dir: Path | None = None) -> Path:
    explicit = os.environ.get("TRAUMA_PREDICT_DATA_ROOT")
    if explicit:
        root = Path(explicit).resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        return root
    try:
        return find_exact_attached_dataset(KAGGLE_INPUT)
    except FileNotFoundError:
        if log_dir is None:
            raise
        return download_private_dataset(log_dir)


def download_private_dataset(log_dir: Path) -> Path:
    if DOWNLOAD_ROOT.is_dir():
        try:
            existing = find_exact_attached_dataset(DOWNLOAD_ROOT)
            if has_usable_shard_payload(existing):
                print("using_existing_download", existing, flush=True)
                return existing
            raise FileNotFoundError(
                f"existing exact manifest has no preserved shard archive/tree: {existing}"
            )
        except FileNotFoundError:
            archive = DOWNLOAD_ROOT.with_name(f"{DOWNLOAD_ROOT.name}.invalid-{os.getpid()}")
            DOWNLOAD_ROOT.rename(archive)
            print("archived_invalid_download", archive, flush=True)
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    configure_kaggle_credentials()
    run_to_log(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET_REF,
            "-p",
            DOWNLOAD_ROOT,
        ],
        log_dir / "dataset_download.log",
        env=os.environ.copy(),
        label="DATASET_DOWNLOAD",
    )
    package_archives = sorted(DOWNLOAD_ROOT.glob("*.zip"))
    if len(package_archives) != 1:
        raise RuntimeError(
            "controlled Kaggle download must produce exactly one outer dataset ZIP; "
            f"found {[path.name for path in package_archives]}"
        )
    package_root = DOWNLOAD_ROOT / "dataset-package"
    safe_extract_dataset_package(package_archives[0], package_root)
    print(
        "KAGGLE_DATASET_PACKAGE_LAYOUT",
        json.dumps(summarize_dataset_layout(package_root), sort_keys=True),
        flush=True,
    )
    downloaded = find_exact_attached_dataset(DOWNLOAD_ROOT)
    if not has_usable_shard_payload(downloaded):
        raise FileNotFoundError(
            f"downloaded package has the exact manifest but no preserved shard payload: {downloaded}"
        )
    print("DATASET_DOWNLOAD_EXACT_IDENTITY_OK", downloaded, flush=True)
    return downloaded


def safe_extract_dataset_package(archive_path: Path, destination: Path) -> int:
    """Extract one outer Kaggle package without recursively unpacking its files."""

    temporary = destination.with_name(f".{destination.name}.extract-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    count = 0
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe dataset package member: {info.filename}")
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            count += 1
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)
    print(
        "KAGGLE_DATASET_PACKAGE_EXTRACT_OK",
        json.dumps({"archive": archive_path.name, "files": count}, sort_keys=True),
        flush=True,
    )
    return count


def summarize_dataset_layout(root: Path) -> dict[str, Any]:
    """Return bounded diagnostics for Kaggle's server-side archive transforms."""

    summary: dict[str, Any] = {
        "files": 0,
        "jsonl": 0,
        "jsonl_gz": 0,
        "zip": 0,
        "top_level": {},
    }
    top_level: dict[str, int] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        summary["files"] += 1
        if path.name.endswith(".jsonl.gz"):
            summary["jsonl_gz"] += 1
        elif path.suffix == ".jsonl":
            summary["jsonl"] += 1
        elif path.suffix == ".zip":
            summary["zip"] += 1
        key = relative.parts[0]
        top_level[key] = top_level.get(key, 0) + 1
    summary["top_level"] = dict(sorted(top_level.items()))
    return summary


def configure_kaggle_credentials() -> None:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    try:
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        username = client.get_secret("KAGGLE_USERNAME")
        key = client.get_secret("KAGGLE_KEY")
    except Exception:
        return
    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key


def prepare_dataset_root(source_root: Path, destination: Path, log_dir: Path) -> Path:
    """Normalize local or Kaggle-hosted payloads into gzip split shards.

    Kaggle may expand the uploaded ``shards.zip`` and gunzip each ``.jsonl.gz``
    member during Dataset ingestion.  The hosted form is therefore commonly
    ``shards/<split>/*.jsonl`` even though the canonical artifact is gzip.
    """
    if is_prepared_dataset(destination):
        print("using_existing_prepared_dataset", destination, flush=True)
        return destination.resolve()
    shards_zip = source_root / "shards.zip"
    if not shards_zip.is_file():
        # Local validation may point directly at the canonical unzipped artifact.
        if is_prepared_dataset(source_root):
            print("using_unzipped_dataset_source", source_root, flush=True)
            return source_root.resolve()

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.prepare-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    root_files = (
        "dataset_manifest.json",
        "sample_manifest.csv",
        "subject_split.csv",
        "event_templates.json",
        "time_blocks.json",
        "SUCCEEDED",
    )
    for name in root_files:
        source = source_root / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, temporary / name)
    if shards_zip.is_file():
        shard_source_layout = "shards_zip"
        extracted = safe_extract_shards(shards_zip, temporary)
    else:
        shard_source_layout = "kaggle_hosted_extracted_split_tree"
        extracted = copy_extracted_shards(source_root, temporary)
    if extracted != EXPECTED_COUNTS["shards"]:
        raise RuntimeError(
            "dataset must provide the exact 52-shard archive or extracted split tree; "
            f"materialized {extracted}"
        )
    if destination.exists():
        archive = destination.with_name(f"{destination.name}.invalid-{os.getpid()}")
        destination.rename(archive)
        print("archived_invalid_prepared_dataset", archive, flush=True)
    temporary.replace(destination)
    atomic_write_json(log_dir / "dataset_prepare.json", {
        "schema_version": "trauma_predict.multires_dataset_prepare.v1",
        "created_at": utc_now(),
        "source_root": str(source_root),
        "destination": str(destination),
        "copied_root_files": list(root_files),
        "shard_source_layout": shard_source_layout,
        "extracted_shards": extracted,
        "skipped_archives": ["manifests.zip", "validation.zip", "audit.zip"],
    })
    return destination.resolve()


def has_usable_shard_payload(root: Path) -> bool:
    if (root / "shards.zip").is_file() or is_prepared_dataset(root):
        return True
    return {
        split: len(paths)
        for split, paths in discover_extracted_shards(root).items()
    } == EXPECTED_SHARD_COUNTS


def discover_extracted_shards(source_root: Path) -> dict[str, list[Path]]:
    discovered: dict[str, list[Path]] = {split: [] for split in EXPECTED_SHARD_COUNTS}
    candidates = set(source_root.rglob("*.jsonl.gz"))
    candidates.update(source_root.rglob("*.jsonl"))
    for source in sorted(candidates):
        relative = source.relative_to(source_root)
        parts = relative.parts
        logical_parts = tuple(part.removesuffix(".zip") for part in parts)
        if "validation" in logical_parts or "manifests" in logical_parts:
            continue
        if "shards" in logical_parts:
            shard_index = logical_parts.index("shards")
            split = logical_parts[shard_index + 1] if len(logical_parts) > shard_index + 1 else None
        else:
            split_parts = [part for part in logical_parts[:-1] if part in discovered]
            split = split_parts[0] if len(split_parts) == 1 else None
        if split in discovered and source.name.startswith(f"{split}-"):
            discovered[split].append(source)
    return discovered


def copy_extracted_shards(source_root: Path, destination: Path) -> int:
    """Restore Kaggle-hosted plain or gzip JSONL shards to canonical gzip."""

    discovered = discover_extracted_shards(source_root)

    observed = {split: len(paths) for split, paths in discovered.items()}
    if observed != EXPECTED_SHARD_COUNTS:
        raise FileNotFoundError(
            f"shards.zip is absent and extracted shard counts are {observed}; "
            f"expected {EXPECTED_SHARD_COUNTS}"
        )

    seen_names: set[tuple[str, str]] = set()
    converted_plain_jsonl = 0
    for split, paths in discovered.items():
        for source in paths:
            target_name = source.name if source.name.endswith(".jsonl.gz") else f"{source.name}.gz"
            key = (split, target_name)
            if key in seen_names:
                raise RuntimeError(f"duplicate extracted shard: {split}/{target_name}")
            seen_names.add(key)
            target = destination / "shards" / split / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == ".jsonl":
                with source.open("rb") as input_handle, target.open("wb") as raw_output:
                    with gzip.GzipFile(
                        filename="",
                        mode="wb",
                        compresslevel=1,
                        fileobj=raw_output,
                        mtime=0,
                    ) as output_handle:
                        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
                converted_plain_jsonl += 1
            else:
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
    print(
        "KAGGLE_EXTRACTED_SPLIT_TREE_OK",
        json.dumps(
            {"counts": observed, "plain_jsonl_recompressed": converted_plain_jsonl},
            sort_keys=True,
        ),
        flush=True,
    )
    return sum(observed.values())


def safe_extract_shards(archive_path: Path, destination: Path) -> int:
    count = 0
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe shards.zip member: {info.filename}")
            parts = list(member.parts)
            if "shards" in parts:
                parts = parts[parts.index("shards") + 1 :]
            if not parts:
                continue
            relative = Path(*parts)
            if relative.suffixes[-2:] != [".jsonl", ".gz"]:
                raise ValueError(f"unexpected non-shard member in shards.zip: {info.filename}")
            if relative.parts[0] not in {"train", "val", "test"}:
                raise ValueError(f"shard member lacks split directory: {info.filename}")
            target = destination / "shards" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            count += 1
    return count


def is_prepared_dataset(root: Path) -> bool:
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file() or not (root / "sample_manifest.csv").is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        manifest.get("dataset_id") != EXPECTED_DATASET_ID
        or manifest.get("fingerprint") != EXPECTED_DATASET_FINGERPRINT
    ):
        return False
    observed = {
        split: len(list((root / "shards" / split).glob("*.jsonl.gz")))
        for split in EXPECTED_SHARD_COUNTS
    }
    return observed == EXPECTED_SHARD_COUNTS


def repo_env(dataset_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["TRAUMA_PREDICT_DATA_ROOT"] = str(dataset_root)
    env["TRAUMA_PREDICT_OUTPUT_ROOT"] = str(OUTPUT_ROOT)
    env["PYTHONPATH"] = str(SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["REQUIRED_GIT_REF"] = EXPECTED_GIT_REF
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def run_torchrun(config: str, log_path: Path, *, env: dict[str, str], label: str) -> None:
    run_to_log(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "notebooks/kaggle/train_multires_event_v1.py",
            "--config",
            config,
        ],
        log_path,
        env=env,
        label=label,
    )


def print_run_contract(config_path: str, log_path: Path) -> None:
    from trauma_predict.training.config import load_yaml_config

    config = load_yaml_config(REPO_ROOT / config_path)
    training = config["training"]
    output_dir = str(config["outputs"]["output_dir"]).replace(
        "${TRAUMA_PREDICT_OUTPUT_ROOT}", str(OUTPUT_ROOT)
    )
    metrics_jsonl = str(config["outputs"]["metrics_jsonl"]).replace(
        "${TRAUMA_PREDICT_OUTPUT_ROOT}", str(OUTPUT_ROOT)
    )
    print("MULTIRES_EVENT_RUN_CONTRACT", json.dumps({
        "run_name": config["run_name"],
        "route": config["route"],
        "git_ref": EXPECTED_GIT_REF,
        "max_steps": int(training["max_steps"]),
        "logging_steps": int(training["logging_steps"]),
        "eval_steps": int(training["eval_steps"]),
        "save_steps": int(training["save_steps"]),
        "output_dir": output_dir,
        "metrics_jsonl": metrics_jsonl,
        "full_log": str(log_path),
    }, sort_keys=True), flush=True)


def run_to_log(
    command: list[Any],
    log_path: Path,
    *,
    env: dict[str, str] | None = None,
    label: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [str(part) for part in command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$", " ".join(command), ">>", log_path, flush=True)
    with log_path.open("a", encoding="utf-8") as log, heartbeat(label, log_path, seconds=300):
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            stripped = line.rstrip()
            if stripped.startswith(STREAM_PREFIXES):
                print(stripped, flush=True)
        returncode = process.wait()
    if returncode != 0:
        if check:
            print_failure_tail(log_path)
            raise subprocess.CalledProcessError(returncode, command)
        print(f"{label}_NONZERO returncode={returncode} log={log_path}", flush=True)
        return subprocess.CompletedProcess(command, returncode)
    print(f"{label}_OK log={log_path}", flush=True)
    return subprocess.CompletedProcess(command, returncode)


def install_requirements(log_dir: Path) -> None:
    if os.environ.get("TRAUMA_PREDICT_SKIP_INSTALL") == "1":
        print("SKIP_MULTIRES_PIP_INSTALL", flush=True)
        return
    run_to_log(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-multires-kaggle.txt"],
        log_dir / "pip_install.log",
        env=os.environ.copy(),
        label="PIP_INSTALL",
    )
    run_to_log(
        [sys.executable, "-m", "pip", "check"],
        log_dir / "pip_check.log",
        env=os.environ.copy(),
        label="PIP_CHECK",
        check=False,
    )


def runtime_guard() -> None:
    import torch

    payload = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
    print("runtime", json.dumps(payload, sort_keys=True), flush=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("multires_event_v1 requires two visible CUDA devices")
    for index in range(2):
        value = torch.ones(1, device=f"cuda:{index}")
        if float(value.item()) != 1.0:
            raise RuntimeError(f"CUDA tensor smoke failed on device {index}")
    print("MULTIRES_EVENT_RUNTIME_GUARD_OK", flush=True)


def require_t4x2_runtime() -> None:
    result = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout.strip(), flush=True)
    gpu_count = sum(1 for line in result.stdout.splitlines() if line.startswith("GPU "))
    if gpu_count < 2:
        raise RuntimeError(f"select Kaggle T4 x2; detected {gpu_count} GPU(s)")


def verify_source_identity() -> None:
    required = os.environ.get("REQUIRED_GIT_REF", EXPECTED_GIT_REF)
    if required != EXPECTED_GIT_REF:
        raise RuntimeError(f"REQUIRED_GIT_REF must be immutable tag {EXPECTED_GIT_REF}")
    head = _git_text("rev-parse", "HEAD")
    tagged = _git_text("rev-parse", f"{required}^{{commit}}")
    if head != tagged:
        raise RuntimeError(f"HEAD {head} does not match pinned tag {required} ({tagged})")
    print("source_identity", json.dumps({"git_ref": required, "commit": head}), flush=True)


def _git_text(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def archive_previous_smoke_output() -> None:
    smoke_dir = OUTPUT_ROOT / SMOKE_RUN_NAME
    if not smoke_dir.exists():
        return
    archive_root = OUTPUT_ROOT / "smoke-history"
    archive_root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (archive_root / f"{SMOKE_RUN_NAME}-attempt-{index:04d}").exists():
        index += 1
    destination = archive_root / f"{SMOKE_RUN_NAME}-attempt-{index:04d}"
    smoke_dir.rename(destination)
    print("archived_previous_smoke", destination, flush=True)


def require_success_marker(run_dir: Path) -> None:
    marker = run_dir / "SUCCESS"
    if not marker.is_file():
        raise FileNotFoundError(f"training subprocess returned but SUCCESS is absent: {marker}")


def validate_final_outputs(run_dir: Path) -> None:
    required = [
        "resolved_config.json",
        "source_identity.json",
        "dataset_fingerprint.json",
        "target_contract.json",
        "normalization.json",
        "runtime_environment.json",
        "metrics.jsonl",
        "best_checkpoint.json",
        "final_model/model_manifest.json",
        "val_predictions.jsonl.gz",
        "evaluation.json",
        "run_manifest.json",
        "SUCCESS",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"completed run lacks required outputs: {missing}")


def print_failure_tail(path: Path) -> None:
    print(f"FAILURE_LOG_TAIL log={path}", flush=True)
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-FAILURE_TAIL_LINES:]:
        print(line, flush=True)


if __name__ == "__main__":
    main()
