from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_REF = "vanilaaaa/trauma-predict-main-route-first-train-8h-v2"
DATASET_SLUG = "trauma-predict-main-route-first-train-8h-v2"
PREFERRED_BASE_MODEL = "answerdotai/ModernBERT-base"
EXPECTED_TRAINING_STAGE = "stage_a_next_hour"
EXPECTED_SPLITS = {"train": 31980, "val": 4378, "test": 3895}
EXPECTED_SAMPLES = 40253
EXPECTED_DEPENDENCIES = {
    "transformers": "4.48.3",
    "accelerate": "0.34.2",
    "tokenizers": "0.21.4",
    "huggingface_hub": "0.36.2",
}
FAILURE_TAIL_LINES = int(os.environ.get("TRAUMA_PREDICT_FAILURE_TAIL_LINES", "60"))

KAGGLE_WORKING = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
DATA_ROOT = Path(os.environ.get("TRAUMA_PREDICT_DATA_ROOT", KAGGLE_WORKING / DATASET_SLUG))
DOWNLOAD_ROOT = Path(os.environ.get(
    "TRAUMA_PREDICT_DOWNLOAD_ROOT",
    KAGGLE_WORKING / "kaggle_dataset_download",
))
OUTPUT_ROOT = Path(os.environ.get(
    "TRAUMA_PREDICT_OUTPUT_ROOT",
    KAGGLE_WORKING / "trauma-predict-runs",
))

T4X2_TRAIN_CONFIG = "configs/train/t4x2_stage_a_hour.yaml"
P100_TRAIN_CONFIG = "configs/train/p100_stage_a_hour.yaml"
SMOKE_CONFIG = "configs/train/t4x2_stage_a_hour_smoke.yaml"

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
os.environ.setdefault("ACCELERATE_LOG_LEVEL", "error")


def main() -> None:
    print("repo_root", REPO_ROOT)
    print("data_root", DATA_ROOT)
    print("output_root", OUTPUT_ROOT)
    gpu_count = detect_gpu_count()
    if gpu_count < 1:
        raise RuntimeError("No Kaggle GPU is visible. Enable a GPU accelerator first.")
    train_config, run_name, nproc = select_training_route(gpu_count)
    log_dir = OUTPUT_ROOT / run_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print("gpu_count", gpu_count)
    print("selected_train_config", train_config)
    print("selected_run_name", run_name)
    print("nproc_per_node", nproc)
    print("log_dir", log_dir)

    install_requirements(log_dir)
    runtime_guard()
    prepare_data_root(log_dir)
    run_stage_a_preflight(train_config, run_name, log_dir)
    scan_token_lengths(train_config, run_name, log_dir)
    if os.environ.get("TRAUMA_PREDICT_DRY_RUN_ONLY") == "1":
        print("STAGE_A_DRY_RUN_ONLY_FINISHED")
        return
    if os.environ.get("TRAUMA_PREDICT_SKIP_SMOKE") != "1":
        run_smoke(gpu_count, log_dir)
    run_full_training(train_config, run_name, nproc, log_dir)
    summarize_and_archive(run_name, log_dir)
    print("STAGE_A_AUTOMATED_RUN_FINISHED")


def run_to_log(
    command: list[Any],
    log_path: Path,
    *,
    cwd: Path | None = REPO_ROOT,
    env: dict[str, str] | None = None,
    check: bool = True,
    status_label: str | None = None,
    stream_patterns: tuple[str, ...] = (),
    heartbeat_seconds: int = 0,
) -> subprocess.CompletedProcess[str]:
    command = [str(part) for part in command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$", " ".join(command), ">", log_path, flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None

        def read_stdout() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if stream_patterns and any(pattern in line for pattern in stream_patterns):
                    print(line, end="", flush=True)

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()
        start = time.monotonic()
        last_heartbeat = start
        while proc.poll() is None:
            if heartbeat_seconds <= 0:
                time.sleep(1)
                continue
            time.sleep(min(30, heartbeat_seconds))
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_seconds and proc.poll() is None:
                log_bytes = log_path.stat().st_size if log_path.exists() else 0
                elapsed = int(now - start)
                print(
                    f"{status_label or 'COMMAND'}_HEARTBEAT elapsed_s={elapsed} log_bytes={log_bytes}",
                    flush=True,
                )
                last_heartbeat = now
        returncode = proc.wait()
        reader.join(timeout=30)

    if returncode != 0:
        if check:
            print_failure_tail(log_path)
            raise subprocess.CalledProcessError(returncode, command)
        print(f"{status_label or 'COMMAND'}_NONZERO returncode={returncode} log={log_path}")
    elif status_label:
        print(f"{status_label}_OK log={log_path}")
    return subprocess.CompletedProcess(command, returncode)


def print_failure_tail(log_path: Path) -> None:
    print(f"FAILURE_LOG_TAIL log={log_path}")
    for line in read_tail(log_path, FAILURE_TAIL_LINES):
        print(line)


def read_tail(path: Path, line_count: int) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(errors="replace").splitlines()
    return lines[-line_count:]


def repo_env() -> dict[str, str]:
    env = os.environ.copy()
    env["TRAUMA_PREDICT_DATA_ROOT"] = str(DATA_ROOT)
    env["TRAUMA_PREDICT_OUTPUT_ROOT"] = str(OUTPUT_ROOT)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["TRANSFORMERS_VERBOSITY"] = "warning"
    env["ACCELERATE_LOG_LEVEL"] = "error"
    return env


def is_kaggle() -> bool:
    return Path("/kaggle").exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def detect_gpu_count() -> int:
    result = subprocess.run(
        ["nvidia-smi", "-L"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout.strip())
    elif result.stderr:
        print(result.stderr.strip())
    return sum(1 for line in result.stdout.splitlines() if line.startswith("GPU "))


def select_training_route(gpu_count: int) -> tuple[str, str, int]:
    if gpu_count >= 2:
        return T4X2_TRAIN_CONFIG, "t4x2_stage_a_hour", 2
    return P100_TRAIN_CONFIG, "p100_stage_a_hour", 1


def install_requirements(log_dir: Path) -> None:
    if not is_kaggle() and os.environ.get("TRAUMA_PREDICT_ALLOW_LOCAL_INSTALL") != "1":
        print("SKIP_PIP_INSTALL_OUTSIDE_KAGGLE")
        return
    if os.environ.get("TRAUMA_PREDICT_SKIP_INSTALL") == "1":
        print("SKIP_PIP_INSTALL")
        return

    if os.environ.get("TRAUMA_PREDICT_SKIP_VISION_UNINSTALL") != "1":
        run_to_log(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "timm"],
            log_dir / "pip_uninstall_vision.log",
            check=False,
            status_label="PIP_UNINSTALL_VISION",
        )
    run_to_log(
        [sys.executable, "-m", "pip", "install", "-q", "-r", REPO_ROOT / "requirements-kaggle.txt"],
        log_dir / "pip_install.log",
        status_label="PIP_INSTALL",
    )
    pip_check = run_to_log(
        [sys.executable, "-m", "pip", "check"],
        log_dir / "pip_check.log",
        check=False,
        status_label="PIP_CHECK",
    )
    if pip_check.returncode != 0:
        print(f"PIP_CHECK_NON_BLOCKING log={log_dir / 'pip_check.log'}")


def runtime_guard() -> None:
    import torch
    import accelerate
    import huggingface_hub
    import tokenizers
    import transformers
    from transformers import AutoConfig, Trainer, TrainingArguments

    actual = {
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "tokenizers": tokenizers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
    }
    print(
        "runtime",
        json.dumps({
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_count": torch.cuda.device_count(),
            "versions": actual,
        }, sort_keys=True),
    )
    if actual != EXPECTED_DEPENDENCIES:
        raise RuntimeError(
            f"Hugging Face stack mismatch: expected {EXPECTED_DEPENDENCIES}, got {actual}"
        )
    if torch.__version__.startswith("2.12.1+cu130"):
        raise RuntimeError("Kaggle session has pip-upgraded torch 2.12.1+cu130; restart the session.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is not available to PyTorch.")
    config = AutoConfig.from_pretrained(PREFERRED_BASE_MODEL)
    if getattr(config, "model_type", None) != "modernbert":
        raise RuntimeError(f"Expected ModernBERT config, got {getattr(config, 'model_type', None)}")
    if int(getattr(config, "max_position_embeddings", 0) or 0) < 4096:
        raise RuntimeError("Preferred encoder does not expose the required 4096-token window.")
    _ = Trainer
    _ = TrainingArguments
    x = torch.ones(1, device="cuda")
    if float(x.item()) != 1.0:
        raise RuntimeError("CUDA tensor smoke check failed.")
    print("main_route_runtime_guard OK")


def prepare_data_root(log_dir: Path) -> None:
    if os.environ.get("TRAUMA_PREDICT_USE_EXISTING_DATA_ROOT") == "1":
        if not is_prepared_artifact(DATA_ROOT):
            raise FileNotFoundError(f"Existing data root is not prepared: {DATA_ROOT}")
        print("using_existing_data_root", DATA_ROOT)
        return

    source_root = explicit_source_root() or attached_dataset_root() or download_dataset_root(log_dir)
    print("dataset_source", source_root)

    if DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    copy_metadata_files(source_root, DATA_ROOT)
    for split in ["train", "val", "test"]:
        shards = reconstruct_split(source_root, split)
        print(split, len(shards))

    manifest = json.loads((DATA_ROOT / "dataset_manifest.json").read_text(encoding="utf-8"))
    print("dataset_manifest", json.dumps({
        "dataset_id": manifest.get("dataset_id"),
        "counts": manifest.get("counts"),
    }, sort_keys=True))


def explicit_source_root() -> Path | None:
    value = os.environ.get("TRAUMA_PREDICT_SOURCE_DATA_ROOT")
    if not value:
        return None
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def attached_dataset_root() -> Path | None:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return None
    candidates = [input_root / DATASET_SLUG]
    candidates.extend(sorted(input_root.glob(f"{DATASET_SLUG}*")))
    candidates.extend(sorted(input_root.glob("*")))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "dataset_manifest.json").exists() or (candidate / "train.zip").exists():
            return candidate
    return None


def download_dataset_root(log_dir: Path) -> Path:
    configure_kaggle_credentials_from_secrets()
    if DOWNLOAD_ROOT.exists():
        shutil.rmtree(DOWNLOAD_ROOT)
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    run_to_log(
        ["kaggle", "datasets", "download", "-d", DATASET_REF, "-p", DOWNLOAD_ROOT, "--unzip"],
        log_dir / "kaggle_dataset_download.log",
        status_label="KAGGLE_DATASET_DOWNLOAD",
    )
    return DOWNLOAD_ROOT


def configure_kaggle_credentials_from_secrets() -> None:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    try:
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        os.environ.setdefault("KAGGLE_USERNAME", client.get_secret("KAGGLE_USERNAME"))
        os.environ.setdefault("KAGGLE_KEY", client.get_secret("KAGGLE_KEY"))
    except Exception:
        return


def is_prepared_artifact(root: Path) -> bool:
    if not (root / "dataset_manifest.json").exists():
        return False
    if not (root / "sample_manifest.csv").exists():
        return False
    return all(any((root / split).glob("shard-*.jsonl.gz")) for split in ["train", "val", "test"])


def copy_metadata_files(source_root: Path, data_root: Path) -> None:
    for name in ["dataset_manifest.json", "sample_manifest.csv"]:
        src = source_root / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, data_root / name)
    for name in ["patient_split.csv", "anchor_plan.csv"]:
        src = source_root / name
        if src.exists():
            shutil.copy2(src, data_root / name)


def reconstruct_split(source_root: Path, split: str) -> list[Path]:
    split_dir = DATA_ROOT / split
    split_dir.mkdir(parents=True, exist_ok=True)
    zip_path = source_root / f"{split}.zip"
    if zip_path.exists():
        extract_zip_members(zip_path, split_dir)
    source_split_dir = source_root / split
    if source_split_dir.exists():
        for src in sorted(source_split_dir.glob("shard-*.jsonl*")):
            shutil.copy2(src, split_dir / src.name)
    for plain in sorted(split_dir.glob("*.jsonl")):
        gz_path = split_dir / f"{plain.name}.gz"
        with plain.open("rb") as src, gzip.open(gz_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        plain.unlink()
    shards = sorted(split_dir.glob("shard-*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"No {split} shards under {split_dir}")
    return shards


def extract_zip_members(zip_path: Path, split_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            name = Path(member).name
            if not name:
                continue
            with archive.open(member) as src, (split_dir / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)


def run_stage_a_preflight(train_config: str, run_name: str, log_dir: Path) -> None:
    run_to_log(
        [
            sys.executable,
            "notebooks/kaggle/train_kaggle.py",
            "--config",
            train_config,
            "--dry-run",
        ],
        log_dir / "stage_a_preflight.log",
        env=repo_env(),
        status_label="STAGE_A_PREFLIGHT",
    )

    snapshot = json.loads(
        (OUTPUT_ROOT / run_name / "run_config_snapshot.json").read_text(encoding="utf-8")
    )
    config = snapshot["config"]
    if config["training_stage"] != EXPECTED_TRAINING_STAGE:
        raise RuntimeError(config["training_stage"])
    if config["model"]["base_model"] != PREFERRED_BASE_MODEL:
        raise RuntimeError(config["model"]["base_model"])
    if config["training"].get("resume") is not True:
        raise RuntimeError("Formal Stage A config must be resumable.")
    active_losses = config["training"]["active_losses"]
    loss_weights = config["training"]["loss_weights"]
    expected_active_losses = {
        "next_hour_values": True,
        "next_hour_vent": False,
        "next24_domain": False,
        "next24_binary": False,
        "next24_multiclass": False,
    }
    if active_losses != expected_active_losses:
        raise RuntimeError(active_losses)
    inactive_keys = ["next_hour_vent", "next24_domain", "next24_binary", "next24_multiclass"]
    for key in inactive_keys:
        if float(loss_weights[key]) != 0.0:
            raise RuntimeError(loss_weights)
    print("STAGE_A_CONFIG_OK")

    summary = json.loads(
        (OUTPUT_ROOT / run_name / "data_preflight_summary.json").read_text(encoding="utf-8")
    )
    print("preflight_summary", json.dumps({
        "manifest_samples": summary["manifest_samples"],
        "sample_manifest_rows": summary["sample_manifest_rows"],
        "shard_rows": summary["shard_rows"],
        "split_counts": summary["split_counts"],
    }, sort_keys=True))
    expected = {
        "manifest_samples": EXPECTED_SAMPLES,
        "sample_manifest_rows": EXPECTED_SAMPLES,
        "shard_rows": EXPECTED_SAMPLES,
        "split_counts": EXPECTED_SPLITS,
    }
    observed = {key: summary[key] for key in expected}
    if observed != expected:
        raise RuntimeError(f"preflight mismatch: expected {expected}, got {observed}")
    print("STAGE_A_ARTIFACT_PREFLIGHT_OK")


def scan_token_lengths(train_config: str, run_name: str, log_dir: Path) -> None:
    output_json = OUTPUT_ROOT / run_name / "token_length_summary.json"
    run_to_log(
        [
            sys.executable,
            "notebooks/kaggle/scan_token_lengths.py",
            "--dataset-config",
            "configs/dataset/first_train.yaml",
            "--train-config",
            train_config,
            "--output-json",
            output_json,
        ],
        log_dir / "token_length_scan.log",
        env=repo_env(),
        status_label="TOKEN_LENGTH_SCAN",
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    compact = {
        split: {
            "count": values["count"],
            "p95": values["p95"],
            "p99": values["p99"],
            "max": values["max"],
        }
        for split, values in payload["by_split"].items()
    }
    print("token_length_summary", json.dumps({
        "base_model": payload["base_model"],
        "max_input_tokens": payload["max_input_tokens"],
        "failure_count": payload["failure_count"],
        "by_split": compact,
    }, sort_keys=True))
    print("TOKEN_LENGTH_SCAN_OK")


def run_smoke(gpu_count: int, log_dir: Path) -> None:
    smoke_nproc = min(gpu_count, 2)
    run_to_log(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(smoke_nproc),
            "notebooks/kaggle/train_kaggle.py",
            "--config",
            SMOKE_CONFIG,
        ],
        log_dir / "stage_a_smoke.log",
        env=repo_env(),
        status_label="STAGE_A_SMOKE_RUN",
    )
    print("STAGE_A_SMOKE_RUN_OK")


def run_full_training(train_config: str, run_name: str, nproc: int, log_dir: Path) -> None:
    run_dir = OUTPUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_log = log_dir / "torchrun_train.log"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(nproc),
        "notebooks/kaggle/train_kaggle.py",
        "--config",
        train_config,
    ]
    stream_patterns = (
        "TRAIN_LOSS=",
        "EVAL_LOSS=",
        "training_status=",
        "run_config_snapshot=",
        "metrics_jsonl=",
        "training_result=",
    )
    run_to_log(
        command,
        train_log,
        env=repo_env(),
        status_label="STAGE_A_FULL_RUN",
        stream_patterns=stream_patterns,
        heartbeat_seconds=int(os.environ.get("TRAUMA_PREDICT_HEARTBEAT_SECONDS", "300")),
    )
    print("STAGE_A_TRAINING_FINISHED")


def summarize_and_archive(run_name: str, log_dir: Path) -> None:
    run_dir = OUTPUT_ROOT / run_name
    print("run_dir", run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        lines = metrics_path.read_text(errors="replace").splitlines()
        print("metrics_rows", len(lines))

    metadata_files = sorted(run_dir.glob("checkpoint-*/training_stage_metadata.json"))
    print("checkpoint_metadata_count", len(metadata_files))
    if metadata_files:
        print("latest_checkpoint_metadata", metadata_files[-1])

    result_path = run_dir / "training_result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        print("training_result", json.dumps({
            "checkpoint": result.get("checkpoint"),
            "final_model": result.get("final_model"),
            "prediction_path": result.get("prediction_path"),
        }, sort_keys=True))

    archive = Path(os.environ.get(
        "TRAUMA_PREDICT_OUTPUT_ARCHIVE",
        KAGGLE_WORKING / f"{run_name}_outputs.tar.gz",
    ))
    if run_dir.exists():
        if archive.exists():
            archive.unlink()
        run_to_log(
            ["tar", "-czf", archive, "-C", KAGGLE_WORKING, f"trauma-predict-runs/{run_name}"],
            log_dir / "archive_outputs.log",
            status_label="ARCHIVE_OUTPUTS",
        )
        print("archive", archive)


if __name__ == "__main__":
    main()
