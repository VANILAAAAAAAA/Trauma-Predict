from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_REF = "vanilaaaa/trauma-predict-main-route-first-train-8h-v2"
DATASET_SLUG = "trauma-predict-main-route-first-train-8h-v2"
PREFERRED_BASE_MODEL = "answerdotai/ModernBERT-base"
EXPECTED_TRAINING_STAGE = "stage_a1_residual"
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

TRAIN_CONFIG = "configs/train/t4x2_stage_a1_residual.yaml"
SMOKE_CONFIG = "configs/train/t4x2_stage_a1_residual_smoke.yaml"
RUN_NAME = "t4x2_stage_a1_residual"
SMOKE_RUN_NAME = "t4x2_stage_a1_residual_smoke"

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
os.environ.setdefault("ACCELERATE_LOG_LEVEL", "error")


def main() -> None:
    print("repo_root", REPO_ROOT)
    print("data_root", DATA_ROOT)
    print("output_root", OUTPUT_ROOT)
    gpu_count = detect_gpu_count()
    if gpu_count < 2:
        raise RuntimeError("Stage A.1 T4x2 config requires 2 visible Kaggle GPUs.")
    log_dir = OUTPUT_ROOT / RUN_NAME / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print("gpu_count", gpu_count)
    print("selected_train_config", TRAIN_CONFIG)
    print("selected_run_name", RUN_NAME)
    print("nproc_per_node", 2)
    print("log_dir", log_dir)

    install_requirements(log_dir)
    runtime_guard()
    prepare_data_root(log_dir)
    checkpoint = discover_stage_a_checkpoint()
    print("stage_a_checkpoint_dir", checkpoint)
    run_preflight(TRAIN_CONFIG, RUN_NAME, checkpoint, log_dir)
    scan_token_lengths(TRAIN_CONFIG, RUN_NAME, checkpoint, log_dir)
    if os.environ.get("TRAUMA_PREDICT_DRY_RUN_ONLY") == "1":
        print("STAGE_A1_DRY_RUN_ONLY_FINISHED")
        return
    if os.environ.get("TRAUMA_PREDICT_SKIP_SMOKE") != "1":
        run_smoke(checkpoint, log_dir)
    run_full_training(checkpoint, log_dir)
    summarize_and_archive(log_dir)
    print("STAGE_A1_AUTOMATED_RUN_FINISHED")


def run_to_log(
    command: list[Any],
    log_path: Path,
    *,
    cwd: Path | None = REPO_ROOT,
    env: dict[str, str] | None = None,
    check: bool = True,
    status_label: str | None = None,
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
        for line in proc.stdout:
            log.write(line)
        returncode = proc.wait()

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
    return path.read_text(errors="replace").splitlines()[-line_count:]


def repo_env(checkpoint: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["TRAUMA_PREDICT_DATA_ROOT"] = str(DATA_ROOT)
    env["TRAUMA_PREDICT_OUTPUT_ROOT"] = str(OUTPUT_ROOT)
    if checkpoint is not None:
        env["STAGE_A_CHECKPOINT_DIR"] = str(checkpoint)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["TRANSFORMERS_VERBOSITY"] = "warning"
    env["ACCELERATE_LOG_LEVEL"] = "error"
    return env


def is_kaggle() -> bool:
    return Path("/kaggle").exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def detect_gpu_count() -> int:
    result = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout.strip())
    elif result.stderr:
        print(result.stderr.strip())
    return sum(1 for line in result.stdout.splitlines() if line.startswith("GPU "))


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
        raise RuntimeError(f"Hugging Face stack mismatch: expected {EXPECTED_DEPENDENCIES}, got {actual}")
    if torch.__version__.startswith("2.12.1+cu130"):
        raise RuntimeError("Kaggle session has pip-upgraded torch 2.12.1+cu130; restart the session.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Stage A.1 requires two CUDA devices.")
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
    for candidate in candidates:
        if (candidate / "dataset_manifest.json").exists() or (candidate / "train.zip").exists():
            return candidate
    return None


def download_dataset_root(log_dir: Path) -> Path:
    if DOWNLOAD_ROOT.exists():
        shutil.rmtree(DOWNLOAD_ROOT)
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    run_to_log(
        ["kaggle", "datasets", "download", "-d", DATASET_REF, "-p", DOWNLOAD_ROOT, "--unzip"],
        log_dir / "kaggle_dataset_download.log",
        status_label="KAGGLE_DATASET_DOWNLOAD",
    )
    return DOWNLOAD_ROOT


def copy_metadata_files(source_root: Path, target_root: Path) -> None:
    for name in ["dataset_manifest.json", "sample_manifest.csv", "patient_split.csv", "anchor_plan.csv"]:
        src = source_root / name
        if src.exists():
            shutil.copy2(src, target_root / name)
    if not (target_root / "dataset_manifest.json").exists():
        raise FileNotFoundError(f"dataset_manifest.json not found under {source_root}")
    if not (target_root / "sample_manifest.csv").exists():
        raise FileNotFoundError(f"sample_manifest.csv not found under {source_root}")


def reconstruct_split(source_root: Path, split: str) -> list[Path]:
    split_dir = DATA_ROOT / split
    split_dir.mkdir(parents=True, exist_ok=True)
    zip_path = source_root / f"{split}.zip"
    if zip_path.exists():
        extract_zip_members(zip_path, split_dir)
    elif (source_root / split).is_dir():
        for src in sorted((source_root / split).glob("*")):
            if src.is_file():
                shutil.copy2(src, split_dir / src.name)
    else:
        flat_files = sorted(source_root.glob(f"{split}*.jsonl*"))
        for src in flat_files:
            shutil.copy2(src, split_dir / src.name)
    shards = sorted(split_dir.glob("*.jsonl.gz"))
    plain_shards = sorted(split_dir.glob("*.jsonl"))
    for plain in plain_shards:
        gz_path = plain.with_suffix(plain.suffix + ".gz")
        with plain.open("rb") as src, gzip.open(gz_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        plain.unlink()
    shards = sorted(split_dir.glob("*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"no {split} shards found after reconstruction from {source_root}")
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


def is_prepared_artifact(path: Path) -> bool:
    return (
        (path / "dataset_manifest.json").exists()
        and (path / "sample_manifest.csv").exists()
        and all((path / split).is_dir() for split in ("train", "val", "test"))
    )


def discover_stage_a_checkpoint() -> Path:
    explicit = os.environ.get("STAGE_A_CHECKPOINT_DIR")
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        if not _has_weight_file(path):
            raise FileNotFoundError(f"STAGE_A_CHECKPOINT_DIR has no supported weight file: {path}")
        return path

    roots = [
        Path(item)
        for item in os.environ.get("TRAUMA_PREDICT_CHECKPOINT_SEARCH_ROOTS", "/kaggle/input:/kaggle/working").split(":")
        if item
    ]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for weight in root.rglob("model.safetensors"):
            candidates.append(weight.parent)
        for weight in root.rglob("main_route_model.pt"):
            candidates.append(weight.parent)
    candidates = sorted(set(candidates), key=lambda path: (_checkpoint_score(path), str(path)), reverse=True)
    if not candidates:
        raise FileNotFoundError(
            "No Stage A checkpoint was found. Attach a checkpoint Kaggle Dataset or set STAGE_A_CHECKPOINT_DIR."
        )
    return candidates[0]


def _has_weight_file(path: Path) -> bool:
    if path.is_file():
        return True
    return any((path / name).exists() for name in ("model.safetensors", "pytorch_model.bin", "main_route_model.pt"))


def _checkpoint_score(path: Path) -> tuple[int, int]:
    text = str(path).lower()
    score = 0
    if (path / "training_stage_metadata.json").exists():
        score += 4
    if "stage_a" in text or "stage-a" in text:
        score += 2
    if "stage_a1" in text or "stage-a1" in text:
        score -= 5
    step = 0
    if path.name.startswith("checkpoint-"):
        score += 1
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            step = 0
    return score, step


def run_preflight(config: str, run_name: str, checkpoint: Path, log_dir: Path) -> None:
    run_to_log(
        [sys.executable, "notebooks/kaggle/train_kaggle.py", "--config", config, "--dry-run"],
        log_dir / "stage_a1_preflight.log",
        env=repo_env(checkpoint),
        status_label="STAGE_A1_PREFLIGHT",
    )
    snapshot_path = OUTPUT_ROOT / run_name / "run_config_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    config_payload = snapshot["config"]
    if config_payload["training_stage"] != EXPECTED_TRAINING_STAGE:
        raise RuntimeError(config_payload["training_stage"])
    if config_payload["model"]["base_model"] != PREFERRED_BASE_MODEL:
        raise RuntimeError(config_payload["model"]["base_model"])
    if int(config_payload["model"]["hour_field_hidden"]) != 64:
        raise RuntimeError(config_payload["model"]["hour_field_hidden"])
    if config_payload["model"]["next_hour_value_mode"] != "h0_residual":
        raise RuntimeError(config_payload["model"]["next_hour_value_mode"])
    if config_payload["training"]["active_losses"]["next_hour_vent"]:
        raise RuntimeError("Stage A.1 must keep next_hour_vent inactive")
    if int(config_payload["training"]["max_steps"]) != 2000:
        raise RuntimeError(config_payload["training"]["max_steps"])
    print("STAGE_A1_CONFIG_OK")
    preflight = json.loads((OUTPUT_ROOT / run_name / "data_preflight_summary.json").read_text(encoding="utf-8"))
    print("preflight_summary", json.dumps({
        "manifest_samples": preflight["manifest_samples"],
        "sample_manifest_rows": preflight["sample_manifest_rows"],
        "shard_rows": preflight["shard_rows"],
        "split_counts": preflight["split_counts"],
    }, sort_keys=True))


def scan_token_lengths(config: str, run_name: str, checkpoint: Path, log_dir: Path) -> None:
    output_json = OUTPUT_ROOT / run_name / "token_length_summary.json"
    run_to_log(
        [
            sys.executable,
            "notebooks/kaggle/scan_token_lengths.py",
            "--dataset-config",
            "configs/dataset/first_train.yaml",
            "--train-config",
            config,
            "--output-json",
            output_json,
        ],
        log_dir / "token_length_scan.log",
        env=repo_env(checkpoint),
        status_label="TOKEN_LENGTH_SCAN",
    )
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    print("token_length_summary", json.dumps({
        "base_model": summary.get("base_model"),
        "max_input_tokens": summary.get("max_input_tokens"),
        "failure_count": summary.get("failure_count"),
        "by_split": summary.get("by_split"),
    }, sort_keys=True))


def run_smoke(checkpoint: Path, log_dir: Path) -> None:
    run_to_log(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            "2",
            "notebooks/kaggle/train_kaggle.py",
            "--config",
            SMOKE_CONFIG,
        ],
        log_dir / "stage_a1_smoke.log",
        env=repo_env(checkpoint),
        status_label="STAGE_A1_SMOKE_RUN",
    )
    report_path = OUTPUT_ROOT / SMOKE_RUN_NAME / "warm_start_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print("warm_start_smoke", json.dumps({
            "loaded_key_count": report.get("loaded_key_count"),
            "skipped_reset_keys": len(report.get("skipped_reset_keys") or []),
            "skipped_shape_keys": len(report.get("skipped_shape_keys") or []),
        }, sort_keys=True))
    print("STAGE_A1_SMOKE_RUN_OK")


def run_full_training(checkpoint: Path, log_dir: Path) -> None:
    run_to_log(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            "2",
            "notebooks/kaggle/train_kaggle.py",
            "--config",
            TRAIN_CONFIG,
        ],
        log_dir / "torchrun_train.log",
        env=repo_env(checkpoint),
        status_label="STAGE_A1_FULL_RUN",
    )


def summarize_and_archive(log_dir: Path) -> None:
    run_dir = OUTPUT_ROOT / RUN_NAME
    result_path = run_dir / "training_result.json"
    if result_path.exists():
        print("training_result", result_path.read_text(encoding="utf-8"))
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        lines = metrics_path.read_text(errors="replace").splitlines()
        print("metrics_rows", len(lines))
        for line in lines[-10:]:
            print(line)
    for path in sorted(run_dir.glob("checkpoint-*/training_stage_metadata.json"))[-3:]:
        print("checkpoint_metadata", path)
    archive = KAGGLE_WORKING / f"{RUN_NAME}_outputs.tar.gz"
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(run_dir, arcname=f"trauma-predict-runs/{RUN_NAME}")
    print("archive", archive)
    print("logs", log_dir)


if __name__ == "__main__":
    main()
