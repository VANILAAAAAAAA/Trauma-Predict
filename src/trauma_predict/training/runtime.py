from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trauma_predict.data.preflight import ArtifactPreflightResult
from trauma_predict.training.checkpoints import sorted_checkpoints


def maybe_cap_records(records: list[dict[str, Any]], cap: Any) -> list[dict[str, Any]]:
    if cap in (None, "", 0):
        return records
    cap_int = int(cap)
    if cap_int < 1:
        raise ValueError("sample caps must be positive")
    return records[:cap_int]


def latest_checkpoint(output_dir: Path) -> str | None:
    checkpoints = sorted_checkpoints(output_dir)
    return str(checkpoints[-1]) if checkpoints else None


def quarantine_rng_state_files(checkpoint: str | None) -> list[str]:
    if not checkpoint:
        return []
    checkpoint_path = Path(checkpoint)
    paths = [checkpoint_path / "rng_state.pth"]
    paths.extend(sorted(checkpoint_path.glob("rng_state_*.pth")))

    quarantined: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        target = _unused_quarantine_path(path)
        try:
            path.rename(target)
        except FileNotFoundError:
            continue
        quarantined.append(str(target))
    return quarantined


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_environment_snapshot(path: Path, preflight: ArtifactPreflightResult) -> None:
    payload = {
        "created_at": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "preflight": preflight.to_dict(),
        "git": _git_snapshot(Path.cwd()),
        "packages": _package_snapshot(),
        "torch_cuda": _torch_cuda_snapshot(),
        "nvidia_smi": _command_lines(["nvidia-smi", "-L"]),
        "pip_freeze": _command_lines([sys.executable, "-m", "pip", "freeze"]),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _unused_quarantine_path(path: Path) -> Path:
    base = path.with_name(f"{path.name}.ignored_for_torch_weights_only")
    if not base.exists():
        return base
    index = 1
    while True:
        candidate = path.with_name(f"{path.name}.ignored_for_torch_weights_only.{index}")
        if not candidate.exists():
            return candidate
        index += 1


def _git_snapshot(cwd: Path) -> dict[str, Any]:
    return {
        "commit": _command_text(["git", "rev-parse", "HEAD"], cwd=cwd),
        "branch": _command_text(["git", "branch", "--show-current"], cwd=cwd),
        "status_short": _command_lines(["git", "status", "--short"], cwd=cwd),
    }


def _package_snapshot() -> dict[str, str | None]:
    packages = ("torch", "transformers", "accelerate", "tokenizers", "huggingface_hub")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            module = __import__(package)
        except Exception:
            versions[package] = None
            continue
        versions[package] = str(getattr(module, "__version__", "unknown"))
    return versions


def _torch_cuda_snapshot() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    return {
        "torch": str(getattr(torch, "__version__", "")),
        "cuda_version": str(getattr(torch.version, "cuda", "")),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


def _command_text(command: list[str], cwd: Path | None = None) -> str | None:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _command_lines(command: list[str], cwd: Path | None = None) -> list[str]:
    text = _command_text(command, cwd=cwd)
    return [] if text is None else text.splitlines()
