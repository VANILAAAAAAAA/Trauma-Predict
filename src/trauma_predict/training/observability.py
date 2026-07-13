from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def rank() -> int:
    return int(os.environ.get("RANK", "0") or 0)


def is_rank_zero() -> bool:
    return rank() == 0


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    if not is_rank_zero():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


class LossAccumulator:
    """Accumulate actual weighted loss values for one train or evaluation interval."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = defaultdict(float)
        self._weights: dict[str, float] = defaultdict(float)

    def update(self, total: float, parts: Mapping[str, float] | None = None, *, weight: float = 1.0) -> None:
        if weight <= 0:
            return
        self._sums["total"] += float(total) * weight
        self._weights["total"] += weight
        for name, value in (parts or {}).items():
            self._sums[str(name)] += float(value) * weight
            self._weights[str(name)] += weight

    def update_aggregates(
        self,
        total_numerator: float,
        total_denominator: float,
        parts: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        """Accumulate additive numerators/denominators supplied by the loss contract."""
        if total_denominator > 0:
            self._sums["total"] += float(total_numerator)
            self._weights["total"] += float(total_denominator)
        for name, (numerator, denominator) in (parts or {}).items():
            if denominator <= 0:
                continue
            self._sums[str(name)] += float(numerator)
            self._weights[str(name)] += float(denominator)

    def state(self) -> dict[str, dict[str, float]]:
        return {
            "sums": dict(self._sums),
            "weights": dict(self._weights),
        }

    def load_state(self, payload: Mapping[str, Mapping[str, float]]) -> None:
        self.reset()
        for name, value in payload.get("sums", {}).items():
            self._sums[str(name)] = float(value)
        for name, value in payload.get("weights", {}).items():
            self._weights[str(name)] = float(value)

    def summary(self) -> dict[str, float]:
        return {
            name: self._sums[name] / self._weights[name]
            for name in sorted(self._sums)
            if self._weights[name] > 0
        }

    def reset(self) -> None:
        self._sums.clear()
        self._weights.clear()


def emit_loss(kind: str, step: int, summary: Mapping[str, float], metrics_path: Path) -> None:
    if kind not in {"TRAIN", "EVAL"}:
        raise ValueError(f"unknown loss kind: {kind}")
    if "total" not in summary:
        raise ValueError("loss summary lacks total")
    if not is_rank_zero():
        return
    print(f"{kind}_LOSS step={int(step)} total={float(summary['total']):.6f}", flush=True)
    append_jsonl(metrics_path, {
        "created_at": utc_now(),
        "event": f"{kind.lower()}_loss",
        "step": int(step),
        "loss": {key: float(value) for key, value in summary.items()},
    })


@contextmanager
def heartbeat(label: str, log_path: Path, *, seconds: int = 300) -> Iterator[None]:
    """Print a bounded rank-zero heartbeat while a hosted subprocess is active."""

    stop = threading.Event()
    started = time.monotonic()

    def run() -> None:
        while not stop.wait(max(1, int(seconds))):
            if not is_rank_zero():
                continue
            elapsed = int(time.monotonic() - started)
            size = log_path.stat().st_size if log_path.exists() else 0
            print(f"{label}_HEARTBEAT elapsed_s={elapsed} log_bytes={size}", flush=True)

    thread = threading.Thread(target=run, name=f"{label.lower()}-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)


def next_attempt_dir(run_dir: Path) -> Path:
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in logs.glob("attempt-*"):
        try:
            indices.append(int(path.name.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    attempt = logs / f"attempt-{max(indices, default=0) + 1:04d}"
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt
