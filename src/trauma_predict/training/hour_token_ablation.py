from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any

from trauma_predict.data.main_route_contract import (
    HOUR_TOKENIZATION_FIELD_HOUR,
    HOUR_TOKENIZATION_HOUR,
    resolve_hour_tokenization,
)


ABLATION_ID = "stage_a_hour_24_vs_field_hour_168"
CONTROL_GIT_COMMIT = "5ce25c1"
CONTROL_RUN_NAME = "t4x2_stage_a_hour"
CONTROL_ARCHIVE = "kaggle_stage_a_v1_hour24_numeric_projection_20260709"


def validate_field_hour_ablation_config(
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    control_mode = resolve_hour_tokenization(control.get("model", {}).get("hour_tokenization"))
    candidate_mode = resolve_hour_tokenization(candidate.get("model", {}).get("hour_tokenization"))
    if control_mode != HOUR_TOKENIZATION_HOUR:
        raise ValueError(f"control must use hour tokenization, got {control_mode}")
    if candidate_mode != HOUR_TOKENIZATION_FIELD_HOUR:
        raise ValueError(f"candidate must use field_hour tokenization, got {candidate_mode}")
    if str(control.get("run_name")) != CONTROL_RUN_NAME:
        raise ValueError(f"control run_name must be {CONTROL_RUN_NAME}")

    normalized_control = deepcopy(control)
    normalized_candidate = deepcopy(candidate)
    normalized_control.setdefault("model", {})["hour_tokenization"] = HOUR_TOKENIZATION_HOUR
    normalized_candidate.setdefault("model", {})["hour_tokenization"] = HOUR_TOKENIZATION_HOUR
    normalized_candidate["run_name"] = normalized_control["run_name"]
    normalized_candidate["outputs"] = deepcopy(normalized_control.get("outputs"))

    if normalized_candidate != normalized_control:
        differences = _differences(normalized_control, normalized_candidate)
        raise ValueError(
            "field-hour candidate differs from Stage A v1 outside the allowed factor: "
            + "; ".join(differences[:20])
        )

    return {
        "ablation_id": ABLATION_ID,
        "control": {
            "git_commit": CONTROL_GIT_COMMIT,
            "run_name": CONTROL_RUN_NAME,
            "archive": CONTROL_ARCHIVE,
            "hour_tokenization": HOUR_TOKENIZATION_HOUR,
            "config_sha256": config_sha256(control),
        },
        "candidate": {
            "run_name": str(candidate["run_name"]),
            "hour_tokenization": HOUR_TOKENIZATION_FIELD_HOUR,
            "config_sha256": config_sha256(candidate),
        },
        "varied_factor": "model.hour_tokenization",
        "allowed_non_model_differences": [
            "run_name",
            "outputs.output_dir",
            "outputs.metrics_jsonl",
        ],
        "sample_contract_changed": False,
        "warm_start": False,
    }


def config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


def _differences(expected: Any, observed: Any, path: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(observed, dict):
        out: list[str] = []
        for key in sorted(set(expected) | set(observed)):
            child = f"{path}.{key}" if path else str(key)
            if key not in expected:
                out.append(f"{child}=unexpected")
            elif key not in observed:
                out.append(f"{child}=missing")
            else:
                out.extend(_differences(expected[key], observed[key], child))
        return out
    if isinstance(expected, list) and isinstance(observed, list):
        if expected == observed:
            return []
        return [f"{path}: expected {expected!r}, got {observed!r}"]
    if expected != observed:
        return [f"{path}: expected {expected!r}, got {observed!r}"]
    return []
