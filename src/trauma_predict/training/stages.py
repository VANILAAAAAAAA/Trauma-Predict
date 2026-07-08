from __future__ import annotations

from dataclasses import dataclass
from typing import Any


LOSS_KEYS = (
    "next_hour_values",
    "next_hour_vent",
    "next24_domain",
    "next24_binary",
    "next24_multiclass",
)

NEXT_HOUR_LOSSES = ("next_hour_values", "next_hour_vent")
NEXT24_LOSSES = ("next24_domain", "next24_binary", "next24_multiclass")

STAGE_A_NEXT_HOUR = "stage_a_next_hour"
STAGE_B_NEXT24 = "stage_b_next24"
STAGE_C_ALTERNATING = "stage_c_alternating"
JOINT_BASELINE = "joint_baseline"

ALLOWED_TRAINING_STAGES = {
    STAGE_A_NEXT_HOUR,
    STAGE_B_NEXT24,
    STAGE_C_ALTERNATING,
    JOINT_BASELINE,
}

STAGE_DEFAULT_ACTIVE_LOSSES: dict[str, dict[str, bool]] = {
    STAGE_A_NEXT_HOUR: {
        "next_hour_values": True,
        "next_hour_vent": True,
        "next24_domain": False,
        "next24_binary": False,
        "next24_multiclass": False,
    },
    STAGE_B_NEXT24: {
        "next_hour_values": False,
        "next_hour_vent": False,
        "next24_domain": True,
        "next24_binary": True,
        "next24_multiclass": True,
    },
    STAGE_C_ALTERNATING: {
        "next_hour_values": True,
        "next_hour_vent": True,
        "next24_domain": True,
        "next24_binary": True,
        "next24_multiclass": True,
    },
    JOINT_BASELINE: {
        "next_hour_values": True,
        "next_hour_vent": True,
        "next24_domain": True,
        "next24_binary": True,
        "next24_multiclass": True,
    },
}


@dataclass(frozen=True)
class TrainingStageContract:
    training_stage: str
    active_losses: dict[str, bool]
    loss_weights: dict[str, float]

    @property
    def active_loss_names(self) -> list[str]:
        return [key for key in LOSS_KEYS if self.active_losses[key]]


def resolve_training_stage_contract(config: dict[str, Any]) -> TrainingStageContract:
    training_stage = str(config.get("training_stage") or "")
    if training_stage not in ALLOWED_TRAINING_STAGES:
        allowed = ", ".join(sorted(ALLOWED_TRAINING_STAGES))
        raise ValueError(f"training_stage must be one of: {allowed}")

    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("train config training must be an object")

    active_losses = _resolve_active_losses(training_stage, training.get("active_losses"))
    loss_weights = _resolve_loss_weights(training.get("loss_weights"))
    _validate_stage_loss_contract(training_stage, active_losses, loss_weights, training, config)
    return TrainingStageContract(
        training_stage=training_stage,
        active_losses=active_losses,
        loss_weights=loss_weights,
    )


def labels_for_active_losses(active_losses: dict[str, bool]) -> list[str]:
    labels: list[str] = []
    if active_losses["next_hour_values"]:
        labels.extend(["next_hour_values", "next_hour_mask"])
    if active_losses["next_hour_vent"]:
        labels.append("next_hour_vent")
    if active_losses["next24_domain"]:
        labels.append("next24_domain_labels")
    if active_losses["next24_binary"]:
        labels.append("next24_binary_labels")
    if active_losses["next24_multiclass"]:
        labels.append("next24_multiclass_labels")
    return labels


def is_next24_active(active_losses: dict[str, bool]) -> bool:
    return any(active_losses[key] for key in NEXT24_LOSSES)


def is_next_hour_active(active_losses: dict[str, bool]) -> bool:
    return any(active_losses[key] for key in NEXT_HOUR_LOSSES)


def _resolve_active_losses(training_stage: str, value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        raise ValueError("training.active_losses must explicitly declare every loss key")
    invalid = sorted(set(value) - set(LOSS_KEYS))
    missing = sorted(set(LOSS_KEYS) - set(value))
    if invalid:
        raise ValueError(f"training.active_losses contains unknown keys: {invalid}")
    if missing:
        raise ValueError(f"training.active_losses is missing keys: {missing}")
    resolved = {key: bool(value[key]) for key in LOSS_KEYS}
    expected = STAGE_DEFAULT_ACTIVE_LOSSES[training_stage]
    if resolved != expected:
        raise ValueError(
            f"training.active_losses does not match {training_stage}: "
            f"expected {expected}, got {resolved}"
        )
    return resolved


def _resolve_loss_weights(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("training.loss_weights must explicitly declare every loss key")
    invalid = sorted(set(value) - set(LOSS_KEYS))
    missing = sorted(set(LOSS_KEYS) - set(value))
    if invalid:
        raise ValueError(f"training.loss_weights contains unknown keys: {invalid}")
    if missing:
        raise ValueError(f"training.loss_weights is missing keys: {missing}")
    return {key: float(value[key]) for key in LOSS_KEYS}


def _validate_stage_loss_contract(
    training_stage: str,
    active_losses: dict[str, bool],
    loss_weights: dict[str, float],
    training: dict[str, Any],
    config: dict[str, Any],
) -> None:
    for key in LOSS_KEYS:
        if active_losses[key] and loss_weights[key] <= 0.0:
            raise ValueError(f"{training_stage} active loss {key} must have positive weight")
        if not active_losses[key] and loss_weights[key] != 0.0:
            raise ValueError(f"{training_stage} inactive loss {key} must have weight 0")

    if training_stage == STAGE_A_NEXT_HOUR:
        if any(active_losses[key] for key in NEXT24_LOSSES):
            raise ValueError("Stage A must not activate NEXT_24H losses")
        if not all(active_losses[key] for key in NEXT_HOUR_LOSSES):
            raise ValueError("Stage A must activate both NEXT_HOUR losses")
        run_name = str(config.get("run_name") or "")
        if "joint" in run_name or "full" in run_name:
            raise ValueError("Stage A run name must not contain joint/full labels")

    if training_stage == STAGE_B_NEXT24:
        checkpoint = training.get("stage_a_checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("Stage B must declare training.stage_a_checkpoint")

    if training_stage == STAGE_C_ALTERNATING:
        k_value = training.get("alternating_summary_steps")
        if int(k_value or 0) < 1:
            raise ValueError("Stage C must declare training.alternating_summary_steps >= 1")
