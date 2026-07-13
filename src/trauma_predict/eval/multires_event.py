from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import torch
from torch import Tensor

from .f24_composition import derive_f24_prediction_summary, normalize_raw_value


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _f24_slots(target_contract: Any) -> tuple[Any, ...]:
    slots = getattr(target_contract, "slots", None)
    indices = getattr(target_contract, "derived_primary_f24_indices", None)
    if isinstance(target_contract, Mapping):
        slots = slots or target_contract.get("slots")
        indices = indices or target_contract.get("derived_primary_f24_indices")
    if slots is None or indices is None:
        raise ValueError("F24 evaluation requires canonical slots and derived F24 indices")
    return tuple(slots[int(index)] for index in indices)


def _query_error(
    prediction: Tensor,
    target: Tensor,
    slot: Any,
    normalizer: Any,
) -> Tensor:
    family = str(_value(slot, "loss_family"))
    if family in {"continuous", "count", "nonnegative"}:
        predicted = normalize_raw_value(prediction, slot, normalizer)
        observed = normalize_raw_value(target, slot, normalizer)
        return (predicted - observed).abs()
    if family == "ordinal":
        classes = max(int(_value(slot, "ordinal_classes", 0)), 2)
        return (prediction - target).abs() / float(classes - 1)
    if family == "duration":
        span = max(float(_value(slot, "span_hours", 24.0)), 1e-6)
        return (prediction - target).abs() / span
    if family == "binary":
        return (prediction.clamp(0.0, 1.0) - target).square()
    raise ValueError(f"unsupported F24 evaluation family {family!r}")


def _additive_parts(
    errors: Tensor,
    valid: Tensor,
    slots: tuple[Any, ...],
) -> tuple[dict[str, dict[str, Tensor]], Tensor]:
    parts: dict[str, dict[str, Tensor]] = {}
    families = sorted({str(_value(slot, "loss_family")) for slot in slots})
    for family in families:
        positions = [
            index for index, slot in enumerate(slots)
            if str(_value(slot, "loss_family")) == family
        ]
        index = torch.tensor(positions, device=errors.device)
        family_valid = valid.index_select(1, index)
        family_error = errors.index_select(1, index)
        parts[f"family/{family}"] = {
            "numerator": (family_error * family_valid.float()).sum(),
            "denominator": family_valid.sum().to(errors.dtype),
        }

    component_map: dict[tuple[int, str], list[int]] = defaultdict(list)
    for position, slot in enumerate(slots):
        component_map[
            (int(_value(slot, "field_id")), str(_value(slot, "semantic_component")))
        ].append(position)
    component_values: list[Tensor] = []
    component_masks: list[Tensor] = []
    component_fields: list[int] = []
    for (field_id, _), positions in component_map.items():
        index = torch.tensor(positions, device=errors.device)
        selected_valid = valid.index_select(1, index)
        selected_error = errors.index_select(1, index)
        denominator = selected_valid.sum(dim=1)
        component_values.append(
            (selected_error * selected_valid.float()).sum(dim=1)
            / denominator.clamp_min(1).float()
        )
        component_masks.append(denominator.gt(0))
        component_fields.append(field_id)
    component_tensor = torch.stack(component_values, dim=1)
    component_mask = torch.stack(component_masks, dim=1)

    field_values: list[Tensor] = []
    field_masks: list[Tensor] = []
    for field_id in sorted(set(component_fields)):
        positions = [
            index for index, observed_field in enumerate(component_fields)
            if observed_field == field_id
        ]
        index = torch.tensor(positions, device=errors.device)
        selected_valid = component_mask.index_select(1, index)
        selected_values = component_tensor.index_select(1, index)
        denominator = selected_valid.sum(dim=1)
        field_values.append(
            (selected_values * selected_valid.float()).sum(dim=1)
            / denominator.clamp_min(1).float()
        )
        field_masks.append(denominator.gt(0))
    field_tensor = torch.stack(field_values, dim=1)
    field_mask = torch.stack(field_masks, dim=1)
    field_denominator = field_mask.sum(dim=1)
    sample_metric = (
        (field_tensor * field_mask.float()).sum(dim=1)
        / field_denominator.clamp_min(1).float()
    )
    sample_valid = field_denominator.gt(0)
    parts["total"] = {
        "numerator": (sample_metric * sample_valid.float()).sum(),
        "denominator": sample_valid.sum().to(errors.dtype),
    }
    return parts, sample_metric


def evaluate_f24_raw(
    prediction_summary: Mapping[str, Tensor] | None,
    target_contract: Any,
    *,
    normalizer: Any | None,
    f24_target_raw_values: Tensor | None,
    f24_target_mask: Tensor | None,
) -> dict[str, Any]:
    """Evaluate derived F24 only when raw truth and train-fitted scaling exist.

    The reported total is a dimensionless component-to-field macro error.  It is
    never mixed into the H1/M4 training loss.
    """

    missing = [
        name
        for name, value in (
            ("prediction_summary", prediction_summary),
            ("normalizer", normalizer),
            ("f24_target_raw_values", f24_target_raw_values),
            ("f24_target_mask", f24_target_mask),
        )
        if value is None
    ]
    if missing:
        return {
            "status": "not_evaluated",
            "reason": "missing " + ", ".join(missing),
            "parts": {},
        }
    if not isinstance(prediction_summary, Mapping):
        raise TypeError("prediction_summary must be a mapping of raw tensor banks")
    derived_summary = derive_f24_prediction_summary(prediction_summary, target_contract)
    predictions = derived_summary["expected_raw_value"]
    target = f24_target_raw_values.float()
    mask = f24_target_mask.bool()
    if predictions.shape != target.shape or mask.shape != target.shape:
        raise ValueError("F24 predictions, raw targets, and target mask must align")
    slots = _f24_slots(target_contract)
    if target.shape[1] != len(slots):
        raise ValueError("F24 target width does not match the target contract")
    valid = mask & torch.isfinite(target) & torch.isfinite(predictions)
    errors = torch.zeros_like(predictions)
    for position, slot in enumerate(slots):
        errors[:, position] = _query_error(
            predictions[:, position],
            target[:, position],
            slot,
            normalizer,
        )
    parts, sample_metric = _additive_parts(errors, valid, slots)
    return {
        "status": "evaluated",
        "prediction_summary": derived_summary,
        "predictions": predictions,
        "prediction_mask": torch.ones_like(mask),
        "per_query_error": errors,
        "per_query_valid": valid,
        "sample_metric": sample_metric,
        "metric_numerator": parts["total"]["numerator"],
        "metric_denominator": parts["total"]["denominator"],
        "parts": parts,
    }


__all__ = ["evaluate_f24_raw"]
