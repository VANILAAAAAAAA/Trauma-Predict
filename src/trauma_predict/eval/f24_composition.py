from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


EPS = 1e-6


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _contract_slots(target_contract: Any) -> tuple[Any, ...]:
    slots = getattr(target_contract, "slots", None)
    if slots is None and isinstance(target_contract, Mapping):
        slots = target_contract.get("slots")
    if slots is None:
        raise ValueError("F24 composition requires the canonical target slots")
    return tuple(slots)


def _active_slots(target_contract: Any) -> tuple[Any, ...]:
    slots = getattr(target_contract, "active_slots", None)
    if slots is None:
        slots = getattr(target_contract, "queries", None)
    if slots is None and isinstance(target_contract, Mapping):
        slots = target_contract.get("active_slots") or target_contract.get("queries")
    if slots is None:
        raise ValueError("F24 composition requires active direct query slots")
    return tuple(slots)


def _f24_indices(target_contract: Any) -> tuple[int, ...]:
    values = getattr(target_contract, "derived_primary_f24_indices", None)
    if values is None and isinstance(target_contract, Mapping):
        values = target_contract.get("derived_primary_f24_indices")
    if values is None:
        raise ValueError("target contract lacks derived_primary_f24_indices")
    return tuple(int(value) for value in values)


def _f24_mapping(target_contract: Any) -> Mapping[int, Sequence[int]]:
    value = getattr(target_contract, "f24_to_m4_indices", None)
    if value is None and isinstance(target_contract, Mapping):
        value = target_contract.get("f24_to_m4_indices")
    if not isinstance(value, Mapping):
        raise ValueError("target contract lacks f24_to_m4_indices")
    return {int(key): tuple(int(item) for item in values) for key, values in value.items()}


@dataclass(frozen=True)
class F24ComposeItem:
    canonical_index: int
    slot_id: str
    rule: str
    loss_family: str
    operator: str
    ordinal_classes: int
    source_positions: tuple[int, ...]
    source_spans: tuple[float, ...]
    observation_positions: tuple[int, ...]


@dataclass(frozen=True)
class F24CompositionPlan:
    items: tuple[F24ComposeItem, ...]
    direct_query_count: int


_PLAN_CACHE: dict[int, F24CompositionPlan] = {}


def build_f24_composition_plan(target_contract: Any) -> F24CompositionPlan:
    cached = _PLAN_CACHE.get(id(target_contract))
    if cached is not None:
        return cached
    slots = _contract_slots(target_contract)
    active = _active_slots(target_contract)
    canonical_to_position = {
        int(_value(slot, "canonical_index", _value(slot, "source_index"))): position
        for position, slot in enumerate(active)
    }
    direct_by_semantics_and_time = {
        (
            int(_value(slot, "field_id")),
            str(_value(slot, "operator")),
            str(_value(slot, "condition")),
            int(_value(slot, "time_index")),
        ): position
        for position, slot in enumerate(active)
    }
    mapping = _f24_mapping(target_contract)
    items: list[F24ComposeItem] = []
    for f24_index in _f24_indices(target_contract):
        f24_slot = slots[f24_index]
        source_indices = tuple(mapping.get(f24_index, ()))
        if len(source_indices) != 6:
            raise ValueError(
                f"F24 slot {_value(f24_slot, 'slot_id', f24_index)} requires six M4 sources"
            )
        source_slots = tuple(slots[index] for index in source_indices)
        source_slots = tuple(sorted(source_slots, key=lambda slot: int(_value(slot, "time_index"))))
        source_positions = tuple(
            canonical_to_position[int(_value(slot, "canonical_index", _value(slot, "source_index")))]
            for slot in source_slots
        )
        rule = str(_value(f24_slot, "compose_rule"))
        observation_positions: list[int] = []
        for slot in source_slots:
            prefix = (int(_value(slot, "field_id")),)
            suffix = (int(_value(slot, "time_index")),)
            position = direct_by_semantics_and_time.get(
                prefix + ("DURATION", "OBSERVED") + suffix
            )
            if position is None:
                position = direct_by_semantics_and_time.get(
                    prefix + ("COUNT", "OBSERVED") + suffix
                )
            observation_positions.append(-1 if position is None else position)
        if rule == "weighted_mean" and any(position < 0 for position in observation_positions):
            raise ValueError(
                f"weighted F24 slot {_value(f24_slot, 'slot_id', f24_index)} "
                "lacks M4 predicted coverage/count"
            )
        items.append(F24ComposeItem(
            canonical_index=f24_index,
            slot_id=str(_value(f24_slot, "slot_id", f24_index)),
            rule=rule,
            loss_family=str(_value(f24_slot, "loss_family")),
            operator=str(_value(f24_slot, "operator")),
            ordinal_classes=int(_value(f24_slot, "ordinal_classes", 0)),
            source_positions=source_positions,
            source_spans=tuple(float(_value(slot, "span_hours", 4.0)) for slot in source_slots),
            observation_positions=tuple(observation_positions),
        ))
    plan = F24CompositionPlan(items=tuple(items), direct_query_count=len(active))
    _PLAN_CACHE[id(target_contract)] = plan
    return plan


def _normalizer_stat(slot: Any, normalizer: Any) -> tuple[Any, float]:
    if normalizer is None:
        raise ValueError("raw-scale prediction requires the frozen train-fitted normalizer")
    exact_key = (
        f"{int(_value(slot, 'field_id'))}:"
        f"{int(_value(slot, 'operator_id'))}:"
        f"{int(_value(slot, 'condition_id'))}:"
        f"{str(_value(slot, 'resolution'))}"
    )
    if isinstance(normalizer, Mapping):
        stats = normalizer.get("event_stats", {})
        fallback_stats = normalizer.get("fallback_event_stats", {})
        fallback_keys = normalizer.get("fallback_event_keys", {})
        epsilon = float(normalizer.get("epsilon", EPS))
        stat = stats.get(exact_key) if isinstance(stats, Mapping) else None
        if stat is None:
            fallback_key = (
                fallback_keys.get(exact_key)
                if isinstance(fallback_keys, Mapping)
                else None
            )
            if fallback_key is None:
                value_type = str(
                    _value(slot, "value_type", _value(slot, "loss_family", "continuous"))
                )
                mode = "log1p" if value_type in {"nonnegative", "count"} else "linear"
                candidates = (
                    f"template:{int(_value(slot, 'field_id'))}:"
                    f"{int(_value(slot, 'operator_id'))}:"
                    f"{int(_value(slot, 'condition_id'))}:{mode}",
                    f"field:{int(_value(slot, 'field_id'))}:{mode}",
                    f"global:{value_type}:{mode}",
                )
                fallback_key = next(
                    (
                        key for key in candidates
                        if isinstance(fallback_stats, Mapping) and key in fallback_stats
                    ),
                    None,
                )
            stat = (
                fallback_stats.get(fallback_key)
                if isinstance(fallback_stats, Mapping) and fallback_key is not None
                else None
            )
    else:
        epsilon = float(getattr(normalizer, "epsilon", EPS))
        resolver = getattr(normalizer, "_event_stat", None)
        if callable(resolver):
            proxy = SimpleNamespace(
                field_id=int(_value(slot, "field_id")),
                operator_id=int(_value(slot, "operator_id")),
                condition_id=int(_value(slot, "condition_id")),
                value_type=str(
                    _value(slot, "value_type", _value(slot, "loss_family", "continuous"))
                ),
            )
            stat = resolver(exact_key, proxy)
        else:
            stats = getattr(normalizer, "event_stats", None)
            stat = stats.get(exact_key) if isinstance(stats, Mapping) else None
    if stat is None:
        raise ValueError(
            f"normalizer has neither an exact nor frozen fallback statistic for {exact_key}"
        )
    return stat, epsilon


def inverse_normalized_prediction(
    normalized: Tensor,
    slot: Any,
    normalizer: Any,
) -> Tensor:
    """Invert one continuous query using its train-only M4 robust statistic."""

    stat, epsilon = _normalizer_stat(slot, normalizer)
    median = float(_value(stat, "median"))
    iqr = max(float(_value(stat, "iqr")), epsilon)
    transformed = normalized * iqr + median
    if bool(_value(stat, "log1p", False)):
        transformed = torch.expm1(transformed)
    return transformed


def normalize_raw_value(raw: Tensor, slot: Any, normalizer: Any) -> Tensor:
    """Map a raw target/prediction to its frozen registry-specific robust scale."""

    stat, epsilon = _normalizer_stat(slot, normalizer)
    transformed = raw
    if bool(_value(stat, "log1p", False)):
        transformed = torch.log1p(raw.clamp_min(0.0))
    median = float(_value(stat, "median"))
    iqr = max(float(_value(stat, "iqr")), epsilon)
    return (transformed - median) / iqr


def _probability_any(probability: Tensor) -> Tensor:
    return 1.0 - torch.prod((1.0 - probability).clamp(0.0, 1.0), dim=1)


def _selected_value(
    values: Tensor,
    probability: Tensor,
    *,
    order: Tensor,
) -> tuple[Tensor, Tensor]:
    ordered_values = torch.gather(values, 1, order)
    ordered_probability = torch.gather(probability, 1, order)
    previous_absent = torch.cumprod(
        torch.cat(
            (
                torch.ones_like(ordered_probability[:, :1]),
                (1.0 - ordered_probability[:, :-1]).clamp(0.0, 1.0),
            ),
            dim=1,
        ),
        dim=1,
    )
    selection = ordered_probability * previous_absent
    probability_any = selection.sum(dim=1)
    conditional = (selection * ordered_values).sum(dim=1) / probability_any.clamp_min(EPS)
    fallback = (values * probability).sum(dim=1) / probability.sum(dim=1).clamp_min(EPS)
    conditional = torch.where(probability_any.gt(EPS), conditional, fallback)
    return conditional, probability_any


def _ordinal_selected_distribution(
    probabilities: Tensor,
    observed: Tensor,
    *,
    rule: str,
) -> tuple[Tensor, Tensor]:
    batch_size, block_count, classes = probabilities.shape
    none_probability = torch.prod(1.0 - observed, dim=1)
    any_probability = 1.0 - none_probability
    if rule == "last":
        reverse_observed = observed.flip(1)
        prior_absent = torch.cumprod(
            torch.cat(
                (
                    torch.ones_like(reverse_observed[:, :1]),
                    1.0 - reverse_observed[:, :-1],
                ),
                dim=1,
            ),
            dim=1,
        )
        weights = (reverse_observed * prior_absent).flip(1)
        distribution = (
            probabilities * weights.unsqueeze(-1)
        ).sum(dim=1) / any_probability.clamp_min(EPS).unsqueeze(1)
    elif rule == "min":
        survival = probabilities.flip(-1).cumsum(-1).flip(-1)
        joint = torch.prod(
            (1.0 - observed).unsqueeze(-1) + observed.unsqueeze(-1) * survival,
            dim=1,
        )
        conditional_survival = (
            joint - none_probability.unsqueeze(1)
        ) / any_probability.clamp_min(EPS).unsqueeze(1)
        distribution = torch.cat(
            (
                conditional_survival[:, :-1] - conditional_survival[:, 1:],
                conditional_survival[:, -1:],
            ),
            dim=1,
        )
    elif rule == "max":
        cdf = probabilities.cumsum(-1)
        joint = torch.prod(
            (1.0 - observed).unsqueeze(-1) + observed.unsqueeze(-1) * cdf,
            dim=1,
        )
        conditional_cdf = (
            joint - none_probability.unsqueeze(1)
        ) / any_probability.clamp_min(EPS).unsqueeze(1)
        distribution = torch.cat(
            (conditional_cdf[:, :1], conditional_cdf[:, 1:] - conditional_cdf[:, :-1]),
            dim=1,
        )
    else:
        raise ValueError(f"ordinal selection does not support {rule}")
    fallback = probabilities.mean(dim=1)
    distribution = torch.where(any_probability.view(batch_size, 1).gt(EPS), distribution, fallback)
    distribution = distribution.clamp_min(0.0)
    return distribution / distribution.sum(dim=1, keepdim=True).clamp_min(EPS), any_probability


def derive_f24_prediction_summary(
    prediction_summary: Mapping[str, Tensor],
    target_contract: Any,
) -> dict[str, Tensor]:
    """Compose 149 raw-scale F24 point forecasts from the six M4 forecasts.

    This function deliberately rejects a bare model-space prediction tensor.  Its
    input must be a typed summary whose expected values were already inverted with
    the frozen training normalizer.
    """

    if not isinstance(prediction_summary, Mapping):
        raise TypeError("F24 composition requires a raw typed prediction_summary")
    direct = prediction_summary.get("expected_raw_value")
    conditional_direct = prediction_summary.get("conditional_raw_value")
    direct_presence = prediction_summary.get("presence_probability")
    if direct is None or conditional_direct is None or direct_presence is None:
        raise ValueError(
            "prediction_summary requires expected_raw_value, conditional_raw_value, "
            "and presence_probability"
        )
    if direct.ndim != 2:
        raise ValueError("expected_raw_value must have shape [batch, direct_query]")
    plan = build_f24_composition_plan(target_contract)
    if direct.shape[1] != plan.direct_query_count:
        raise ValueError(
            f"raw prediction summary has {direct.shape[1]} queries; "
            f"expected {plan.direct_query_count}"
        )

    composed_expected: list[Tensor] = []
    composed_conditional: list[Tensor] = []
    composed_presence: list[Tensor] = []
    max_ordinal_classes = max((item.ordinal_classes for item in plan.items), default=0)
    ordinal_output = direct.new_zeros((direct.shape[0], len(plan.items), max_ordinal_classes))
    ordinal_mask = torch.zeros(
        (len(plan.items), max_ordinal_classes),
        dtype=torch.bool,
        device=direct.device,
    )
    direct_ordinal = prediction_summary.get("ordinal_probability")
    for item in plan.items:
        index = torch.tensor(item.source_positions, device=direct.device)
        values = direct.index_select(1, index)
        conditional_values = conditional_direct.index_select(1, index)
        source_presence = direct_presence.index_select(1, index).clamp(0.0, 1.0)
        linked = [position >= 0 for position in item.observation_positions]
        if any(linked):
            observation_index = torch.tensor(
                [max(position, 0) for position in item.observation_positions],
                device=direct.device,
            )
            linked_presence = direct_presence.index_select(1, observation_index).clamp(0.0, 1.0)
            observed = torch.where(
                direct.new_tensor(linked, dtype=torch.bool).unsqueeze(0),
                linked_presence,
                source_presence,
            )
        elif item.operator == "STATE":
            observed = torch.ones_like(source_presence)
        else:
            observed = source_presence
        probability_any = _probability_any(observed)
        if item.rule == "sum":
            expected_result = values.sum(dim=1)
            conditional_result = expected_result / probability_any.clamp_min(EPS)
        elif item.rule == "min":
            if item.loss_family == "ordinal" and direct_ordinal is not None:
                classes = item.ordinal_classes
                probabilities = direct_ordinal.index_select(1, index)[..., :classes]
                distribution, probability_any = _ordinal_selected_distribution(
                    probabilities,
                    observed,
                    rule="min",
                )
                class_values = torch.arange(
                    1, classes + 1, device=direct.device, dtype=direct.dtype
                )
                conditional_result = (distribution * class_values).sum(dim=1)
                ordinal_output[:, len(composed_expected), :classes] = distribution
                ordinal_mask[len(composed_expected), :classes] = True
            else:
                order = torch.argsort(conditional_values, dim=1)
                conditional_result, probability_any = _selected_value(
                    conditional_values, observed, order=order
                )
            expected_result = conditional_result
        elif item.rule == "max":
            if item.loss_family == "ordinal" and direct_ordinal is not None:
                classes = item.ordinal_classes
                probabilities = direct_ordinal.index_select(1, index)[..., :classes]
                distribution, probability_any = _ordinal_selected_distribution(
                    probabilities,
                    observed,
                    rule="max",
                )
                class_values = torch.arange(
                    1, classes + 1, device=direct.device, dtype=direct.dtype
                )
                conditional_result = (distribution * class_values).sum(dim=1)
                ordinal_output[:, len(composed_expected), :classes] = distribution
                ordinal_mask[len(composed_expected), :classes] = True
            else:
                order = torch.argsort(conditional_values, dim=1, descending=True)
                conditional_result, probability_any = _selected_value(
                    conditional_values, observed, order=order
                )
            expected_result = conditional_result
        elif item.rule == "last":
            if any(linked):
                if item.loss_family == "ordinal" and direct_ordinal is not None:
                    classes = item.ordinal_classes
                    probabilities = direct_ordinal.index_select(1, index)[..., :classes]
                    distribution, probability_any = _ordinal_selected_distribution(
                        probabilities,
                        observed,
                        rule="last",
                    )
                    class_values = torch.arange(
                        1, classes + 1, device=direct.device, dtype=direct.dtype
                    )
                    conditional_result = (distribution * class_values).sum(dim=1)
                    ordinal_output[:, len(composed_expected), :classes] = distribution
                    ordinal_mask[len(composed_expected), :classes] = True
                else:
                    order = torch.arange(5, -1, -1, device=direct.device).view(1, -1)
                    order = order.expand(direct.shape[0], -1)
                    conditional_result, probability_any = _selected_value(
                        conditional_values, observed, order=order
                    )
                expected_result = conditional_result
            else:
                expected_result = values[:, -1]
                conditional_result = conditional_values[:, -1]
                probability_any = observed[:, -1]
        elif item.rule == "weighted_mean":
            coverage_index = torch.tensor(item.observation_positions, device=direct.device)
            weights = direct.index_select(1, coverage_index).clamp_min(0.0)
            denominator = weights.sum(dim=1)
            weighted = (conditional_values * weights).sum(dim=1) / denominator.clamp_min(EPS)
            fallback = (
                (conditional_values * observed).sum(dim=1)
                / observed.sum(dim=1).clamp_min(EPS)
            )
            conditional_result = torch.where(denominator.gt(EPS), weighted, fallback)
            expected_result = conditional_result
        elif item.rule == "block_weighted_mean":
            weights = direct.new_tensor(item.source_spans).unsqueeze(0)
            expected_result = (values * weights).sum(dim=1) / weights.sum().clamp_min(EPS)
            conditional_result = expected_result / probability_any.clamp_min(EPS)
        else:
            raise ValueError(f"unsupported F24 compose rule {item.rule!r} for {item.slot_id}")
        composed_expected.append(expected_result)
        composed_conditional.append(conditional_result)
        composed_presence.append(probability_any)
    if not composed_expected:
        raise ValueError("target contract exposes no derived primary F24 slots")
    expected_tensor = torch.stack(composed_expected, dim=1)
    binary_probability = torch.full_like(expected_tensor, torch.nan)
    for position, item in enumerate(plan.items):
        if item.loss_family == "binary":
            binary_probability[:, position] = expected_tensor[:, position]
    return {
        "conditional_raw_value": torch.stack(composed_conditional, dim=1),
        "expected_raw_value": expected_tensor,
        "presence_probability": torch.stack(composed_presence, dim=1),
        "binary_probability": binary_probability,
        "ordinal_probability": ordinal_output,
        "ordinal_class_mask": ordinal_mask,
    }


def derive_f24_predictions(
    prediction_summary: Mapping[str, Tensor],
    target_contract: Any,
) -> Tensor:
    return derive_f24_prediction_summary(prediction_summary, target_contract)[
        "expected_raw_value"
    ]


__all__ = [
    "F24CompositionPlan",
    "build_f24_composition_plan",
    "derive_f24_prediction_summary",
    "derive_f24_predictions",
    "inverse_normalized_prediction",
    "normalize_raw_value",
]
