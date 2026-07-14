from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


STUDENT_T_DF = 3.0
EPS = 1e-6
RESPIRATORY_SUPPORT_FIELD_ID = 11
VASOPRESSOR_SUPPORT_FIELD_ID = 27
ANTIBIOTICS_FIELD_ID = 34
SUPPORT_FIELD_IDS = {
    RESPIRATORY_SUPPORT_FIELD_ID,
    VASOPRESSOR_SUPPORT_FIELD_ID,
    ANTIBIOTICS_FIELD_ID,
}


@dataclass(frozen=True)
class QueryLossSpec:
    position: int
    field_id: int
    operator_id: int
    condition_id: int
    field: str
    operator: str
    condition: str
    resolution: str
    resolution_id: int
    time_index: int
    span_hours: float
    value_type: str
    loss_family: str
    semantic_component: str
    duration_kind: str | None
    coverage_query_position: int | None
    ordinal_classes: int | None
    compose_rule: str
    source_index: int | None

    @property
    def block_key(self) -> tuple[str, int]:
        return (self.resolution, self.time_index)


@dataclass(frozen=True)
class MacroLossLayout:
    queries: tuple[QueryLossSpec, ...]
    component_groups: tuple[tuple[tuple[str, int], int, str, tuple[int, ...]], ...]
    field_groups: tuple[tuple[tuple[str, int], int, tuple[int, ...]], ...]
    block_groups: tuple[tuple[tuple[str, int], tuple[int, ...]], ...]
    respiratory_state_groups: tuple[tuple[int, ...], ...]
    respiratory_duration_groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class DeviceLossLayout:
    family_positions: Mapping[str, Tensor]
    ordinal_positions: Mapping[int, Tensor]
    point_positions: Tensor
    point_spans: Tensor
    point_coverage_positions: Tensor
    interval_positions: Tensor
    interval_spans: Tensor
    respiratory_state_groups: tuple[Tensor, ...]
    respiratory_duration_groups: tuple[Tensor, ...]
    query_to_component: Tensor
    component_to_field: Tensor
    field_to_block: Tensor
    h1_block_positions: Tensor
    m4_block_positions: Tensor
    family_part_positions: tuple[tuple[str, Tensor], ...]
    component_part_positions: tuple[tuple[str, Tensor], ...]


_LAYOUT_CACHE: dict[int, MacroLossLayout] = {}
_DEVICE_LAYOUT_CACHE: dict[tuple[int, str], DeviceLossLayout] = {}


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _contract_queries(target_contract: Any) -> Sequence[Any]:
    queries = getattr(target_contract, "queries", None)
    if queries is None and isinstance(target_contract, Mapping):
        queries = target_contract.get("queries")
    if queries is None:
        queries = getattr(target_contract, "active_slots", None)
    if queries is None:
        slots = getattr(target_contract, "slots", None)
        if slots is not None:
            queries = [slot for slot in slots if bool(_value(slot, "enabled_in_baseline", False))]
    if queries is None:
        raise ValueError("target_contract must expose queries or active_slots")
    return tuple(queries)


def _fallback_component(field_id: int, operator: str, condition: str) -> str:
    if operator in {"OBS", "LAST", "MIN", "MAX", "MEAN"}:
        return "value_summary"
    if operator == "COUNT":
        return "observation_count"
    if operator == "DURATION":
        if condition == "OBSERVED":
            return "observation_coverage"
        return "support_duration" if field_id in SUPPORT_FIELD_IDS else "abnormal_duration"
    if operator == "STATE":
        return "state"
    if operator == "START":
        return "start"
    if operator == "SUM":
        return "amount_sum"
    raise ValueError(f"cannot infer semantic component for {field_id}/{operator}/{condition}")


def _duration_kind(item: Any, field_id: int, component: str) -> str | None:
    explicit = _value(item, "duration_kind")
    if explicit:
        return str(explicit)
    if component not in {"observation_coverage", "abnormal_duration", "support_duration"}:
        return None
    if field_id == RESPIRATORY_SUPPORT_FIELD_ID and component == "support_duration":
        return "respiratory_grouped"
    if field_id in SUPPORT_FIELD_IDS and component == "support_duration":
        return "interval_zoib"
    return "point_binomial"


def _query_spec(position: int, item: Any) -> QueryLossSpec:
    field_id = int(_value(item, "field_id"))
    operator = str(_value(item, "operator", ""))
    condition = str(_value(item, "condition", ""))
    component = str(
        _value(item, "semantic_component", "")
        or _fallback_component(field_id, operator, condition)
    )
    field = str(_value(item, "field", f"field_{field_id}"))
    ordinal_classes = _value(item, "ordinal_classes")
    if ordinal_classes is None:
        ordinal_classes = {"gcs_eye": 4, "gcs_motor": 6}.get(field)
    span = _value(item, "span_hours", _value(item, "span", 1.0))
    coverage_position = _value(item, "coverage_query_position")
    if coverage_position is not None and int(coverage_position) < 0:
        coverage_position = None
    return QueryLossSpec(
        position=position,
        field_id=field_id,
        operator_id=int(_value(item, "operator_id", 0)),
        condition_id=int(_value(item, "condition_id", 0)),
        field=field,
        operator=operator,
        condition=condition,
        resolution=str(_value(item, "resolution", "")),
        resolution_id=int(_value(item, "resolution_id", 0)),
        time_index=int(_value(item, "time_index", 0)),
        span_hours=float(span),
        value_type=str(_value(item, "value_type", _value(item, "loss_family", "continuous"))),
        loss_family=str(_value(item, "loss_family")),
        semantic_component=component,
        duration_kind=_duration_kind(item, field_id, component),
        coverage_query_position=(None if coverage_position is None else int(coverage_position)),
        ordinal_classes=(None if ordinal_classes is None else int(ordinal_classes)),
        compose_rule=str(_value(item, "compose_rule", "not_cross_resolution")),
        source_index=(
            None if _value(item, "source_index") is None else int(_value(item, "source_index"))
        ),
    )


def _build_layout(target_contract: Any) -> MacroLossLayout:
    cache_key = id(target_contract)
    cached = _LAYOUT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    queries = tuple(_query_spec(index, item) for index, item in enumerate(_contract_queries(target_contract)))

    coverage_lookup = {
        (query.field_id, query.block_key): query.position
        for query in queries
        if query.operator == "DURATION" and query.condition == "OBSERVED"
    }
    resolved: list[QueryLossSpec] = []
    for query in queries:
        coverage = query.coverage_query_position
        if coverage is None and query.duration_kind == "point_binomial" and query.condition != "OBSERVED":
            coverage = coverage_lookup.get((query.field_id, query.block_key))
        resolved.append(QueryLossSpec(**{**query.__dict__, "coverage_query_position": coverage}))
    queries = tuple(resolved)

    component_map: dict[tuple[tuple[str, int], int, str], list[int]] = defaultdict(list)
    for query in queries:
        component_map[(query.block_key, query.field_id, query.semantic_component)].append(query.position)
    component_groups = tuple(
        (block_key, field_id, component, tuple(positions))
        for (block_key, field_id, component), positions in component_map.items()
    )
    field_map: dict[tuple[tuple[str, int], int], list[int]] = defaultdict(list)
    for component_index, (block_key, field_id, _, _) in enumerate(component_groups):
        field_map[(block_key, field_id)].append(component_index)
    field_groups = tuple(
        (block_key, field_id, tuple(component_indices))
        for (block_key, field_id), component_indices in field_map.items()
    )
    block_map: dict[tuple[str, int], list[int]] = defaultdict(list)
    for field_index, (block_key, _, _) in enumerate(field_groups):
        block_map[block_key].append(field_index)
    block_groups = tuple((block_key, tuple(indices)) for block_key, indices in block_map.items())

    respiratory_state: dict[tuple[str, int], list[int]] = defaultdict(list)
    respiratory_duration: dict[tuple[str, int], list[int]] = defaultdict(list)
    for query in queries:
        if query.field_id != RESPIRATORY_SUPPORT_FIELD_ID:
            continue
        if query.operator == "STATE":
            respiratory_state[query.block_key].append(query.position)
        elif query.operator == "DURATION":
            respiratory_duration[query.block_key].append(query.position)
    layout = MacroLossLayout(
        queries=queries,
        component_groups=component_groups,
        field_groups=field_groups,
        block_groups=block_groups,
        respiratory_state_groups=tuple(tuple(value) for value in respiratory_state.values()),
        respiratory_duration_groups=tuple(tuple(value) for value in respiratory_duration.values()),
    )
    _LAYOUT_CACHE[cache_key] = layout
    return layout


def _device_layout(layout: MacroLossLayout, device: torch.device) -> DeviceLossLayout:
    cache_key = (id(layout), str(device))
    cached = _DEVICE_LAYOUT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    def tensor(values: Sequence[int]) -> Tensor:
        return torch.tensor(tuple(values), dtype=torch.long, device=device)

    family_map: dict[str, list[int]] = defaultdict(list)
    ordinal_map: dict[int, list[int]] = defaultdict(list)
    point_queries: list[QueryLossSpec] = []
    interval_queries: list[QueryLossSpec] = []
    for query in layout.queries:
        family_map[query.loss_family].append(query.position)
        if query.loss_family == "ordinal":
            if query.ordinal_classes is None:
                raise ValueError(f"ordinal query {query.position} lacks ordinal_classes")
            ordinal_map[query.ordinal_classes].append(query.position)
        if query.loss_family == "duration" and query.duration_kind == "point_binomial":
            point_queries.append(query)
        if query.loss_family == "duration" and query.duration_kind == "interval_zoib":
            interval_queries.append(query)

    query_to_component = torch.empty(len(layout.queries), dtype=torch.long, device=device)
    for component_index, (_, _, _, positions) in enumerate(layout.component_groups):
        query_to_component[tensor(positions)] = component_index
    component_to_field = torch.empty(len(layout.component_groups), dtype=torch.long, device=device)
    for field_index, (_, _, component_indices) in enumerate(layout.field_groups):
        component_to_field[tensor(component_indices)] = field_index
    field_to_block = torch.empty(len(layout.field_groups), dtype=torch.long, device=device)
    for block_index, (_, field_indices) in enumerate(layout.block_groups):
        field_to_block[tensor(field_indices)] = block_index

    value = DeviceLossLayout(
        family_positions={key: tensor(positions) for key, positions in family_map.items()},
        ordinal_positions={key: tensor(positions) for key, positions in ordinal_map.items()},
        point_positions=tensor([query.position for query in point_queries]),
        point_spans=torch.tensor(
            [query.span_hours for query in point_queries], dtype=torch.float32, device=device
        ),
        point_coverage_positions=tensor([
            -1 if query.coverage_query_position is None else query.coverage_query_position
            for query in point_queries
        ]),
        interval_positions=tensor([query.position for query in interval_queries]),
        interval_spans=torch.tensor(
            [query.span_hours for query in interval_queries], dtype=torch.float32, device=device
        ),
        respiratory_state_groups=tuple(tensor(group) for group in layout.respiratory_state_groups),
        respiratory_duration_groups=tuple(
            tensor(group) for group in layout.respiratory_duration_groups
        ),
        query_to_component=query_to_component,
        component_to_field=component_to_field,
        field_to_block=field_to_block,
        h1_block_positions=tensor([
            index for index, (block_key, _) in enumerate(layout.block_groups)
            if block_key[0] == "H1"
        ]),
        m4_block_positions=tensor([
            index for index, (block_key, _) in enumerate(layout.block_groups)
            if block_key[0] == "M4"
        ]),
        family_part_positions=tuple(
            (label, tensor(positions)) for label, positions in sorted(family_map.items())
        ),
        component_part_positions=tuple(
            (
                label,
                tensor([query.position for query in layout.queries if query.semantic_component == label]),
            )
            for label in sorted({query.semantic_component for query in layout.queries})
        ),
    )
    _DEVICE_LAYOUT_CACHE[cache_key] = value
    return value


def _student_t_nll(target: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
    distribution = torch.distributions.StudentT(
        df=torch.full_like(loc, STUDENT_T_DF),
        loc=loc,
        scale=scale,
    )
    return -distribution.log_prob(target)


def _hurdle_nb_nll(
    raw_target: Tensor,
    gate_logits: Tensor,
    total_count: Tensor,
    nb_logits: Tensor,
) -> Tensor:
    positive = raw_target.gt(0)
    gate = F.binary_cross_entropy_with_logits(
        gate_logits,
        positive.float(),
        reduction="none",
    )
    shifted = (raw_target - 1.0).clamp_min(0.0)
    count_nll = -torch.distributions.NegativeBinomial(
        total_count=total_count,
        logits=nb_logits,
    ).log_prob(shifted)
    return gate + positive.float() * count_nll


def _interval_zoib_nll(
    raw_target: Tensor,
    span: Tensor,
    mixture_logits: Tensor,
    alpha: Tensor,
    beta: Tensor,
) -> Tensor:
    fraction = (raw_target / span.clamp_min(EPS)).clamp(0.0, 1.0)
    log_weights = F.log_softmax(mixture_logits, dim=-1)
    at_zero = fraction.le(EPS)
    at_one = fraction.ge(1.0 - EPS)
    safe_fraction = fraction.clamp(EPS, 1.0 - EPS)
    beta_nll = -torch.distributions.Beta(alpha, beta).log_prob(safe_fraction)
    return torch.where(
        at_zero,
        -log_weights[..., 0],
        torch.where(at_one, -log_weights[..., 2], -log_weights[..., 1] + beta_nll),
    )


def _nonnegative_nll(
    normalized_target: Tensor,
    raw_target: Tensor,
    gate_logits: Tensor,
    loc: Tensor,
    scale: Tensor,
) -> Tensor:
    positive = raw_target.gt(0)
    gate = F.binary_cross_entropy_with_logits(
        gate_logits,
        positive.float(),
        reduction="none",
    )
    value_nll = _student_t_nll(normalized_target, loc, scale)
    return gate + positive.float() * value_nll


def _ordered_ordinal_logits(raw_logits: Tensor, threshold_count: int) -> Tensor:
    first = raw_logits[..., :1]
    if threshold_count == 1:
        return first
    gaps = F.softplus(raw_logits[..., 1:threshold_count])
    return torch.cat((first, first - torch.cumsum(gaps, dim=-1)), dim=-1)


def _set_structured_respiratory_losses(
    outputs: Mapping[str, Tensor],
    raw_target: Tensor,
    target_mask: Tensor,
    query_loss: Tensor,
    query_valid: Tensor,
    predictions: Tensor,
    layout: MacroLossLayout,
    indices: DeviceLossLayout,
) -> None:
    for positions, index in zip(
        layout.respiratory_state_groups,
        indices.respiratory_state_groups,
        strict=True,
    ):
        valid = target_mask.index_select(1, index).all(dim=1)
        values = raw_target.index_select(1, index).clamp(0.0, 1.0)
        none = (1.0 - values.sum(dim=1, keepdim=True)).clamp(0.0, 1.0)
        target_distribution = torch.cat((none, values), dim=1)
        logits = torch.cat(
            (
                torch.zeros_like(none),
                outputs["structured_scores"].float().index_select(1, index),
            ),
            dim=1,
        )
        group_loss = -(target_distribution * F.log_softmax(logits, dim=1)).sum(dim=1)
        query_loss[:, positions[0]] = group_loss
        query_valid[:, positions[0]] = valid
        for position in positions[1:]:
            query_valid[:, position] = False
        probabilities = F.softmax(logits, dim=1)[:, 1:]
        predictions[:, index] = probabilities

    for positions, index in zip(
        layout.respiratory_duration_groups,
        indices.respiratory_duration_groups,
        strict=True,
    ):
        valid = target_mask.index_select(1, index).all(dim=1)
        span = raw_target.new_tensor([layout.queries[position].span_hours for position in positions])
        # Every query in one respiratory group has the same block span.
        block_span = span[0].clamp_min(EPS)
        values = raw_target.index_select(1, index).clamp_min(0.0)
        fractions = values / block_span
        none = (1.0 - fractions.sum(dim=1, keepdim=True)).clamp(0.0, 1.0)
        target_distribution = torch.cat((none, fractions), dim=1)
        target_distribution = target_distribution / target_distribution.sum(dim=1, keepdim=True).clamp_min(EPS)
        logits = torch.cat(
            (
                torch.zeros_like(none),
                outputs["structured_scores"].float().index_select(1, index),
            ),
            dim=1,
        )
        group_loss = -(target_distribution * F.log_softmax(logits, dim=1)).sum(dim=1)
        query_loss[:, positions[0]] = group_loss
        query_valid[:, positions[0]] = valid
        for position in positions[1:]:
            query_valid[:, position] = False
        probabilities = F.softmax(logits, dim=1)[:, 1:]
        predictions[:, index] = block_span * probabilities


def _ordinal_probabilities(logits: Tensor, classes: int) -> Tensor:
    cumulative = torch.sigmoid(_ordered_ordinal_logits(logits, classes - 1))
    pieces = [1.0 - cumulative[:, :1]]
    if classes > 2:
        pieces.append(cumulative[:, :-1] - cumulative[:, 1:])
    pieces.append(cumulative[:, -1:])
    probabilities = torch.cat(pieces, dim=1).clamp_min(0.0)
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(EPS)


def summarize_typed_predictions(
    head_outputs: Mapping[str, Tensor],
    target_contract: Any,
    normalizer: Any,
) -> dict[str, Tensor]:
    """Convert typed head parameters into compact raw-scale query summaries.

    ``conditional_raw_value`` preserves the positive/observed branch estimate;
    ``expected_raw_value`` integrates the hurdle or class probability.  The latter
    is the only scalar bank used for cross-resolution F24 composition.
    """

    from trauma_predict.eval.f24_composition import inverse_normalized_prediction

    head_outputs = {
        key: value.float() if torch.is_floating_point(value) else value
        for key, value in head_outputs.items()
    }
    layout = _build_layout(target_contract)
    reference = head_outputs["continuous_loc"]
    batch_size, query_count = reference.shape
    if query_count != len(layout.queries):
        raise ValueError("head outputs do not align with the active target contract")
    conditional = torch.zeros_like(reference)
    expected = torch.zeros_like(reference)
    presence = torch.ones_like(reference)
    binary_probability = torch.full_like(reference, torch.nan)
    max_ordinal_classes = max(
        (query.ordinal_classes or 0 for query in layout.queries),
        default=0,
    )
    ordinal_probability = reference.new_zeros((batch_size, query_count, max_ordinal_classes))
    ordinal_class_mask = torch.zeros(
        (query_count, max_ordinal_classes),
        dtype=torch.bool,
        device=reference.device,
    )

    deferred_point_durations: list[QueryLossSpec] = []
    for query in layout.queries:
        position = query.position
        family = query.loss_family
        if family == "continuous":
            raw = inverse_normalized_prediction(
                head_outputs["continuous_loc"][:, position],
                query,
                normalizer,
            )
            conditional[:, position] = raw
            expected[:, position] = raw
        elif family == "ordinal":
            classes = query.ordinal_classes
            if classes is None:
                raise ValueError(f"ordinal query {position} lacks ordinal_classes")
            probabilities = _ordinal_probabilities(
                head_outputs["ordinal_logits"][:, position],
                classes,
            )
            class_values = torch.arange(
                1,
                classes + 1,
                device=reference.device,
                dtype=reference.dtype,
            )
            mean = (probabilities * class_values).sum(dim=1)
            conditional[:, position] = mean
            expected[:, position] = mean
            ordinal_probability[:, position, :classes] = probabilities
            ordinal_class_mask[position, :classes] = True
        elif family == "count":
            probability = torch.sigmoid(head_outputs["count_gate_logits"][:, position])
            positive_mean = 1.0 + head_outputs["count_total_count"][:, position] * torch.exp(
                head_outputs["count_nb_logits"][:, position]
            )
            conditional[:, position] = positive_mean
            expected[:, position] = probability * positive_mean
            presence[:, position] = probability
        elif family == "duration":
            if query.duration_kind == "respiratory_grouped":
                continue
            if query.duration_kind == "point_binomial":
                if query.condition != "OBSERVED" and query.coverage_query_position is not None:
                    deferred_point_durations.append(query)
                    continue
                probability = torch.sigmoid(head_outputs["point_duration_logits"][:, position])
                trials = torch.full_like(probability, query.span_hours)
                mean = trials * probability
                probability_positive = 1.0 - torch.pow(
                    (1.0 - probability).clamp(0.0, 1.0),
                    trials,
                )
                conditional[:, position] = mean / probability_positive.clamp_min(EPS)
                expected[:, position] = mean
                presence[:, position] = probability_positive
            elif query.duration_kind == "interval_zoib":
                weights = F.softmax(
                    head_outputs["interval_mixture_logits"][:, position],
                    dim=-1,
                )
                beta_mean = head_outputs["interval_alpha"][:, position] / (
                    head_outputs["interval_alpha"][:, position]
                    + head_outputs["interval_beta"][:, position]
                )
                positive_probability = weights[:, 1] + weights[:, 2]
                positive_fraction = (
                    weights[:, 2] + weights[:, 1] * beta_mean
                ) / positive_probability.clamp_min(EPS)
                conditional[:, position] = query.span_hours * positive_fraction
                expected[:, position] = query.span_hours * (
                    weights[:, 2] + weights[:, 1] * beta_mean
                )
                presence[:, position] = positive_probability
            else:
                raise ValueError(f"unsupported duration kind {query.duration_kind}")
        elif family == "binary":
            probability = torch.sigmoid(head_outputs["binary_logits"][:, position])
            conditional[:, position] = probability
            expected[:, position] = probability
            presence[:, position] = probability
            binary_probability[:, position] = probability
        elif family == "nonnegative":
            probability = torch.sigmoid(
                head_outputs["nonnegative_gate_logits"][:, position]
            )
            raw_positive = inverse_normalized_prediction(
                head_outputs["nonnegative_loc"][:, position],
                query,
                normalizer,
            ).clamp_min(0.0)
            conditional[:, position] = raw_positive
            expected[:, position] = probability * raw_positive
            presence[:, position] = probability
        else:
            raise ValueError(f"unsupported loss family {family}")

    for query in deferred_point_durations:
        position = query.position
        if query.coverage_query_position is None:
            raise AssertionError("deferred point duration lacks coverage query")
        probability = torch.sigmoid(head_outputs["point_duration_logits"][:, position])
        trials = expected[:, query.coverage_query_position].clamp_min(0.0)
        mean = trials * probability
        probability_positive = 1.0 - torch.pow(
            (1.0 - probability).clamp(0.0, 1.0),
            trials,
        )
        conditional[:, position] = mean / probability_positive.clamp_min(EPS)
        expected[:, position] = mean
        presence[:, position] = probability_positive

    for positions in layout.respiratory_state_groups:
        index = torch.tensor(positions, device=reference.device)
        none = torch.zeros((batch_size, 1), device=reference.device, dtype=reference.dtype)
        logits = torch.cat((none, head_outputs["structured_scores"].index_select(1, index)), dim=1)
        probabilities = F.softmax(logits, dim=1)[:, 1:]
        conditional[:, index] = probabilities
        expected[:, index] = probabilities
        presence[:, index] = probabilities
        binary_probability[:, index] = probabilities

    for positions in layout.respiratory_duration_groups:
        index = torch.tensor(positions, device=reference.device)
        none = torch.zeros((batch_size, 1), device=reference.device, dtype=reference.dtype)
        logits = torch.cat((none, head_outputs["structured_scores"].index_select(1, index)), dim=1)
        probabilities = F.softmax(logits, dim=1)
        condition_probability = probabilities[:, 1:]
        block_span = float(layout.queries[positions[0]].span_hours)
        expected[:, index] = block_span * condition_probability
        conditional[:, index] = block_span
        presence[:, index] = condition_probability

    return {
        "conditional_raw_value": conditional,
        "expected_raw_value": expected,
        "presence_probability": presence,
        "binary_probability": binary_probability,
        "ordinal_probability": ordinal_probability,
        "ordinal_class_mask": ordinal_class_mask,
    }


def _raw_parts(
    values: Tensor,
    valid: Tensor,
    groups: Sequence[tuple[str, Tensor]],
) -> dict[str, dict[str, Tensor]]:
    result: dict[str, dict[str, Tensor]] = {}
    for label, index in groups:
        selected_valid = valid.index_select(1, index)
        selected_values = values.index_select(1, index)
        result[label] = {
            "numerator": (selected_values * selected_valid.float()).sum(),
            "denominator": selected_valid.sum().to(values.dtype),
        }
    return result


def _macro_loss(
    query_loss: Tensor,
    query_valid: Tensor,
    layout: MacroLossLayout,
    indices: DeviceLossLayout,
) -> tuple[Tensor, Tensor, dict[str, dict[str, Tensor]]]:
    batch_size = query_loss.shape[0]
    query_valid_float = query_valid.to(query_loss.dtype)
    component_count = len(layout.component_groups)
    component_sum = query_loss.new_zeros((batch_size, component_count)).scatter_add(
        1,
        indices.query_to_component.unsqueeze(0).expand(batch_size, -1),
        query_loss * query_valid_float,
    )
    component_denominator = query_loss.new_zeros((batch_size, component_count)).scatter_add(
        1,
        indices.query_to_component.unsqueeze(0).expand(batch_size, -1),
        query_valid_float,
    )
    component_tensor = component_sum / component_denominator.clamp_min(1.0)
    component_mask = component_denominator.gt(0)

    field_count = len(layout.field_groups)
    field_sum = query_loss.new_zeros((batch_size, field_count)).scatter_add(
        1,
        indices.component_to_field.unsqueeze(0).expand(batch_size, -1),
        component_tensor * component_mask.to(query_loss.dtype),
    )
    field_denominator = query_loss.new_zeros((batch_size, field_count)).scatter_add(
        1,
        indices.component_to_field.unsqueeze(0).expand(batch_size, -1),
        component_mask.to(query_loss.dtype),
    )
    field_tensor = field_sum / field_denominator.clamp_min(1.0)
    field_mask = field_denominator.gt(0)

    block_count = len(layout.block_groups)
    block_sum = query_loss.new_zeros((batch_size, block_count)).scatter_add(
        1,
        indices.field_to_block.unsqueeze(0).expand(batch_size, -1),
        field_tensor * field_mask.to(query_loss.dtype),
    )
    block_denominator = query_loss.new_zeros((batch_size, block_count)).scatter_add(
        1,
        indices.field_to_block.unsqueeze(0).expand(batch_size, -1),
        field_mask.to(query_loss.dtype),
    )
    block_tensor = block_sum / block_denominator.clamp_min(1.0)
    block_mask = block_denominator.gt(0)

    parts: dict[str, dict[str, Tensor]] = {}
    resolution_values: dict[str, Tensor] = {}
    resolution_valid: dict[str, Tensor] = {}
    for resolution, index in (
        ("H1", indices.h1_block_positions),
        ("M4", indices.m4_block_positions),
    ):
        if index.numel() == 0:
            raise ValueError(f"target contract contains no {resolution} direct block")
        valid = block_mask.index_select(1, index)
        values = block_tensor.index_select(1, index)
        denominator = valid.sum(dim=1)
        per_sample = (values * valid.float()).sum(dim=1) / denominator.clamp_min(1).float()
        per_sample_valid = denominator.gt(0)
        resolution_values[resolution] = per_sample
        resolution_valid[resolution] = per_sample_valid
        parts[f"resolution/{resolution}"] = {
            "numerator": (per_sample * per_sample_valid.float()).sum(),
            "denominator": per_sample_valid.sum().to(query_loss.dtype),
        }

    both = resolution_valid["H1"] & resolution_valid["M4"]
    sample_loss = 0.5 * resolution_values["H1"] + 0.5 * resolution_values["M4"]
    numerator = (sample_loss * both.float()).sum()
    denominator = both.sum().to(query_loss.dtype)
    parts["total"] = {"numerator": numerator, "denominator": denominator}
    return numerator / denominator.clamp_min(1.0), sample_loss, parts


def compute_multires_loss(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Any],
    target_contract: Any,
    *,
    normalizer: Any | None = None,
) -> dict[str, Any]:
    """Compute the frozen typed loss without treating target_mask as a prediction target."""

    layout = _build_layout(target_contract)
    target = batch["target_values"].float()
    raw_target = batch.get("target_raw_values", target).float()
    target_mask = batch["target_mask"].bool()
    if target.shape != target_mask.shape or raw_target.shape != target.shape:
        raise ValueError("target_values, target_raw_values, and target_mask must align")
    batch_size, query_count = target.shape
    if query_count != len(layout.queries):
        raise ValueError(
            f"batch has {query_count} active targets but target_contract has {len(layout.queries)} queries"
        )
    model_query_mask = outputs.get("query_mask")
    if model_query_mask is None:
        model_query_mask = torch.ones_like(target_mask)
    query_valid = target_mask & model_query_mask.bool()
    query_loss = torch.zeros_like(target)
    predictions = torch.zeros_like(target)
    indices = _device_layout(layout, target.device)

    # Family dispatch is vectorized over the frozen query positions. Respiratory
    # mutually-exclusive groups replace their member terms below.
    index = indices.family_positions.get("continuous")
    if index is not None and index.numel():
        y = target.index_select(1, index)
        loc = outputs["continuous_loc"].float().index_select(1, index)
        scale = outputs["continuous_scale"].float().index_select(1, index)
        query_loss[:, index] = _student_t_nll(y, loc, scale)
        predictions[:, index] = loc

    for classes, index in indices.ordinal_positions.items():
        y_raw = raw_target.index_select(1, index)
        logits = _ordered_ordinal_logits(
            outputs["ordinal_logits"].float().index_select(1, index),
            classes - 1,
        )
        thresholds = torch.arange(1, classes, device=target.device).view(1, 1, -1)
        labels = y_raw.round().long().unsqueeze(-1).gt(thresholds).float()
        query_loss[:, index] = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        ).mean(dim=-1)
        predictions[:, index] = 1.0 + torch.sigmoid(logits).sum(dim=-1)

    index = indices.family_positions.get("count")
    if index is not None and index.numel():
        y_raw = raw_target.index_select(1, index)
        gate = outputs["count_gate_logits"].float().index_select(1, index)
        total = outputs["count_total_count"].float().index_select(1, index)
        logits = outputs["count_nb_logits"].float().index_select(1, index)
        query_loss[:, index] = _hurdle_nb_nll(y_raw, gate, total, logits)
        predictions[:, index] = torch.sigmoid(gate) * (
            1.0 + total * torch.exp(logits)
        )

    index = indices.point_positions
    if index.numel():
        y_raw = raw_target.index_select(1, index)
        spans = indices.point_spans.to(target.dtype).unsqueeze(0).expand(batch_size, -1)
        coverage_position = indices.point_coverage_positions
        gathered_coverage = raw_target.index_select(
            1,
            coverage_position.clamp_min(0),
        ).clamp_min(0.0)
        total_count = torch.where(
            coverage_position.unsqueeze(0).ge(0),
            gathered_coverage,
            spans,
        )
        logits = outputs["point_duration_logits"].float().index_select(1, index)
        bounded_target = torch.minimum(y_raw.clamp_min(0.0), total_count)
        query_loss[:, index] = -torch.distributions.Binomial(
            total_count=total_count,
            logits=logits,
        ).log_prob(bounded_target)
        conditional_without_coverage = (
            coverage_position.unsqueeze(0).ge(0) & total_count.le(0.0)
        )
        query_valid[:, index] = query_valid[:, index] & ~conditional_without_coverage
        probability = torch.sigmoid(logits)
        initial_prediction = spans * probability
        predictions[:, index] = initial_prediction
        predicted_coverage = predictions.index_select(
            1,
            coverage_position.clamp_min(0),
        ).clamp_min(0.0)
        predictions[:, index] = torch.where(
            coverage_position.unsqueeze(0).ge(0),
            predicted_coverage * probability,
            initial_prediction,
        )

    index = indices.interval_positions
    if index.numel():
        y_raw = raw_target.index_select(1, index)
        span = indices.interval_spans.to(target.dtype).unsqueeze(0).expand(batch_size, -1)
        mixture = outputs["interval_mixture_logits"].float().index_select(1, index)
        alpha = outputs["interval_alpha"].float().index_select(1, index)
        beta = outputs["interval_beta"].float().index_select(1, index)
        query_loss[:, index] = _interval_zoib_nll(y_raw, span, mixture, alpha, beta)
        weights = F.softmax(mixture, dim=-1)
        beta_mean = alpha / (alpha + beta)
        predictions[:, index] = span * (weights[..., 2] + weights[..., 1] * beta_mean)

    index = indices.family_positions.get("binary")
    if index is not None and index.numel():
        y_raw = raw_target.index_select(1, index)
        logits = outputs["binary_logits"].float().index_select(1, index)
        query_loss[:, index] = F.binary_cross_entropy_with_logits(
            logits,
            y_raw,
            reduction="none",
        )
        predictions[:, index] = torch.sigmoid(logits)

    index = indices.family_positions.get("nonnegative")
    if index is not None and index.numel():
        y = target.index_select(1, index)
        y_raw = raw_target.index_select(1, index)
        gate = outputs["nonnegative_gate_logits"].float().index_select(1, index)
        loc = outputs["nonnegative_loc"].float().index_select(1, index)
        scale = outputs["nonnegative_scale"].float().index_select(1, index)
        query_loss[:, index] = _nonnegative_nll(y, y_raw, gate, loc, scale)
        predictions[:, index] = torch.sigmoid(gate) * loc

    _set_structured_respiratory_losses(
        outputs,
        raw_target,
        target_mask,
        query_loss,
        query_valid,
        predictions,
        layout,
        indices,
    )
    loss, sample_loss, parts = _macro_loss(query_loss, query_valid, layout, indices)
    for family, part in _raw_parts(
        query_loss, query_valid, indices.family_part_positions
    ).items():
        parts[f"family/{family}"] = part
    for component, part in _raw_parts(
        query_loss,
        query_valid,
        indices.component_part_positions,
    ).items():
        parts[f"component/{component}"] = part

    prediction_summary: dict[str, Tensor] | None = None
    derived_f24_prediction_summary: dict[str, Tensor] | None = None
    derived_f24_predictions: Tensor | None = None
    f24_parts: dict[str, dict[str, Tensor]] = {}
    f24_status = "not_evaluated"
    if normalizer is not None:
        prediction_summary = summarize_typed_predictions(
            outputs,
            target_contract,
            normalizer,
        )
        predictions = prediction_summary["expected_raw_value"]
        from trauma_predict.eval.f24_composition import derive_f24_prediction_summary

        has_f24_contract = (
            getattr(target_contract, "derived_primary_f24_indices", None) is not None
            or (
                isinstance(target_contract, Mapping)
                and target_contract.get("derived_primary_f24_indices") is not None
            )
        )
        if has_f24_contract:
            derived_f24_prediction_summary = derive_f24_prediction_summary(
                prediction_summary,
                target_contract,
            )
            derived_f24_predictions = derived_f24_prediction_summary["expected_raw_value"]
            from trauma_predict.eval.multires_event import evaluate_f24_raw

            f24_evaluation = evaluate_f24_raw(
                prediction_summary,
                target_contract,
                normalizer=normalizer,
                f24_target_raw_values=batch.get("f24_target_raw_values"),
                f24_target_mask=batch.get("f24_target_mask"),
            )
            f24_status = str(f24_evaluation["status"])
            f24_parts = f24_evaluation.get("parts", {})

    result: dict[str, Any] = {
        "loss": loss,
        "loss_numerator": parts["total"]["numerator"],
        "loss_denominator": parts["total"]["denominator"],
        "parts": parts,
        "sample_loss": sample_loss,
        "predictions": predictions,
        "prediction_mask": model_query_mask.bool(),
        "per_query_loss": query_loss,
        "per_query_valid": query_valid,
        "prediction_summary": prediction_summary,
        "prediction_space": "raw" if prediction_summary is not None else "head_native",
        "derived_f24_prediction_summary": derived_f24_prediction_summary,
        "derived_f24_predictions": derived_f24_predictions,
        "f24_status": f24_status,
        "f24_parts": f24_parts,
    }
    return result


__all__ = ["compute_multires_loss", "summarize_typed_predictions"]
