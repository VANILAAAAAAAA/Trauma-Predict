from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from trauma_predict.modeling.multires_event_v2.emissions import (
    CORE_EMISSION_LOG_PROBS,
    DenseValueParameters,
    DenseValueTarget,
    LabValueParameters,
    LabValueTarget,
    NEDParameters,
    NEDTarget,
    RespiratoryOccupancyParameters,
    StudentTParameters,
    UOPParameters,
    ZOILogitNormalParameters,
    autoregressive_binary_vector_log_prob,
    bernoulli_log_prob,
    categorical_log_prob,
    dense_abnormal_duration_log_prob,
    dense_joint_value_log_prob,
    gcs_verbal_gradable_triple_log_prob,
    gcs_verbal_latest_status_log_prob,
    gcs_verbal_ungradable_hours_log_prob,
    hurdle_negative_binomial_log_prob,
    lab_joint_value_log_prob,
    legal_gcs_triple_log_prob,
    legal_ordinal_triples,
    ordinal_triple_class_mask,
    ned_joint_value_log_prob,
    respiratory_edge_evidence_log_prob,
    respiratory_edge_state_log_prob,
    respiratory_occupancy_log_prob,
    respiratory_onset_log_prob,
    sample_autoregressive_binary_vector,
    sample_autoregressive_hurdle_count_vector,
    sample_autoregressive_zoi_logit_normal_vector,
    sample_categorical,
    sample_dense_abnormal_duration,
    sample_hurdle_negative_binomial,
    sample_student_t,
    sample_zoi_logit_normal,
    uop_sum_log_prob,
    vasopressor_duration_log_prob,
    vasopressor_onset_log_prob,
)


REGISTERED_CORE_FIELD_IDS = (
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    9,
    10,
    11,
    12,
    13,
    7,
    *range(14, 29),
    35,
)
V2_PROCESS_REGISTRY_VERSION = "2026-07-14-r9"
V2_PROCESS_REGISTRY_SHA256 = "2cd5fd86e42f2dc582080a1d147495a24ac6eebb5c9b007f9575918a79f2b33b"
V2_EMISSION_REGISTRY_VERSION = "2026-07-14-r9"
V2_EMISSION_REGISTRY_SHA256 = "d41a0965e0ba2170c28c35c0320fc5c78247982548ba354cb8c137113ae6f48c"
EXPECTED_ENABLED_CORE_PRIMITIVES = 414

# One shared parameter bank is emitted at every [block, registered-field] state.
# The registry selects exactly one applicable bank for each stochastic primitive.
V2_PRIMITIVE_HEAD_DIMS: Mapping[str, int] = {
    "categorical_hours_0_4": 5,
    "dense_joint_value_state": 27,
    "dense_abnormal_duration_vector": 30,
    "gcs_ordinal_triple": 56,
    "gcs_verbal_ungradable_hours_given_observed": 5,
    "gcs_verbal_latest_status": 2,
    "gcs_verbal_gradable_ordinal_triple": 35,
    "hurdle_negative_binomial_count": 3,
    "lab_joint_value_state": 18,
    "respiratory_block_evidence": 1,
    "respiratory_edge_evidence_given_block": 1,
    "respiratory_occupancy_vector": 39,
    "respiratory_edge_state": 4,
    "respiratory_onset_vector": 30,
    "vasopressor_duration_vector": 105,
    "vasopressor_edge_state_vector": 21,
    "vasopressor_onset_vector": 63,
    "ned_joint_value_state": 14,
    "uop_sum_given_count": 4,
}

# Raw structured truth widths consumed by the feedback encoder. Scalar banks are
# represented with one channel when sampled for feedback, even though collated
# training tensors may omit the final singleton dimension. Physical-unit feedback
# transformation (signed_log1p) belongs to the model encoder, never to likelihoods.
V2_PRIMITIVE_FEEDBACK_DIMS: Mapping[str, int] = {
    "categorical_hours_0_4": 1,
    "dense_joint_value_state": 4,
    "dense_abnormal_duration_vector": 2,
    "gcs_ordinal_triple": 3,
    "gcs_verbal_ungradable_hours_given_observed": 1,
    "gcs_verbal_latest_status": 1,
    "gcs_verbal_gradable_ordinal_triple": 3,
    "hurdle_negative_binomial_count": 1,
    "lab_joint_value_state": 3,
    "respiratory_block_evidence": 1,
    "respiratory_edge_evidence_given_block": 1,
    "respiratory_occupancy_vector": 5,
    "respiratory_edge_state": 1,
    "respiratory_onset_vector": 4,
    "vasopressor_duration_vector": 6,
    "vasopressor_edge_state_vector": 6,
    "vasopressor_onset_vector": 6,
    "ned_joint_value_state": 3,
    "uop_sum_given_count": 1,
}

V2_FLOAT64_PHYSICAL_TARGETS = frozenset(
    {
        "dense_joint_value_state",
        "lab_joint_value_state",
        "respiratory_occupancy_vector",
        "vasopressor_duration_vector",
        "ned_joint_value_state",
        "uop_sum_given_count",
    }
)


def _require_tensor_condition(
    condition: Tensor,
    message: str,
    error_type: type[Exception],
) -> None:
    """Fail closed without forcing CUDA reductions through the host."""

    if condition.numel() == 0:
        raise error_type(message)
    valid = condition.all()
    if condition.device.type == "cuda":
        torch._assert_async(valid, message)
    elif not bool(valid.item()):
        raise error_type(message)


def validate_emission_registry_head_contract(
    emission_registry: Mapping[str, Any],
) -> None:
    """Fail closed when the attached emission registry and model heads drift."""

    if emission_registry.get("version") != V2_EMISSION_REGISTRY_VERSION:
        raise ValueError(
            "enabled-core heads require emission registry "
            f"{V2_EMISSION_REGISTRY_VERSION}, got {emission_registry.get('version')!r}"
        )
    global_contract = emission_registry.get("global_contract")
    numerical = (
        global_contract.get("numerical_constants")
        if isinstance(global_contract, Mapping)
        else None
    )
    if not isinstance(numerical, Mapping):
        raise ValueError("emission registry lacks global_contract.numerical_constants")
    expected_numerical = {
        "positive_scale_floor": 0.0001,
        "unit_interval_interior_family": "zero_one_inflated_logit_normal",
        "unit_interval_interior_measure": "Lebesgue_dq",
    }
    observed_numerical = {
        key: numerical.get(key) for key in expected_numerical
    }
    if observed_numerical != expected_numerical:
        raise ValueError(
            "emission numerical constants differ from the r8 model contract: "
            f"{observed_numerical}"
        )
    head_contract = emission_registry.get("enabled_core_head_contract")
    if not isinstance(head_contract, Mapping):
        raise ValueError("emission registry lacks enabled_core_head_contract")
    layouts = head_contract.get("layouts")
    if not isinstance(layouts, Mapping) or set(layouts) != set(V2_PRIMITIVE_HEAD_DIMS):
        raise ValueError("emission head layouts do not exactly cover the 19 enabled likelihoods")
    observed = {
        str(likelihood_id): int(row["width"])
        for likelihood_id, row in layouts.items()
        if isinstance(row, Mapping) and "width" in row
    }
    if observed != dict(V2_PRIMITIVE_HEAD_DIMS):
        raise ValueError(f"emission head widths differ from the model contract: {observed}")


@dataclass(frozen=True)
class EnabledPrimitiveSpec:
    primitive_id: str
    log_prob_id: str
    likelihood_id: str
    field: str
    field_id: int
    field_index: int
    block: str
    block_index: int
    process_order: int
    primitive_order: int


@dataclass(frozen=True)
class PrimitiveLogProb:
    primitive_id: str
    likelihood_id: str
    log_prob: Tensor
    objective_status: str = "enabled_core"
    source_kind: str = "stochastic_primitive"


def _format(template: str, *, field: str, block: str) -> str:
    return template.format(field=field, block=block)


def expand_enabled_core_primitives(registry: Mapping[str, Any]) -> tuple[EnabledPrimitiveSpec, ...]:
    """Expand the process registry in block-major, registered-field order."""

    if registry.get("version") != V2_PROCESS_REGISTRY_VERSION:
        raise ValueError(
            "enabled-core loss requires process registry "
            f"{V2_PROCESS_REGISTRY_VERSION}, got {registry.get('version')!r}"
        )
    registered = registry.get("registered_core_field_order")
    if not isinstance(registered, Sequence):
        raise ValueError("registry must define registered_core_field_order")
    ordered_rows = sorted(registered, key=lambda row: int(row["position"]))
    positions = [int(row["position"]) for row in ordered_rows]
    field_ids = tuple(int(row["field_id"]) for row in ordered_rows)
    if positions != list(range(len(ordered_rows))):
        raise ValueError("registered core field positions must be contiguous from zero")
    if field_ids != REGISTERED_CORE_FIELD_IDS:
        raise ValueError(
            "registered core field IDs must match the r8 explicit order "
            f"{REGISTERED_CORE_FIELD_IDS}, got {field_ids}"
        )
    field_lookup = {
        str(row["field"]): (int(row["field_id"]), int(row["position"])) for row in ordered_rows
    }
    blocks = tuple(str(block) for block in registry["scope"]["future_blocks"])
    if len(blocks) != 6:
        raise ValueError("V2 requires exactly six future M4 blocks")
    field_sets = registry["field_sets"]
    expanded: list[EnabledPrimitiveSpec] = []
    seen_primitives: set[str] = set()
    seen_log_probs: set[str] = set()
    for process_order, process in enumerate(registry["process_templates"]):
        if process["objective_status"] != "enabled_core":
            continue
        if process["scope"] != "per_block":
            raise ValueError("the initial enabled-core objective only supports per-block processes")
        fields = field_sets[process["fields_from"]]
        for field in fields:
            if field not in field_lookup:
                raise ValueError(f"enabled process references non-core field {field!r}")
            field_id, field_index = field_lookup[field]
            for block_index, block in enumerate(blocks):
                for primitive_order, primitive in enumerate(process["primitives"]):
                    primitive_id = _format(
                        primitive["primitive_id_template"], field=field, block=block
                    )
                    log_prob_id = _format(
                        primitive["log_prob_id_template"], field=field, block=block
                    )
                    if primitive_id in seen_primitives:
                        raise ValueError(f"duplicate enabled primitive ID: {primitive_id}")
                    if log_prob_id in seen_log_probs:
                        raise ValueError(f"duplicate enabled log-prob ID: {log_prob_id}")
                    likelihood_id = str(primitive["likelihood_id"])
                    if likelihood_id not in CORE_EMISSION_LOG_PROBS:
                        raise ValueError(f"no core emission for likelihood {likelihood_id!r}")
                    seen_primitives.add(primitive_id)
                    seen_log_probs.add(log_prob_id)
                    expanded.append(
                        EnabledPrimitiveSpec(
                            primitive_id=primitive_id,
                            log_prob_id=log_prob_id,
                            likelihood_id=likelihood_id,
                            field=field,
                            field_id=field_id,
                            field_index=field_index,
                            block=block,
                            block_index=block_index,
                            process_order=process_order,
                            primitive_order=primitive_order,
                        )
                    )
    expanded.sort(
        key=lambda item: (
            item.block_index,
            item.field_index,
            item.process_order,
            item.primitive_order,
        )
    )
    declared_count = registry["scope"].get("expanded_enabled_core_primitives")
    if declared_count is not None and len(expanded) != int(declared_count):
        raise ValueError(
            f"registry declares {declared_count} enabled primitives but expands to {len(expanded)}"
        )
    return tuple(expanded)


def _zoi_logit_normal_from_flat(value: Tensor, offset: int) -> ZOILogitNormalParameters:
    return ZOILogitNormalParameters(
        mixture_logits=value[..., offset : offset + 3],
        interior_loc=value[..., offset + 3],
        interior_scale_raw=value[..., offset + 4],
    )


def _student_t_from_flat(value: Tensor, offset: int) -> StudentTParameters:
    return StudentTParameters(
        location=value[..., offset],
        scale_raw=value[..., offset + 1],
        df_raw=value[..., offset + 2],
    )


def _parameter_slice(
    primitive_parameters: Mapping[str, Tensor],
    spec: EnabledPrimitiveSpec,
) -> Tensor:
    if spec.likelihood_id not in primitive_parameters:
        raise KeyError(f"model output lacks primitive head {spec.likelihood_id!r}")
    bank = primitive_parameters[spec.likelihood_id]
    width = V2_PRIMITIVE_HEAD_DIMS[spec.likelihood_id]
    if not isinstance(bank, Tensor) or bank.ndim != 4:
        raise ValueError(
            f"primitive head {spec.likelihood_id} must be [batch,6,29,width], got {bank.shape}"
        )
    if bank.shape[1:] != (6, 29, width):
        raise ValueError(
            f"primitive head {spec.likelihood_id} must end in (6,29,{width}), got {bank.shape}"
        )
    # Likelihood support/canonicalization uses float64 physical truth, but the
    # 414 density evaluations and their gradients stay in float32 on T4.
    return bank[:, spec.block_index, spec.field_index, :].float()


def _target_bank(
    target_primitives: Mapping[str, Tensor],
    likelihood_id: str,
) -> Tensor:
    if likelihood_id not in target_primitives:
        raise KeyError(f"batch lacks target primitive bank {likelihood_id!r}")
    value = target_primitives[likelihood_id]
    if not isinstance(value, Tensor) or value.ndim < 3 or value.shape[1:3] != (6, 29):
        raise ValueError(
            f"target bank {likelihood_id} must start [batch,6,29], got "
            f"{getattr(value, 'shape', None)}"
        )
    return value


def _target_slice(
    target_primitives: Mapping[str, Tensor],
    likelihood_id: str,
    spec: EnabledPrimitiveSpec,
) -> Tensor:
    bank = _target_bank(target_primitives, likelihood_id)
    return bank[:, spec.block_index, spec.field_index, ...]


def _mask_slice(
    masks: Mapping[str, Tensor],
    likelihood_id: str,
    spec: EnabledPrimitiveSpec,
) -> Tensor:
    if likelihood_id not in masks:
        raise KeyError(f"batch lacks target primitive mask {likelihood_id!r}")
    mask = masks[likelihood_id]
    if not isinstance(mask, Tensor) or mask.ndim != 3 or mask.shape[1:] != (6, 29):
        raise ValueError(f"target mask {likelihood_id} must be [batch,6,29]")
    return mask[:, spec.block_index, spec.field_index].bool()


def _validate_batch_metadata(metadata: Mapping[str, Any], registry: Mapping[str, Any]) -> None:
    raw_field_ids = metadata.get("field_ids")
    if raw_field_ids is None:
        raise ValueError("target_primitive_metadata must include field_ids")
    registered_ids = tuple(
        int(row["field_id"])
        for row in sorted(registry["registered_core_field_order"], key=lambda row: row["position"])
    )
    if registered_ids != REGISTERED_CORE_FIELD_IDS:
        raise ValueError(
            f"process registry does not carry the r8 explicit field order: {registered_ids}"
        )
    if isinstance(raw_field_ids, Tensor):
        if raw_field_ids.shape != (len(registered_ids),):
            raise ValueError("target field_ids tensor must contain the registered field axis")
        expected = torch.tensor(
            registered_ids,
            dtype=raw_field_ids.dtype,
            device=raw_field_ids.device,
        )
        _require_tensor_condition(
            raw_field_ids.eq(expected),
            f"target field axis must be registry order {registered_ids}",
            ValueError,
        )
        field_ids = registered_ids
    else:
        field_ids = tuple(int(value) for value in raw_field_ids)
    if field_ids != registered_ids:
        raise ValueError(
            f"target field axis must be registry order {registered_ids}, got {field_ids}"
        )


def _valid_range(metadata: Mapping[str, Any], spec: EnabledPrimitiveSpec) -> tuple[float, float]:
    ranges = metadata.get("valid_ranges")
    if not isinstance(ranges, Mapping):
        raise ValueError("target_primitive_metadata must include valid_ranges for dense/lab fields")
    candidate = None
    for key in (spec.field, spec.field_id, str(spec.field_id)):
        if key in ranges:
            candidate = ranges[key]
            break
    if not isinstance(candidate, Sequence) or len(candidate) != 2:
        raise ValueError(f"valid range is missing for field {spec.field!r}")
    lower, upper = float(candidate[0]), float(candidate[1])
    if not upper > lower:
        raise ValueError(f"invalid registered range for field {spec.field!r}")
    return lower, upper


def _validate_lab_scale_metadata(
    metadata: Mapping[str, Any],
    registry: Mapping[str, Any],
    expected_artifact_hash: str,
) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected_artifact_hash)):
        raise ValueError("expected_lab_scale_artifact_hash must be a lowercase SHA-256")
    contract = metadata.get("lab_scale")
    if not isinstance(contract, Mapping):
        raise ValueError("target_primitive_metadata must include lab_scale")
    if contract.get("schema") != "multires_event_v2_lab_affine_scale_v1":
        raise ValueError("lab scale schema does not match the frozen runtime contract")
    if contract.get("version") != "2026-07-13-train-target-windows-v1":
        raise ValueError("lab scale version does not match the frozen runtime contract")
    if contract.get("coordinate_contract") != "lab_shared_affine_canonical_v1":
        raise ValueError("lab scale coordinate contract is not canonical V1")
    actual_hash = str(contract.get("content_sha256", ""))
    if actual_hash != expected_artifact_hash:
        raise ValueError(
            "lab scale artifact hash mismatch: "
            f"expected {expected_artifact_hash}, got {actual_hash or '<missing>'}"
        )
    fields = contract.get("fields")
    if not isinstance(fields, Mapping):
        raise ValueError("lab scale metadata must contain a fields mapping")
    expected_fields = tuple(str(field) for field in registry["field_sets"]["intermittent_labs"])
    if set(fields) != set(expected_fields) or len(fields) != len(expected_fields):
        raise ValueError("lab scale fields must exactly equal the 13 registered lab fields")
    for field in expected_fields:
        row = fields[field]
        if not isinstance(row, Mapping) or set(row) != {"unit", "center", "scale"}:
            raise ValueError(f"lab scale row for {field!r} must contain unit/center/scale")
        if not isinstance(row["unit"], str) or not row["unit"]:
            raise ValueError(f"lab scale unit is missing for {field!r}")
        center = float(row["center"])
        scale = float(row["scale"])
        if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"lab scale for {field!r} must be finite and strictly positive")


def _lab_affine(metadata: Mapping[str, Any], spec: EnabledPrimitiveSpec) -> tuple[float, float]:
    row = metadata["lab_scale"]["fields"][spec.field]
    return float(row["center"]), float(row["scale"])


def _cross_target(
    target_primitives: Mapping[str, Tensor],
    likelihood_id: str,
    spec: EnabledPrimitiveSpec,
) -> Tensor:
    return _target_slice(target_primitives, likelihood_id, spec)


def _assert_activation(mask: Tensor, expected: Tensor, primitive_id: str) -> None:
    if mask.shape != expected.shape:
        raise ValueError(f"target mask violates the registered activation gate for {primitive_id}")
    _require_tensor_condition(
        mask.eq(expected.bool()),
        f"target mask violates the registered activation gate for {primitive_id}",
        ValueError,
    )


def _evaluate_log_prob(
    spec: EnabledPrimitiveSpec,
    raw_parameters: Tensor,
    target_primitives: Mapping[str, Tensor],
    mask: Tensor,
    metadata: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> Tensor:
    likelihood = spec.likelihood_id
    target = _target_slice(target_primitives, likelihood, spec)

    if likelihood == "categorical_hours_0_4":
        _assert_activation(mask, torch.ones_like(mask), spec.primitive_id)
        return categorical_log_prob(raw_parameters, target)

    if likelihood == "dense_joint_value_state":
        observed = _cross_target(target_primitives, "categorical_hours_0_4", spec)
        _assert_activation(mask, observed.gt(0), spec.primitive_id)
        if target.shape[-1:] != (4,):
            raise ValueError("dense target order must be (LAST,MIN,MAX,MEAN)")
        parameters = DenseValueParameters(
            range_logits=raw_parameters[..., :2],
            constant_value=_zoi_logit_normal_from_flat(raw_parameters, 2),
            minimum_coordinate=_zoi_logit_normal_from_flat(raw_parameters, 7),
            range_coordinate=_zoi_logit_normal_from_flat(raw_parameters, 12),
            last_coordinate=_zoi_logit_normal_from_flat(raw_parameters, 17),
            mean_coordinate=_zoi_logit_normal_from_flat(raw_parameters, 22),
        )
        lower, upper = _valid_range(metadata, spec)
        return dense_joint_value_log_prob(
            parameters,
            DenseValueTarget(
                observed, target[..., 1], target[..., 0], target[..., 2], target[..., 3]
            ),
            lower=lower,
            upper=upper,
        )

    if likelihood == "dense_abnormal_duration_vector":
        observed = _cross_target(target_primitives, "categorical_hours_0_4", spec)
        _assert_activation(mask, observed.gt(0), spec.primitive_id)
        conditions = tuple(registry["condition_sets"]["dense_abnormal"][spec.field])
        condition_count = len(conditions)
        if target.shape[-1] < condition_count:
            raise ValueError(f"dense abnormal target lacks channels for {spec.field}")
        dense_state = _cross_target(target_primitives, "dense_joint_value_state", spec)
        if dense_state.shape[-1:] != (4,):
            raise ValueError("dense target order must be (LAST,MIN,MAX,MEAN)")
        return dense_abnormal_duration_log_prob(
            raw_parameters,
            target[..., :condition_count],
            observed,
            field=spec.field,
            condition_keys=conditions,
            minimum=dense_state[..., 1],
            maximum=dense_state[..., 2],
        )

    if likelihood == "gcs_ordinal_triple":
        observed = _cross_target(target_primitives, "categorical_hours_0_4", spec)
        _assert_activation(mask, observed.gt(0), spec.primitive_id)
        if target.shape[-1:] != (3,):
            raise ValueError("GCS target order must be (LAST,MIN,MAX)")
        field_parameters = registry["field_parameters"][spec.field]
        maximum = int(field_parameters["ordinal_max"])
        classes = int(field_parameters["legal_triple_count"])
        triple = torch.stack((target[..., 1], target[..., 0], target[..., 2]), dim=-1)
        safe_triple = torch.where(observed.gt(0).unsqueeze(-1), triple, torch.ones_like(triple))
        return legal_gcs_triple_log_prob(
            raw_parameters[..., :classes],
            safe_triple,
            maximum=maximum,
            observation_count=observed,
            source_semantics="raw_point",
        )

    if likelihood == "gcs_verbal_ungradable_hours_given_observed":
        observed = _cross_target(target_primitives, "categorical_hours_0_4", spec)
        _assert_activation(mask, torch.ones_like(mask), spec.primitive_id)
        return gcs_verbal_ungradable_hours_log_prob(raw_parameters, target, observed)

    if likelihood == "gcs_verbal_latest_status":
        observed = _cross_target(target_primitives, "categorical_hours_0_4", spec)
        ungradable = _cross_target(
            target_primitives, "gcs_verbal_ungradable_hours_given_observed", spec
        )
        _assert_activation(mask, observed.gt(0), spec.primitive_id)
        # Collator IDs are 0=undefined, 1=GRADABLE, 2=UNGRADABLE.
        status = torch.where(observed.gt(0), target - 1, torch.zeros_like(target))
        return gcs_verbal_latest_status_log_prob(
            raw_parameters,
            status,
            observed,
            ungradable,
        )

    if likelihood == "gcs_verbal_gradable_ordinal_triple":
        observed = _cross_target(target_primitives, "categorical_hours_0_4", spec)
        ungradable = _cross_target(
            target_primitives, "gcs_verbal_ungradable_hours_given_observed", spec
        )
        _assert_activation(mask, observed.gt(ungradable), spec.primitive_id)
        if target.shape[-1:] != (3,):
            raise ValueError("GCS verbal target order must be (LAST,MIN,MAX)")
        triple = torch.stack((target[..., 1], target[..., 0], target[..., 2]), dim=-1)
        return gcs_verbal_gradable_triple_log_prob(
            raw_parameters,
            triple,
            observed,
            ungradable,
        )

    if likelihood == "hurdle_negative_binomial_count":
        _assert_activation(mask, torch.ones_like(mask), spec.primitive_id)
        return hurdle_negative_binomial_log_prob(
            target,
            raw_parameters[..., 0],
            raw_parameters[..., 1],
            raw_parameters[..., 2],
        )

    if likelihood == "lab_joint_value_state":
        count = _cross_target(target_primitives, "hurdle_negative_binomial_count", spec)
        _assert_activation(mask, count.gt(0), spec.primitive_id)
        if target.shape[-1:] != (3,):
            raise ValueError("lab target order must be (LAST,MIN,MAX)")
        parameters = LabValueParameters(
            single_value=_student_t_from_flat(raw_parameters, 0),
            range_logits=raw_parameters[..., 3:5],
            constant_value=_student_t_from_flat(raw_parameters, 5),
            minimum=_student_t_from_flat(raw_parameters, 8),
            log_range_loc=raw_parameters[..., 11],
            log_range_scale_raw=raw_parameters[..., 12],
            last_coordinate=_zoi_logit_normal_from_flat(raw_parameters, 13),
        )
        center, scale = _lab_affine(metadata, spec)
        standardized = (target - center) / scale
        return lab_joint_value_log_prob(
            parameters,
            LabValueTarget(
                count,
                standardized[..., 1],
                standardized[..., 0],
                standardized[..., 2],
            ),
        )

    if likelihood == "respiratory_block_evidence":
        _assert_activation(mask, torch.ones_like(mask), spec.primitive_id)
        return bernoulli_log_prob(raw_parameters[..., 0], target)

    if likelihood == "respiratory_edge_evidence_given_block":
        block = _cross_target(target_primitives, "respiratory_block_evidence", spec)
        _assert_activation(mask, torch.ones_like(mask), spec.primitive_id)
        return respiratory_edge_evidence_log_prob(raw_parameters[..., 0], target, block)

    if likelihood == "respiratory_occupancy_vector":
        block = _cross_target(target_primitives, "respiratory_block_evidence", spec)
        _assert_activation(mask, block.gt(0), spec.primitive_id)
        if target.shape[-1:] != (5,):
            raise ValueError("respiratory occupancy order must be four modalities then uncovered")
        durations = torch.cat((target[..., 4:5], target[..., :4]), dim=-1)
        return respiratory_occupancy_log_prob(
            RespiratoryOccupancyParameters(
                active_set_logits=raw_parameters[..., :31],
                alr_location=raw_parameters[..., 31:35],
                alr_scale_raw=raw_parameters[..., 35:39],
            ),
            durations,
            block_evidence=block,
        )

    if likelihood == "respiratory_edge_state":
        edge = _cross_target(target_primitives, "respiratory_edge_evidence_given_block", spec)
        _assert_activation(mask, edge.gt(0), spec.primitive_id)
        state = torch.where(edge.gt(0), target - 1, torch.zeros_like(target))
        return respiratory_edge_state_log_prob(raw_parameters, state, edge)

    if likelihood == "respiratory_onset_vector":
        block = _cross_target(target_primitives, "respiratory_block_evidence", spec)
        edge = _cross_target(
            target_primitives,
            "respiratory_edge_evidence_given_block",
            spec,
        )
        edge_state_raw = _cross_target(target_primitives, "respiratory_edge_state", spec)
        edge_state = torch.where(edge.gt(0), edge_state_raw - 1, torch.zeros_like(edge_state_raw))
        occupancy = _cross_target(target_primitives, "respiratory_occupancy_vector", spec)
        _assert_activation(mask, block.gt(0), spec.primitive_id)
        if target.shape[-1:] != (4,):
            raise ValueError("respiratory onset target must have four modalities")
        return respiratory_onset_log_prob(
            target,
            raw_parameters,
            block,
            occupancy[..., :4],
            edge,
            edge_state,
        )

    if likelihood == "vasopressor_duration_vector":
        _assert_activation(mask, torch.ones_like(mask), spec.primitive_id)
        if target.shape[-1:] != (6,):
            raise ValueError("vasopressor duration target must have six agents")
        return vasopressor_duration_log_prob(target, raw_parameters)

    if likelihood == "vasopressor_edge_state_vector":
        _assert_activation(mask, torch.ones_like(mask), spec.primitive_id)
        return autoregressive_binary_vector_log_prob(raw_parameters, target)

    if likelihood == "vasopressor_onset_vector":
        _assert_activation(mask, torch.ones_like(mask), spec.primitive_id)
        vaso_duration = _cross_target(target_primitives, "vasopressor_duration_vector", spec)
        vaso_edge = _cross_target(target_primitives, "vasopressor_edge_state_vector", spec)
        return vasopressor_onset_log_prob(
            target,
            raw_parameters,
            vaso_duration,
            vaso_edge,
        )

    if likelihood == "ned_joint_value_state":
        _assert_activation(mask, torch.ones_like(mask), spec.primitive_id)
        if target.shape[-1:] != (3,):
            raise ValueError("NED target order must be (LAST,MAX,MEAN)")
        vaso_spec = EnabledPrimitiveSpec(
            **{
                **spec.__dict__,
                "field": "vasopressor_support",
                "field_id": 27,
                "field_index": next(
                    int(row["position"])
                    for row in registry["registered_core_field_order"]
                    if int(row["field_id"]) == 27
                ),
                "likelihood_id": "vasopressor_duration_vector",
            }
        )
        vaso_duration = _cross_target(target_primitives, "vasopressor_duration_vector", vaso_spec)
        vaso_edge = _cross_target(target_primitives, "vasopressor_edge_state_vector", vaso_spec)
        compatible_duration = vaso_duration[..., :5].gt(0).any(dim=-1)
        compatible_edge = vaso_edge[..., :5].gt(0).any(dim=-1)
        return ned_joint_value_log_prob(
            NEDParameters(
                zero_positive_logits=raw_parameters[..., :2],
                positive_max_loc=raw_parameters[..., 2],
                positive_max_scale_raw=raw_parameters[..., 3],
                last_ratio=_zoi_logit_normal_from_flat(raw_parameters, 4),
                mean_ratio=_zoi_logit_normal_from_flat(raw_parameters, 9),
            ),
            NEDTarget(
                target[..., 1],
                target[..., 0],
                target[..., 2],
                compatible_duration,
                compatible_edge,
            ),
        )

    if likelihood == "uop_sum_given_count":
        count = _cross_target(target_primitives, "hurdle_negative_binomial_count", spec)
        _assert_activation(mask, count.gt(0), spec.primitive_id)
        return uop_sum_log_prob(
            UOPParameters(
                zero_positive_logits=raw_parameters[..., :2],
                positive_loc=raw_parameters[..., 2],
                positive_scale_raw=raw_parameters[..., 3],
            ),
            target,
            count,
        )

    raise AssertionError(f"unhandled enabled-core likelihood: {likelihood}")


def _sample_dense_value_state(
    raw: Tensor,
    observed_hours: Tensor,
    *,
    lower: float,
    upper: float,
) -> Tensor:
    active = observed_hours.gt(0)
    branch = sample_categorical(raw[..., :2])
    constant_coordinate = sample_zoi_logit_normal(
        _zoi_logit_normal_from_flat(raw, 2)
    ).double()
    lower_tensor = torch.as_tensor(lower, dtype=torch.float64, device=raw.device)
    upper_tensor = torch.as_tensor(upper, dtype=torch.float64, device=raw.device)
    physical_span = upper_tensor - lower_tensor
    constant = lower_tensor + physical_span * constant_coordinate
    no_upper_atom = torch.tensor([True, True, False], device=raw.device)
    no_zero_atom = torch.tensor([False, True, True], device=raw.device)
    alpha = sample_zoi_logit_normal(
        _zoi_logit_normal_from_flat(raw, 7), component_mask=no_upper_atom
    ).double()
    minimum_positive = lower_tensor + physical_span * alpha
    beta = sample_zoi_logit_normal(
        _zoi_logit_normal_from_flat(raw, 12), component_mask=no_zero_atom
    ).double()
    value_range = (upper_tensor - minimum_positive) * beta
    maximum_positive = minimum_positive + value_range
    last_coordinate = sample_zoi_logit_normal(
        _zoi_logit_normal_from_flat(raw, 17)
    ).double()
    last_positive = minimum_positive + value_range * last_coordinate
    mean_coordinate = sample_zoi_logit_normal(
        _zoi_logit_normal_from_flat(raw, 22)
    ).double()
    hours = observed_hours.clamp_min(1).double()
    lower_mean = (last_positive + (hours - 1.0) * minimum_positive) / hours
    upper_mean = (last_positive + (hours - 1.0) * maximum_positive) / hours
    mean_positive = lower_mean + (upper_mean - lower_mean) * mean_coordinate
    mean_positive = torch.where(mean_coordinate.eq(0.0), lower_mean, mean_positive)
    mean_positive = torch.where(mean_coordinate.eq(1.0), upper_mean, mean_positive)
    mean_positive = torch.where(observed_hours.eq(1), last_positive, mean_positive)
    zero_range = branch.eq(0)
    minimum = torch.where(zero_range, constant, minimum_positive)
    maximum = torch.where(zero_range, constant, maximum_positive)
    last = torch.where(zero_range, constant, last_positive)
    mean = torch.where(zero_range, constant, mean_positive)
    result = torch.stack((last, minimum, maximum, mean), dim=-1)
    return torch.where(active.unsqueeze(-1), result, torch.zeros_like(result))


def _sample_lab_value_state(
    raw: Tensor,
    observation_count: Tensor,
    *,
    center: float,
    scale: float,
) -> Tensor:
    active = observation_count.gt(0)
    single = sample_student_t(_student_t_from_flat(raw, 0)).double()
    branch = sample_categorical(raw[..., 3:5])
    constant = sample_student_t(_student_t_from_flat(raw, 5)).double()
    minimum_positive = sample_student_t(_student_t_from_flat(raw, 8)).double()
    log_range = torch.distributions.Normal(
        raw[..., 11],
        F.softplus(raw[..., 12]) + 1e-4,
        validate_args=False,
    ).sample().double()
    value_range = torch.exp(log_range)
    maximum_positive = minimum_positive + value_range
    endpoint_components = torch.tensor([True, False, True], dtype=torch.bool, device=raw.device)
    all_components = torch.ones(
        observation_count.shape + (3,),
        dtype=torch.bool,
        device=raw.device,
    )
    last_mask = torch.where(
        observation_count.eq(2).unsqueeze(-1),
        endpoint_components,
        all_components,
    )
    last_coordinate = sample_zoi_logit_normal(
        _zoi_logit_normal_from_flat(raw, 13),
        component_mask=last_mask,
    ).double()
    last_positive = minimum_positive + value_range * last_coordinate
    zero_range = branch.eq(0)
    repeated_minimum = torch.where(zero_range, constant, minimum_positive)
    repeated_maximum = torch.where(zero_range, constant, maximum_positive)
    repeated_last = torch.where(zero_range, constant, last_positive)
    single_branch = observation_count.eq(1)
    minimum = torch.where(single_branch, single, repeated_minimum)
    maximum = torch.where(single_branch, single, repeated_maximum)
    last = torch.where(single_branch, single, repeated_last)
    scale_tensor = torch.as_tensor(scale, dtype=torch.float64, device=raw.device)
    center_tensor = torch.as_tensor(center, dtype=torch.float64, device=raw.device)
    result = torch.stack((last, minimum, maximum), dim=-1) * scale_tensor + center_tensor
    return torch.where(active.unsqueeze(-1), result, torch.zeros_like(result))


def _sample_ordinal_triple(
    logits: Tensor,
    *,
    maximum: int,
    observation_count: Tensor,
    source_semantics: Literal["raw_point", "hourly_sequence"],
) -> Tensor:
    states = legal_ordinal_triples(maximum, device=logits.device)
    valid = ordinal_triple_class_mask(
        states,
        observation_count,
        source_semantics=source_semantics,
    )
    index = sample_categorical(logits, valid)
    triple = states[index]
    # Emission state order is MIN,LAST,MAX; collator feedback order is LAST,MIN,MAX.
    return torch.stack((triple[..., 1], triple[..., 0], triple[..., 2]), dim=-1)


def _sample_respiratory_occupancy(raw: Tensor, active: Tensor) -> Tensor:
    active_index = sample_categorical(raw[..., :31])
    # Draw the maximum ALR width for every row before branch selection.  Using
    # a branch-dependent Normal shape would advance the global RNG by a
    # mode-dependent amount and invalidate factor-aligned common random numbers
    # for all downstream primitives.  Slicing independent coordinates preserves
    # every lower-dimensional ALR marginal while fixing draw consumption.
    full_alr = torch.distributions.Normal(
        raw[..., 31:35].double(),
        F.softplus(raw[..., 35:39].double()) + 1e-4,
        validate_args=False,
    ).sample()
    codes = active_index + 1
    bit_values = torch.tensor((1, 2, 4, 8, 16), device=raw.device)
    selected = codes.unsqueeze(-1).bitwise_and(bit_values).ne(0)
    component_count = selected.sum(dim=-1)
    selected_rank = selected.cumsum(dim=-1) - 1
    duration = full_alr.new_zeros((raw.shape[0], 5))
    for cardinality in range(1, 6):
        logits = torch.cat(
            (full_alr[..., : cardinality - 1], full_alr.new_zeros((raw.shape[0], 1))),
            dim=-1,
        )
        proportion = torch.softmax(logits, dim=-1)
        ranked = proportion.gather(
            -1,
            selected_rank.clamp(min=0, max=cardinality - 1),
        )
        row_mask = active.bool() & component_count.eq(cardinality)
        duration = torch.where(
            row_mask.unsqueeze(-1) & selected,
            ranked * 4.0,
            duration,
        )
    # Internal order is uncovered plus modalities; feedback order is modalities then uncovered.
    return torch.cat((duration[..., 1:], duration[..., :1]), dim=-1)


def _require_finite_sampling_parameters(parameters: Tensor, position: int) -> None:
    message = f"non-finite raw sampling parameters at block-major position {position}"
    _require_tensor_condition(torch.isfinite(parameters), message, FloatingPointError)


def _sample_ned_state(
    raw: Tensor,
    compatible_duration: Tensor,
    compatible_edge: Tensor,
) -> Tensor:
    valid_branch = torch.stack(
        (torch.ones_like(compatible_duration), compatible_duration),
        dim=-1,
    )
    positive = sample_categorical(raw[..., :2], valid_branch).bool()
    log_maximum = torch.distributions.Normal(
        raw[..., 2],
        F.softplus(raw[..., 3]) + 1e-4,
        validate_args=False,
    ).sample()
    maximum = torch.exp(log_maximum.double())
    last_mask = torch.stack(
        (
            torch.ones_like(compatible_edge),
            compatible_edge,
            compatible_edge,
        ),
        dim=-1,
    )
    mean_mask = torch.tensor(
        [False, True, True],
        dtype=torch.bool,
        device=raw.device,
    )
    last_ratio = sample_zoi_logit_normal(
        _zoi_logit_normal_from_flat(raw, 4),
        component_mask=last_mask,
    ).double()
    mean_ratio = sample_zoi_logit_normal(
        _zoi_logit_normal_from_flat(raw, 9),
        component_mask=mean_mask,
    ).double()
    maximum = torch.where(positive, maximum, torch.zeros_like(maximum))
    last = torch.where(positive, maximum * last_ratio, torch.zeros_like(maximum))
    mean = torch.where(positive, maximum * mean_ratio, torch.zeros_like(maximum))
    return torch.stack((last, maximum, mean), dim=-1)


def _sample_uop_sum(raw: Tensor, observation_count: Tensor) -> Tensor:
    active = observation_count.gt(0)
    positive = sample_categorical(raw[..., :2]).bool()
    log_amount = torch.distributions.Normal(
        raw[..., 2],
        F.softplus(raw[..., 3]) + 1e-4,
        validate_args=False,
    ).sample()
    amount = torch.exp(log_amount.double())
    return torch.where(active & positive, amount, torch.zeros_like(amount))


class RegistryPrimitiveSampler:
    """Stateful rollout sampler for the r8 registered stochastic process.

    One instance serves one sequential block-major rollout.  It returns every
    feedback bank at every field position, with component masks marking only
    the primitives registered and active at that position.  Cross-primitive
    gates and lower-triangular vector conditionals use samples generated earlier
    in the same call or earlier registered field positions.
    """

    def __init__(
        self,
        registry: Mapping[str, Any],
        metadata: Mapping[str, Any],
        *,
        expected_lab_scale_artifact_hash: str,
    ) -> None:
        _validate_batch_metadata(metadata, registry)
        _validate_lab_scale_metadata(metadata, registry, expected_lab_scale_artifact_hash)
        self.registry = registry
        self.metadata = metadata
        self.specs = expand_enabled_core_primitives(registry)
        grouped: dict[tuple[int, int], list[EnabledPrimitiveSpec]] = {}
        for spec in self.specs:
            grouped.setdefault((spec.block_index, spec.field_index), []).append(spec)
        self.grouped_specs = {key: tuple(value) for key, value in grouped.items()}
        self._next_position = 0
        self._history: dict[tuple[int, int, str], Tensor] = {}

    def reset(self) -> None:
        self._next_position = 0
        self._history.clear()

    def required_likelihood_ids(
        self,
        block_index: int,
        field_index: int,
    ) -> tuple[str, ...]:
        """Return r8 heads used at one position in registered process order."""

        specs = self.grouped_specs.get((int(block_index), int(field_index)), ())
        return tuple(dict.fromkeys(spec.likelihood_id for spec in specs))

    def __call__(
        self,
        block_index: int,
        field_index: int,
        parameters: Mapping[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        position = int(block_index) * 29 + int(field_index)
        if position == 0:
            self.reset()
        if position != self._next_position:
            raise ValueError(
                "registry sampler requires one block-major call per field; "
                f"expected position {self._next_position}, got {position}"
            )
        if set(parameters) != set(V2_PRIMITIVE_HEAD_DIMS):
            missing = set(V2_PRIMITIVE_HEAD_DIMS).difference(parameters)
            extra = set(parameters).difference(V2_PRIMITIVE_HEAD_DIMS)
            raise ValueError(
                f"sampler head mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        reference = next(iter(parameters.values()))
        if reference.ndim != 2:
            raise ValueError("rollout primitive parameters must be [batch,width]")
        batch_size = reference.shape[0]
        for likelihood_id, width in V2_PRIMITIVE_HEAD_DIMS.items():
            value = parameters[likelihood_id]
            if value.shape != (batch_size, width):
                raise ValueError(
                    f"sampler head {likelihood_id} must be {(batch_size, width)}, got {value.shape}"
                )
        sampling_parameters = {
            likelihood_id: value.float() for likelihood_id, value in parameters.items()
        }
        specs = self.grouped_specs.get((int(block_index), int(field_index)), ())
        required_likelihoods = tuple(
            dict.fromkeys(spec.likelihood_id for spec in specs)
        )
        if required_likelihoods:
            _require_finite_sampling_parameters(
                torch.cat(
                    tuple(sampling_parameters[likelihood] for likelihood in required_likelihoods),
                    dim=-1,
                ),
                position,
            )
        values = {
            likelihood_id: torch.zeros(
                (batch_size, width),
                dtype=torch.float64,
                device=reference.device,
            )
            for likelihood_id, width in V2_PRIMITIVE_FEEDBACK_DIMS.items()
        }
        masks = {
            likelihood_id: torch.zeros(
                (batch_size, width), dtype=torch.bool, device=reference.device
            )
            for likelihood_id, width in V2_PRIMITIVE_FEEDBACK_DIMS.items()
        }

        for spec in specs:
            likelihood = spec.likelihood_id
            raw = sampling_parameters[likelihood]
            sampled, component_mask = self._sample_spec(
                spec,
                raw,
                values,
                block_index=int(block_index),
            )
            width = V2_PRIMITIVE_FEEDBACK_DIMS[likelihood]
            if sampled.shape != (batch_size, width):
                raise ValueError(f"sample shape mismatch for {spec.primitive_id}: {sampled.shape}")
            if component_mask.shape != (batch_size, width):
                raise ValueError(
                    f"sample mask mismatch for {spec.primitive_id}: {component_mask.shape}"
                )
            values[likelihood] = sampled.double()
            masks[likelihood] = component_mask.bool()

        for likelihood_id in V2_PRIMITIVE_FEEDBACK_DIMS:
            self._history[(int(block_index), int(field_index), likelihood_id)] = values[
                likelihood_id
            ]
        self._next_position += 1
        return values, masks

    def _sample_spec(
        self,
        spec: EnabledPrimitiveSpec,
        raw: Tensor,
        values: Mapping[str, Tensor],
        *,
        block_index: int,
    ) -> tuple[Tensor, Tensor]:
        likelihood = spec.likelihood_id
        batch_size = raw.shape[0]

        def scalar(name: str) -> Tensor:
            return values[name][..., 0]

        def scalar_mask(active: Tensor | bool) -> Tensor:
            if isinstance(active, bool):
                active = torch.full((batch_size,), active, dtype=torch.bool, device=raw.device)
            return active.bool().unsqueeze(-1)

        if likelihood == "categorical_hours_0_4":
            sampled = sample_categorical(raw).unsqueeze(-1)
            return sampled, scalar_mask(True)

        if likelihood == "dense_joint_value_state":
            observed = scalar("categorical_hours_0_4")
            lower, upper = _valid_range(self.metadata, spec)
            sampled = _sample_dense_value_state(raw, observed, lower=lower, upper=upper)
            return sampled, scalar_mask(observed.gt(0)).expand(-1, 4)

        if likelihood == "dense_abnormal_duration_vector":
            observed = scalar("categorical_hours_0_4")
            condition_keys = tuple(
                self.registry["condition_sets"]["dense_abnormal"][spec.field]
            )
            condition_count = len(condition_keys)
            dense_state = values["dense_joint_value_state"]
            chain = sample_dense_abnormal_duration(
                raw,
                observed,
                field=spec.field,
                condition_keys=condition_keys,
                minimum=dense_state[..., 1],
                maximum=dense_state[..., 2],
            )
            sampled = raw.new_zeros((batch_size, 2))
            sampled[..., :condition_count] = chain
            component_mask = torch.zeros_like(sampled, dtype=torch.bool)
            component_mask[..., :condition_count] = observed.gt(0).unsqueeze(-1)
            return sampled, component_mask

        if likelihood == "gcs_ordinal_triple":
            observed = scalar("categorical_hours_0_4")
            field_parameters = self.registry["field_parameters"][spec.field]
            classes = int(field_parameters["legal_triple_count"])
            sampled = _sample_ordinal_triple(
                raw[..., :classes],
                maximum=int(field_parameters["ordinal_max"]),
                observation_count=observed,
                source_semantics="raw_point",
            )
            active = observed.gt(0)
            sampled = torch.where(active.unsqueeze(-1), sampled, torch.zeros_like(sampled))
            return sampled, scalar_mask(active).expand(-1, 3)

        if likelihood == "gcs_verbal_ungradable_hours_given_observed":
            observed = scalar("categorical_hours_0_4").long()
            valid = torch.arange(5, device=raw.device).le(observed.unsqueeze(-1))
            sampled = sample_categorical(raw, valid).unsqueeze(-1)
            return sampled, scalar_mask(True)

        if likelihood == "gcs_verbal_latest_status":
            observed = scalar("categorical_hours_0_4").long()
            ungradable = scalar("gcs_verbal_ungradable_hours_given_observed").long()
            active = observed.gt(0)
            gradable = observed - ungradable
            valid = torch.stack((gradable.gt(0), ungradable.gt(0)), dim=-1)
            safe_valid = torch.where(active.unsqueeze(-1), valid, torch.ones_like(valid))
            sampled = sample_categorical(raw, safe_valid) + 1
            sampled = torch.where(active, sampled, torch.zeros_like(sampled))
            return sampled.unsqueeze(-1), scalar_mask(active)

        if likelihood == "gcs_verbal_gradable_ordinal_triple":
            observed = scalar("categorical_hours_0_4").long()
            ungradable = scalar("gcs_verbal_ungradable_hours_given_observed").long()
            gradable = observed - ungradable
            active = gradable.gt(0)
            sampled = _sample_ordinal_triple(
                raw,
                maximum=5,
                observation_count=gradable,
                source_semantics="hourly_sequence",
            )
            sampled = torch.where(active.unsqueeze(-1), sampled, torch.zeros_like(sampled))
            return sampled, scalar_mask(active).expand(-1, 3)

        if likelihood == "hurdle_negative_binomial_count":
            sampled = sample_hurdle_negative_binomial(
                raw[..., 0], raw[..., 1], raw[..., 2]
            ).unsqueeze(-1)
            return sampled, scalar_mask(True)

        if likelihood == "lab_joint_value_state":
            count = scalar("hurdle_negative_binomial_count")
            center, scale = _lab_affine(self.metadata, spec)
            sampled = _sample_lab_value_state(raw, count, center=center, scale=scale)
            return sampled, scalar_mask(count.gt(0)).expand(-1, 3)

        if likelihood == "respiratory_block_evidence":
            sampled = torch.distributions.Bernoulli(
                logits=raw[..., 0],
                validate_args=False,
            ).sample()
            return sampled.unsqueeze(-1), scalar_mask(True)

        if likelihood == "respiratory_edge_evidence_given_block":
            block = scalar("respiratory_block_evidence").bool()
            stochastic = torch.distributions.Bernoulli(
                logits=raw[..., 0],
                validate_args=False,
            ).sample()
            sampled = torch.where(block, stochastic, torch.zeros_like(stochastic))
            return sampled.unsqueeze(-1), scalar_mask(True)

        if likelihood == "respiratory_occupancy_vector":
            block = scalar("respiratory_block_evidence").bool()
            sampled = _sample_respiratory_occupancy(raw, block)
            return sampled, scalar_mask(block).expand(-1, 5)

        if likelihood == "respiratory_edge_state":
            edge = scalar("respiratory_edge_evidence_given_block").bool()
            sampled = sample_categorical(raw) + 1
            sampled = torch.where(edge, sampled, torch.zeros_like(sampled))
            return sampled.unsqueeze(-1), scalar_mask(edge)

        if likelihood == "respiratory_onset_vector":
            block = scalar("respiratory_block_evidence").bool()
            edge = scalar("respiratory_edge_evidence_given_block").bool()
            edge_state = scalar("respiratory_edge_state").long() - 1
            occupancy = values["respiratory_occupancy_vector"][..., :4]
            one_hot_edge = F.one_hot(edge_state.clamp(0, 3), num_classes=4).bool()
            required = edge.unsqueeze(-1) & one_hot_edge & occupancy.eq(0.0)
            require_any = block & (~occupancy.gt(0.0).any(dim=-1))
            sampled = sample_autoregressive_hurdle_count_vector(
                raw,
                component_count=4,
                required_positive=required,
                require_any_positive=require_any,
            )
            sampled = torch.where(block.unsqueeze(-1), sampled, torch.zeros_like(sampled))
            return sampled, scalar_mask(block).expand(-1, 4)

        if likelihood == "vasopressor_duration_vector":
            sampled = sample_autoregressive_zoi_logit_normal_vector(
                raw,
                component_count=6,
            )
            return sampled, scalar_mask(True).expand(-1, 6)

        if likelihood == "vasopressor_edge_state_vector":
            sampled = sample_autoregressive_binary_vector(raw, component_count=6)
            return sampled, scalar_mask(True).expand(-1, 6)

        if likelihood == "vasopressor_onset_vector":
            duration = values["vasopressor_duration_vector"]
            edge = values["vasopressor_edge_state_vector"].bool()
            sampled = sample_autoregressive_hurdle_count_vector(
                raw,
                component_count=6,
                required_positive=edge & duration.eq(0.0),
            )
            return sampled, scalar_mask(True).expand(-1, 6)

        if likelihood == "ned_joint_value_state":
            vasopressor_index = next(
                int(row["position"])
                for row in self.registry["registered_core_field_order"]
                if int(row["field_id"]) == 27
            )
            vaso_duration = self._history.get(
                (block_index, vasopressor_index, "vasopressor_duration_vector")
            )
            vaso_edge = self._history.get(
                (block_index, vasopressor_index, "vasopressor_edge_state_vector")
            )
            if vaso_duration is None or vaso_edge is None:
                raise ValueError("NED sampling requires preceding vasopressor duration and edge")
            compatible_duration = vaso_duration[..., :5].gt(0).any(dim=-1)
            compatible_edge = vaso_edge[..., :5].gt(0).any(dim=-1)
            sampled = _sample_ned_state(raw, compatible_duration, compatible_edge)
            return sampled, scalar_mask(True).expand(-1, 3)

        if likelihood == "uop_sum_given_count":
            count = scalar("hurdle_negative_binomial_count")
            sampled = _sample_uop_sum(raw, count)
            return sampled.unsqueeze(-1), scalar_mask(count.gt(0))

        raise AssertionError(f"unhandled rollout likelihood: {likelihood}")


def build_enabled_core_factors(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    expected_lab_scale_artifact_hash: str,
) -> tuple[PrimitiveLogProb, ...]:
    """Build all 414 registry factors from model banks and packed structured truth.

    Required model output: ``outputs['primitive_parameters'][likelihood_id]`` with
    shape ``[B,6,29,V2_PRIMITIVE_HEAD_DIMS[likelihood_id]]``.

    Required batch keys are ``target_primitives``, ``target_primitive_masks``, and
    ``target_primitive_metadata``. Target and mask leading shapes are ``[B,6,29]``.
    The field axis must be registry order ``1..28,35``.
    """

    primitive_parameters = outputs.get("primitive_parameters")
    targets = batch.get("target_primitives")
    masks = batch.get("target_primitive_masks")
    metadata = batch.get("target_primitive_metadata")
    if not isinstance(primitive_parameters, Mapping):
        raise ValueError("outputs must contain a primitive_parameters mapping")
    if not isinstance(targets, Mapping) or not isinstance(masks, Mapping):
        raise ValueError("batch must contain target_primitives and target_primitive_masks")
    if not isinstance(metadata, Mapping):
        raise ValueError("batch must contain target_primitive_metadata")
    _validate_batch_metadata(metadata, registry)
    _validate_lab_scale_metadata(metadata, registry, expected_lab_scale_artifact_hash)
    specs = expand_enabled_core_primitives(registry)
    required_likelihoods = {spec.likelihood_id for spec in specs}
    missing_heads = required_likelihoods.difference(primitive_parameters)
    missing_targets = required_likelihoods.difference(targets)
    missing_masks = required_likelihoods.difference(masks)
    if missing_heads or missing_targets or missing_masks:
        raise ValueError(
            "missing enabled-core banks: "
            f"heads={sorted(missing_heads)}, targets={sorted(missing_targets)}, "
            f"masks={sorted(missing_masks)}"
        )
    wrong_physical_dtype = {
        likelihood_id: getattr(targets[likelihood_id], "dtype", None)
        for likelihood_id in V2_FLOAT64_PHYSICAL_TARGETS
        if not isinstance(targets[likelihood_id], Tensor)
        or targets[likelihood_id].dtype != torch.float64
    }
    if wrong_physical_dtype:
        raise ValueError(
            "physical target banks must remain serialized float64 through likelihood "
            f"canonicalization: {wrong_physical_dtype}"
        )
    ordered_likelihoods = tuple(
        likelihood_id
        for likelihood_id in V2_PRIMITIVE_HEAD_DIMS
        if likelihood_id in required_likelihoods
    )
    finite_parameter_banks: list[Tensor] = []
    for likelihood_id in ordered_likelihoods:
        bank = primitive_parameters[likelihood_id]
        width = V2_PRIMITIVE_HEAD_DIMS[likelihood_id]
        if not isinstance(bank, Tensor) or bank.ndim != 4:
            raise ValueError(
                f"primitive head {likelihood_id} must be [batch,6,29,width], "
                f"got {getattr(bank, 'shape', None)}"
            )
        if bank.shape[1:] != (6, 29, width):
            raise ValueError(
                f"primitive head {likelihood_id} must end in (6,29,{width}), got {bank.shape}"
            )
        finite_parameter_banks.append(torch.isfinite(bank).all())
    _require_tensor_condition(
        torch.stack(finite_parameter_banks),
        "non-finite raw parameters in enabled-core emission banks",
        FloatingPointError,
    )
    factors: list[PrimitiveLogProb] = []
    for spec in specs:
        raw_parameters = _parameter_slice(primitive_parameters, spec)
        mask = _mask_slice(masks, spec.likelihood_id, spec)
        log_prob = _evaluate_log_prob(
            spec,
            raw_parameters,
            targets,
            mask,
            metadata,
            registry,
        )
        if log_prob.shape != mask.shape:
            raise ValueError(f"log probability shape mismatch for {spec.primitive_id}")
        log_prob = torch.where(mask, log_prob, torch.zeros_like(log_prob))
        factors.append(PrimitiveLogProb(spec.primitive_id, spec.likelihood_id, log_prob))
    if len(factors) != len({factor.primitive_id for factor in factors}):
        raise ValueError("an enabled primitive produced more than one log probability")
    if len(factors) != len(specs):
        raise AssertionError("not every enabled primitive produced one factor")
    return tuple(factors)


def compute_multires_event_v2_loss(
    factors: Sequence[PrimitiveLogProb],
    *,
    expected_primitive_ids: Sequence[str] | None = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> dict[str, Any]:
    """Exact enabled-core negative canonical-coordinate log score.

    Every normalized primitive factor is summed once per sample. ``mean``
    averages only over sampled anchors in the minibatch; it never divides by
    active targets or registered factors.  Continuous mixed branches are scored
    in their frozen generative coordinates, not mislabeled as one raw-tuple NLL.
    """

    if not factors:
        raise ValueError("at least one stochastic primitive factor is required")
    identifiers = tuple(factor.primitive_id for factor in factors)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("every stochastic primitive must appear exactly once")
    if expected_primitive_ids is not None and set(identifiers) != set(expected_primitive_ids):
        missing = set(expected_primitive_ids).difference(identifiers)
        extra = set(identifiers).difference(expected_primitive_ids)
        raise ValueError(
            f"primitive registry mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    reference_shape = factors[0].log_prob.shape
    if len(reference_shape) != 1:
        raise ValueError("primitive log probabilities must be one scalar per batch sample")
    for factor in factors:
        if factor.source_kind != "stochastic_primitive":
            raise ValueError("deterministic projections cannot contribute a loss term")
        if factor.objective_status != "enabled_core":
            raise ValueError("only enabled_core primitives enter the initial joint objective")
        if factor.likelihood_id not in CORE_EMISSION_LOG_PROBS:
            raise ValueError(f"unknown enabled-core likelihood {factor.likelihood_id!r}")
        if factor.log_prob.shape != reference_shape:
            raise ValueError("all primitive log probabilities must share the batch shape")
        if factor.log_prob.device.type != "cuda" and not bool(
            torch.isfinite(factor.log_prob).all().item()
        ):
            raise FloatingPointError(f"non-finite log probability for {factor.primitive_id}")
    primitive_log_prob = torch.stack([factor.log_prob for factor in factors], dim=-1)
    if primitive_log_prob.device.type == "cuda":
        _require_tensor_condition(
            torch.isfinite(primitive_log_prob),
            "non-finite enabled-core primitive log probability",
            FloatingPointError,
        )
    joint_log_prob = primitive_log_prob.sum(dim=-1)
    _require_tensor_condition(
        torch.isfinite(joint_log_prob),
        "non-finite enabled-core joint log probability",
        FloatingPointError,
    )
    per_sample_nll = -joint_log_prob
    if reduction == "mean":
        loss = per_sample_nll.mean()
    elif reduction == "sum":
        loss = per_sample_nll.sum()
    elif reduction == "none":
        loss = per_sample_nll
    else:
        raise ValueError(f"unsupported reduction: {reduction!r}")
    return {
        "loss": loss,
        "joint_log_prob": joint_log_prob,
        "per_sample_nll": per_sample_nll,
        "primitive_log_prob": primitive_log_prob,
        "primitive_ids": identifiers,
        "primitive_count": len(factors),
    }


def compute_registry_multires_event_v2_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    expected_lab_scale_artifact_hash: str,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> dict[str, Any]:
    specs = expand_enabled_core_primitives(registry)
    factors = build_enabled_core_factors(
        outputs,
        batch,
        registry,
        expected_lab_scale_artifact_hash=expected_lab_scale_artifact_hash,
    )
    return compute_multires_event_v2_loss(
        factors,
        expected_primitive_ids=[spec.primitive_id for spec in specs],
        reduction=reduction,
    )


__all__ = [
    "REGISTERED_CORE_FIELD_IDS",
    "EXPECTED_ENABLED_CORE_PRIMITIVES",
    "EnabledPrimitiveSpec",
    "PrimitiveLogProb",
    "RegistryPrimitiveSampler",
    "V2_PRIMITIVE_FEEDBACK_DIMS",
    "V2_PRIMITIVE_HEAD_DIMS",
    "V2_EMISSION_REGISTRY_SHA256",
    "V2_EMISSION_REGISTRY_VERSION",
    "V2_PROCESS_REGISTRY_SHA256",
    "V2_PROCESS_REGISTRY_VERSION",
    "build_enabled_core_factors",
    "compute_multires_event_v2_loss",
    "compute_registry_multires_event_v2_loss",
    "expand_enabled_core_primitives",
    "validate_emission_registry_head_contract",
]
