from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
import re
import json
import math
import random
from typing import Any, Iterable, Mapping

import torch

from trauma_predict.training.multires_event_v2_loss import (
    V2_PRIMITIVE_FEEDBACK_DIMS,
    compute_registry_multires_event_v2_loss,
    expand_enabled_core_primitives,
)
from trauma_predict.training.observability import (
    append_jsonl,
    is_rank_zero,
    sha256_file,
    utc_now,
)


MODEL_INPUT_KEYS = (
    "event_field_ids",
    "event_operator_ids",
    "event_condition_ids",
    "event_values",
    "event_value_mask",
    "event_study_slot_ids",
    "block_index",
    "event_mask",
    "block_role_ids",
    "resolution_ids",
    "relative_start",
    "relative_end",
    "span",
    "block_mask",
    "static_numeric",
    "static_numeric_mask",
    "static_categorical",
)


def teacher_forced_model_inputs(batch: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    """Create the model view without mutating the exact raw-unit loss targets."""

    input_batch = batch.get("input_batch")
    primitives = batch.get("target_primitives")
    masks = batch.get("target_primitive_masks")
    if not isinstance(input_batch, Mapping):
        raise ValueError("V2 batch lacks input_batch")
    if not isinstance(primitives, Mapping) or not isinstance(masks, Mapping):
        raise ValueError("V2 batch lacks target primitive banks")
    missing = set(MODEL_INPUT_KEYS).difference(input_batch)
    if missing:
        raise ValueError(f"V2 input_batch lacks model tensors: {sorted(missing)}")

    # The collator stores scalar likelihood truth as [B,6,29], while the feedback
    # encoder represents every likelihood with an explicit component axis.  This
    # adapter is model-only: the loss receives the untouched collator banks below.
    feedback: dict[str, torch.Tensor] = {}
    for likelihood_id, width in V2_PRIMITIVE_FEEDBACK_DIMS.items():
        value = primitives.get(likelihood_id)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"missing feedback primitive {likelihood_id!r}")
        if int(width) == 1 and value.ndim == 3:
            value = value.unsqueeze(-1)
        feedback[likelihood_id] = value
    return {
        **{key: input_batch[key] for key in MODEL_INPUT_KEYS},
        "target_primitives": feedback,
        "target_primitive_masks": masks,
        "relation_adjacency": batch.get("relation_adjacency"),
        "relation_type_lags": batch.get("relation_type_lags"),
        "mode": mode,
    }


def exact_teacher_forced_loss(
    model: Any,
    batch: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    mode: str,
    expected_lab_scale_artifact_hash: str | None = None,
    autocast: Any = nullcontext,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Forward under teacher forcing and evaluate the exact float32 joint NLL."""

    if not re.fullmatch(r"[0-9a-f]{64}", str(expected_lab_scale_artifact_hash or "")):
        raise ValueError(
            "teacher-forced V2 loss requires the configured train-only lab scale artifact hash"
        )
    with autocast():
        outputs = model(**teacher_forced_model_inputs(batch, mode=mode))
    parameters = outputs.get("primitive_parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("V2 model output lacks primitive parameter banks")
    # The probabilistic emissions are audited in float32 even when the neural
    # network runs under fp16 autocast.  Tensor.float() preserves autograd.
    loss_outputs = dict(outputs)
    loss_outputs["primitive_parameters"] = {
        key: value.float() for key, value in parameters.items()
    }
    loss_result = compute_registry_multires_event_v2_loss(
        loss_outputs,
        batch,
        registry,
        expected_lab_scale_artifact_hash=str(expected_lab_scale_artifact_hash),
        reduction="mean",
    )
    if int(loss_result["primitive_count"]) != 414:
        raise AssertionError("V2 objective did not expand to exactly 414 primitive factors")
    return outputs, loss_result


def evaluate_teacher_forced(
    *,
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    registry: Mapping[str, Any],
    device: torch.device,
    mode: str,
    expected_samples: int,
    phase: str,
    step: int,
    precision: str = "fp16",
    metrics_path: Path | None = None,
    expected_lab_scale_artifact_hash: str | None = None,
    per_anchor_output_path: Path | None = None,
    evaluation_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate anchor-mean and within-subject-then-subject-macro joint NLL.

    The only evaluation denominator is the fixed persisted anchor set (or the
    number of anchors belonging to a subject).  It is never an active-target or
    observed-variable count.
    """

    if phase not in {"interval", "final"}:
        raise ValueError("V2 evaluation phase must be interval or final")
    if expected_samples < 1:
        raise ValueError("expected_samples must be positive")
    model.eval()
    local_rows: list[dict[str, Any]] = []
    autocast = _autocast_factory(device, precision)
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_to_device(raw_batch, device)
            _, result = exact_teacher_forced_loss(
                model,
                batch,
                registry,
                mode=mode,
                expected_lab_scale_artifact_hash=expected_lab_scale_artifact_hash,
                autocast=autocast,
            )
            per_sample = result["per_sample_nll"].detach().float().cpu().tolist()
            sample_ids = _string_batch(batch.get("sample_id"))
            subject_ids = _string_batch(batch.get("subject_id"))
            if not (len(per_sample) == len(sample_ids) == len(subject_ids)):
                raise ValueError("V2 evaluation identities do not align with per-sample NLL")
            decompositions = _teacher_nll_decomposition_rows(
                result,
                registry,
                batch_size=len(per_sample),
            )
            if decompositions is None:
                decompositions = [None] * len(per_sample)
            local_rows.extend(
                {
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                    "joint_nll": float(nll),
                    "decomposition": decomposition,
                }
                for sample_id, subject_id, nll, decomposition in zip(
                    sample_ids,
                    subject_ids,
                    per_sample,
                    decompositions,
                    strict=True,
                )
            )

    rows = [row for rank_rows in _gather_objects(local_rows) for row in rank_rows]
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("V2 validation sampler introduced duplicate persisted anchors")
    if len(rows) != expected_samples:
        raise RuntimeError(
            f"V2 {phase} evaluation expected {expected_samples} anchors, got {len(rows)}"
        )
    by_subject: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject_id"])].append(float(row["joint_nll"]))
    if not by_subject:
        raise RuntimeError("V2 evaluation produced no subjects")
    subject_means = {
        subject_id: sum(values) / len(values)
        for subject_id, values in by_subject.items()
    }
    result = {
        "schema_version": "trauma_predict.multires_event_v2_evaluation.v1",
        "evaluated_at": utc_now(),
        "phase": phase,
        "step": int(step),
        "mode": mode,
        "samples": len(rows),
        "subjects": len(subject_means),
        "joint_nll_anchor_mean": sum(float(row["joint_nll"]) for row in rows)
        / len(rows),
        "joint_nll_subject_macro": sum(subject_means.values()) / len(subject_means),
        "primitive_factors_per_anchor": 414,
        "aggregation": "sum_414_log_prob_terms_per_anchor_then_within_subject_mean_then_subject_macro",
        "active_target_denominator": False,
        "deterministic_projection_loss": False,
    }
    result["joint_nll_anchor_mean_nats_per_block"] = (
        result["joint_nll_anchor_mean"] / 6.0
    )
    result["joint_nll_anchor_mean_bits_per_block"] = (
        result["joint_nll_anchor_mean"] / (6.0 * math.log(2.0))
    )
    result["joint_nll_subject_macro_nats_per_block"] = (
        result["joint_nll_subject_macro"] / 6.0
    )
    result["joint_nll_subject_macro_bits_per_block"] = (
        result["joint_nll_subject_macro"] / (6.0 * math.log(2.0))
    )
    available_decomposition = all(row["decomposition"] is not None for row in rows)
    if available_decomposition:
        result["teacher_nll_decomposition"] = _summarize_teacher_decomposition(rows)
    else:
        result["teacher_nll_decomposition"] = {
            "status": "not_available_in_mock_or_legacy_loss_result"
        }
    if evaluation_identity is not None:
        result["identity"] = dict(evaluation_identity)
    def persist_rank_zero_result() -> dict[str, Any] | None:
        if not is_rank_zero():
            return None
        persisted = dict(result)
        if per_anchor_output_path is not None:
            per_anchor_output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = per_anchor_output_path.with_suffix(
                per_anchor_output_path.suffix + ".tmp"
            )
            with temporary.open("w", encoding="utf-8") as handle:
                for row in sorted(
                    rows,
                    key=lambda item: (
                        str(item["subject_id"]),
                        str(item["sample_id"]),
                    ),
                ):
                    handle.write(
                        json.dumps(
                            {
                                "sample_id": row["sample_id"],
                                "subject_id": row["subject_id"],
                                "joint_nll": row["joint_nll"],
                                "primitive_factors": 414,
                                "mode": mode,
                                "step": int(step),
                                "teacher_nll_decomposition": row["decomposition"],
                                **(
                                    {"identity": dict(evaluation_identity)}
                                    if evaluation_identity is not None
                                    else {}
                                ),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            temporary.replace(per_anchor_output_path)
            persisted["per_anchor_output_path"] = str(per_anchor_output_path)
            persisted["per_anchor_output_sha256"] = sha256_file(
                per_anchor_output_path
            )
        if metrics_path is not None:
            append_jsonl(
                metrics_path,
                {"event": f"v2_{phase}_evaluation", **persisted},
            )
        return persisted

    persisted_results = _collect_distributed_phase(
        f"V2 {phase} evaluation persistence",
        persist_rank_zero_result,
    )
    persisted_result = persisted_results[0]
    if not isinstance(persisted_result, Mapping):
        raise RuntimeError(f"V2 {phase} evaluation result persistence failed")
    return dict(persisted_result)


def _teacher_nll_decomposition_rows(
    loss_result: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    batch_size: int,
) -> list[dict[str, Any]] | None:
    primitive_log_prob = loss_result.get("primitive_log_prob")
    primitive_ids = loss_result.get("primitive_ids")
    if not isinstance(primitive_log_prob, torch.Tensor) or not isinstance(
        primitive_ids, (tuple, list)
    ):
        if registry.get("version") is not None:
            raise ValueError(
                "audited teacher loss must expose primitive_log_prob and primitive_ids"
            )
        return None
    specs = expand_enabled_core_primitives(registry)
    expected_ids = tuple(spec.primitive_id for spec in specs)
    if tuple(str(item) for item in primitive_ids) != expected_ids:
        raise ValueError("teacher decomposition primitive order differs from the registry")
    if primitive_log_prob.shape != (batch_size, len(specs)):
        raise ValueError(
            "teacher primitive_log_prob must be [batch,registered_primitive_count]"
        )
    nll = -primitive_log_prob.detach().double().cpu()
    if not bool(torch.isfinite(nll).all().item()):
        raise FloatingPointError("teacher decomposition contains non-finite factor NLL")
    total = loss_result.get("per_sample_nll")
    if not isinstance(total, torch.Tensor) or total.shape != (batch_size,):
        raise ValueError("teacher decomposition requires aligned per_sample_nll")
    reconstructed = -primitive_log_prob.detach().sum(dim=-1)
    if not torch.allclose(
        reconstructed,
        total.detach().to(device=reconstructed.device, dtype=reconstructed.dtype),
        rtol=1e-6,
        atol=1e-5,
    ):
        raise AssertionError("teacher factor decomposition does not sum to joint NLL")

    rows: list[dict[str, Any]] = []
    family_by_field: dict[str, str] = {}
    field_sets = registry.get("field_sets")
    if not isinstance(field_sets, Mapping):
        raise ValueError("teacher decomposition registry lacks field_sets")
    family_names = (
        "dense_continuous",
        "gcs_ordinal_enabled",
        "gcs_verbal_reaggregated",
        "intermittent_labs",
        "respiratory_support",
        "vasopressor_support",
        "ned",
        "uop",
    )
    for family in family_names:
        fields = field_sets.get(family)
        if not isinstance(fields, (tuple, list)):
            raise ValueError(f"teacher decomposition registry lacks field set {family}")
        for field in fields:
            if str(field) in family_by_field:
                raise ValueError("teacher decomposition field belongs to two process families")
            family_by_field[str(field)] = family
    observation_likelihoods = {
        "categorical_hours_0_4",
        "gcs_verbal_ungradable_hours_given_observed",
        "gcs_verbal_latest_status",
        "hurdle_negative_binomial_count",
        "respiratory_block_evidence",
        "respiratory_edge_evidence_given_block",
    }
    for sample_index in range(batch_size):
        by_block: dict[str, float] = defaultdict(float)
        by_field: dict[str, float] = defaultdict(float)
        by_likelihood: dict[str, float] = defaultdict(float)
        by_process_family: dict[str, float] = defaultdict(float)
        by_objective_branch: dict[str, float] = defaultdict(float)
        for factor_index, spec in enumerate(specs):
            value = float(nll[sample_index, factor_index].item())
            by_block[spec.block] += value
            by_field[spec.field] += value
            by_likelihood[spec.likelihood_id] += value
            family = family_by_field.get(spec.field)
            if family is None:
                raise ValueError(f"teacher decomposition has unclassified field {spec.field}")
            by_process_family[family] += value
            branch = (
                "observation_branch"
                if spec.likelihood_id in observation_likelihoods
                else "conditional_value_branch"
            )
            by_objective_branch[branch] += value
        rows.append(
            {
                "units": "canonical_factor_nats_per_fixed_six_block_anchor",
                "by_block": dict(sorted(by_block.items())),
                "by_field": dict(sorted(by_field.items())),
                "by_likelihood": dict(sorted(by_likelihood.items())),
                "by_process_family": dict(sorted(by_process_family.items())),
                "by_objective_branch": dict(sorted(by_objective_branch.items())),
            }
        )
    return rows


def _summarize_teacher_decomposition(
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "complete",
        "units": "canonical_factor_nats_per_fixed_six_block_anchor",
        "aggregation": "anchor_mean_and_within_subject_anchor_mean_then_subject_macro",
    }
    for dimension in (
        "by_block",
        "by_field",
        "by_likelihood",
        "by_process_family",
        "by_objective_branch",
    ):
        first = rows[0]["decomposition"][dimension]
        if not isinstance(first, Mapping):
            raise ValueError("teacher decomposition dimension must be a mapping")
        keys = set(str(key) for key in first)
        summary: dict[str, Any] = {}
        for row in rows:
            values = row["decomposition"][dimension]
            if not isinstance(values, Mapping) or set(values) != keys:
                raise ValueError("teacher decomposition support changed across anchors")
        for key in sorted(keys):
            values = [float(row["decomposition"][dimension][key]) for row in rows]
            by_subject: dict[str, list[float]] = defaultdict(list)
            for row, value in zip(rows, values, strict=True):
                by_subject[str(row["subject_id"])].append(value)
            subject_means = [
                sum(items) / len(items) for items in by_subject.values()
            ]
            summary[key] = {
                "anchor_mean": sum(values) / len(values),
                "subject_macro": sum(subject_means) / len(subject_means),
            }
        result[dimension] = summary
    return result


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def paired_subject_bootstrap_joint_nll(
    control_path: str | Path,
    candidate_path: str | Path,
    *,
    repetitions: int = 2_000,
    seed: int = 20260713,
    expected_anchors: int = 6_309,
) -> dict[str, Any]:
    """Describe a matched-mode delta without making a promotion decision."""

    if repetitions != 2_000 or seed != 20260713:
        raise ValueError("paired V2 bootstrap is frozen to 2,000 draws with seed 20260713")
    if expected_anchors < 1:
        raise ValueError("paired V2 comparison expected_anchors must be positive")
    control = _read_per_anchor_nll(Path(control_path))
    candidate = _read_per_anchor_nll(Path(candidate_path))
    if set(control) != set(candidate):
        missing = set(control).difference(candidate)
        extra = set(candidate).difference(control)
        raise ValueError(
            "paired V2 comparison requires identical persisted validation anchors: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    if len(control) != expected_anchors:
        raise ValueError(
            f"paired V2 comparison requires {expected_anchors} persisted anchors, "
            f"got {len(control)}"
        )
    by_subject: dict[str, list[float]] = defaultdict(list)
    for sample_id, (subject_id, control_nll) in control.items():
        candidate_subject, candidate_nll = candidate[sample_id]
        if candidate_subject != subject_id:
            raise ValueError(f"subject identity changed for paired anchor {sample_id}")
        by_subject[subject_id].append(candidate_nll - control_nll)
    subject_delta = [sum(values) / len(values) for values in by_subject.values()]
    if not subject_delta:
        raise ValueError("paired V2 comparison contains no subjects")
    observed = sum(subject_delta) / len(subject_delta)
    rng = random.Random(seed)
    count = len(subject_delta)
    bootstrap = sorted(
        sum(subject_delta[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(repetitions)
    )
    lower = _linear_percentile(bootstrap, 0.025)
    upper = _linear_percentile(bootstrap, 0.975)
    return {
        "schema_version": "trauma_predict.multires_event_v2_paired_bootstrap.v2",
        "created_at": utc_now(),
        "anchors": len(control),
        "subjects": len(subject_delta),
        "estimand": "candidate_minus_control_subject_macro_joint_nll",
        "observed_delta": observed,
        "ci95": {"lower": lower, "upper": upper},
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
        "ci95_upper_lt_zero": upper < 0.0,
        "decision_authority": "none_pending_final_mathematical_audit",
        "promotion_contract_valid": False,
        "promotion_decision": None,
    }


def _read_per_anchor_nll(path: Path) -> dict[str, tuple[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, tuple[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or "")
            subject_id = str(row.get("subject_id") or "")
            nll = float(row.get("joint_nll"))
            if not sample_id or not subject_id or not math.isfinite(nll):
                raise ValueError(f"invalid per-anchor V2 NLL row at {path}:{line_number}")
            if sample_id in result:
                raise ValueError(f"duplicate per-anchor V2 sample_id={sample_id}")
            result[sample_id] = (subject_id, nll)
    if not result:
        raise ValueError(f"per-anchor V2 NLL file is empty: {path}")
    return result


def _linear_percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _autocast_factory(device: torch.device, precision: str) -> Any:
    if device.type != "cuda" or precision != "fp16":
        return nullcontext

    def factory() -> Any:
        try:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        except AttributeError:  # pragma: no cover - older supported torch
            return torch.cuda.amp.autocast(dtype=torch.float16)

    return factory


def _gather_objects(value: Any) -> list[Any]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return [value]
    gathered: list[Any] = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(gathered, value)
    return gathered


def _collect_distributed_phase(stage: str, factory: Any) -> list[Any]:
    """Propagate local evaluation/persistence failures to every active rank."""

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return [factory()]
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    try:
        value = factory()
        envelope = {
            "rank": rank,
            "ok": True,
            "value": value,
            "error_type": None,
            "error_message": None,
        }
    except Exception as error:
        envelope = {
            "rank": rank,
            "ok": False,
            "value": None,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    envelopes: list[Any] = [None] * world_size
    torch.distributed.all_gather_object(envelopes, envelope)
    failures = [row for row in envelopes if not bool(row.get("ok"))]
    if failures:
        details = "; ".join(
            "rank {rank} {error_type}: {error_message}".format(**row)
            for row in failures
        )
        raise RuntimeError(f"distributed {stage} failed: {details}")
    ordered = sorted(envelopes, key=lambda row: int(row["rank"]))
    if [int(row["rank"]) for row in ordered] != list(range(world_size)):
        raise RuntimeError(f"distributed {stage} returned an invalid rank set")
    return [row["value"] for row in ordered]


def _string_batch(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (tuple, list)):
        return [str(item) for item in value]
    return [str(value)]


__all__ = [
    "MODEL_INPUT_KEYS",
    "evaluate_teacher_forced",
    "exact_teacher_forced_loss",
    "move_to_device",
    "paired_subject_bootstrap_joint_nll",
    "teacher_forced_model_inputs",
]
