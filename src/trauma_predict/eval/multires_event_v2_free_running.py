from __future__ import annotations

from contextlib import contextmanager, nullcontext
from collections import defaultdict
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from trauma_predict.data.multires_event_v2 import MultiresEventV2Contract
from trauma_predict.eval.multires_event_v2 import MODEL_INPUT_KEYS, move_to_device
from trauma_predict.eval.multires_event_v2_projections import (
    build_physical_projection_schema,
    build_standardized_primitive_schema,
    generated_coherence_report,
    load_standardized_primitive_scale_artifact,
    project_physical_primitives,
    score_physical_ensemble,
    score_standardized_primitive_ensemble,
    standardize_primitive_trajectory,
)
from trauma_predict.training.multires_event_v2_loss import (
    RegistryPrimitiveSampler,
    V2_PRIMITIVE_FEEDBACK_DIMS,
    expand_enabled_core_primitives,
)
from trauma_predict.training.observability import (
    append_jsonl,
    append_rank_local_jsonl,
    sha256_file,
    sha256_payload,
    utc_now,
)


FREE_RUNNING_SCHEMA = "trauma_predict.multires_event_v2_free_running_evaluation.v1"
CRN_SCHEDULE_SCHEMA = "multires_event_v2_common_random_numbers_sha256_v1"
PRODUCTION_TRAJECTORIES_PER_ANCHOR = 100
PRODUCTION_CRN_SEED = 20260713
RETAINED_AUDIT_TRAJECTORIES_PER_ANCHOR = 1
RANK_ARTIFACT_PREFLIGHT_SCHEMA = (
    "trauma_predict.multires_event_v2_rank_artifact_preflight.v1"
)


def evaluate_free_running_v2(
    *,
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    contract: MultiresEventV2Contract,
    device: torch.device,
    mode: str,
    expected_samples: int,
    step: int,
    output_dir: str | Path,
    expected_lab_scale_artifact_hash: str,
    standardized_primitive_scale_path: str | Path,
    expected_standardized_primitive_scale_hash: str,
    input_normalization_sha256: str,
    promotion_metric_contract: Mapping[str, Any],
    evaluation_identity: Mapping[str, Any] | None = None,
    trajectories_per_anchor: int = PRODUCTION_TRAJECTORIES_PER_ANCHOR,
    trajectory_batch_size: int | None = None,
    crn_seed: int = PRODUCTION_CRN_SEED,
    metrics_path: Path | None = None,
    precision: str = "fp16",
) -> dict[str, Any]:
    """Evaluate generated six-block trajectories without future truth feedback.

    The model input is encoded once per anchor. The cached memory/query state is
    expanded to the ensemble dimension and decoded autoregressively with the
    registry sampler. Production configuration freezes 100 trajectories; the
    function accepts smaller positive counts for unit/integration tests.
    """

    if mode not in {"block", "trajectory", "relational"}:
        raise ValueError("free-running V2 mode must be block/trajectory/relational")
    if expected_samples < 1 or trajectories_per_anchor < 1:
        raise ValueError("free-running sample and trajectory counts must be positive")
    batch_size = trajectories_per_anchor if trajectory_batch_size is None else int(
        trajectory_batch_size
    )
    if batch_size < 1 or batch_size > trajectories_per_anchor:
        raise ValueError("trajectory_batch_size must lie in 1..trajectories_per_anchor")
    if not isinstance(crn_seed, int) or crn_seed < 0:
        raise ValueError("free-running CRN seed must be a nonnegative integer")
    if precision not in {"fp16", "fp32"}:
        raise ValueError("free-running neural precision must be fp16 or fp32")
    if len(input_normalization_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in input_normalization_sha256
    ):
        raise ValueError("free-running input normalization identity must be SHA-256")

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rank, world_size = _rank_world()
    rank_started_at = time.monotonic()
    rank_progress_path = output_root / f"progress.rank{rank:05d}.jsonl"
    core_model = model.module if hasattr(model, "module") else model
    if not callable(getattr(core_model, "encode_for_rollout", None)) or not callable(
        getattr(core_model, "rollout_from_encoded", None)
    ):
        raise RuntimeError(
            "free-running evaluation requires encode_for_rollout and "
            "rollout_from_encoded so each anchor is encoded exactly once"
        )
    core_model.eval()

    physical_schema = build_physical_projection_schema(contract)
    primitive_schema = build_standardized_primitive_schema(contract)
    scale_artifact = load_standardized_primitive_scale_artifact(
        standardized_primitive_scale_path,
        expected_content_sha256=expected_standardized_primitive_scale_hash,
        contract=contract,
        expected_lab_scale_artifact_hash=expected_lab_scale_artifact_hash,
    )
    primitive_specs = expand_enabled_core_primitives(contract.process_registry)
    primitive_order: list[dict[str, Any]] = []
    component_cursor = 0
    for spec in primitive_specs:
        raw_width = V2_PRIMITIVE_FEEDBACK_DIMS.get(spec.likelihood_id)
        if raw_width is None:
            if contract.process_registry.get("version") is not None:
                raise ValueError(
                    f"registered primitive lacks a feedback width: {spec.likelihood_id}"
                )
            raw_width = 1
        width = int(raw_width)
        primitive_order.append(
            {
                "primitive_id": spec.primitive_id,
                "likelihood_id": spec.likelihood_id,
                "block_index": spec.block_index,
                "field_index": spec.field_index,
                "field": spec.field,
                "component_width": width,
                "flat_component_start": component_cursor,
                "flat_component_end": component_cursor + width,
            }
        )
        component_cursor += width
    crn_contract = {
        "schema": CRN_SCHEDULE_SCHEMA,
        "base_seed": int(crn_seed),
        "seed_input": ["sample_id", "trajectory_start", "trajectory_count"],
        "digest": "sha256_first_63_bits_big_endian",
        "trajectory_batch_size": batch_size,
        "trajectories_per_anchor": trajectories_per_anchor,
        "mode_in_seed": False,
        "registered_sampling_order": "block_major_then_topological_field_then_likelihood",
        "draw_consumption": "fixed_by_registered_factor_and_ensemble_shape",
    }
    crn_contract_hash = sha256_payload(crn_contract)
    identity = {
        "dataset_id": str(contract.manifest["dataset_id"]),
        "contract_bundle_hash": contract.contract_bundle_hash,
        "process_contract_sha256": contract.contract_hashes["process"],
        "emission_contract_sha256": contract.contract_hashes["emission"],
        "projection_contract_sha256": contract.contract_hashes["projection"],
        "relation_contract_sha256": contract.contract_hashes["relation"],
        "sidecar_schema_sha256": contract.contract_hashes["sidecar_schema"],
        "lab_scale_artifact_sha256": expected_lab_scale_artifact_hash,
        "standardized_primitive_scale_sha256": expected_standardized_primitive_scale_hash,
        "input_normalization_sha256": input_normalization_sha256,
        "crn_contract_sha256": crn_contract_hash,
        "neural_precision": precision,
    }
    if evaluation_identity is not None:
        required_model_identity = (
            "source_tree_sha256",
            "source_identity_sha256",
            "git_commit",
            "git_head_tree",
            "matched_design_signature",
            "selected_checkpoint_step",
            "selected_checkpoint_model_sha256",
        )
        missing = [
            key
            for key in required_model_identity
            if evaluation_identity.get(key) in (None, "")
        ]
        if missing:
            raise ValueError(
                f"free-running final model identity is incomplete: {missing}"
            )
        for key, value in evaluation_identity.items():
            if key in identity and str(identity[key]) != str(value):
                raise ValueError(
                    f"free-running evaluation identity differs for {key}: "
                    f"{identity[key]!r} != {value!r}"
                )
            identity[key] = value
    schema_path = output_root / "sample_schema.json"
    schema_payload = {
        "schema_version": "trauma_predict.multires_event_v2_sample_export.v2",
        "created_at": utc_now(),
        "identity": identity,
        "crn_contract": crn_contract,
        "primitive_order": primitive_order,
        "primitive_flat_component_count": component_cursor,
        "physical_projection_order": [row.as_dict() for row in physical_schema],
        "standardized_primitive_coordinates": [
            row.as_dict() for row in primitive_schema
        ],
        "care_and_procedure": {
            "status": "not_applicable",
            "reason": "care_and_procedure_joint_objective_off",
        },
        "trajectory_retention": {
            "scored_trajectories_per_anchor": trajectories_per_anchor,
            "retained_audit_trajectories_per_anchor": (
                RETAINED_AUDIT_TRAJECTORIES_PER_ANCHOR
            ),
            "selection": "trajectory_index_zero_for_every_anchor",
            "rationale": (
                "all_ensemble_members_contribute_to_metrics; one deterministic "
                "member_per_anchor_is_retained_for_trace_audit"
            ),
        },
    }
    _collect_distributed_phase(
        "free-running sample-schema materialization",
        lambda: _atomic_json(schema_path, schema_payload) if rank == 0 else None,
    )

    primitive_path = output_root / f"audit_trajectory_samples.rank{rank:05d}.jsonl.gz"
    anchor_path = output_root / f"per_anchor_scores.rank{rank:05d}.jsonl"
    local_anchor_count = 0
    local_calibration = _empty_calibration_state()
    local_coverage = _empty_coverage_state()
    local_sample_ids: set[str] = set()
    autocast = _autocast_factory(device, precision)
    _emit_rank_progress(
        path=rank_progress_path,
        rank=rank,
        mode=mode,
        completed_anchors=0,
        started_at=rank_started_at,
    )
    with _atomic_gzip_text(primitive_path) as primitive_handle, _atomic_text(
        anchor_path
    ) as anchor_handle:
        with torch.no_grad():
            for raw_batch in loader:
                batch = move_to_device(raw_batch, device)
                sample_ids = _string_batch(batch.get("sample_id"))
                subject_ids = _string_batch(batch.get("subject_id"))
                if not sample_ids or len(sample_ids) != len(subject_ids):
                    raise ValueError("free-running identities do not align within the batch")
                metadata = batch.get("target_primitive_metadata")
                if not isinstance(metadata, Mapping):
                    raise ValueError("free-running batch lacks target primitive metadata")
                with autocast():
                    encoded_batch = _encode_batch_once(
                        core_model, batch, expected_batch_size=len(sample_ids)
                    )
                truth_batch = _primitive_batch_slice_inputs(
                    batch, expected_batch_size=len(sample_ids)
                )
                for anchor_index, (sample_id, subject_id) in enumerate(
                    zip(sample_ids, subject_ids, strict=True)
                ):
                    if sample_id in local_sample_ids:
                        raise RuntimeError(f"duplicate local free-running anchor {sample_id}")
                    local_sample_ids.add(sample_id)
                    encoded = {
                        key: value[anchor_index : anchor_index + 1]
                        for key, value in encoded_batch.items()
                    }
                    truth_primitives = {
                        key: value[anchor_index : anchor_index + 1]
                        for key, value in truth_batch["target_primitives"].items()
                    }
                    truth_primitive_masks = {
                        key: value[anchor_index : anchor_index + 1]
                        for key, value in truth_batch["target_primitive_masks"].items()
                    }
                    truth_values, truth_masks = project_physical_primitives(
                        truth_primitives, contract, physical_schema
                    )
                    truth_phi = standardize_primitive_trajectory(
                        truth_primitives,
                        truth_primitive_masks,
                        primitive_schema,
                        scale_artifact,
                    )
                    generated_values: list[torch.Tensor] = []
                    generated_masks: list[torch.Tensor] = []
                    generated_phi: list[torch.Tensor] = []
                    coherence_rows: list[dict[str, Any]] = []
                    trajectory_start = 0
                    while trajectory_start < trajectories_per_anchor:
                        count = min(
                            batch_size, trajectories_per_anchor - trajectory_start
                        )
                        seed = common_random_seed(
                            crn_seed,
                            sample_id,
                            trajectory_start=trajectory_start,
                            trajectory_count=count,
                        )
                        sampler = RegistryPrimitiveSampler(
                            contract.process_registry,
                            metadata,
                            expected_lab_scale_artifact_hash=expected_lab_scale_artifact_hash,
                        )
                        expanded = _expand_encoded(encoded, count)
                        with _fork_rng(device, seed):
                            with autocast():
                                outputs = core_model.rollout_from_encoded(
                                    expanded["memory"],
                                    expanded["memory_mask"],
                                    expanded["query_tokens"],
                                    sampler=sampler,
                                    relation_adjacency=batch.get("relation_adjacency"),
                                    relation_type_lags=batch.get("relation_type_lags"),
                                    mode=mode,
                                )
                        primitives = outputs.get("generated_primitives")
                        primitive_masks = outputs.get("generated_primitive_masks")
                        if not isinstance(primitives, Mapping) or not isinstance(
                            primitive_masks, Mapping
                        ):
                            raise ValueError(
                                "cached rollout did not return generated primitives"
                            )
                        reports = generated_coherence_report(
                            primitives, primitive_masks, contract
                        )
                        if len(reports) != count:
                            raise AssertionError(
                                "coherence report does not align with trajectories"
                            )
                        physical_values, physical_masks = project_physical_primitives(
                            primitives, contract, physical_schema
                        )
                        phi = standardize_primitive_trajectory(
                            primitives,
                            primitive_masks,
                            primitive_schema,
                            scale_artifact,
                        )
                        generated_values.append(physical_values)
                        generated_masks.append(physical_masks)
                        generated_phi.append(phi)
                        coherence_rows.extend(reports)
                        if trajectory_start == 0:
                            row_index = 0
                            export = _trajectory_export_row(
                                sample_id=sample_id,
                                subject_id=subject_id,
                                mode=mode,
                                trajectory_index=0,
                                crn_seed=seed,
                                rng_row_index=row_index,
                                primitives=primitives,
                                primitive_masks=primitive_masks,
                                primitive_specs=primitive_specs,
                                physical_values=physical_values,
                                physical_masks=physical_masks,
                                row_index=row_index,
                            )
                            primitive_handle.write(_json_line(export))
                        trajectory_start += count

                    # Scoring is a small [100,6,D] workload with many scalar
                    # summaries.  Move each complete ensemble once so the
                    # projection loop cannot trigger hundreds of tiny GPU-host
                    # synchronizations per anchor.
                    ensemble_values = torch.cat(generated_values, dim=0).detach().cpu()
                    ensemble_masks = torch.cat(generated_masks, dim=0).detach().cpu()
                    ensemble_phi = torch.cat(generated_phi, dim=0).detach().cpu()
                    physical_scores = score_physical_ensemble(
                        ensemble_values,
                        ensemble_masks,
                        truth_values[0].detach().cpu(),
                        truth_masks[0].detach().cpu(),
                        physical_schema,
                    )
                    _update_calibration_state(
                        local_calibration,
                        physical_scores.pop("branch_calibration_rows"),
                    )
                    _update_coverage_state(
                        local_coverage,
                        physical_scores.pop("coverage_by_projection"),
                    )
                    trajectory_scores = score_standardized_primitive_ensemble(
                        ensemble_phi,
                        truth_phi[0].detach().cpu(),
                        primitive_schema,
                        contract.active_core_relation_edges,
                        promotion_metric_contract,
                    )
                    coherent = sum(bool(row["coherent"]) for row in coherence_rows)
                    violation_counts: dict[str, int] = defaultdict(int)
                    for report in coherence_rows:
                        for code in report["violations"]:
                            violation_counts[str(code)] += 1
                    anchor_row = {
                        "schema_version": "trauma_predict.multires_event_v2_free_running_anchor.v1",
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "mode": mode,
                        "step": int(step),
                        "trajectories": trajectories_per_anchor,
                        "energy_score": trajectory_scores["energy_score"],
                        "lag1_variogram_score_p0_5": trajectory_scores[
                            "lag1_variogram_score_p0_5"
                        ],
                        "field_macro_lag1_variogram_score_p0_5": trajectory_scores[
                            "field_macro_lag1_variogram_score_p0_5"
                        ],
                        "relation_edge_macro_variogram_score_p0_5": trajectory_scores[
                            "relation_edge_macro_variogram_score_p0_5"
                        ],
                        "marginal_value_crps": trajectory_scores[
                            "marginal_value_crps"
                        ],
                        "marginal_state_crps": trajectory_scores[
                            "marginal_state_crps"
                        ],
                        "relation_variogram_by_type": trajectory_scores[
                            "relation_variogram_by_type"
                        ],
                        "physical_scores": physical_scores,
                        "coherence_rate": coherent / trajectories_per_anchor,
                        "coherent_trajectories": coherent,
                        "coherence_violations": dict(sorted(violation_counts.items())),
                        "identity": identity,
                    }
                    local_anchor_count += 1
                    anchor_handle.write(_json_line(anchor_row))
                    if local_anchor_count % 25 == 0:
                        _emit_rank_progress(
                            path=rank_progress_path,
                            rank=rank,
                            mode=mode,
                            completed_anchors=local_anchor_count,
                            started_at=rank_started_at,
                        )

    if local_anchor_count % 25 != 0:
        _emit_rank_progress(
            path=rank_progress_path,
            rank=rank,
            mode=mode,
            completed_anchors=local_anchor_count,
            started_at=rank_started_at,
        )

    def build_rank_payload() -> dict[str, Any]:
        return {
            "manifest": {
                "rank": rank,
                "anchors": local_anchor_count,
                "audit_trajectory_sample_path": primitive_path.name,
                "audit_trajectory_sample_sha256": sha256_file(primitive_path),
                "retained_audit_trajectories": local_anchor_count,
                "per_anchor_score_path": anchor_path.name,
                "per_anchor_score_sha256": sha256_file(anchor_path),
                "progress_metrics_path": rank_progress_path.name,
                "progress_metrics_sha256": sha256_file(rank_progress_path),
            },
            "calibration": local_calibration,
            "coverage": local_coverage,
        }

    rank_payloads = _collect_distributed_phase(
        "free-running rank artifact finalization",
        build_rank_payload,
    )
    manifests = [payload["manifest"] for payload in rank_payloads]
    calibration_parts = [payload["calibration"] for payload in rank_payloads]
    coverage_parts = [payload["coverage"] for payload in rank_payloads]

    def assemble_rank_zero_result() -> dict[str, Any] | None:
        if rank != 0:
            return None
        rows: list[dict[str, Any]] = []
        for manifest in sorted(manifests, key=lambda item: int(item["rank"])):
            rows.extend(
                _read_jsonl(output_root / str(manifest["per_anchor_score_path"]))
            )
        if len(rows) != expected_samples:
            raise RuntimeError(
                f"free-running evaluation expected {expected_samples} anchors, got {len(rows)}"
            )
        ids = [str(row["sample_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError("free-running sampler introduced duplicate persisted anchors")
        result = _summarize_free_running_rows(
            rows,
            mode=mode,
            step=step,
            trajectories=trajectories_per_anchor,
            identity=identity,
            crn_contract=crn_contract,
            calibration=_merge_calibration_states(calibration_parts),
            coverage=_merge_coverage_states(coverage_parts),
        )
        result["sample_schema_path"] = schema_path.name
        result["sample_schema_sha256"] = sha256_file(schema_path)
        result["shards"] = manifests
        manifest_path = output_root / "manifest.json"
        _atomic_json(
            manifest_path,
            {
                "schema_version": "trauma_predict.multires_event_v2_free_running_manifest.v1",
                "created_at": utc_now(),
                "evaluation": result,
                "per_anchor_score_shards": manifests,
            },
        )
        result["manifest_path"] = manifest_path.name
        result["manifest_sha256"] = sha256_file(manifest_path)
        _atomic_json(output_root / "evaluation.json", result)
        if metrics_path is not None:
            append_jsonl(metrics_path, {"event": "v2_free_running_evaluation", **result})
        return result

    assembled = _collect_distributed_phase(
        "free-running rank-zero report assembly",
        assemble_rank_zero_result,
    )
    result = assembled[0]
    if not isinstance(result, Mapping):
        raise RuntimeError("free-running evaluation result assembly failed")
    return dict(result)


def evaluate_multires_event_v2_promotion(
    *,
    block_teacher_path: str | Path,
    trajectory_teacher_path: str | Path,
    relational_teacher_path: str | Path,
    block_free_running_path: str | Path,
    trajectory_free_running_path: str | Path,
    relational_free_running_path: str | Path,
    promotion_metric_contract: Mapping[str, Any],
    expected_anchors: int = 6309,
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int = 20260713,
) -> dict[str, Any]:
    """Apply the frozen structural sequential promotion decision."""

    bootstrap_contract = promotion_metric_contract.get("bootstrap")
    if not isinstance(bootstrap_contract, Mapping):
        raise ValueError("V2 promotion metric contract lacks bootstrap settings")
    if (
        bootstrap_repetitions != int(bootstrap_contract["repetitions"])
        or bootstrap_seed != int(bootstrap_contract["seed"])
    ):
        raise ValueError("V2 promotion bootstrap is frozen to 2,000 draws/seed 20260713")
    teacher = {
        "block": _index_rows(_read_jsonl(Path(block_teacher_path)), expected_anchors),
        "trajectory": _index_rows(
            _read_jsonl(Path(trajectory_teacher_path)), expected_anchors
        ),
        "relational": _index_rows(
            _read_jsonl(Path(relational_teacher_path)), expected_anchors
        ),
    }
    free = {
        "block": _index_rows(
            _read_free_running_rows(Path(block_free_running_path)), expected_anchors
        ),
        "trajectory": _index_rows(
            _read_free_running_rows(Path(trajectory_free_running_path)), expected_anchors
        ),
        "relational": _index_rows(
            _read_free_running_rows(Path(relational_free_running_path)), expected_anchors
        ),
    }
    _assert_matched_identity(teacher)
    _assert_matched_identity(free)
    _assert_teacher_free_anchor_identity(teacher, free)
    _assert_common_random_contract(free)
    _assert_teacher_free_contract_identity(teacher, free)
    subject_order = _shared_subject_order(teacher["block"])
    expected_subjects = int(promotion_metric_contract["population"]["subjects"])
    if expected_anchors == int(promotion_metric_contract["population"]["anchors"]) and len(
        subject_order
    ) != expected_subjects:
        raise ValueError(
            f"production promotion requires {expected_subjects} subjects, got "
            f"{len(subject_order)}"
        )
    bootstrap_schedule = _shared_subject_bootstrap_schedule(
        len(subject_order),
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )

    trajectory_teacher = _paired_subject_bootstrap_delta(
        teacher["block"],
        teacher["trajectory"],
        value=lambda row: float(row["joint_nll"]),
        subject_order=subject_order,
        bootstrap_schedule=bootstrap_schedule,
        metric="teacher_joint_nll_nats_per_anchor",
    )
    relational_teacher = _paired_subject_bootstrap_delta(
        teacher["trajectory"],
        teacher["relational"],
        value=lambda row: float(row["joint_nll"]),
        subject_order=subject_order,
        bootstrap_schedule=bootstrap_schedule,
        metric="teacher_joint_nll_nats_per_anchor",
    )
    _add_log_score_units(trajectory_teacher)
    _add_log_score_units(relational_teacher)
    _add_log_density_ratios(trajectory_teacher)
    _add_log_density_ratios(relational_teacher)
    gates = promotion_metric_contract.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("V2 promotion metric contract lacks gates")
    trajectory_gate = gates["trajectory_over_block"]
    relational_gate = gates["relational_over_trajectory"]
    if not isinstance(trajectory_gate, Mapping) or not isinstance(
        relational_gate, Mapping
    ):
        raise ValueError("V2 promotion structural gates are invalid")
    trajectory_teacher_pass = (
        float(trajectory_teacher["ci95"]["upper"])
        < float(trajectory_gate["teacher_joint_nll_delta_ci95_upper_lt"])
    )
    relational_teacher_pass = (
        float(relational_teacher["ci95"]["upper"])
        < float(relational_gate["teacher_joint_nll_delta_ci95_upper_lt"])
    )

    trajectory_temporal = _paired_subject_bootstrap_ratio(
        free["block"],
        free["trajectory"],
        value=lambda row: float(row["field_macro_lag1_variogram_score_p0_5"]),
        subject_order=subject_order,
        bootstrap_schedule=bootstrap_schedule,
        metric="field_macro_lag1_variogram_score_p0_5",
    )
    relational_structure = _paired_subject_bootstrap_ratio(
        free["trajectory"],
        free["relational"],
        value=lambda row: float(row["relation_edge_macro_variogram_score_p0_5"]),
        subject_order=subject_order,
        bootstrap_schedule=bootstrap_schedule,
        metric="relation_edge_macro_variogram_score_p0_5",
    )
    trajectory_marginal = {
        partition: _paired_subject_bootstrap_ratio(
            free["block"],
            free["trajectory"],
            value=lambda row, name=partition: float(row[f"marginal_{name}_crps"]),
            subject_order=subject_order,
            bootstrap_schedule=bootstrap_schedule,
            metric=f"marginal_{partition}_crps",
        )
        for partition in ("value", "state")
    }
    relational_marginal = {
        partition: _paired_subject_bootstrap_ratio(
            free["trajectory"],
            free["relational"],
            value=lambda row, name=partition: float(row[f"marginal_{name}_crps"]),
            subject_order=subject_order,
            bootstrap_schedule=bootstrap_schedule,
            metric=f"marginal_{partition}_crps",
        )
        for partition in ("value", "state")
    }
    trajectory_physical = _physical_noninferiority_gate(
        free["block"],
        free["trajectory"],
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    relational_physical = _physical_noninferiority_gate(
        free["trajectory"],
        free["relational"],
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    coherence = {
        mode: _coherence_gate(rows) for mode, rows in free.items()
    }
    trajectory_coherence_pass = coherence["block"]["passed"] and coherence[
        "trajectory"
    ]["passed"]
    relational_coherence_pass = coherence["trajectory"]["passed"] and coherence[
        "relational"
    ]["passed"]
    trajectory_temporal_pass = (
        float(trajectory_temporal["observed_ratio"])
        <= float(trajectory_gate["temporal_score_observed_ratio_lte"])
        and float(trajectory_temporal["ci95"]["upper"])
        < float(trajectory_gate["temporal_score_ci95_upper_lt"])
    )
    relational_structure_pass = (
        float(relational_structure["observed_ratio"])
        <= float(relational_gate["relation_score_observed_ratio_lte"])
        and float(relational_structure["ci95"]["upper"])
        < float(relational_gate["relation_score_ci95_upper_lt"])
    )
    trajectory_marginal_pass = all(
        float(trajectory_marginal[name]["ci95"]["upper"])
        < float(trajectory_gate[f"{name}_marginal_score_ci95_upper_lt"])
        for name in ("value", "state")
    )
    relational_marginal_pass = all(
        float(relational_marginal[name]["ci95"]["upper"])
        < float(relational_gate[f"{name}_marginal_score_ci95_upper_lt"])
        for name in ("value", "state")
    )
    trajectory_pass = (
        trajectory_teacher_pass
        and trajectory_temporal_pass
        and trajectory_marginal_pass
        and trajectory_coherence_pass
    )
    relational_increment_pass = (
        relational_teacher_pass
        and relational_structure_pass
        and relational_marginal_pass
        and relational_coherence_pass
    )
    relational_pass = trajectory_pass and relational_increment_pass
    winner = "relational" if relational_pass else "trajectory" if trajectory_pass else "block"
    return {
        "schema_version": "trauma_predict.multires_event_v2_promotion.v2",
        "created_at": utc_now(),
        "anchors": expected_anchors,
        "bootstrap": {
            "unit": "subject_id",
            "repetitions": bootstrap_repetitions,
            "seed": bootstrap_seed,
            "interval": "paired_percentile_95",
            "shared_subject_index_schedule": True,
        },
        "teacher_log_score": {
            "units": "fixed_six_block_joint_target_nats_per_anchor",
            "trajectory_minus_block": trajectory_teacher,
            "relational_minus_trajectory": relational_teacher,
            "not_coordinate_percentage": True,
        },
        "structural_scores": {
            "trajectory_minus_block": trajectory_temporal,
            "relational_minus_trajectory": relational_structure,
        },
        "marginal_noninferiority": {
            "rule": "field_balanced_phi_marginal_score_ratio_CI95_upper_lt_1.01",
            "trajectory_over_block": trajectory_marginal,
            "relational_over_trajectory": relational_marginal,
        },
        "physical_report": {
            "decision_role": "report_only_no_veto",
            "trajectory_minus_block": trajectory_physical,
            "relational_minus_trajectory": relational_physical,
        },
        "coherence": {
            "rule": "100_percent_generated_trajectories",
            "modes": coherence,
        },
        "care_and_procedure": {
            "status": "not_applicable",
            "reason": "care_and_procedure_joint_objective_off",
        },
        "gates": {
            "trajectory_over_block": {
                "teacher_joint_nll_superiority": trajectory_teacher_pass,
                "temporal_structure": trajectory_temporal_pass,
                "marginal_noninferiority": trajectory_marginal_pass,
                "coherence": trajectory_coherence_pass,
                "passed": trajectory_pass,
            },
            "relational_over_trajectory": {
                "teacher_joint_nll_superiority": relational_teacher_pass,
                "relation_structure": relational_structure_pass,
                "marginal_noninferiority": relational_marginal_pass,
                "coherence": relational_coherence_pass,
                "increment_passed": relational_increment_pass,
                "passed": relational_pass,
            },
        },
        "winner": winner,
        "trajectory_promoted": trajectory_pass,
        "relational_promoted": relational_pass,
        "promoted": winner != "block",
    }


def common_random_seed(
    base_seed: int,
    sample_id: str,
    *,
    trajectory_start: int,
    trajectory_count: int,
) -> int:
    payload = {
        "schema": CRN_SCHEDULE_SCHEMA,
        "base_seed": int(base_seed),
        "sample_id": str(sample_id),
        "trajectory_start": int(trajectory_start),
        "trajectory_count": int(trajectory_count),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def _encode_batch_once(
    model: Any,
    batch: Mapping[str, Any],
    *,
    expected_batch_size: int,
) -> Mapping[str, torch.Tensor]:
    input_batch = batch.get("input_batch")
    if not isinstance(input_batch, Mapping):
        raise ValueError("free-running batch lacks input_batch")
    missing = set(MODEL_INPUT_KEYS).difference(input_batch)
    if missing:
        raise ValueError(f"free-running input lacks model tensors: {sorted(missing)}")
    encoded_result = model.encode_for_rollout(
        **{key: input_batch[key] for key in MODEL_INPUT_KEYS}
    )
    required = {"memory", "memory_mask", "query_tokens"}
    if isinstance(encoded_result, Mapping):
        encoded = dict(encoded_result)
        if set(encoded) != required:
            raise ValueError(
                f"encoded rollout state must contain exactly {sorted(required)}"
            )
    elif isinstance(encoded_result, tuple) and len(encoded_result) == 3:
        encoded = dict(zip(("memory", "memory_mask", "query_tokens"), encoded_result))
    else:
        raise ValueError(
            "encode_for_rollout must return (memory,memory_mask,query_tokens) or an exact mapping"
        )
    for key in required:
        value = encoded[key]
        if not isinstance(value, torch.Tensor) or value.shape[0] != expected_batch_size:
            raise ValueError(
                f"encoded rollout {key} must have batch size {expected_batch_size}"
            )
    return dict(encoded)


def _primitive_batch_slice_inputs(
    batch: Mapping[str, Any],
    *,
    expected_batch_size: int,
) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    for bank_name in ("target_primitives", "target_primitive_masks"):
        source = batch.get(bank_name)
        if not isinstance(source, Mapping):
            raise ValueError(f"free-running batch lacks {bank_name}")
        banks: dict[str, torch.Tensor] = {}
        for likelihood_id, value in source.items():
            if not isinstance(value, torch.Tensor) or value.shape[0] != expected_batch_size:
                raise ValueError(
                    f"{bank_name}.{likelihood_id} must have batch size "
                    f"{expected_batch_size}"
                )
            banks[str(likelihood_id)] = value
        result[bank_name] = banks
    if set(result["target_primitives"]) != set(result["target_primitive_masks"]):
        raise ValueError("free-running target primitive and mask banks differ")
    return result


def _expand_encoded(encoded: Mapping[str, Any], count: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in encoded.items():
        if not isinstance(value, torch.Tensor) or value.shape[0] != 1:
            raise ValueError(f"encoded rollout {key} cannot be ensemble-expanded")
        result[key] = value.expand((count,) + tuple(value.shape[1:])).contiguous()
    return result


def _autocast_factory(device: torch.device, precision: str) -> Any:
    if device.type != "cuda" or precision == "fp32":
        return nullcontext

    def factory() -> Any:
        try:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        except AttributeError:  # pragma: no cover - older supported torch
            return torch.cuda.amp.autocast(dtype=torch.float16)

    return factory


@contextmanager
def _fork_rng(device: torch.device, seed: int) -> Any:
    devices = [int(device.index or 0)] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        yield


def _trajectory_export_row(
    *,
    sample_id: str,
    subject_id: str,
    mode: str,
    trajectory_index: int,
    crn_seed: int,
    rng_row_index: int,
    primitives: Mapping[str, torch.Tensor],
    primitive_masks: Mapping[str, torch.Tensor],
    primitive_specs: Sequence[Any],
    physical_values: torch.Tensor,
    physical_masks: torch.Tensor,
    row_index: int,
) -> dict[str, Any]:
    # Concatenate on device and cross the device boundary once per bank.  The
    # schema carries offsets/widths needed to recover each primitive vector.
    flat_values = torch.cat(
        tuple(
            _bank4(primitives[spec.likelihood_id])[
                :, spec.block_index, spec.field_index, :
            ]
            for spec in primitive_specs
        ),
        dim=-1,
    )
    flat_masks = torch.cat(
        tuple(
            _bank4(primitive_masks[spec.likelihood_id])[
                :, spec.block_index, spec.field_index, :
            ]
            for spec in primitive_specs
        ),
        dim=-1,
    )
    retained_physical_values = physical_values[row_index].detach().cpu()
    retained_physical_masks = physical_masks[row_index].detach().cpu()
    return {
        "schema_version": "trauma_predict.multires_event_v2_primitive_sample.v2",
        "sample_id": sample_id,
        "subject_id": subject_id,
        "mode": mode,
        "trajectory_index": int(trajectory_index),
        "crn_seed": int(crn_seed),
        "rng_row_index": int(rng_row_index),
        "primitive_values_flat": [
            float(item) for item in flat_values[row_index].detach().cpu().tolist()
        ],
        "primitive_component_masks_flat": [
            int(bool(item))
            for item in flat_masks[row_index].detach().cpu().tolist()
        ],
        "physical_projection_values": _masked_projection_json(
            retained_physical_values, retained_physical_masks
        ),
        "physical_projection_masks": retained_physical_masks
        .to(dtype=torch.uint8)
        .tolist(),
    }


def _summarize_free_running_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    step: int,
    trajectories: int,
    identity: Mapping[str, Any],
    crn_contract: Mapping[str, Any],
    calibration: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    coherent = sum(int(row["coherent_trajectories"]) for row in rows)
    total_trajectories = len(rows) * trajectories
    result = {
        "schema_version": FREE_RUNNING_SCHEMA,
        "evaluated_at": utc_now(),
        "mode": mode,
        "step": int(step),
        "anchors": len(rows),
        "subjects": len({str(row["subject_id"]) for row in rows}),
        "trajectories_per_anchor": trajectories,
        "energy_score": _scalar_summary(rows, "energy_score"),
        "lag1_variogram_score_p0_5": _scalar_summary(
            rows, "lag1_variogram_score_p0_5"
        ),
        "field_macro_lag1_variogram_score_p0_5": _scalar_summary(
            rows, "field_macro_lag1_variogram_score_p0_5"
        ),
        "relation_edge_macro_variogram_score_p0_5": _scalar_summary(
            rows, "relation_edge_macro_variogram_score_p0_5"
        ),
        "marginal_value_crps": _scalar_summary(rows, "marginal_value_crps"),
        "marginal_state_crps": _scalar_summary(rows, "marginal_state_crps"),
        "relation_variogram_by_type": _mapping_summary(
            rows, ("relation_variogram_by_type",)
        ),
        "branch_calibration": dict(calibration),
        "physical_crps_by_projection": _mapping_summary(
            rows, ("physical_scores", "crps_by_projection")
        ),
        "physical_brier_by_projection": _mapping_summary(
            rows, ("physical_scores", "brier_by_projection")
        ),
        "physical_median_mae_by_projection": _mapping_summary(
            rows, ("physical_scores", "median_mae_by_projection")
        ),
        "physical_rps_by_projection": _mapping_summary(
            rows, ("physical_scores", "rps_by_projection")
        ),
        "conditional_sample_coverage_by_projection": dict(coverage),
        "physical_metric_contract_status": (
            "complete"
            if all(
                row["physical_scores"]["physical_metric_contract_status"] == "complete"
                for row in rows
            )
            else "incomplete_conditional_sample_coverage"
        ),
        "coherence": {
            "coherent_trajectories": coherent,
            "total_trajectories": total_trajectories,
            "rate": coherent / total_trajectories,
            "required_rate": 1.0,
        },
        "identity": dict(identity),
        "crn_contract": dict(crn_contract),
        "standardized_primitive_vector": {
            "status": "active",
            "aggregation": "one_coordinate_per_registered_primitive_state_no_projection_duplicates",
        },
        "physical_projection_table": {
            "views_per_block": 155,
            "cross_unit_aggregation": "forbidden",
            "conditional_score_subset": "truth_active_blocks",
        },
        "care_and_procedure": {
            "status": "not_applicable",
            "reason": "care_and_procedure_joint_objective_off",
        },
    }
    return result


def _emit_rank_progress(
    *,
    path: Path,
    rank: int,
    mode: str,
    completed_anchors: int,
    started_at: float,
) -> None:
    elapsed = max(time.monotonic() - started_at, 1e-12)
    row = {
        "event": "v2_free_running_rank_progress",
        "evaluated_at": utc_now(),
        "rank": int(rank),
        "mode": str(mode),
        "completed_anchors": int(completed_anchors),
        "elapsed_seconds": elapsed,
        "anchors_per_second": completed_anchors / elapsed,
    }
    append_rank_local_jsonl(path, row, rank_value=rank)
    print(
        "V2_FREE_PROGRESS "
        f"rank={rank} mode={mode} anchors={completed_anchors} "
        f"elapsed={elapsed:.1f}s anchors_per_second={completed_anchors / elapsed:.4f}",
        flush=True,
    )


def verify_rank_local_artifact_preflight(
    *,
    output_dir: str | Path,
    mode: str,
) -> dict[str, Any]:
    """Exercise rank-local write, hash, gather, and rank-zero assembly.

    The capacity route runs this immediately after DDP initialization so an
    invalid per-rank artifact contract fails before model construction or any
    expensive rollout.
    """

    if mode not in {"block", "trajectory", "relational"}:
        raise ValueError("rank artifact preflight mode must be block/trajectory/relational")
    output_root = Path(output_dir).resolve()
    rank, world_size = _rank_world()
    _collect_distributed_phase(
        "rank artifact preflight root materialization",
        lambda: output_root.mkdir(parents=True, exist_ok=True) if rank == 0 else None,
    )
    progress_path = output_root / f"progress.rank{rank:05d}.jsonl"
    started_at = time.monotonic()
    _emit_rank_progress(
        path=progress_path,
        rank=rank,
        mode=mode,
        completed_anchors=0,
        started_at=started_at,
    )

    def inspect_local_artifact() -> dict[str, Any]:
        rows = _read_jsonl(progress_path)
        if len(rows) != 1:
            raise RuntimeError(
                f"rank artifact preflight expected one progress row, got {len(rows)}"
            )
        row = rows[0]
        if (
            int(row.get("rank", -1)) != rank
            or int(row.get("completed_anchors", -1)) != 0
            or str(row.get("mode")) != mode
        ):
            raise RuntimeError("rank artifact preflight progress identity mismatch")
        return {
            "rank": rank,
            "path": progress_path.name,
            "sha256": sha256_file(progress_path),
            "rows": len(rows),
        }

    artifacts = _collect_distributed_phase(
        "rank artifact preflight local verification",
        inspect_local_artifact,
    )

    def assemble_preflight() -> dict[str, Any] | None:
        if rank != 0:
            return None
        ranks = [int(row["rank"]) for row in artifacts]
        if ranks != list(range(world_size)):
            raise RuntimeError(
                f"rank artifact preflight expected ranks 0..{world_size - 1}, got {ranks}"
            )
        payload = {
            "schema_version": RANK_ARTIFACT_PREFLIGHT_SCHEMA,
            "created_at": utc_now(),
            "status": "PASSED",
            "mode": mode,
            "world_size": world_size,
            "rank_artifacts": artifacts,
        }
        manifest_path = output_root / "manifest.json"
        _atomic_json(manifest_path, payload)
        return {
            **payload,
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
        }

    assembled = _collect_distributed_phase(
        "rank artifact preflight manifest assembly",
        assemble_preflight,
    )
    result = assembled[0]
    if not isinstance(result, Mapping):
        raise RuntimeError("rank artifact preflight result assembly failed")
    return dict(result)


def validate_rank_local_artifact_preflight(
    output_dir: str | Path,
    *,
    expected_mode: str,
    expected_world_size: int = 2,
) -> dict[str, Any]:
    """Re-open every retained rank-local canary byte and validate its contract."""

    if expected_mode not in {"block", "trajectory", "relational"}:
        raise ValueError("rank artifact validation mode is invalid")
    if expected_world_size < 1:
        raise ValueError("rank artifact validation world size must be positive")
    output_root = Path(output_dir).resolve()
    manifest_path = output_root / "manifest.json"
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.resolve() != manifest_path
    ):
        raise ValueError("rank artifact preflight manifest is missing, linked, or escaped")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("rank artifact preflight manifest is not a mapping")
    artifacts = manifest.get("rank_artifacts")
    if (
        manifest.get("schema_version") != RANK_ARTIFACT_PREFLIGHT_SCHEMA
        or manifest.get("status") != "PASSED"
        or manifest.get("mode") != expected_mode
        or int(manifest.get("world_size", -1)) != expected_world_size
        or not isinstance(artifacts, list)
        or len(artifacts) != expected_world_size
    ):
        raise ValueError("rank artifact preflight manifest contract failed")
    for expected_rank, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            raise ValueError("rank artifact preflight row is not a mapping")
        expected_name = f"progress.rank{expected_rank:05d}.jsonl"
        if (
            int(artifact.get("rank", -1)) != expected_rank
            or str(artifact.get("path") or "") != expected_name
            or int(artifact.get("rows", -1)) != 1
        ):
            raise ValueError("rank artifact preflight rank/path contract failed")
        progress_path = output_root / expected_name
        if (
            progress_path.is_symlink()
            or not progress_path.is_file()
            or progress_path.resolve().parent != output_root
            or sha256_file(progress_path) != str(artifact.get("sha256") or "")
        ):
            raise ValueError("rank artifact preflight retained file/hash failed")
        rows = _read_jsonl(progress_path)
        if (
            len(rows) != 1
            or rows[0].get("event") != "v2_free_running_rank_progress"
            or int(rows[0].get("rank", -1)) != expected_rank
            or rows[0].get("mode") != expected_mode
            or int(rows[0].get("completed_anchors", -1)) != 0
        ):
            raise ValueError("rank artifact preflight retained row contract failed")
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _empty_calibration_state(*, bins: int = 10) -> dict[str, Any]:
    if bins < 2:
        raise ValueError("calibration requires at least two bins")
    return {"bins": int(bins), "families": {}}


def _update_calibration_state(
    state: dict[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    bins = int(state["bins"])
    families = state["families"]
    if not isinstance(families, dict):
        raise ValueError("calibration accumulator families must be mutable")
    for row in rows:
        probability = float(row["probability"])
        outcome = int(row["outcome"])
        if not 0.0 <= probability <= 1.0 or outcome not in {0, 1}:
            raise ValueError("calibration event lies outside Bernoulli support")
        bin_index = min(int(probability * bins), bins - 1)
        for family in ("overall", str(row["family"])):
            accumulator = families.setdefault(
                family,
                {
                    "events": 0,
                    "brier_sum": 0.0,
                    "counts": [0] * bins,
                    "probability_sum": [0.0] * bins,
                    "outcome_sum": [0.0] * bins,
                },
            )
            accumulator["events"] += 1
            accumulator["brier_sum"] += (probability - outcome) ** 2
            accumulator["counts"][bin_index] += 1
            accumulator["probability_sum"][bin_index] += probability
            accumulator["outcome_sum"][bin_index] += outcome


def _merge_calibration_states(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not states:
        raise ValueError("calibration merge requires rank states")
    bins = int(states[0]["bins"])
    merged = _empty_calibration_state(bins=bins)
    merged_families = merged["families"]
    for state in states:
        if int(state.get("bins", -1)) != bins:
            raise ValueError("calibration bins differ across ranks")
        families = state.get("families")
        if not isinstance(families, Mapping):
            raise ValueError("calibration rank state lacks families")
        for family, source in families.items():
            if not isinstance(source, Mapping):
                raise ValueError("calibration family state is invalid")
            target = merged_families.setdefault(
                str(family),
                {
                    "events": 0,
                    "brier_sum": 0.0,
                    "counts": [0] * bins,
                    "probability_sum": [0.0] * bins,
                    "outcome_sum": [0.0] * bins,
                },
            )
            target["events"] += int(source["events"])
            target["brier_sum"] += float(source["brier_sum"])
            for key in ("counts", "probability_sum", "outcome_sum"):
                values = source[key]
                if not isinstance(values, Sequence) or len(values) != bins:
                    raise ValueError("calibration rank bin width changed")
                for index, value in enumerate(values):
                    target[key][index] += value

    result: dict[str, Any] = {}
    for family, source in sorted(merged_families.items()):
        total = int(source["events"])
        bin_rows: list[dict[str, Any]] = []
        ece = 0.0
        for index in range(bins):
            count = int(source["counts"][index])
            predicted = source["probability_sum"][index] / count if count else None
            observed = source["outcome_sum"][index] / count if count else None
            if count:
                ece += count / total * abs(float(predicted) - float(observed))
            bin_rows.append(
                {
                    "bin": index,
                    "lower": index / bins,
                    "upper": (index + 1) / bins,
                    "count": count,
                    "mean_probability": predicted,
                    "observed_frequency": observed,
                }
            )
        result[family] = {
            "events": total,
            "brier": float(source["brier_sum"]) / total if total else None,
            "ece": ece if total else None,
            "bins": bin_rows,
        }
    return result


def _physical_noninferiority_gate(
    control: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    for label, rows in (("control", control), ("candidate", candidate)):
        incomplete = [
            sample_id
            for sample_id, row in rows.items()
            if row["physical_scores"]["physical_metric_contract_status"] != "complete"
        ]
        if incomplete:
            blockers.append(f"{label}_conditional_coverage_incomplete:{len(incomplete)}")
    metrics: dict[str, Any] = {}
    for key in ("energy_score", "lag1_variogram_score_p0_5"):
        metrics[key] = _paired_subject_bootstrap(
            control,
            candidate,
            value=lambda row, name=key: float(row[name]),
            repetitions=repetitions,
            seed=seed,
            metric=key,
        )
    control_keys = _nested_metric_keys(control, ("physical_scores", "crps_by_projection"))
    candidate_keys = _nested_metric_keys(
        candidate, ("physical_scores", "crps_by_projection")
    )
    if control_keys != candidate_keys:
        blockers.append("physical_crps_projection_key_mismatch")
    crps: dict[str, Any] = {}
    for projection_id in sorted(control_keys & candidate_keys):
        control_support = _nested_metric_anchor_support(
            control, ("physical_scores", "crps_by_projection"), projection_id
        )
        candidate_support = _nested_metric_anchor_support(
            candidate, ("physical_scores", "crps_by_projection"), projection_id
        )
        if control_support != candidate_support:
            blockers.append(f"physical_crps:{projection_id}:anchor_support_mismatch")
            continue
        try:
            crps[projection_id] = _paired_subject_bootstrap(
                control,
                candidate,
                value=lambda row, name=projection_id: float(
                    row["physical_scores"]["crps_by_projection"][name]
                ),
                repetitions=repetitions,
                seed=seed,
                metric=f"physical_crps:{projection_id}",
                allow_missing=True,
            )
        except ValueError as exc:
            blockers.append(f"physical_crps:{projection_id}:{exc}")
    scalar_pass = all(float(row["ci95"]["upper"]) <= 0.0 for row in metrics.values())
    crps_pass = bool(crps) and all(
        float(row["ci95"]["upper"]) <= 0.0 for row in crps.values()
    )
    return {
        "rule": "all_CI95_upper_le_zero",
        "standardized_trajectory_scores": metrics,
        "physical_crps_by_projection": crps,
        "blockers": blockers,
        "passed": not blockers and scalar_pass and crps_pass,
    }


def _shared_subject_order(
    rows: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    subjects = sorted({str(row["subject_id"]) for row in rows.values()})
    if not subjects or any(not subject for subject in subjects):
        raise ValueError("promotion rows contain no valid subjects")
    return tuple(subjects)


def _shared_subject_bootstrap_schedule(
    subject_count: int,
    *,
    repetitions: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    if subject_count < 1 or repetitions < 1:
        raise ValueError("shared subject bootstrap dimensions must be positive")
    rng = random.Random(seed)
    return tuple(
        tuple(rng.randrange(subject_count) for _ in range(subject_count))
        for _ in range(repetitions)
    )


def _subject_metric_values(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    value: Any,
    subject_order: Sequence[str],
    metric: str,
) -> list[float]:
    by_subject: dict[str, list[float]] = defaultdict(list)
    for sample_id in sorted(rows):
        row = rows[sample_id]
        subject_id = str(row["subject_id"])
        observed = float(value(row))
        if not math.isfinite(observed):
            raise ValueError(f"non-finite subject-bootstrap value for {metric}")
        by_subject[subject_id].append(observed)
    if set(by_subject) != set(subject_order):
        raise ValueError(f"subject support differs for promotion metric {metric}")
    return [sum(by_subject[subject]) / len(by_subject[subject]) for subject in subject_order]


def _paired_subject_bootstrap_delta(
    control: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    *,
    value: Any,
    subject_order: Sequence[str],
    bootstrap_schedule: Sequence[Sequence[int]],
    metric: str,
) -> dict[str, Any]:
    if set(control) != set(candidate):
        raise ValueError("paired bootstrap requires identical anchors")
    control_values = _subject_metric_values(
        control, value=value, subject_order=subject_order, metric=metric
    )
    candidate_values = _subject_metric_values(
        candidate, value=value, subject_order=subject_order, metric=metric
    )
    subject_delta = [
        candidate_value - control_value
        for control_value, candidate_value in zip(
            control_values, candidate_values, strict=True
        )
    ]
    observed = sum(subject_delta) / len(subject_delta)
    draws = sorted(
        sum(subject_delta[index] for index in indices) / len(indices)
        for indices in bootstrap_schedule
    )
    return {
        "metric": metric,
        "anchors": len(control),
        "subjects": len(subject_delta),
        "estimand": "candidate_minus_control_subject_macro_mean",
        "observed_delta": observed,
        "ci95": {
            "lower": _linear_percentile(draws, 0.025),
            "upper": _linear_percentile(draws, 0.975),
        },
    }


def _paired_subject_bootstrap_ratio(
    control: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    *,
    value: Any,
    subject_order: Sequence[str],
    bootstrap_schedule: Sequence[Sequence[int]],
    metric: str,
) -> dict[str, Any]:
    if set(control) != set(candidate):
        raise ValueError("paired bootstrap requires identical anchors")
    control_values = _subject_metric_values(
        control, value=value, subject_order=subject_order, metric=metric
    )
    candidate_values = _subject_metric_values(
        candidate, value=value, subject_order=subject_order, metric=metric
    )
    control_mean = sum(control_values) / len(control_values)
    candidate_mean = sum(candidate_values) / len(candidate_values)
    if control_mean <= 0.0 or candidate_mean < 0.0:
        raise ValueError(f"nonpositive ratio denominator or negative score for {metric}")
    observed = candidate_mean / control_mean
    draws: list[float] = []
    for indices in bootstrap_schedule:
        control_sum = sum(control_values[index] for index in indices)
        candidate_sum = sum(candidate_values[index] for index in indices)
        if control_sum <= 0.0 or candidate_sum < 0.0:
            raise ValueError(
                f"nonpositive bootstrap ratio denominator or negative score for {metric}"
            )
        draws.append(candidate_sum / control_sum)
    draws.sort()
    return {
        "metric": metric,
        "anchors": len(control),
        "subjects": len(control_values),
        "estimand": "candidate_subject_macro_mean_over_control_subject_macro_mean",
        "control_subject_macro_mean": control_mean,
        "candidate_subject_macro_mean": candidate_mean,
        "observed_ratio": observed,
        "ci95": {
            "lower": _linear_percentile(draws, 0.025),
            "upper": _linear_percentile(draws, 0.975),
        },
    }


def _paired_subject_bootstrap(
    control: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    *,
    value: Any,
    repetitions: int,
    seed: int,
    metric: str,
    allow_missing: bool = False,
) -> dict[str, Any]:
    if set(control) != set(candidate):
        raise ValueError("paired bootstrap requires identical anchors")
    by_subject: dict[str, list[float]] = defaultdict(list)
    used_anchors = 0
    for sample_id in sorted(control):
        control_row = control[sample_id]
        candidate_row = candidate[sample_id]
        if str(control_row["subject_id"]) != str(candidate_row["subject_id"]):
            raise ValueError(f"paired subject changed for {sample_id}")
        try:
            delta = float(value(candidate_row)) - float(value(control_row))
        except (KeyError, TypeError):
            if allow_missing:
                continue
            raise
        if not math.isfinite(delta):
            raise ValueError(f"non-finite paired delta for {metric}")
        by_subject[str(control_row["subject_id"])].append(delta)
        used_anchors += 1
    subject_delta = [
        sum(values) / len(values) for _, values in sorted(by_subject.items()) if values
    ]
    if not subject_delta:
        raise ValueError("no paired subjects with metric support")
    observed = sum(subject_delta) / len(subject_delta)
    rng = random.Random(seed)
    count = len(subject_delta)
    draws = sorted(
        sum(subject_delta[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(repetitions)
    )
    return {
        "metric": metric,
        "anchors": used_anchors,
        "subjects": count,
        "observed_delta": observed,
        "ci95": {
            "lower": _linear_percentile(draws, 0.025),
            "upper": _linear_percentile(draws, 0.975),
        },
    }


def _add_log_score_units(result: dict[str, Any]) -> None:
    observed = float(result["observed_delta"])
    lower = float(result["ci95"]["lower"])
    upper = float(result["ci95"]["upper"])
    result["observed_delta_nats_per_block"] = observed / 6.0
    result["observed_delta_bits_per_block"] = observed / (6.0 * math.log(2.0))
    result["ci95_nats_per_block"] = {"lower": lower / 6.0, "upper": upper / 6.0}
    result["ci95_bits_per_block"] = {
        "lower": lower / (6.0 * math.log(2.0)),
        "upper": upper / (6.0 * math.log(2.0)),
    }


def _add_log_density_ratios(result: dict[str, Any]) -> None:
    observed = float(result["observed_delta"])
    lower = float(result["ci95"]["lower"])
    upper = float(result["ci95"]["upper"])
    result["joint_geometric_density_ratio_candidate_over_control"] = math.exp(
        -observed
    )
    result["joint_geometric_density_ratio_ci95"] = {
        "lower": math.exp(-upper),
        "upper": math.exp(-lower),
    }
    result["per_factor_geometric_density_ratio_candidate_over_control"] = math.exp(
        -observed / 414.0
    )
    result["per_factor_geometric_density_ratio_ci95"] = {
        "lower": math.exp(-upper / 414.0),
        "upper": math.exp(-lower / 414.0),
    }


def _coherence_gate(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    coherent = sum(int(row["coherent_trajectories"]) for row in rows.values())
    total = sum(int(row["trajectories"]) for row in rows.values())
    return {
        "coherent_trajectories": coherent,
        "total_trajectories": total,
        "rate": coherent / total,
        "passed": coherent == total,
    }


def _assert_matched_identity(groups: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> None:
    modes = tuple(groups)
    anchor_ids = set(groups[modes[0]])
    for mode in modes[1:]:
        if set(groups[mode]) != anchor_ids:
            raise ValueError("matched V2 files do not contain identical anchors")
    for sample_id in anchor_ids:
        subjects = {str(groups[mode][sample_id]["subject_id"]) for mode in modes}
        if len(subjects) != 1:
            raise ValueError(f"subject identity differs across modes for {sample_id}")


def _assert_teacher_free_anchor_identity(
    teacher: Mapping[str, Mapping[str, Mapping[str, Any]]],
    free: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    for mode in teacher:
        teacher_sample_ids = set(teacher[mode])
        if teacher_sample_ids != set(free[mode]):
            raise ValueError(f"teacher/free-running anchors differ for {mode}")
        for sample_id in teacher_sample_ids:
            teacher_subject = str(teacher[mode][sample_id].get("subject_id") or "")
            free_subject = str(free[mode][sample_id].get("subject_id") or "")
            if not teacher_subject or teacher_subject != free_subject:
                raise ValueError(
                    "teacher/free-running subject identity differs for "
                    f"{mode}:{sample_id}"
                )


def _assert_teacher_free_contract_identity(
    teacher: Mapping[str, Mapping[str, Mapping[str, Any]]],
    free: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    keys = (
        "dataset_id",
        "contract_bundle_hash",
        "process_contract_sha256",
        "emission_contract_sha256",
        "projection_contract_sha256",
        "relation_contract_sha256",
        "sidecar_schema_sha256",
        "lab_scale_artifact_sha256",
        "standardized_primitive_scale_sha256",
        "input_normalization_sha256",
        "promotion_metric_contract_sha256",
        "semantic_runtime_identity_sha256",
        "source_tree_sha256",
        "source_identity_sha256",
        "git_commit",
        "git_head_tree",
        "matched_design_signature",
        "selected_checkpoint_step",
        "selected_checkpoint_model_sha256",
    )
    for mode in teacher:
        for sample_id in teacher[mode]:
            teacher_identity = teacher[mode][sample_id].get("identity")
            free_identity = free[mode][sample_id].get("identity")
            if not isinstance(teacher_identity, Mapping) or not isinstance(
                free_identity, Mapping
            ):
                raise ValueError("teacher/free-running row lacks contract identity")
            teacher_values = tuple(str(teacher_identity.get(key) or "") for key in keys)
            free_values = tuple(str(free_identity.get(key) or "") for key in keys)
            if any(not value for value in teacher_values) or teacher_values != free_values:
                raise ValueError(
                    f"teacher/free-running contract identity differs for {mode}:{sample_id}"
                )


def _assert_common_random_contract(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> None:
    identity_keys = (
        "dataset_id",
        "contract_bundle_hash",
        "process_contract_sha256",
        "emission_contract_sha256",
        "projection_contract_sha256",
        "relation_contract_sha256",
        "sidecar_schema_sha256",
        "lab_scale_artifact_sha256",
        "standardized_primitive_scale_sha256",
        "input_normalization_sha256",
        "promotion_metric_contract_sha256",
        "semantic_runtime_identity_sha256",
        "source_tree_sha256",
        "source_identity_sha256",
        "git_commit",
        "git_head_tree",
        "matched_design_signature",
        "crn_contract_sha256",
    )
    identities: set[tuple[str, ...]] = set()
    for rows in groups.values():
        for row in rows.values():
            identity = row.get("identity")
            if not isinstance(identity, Mapping):
                raise ValueError("free-running row lacks identity")
            values = tuple(str(identity.get(key) or "") for key in identity_keys)
            if any(not value for value in values):
                raise ValueError("free-running row has an incomplete contract identity")
            identities.add(values)
    if len(identities) != 1:
        raise ValueError(
            "matched free-running modes do not share data/contract/scale/CRN identity"
        )


def _index_rows(
    rows: Sequence[Mapping[str, Any]], expected_anchors: int
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in result:
            raise ValueError("evaluation rows contain an empty/duplicate sample_id")
        result[sample_id] = row
    if len(result) != expected_anchors:
        raise ValueError(f"promotion requires {expected_anchors} anchors, got {len(result)}")
    return result


def _read_free_running_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        manifest_path = path / "manifest.json"
    else:
        manifest_path = path
    if manifest_path.name.endswith(".jsonl"):
        return _read_jsonl(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = payload.get("per_anchor_score_shards")
    if not isinstance(shards, Sequence):
        raise ValueError("free-running manifest lacks per_anchor_score_shards")
    rows: list[dict[str, Any]] = []
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("free-running shard manifest row is invalid")
        shard_path = Path(str(shard["per_anchor_score_path"]))
        if not shard_path.is_absolute():
            shard_path = manifest_path.parent / shard_path
        if sha256_file(shard_path) != str(shard["per_anchor_score_sha256"]):
            raise ValueError(f"free-running score shard hash mismatch: {shard_path}")
        rows.extend(_read_jsonl(shard_path))
    return rows


def _scalar_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    by_subject: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        by_subject[str(row["subject_id"])].append(value)
    subject_means = [sum(items) / len(items) for items in by_subject.values()]
    return {
        "anchor_mean": sum(values) / len(values),
        "subject_macro": sum(subject_means) / len(subject_means),
    }


def _mapping_summary(
    rows: Sequence[Mapping[str, Any]], path: Sequence[str]
) -> dict[str, Any]:
    keys = _nested_metric_keys(
        {str(index): row for index, row in enumerate(rows)}, path
    )
    result: dict[str, Any] = {}
    for key in sorted(keys):
        supported: list[tuple[str, float]] = []
        for row in rows:
            value: Any = row
            for part in path:
                value = value[part]
            if key in value:
                supported.append((str(row["subject_id"]), float(value[key])))
        by_subject: dict[str, list[float]] = defaultdict(list)
        for subject_id, value in supported:
            by_subject[subject_id].append(value)
        subject_means = [sum(items) / len(items) for items in by_subject.values()]
        result[key] = {
            "anchors": len(supported),
            "subjects": len(subject_means),
            "anchor_mean": sum(value for _, value in supported) / len(supported),
            "subject_macro": sum(subject_means) / len(subject_means),
        }
    return result


def _empty_coverage_state() -> dict[str, Any]:
    return {"projections": {}}


def _update_coverage_state(
    state: dict[str, Any], coverage: Mapping[str, Any]
) -> None:
    projections = state["projections"]
    if not isinstance(projections, dict):
        raise ValueError("coverage accumulator projections must be mutable")
    for projection_id, item in coverage.items():
        if not isinstance(item, Mapping):
            raise ValueError("physical conditional coverage row is invalid")
        counts = item.get("generated_active_counts")
        if not isinstance(counts, Sequence):
            raise ValueError("physical conditional generated counts are invalid")
        numeric_counts = [int(value) for value in counts]
        target = projections.setdefault(
            str(projection_id),
            {
                "anchors": 0,
                "truth_active_blocks": 0,
                "scored_blocks": 0,
                "generated_count_events": 0,
                "generated_count_sum": 0,
                "generated_count_min": None,
            },
        )
        target["anchors"] += 1
        target["truth_active_blocks"] += int(item["truth_active_blocks"])
        target["scored_blocks"] += int(item["scored_blocks"])
        target["generated_count_events"] += len(numeric_counts)
        target["generated_count_sum"] += sum(numeric_counts)
        if numeric_counts:
            minimum = min(numeric_counts)
            current = target["generated_count_min"]
            target["generated_count_min"] = (
                minimum if current is None else min(int(current), minimum)
            )


def _merge_coverage_states(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    merged = _empty_coverage_state()
    projections = merged["projections"]
    for state in states:
        source = state.get("projections")
        if not isinstance(source, Mapping):
            raise ValueError("coverage rank state lacks projections")
        for projection_id, item in source.items():
            if not isinstance(item, Mapping):
                raise ValueError("coverage rank projection row is invalid")
            target = projections.setdefault(
                str(projection_id),
                {
                    "anchors": 0,
                    "truth_active_blocks": 0,
                    "scored_blocks": 0,
                    "generated_count_events": 0,
                    "generated_count_sum": 0,
                    "generated_count_min": None,
                },
            )
            for key in (
                "anchors",
                "truth_active_blocks",
                "scored_blocks",
                "generated_count_events",
                "generated_count_sum",
            ):
                target[key] += int(item[key])
            minimum = item.get("generated_count_min")
            if minimum is not None:
                current = target["generated_count_min"]
                target["generated_count_min"] = (
                    int(minimum)
                    if current is None
                    else min(int(current), int(minimum))
                )
    result: dict[str, Any] = {}
    for projection_id, item in sorted(projections.items()):
        events = int(item["generated_count_events"])
        truth_active = int(item["truth_active_blocks"])
        scored = int(item["scored_blocks"])
        result[projection_id] = {
            "anchors": int(item["anchors"]),
            "truth_active_blocks": truth_active,
            "scored_blocks": scored,
            "complete": scored == truth_active,
            "generated_active_count_min": item["generated_count_min"],
            "generated_active_count_mean": (
                int(item["generated_count_sum"]) / events if events else None
            ),
        }
    return result


def _nested_metric_keys(
    rows: Mapping[str, Mapping[str, Any]], path: Sequence[str]
) -> set[str]:
    result: set[str] = set()
    for row in rows.values():
        value: Any = row
        for part in path:
            value = value[part]
        if not isinstance(value, Mapping):
            raise ValueError(f"metric path {path} is not a mapping")
        result.update(str(key) for key in value)
    return result


def _nested_metric_anchor_support(
    rows: Mapping[str, Mapping[str, Any]],
    path: Sequence[str],
    metric_key: str,
) -> set[str]:
    support: set[str] = set()
    for sample_id, row in rows.items():
        value: Any = row
        for part in path:
            value = value[part]
        if not isinstance(value, Mapping):
            raise ValueError(f"metric path {path} is not a mapping")
        if metric_key in value:
            support.add(sample_id)
    return support


def _masked_projection_json(values: torch.Tensor, masks: torch.Tensor) -> list[list[float | None]]:
    value_rows = values.detach().cpu().tolist()
    mask_rows = masks.detach().cpu().tolist()
    result: list[list[float | None]] = []
    for block_values, block_masks in zip(value_rows, mask_rows, strict=True):
        result.append(
            [
                float(value) if bool(mask) else None
                for value, mask in zip(
                    block_values,
                    block_masks,
                    strict=True,
                )
            ]
        )
    return result


def _bank4(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError("primitive export bank must be a tensor")
    if value.ndim == 3:
        value = value.unsqueeze(-1)
    if value.ndim != 4 or value.shape[1:3] != (6, 29):
        raise ValueError("primitive export bank must be [B,6,29,W]")
    return value


def _string_batch(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    raise ValueError("evaluation identity must be a string batch")


def _linear_percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires values")
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"invalid JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def _json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ) + "\n"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@contextmanager
def _atomic_text(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yield handle
    temporary.replace(path)


@contextmanager
def _atomic_gzip_text(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                yield text
    temporary.replace(path)


def _rank_world() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _gather_objects(value: Any) -> list[Any]:
    rank, world_size = _rank_world()
    if world_size == 1:
        return [value]
    result: list[Any] = [None] * world_size
    torch.distributed.all_gather_object(result, value)
    return result


def _collect_distributed_phase(
    stage: str,
    factory: Callable[[], Any],
) -> list[Any]:
    """Run rank-local work, then make every rank observe any local failure.

    This boundary prevents one rank from entering the next collective while a
    peer has already failed during file hashing or rank-zero assembly.
    """

    rank, world_size = _rank_world()
    if world_size == 1:
        return [factory()]
    try:
        value = factory()
        envelope: dict[str, Any] = {
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
    envelopes = _gather_objects(envelope)
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


def _broadcast_object(value: Any) -> Any:
    rank, world_size = _rank_world()
    if world_size == 1:
        return value
    payload = [value if rank == 0 else None]
    torch.distributed.broadcast_object_list(payload, src=0)
    return payload[0]


__all__ = [
    "CRN_SCHEDULE_SCHEMA",
    "FREE_RUNNING_SCHEMA",
    "PRODUCTION_CRN_SEED",
    "PRODUCTION_TRAJECTORIES_PER_ANCHOR",
    "common_random_seed",
    "evaluate_free_running_v2",
    "evaluate_multires_event_v2_promotion",
    "validate_rank_local_artifact_preflight",
    "verify_rank_local_artifact_preflight",
]
