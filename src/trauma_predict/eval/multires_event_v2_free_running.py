from __future__ import annotations

from contextlib import contextmanager, nullcontext
from collections import defaultdict
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from trauma_predict.data.multires_event_v2 import (
    MultiresEventV2Contract,
    MultiresEventV2RelationContract,
)
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
PRODUCTION_CHUNK_TARGET_ANCHORS = 100
FREE_RUNNING_CHUNK_SCHEMA = (
    "trauma_predict.multires_event_v2_free_running_atomic_chunk.v1"
)
FREE_RUNNING_CHUNK_STATS_SCHEMA = (
    "trauma_predict.multires_event_v2_free_running_chunk_sufficient_stats.v1"
)
FREE_RUNNING_RESUME_SCHEMA = (
    "trauma_predict.multires_event_v2_free_running_resume_status.v1"
)
FREE_RUNNING_HOSTED_PROGRESS_SCHEMA = (
    "trauma_predict.multires_event_v2_free_running_hosted_progress.v1"
)
RANK_ARTIFACT_PREFLIGHT_SCHEMA = (
    "trauma_predict.multires_event_v2_rank_artifact_preflight.v1"
)


def evaluate_free_running_v2(
    *,
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    contract: MultiresEventV2Contract,
    relation_contract: MultiresEventV2RelationContract,
    device: torch.device,
    expected_samples: int,
    step: int,
    output_dir: str | Path,
    expected_lab_scale_artifact_hash: str,
    standardized_primitive_scale_path: str | Path,
    expected_standardized_primitive_scale_hash: str,
    input_normalization_sha256: str,
    trajectory_metric_contract: Mapping[str, Any],
    evaluation_identity: Mapping[str, Any] | None = None,
    trajectories_per_anchor: int = PRODUCTION_TRAJECTORIES_PER_ANCHOR,
    trajectory_batch_size: int | None = None,
    crn_seed: int = PRODUCTION_CRN_SEED,
    metrics_path: Path | None = None,
    precision: str = "fp16",
    chunk_target_anchors: int = PRODUCTION_CHUNK_TARGET_ANCHORS,
    max_new_anchors: int | None = None,
) -> dict[str, Any]:
    """Evaluate generated six-block trajectories without future truth feedback.

    The model input is encoded once per anchor. The cached memory/query state is
    expanded to the ensemble dimension and decoded autoregressively with the
    registry sampler. Production configuration freezes 100 trajectories; the
    function accepts smaller positive counts for unit/integration tests. Atomic
    chunks commit only after complete loader batches; ``max_new_anchors`` may
    return an ``INCOMPLETE`` status after the next durable chunk boundary.
    """

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
    if chunk_target_anchors < 1:
        raise ValueError("free-running chunk target must be positive")
    if max_new_anchors is not None and max_new_anchors < 1:
        raise ValueError("free-running per-invocation anchor limit must be positive")
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
        "target_sidecar_relation_contract_sha256": contract.contract_hashes["relation"],
        "relation_contract_version": relation_contract.version,
        "relation_contract_sha256": relation_contract.bundle_hash,
        "relation_contract_file_sha256": dict(relation_contract.file_hashes),
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
            "run_contract_signature",
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
        if int(evaluation_identity["selected_checkpoint_step"]) != int(step):
            raise ValueError(
                "free-running selected checkpoint identity differs from evaluation step"
            )
        for key, value in evaluation_identity.items():
            if key in identity and str(identity[key]) != str(value):
                raise ValueError(
                    f"free-running evaluation identity differs for {key}: "
                    f"{identity[key]!r} != {value!r}"
                )
            identity[key] = value
    schema_path = output_root / "sample_schema.json"
    schema_contract = {
        "schema_version": "trauma_predict.multires_event_v2_sample_export.v2",
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
        "free-running chunk-root validation",
        lambda: _validate_chunk_root_layout(output_root, world_size)
        if rank == 0
        else None,
    )
    schema_results = _collect_distributed_phase(
        "free-running sample-schema materialization",
        lambda: _materialize_or_validate_sample_schema(schema_path, schema_contract)
        if rank == 0
        else None,
    )
    schema_info = schema_results[0]
    if not isinstance(schema_info, Mapping):
        raise RuntimeError("free-running sample schema identity was not assembled")
    sample_schema_sha256 = str(schema_info["file_sha256"])
    sample_schema_identity_sha256 = str(schema_info["identity_sha256"])

    loaded_chunks = _collect_distributed_phase(
        "free-running atomic-chunk validation",
        lambda: _load_completed_free_running_chunks(
            output_root=output_root,
            rank=rank,
            world_size=world_size,
            step=step,
            trajectories_per_anchor=trajectories_per_anchor,
            chunk_target_anchors=chunk_target_anchors,
            identity=identity,
            crn_contract=crn_contract,
            sample_schema_sha256=sample_schema_sha256,
            sample_schema_identity_sha256=sample_schema_identity_sha256,
        ),
    )
    completed_chunks = list(loaded_chunks[rank])
    completed_loader_batches = [
        [str(sample_id) for sample_id in batch_ids]
        for chunk in completed_chunks
        for batch_ids in chunk["loader_batch_sample_ids"]
    ]
    completed_sample_ids = [
        str(sample_id)
        for chunk in completed_chunks
        for sample_id in chunk["sample_ids"]
    ]
    completed_terminal_partial = bool(
        completed_chunks
        and int(completed_chunks[-1]["anchors"]) < chunk_target_anchors
    )

    local_anchor_count = len(completed_sample_ids)
    local_sample_ids: set[str] = set()
    completed_batch_cursor = 0
    chunk_anchor_rows: list[dict[str, Any]] = []
    chunk_audit_rows: list[dict[str, Any]] = []
    chunk_stats_rows: list[dict[str, Any]] = []
    chunk_loader_batches: list[list[str]] = []
    new_anchor_count = 0
    stopped_for_limit = False
    autocast = _autocast_factory(device, precision)
    _emit_rank_progress(
        path=rank_progress_path,
        rank=rank,
        completed_anchors=local_anchor_count,
        started_at=rank_started_at,
    )
    with torch.no_grad():
        for raw_batch in loader:
            sample_ids = _string_batch(raw_batch.get("sample_id"))
            subject_ids = _string_batch(raw_batch.get("subject_id"))
            if not sample_ids or len(sample_ids) != len(subject_ids):
                raise ValueError("free-running identities do not align within the batch")
            duplicates = local_sample_ids.intersection(sample_ids)
            if duplicates or len(sample_ids) != len(set(sample_ids)):
                within_batch = {
                    item for item in sample_ids if sample_ids.count(item) > 1
                }
                duplicate = sorted(duplicates or within_batch)[0]
                raise RuntimeError(f"duplicate local free-running anchor {duplicate}")
            local_sample_ids.update(sample_ids)

            if completed_batch_cursor < len(completed_loader_batches):
                expected_batch_ids = completed_loader_batches[completed_batch_cursor]
                if sample_ids != expected_batch_ids:
                    raise RuntimeError(
                        "free-running resume loader order or batch boundary changed before "
                        f"completed chunk {completed_batch_cursor}: {sample_ids[:3]} != "
                        f"{expected_batch_ids[:3]}"
                    )
                completed_batch_cursor += 1
                # A completed batch is rejected before device transfer and before
                # encode_for_rollout, so resume never repeats model computation.
                continue
            if completed_terminal_partial:
                raise RuntimeError(
                    "free-running resume found anchors after a committed terminal "
                    "partial chunk"
                )

            batch = move_to_device(raw_batch, device)
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
                    retained_audit_row: dict[str, Any] | None = None
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
                            retained_audit_row = export
                        trajectory_start += count

                    if retained_audit_row is None:
                        raise AssertionError("free-running audit trajectory was not retained")

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
                    calibration_rows = physical_scores.pop("branch_calibration_rows")
                    coverage_by_projection = physical_scores.pop(
                        "coverage_by_projection"
                    )
                    trajectory_scores = score_standardized_primitive_ensemble(
                        ensemble_phi,
                        truth_phi[0].detach().cpu(),
                        primitive_schema,
                        relation_contract.target_edges,
                        trajectory_metric_contract,
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
                        "model_contract": "relation_v2",
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
                    chunk_anchor_rows.append(anchor_row)
                    chunk_audit_rows.append(retained_audit_row)
                    chunk_stats_rows.append(
                        {
                            "schema_version": FREE_RUNNING_CHUNK_STATS_SCHEMA,
                            "sample_id": sample_id,
                            "branch_calibration_rows": calibration_rows,
                            "coverage_by_projection": coverage_by_projection,
                        }
                    )
                    new_anchor_count += 1
                    local_anchor_count += 1

            # Chunks end only at an original loader-batch boundary.  This makes
            # resumed and uninterrupted encoding use identical batch shapes.
            chunk_loader_batches.append(list(sample_ids))
            if len(chunk_anchor_rows) >= chunk_target_anchors:
                completed_chunks.append(
                    _commit_free_running_chunk(
                        output_root=output_root,
                        rank=rank,
                        world_size=world_size,
                        chunk_index=len(completed_chunks),
                        step=step,
                        trajectories_per_anchor=trajectories_per_anchor,
                        chunk_target_anchors=chunk_target_anchors,
                        identity=identity,
                        crn_contract=crn_contract,
                        sample_schema_sha256=sample_schema_sha256,
                        sample_schema_identity_sha256=sample_schema_identity_sha256,
                        score_rows=chunk_anchor_rows,
                        audit_rows=chunk_audit_rows,
                        stats_rows=chunk_stats_rows,
                        loader_batch_sample_ids=chunk_loader_batches,
                    )
                )
                chunk_anchor_rows = []
                chunk_audit_rows = []
                chunk_stats_rows = []
                chunk_loader_batches = []
                _emit_rank_progress(
                    path=rank_progress_path,
                    rank=rank,
                    completed_anchors=local_anchor_count,
                    started_at=rank_started_at,
                )
                if (
                    max_new_anchors is not None
                    and new_anchor_count >= max_new_anchors
                ):
                    stopped_for_limit = True
                    break

    if completed_batch_cursor != len(completed_loader_batches):
        raise RuntimeError(
            "free-running loader ended before all committed resume batches were seen"
        )
    if stopped_for_limit:
        if chunk_anchor_rows or chunk_audit_rows or chunk_stats_rows or chunk_loader_batches:
            raise AssertionError("free-running invocation stopped with an uncommitted chunk")
    elif chunk_anchor_rows:
        completed_chunks.append(
            _commit_free_running_chunk(
                output_root=output_root,
                rank=rank,
                world_size=world_size,
                chunk_index=len(completed_chunks),
                step=step,
                trajectories_per_anchor=trajectories_per_anchor,
                chunk_target_anchors=chunk_target_anchors,
                identity=identity,
                crn_contract=crn_contract,
                sample_schema_sha256=sample_schema_sha256,
                sample_schema_identity_sha256=sample_schema_identity_sha256,
                score_rows=chunk_anchor_rows,
                audit_rows=chunk_audit_rows,
                stats_rows=chunk_stats_rows,
                loader_batch_sample_ids=chunk_loader_batches,
            )
        )
        chunk_anchor_rows = []
        chunk_audit_rows = []
        chunk_stats_rows = []
        chunk_loader_batches = []

    _emit_rank_progress(
        path=rank_progress_path,
        rank=rank,
        completed_anchors=local_anchor_count,
        started_at=rank_started_at,
    )

    def build_rank_payload() -> dict[str, Any]:
        return {
            "rank": rank,
            "anchors": local_anchor_count,
            "new_anchors": new_anchor_count,
            "stopped_for_limit": stopped_for_limit,
            "chunks": completed_chunks,
            "sample_ids": [
                str(sample_id)
                for chunk in completed_chunks
                for sample_id in chunk["sample_ids"]
            ],
        }

    rank_payloads = _collect_distributed_phase(
        "free-running rank artifact finalization",
        build_rank_payload,
    )
    def assemble_rank_zero_result() -> dict[str, Any] | None:
        if rank != 0:
            return None
        ordered_rank_payloads = sorted(rank_payloads, key=lambda item: int(item["rank"]))
        ids = [
            str(sample_id)
            for payload in ordered_rank_payloads
            for sample_id in payload["sample_ids"]
        ]
        if len(ids) != len(set(ids)):
            raise RuntimeError("free-running sampler introduced duplicate persisted anchors")
        if len(ids) > expected_samples:
            raise RuntimeError(
                f"free-running evaluation expected {expected_samples} anchors, got {len(ids)}"
            )
        if len(ids) < expected_samples:
            if not any(bool(payload["stopped_for_limit"]) for payload in ordered_rank_payloads):
                raise RuntimeError(
                    f"free-running evaluation expected {expected_samples} anchors, got {len(ids)}"
                )
            partial = {
                "schema_version": FREE_RUNNING_RESUME_SCHEMA,
                "status": "INCOMPLETE",
                "updated_at": utc_now(),
                "model_contract": "relation_v2",
                "step": int(step),
                "anchors": len(ids),
                "expected_anchors": expected_samples,
                "new_anchors_this_invocation": sum(
                    int(payload["new_anchors"]) for payload in ordered_rank_payloads
                ),
                "trajectories_per_anchor": trajectories_per_anchor,
                "chunk_target_anchors": chunk_target_anchors,
                "identity": identity,
                "crn_contract": crn_contract,
                "sample_schema_path": schema_path.name,
                "sample_schema_sha256": sample_schema_sha256,
                "sample_schema_identity_sha256": sample_schema_identity_sha256,
                "ranks": ordered_rank_payloads,
            }
            _atomic_json(output_root / "resume_status.json", partial)
            _write_free_running_hosted_progress(
                output_root=output_root,
                status="INCOMPLETE",
                completed_anchors=len(ids),
                expected_anchors=expected_samples,
                new_anchors=sum(
                    int(payload["new_anchors"])
                    for payload in ordered_rank_payloads
                ),
                identity=identity,
                rank_payloads=ordered_rank_payloads,
            )
            return partial

        rows: list[dict[str, Any]] = []
        manifests: list[dict[str, Any]] = []
        calibration_parts: list[dict[str, Any]] = []
        coverage_parts: list[dict[str, Any]] = []
        for payload in ordered_rank_payloads:
            merged = _merge_free_running_rank_chunks(
                output_root=output_root,
                rank_payload=payload,
            )
            manifests.append(merged["manifest"])
            calibration_parts.append(merged["calibration"])
            coverage_parts.append(merged["coverage"])
            rows.extend(merged["rows"])
        result = _summarize_free_running_rows(
            rows,
            step=step,
            trajectories=trajectories_per_anchor,
            identity=identity,
            crn_contract=crn_contract,
            calibration=_merge_calibration_states(calibration_parts),
            coverage=_merge_coverage_states(coverage_parts),
        )
        result["sample_schema_path"] = schema_path.name
        result["sample_schema_sha256"] = sample_schema_sha256
        result["sample_schema_identity_sha256"] = sample_schema_identity_sha256
        result["shards"] = manifests
        result["atomic_chunks"] = [
            chunk
            for payload in ordered_rank_payloads
            for chunk in payload["chunks"]
        ]
        manifest_path = output_root / "manifest.json"
        _atomic_json(
            manifest_path,
            {
                "schema_version": "trauma_predict.multires_event_v2_free_running_manifest.v2",
                "created_at": utc_now(),
                "evaluation": result,
                "per_anchor_score_shards": manifests,
                "atomic_chunks": result["atomic_chunks"],
            },
        )
        result["manifest_path"] = manifest_path.name
        result["manifest_sha256"] = sha256_file(manifest_path)
        _atomic_json(output_root / "evaluation.json", result)
        _atomic_json(
            output_root / "resume_status.json",
            {
                "schema_version": FREE_RUNNING_RESUME_SCHEMA,
                "status": "COMPLETE",
                "updated_at": utc_now(),
                "anchors": len(rows),
                "expected_anchors": expected_samples,
                "step": int(step),
                "identity": identity,
                "manifest_path": manifest_path.name,
                "manifest_sha256": result["manifest_sha256"],
            },
        )
        _write_free_running_hosted_progress(
            output_root=output_root,
            status="COMPLETE",
            completed_anchors=len(rows),
            expected_anchors=expected_samples,
            new_anchors=sum(
                int(payload["new_anchors"]) for payload in ordered_rank_payloads
            ),
            identity=identity,
            rank_payloads=ordered_rank_payloads,
        )
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


def _validate_chunk_root_layout(output_root: Path, world_size: int) -> None:
    chunk_root = output_root / "chunks"
    if not chunk_root.exists():
        legacy = (
            list(output_root.glob("per_anchor_scores.rank*.jsonl"))
            + list(output_root.glob("audit_trajectory_samples.rank*.jsonl.gz"))
            + [
                path
                for path in (
                    output_root / "manifest.json",
                    output_root / "evaluation.json",
                    output_root / "hosted_progress.json",
                )
                if path.exists()
            ]
        )
        if legacy:
            raise RuntimeError(
                "free-running output contains legacy formal artifacts without atomic chunks"
            )
        return
    if chunk_root.is_symlink() or not chunk_root.is_dir():
        raise RuntimeError("free-running chunk root is linked or not a directory")
    formal_rank_roots: list[Path] = []
    for entry in chunk_root.iterdir():
        if ".tmp" in entry.name:
            continue
        match = re.fullmatch(r"rank([0-9]{5})", entry.name)
        if match is None or entry.is_symlink() or not entry.is_dir():
            raise RuntimeError(f"invalid formal free-running rank chunk path: {entry}")
        persisted_rank = int(match.group(1))
        if persisted_rank < 0 or persisted_rank >= world_size:
            raise RuntimeError(
                "free-running chunks were created under a different distributed world"
            )
        formal_rank_roots.append(entry)

    actual_manifest_paths = {
        str(path.relative_to(output_root))
        for rank_root in formal_rank_roots
        for path in rank_root.glob("chunk[0-9][0-9][0-9][0-9][0-9][0-9]/manifest.json")
        if ".tmp" not in str(path.relative_to(output_root))
    }
    hosted_progress_path = output_root / "hosted_progress.json"
    if hosted_progress_path.exists():
        if hosted_progress_path.is_symlink() or not hosted_progress_path.is_file():
            raise RuntimeError("free-running hosted progress is linked or invalid")
        hosted = json.loads(hosted_progress_path.read_text(encoding="utf-8"))
        hosted_chunks = hosted.get("chunk_manifests") if isinstance(hosted, dict) else None
        hosted_identity = hosted.get("identity") if isinstance(hosted, dict) else None
        hosted_status = hosted.get("status") if isinstance(hosted, dict) else None
        hosted_completed = int(hosted.get("completed", -1)) if isinstance(hosted, dict) else -1
        hosted_expected = int(hosted.get("expected", -1)) if isinstance(hosted, dict) else -1
        hosted_chunk_anchors = (
            sum(
                int(chunk.get("anchors", -1))
                for chunk in hosted_chunks
                if isinstance(chunk, Mapping)
            )
            if isinstance(hosted_chunks, list)
            else -1
        )
        if (
            not isinstance(hosted, dict)
            or hosted.get("schema_version") != FREE_RUNNING_HOSTED_PROGRESS_SCHEMA
            or hosted_status not in {"INCOMPLETE", "COMPLETE"}
            or not isinstance(hosted_chunks, list)
            or not isinstance(hosted_identity, Mapping)
            or hosted.get("identity_sha256") != sha256_payload(hosted_identity)
            or hosted.get("completed_anchors") != hosted_completed
            or hosted.get("expected_anchors") != hosted_expected
            or hosted_completed < 0
            or hosted_expected < 1
            or hosted_completed > hosted_expected
            or (hosted_status == "COMPLETE" and hosted_completed != hosted_expected)
            or (hosted_status == "INCOMPLETE" and hosted_completed >= hosted_expected)
            or int(hosted.get("new_anchors", -1)) < 0
            or hosted_chunk_anchors != hosted_completed
            or hosted.get("set_sha256") != sha256_payload(hosted_chunks)
            or hosted.get("chunk_manifest_set_sha256") != sha256_payload(hosted_chunks)
        ):
            raise RuntimeError("free-running hosted progress contract failed")
        hosted_manifest_paths: set[str] = set()
        for chunk in hosted_chunks:
            if not isinstance(chunk, Mapping):
                raise RuntimeError("free-running hosted chunk pointer is invalid")
            relative = str(chunk.get("manifest_path") or "")
            path = output_root / relative
            if (
                not relative
                or path.is_symlink()
                or not path.is_file()
                or not path.resolve().is_relative_to(output_root)
                or sha256_file(path) != str(chunk.get("manifest_sha256") or "")
            ):
                raise RuntimeError("free-running hosted chunk manifest pointer/hash failed")
            hosted_manifest_paths.add(relative)
        # A process may die after atomically committing a new chunk but before
        # refreshing hosted_progress.json.  Such an extra complete chunk is
        # independently validated below; every previously declared chunk is fixed.
        if not hosted_manifest_paths.issubset(actual_manifest_paths):
            raise RuntimeError("free-running hosted chunk set is missing formal evidence")

    final_manifest_path = output_root / "manifest.json"
    final_evaluation_path = output_root / "evaluation.json"
    if final_manifest_path.exists() != final_evaluation_path.exists():
        raise RuntimeError("free-running final manifest/evaluation pair is incomplete")
    if not final_manifest_path.exists():
        return
    if (
        final_manifest_path.is_symlink()
        or not final_manifest_path.is_file()
        or final_evaluation_path.is_symlink()
        or not final_evaluation_path.is_file()
    ):
        raise RuntimeError("free-running final manifest/evaluation is linked or invalid")
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    chunks = final_manifest.get("atomic_chunks") if isinstance(final_manifest, dict) else None
    if (
        not isinstance(final_manifest, dict)
        or final_manifest.get("schema_version")
        != "trauma_predict.multires_event_v2_free_running_manifest.v2"
        or not isinstance(chunks, list)
    ):
        raise RuntimeError("free-running final manifest does not describe atomic chunks")
    declared_manifest_paths: set[str] = set()
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            raise RuntimeError("free-running final chunk pointer is invalid")
        relative = str(chunk.get("manifest_path") or "")
        path = output_root / relative
        if (
            not relative
            or path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(output_root)
            or sha256_file(path) != str(chunk.get("manifest_sha256") or "")
        ):
            raise RuntimeError("free-running final chunk manifest pointer/hash failed")
        declared_manifest_paths.add(relative)
    if declared_manifest_paths != actual_manifest_paths:
        raise RuntimeError("free-running final manifest chunk set is incomplete")


def _materialize_or_validate_sample_schema(
    path: Path,
    schema_contract: Mapping[str, Any],
) -> dict[str, str]:
    identity_sha256 = sha256_payload(schema_contract)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("free-running sample schema is linked or not a file")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("free-running sample schema is not a mapping")
        created_at = payload.pop("created_at", None)
        if not isinstance(created_at, str) or not created_at:
            raise RuntimeError("free-running sample schema lacks its creation identity")
        if sha256_payload(payload) != identity_sha256:
            raise RuntimeError(
                "free-running sample schema differs from the resumed evaluation contract"
            )
    else:
        _atomic_json(path, {**dict(schema_contract), "created_at": utc_now()})
    return {
        "file_sha256": sha256_file(path),
        "identity_sha256": identity_sha256,
    }


def _commit_free_running_chunk(
    *,
    output_root: Path,
    rank: int,
    world_size: int,
    chunk_index: int,
    step: int,
    trajectories_per_anchor: int,
    chunk_target_anchors: int,
    identity: Mapping[str, Any],
    crn_contract: Mapping[str, Any],
    sample_schema_sha256: str,
    sample_schema_identity_sha256: str,
    score_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    stats_rows: Sequence[Mapping[str, Any]],
    loader_batch_sample_ids: Sequence[Sequence[str]],
) -> dict[str, Any]:
    anchors = len(score_rows)
    if anchors < 1 or len(audit_rows) != anchors or len(stats_rows) != anchors:
        raise ValueError("free-running chunk rows do not align")
    score_ids = [str(row.get("sample_id")) for row in score_rows]
    audit_ids = [str(row.get("sample_id")) for row in audit_rows]
    stats_ids = [str(row.get("sample_id")) for row in stats_rows]
    loader_ids = [str(item) for batch in loader_batch_sample_ids for item in batch]
    if (
        score_ids != audit_ids
        or score_ids != stats_ids
        or score_ids != loader_ids
        or len(score_ids) != len(set(score_ids))
    ):
        raise ValueError("free-running chunk sample identities do not align")

    rank_root = output_root / "chunks" / f"rank{rank:05d}"
    rank_root.mkdir(parents=True, exist_ok=True)
    chunk_name = f"chunk{chunk_index:06d}"
    final_path = rank_root / chunk_name
    if final_path.exists() or final_path.is_symlink():
        raise RuntimeError(f"refusing to replace formal free-running chunk {final_path}")
    temporary = rank_root / (
        f"{chunk_name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    temporary.mkdir()
    score_path = temporary / "scores.jsonl"
    audit_path = temporary / "audit_trajectories.jsonl.gz"
    stats_path = temporary / "calibration_coverage_stats.jsonl.gz"
    with _atomic_text(score_path) as handle:
        for row in score_rows:
            handle.write(_json_line(row))
    with _atomic_gzip_text(audit_path) as handle:
        for row in audit_rows:
            handle.write(_json_line(row))
    with _atomic_gzip_text(stats_path) as handle:
        for row in stats_rows:
            handle.write(_json_line(row))

    def file_row(path: Path, rows: int) -> dict[str, Any]:
        return {
            "path": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "rows": rows,
        }

    identity_payload = dict(identity)
    source_model_run_identity = {
        key: identity_payload.get(key)
        for key in (
            "source_tree_sha256",
            "source_identity_sha256",
            "git_commit",
            "git_head_tree",
            "run_contract_signature",
            "selected_checkpoint_step",
            "selected_checkpoint_model_sha256",
        )
    }
    manifest = {
        "schema_version": FREE_RUNNING_CHUNK_SCHEMA,
        "status": "COMPLETE",
        "created_at": utc_now(),
        "model_contract": "relation_v2",
        "rank": int(rank),
        "world_size": int(world_size),
        "chunk_index": int(chunk_index),
        "chunk_target_anchors": int(chunk_target_anchors),
        "commit_boundary": "complete_loader_batch_at_or_above_target",
        "anchors": anchors,
        "sample_ids": score_ids,
        "loader_batch_sample_ids": [
            [str(item) for item in batch] for batch in loader_batch_sample_ids
        ],
        "step": int(step),
        "trajectories_per_anchor": int(trajectories_per_anchor),
        "identity": identity_payload,
        "identity_sha256": sha256_payload(identity_payload),
        "source_model_run_identity": source_model_run_identity,
        "crn_contract": dict(crn_contract),
        "crn_contract_sha256": sha256_payload(crn_contract),
        "sample_schema_sha256": sample_schema_sha256,
        "sample_schema_identity_sha256": sample_schema_identity_sha256,
        "files": {
            "scores": file_row(score_path, anchors),
            "audit_trajectories": file_row(audit_path, anchors),
            "calibration_coverage_sufficient_stats": file_row(stats_path, anchors),
        },
    }
    _atomic_json(temporary / "manifest.json", manifest)
    temporary.replace(final_path)
    return _free_running_chunk_descriptor(output_root, final_path, manifest)


def _free_running_chunk_descriptor(
    output_root: Path,
    chunk_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = chunk_path / "manifest.json"
    return {
        "rank": int(manifest["rank"]),
        "chunk_index": int(manifest["chunk_index"]),
        "anchors": int(manifest["anchors"]),
        "sample_ids": [str(item) for item in manifest["sample_ids"]],
        "loader_batch_sample_ids": [
            [str(item) for item in batch]
            for batch in manifest["loader_batch_sample_ids"]
        ],
        "manifest_path": str(manifest_path.relative_to(output_root)),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _load_completed_free_running_chunks(
    *,
    output_root: Path,
    rank: int,
    world_size: int,
    step: int,
    trajectories_per_anchor: int,
    chunk_target_anchors: int,
    identity: Mapping[str, Any],
    crn_contract: Mapping[str, Any],
    sample_schema_sha256: str,
    sample_schema_identity_sha256: str,
) -> list[dict[str, Any]]:
    rank_root = output_root / "chunks" / f"rank{rank:05d}"
    rank_root.mkdir(parents=True, exist_ok=True)
    formal_paths: list[Path] = []
    for entry in rank_root.iterdir():
        if ".tmp" in entry.name:
            # Interrupted temporary directories are never adopted as evidence.
            continue
        if re.fullmatch(r"chunk[0-9]{6}", entry.name) is None:
            raise RuntimeError(f"invalid formal free-running chunk path: {entry}")
        if entry.is_symlink() or not entry.is_dir():
            raise RuntimeError(f"formal free-running chunk is linked or not a directory: {entry}")
        formal_paths.append(entry)
    formal_paths.sort(key=lambda path: path.name)
    descriptors: list[dict[str, Any]] = []
    for expected_index, path in enumerate(formal_paths):
        if path.name != f"chunk{expected_index:06d}":
            raise RuntimeError("free-running formal chunks are not contiguous")
        descriptors.append(
            _validate_free_running_chunk(
                output_root=output_root,
                chunk_path=path,
                rank=rank,
                world_size=world_size,
                chunk_index=expected_index,
                step=step,
                trajectories_per_anchor=trajectories_per_anchor,
                chunk_target_anchors=chunk_target_anchors,
                identity=identity,
                crn_contract=crn_contract,
                sample_schema_sha256=sample_schema_sha256,
                sample_schema_identity_sha256=sample_schema_identity_sha256,
            )
        )
    for descriptor in descriptors[:-1]:
        if int(descriptor["anchors"]) < chunk_target_anchors:
            raise RuntimeError("only the terminal free-running chunk may be below target size")
    sample_ids = [
        sample_id for descriptor in descriptors for sample_id in descriptor["sample_ids"]
    ]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("free-running formal chunks contain duplicate sample IDs")
    return descriptors


def _validate_free_running_chunk(
    *,
    output_root: Path,
    chunk_path: Path,
    rank: int,
    world_size: int,
    chunk_index: int,
    step: int,
    trajectories_per_anchor: int,
    chunk_target_anchors: int,
    identity: Mapping[str, Any],
    crn_contract: Mapping[str, Any],
    sample_schema_sha256: str,
    sample_schema_identity_sha256: str,
) -> dict[str, Any]:
    manifest_path = chunk_path / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError(f"formal free-running chunk lacks its manifest: {chunk_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("free-running chunk manifest is not a mapping")
    if (
        manifest.get("schema_version") != FREE_RUNNING_CHUNK_SCHEMA
        or manifest.get("status") != "COMPLETE"
        or manifest.get("model_contract") != "relation_v2"
        or int(manifest.get("rank", -1)) != rank
        or int(manifest.get("world_size", -1)) != world_size
        or int(manifest.get("chunk_index", -1)) != chunk_index
        or int(manifest.get("step", -1)) != step
        or int(manifest.get("trajectories_per_anchor", -1))
        != trajectories_per_anchor
        or int(manifest.get("chunk_target_anchors", -1)) != chunk_target_anchors
        or manifest.get("commit_boundary")
        != "complete_loader_batch_at_or_above_target"
    ):
        raise RuntimeError("free-running formal chunk contract identity changed")
    expected_identity_hash = sha256_payload(identity)
    expected_crn_hash = sha256_payload(crn_contract)
    expected_source_model_run_identity = {
        key: identity.get(key)
        for key in (
            "source_tree_sha256",
            "source_identity_sha256",
            "git_commit",
            "git_head_tree",
            "run_contract_signature",
            "selected_checkpoint_step",
            "selected_checkpoint_model_sha256",
        )
    }
    if (
        manifest.get("identity_sha256") != expected_identity_hash
        or sha256_payload(manifest.get("identity")) != expected_identity_hash
        or manifest.get("source_model_run_identity")
        != expected_source_model_run_identity
        or manifest.get("crn_contract_sha256") != expected_crn_hash
        or sha256_payload(manifest.get("crn_contract")) != expected_crn_hash
        or manifest.get("sample_schema_sha256") != sample_schema_sha256
        or manifest.get("sample_schema_identity_sha256")
        != sample_schema_identity_sha256
    ):
        raise RuntimeError("free-running formal chunk source/model/run/CRN/schema drift")

    sample_ids = manifest.get("sample_ids")
    loader_batches = manifest.get("loader_batch_sample_ids")
    files = manifest.get("files")
    anchors = int(manifest.get("anchors", -1))
    if (
        anchors < 1
        or not isinstance(sample_ids, list)
        or len(sample_ids) != anchors
        or len({str(item) for item in sample_ids}) != anchors
        or not isinstance(loader_batches, list)
        or [str(item) for batch in loader_batches for item in batch]
        != [str(item) for item in sample_ids]
        or not isinstance(files, Mapping)
        or set(files)
        != {
            "scores",
            "audit_trajectories",
            "calibration_coverage_sufficient_stats",
        }
    ):
        raise RuntimeError("free-running formal chunk sample/file contract failed")

    paths: dict[str, Path] = {}
    for key, expected_name in (
        ("scores", "scores.jsonl"),
        ("audit_trajectories", "audit_trajectories.jsonl.gz"),
        (
            "calibration_coverage_sufficient_stats",
            "calibration_coverage_stats.jsonl.gz",
        ),
    ):
        row = files[key]
        if not isinstance(row, Mapping) or row.get("path") != expected_name:
            raise RuntimeError("free-running chunk file pointer changed")
        path = chunk_path / expected_name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve().parent != chunk_path.resolve()
            or int(row.get("rows", -1)) != anchors
            or int(row.get("size_bytes", -1)) != path.stat().st_size
            or row.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"free-running chunk file/hash failed: {path}")
        paths[key] = path

    score_rows = _read_jsonl(paths["scores"])
    audit_rows = _read_gzip_jsonl(paths["audit_trajectories"])
    stats_rows = _read_gzip_jsonl(paths["calibration_coverage_sufficient_stats"])
    expected_ids = [str(item) for item in sample_ids]
    if any(len(rows) != anchors for rows in (score_rows, audit_rows, stats_rows)):
        raise RuntimeError("free-running chunk row counts differ from its manifest")
    for label, rows in (
        ("scores", score_rows),
        ("audit", audit_rows),
        ("statistics", stats_rows),
    ):
        if [str(row.get("sample_id")) for row in rows] != expected_ids:
            raise RuntimeError(f"free-running chunk {label} sample order changed")
    for row in score_rows:
        if (
            row.get("model_contract") != "relation_v2"
            or int(row.get("step", -1)) != step
            or int(row.get("trajectories", -1)) != trajectories_per_anchor
            or sha256_payload(row.get("identity")) != expected_identity_hash
        ):
            raise RuntimeError("free-running chunk score identity changed")
    for row in audit_rows:
        if int(row.get("trajectory_index", -1)) != 0:
            raise RuntimeError("free-running chunk audit retention changed")
    for row in stats_rows:
        if (
            row.get("schema_version") != FREE_RUNNING_CHUNK_STATS_SCHEMA
            or not isinstance(row.get("branch_calibration_rows"), list)
            or not isinstance(row.get("coverage_by_projection"), Mapping)
        ):
            raise RuntimeError("free-running chunk sufficient statistics changed")
    return _free_running_chunk_descriptor(output_root, chunk_path, manifest)


def _read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"invalid gzip JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def _merge_free_running_rank_chunks(
    *,
    output_root: Path,
    rank_payload: Mapping[str, Any],
) -> dict[str, Any]:
    rank = int(rank_payload["rank"])
    chunks = rank_payload.get("chunks")
    if not isinstance(chunks, list):
        raise RuntimeError("free-running rank payload lacks atomic chunks")
    primitive_path = output_root / f"audit_trajectory_samples.rank{rank:05d}.jsonl.gz"
    anchor_path = output_root / f"per_anchor_scores.rank{rank:05d}.jsonl"
    rows: list[dict[str, Any]] = []
    calibration = _empty_calibration_state()
    coverage = _empty_coverage_state()
    with _atomic_text(anchor_path) as anchor_handle, _atomic_gzip_text(
        primitive_path
    ) as primitive_handle:
        for descriptor in chunks:
            manifest_path = output_root / str(descriptor["manifest_path"])
            if (
                manifest_path.is_symlink()
                or not manifest_path.is_file()
                or sha256_file(manifest_path) != descriptor["manifest_sha256"]
            ):
                raise RuntimeError("free-running chunk manifest changed during final merge")
            chunk_path = manifest_path.parent
            chunk_rows = _read_jsonl(chunk_path / "scores.jsonl")
            audit_rows = _read_gzip_jsonl(chunk_path / "audit_trajectories.jsonl.gz")
            stats_rows = _read_gzip_jsonl(
                chunk_path / "calibration_coverage_stats.jsonl.gz"
            )
            for row in chunk_rows:
                anchor_handle.write(_json_line(row))
            for row in audit_rows:
                primitive_handle.write(_json_line(row))
            for row in stats_rows:
                _update_calibration_state(calibration, row["branch_calibration_rows"])
                _update_coverage_state(coverage, row["coverage_by_projection"])
            rows.extend(chunk_rows)
    expected_ids = [str(item) for item in rank_payload["sample_ids"]]
    if [str(row["sample_id"]) for row in rows] != expected_ids:
        raise RuntimeError("free-running final rank merge changed sample order")
    progress_path = output_root / f"progress.rank{rank:05d}.jsonl"
    if progress_path.is_symlink() or not progress_path.is_file():
        raise RuntimeError("free-running rank progress artifact is missing")
    manifest = {
        "rank": rank,
        "anchors": len(rows),
        "audit_trajectory_sample_path": primitive_path.name,
        "audit_trajectory_sample_sha256": sha256_file(primitive_path),
        "retained_audit_trajectories": len(rows),
        "per_anchor_score_path": anchor_path.name,
        "per_anchor_score_sha256": sha256_file(anchor_path),
        "progress_metrics_path": progress_path.name,
        "progress_metrics_sha256": sha256_file(progress_path),
        "atomic_chunks": chunks,
    }
    return {
        "manifest": manifest,
        "calibration": calibration,
        "coverage": coverage,
        "rows": rows,
    }


def _write_free_running_hosted_progress(
    *,
    output_root: Path,
    status: str,
    completed_anchors: int,
    expected_anchors: int,
    new_anchors: int,
    identity: Mapping[str, Any],
    rank_payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if status not in {"INCOMPLETE", "COMPLETE"}:
        raise ValueError("free-running hosted progress status is invalid")
    chunks = [
        {
            "rank": int(chunk["rank"]),
            "chunk_index": int(chunk["chunk_index"]),
            "anchors": int(chunk["anchors"]),
            "manifest_path": str(chunk["manifest_path"]),
            "manifest_sha256": str(chunk["manifest_sha256"]),
        }
        for payload in sorted(rank_payloads, key=lambda item: int(item["rank"]))
        for chunk in payload["chunks"]
    ]
    chunk_manifest_set_sha256 = sha256_payload(chunks)
    payload = {
        "schema_version": FREE_RUNNING_HOSTED_PROGRESS_SCHEMA,
        "status": status,
        "updated_at": utc_now(),
        "completed": int(completed_anchors),
        "expected": int(expected_anchors),
        "completed_anchors": int(completed_anchors),
        "expected_anchors": int(expected_anchors),
        "new_anchors": int(new_anchors),
        "identity": dict(identity),
        "identity_sha256": sha256_payload(identity),
        "chunk_manifests": chunks,
        "set_sha256": chunk_manifest_set_sha256,
        "chunk_manifest_set_sha256": chunk_manifest_set_sha256,
    }
    _atomic_json(output_root / "hosted_progress.json", payload)
    return payload


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


def probe_free_running_v2_capacity(
    *,
    model: Any,
    validation_batch: Mapping[str, Any],
    contract: MultiresEventV2Contract,
    device: torch.device,
    expected_lab_scale_artifact_hash: str,
    crn_seed: int = PRODUCTION_CRN_SEED,
    precision: str = "fp16",
    trajectories_per_anchor: int = PRODUCTION_TRAJECTORIES_PER_ANCHOR,
    trajectory_batch_size: int = PRODUCTION_TRAJECTORIES_PER_ANCHOR,
) -> dict[str, Any]:
    """Prove the production 100-trajectory rollout fits without mutating state.

    The first real validation anchor is encoded exactly once and expanded to the
    production ensemble width.  CPU/all-CUDA RNG states and the model's training
    mode are restored even if the probe fails; parameter bytes are hash-checked.
    """

    if trajectories_per_anchor != PRODUCTION_TRAJECTORIES_PER_ANCHOR:
        raise ValueError("formal free-running capacity probe requires 100 trajectories")
    if trajectory_batch_size != PRODUCTION_TRAJECTORIES_PER_ANCHOR:
        raise ValueError("formal free-running capacity probe requires batch size 100")
    if precision not in {"fp16", "fp32"}:
        raise ValueError("formal free-running capacity probe precision is invalid")
    sample_ids = _string_batch(validation_batch.get("sample_id"))
    if not sample_ids:
        raise ValueError("formal free-running capacity probe lacks a validation anchor")
    first_batch = _first_anchor_batch(validation_batch, batch_size=len(sample_ids))
    sample_id = _string_batch(first_batch.get("sample_id"))[0]
    core_model = model.module if hasattr(model, "module") else model
    if not callable(getattr(core_model, "encode_for_rollout", None)) or not callable(
        getattr(core_model, "rollout_from_encoded", None)
    ):
        raise RuntimeError("formal capacity probe requires cached free-running APIs")

    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    parameter_sha_before = _model_parameter_state_sha256(core_model)
    was_training = bool(core_model.training)
    allocated_before: int | None = None
    reserved_before: int | None = None
    peak_allocated: int | None = None
    peak_reserved: int | None = None
    allocated_after: int | None = None
    reserved_after: int | None = None
    encoded_shapes: dict[str, list[int]] = {}
    generated_shapes: dict[str, list[int]] = {}
    parameter_sha_after = ""
    try:
        core_model.eval()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            allocated_before = int(torch.cuda.memory_allocated(device))
            reserved_before = int(torch.cuda.memory_reserved(device))
            torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(
            common_random_seed(
                crn_seed,
                sample_id,
                trajectory_start=0,
                trajectory_count=trajectory_batch_size,
            )
        )
        if device.type == "cuda":
            torch.cuda.manual_seed(
                common_random_seed(
                    crn_seed,
                    sample_id,
                    trajectory_start=0,
                    trajectory_count=trajectory_batch_size,
                )
            )
        batch = move_to_device(first_batch, device)
        metadata = batch.get("target_primitive_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("formal capacity probe lacks target primitive metadata")
        autocast = _autocast_factory(device, precision)
        with torch.no_grad():
            with autocast():
                encoded = _encode_batch_once(core_model, batch, expected_batch_size=1)
                expanded = _expand_encoded(encoded, trajectory_batch_size)
                sampler = RegistryPrimitiveSampler(
                    contract.process_registry,
                    metadata,
                    expected_lab_scale_artifact_hash=expected_lab_scale_artifact_hash,
                )
                outputs = core_model.rollout_from_encoded(
                    expanded["memory"],
                    expanded["memory_mask"],
                    expanded["query_tokens"],
                    sampler=sampler,
                )
        primitives = outputs.get("generated_primitives")
        masks = outputs.get("generated_primitive_masks")
        if (
            not isinstance(primitives, Mapping)
            or not primitives
            or not isinstance(masks, Mapping)
            or set(primitives) != set(masks)
            or set(primitives) != set(V2_PRIMITIVE_FEEDBACK_DIMS)
        ):
            raise RuntimeError("formal capacity probe did not generate every primitive bank")
        encoded_shapes = {key: list(value.shape) for key, value in encoded.items()}
        for likelihood_id, value in primitives.items():
            mask = masks[likelihood_id]
            bank = _bank4(value)
            mask_bank = _bank4(mask)
            if (
                bank.shape[:3] != (trajectory_batch_size, 6, 29)
                or mask_bank.shape != bank.shape
            ):
                raise RuntimeError(
                    f"formal capacity probe primitive shape failed: {likelihood_id}"
                )
            generated_shapes[str(likelihood_id)] = list(bank.shape)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            allocated_after = int(torch.cuda.memory_allocated(device))
            reserved_after = int(torch.cuda.memory_reserved(device))
        parameter_sha_after = _model_parameter_state_sha256(core_model)
        if parameter_sha_after != parameter_sha_before:
            raise RuntimeError("formal capacity probe mutated model parameter state")
    finally:
        torch.set_rng_state(cpu_rng_before)
        if cuda_rng_before:
            torch.cuda.set_rng_state_all(cuda_rng_before)
        core_model.train(was_training)

    cpu_restored = torch.equal(torch.get_rng_state(), cpu_rng_before)
    cuda_restored = not cuda_rng_before or all(
        torch.equal(current, expected)
        for current, expected in zip(
            torch.cuda.get_rng_state_all(), cuda_rng_before, strict=True
        )
    )
    if not cpu_restored or not cuda_restored:
        raise RuntimeError("formal capacity probe failed to restore RNG state")
    return {
        "schema_version": "trauma_predict.multires_event_v2_free_running_capacity_probe.v1",
        "status": "PASSED",
        "sample_id": sample_id,
        "trajectories_per_anchor": trajectories_per_anchor,
        "trajectory_batch_size": trajectory_batch_size,
        "blocks": 6,
        "fields": 29,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "encode_calls": 1,
        "neural_precision": precision,
        "parameter_state_sha256_before": parameter_sha_before,
        "parameter_state_sha256_after": parameter_sha_after,
        "parameter_state_unchanged": True,
        "rng": {
            "cpu_restored": cpu_restored,
            "cuda_device_count": len(cuda_rng_before),
            "all_cuda_restored": cuda_restored,
        },
        "encoded_shapes": encoded_shapes,
        "generated_primitive_shapes": generated_shapes,
        "cuda_memory": {
            "allocated_before_bytes": allocated_before,
            "reserved_before_bytes": reserved_before,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "allocated_after_rollout_bytes": allocated_after,
            "reserved_after_rollout_bytes": reserved_after,
            "peak_allocated_increase_bytes": (
                peak_allocated - allocated_before
                if peak_allocated is not None and allocated_before is not None
                else None
            ),
        },
    }


def _first_anchor_batch(
    batch: Mapping[str, Any],
    *,
    batch_size: int,
) -> dict[str, Any]:
    def select(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.ndim > 0 and value.shape[0] == batch_size:
                return value[:1]
            return value
        if isinstance(value, Mapping):
            return {key: select(item) for key, item in value.items()}
        if isinstance(value, list) and len(value) == batch_size:
            return value[:1]
        if isinstance(value, tuple) and len(value) == batch_size:
            return value[:1]
        return value

    result: dict[str, Any] = {}
    for key, value in batch.items():
        # Primitive metadata is registry-global, not batch-major; its sequences
        # may coincidentally have the same length as a validation batch.
        result[str(key)] = value if key == "target_primitive_metadata" else select(value)
    if len(_string_batch(result.get("sample_id"))) != 1:
        raise ValueError("formal capacity probe could not isolate the first anchor")
    return result


def _model_parameter_state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    named_parameters = sorted(model.named_parameters(), key=lambda item: item[0])
    if not named_parameters:
        raise RuntimeError("formal capacity probe found no model parameters")
    for name, parameter in named_parameters:
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


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
        "model_contract": "relation_v2",
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
        "model_contract": "relation_v2",
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
    completed_anchors: int,
    started_at: float,
) -> None:
    elapsed = max(time.monotonic() - started_at, 1e-12)
    row = {
        "event": "v2_free_running_rank_progress",
        "evaluated_at": utc_now(),
        "rank": int(rank),
        "model_contract": "relation_v2",
        "completed_anchors": int(completed_anchors),
        "elapsed_seconds": elapsed,
        "anchors_per_second": completed_anchors / elapsed,
    }
    append_rank_local_jsonl(path, row, rank_value=rank)
    print(
        "V2_FREE_PROGRESS "
        f"rank={rank} model_contract=relation_v2 anchors={completed_anchors} "
        f"elapsed={elapsed:.1f}s anchors_per_second={completed_anchors / elapsed:.4f}",
        flush=True,
    )


def verify_rank_local_artifact_preflight(
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Exercise rank-local write, hash, gather, and rank-zero assembly.

    The formal route runs this immediately after DDP initialization so an
    invalid per-rank artifact contract fails before model construction or any
    expensive rollout.
    """

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
            or str(row.get("model_contract")) != "relation_v2"
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
            "model_contract": "relation_v2",
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
    expected_world_size: int = 2,
) -> dict[str, Any]:
    """Re-open every retained rank-local canary byte and validate its contract."""

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
        or manifest.get("model_contract") != "relation_v2"
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
            or rows[0].get("model_contract") != "relation_v2"
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
    ``factory`` must not itself enter a distributed collective: every rank
    reaches the all-gather immediately after its rank-local callback returns.
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
    "FREE_RUNNING_HOSTED_PROGRESS_SCHEMA",
    "PRODUCTION_CRN_SEED",
    "PRODUCTION_CHUNK_TARGET_ANCHORS",
    "PRODUCTION_TRAJECTORIES_PER_ANCHOR",
    "common_random_seed",
    "evaluate_free_running_v2",
    "probe_free_running_v2_capacity",
    "validate_rank_local_artifact_preflight",
    "verify_rank_local_artifact_preflight",
]
