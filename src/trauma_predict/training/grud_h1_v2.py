from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import shutil
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from trauma_predict.data.multires_event.sampler import SubjectAnchorDistributedSampler
from trauma_predict.data.multires_event_v2 import MultiresEventV2Contract
from trauma_predict.training.config import expand_env, load_yaml_config_unexpanded
from trauma_predict.training.multires_event import _build_grad_scaler, _build_scheduler
from trauma_predict.training.multires_event_v2_loss import (
    V2_PRIMITIVE_FEEDBACK_DIMS,
    compute_registry_multires_event_v2_loss,
    validate_emission_registry_head_contract,
)
from trauma_predict.training.observability import (
    append_jsonl,
    atomic_write_json,
    sha256_file,
    sha256_payload,
    utc_now,
)


TRAIN_SCHEMA = "trauma_predict.grud_h1_joint_m4_train_config.v1"
DATASET_SCHEMA = "trauma_predict.grud_h1_v2_dataset_config.v1"
MODEL_SCHEMA = "trauma_predict.grud_h1_joint_m4_model_config.v1"
ROUTE = "grud_h1_to_joint_m4_v2"
MODEL_CONTRACT = "grud_h1_joint_m4_v2"
RAW_JOINT_NLL_REDUCTION = "raw_414_factor_joint_nll_batch_mean"
EXPECTED_OPTIMIZER_STEPS = 4000
EXPECTED_PRIMITIVE_FACTORS = 414
EXPECTED_INTERVAL_SUBJECTS = 505
EXPECTED_LOGGING_STEPS = 100
EXPECTED_EVAL_STEPS = 250
EXPECTED_SAVE_STEPS = 500
EXPECTED_MODEL_PARAMETERS = 1_596_987

DEFAULT_LAB_SCALE_ARTIFACT = (
    "configs/dataset/multires_event_v2_c4_lab_affine_scale_r9.json"
)
DEFAULT_LAB_SCALE_ARTIFACT_HASH = (
    "cae827b1f8b1c6a156da4bad340af1b9b0411ca2f5fbe0b9aa8d36ed06cb87bb"
)

MODEL_INPUT_KEYS = (
    "h1_values",
    "h1_observed_mask",
    "h1_delta_hours",
    "h1_sequence_mask",
    "static_numeric",
    "static_numeric_mask",
    "static_categorical",
)


@dataclass(frozen=True)
class GRUDH1V2Runtime:
    train_loader: Any
    interval_loader: Any
    train_sampler: SubjectAnchorDistributedSampler
    interval_sampler: SubjectAnchorDistributedSampler
    train_dataset: Any
    interval_dataset: Any
    contract: MultiresEventV2Contract
    lab_scale_metadata: Mapping[str, Any]
    identity: Mapping[str, Any]


@dataclass(frozen=True)
class GRUDH1V2TrainingResult:
    output_dir: Path
    metrics_path: Path
    completed_step: int
    selected_checkpoint: Path
    selected_checkpoint_sha256: str
    selected_validation_nll: float


class _LabScaleBoundCollator:
    """Attach the frozen target lab scale without changing target truth."""

    def __init__(self, collator: Any, metadata: Mapping[str, Any]) -> None:
        self.collator = collator
        self.metadata = dict(metadata)

    def __call__(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        batch = self.collator(records)
        metadata = dict(batch["target_primitive_metadata"])
        metadata["lab_scale"] = self.metadata
        batch["target_primitive_metadata"] = metadata
        return batch


def resolve_repo_path(value: str | Path, repo_root: str | Path) -> Path:
    path = Path(value)
    if "${" in str(path):
        raise ValueError(f"unexpanded environment variable in path: {path}")
    root = Path(repo_root).resolve()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_grud_h1_v2_configs(
    train_config_path: str | Path,
    *,
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    """Load the three authored YAML contracts and then resolve environment values."""

    root = Path(repo_root).resolve()
    train_path = Path(train_config_path).resolve()
    authored_train = load_yaml_config_unexpanded(train_path)
    dataset_path = resolve_repo_path(authored_train["dataset"]["config_path"], root)
    model_path = resolve_repo_path(authored_train["model"]["config_path"], root)
    authored_dataset = load_yaml_config_unexpanded(dataset_path)
    authored_model = load_yaml_config_unexpanded(model_path)
    validate_grud_h1_v2_configs(authored_train, authored_dataset, authored_model)
    return (
        expand_env(copy.deepcopy(authored_train)),
        expand_env(copy.deepcopy(authored_dataset)),
        expand_env(copy.deepcopy(authored_model)),
        dataset_path,
        model_path,
    )


def validate_grud_h1_v2_configs(
    train: Mapping[str, Any],
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
) -> None:
    """Fail closed if the matched baseline contract is silently changed."""

    if train.get("schema_version") != TRAIN_SCHEMA or train.get("route") != ROUTE:
        raise ValueError("GRU-D train config schema/route mismatch")
    if dataset.get("schema_version") != DATASET_SCHEMA or dataset.get("route") != ROUTE:
        raise ValueError("GRU-D dataset config schema/route mismatch")
    if model.get("schema_version") != MODEL_SCHEMA or model.get("route") != ROUTE:
        raise ValueError("GRU-D model config schema/route mismatch")
    if (
        model.get("role") != "matched_classic_baseline"
        or model.get("initialization") != "from_scratch"
    ):
        raise ValueError("GRU-D must remain a from-scratch matched classic baseline")
    if (
        train.get("lab_scale_artifact") != DEFAULT_LAB_SCALE_ARTIFACT
        or train.get("lab_scale_artifact_hash") != DEFAULT_LAB_SCALE_ARTIFACT_HASH
    ):
        raise ValueError("GRU-D must reuse the frozen r9 target lab-scale artifact")

    h1 = _mapping(dataset.get("h1"), "dataset.h1")
    target = _mapping(dataset.get("target"), "dataset.target")
    expected_counts = _mapping(dataset.get("expected_counts"), "dataset.expected_counts")
    if (
        h1.get("resolution") != "H1"
        or int(h1.get("channel_count", -1)) != 118
        or int(h1.get("max_history_hours", -1)) != 312
    ):
        raise ValueError("GRU-D input must remain the frozen 118-channel H1 history")
    if (
        target.get("resolution") != "M4"
        or int(target.get("ordered_blocks", -1)) != 6
        or int(target.get("field_processes", -1)) != 29
        or int(target.get("stochastic_factors", -1)) != EXPECTED_PRIMITIVE_FACTORS
    ):
        raise ValueError("GRU-D target must remain the frozen joint six-M4 contract")
    if {key: int(expected_counts.get(key, -1)) for key in ("samples", "train", "val", "test")} != {
        "samples": 50350,
        "train": 37734,
        "val": 6309,
        "test": 6307,
    }:
        raise ValueError("GRU-D dataset counts differ from the persisted C4 authority")
    normalization = _mapping(dataset.get("normalization"), "dataset.normalization")
    normalization_contract = {
        "fit_split": "train",
        "fit_by_subject_only": True,
        "fit_at_training_runtime": False,
        "clip_value": 10.0,
        "epsilon": 1.0e-6,
    }
    if {
        key: normalization.get(key) for key in normalization_contract
    } != normalization_contract:
        raise ValueError("GRU-D normalization must remain frozen train-subject-only input")

    objective = _mapping(train.get("objective"), "train.objective")
    expected_objective = {
        "future_resolution": "M4",
        "future_blocks": 6,
        "core_fields": 29,
        "stochastic_primitive_factors": EXPECTED_PRIMITIVE_FACTORS,
        "factor_composition": "joint_log_probability_sum",
        "anchor_reduction": "mean",
        "active_target_denominator": False,
        "deterministic_projection_loss": False,
        "family_weights": None,
    }
    if {key: objective.get(key) for key in expected_objective} != expected_objective:
        raise ValueError("GRU-D objective differs from the raw 414-factor joint NLL")

    evaluation = _mapping(train.get("evaluation"), "train.evaluation")
    expected_evaluation = {
        "checkpoint_metric": "joint_nll_subject_macro",
        "interval_anchor_policy": "one_fixed_anchor_per_validation_subject",
        "interval_expected_subjects": EXPECTED_INTERVAL_SUBJECTS,
        "final_evaluation_in_training_notebook": False,
        "free_running_in_training_notebook": False,
    }
    if {key: evaluation.get(key) for key in expected_evaluation} != expected_evaluation:
        raise ValueError("GRU-D interval/final evaluation boundary changed")

    training = _mapping(train.get("training"), "train.training")
    exact_training = {
        "required_cuda_devices": 1,
        "required_world_size": 1,
        "required_device_name_substring": "P100",
        "fresh_start": True,
        "resume": False,
        "forced_stop": False,
        "precision": "fp16",
        "max_steps": EXPECTED_OPTIMIZER_STEPS,
        "optimizer": "AdamW",
        "loss_reduction": RAW_JOINT_NLL_REDUCTION,
        "logging_steps": EXPECTED_LOGGING_STEPS,
        "eval_steps": EXPECTED_EVAL_STEPS,
        "save_steps": EXPECTED_SAVE_STEPS,
    }
    if {key: training.get(key) for key in exact_training} != exact_training:
        raise ValueError("GRU-D execution schedule or fresh-run contract changed")
    if int(training.get("warmup_steps", -1)) != 400:
        raise ValueError("GRU-D warmup must remain 400 optimizer steps")
    if int(training.get("gradient_accumulation_steps", 0)) < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    batch_size = int(training.get("per_device_train_batch_size", 0))
    accumulation = int(training.get("gradient_accumulation_steps", 0))
    if (
        batch_size < 1
        or int(training.get("effective_train_batch_size", -1))
        != batch_size * accumulation
    ):
        raise ValueError("effective_train_batch_size does not match batch x accumulation")
    if int(training.get("per_device_eval_batch_size", 0)) < 1:
        raise ValueError("per_device_eval_batch_size must be positive")
    if float(training.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if float(training.get("weight_decay", -1.0)) < 0.0:
        raise ValueError("weight_decay must be nonnegative")
    if float(training.get("max_grad_norm", 0.0)) <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    frozen_optimizer = {
        "per_device_train_batch_size": 32,
        "gradient_accumulation_steps": 2,
        "effective_train_batch_size": 64,
        "per_device_eval_batch_size": 32,
        "train_samples_per_epoch": 3072,
        "learning_rate": 3.0e-4,
        "weight_decay": 0.01,
        "adamw_betas": [0.9, 0.999],
        "adamw_eps": 1.0e-8,
        "max_grad_norm": 1.0,
        "grad_scaler_initial_scale": 32.0,
        "grad_scaler_growth_factor": 2.0,
        "grad_scaler_backoff_factor": 0.5,
        "grad_scaler_growth_interval": 1_000_000,
        "keep_last_checkpoints": 3,
        "dataloader_num_workers": 0,
    }
    if {key: training.get(key) for key in frozen_optimizer} != frozen_optimizer:
        raise ValueError("GRU-D optimizer/batch/precision contract changed")

    decoder = _mapping(model.get("decoder"), "model.decoder")
    if (
        decoder.get("type") != "block_major_gru_cell"
        or int(decoder.get("future_blocks", -1)) != 6
        or int(decoder.get("target_fields", -1)) != 29
        or int(decoder.get("causal_positions", -1)) != 174
        or decoder.get("teacher_feedback") != "right_shifted_registered_primitives"
    ):
        raise ValueError("GRU-D decoder no longer matches the joint causal target task")
    output = _mapping(model.get("output"), "model.output")
    if (
        output.get("primitive_parameter_heads") != "registered_v2"
        or int(output.get("stochastic_factors", -1)) != EXPECTED_PRIMITIVE_FACTORS
        or output.get("deterministic_projection_loss") is not False
        or output.get("h1_head") is not False
        or output.get("f24_head") is not False
    ):
        raise ValueError("GRU-D output heads differ from the matched V2 task")
    excluded = _mapping(model.get("excluded_method_components"), "model.excluded_method_components")
    if not excluded or any(value is not True for value in excluded.values()):
        raise ValueError("all primary-method components must remain excluded from GRU-D")
    formal = _mapping(model.get("formal_contract"), "model.formal_contract")
    observed_formal = {
        "model_parameter_count": int(formal.get("model_parameter_count", -1)),
        "causal_field_positions": int(formal.get("causal_field_positions", -1)),
        "stochastic_factors": int(formal.get("stochastic_factors", -1)),
        "relation_parameters": int(formal.get("relation_parameters", -1)),
    }
    expected_formal = {
        "model_parameter_count": EXPECTED_MODEL_PARAMETERS,
        "causal_field_positions": 174,
        "stochastic_factors": EXPECTED_PRIMITIVE_FACTORS,
        "relation_parameters": 0,
    }
    if observed_formal != expected_formal:
        raise ValueError("GRU-D formal model contract changed")


def teacher_forced_model_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Build the GRU-D model view while leaving raw-unit likelihood truth untouched."""

    input_batch = batch.get("input_batch")
    primitives = batch.get("target_primitives")
    masks = batch.get("target_primitive_masks")
    if not isinstance(input_batch, Mapping):
        raise ValueError("GRU-D batch lacks input_batch")
    if not isinstance(primitives, Mapping) or not isinstance(masks, Mapping):
        raise ValueError("GRU-D batch lacks target primitive banks")
    missing = set(MODEL_INPUT_KEYS).difference(input_batch)
    if missing:
        raise ValueError(f"GRU-D input_batch lacks model tensors: {sorted(missing)}")
    feedback: dict[str, Tensor] = {}
    for likelihood_id, width in V2_PRIMITIVE_FEEDBACK_DIMS.items():
        value = primitives.get(likelihood_id)
        if not isinstance(value, Tensor):
            raise ValueError(f"missing feedback primitive {likelihood_id!r}")
        if int(width) == 1 and value.ndim == 3:
            value = value.unsqueeze(-1)
        feedback[likelihood_id] = value
    return {
        **{key: input_batch[key] for key in MODEL_INPUT_KEYS},
        "target_primitives": feedback,
        "target_primitive_masks": masks,
    }


def exact_teacher_forced_loss(
    model: Any,
    batch: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    expected_lab_scale_artifact_hash: str,
    autocast: Any = nullcontext,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_lab_scale_artifact_hash):
        raise ValueError("GRU-D loss requires the frozen target lab-scale content hash")
    with autocast():
        outputs = model(**teacher_forced_model_inputs(batch))
    if not isinstance(outputs, Mapping):
        raise ValueError("GRU-D model output must be a mapping")
    parameters = outputs.get("primitive_parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("GRU-D model output lacks primitive parameter banks")
    loss_outputs = dict(outputs)
    loss_outputs["primitive_parameters"] = {
        key: value.float() for key, value in parameters.items()
    }
    result = compute_registry_multires_event_v2_loss(
        loss_outputs,
        batch,
        registry,
        expected_lab_scale_artifact_hash=expected_lab_scale_artifact_hash,
        reduction="mean",
    )
    if int(result.get("primitive_count", -1)) != EXPECTED_PRIMITIVE_FACTORS:
        raise AssertionError("GRU-D objective did not expand to exactly 414 factors")
    return outputs, result


def raw_414_factor_joint_nll_batch_mean(loss_result: Mapping[str, Any]) -> Tensor:
    """Sum all 414 raw log factors per anchor, then average anchors."""

    if int(loss_result.get("primitive_count", -1)) != EXPECTED_PRIMITIVE_FACTORS:
        raise ValueError("GRU-D optimizer loss requires exactly 414 primitive factors")
    primitive_log_prob = loss_result.get("primitive_log_prob")
    if (
        not isinstance(primitive_log_prob, Tensor)
        or primitive_log_prob.ndim != 2
        or int(primitive_log_prob.shape[0]) < 1
        or int(primitive_log_prob.shape[1]) != EXPECTED_PRIMITIVE_FACTORS
    ):
        raise ValueError("GRU-D optimizer loss requires a nonempty [batch,414] bank")
    return -primitive_log_prob.sum(dim=-1).mean()


def fixed_one_anchor_per_subject_indices(dataset: Any, *, seed: int) -> tuple[int, ...]:
    """Return the sampler's deterministic interval-validation anchor identities."""

    sampler = SubjectAnchorDistributedSampler(
        dataset,
        seed=int(seed),
        mode="one_fixed_per_subject",
        shuffle=False,
        pad_to_world_size=False,
        require_even_divisible=False,
    )
    return tuple(iter(sampler))


def summarize_subject_macro_nll(rows: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot summarize an empty validation result")
    sample_ids = [str(row["sample_id"]) for row in materialized]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("interval validation contains duplicate sample_id values")
    by_subject: dict[str, list[float]] = defaultdict(list)
    for row in materialized:
        value = float(row["joint_nll"])
        if not math.isfinite(value):
            raise FloatingPointError("interval validation produced a non-finite NLL")
        by_subject[str(row["subject_id"])].append(value)
    subject_means = [sum(values) / len(values) for values in by_subject.values()]
    return {
        "samples": len(materialized),
        "subjects": len(subject_means),
        "joint_nll_anchor_mean": sum(float(row["joint_nll"]) for row in materialized)
        / len(materialized),
        "joint_nll_subject_macro": sum(subject_means) / len(subject_means),
    }


def evaluate_interval_teacher_forced(
    *,
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    registry: Mapping[str, Any],
    device: torch.device,
    step: int,
    expected_subjects: int,
    precision: str,
    expected_lab_scale_artifact_hash: str,
    metrics_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate only the fixed one-anchor-per-validation-subject subset."""

    model.eval()
    rows: list[dict[str, Any]] = []
    autocast = _autocast_factory(device, precision)
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_to_device(raw_batch, device)
            _, loss_result = exact_teacher_forced_loss(
                model,
                batch,
                registry,
                expected_lab_scale_artifact_hash=expected_lab_scale_artifact_hash,
                autocast=autocast,
            )
            per_sample = loss_result.get("per_sample_nll")
            if not isinstance(per_sample, Tensor) or per_sample.ndim != 1:
                raise ValueError("GRU-D interval loss lacks one NLL per anchor")
            sample_ids = _string_batch(batch.get("sample_id"))
            subject_ids = _string_batch(batch.get("subject_id"))
            values = per_sample.detach().float().cpu().tolist()
            if not (len(values) == len(sample_ids) == len(subject_ids)):
                raise ValueError("GRU-D interval identities do not align with NLL values")
            rows.extend(
                {"sample_id": sample_id, "subject_id": subject_id, "joint_nll": float(value)}
                for sample_id, subject_id, value in zip(
                    sample_ids, subject_ids, values, strict=True
                )
            )
    summary = summarize_subject_macro_nll(rows)
    if int(summary["subjects"]) != int(expected_subjects):
        raise RuntimeError(
            f"GRU-D interval validation expected {expected_subjects} subjects, "
            f"got {summary['subjects']}"
        )
    if int(summary["samples"]) != int(expected_subjects):
        raise RuntimeError("GRU-D interval policy must contain exactly one anchor per subject")
    result = {
        "schema_version": "trauma_predict.grud_h1_v2_interval_evaluation.v1",
        "evaluated_at": utc_now(),
        "phase": "interval",
        "step": int(step),
        "model_contract": MODEL_CONTRACT,
        **summary,
        "primitive_factors_per_anchor": EXPECTED_PRIMITIVE_FACTORS,
        "aggregation": "sum_414_then_one_anchor_per_subject_macro",
    }
    print(
        "GRUD_V2_VAL_NLL "
        f"step={int(step)} anchors={result['samples']} subjects={result['subjects']} "
        f"subject_macro={float(result['joint_nll_subject_macro']):.6f} "
        f"anchor_mean={float(result['joint_nll_anchor_mean']):.6f}",
        flush=True,
    )
    if metrics_path is not None:
        append_jsonl(metrics_path, {"event": "interval_validation", **result})
    model.train()
    return result


def build_grud_h1_v2_runtime(
    train: Mapping[str, Any],
    dataset: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> GRUDH1V2Runtime:
    """Bind frozen H1 inputs and V2 targets; no sample or normalization is fit here."""

    from trauma_predict.data.grud_h1_v2 import (
        GRUDH1V2Collator,
        GRUDH1V2Dataset,
        H1ChannelRegistry,
        load_frozen_h1_normalizer,
    )

    root = Path(repo_root).resolve()
    h1_config = _mapping(dataset["h1"], "dataset.h1")
    target_config = _mapping(dataset["target"], "dataset.target")
    normalization_config = _mapping(dataset["normalization"], "dataset.normalization")
    loader_config = _mapping(dataset["loader"], "dataset.loader")
    training = _mapping(train["training"], "train.training")
    evaluation = _mapping(train["evaluation"], "train.evaluation")

    h1_root = resolve_repo_path(str(h1_config["root"]), root)
    target_root = resolve_repo_path(str(target_config["root"]), root)
    normalization_path = resolve_repo_path(str(normalization_config["path"]), root)
    if not (h1_root / "SUCCEEDED").is_file():
        raise FileNotFoundError(f"H1 sample authority is incomplete: {h1_root}")
    if not (target_root / "SUCCEEDED").is_file():
        raise FileNotFoundError(f"V2 target authority is incomplete: {target_root}")

    h1_manifest_path = h1_root / "dataset_manifest.json"
    h1_manifest = _read_json(h1_manifest_path)
    if h1_manifest.get("dataset_id") != h1_config.get("dataset_id"):
        raise ValueError("attached H1 dataset_id differs from the YAML contract")
    if sha256_file(h1_manifest_path) != str(h1_config["dataset_manifest_sha256"]):
        raise ValueError("attached H1 dataset manifest hash differs from the YAML contract")
    h1_sample_manifest = h1_root / "sample_manifest.csv"
    if sha256_file(h1_sample_manifest) != str(h1_config["sample_manifest_sha256"]):
        raise ValueError("attached H1 sample manifest hash differs from the YAML contract")
    channel_path = h1_root / "h1_event_templates.json"
    if sha256_file(channel_path) != str(h1_config["channel_registry_sha256"]):
        raise ValueError("attached H1 channel registry hash differs from the YAML contract")

    contract = MultiresEventV2Contract.from_dataset_root(target_root)
    if contract.manifest.get("dataset_id") != target_config.get("dataset_id"):
        raise ValueError("attached V2 target dataset_id differs from the YAML contract")
    if sha256_file(target_root / "dataset_manifest.json") != str(
        target_config["dataset_manifest_sha256"]
    ):
        raise ValueError("attached V2 target manifest hash differs from the YAML contract")
    if sha256_file(target_root / "sample_manifest.csv") != str(
        target_config["sample_manifest_sha256"]
    ):
        raise ValueError("attached V2 target sample manifest differs from the YAML contract")
    if contract.contract_bundle_hash != target_config.get("contract_bundle_hash"):
        raise ValueError("attached V2 target contract bundle differs from the YAML contract")
    validate_emission_registry_head_contract(contract.emission_registry)

    supervision_path = resolve_repo_path(
        str(dataset.get("supervision_path", "configs/model/multires_event_v1_supervision.json")),
        root,
    )
    from trauma_predict.data.multires_event import SupervisionContract

    supervision = SupervisionContract.from_json(supervision_path)
    channel_registry = H1ChannelRegistry.from_json(channel_path)
    normalization = load_frozen_h1_normalizer(
        normalization_path,
        expected_dataset_fingerprint=str(h1_manifest["fingerprint"]),
        expected_supervision_sha256=supervision.source_sha256,
    )

    lab_scale_path = resolve_repo_path(
        str(train.get("lab_scale_artifact", DEFAULT_LAB_SCALE_ARTIFACT)), root
    )
    lab_scale_hash = str(
        train.get("lab_scale_artifact_hash", DEFAULT_LAB_SCALE_ARTIFACT_HASH)
    )
    from trauma_predict.training.multires_event_v2 import load_lab_scale_artifact

    lab_scale_metadata = load_lab_scale_artifact(
        lab_scale_path,
        expected_content_sha256=lab_scale_hash,
        contract=contract,
    )
    base_collator = GRUDH1V2Collator(
        contract=contract,
        supervision=supervision,
        templates=channel_registry.templates,
        normalization=normalization,
        channel_registry=channel_registry,
    )
    collator = _LabScaleBoundCollator(base_collator, lab_scale_metadata)

    common_dataset = {
        "h1_root": h1_root,
        "target_root": target_root,
        "contract": contract,
        "cache_shards": int(loader_config.get("cache_shards", 1)),
        "strict": True,
        "verify_shard_hashes": False,
    }
    train_dataset = GRUDH1V2Dataset(split="train", **common_dataset)
    interval_dataset = GRUDH1V2Dataset(split="val", **common_dataset)
    train_sampler = SubjectAnchorDistributedSampler(
        train_dataset,
        seed=int(train["seed"]),
        mode="subject_uniform_replacement",
        shuffle=True,
        pad_to_world_size=False,
        require_even_divisible=True,
        max_samples=int(training["train_samples_per_epoch"]),
    )
    interval_sampler = SubjectAnchorDistributedSampler(
        interval_dataset,
        seed=int(train["seed"]),
        mode="one_fixed_per_subject",
        shuffle=False,
        pad_to_world_size=False,
        require_even_divisible=False,
    )
    interval_subjects = len(interval_sampler)
    expected_subjects = int(evaluation["interval_expected_subjects"])
    if interval_subjects != expected_subjects:
        raise RuntimeError(
            f"fixed interval validation expected {expected_subjects} subjects, "
            f"found {interval_subjects}"
        )

    num_workers = int(training.get("dataloader_num_workers", 0))
    pin_memory = bool(loader_config.get("pin_memory", True))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["per_device_train_batch_size"]),
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    interval_loader = DataLoader(
        interval_dataset,
        batch_size=int(training["per_device_eval_batch_size"]),
        sampler=interval_sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    identity = {
        "h1_dataset_id": str(h1_manifest["dataset_id"]),
        "h1_dataset_fingerprint": str(h1_manifest["fingerprint"]),
        "h1_dataset_manifest_sha256": sha256_file(h1_manifest_path),
        "h1_sample_manifest_sha256": sha256_file(h1_sample_manifest),
        "target_dataset_id": str(contract.manifest["dataset_id"]),
        "target_dataset_manifest_sha256": sha256_file(target_root / "dataset_manifest.json"),
        "target_contract_bundle_hash": contract.contract_bundle_hash,
        "normalization_sha256": sha256_file(normalization_path),
        "lab_scale_artifact_sha256": lab_scale_hash,
        "interval_samples": interval_subjects,
        "interval_subjects": interval_subjects,
    }
    return GRUDH1V2Runtime(
        train_loader=train_loader,
        interval_loader=interval_loader,
        train_sampler=train_sampler,
        interval_sampler=interval_sampler,
        train_dataset=train_dataset,
        interval_dataset=interval_dataset,
        contract=contract,
        lab_scale_metadata=lab_scale_metadata,
        identity=identity,
    )


def build_grud_h1_v2_model(
    model_config: Mapping[str, Any],
    *,
    contract: MultiresEventV2Contract,
) -> torch.nn.Module:
    from trauma_predict.modeling.grud_h1_v2 import (
        GRUDH1JointM4Config,
        build_grud_h1_joint_m4_model,
    )

    config = GRUDH1JointM4Config.from_mapping(model_config)
    return build_grud_h1_joint_m4_model(config, contract=contract)


def build_grud_h1_v2_optimizer(
    model: torch.nn.Module,
    training: Mapping[str, Any],
) -> torch.optim.AdamW:
    if training.get("optimizer") != "AdamW":
        raise ValueError("GRU-D baseline requires AdamW")
    betas = training.get("adamw_betas", (0.9, 0.999))
    if not isinstance(betas, Sequence) or len(betas) != 2:
        raise ValueError("adamw_betas must contain two values")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("GRU-D model has no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(training.get("adamw_eps", 1.0e-8)),
    )


def require_single_p100(training: Mapping[str, Any]) -> torch.device:
    world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
    if world_size != 1 or int(training.get("required_world_size", -1)) != 1:
        raise RuntimeError("GRU-D formal training requires WORLD_SIZE=1")
    if not torch.cuda.is_available():
        raise RuntimeError("GRU-D formal training requires one CUDA P100")
    if torch.cuda.device_count() != 1 or int(training.get("required_cuda_devices", -1)) != 1:
        raise RuntimeError("GRU-D formal training requires exactly one visible CUDA device")
    name = torch.cuda.get_device_name(0)
    required = str(training.get("required_device_name_substring") or "")
    if required not in name:
        raise RuntimeError(f"GRU-D formal training requires {required}, observed {name}")
    torch.cuda.set_device(0)
    return torch.device("cuda", 0)


def run_grud_h1_v2_training(
    train_config_path: str | Path,
    *,
    repo_root: str | Path,
) -> GRUDH1V2TrainingResult:
    """Run one fresh 4,000-update GRU-D baseline and export its selected checkpoint."""

    root = Path(repo_root).resolve()
    train, dataset, model_config, dataset_path, model_path = load_grud_h1_v2_configs(
        train_config_path,
        repo_root=root,
    )
    training = _mapping(train["training"], "train.training")
    output_dir = resolve_repo_path(str(train["outputs"]["output_dir"]), root)
    metrics_path = resolve_repo_path(str(train["outputs"]["metrics_jsonl"]), root)
    _require_fresh_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if metrics_path.parent != output_dir:
        raise ValueError("GRU-D metrics_jsonl must be directly inside output_dir")

    manifest_path = output_dir / "training_manifest.json"
    config_sha256 = sha256_payload(
        {"train": train, "dataset": dataset, "model": model_config}
    )
    running_manifest = {
        "schema_version": "trauma_predict.grud_h1_v2_training_manifest.v1",
        "status": "RUNNING",
        "started_at": utc_now(),
        "route": ROUTE,
        "completed_step": 0,
        "config_sha256": config_sha256,
    }
    atomic_write_json(manifest_path, running_manifest)
    global_step = 0

    try:
        device = require_single_p100(training)
        _seed_everything(int(train["seed"]))
        runtime = build_grud_h1_v2_runtime(train, dataset, repo_root=root)
        model = build_grud_h1_v2_model(model_config, contract=runtime.contract).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != EXPECTED_MODEL_PARAMETERS:
            raise RuntimeError(
                f"GRU-D parameter count changed: {parameter_count} != "
                f"{EXPECTED_MODEL_PARAMETERS}"
            )
        print(
            "GRUD_V2_MODEL_OK "
            f"parameters={parameter_count} encoder=GRUD decoder_positions=174 "
            "blocks=6 fields=29 factors=414 relations=0",
            flush=True,
        )
        optimizer = build_grud_h1_v2_optimizer(model, training)
        scheduler = _build_scheduler(optimizer, training)
        scaler = _build_grad_scaler(torch, device, training)
        registry = runtime.contract.process_registry
        lab_scale_hash = str(
            train.get("lab_scale_artifact_hash", DEFAULT_LAB_SCALE_ARTIFACT_HASH)
        )
        autocast = _autocast_factory(device, str(training["precision"]))

        append_jsonl(
            metrics_path,
            {
                "event": "training_start",
                "created_at": utc_now(),
                "step": 0,
                "max_steps": EXPECTED_OPTIMIZER_STEPS,
                "identity": dict(runtime.identity),
            },
        )
        print(
            "GRUD_V2_TRAINING_START "
            f"restored_step=0 target_step={EXPECTED_OPTIMIZER_STEPS} forced_stop=false",
            flush=True,
        )
        best_step: int | None = None
        best_metric = math.inf
        train_nll_sum = 0.0
        train_anchor_count = 0
        epoch = 0
        runtime.train_sampler.set_epoch(epoch)
        train_iterator = iter(runtime.train_loader)
        accumulation_steps = int(training["gradient_accumulation_steps"])

        while global_step < EXPECTED_OPTIMIZER_STEPS:
            optimizer.zero_grad(set_to_none=True)
            attempt_nll_sum = 0.0
            attempt_anchor_count = 0
            for _ in range(accumulation_steps):
                try:
                    raw_batch = next(train_iterator)
                except StopIteration:
                    epoch += 1
                    runtime.train_sampler.set_epoch(epoch)
                    train_iterator = iter(runtime.train_loader)
                    raw_batch = next(train_iterator)
                batch = move_to_device(raw_batch, device)
                _, loss_result = exact_teacher_forced_loss(
                    model,
                    batch,
                    registry,
                    expected_lab_scale_artifact_hash=lab_scale_hash,
                    autocast=autocast,
                )
                loss = raw_414_factor_joint_nll_batch_mean(loss_result)
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("GRU-D training NLL is non-finite")
                per_sample = loss_result.get("per_sample_nll")
                if not isinstance(per_sample, Tensor) or per_sample.ndim != 1:
                    raise ValueError("GRU-D loss lacks a per-anchor NLL vector")
                batch_size = int(per_sample.numel())
                attempt_nll_sum += float(per_sample.detach().sum().cpu())
                attempt_anchor_count += batch_size
                scaler.scale(loss / accumulation_steps).backward()

            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["max_grad_norm"])
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                raise FloatingPointError("GRU-D gradient norm is non-finite")
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            if scale_after < scale_before:
                append_jsonl(
                    metrics_path,
                    {
                        "event": "amp_optimizer_skip",
                        "created_at": utc_now(),
                        "completed_step": global_step,
                        "scale_before": scale_before,
                        "scale_after": scale_after,
                    },
                )
                continue
            scheduler.step()
            global_step += 1
            train_nll_sum += attempt_nll_sum
            train_anchor_count += attempt_anchor_count

            if global_step % EXPECTED_LOGGING_STEPS == 0:
                if train_anchor_count < 1:
                    raise RuntimeError("GRU-D training log interval contains no anchors")
                anchor_mean = train_nll_sum / train_anchor_count
                print(
                    f"GRUD_V2_TRAIN_NLL step={global_step} anchor_mean={anchor_mean:.6f}",
                    flush=True,
                )
                append_jsonl(
                    metrics_path,
                    {
                        "event": "train_nll",
                        "created_at": utc_now(),
                        "step": global_step,
                        "anchors": train_anchor_count,
                        "joint_nll_anchor_mean": anchor_mean,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    },
                )
                train_nll_sum = 0.0
                train_anchor_count = 0

            interval_result: Mapping[str, Any] | None = None
            if global_step % EXPECTED_EVAL_STEPS == 0:
                interval_result = evaluate_interval_teacher_forced(
                    model=model,
                    loader=runtime.interval_loader,
                    registry=registry,
                    device=device,
                    step=global_step,
                    expected_subjects=EXPECTED_INTERVAL_SUBJECTS,
                    precision=str(training["precision"]),
                    expected_lab_scale_artifact_hash=lab_scale_hash,
                    metrics_path=metrics_path,
                )

            if global_step % EXPECTED_SAVE_STEPS == 0:
                if interval_result is None:
                    raise AssertionError(
                        "every GRU-D checkpoint step must have interval validation"
                    )
                _save_checkpoint(
                    output_dir=output_dir,
                    step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    train_config=train,
                    dataset_config=dataset,
                    model_config=model_config,
                    runtime_identity=runtime.identity,
                    validation=interval_result,
                )
                print(f"GRUD_V2_CHECKPOINT_OK step={global_step}", flush=True)
                metric = float(interval_result["joint_nll_subject_macro"])
                if metric < best_metric:
                    best_metric = metric
                    best_step = global_step
                _prune_checkpoints(
                    output_dir,
                    keep_last=int(training.get("keep_last_checkpoints", 3)),
                    preserve_step=best_step,
                )

        if global_step != EXPECTED_OPTIMIZER_STEPS or best_step is None:
            raise RuntimeError("GRU-D training did not complete its selected-checkpoint contract")
        final_checkpoint = output_dir / f"checkpoint-{EXPECTED_OPTIMIZER_STEPS}" / "checkpoint.pt"
        if not final_checkpoint.is_file():
            raise RuntimeError("GRU-D step-4000 checkpoint is absent")
        selected_checkpoint = output_dir / f"checkpoint-{best_step}" / "checkpoint.pt"
        if not selected_checkpoint.is_file():
            raise RuntimeError("GRU-D selected checkpoint was not preserved")
        selected_sha256 = sha256_file(selected_checkpoint)
        succeeded = {
            **running_manifest,
            "status": "SUCCEEDED",
            "completed_at": utc_now(),
            "completed_step": EXPECTED_OPTIMIZER_STEPS,
            "selected_checkpoint": {
                "path": str(selected_checkpoint.relative_to(output_dir)),
                "sha256": selected_sha256,
                "step": best_step,
                "joint_nll_subject_macro": best_metric,
            },
            "step_4000_checkpoint": {
                "path": str(final_checkpoint.relative_to(output_dir)),
                "sha256": sha256_file(final_checkpoint),
            },
            "metrics_jsonl": str(metrics_path.relative_to(output_dir)),
            "identity": dict(runtime.identity),
            "config_paths": {
                "train": str(Path(train_config_path).resolve()),
                "dataset": str(dataset_path),
                "model": str(model_path),
            },
        }
        atomic_write_json(manifest_path, succeeded)
        print(
            "GRUD_V2_TRAINING_FINISHED "
            f"status=SUCCEEDED step={EXPECTED_OPTIMIZER_STEPS} selected_step={best_step}",
            flush=True,
        )
        return GRUDH1V2TrainingResult(
            output_dir=output_dir,
            metrics_path=metrics_path,
            completed_step=global_step,
            selected_checkpoint=selected_checkpoint,
            selected_checkpoint_sha256=selected_sha256,
            selected_validation_nll=best_metric,
        )
    except BaseException as exc:
        failed = {
            **running_manifest,
            "status": "FAILED",
            "failed_at": utc_now(),
            "completed_step": global_step,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        atomic_write_json(manifest_path, failed)
        raise


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def _save_checkpoint(
    *,
    output_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    train_config: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> Path:
    checkpoint_dir = output_dir / f"checkpoint-{int(step)}"
    temporary_dir = output_dir / f".checkpoint-{int(step)}.partial"
    if checkpoint_dir.exists() or temporary_dir.exists():
        raise FileExistsError(f"checkpoint destination already exists for step {step}")
    temporary_dir.mkdir(parents=False)
    checkpoint_path = temporary_dir / "checkpoint.pt"
    torch.save(
        {
            "schema_version": "trauma_predict.grud_h1_v2_checkpoint.v1",
            "created_at": utc_now(),
            "step": int(step),
            "model_contract": MODEL_CONTRACT,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "grad_scaler_state_dict": scaler.state_dict(),
            "train_config": dict(train_config),
            "dataset_config": dict(dataset_config),
            "model_config": dict(model_config),
            "runtime_identity": dict(runtime_identity),
            "validation": dict(validation),
        },
        checkpoint_path,
    )
    atomic_write_json(
        temporary_dir / "manifest.json",
        {
            "schema_version": "trauma_predict.grud_h1_v2_checkpoint_manifest.v1",
            "step": int(step),
            "checkpoint": "checkpoint.pt",
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "joint_nll_subject_macro": float(validation["joint_nll_subject_macro"]),
        },
    )
    temporary_dir.replace(checkpoint_dir)
    return checkpoint_dir / "checkpoint.pt"


def _prune_checkpoints(output_dir: Path, *, keep_last: int, preserve_step: int | None) -> None:
    if keep_last < 1:
        raise ValueError("keep_last_checkpoints must be positive")
    checkpoints = sorted(
        (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("-", 1)[1]),
    )
    keep = set(checkpoints[-keep_last:])
    if preserve_step is not None:
        keep.add(output_dir / f"checkpoint-{int(preserve_step)}")
    for checkpoint in checkpoints:
        if checkpoint not in keep:
            shutil.rmtree(checkpoint)


def _require_fresh_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"fresh GRU-D run refuses an existing output directory: {output_dir}"
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _autocast_factory(device: torch.device, precision: str) -> Any:
    if device.type != "cuda" or precision != "fp16":
        return nullcontext

    def factory() -> Any:
        try:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        except AttributeError:  # pragma: no cover - older supported torch
            return torch.cuda.amp.autocast(dtype=torch.float16)

    return factory


def _string_batch(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("batch identity must be a list or tuple")
    return [str(item) for item in value]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


__all__ = [
    "EXPECTED_EVAL_STEPS",
    "EXPECTED_INTERVAL_SUBJECTS",
    "EXPECTED_LOGGING_STEPS",
    "EXPECTED_OPTIMIZER_STEPS",
    "EXPECTED_PRIMITIVE_FACTORS",
    "EXPECTED_SAVE_STEPS",
    "GRUDH1V2Runtime",
    "GRUDH1V2TrainingResult",
    "RAW_JOINT_NLL_REDUCTION",
    "build_grud_h1_v2_model",
    "build_grud_h1_v2_optimizer",
    "build_grud_h1_v2_runtime",
    "evaluate_interval_teacher_forced",
    "exact_teacher_forced_loss",
    "fixed_one_anchor_per_subject_indices",
    "load_grud_h1_v2_configs",
    "raw_414_factor_joint_nll_batch_mean",
    "require_single_p100",
    "run_grud_h1_v2_training",
    "summarize_subject_macro_nll",
    "teacher_forced_model_inputs",
    "validate_grud_h1_v2_configs",
]
