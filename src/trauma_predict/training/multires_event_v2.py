from __future__ import annotations

import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping

import torch

from trauma_predict.data.multires_event import (
    MultiresEventDataset,
    SubjectAnchorDistributedSampler,
    SupervisionContract,
    build_runtime as build_v1_runtime,
)
from trauma_predict.data.multires_event_v2 import (
    MultiresEventV2Collator,
    MultiresEventV2Contract,
    MultiresEventV2Dataset,
)
from trauma_predict.eval.multires_event_v2 import (
    evaluate_teacher_forced,
    exact_teacher_forced_loss,
    move_to_device,
)
from trauma_predict.eval.multires_event_v2_free_running import (
    _collect_distributed_phase,
    evaluate_free_running_v2,
    verify_rank_local_artifact_preflight,
)
from trauma_predict.eval.multires_event_v2_promotion_contract import (
    load_promotion_metric_contract,
)
from trauma_predict.eval.multires_event_v2_projections import (
    load_standardized_primitive_scale_artifact,
)
from trauma_predict.modeling.multires_event_v2.config import MultiResolutionEventV2Config
from trauma_predict.modeling.multires_event_v2.model import MultiResolutionEventV2Model
from trauma_predict.training.config import load_yaml_config
from trauma_predict.training.multires_event import (
    _barrier,
    _build_grad_scaler,
    _build_scheduler,
    _capture_rng_state,
    _maybe_resume,
    _prune_checkpoints,
    _restore_rng_state,
    _seed_everything,
    _set_sampler_epoch,
    _unwrapped_model,
    _world_size,
)
from trauma_predict.training.multires_event_v2_loss import (
    REGISTERED_CORE_FIELD_IDS,
    V2_PRIMITIVE_FEEDBACK_DIMS,
    V2_PRIMITIVE_HEAD_DIMS,
    validate_emission_registry_head_contract,
)
from trauma_predict.training.observability import (
    append_jsonl,
    atomic_write_json,
    is_rank_zero,
    sha256_file,
    sha256_payload,
    utc_now,
)


ROUTE = "multires_event_v2_m4_relational_primary"
MATCHED_MODES = ("block", "trajectory", "relational")
TRAINING_AUTHORIZED = True
AUTHORIZED_TRAINING_RUN_NAMES: tuple[str, ...] = (
    "t4x2_multires_event_v2_relational",
)
TRAINING_AUTHORIZATION_REASON = (
    "the frozen primary is the 47,801,855-parameter relational six-M4 model; "
    "block and trajectory are optional later ablations and cannot gate it"
)
VERIFICATION_AUTHORIZED = True
AUTHORIZED_VERIFICATION_RUN_NAMES = ("t4x2_multires_event_v2_relational",)
VERIFICATION_AUTHORIZATION_REASON = (
    "verification must exercise the exact relational primary model and training state"
)
EXPECTED_BASE_DATASET_ID = "multires_event_v1_c4_full_20260712"
EXPECTED_BASE_FINGERPRINT = "d58d003b6a9b2dd7c1f8d269a1867b534ea475a91118d7d4d44804bee69f9e47"
EXPECTED_BASE_MANIFEST_SHA256 = "4e7742900907e0e2f774099ba1dd485468210ff3da9ddaef3ec3bf67957000c3"
EXPECTED_BASE_SAMPLE_MANIFEST_SHA256 = "b3d4305353997320fe310c4df6e15619026db6f229a124b0c9a5e1d89898f05e"
EXPECTED_SUBJECT_SPLIT_SHA256 = "89deb50c2c6415dff5ce00338a980e25531433e8dee835b004a27d561e7adb6d"
EXPECTED_TARGET_DATASET_ID = "multires_event_m4_target_v2_c4_full_20260714_r9"
EXPECTED_TARGET_MANIFEST_SHA256 = "6c4e1e300686195fb2c58bfcbd74df6c7cb905d7031985cb7a7624d5c7061f1e"
EXPECTED_TARGET_SAMPLE_MANIFEST_SHA256 = "df5eedcee0abf7d09fea86572db471047bdaa82dc28b14dc8bbf0dac0e32dd0e"
EXPECTED_CONTRACT_BUNDLE_HASH = "ee4786d5141c5e0a4abfd1780bbca93244c3b2c8323f3ef59a62e09123d11c05"
EXPECTED_PROCESS_CONTRACT_SHA256 = "2cd5fd86e42f2dc582080a1d147495a24ac6eebb5c9b007f9575918a79f2b33b"
EXPECTED_EMISSION_CONTRACT_SHA256 = "d41a0965e0ba2170c28c35c0320fc5c78247982548ba354cb8c137113ae6f48c"
EXPECTED_PROJECTION_CONTRACT_SHA256 = "3974797e7001e0292a89885a3edc81d09134c4ea44e25ea91bf07e18eaf06b65"
EXPECTED_RELATION_CONTRACT_SHA256 = "65286cd9fb7e1038270de39ea17daafffb160cf9c5ab7bb3beb2556a9aa8eea0"
EXPECTED_SIDECAR_SCHEMA_SHA256 = "58d3673b6e232344709555a7bff2186047b08ea7932b6685553ee1b526d7e0dd"
EXPECTED_LAB_SCALE_ARTIFACT = "configs/dataset/multires_event_v2_c4_lab_affine_scale_r9.json"
EXPECTED_LAB_SCALE_ARTIFACT_HASH = "cae827b1f8b1c6a156da4bad340af1b9b0411ca2f5fbe0b9aa8d36ed06cb87bb"
EXPECTED_STANDARDIZED_PRIMITIVE_SCALE_ARTIFACT = (
    "configs/dataset/multires_event_v2_c4_standardized_primitive_scale_r9.json"
)
EXPECTED_STANDARDIZED_PRIMITIVE_SCALE_ARTIFACT_HASH = (
    "f075a9d2d415028845026b06e746cecf382102dfc0ed2c31631000506030665f"
)
EXPECTED_PROMOTION_METRIC_CONTRACT = (
    "configs/evaluation/multires_event_v2_promotion_v2.json"
)
EXPECTED_PROMOTION_METRIC_CONTRACT_HASH = (
    "7b5b85d5d3b3604308e1fe8b1471bc6c5c0c20bb16e3b9aaffd0c5e3afb53f3f"
)
EXPECTED_COUNTS = {"samples": 50350, "train": 37734, "val": 6309, "test": 6307, "shards": 52}
EXPECTED_FORMAL_MODEL_PARAMETER_COUNT = 47_801_855
OPTIMIZER_CONTRACT_VERSION = "trauma_predict.multires_event_v2_optimizer.v1"
RAW_JOINT_NLL_REDUCTION = "raw_414_factor_joint_nll_batch_mean"
OPTIMIZER_HEALTH_SUMMARY_SCHEMA = (
    "trauma_predict.multires_event_v2_optimizer_health_summary.v1"
)
EXPECTED_OPTIMIZER_CONTRACT = {
    "optimizer_contract_version": OPTIMIZER_CONTRACT_VERSION,
    "loss_reduction": RAW_JOINT_NLL_REDUCTION,
    "optimizer": "AdamW",
    "learning_rate": 2.0e-4,
    "weight_decay": 0.01,
    "adamw_betas": [0.9, 0.999],
    "adamw_eps": 1.0e-8,
    "adamw_amsgrad": False,
    "adamw_maximize": False,
    "adamw_foreach": False,
    "adamw_fused": False,
    "gradient_clipping": "disabled",
}
CAPACITY_PROBE_SCHEMA = "trauma_predict.multires_event_v2_capacity_probe.v3"
CAPACITY_PROBE_OPTIMIZER_STEPS = 2
CAPACITY_PROBE_VALIDATION_ANCHORS = 100
CAPACITY_PROBE_TRAJECTORIES_PER_ANCHOR = 100
CAPACITY_SEMANTIC_CANARY_ANCHORS = 2
CAPACITY_SEMANTIC_CANARY_TRAJECTORIES_PER_ANCHOR = 100
V2_DISTRIBUTED_TIMEOUT_SECONDS = 600
V2_NCCL_MONITOR_HEARTBEAT_TIMEOUT_SECONDS = 120
V2_EARLY_CANARY_PROCESS_GROUP_TIMEOUT_SECONDS = 60
CAPACITY_RUNTIME_POLICY = "background_save_and_run_informational_only"
CAPACITY_STRUCTURAL_METRICS = (
    "field_macro_lag1_variogram_score_p0_5",
    "relation_edge_macro_variogram_score_p0_5",
    "marginal_value_crps",
    "marginal_state_crps",
)
EXPECTED_FORMAL_ARCHITECTURE = {
    "hidden_size": 480,
    "num_attention_heads": 8,
    "trajectory_encoder_layers": 6,
    "target_decoder_layers": 6,
    "block_compressor_layers": 1,
    "block_latent_count": 8,
    "future_block_count": 6,
    "target_field_count": 29,
    "relation_type_count": 14,
}
BEST_CHECKPOINT_SCHEMA = "trauma_predict.multires_event_v2_best_checkpoint.v1"
V2_CHECKPOINT_SCHEMA = "trauma_predict.multires_event_v2_checkpoint.v2"
SELECTED_MODEL_SCHEMA = "trauma_predict.multires_event_v2_selected_model.v1"
RUN_ARTIFACT_PATHS = {
    "input_normalization": "artifacts/input_normalization.json",
    "lab_affine_scale": "artifacts/lab_affine_scale.json",
    "standardized_primitive_scale": "artifacts/standardized_primitive_scale.json",
    "promotion_metric_contract": "artifacts/promotion_metric_contract.json",
    "runtime_environment": "artifacts/runtime_environment.json",
    "train_config": "artifacts/config/train.yaml",
    "dataset_config": "artifacts/config/dataset.yaml",
    "model_config": "artifacts/config/model.yaml",
}


@dataclass(frozen=True)
class MultiresEventV2Runtime:
    train_loader: Any
    eval_loader: Any
    train_sampler: SubjectAnchorDistributedSampler
    eval_sampler: SubjectAnchorDistributedSampler
    train_dataset: MultiresEventV2Dataset
    eval_dataset: MultiresEventV2Dataset
    contract: MultiresEventV2Contract
    normalization: Any
    identity: Mapping[str, Any]


@dataclass(frozen=True)
class _OptimizerUpdateProbe:
    parameter_name: str
    parameter: torch.nn.Parameter
    flat_index: int
    value_before: torch.Tensor


_CAPACITY_AUTHORIZATION_GUARD = object()
_VERIFICATION_AUTHORIZATION_GUARD = object()


@dataclass(frozen=True)
class _CapacityAuthorization:
    guard: object
    config_path: str
    config_file_sha256: str
    train_config_sha256: str
    mode: str
    capacity_report_path: str
    capacity_report_sha256: str


class _LabScaleBoundCollator:
    """Attach a preflighted compact scale view without changing target truth."""

    def __init__(self, collator: MultiresEventV2Collator, metadata: Mapping[str, Any]) -> None:
        self.collator = collator
        self.metadata = dict(metadata)

    def __call__(self, records: Any) -> dict[str, Any]:
        batch = self.collator(records)
        metadata = dict(batch["target_primitive_metadata"])
        metadata["lab_scale"] = self.metadata
        batch["target_primitive_metadata"] = metadata
        return batch


def resolve_repo_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    if "${" in str(path):
        raise ValueError(f"unexpanded environment variable in path: {path}")
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def load_multires_event_v2_configs(
    train_config_path: str | Path,
    *,
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    root = Path(repo_root).resolve()
    train_path = Path(train_config_path).resolve()
    train = load_yaml_config(train_path)
    dataset_path = resolve_repo_path(train["dataset"]["config_path"], root)
    model_path = resolve_repo_path(train["model"]["config_path"], root)
    dataset = load_yaml_config(dataset_path)
    model = load_yaml_config(model_path)
    validate_multires_event_v2_configs(train, dataset, model)
    return train, dataset, model, dataset_path, model_path


def validate_multires_event_v2_configs(
    train: Mapping[str, Any],
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
) -> None:
    for label, payload in (("train", train), ("dataset", dataset), ("model", model)):
        if payload.get("route") != ROUTE:
            raise ValueError(f"{label} route must be {ROUTE!r}")
    mode = str(train.get("mode"))
    if mode not in MATCHED_MODES:
        raise ValueError(f"V2 matched mode must be one of {MATCHED_MODES}")

    base = _mapping(dataset.get("base"), "dataset.base")
    target = _mapping(dataset.get("target"), "dataset.target")
    expected_base = {
        "dataset_id": EXPECTED_BASE_DATASET_ID,
        "fingerprint": EXPECTED_BASE_FINGERPRINT,
        "dataset_manifest_sha256": EXPECTED_BASE_MANIFEST_SHA256,
        "sample_manifest_sha256": EXPECTED_BASE_SAMPLE_MANIFEST_SHA256,
        "subject_split_sha256": EXPECTED_SUBJECT_SPLIT_SHA256,
    }
    expected_target = {
        "dataset_id": EXPECTED_TARGET_DATASET_ID,
        "dataset_manifest_sha256": EXPECTED_TARGET_MANIFEST_SHA256,
        "sample_manifest_sha256": EXPECTED_TARGET_SAMPLE_MANIFEST_SHA256,
        "contract_bundle_hash": EXPECTED_CONTRACT_BUNDLE_HASH,
        "process_contract_sha256": EXPECTED_PROCESS_CONTRACT_SHA256,
        "emission_contract_sha256": EXPECTED_EMISSION_CONTRACT_SHA256,
        "projection_contract_sha256": EXPECTED_PROJECTION_CONTRACT_SHA256,
        "relation_contract_sha256": EXPECTED_RELATION_CONTRACT_SHA256,
        "sidecar_schema_sha256": EXPECTED_SIDECAR_SCHEMA_SHA256,
    }
    for key, expected in expected_base.items():
        if str(base.get(key)) != expected:
            raise ValueError(f"dataset.base.{key} differs from immutable V1 authority")
    for key, expected in expected_target.items():
        if str(target.get(key)) != expected:
            raise ValueError(f"dataset.target.{key} differs from full_r9 authority")
    if dict(dataset.get("expected_counts", {})) != EXPECTED_COUNTS:
        raise ValueError("V2 dataset expected_counts must match the persisted C4 rows")
    if str(train.get("lab_scale_artifact")) != EXPECTED_LAB_SCALE_ARTIFACT:
        raise ValueError("V2 training must use the frozen train-only lab scale artifact")
    if str(train.get("lab_scale_artifact_hash")) != EXPECTED_LAB_SCALE_ARTIFACT_HASH:
        raise ValueError("V2 training lab scale artifact hash differs from the frozen identity")
    if (
        str(train.get("standardized_primitive_scale_artifact"))
        != EXPECTED_STANDARDIZED_PRIMITIVE_SCALE_ARTIFACT
    ):
        raise ValueError("V2 evaluation must use the frozen train-only phi scale artifact")
    if (
        str(train.get("standardized_primitive_scale_artifact_hash"))
        != EXPECTED_STANDARDIZED_PRIMITIVE_SCALE_ARTIFACT_HASH
    ):
        raise ValueError("V2 phi scale artifact hash differs from the frozen identity")
    if str(train.get("promotion_metric_contract")) != EXPECTED_PROMOTION_METRIC_CONTRACT:
        raise ValueError("V2 training must use the frozen promotion metric contract")
    if (
        str(train.get("promotion_metric_contract_hash"))
        != EXPECTED_PROMOTION_METRIC_CONTRACT_HASH
    ):
        raise ValueError("V2 promotion metric contract hash differs from the frozen identity")

    objective = _mapping(train.get("objective"), "train.objective")
    required_objective = {
        "future_resolution": "M4",
        "future_blocks": 6,
        "core_fields": 29,
        "stochastic_primitive_factors": 414,
        "factor_composition": "joint_log_probability_sum",
        "anchor_reduction": "mean",
        "active_target_denominator": False,
        "deterministic_projection_loss": False,
        "h1_training_loss": False,
        "f24_training_loss": False,
        "auxiliary_training_loss": False,
        "family_weights": None,
    }
    for key, expected in required_objective.items():
        if objective.get(key) != expected:
            raise ValueError(f"objective.{key} must equal {expected!r}")

    if model.get("initialization") != "from_scratch":
        raise ValueError("V2 must initialize from scratch")
    if model.get("text_backbone") is not None or model.get("tokenizer") is not None:
        raise ValueError("V2 structured process model cannot use a text backbone/tokenizer")
    if model.get("role") != "primary":
        raise ValueError("V2 formal model config must declare role=primary")
    architecture = _mapping(model.get("architecture"), "model.architecture")
    for key, expected in EXPECTED_FORMAL_ARCHITECTURE.items():
        if int(architecture.get(key, -1)) != expected:
            raise ValueError(f"model.architecture.{key} must equal {expected}")
    configured_field_ids = tuple(int(value) for value in architecture.get("target_field_ids", ()))
    if configured_field_ids != REGISTERED_CORE_FIELD_IDS:
        raise ValueError(
            "formal model.architecture.target_field_ids must exactly match the ordered "
            "full_r9 registered_core_field_ids"
        )

    evaluation = _mapping(train.get("evaluation"), "train.evaluation")
    if evaluation.get("checkpoint_metric") != "joint_nll_subject_macro":
        raise ValueError("V2 checkpoints must be selected by subject-macro joint NLL")
    if evaluation.get("interval_anchor_policy") != "all_validation_anchors":
        raise ValueError("V2 checkpoint validation must use all persisted validation anchors")
    if evaluation.get("final_anchor_policy") != "all_validation_anchors":
        raise ValueError("V2 final validation must use all persisted validation anchors")
    if int(evaluation.get("interval_expected_samples", -1)) != EXPECTED_COUNTS["val"]:
        raise ValueError("V2 checkpoint validation must contain all 6,309 persisted anchors")
    if int(evaluation.get("final_expected_samples", -1)) != EXPECTED_COUNTS["val"]:
        raise ValueError("V2 final validation must contain all 6,309 persisted anchors")
    if evaluation.get("subject_macro") is not True:
        raise ValueError("V2 validation must report subject macro")
    if evaluation.get("no_ddp_padding_duplicates") is not True:
        raise ValueError("V2 validation cannot pad duplicate anchors across ranks")
    expected_free_running = str(train.get("run_name")) != "t4x2_multires_event_v2_smoke"
    if evaluation.get("free_running_final") is not expected_free_running:
        raise ValueError(
            "V2 free-running evaluation must be enabled for matched modes and disabled "
            "only for the route smoke run"
        )
    required_free_running = {
        "free_running_trajectories_per_anchor": 100,
        "free_running_trajectory_batch_size": 100,
        "free_running_crn_seed": 20260713,
    }
    for key, expected in required_free_running.items():
        if int(evaluation.get(key, -1)) != expected:
            raise ValueError(f"evaluation.{key} must equal {expected}")
    comparison = _mapping(train.get("comparison"), "train.comparison")
    required_comparison = {
        "primary_mode": "relational",
        "primary_training_order": ["relational"],
        "optional_ablation_modes": ["trajectory", "block"],
        "ablations_are_prerequisites": False,
        "paired_rows": "all_6309_persisted_validation_anchors",
        "paired_unit": "subject_id",
        "estimands": [
            "candidate_minus_control_subject_macro_joint_nll",
            "candidate_over_control_subject_macro_score_ratio",
        ],
        "bootstrap_repetitions": 2000,
        "bootstrap_seed": 20260713,
        "shared_subject_bootstrap_schedule": True,
        "physical_metrics_decision_role": "report_only",
        "coherence_required_rate": 1.0,
        "promotion_gate": "none_for_primary_training",
        "decision_authority": "relational_primary_then_optional_matched_ablations",
    }
    for key, expected in required_comparison.items():
        if comparison.get(key) != expected:
            raise ValueError(f"comparison.{key} must equal {expected!r}")

    training = _mapping(train.get("training"), "train.training")
    if int(training.get("required_world_size", -1)) != 2:
        raise ValueError("matched hosted V2 runs require two DDP ranks")
    if int(training.get("required_cuda_devices", -1)) != 2:
        raise ValueError("matched hosted V2 runs require two CUDA devices")
    if training.get("precision") != "fp16":
        raise ValueError("matched T4 runs require fp16 neural forward")
    if int(training.get("per_device_train_batch_size", -1)) != 32:
        raise ValueError("matched T4 runs freeze per-device train batch size at 32")
    if int(training.get("gradient_accumulation_steps", -1)) != 1:
        raise ValueError("matched T4 runs freeze gradient accumulation at one")
    if int(training.get("train_samples_per_epoch", -1)) != 3072:
        raise ValueError(
            "matched V2 runs freeze 3,072 uniform-subject replacement draws per epoch"
        )
    if (
        int(training["per_device_train_batch_size"])
        * int(training["required_world_size"])
        * int(training["gradient_accumulation_steps"])
        != 64
    ):
        raise ValueError("matched V2 runs require effective batch size 64")
    if int(training.get("per_device_eval_batch_size", -1)) != 32:
        raise ValueError(
            "matched T4 runs freeze eval batch size at 32; subject macro is row-wise"
        )
    required_scaler = {
        "grad_scaler_initial_scale": 32.0,
        "grad_scaler_growth_factor": 2.0,
        "grad_scaler_backoff_factor": 0.5,
        "grad_scaler_growth_interval": 1_000_000,
        "max_consecutive_scaler_skips": 0,
        "grad_scaler_overflow_policy": "fail_run_preserve_matched_rows",
    }
    for key, expected in required_scaler.items():
        observed = training.get(key)
        if isinstance(expected, float):
            valid = isinstance(observed, (int, float)) and float(observed) == expected
        elif isinstance(expected, int):
            valid = isinstance(observed, int) and not isinstance(observed, bool) and observed == expected
        else:
            valid = observed == expected
        if not valid:
            raise ValueError(f"training.{key} must equal {expected!r}")
    _validate_optimizer_contract(training)
    _validate_run_profile(train, training=training, evaluation=evaluation)
    if training.get("ddp_find_unused_parameters") is not False:
        raise ValueError("neutral relation parameters must use zero gradients, not DDP unused search")


def _validate_run_profile(
    train: Mapping[str, Any],
    *,
    training: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> None:
    run_name = str(train.get("run_name") or "")
    mode = str(train.get("mode") or "")
    smoke_name = "t4x2_multires_event_v2_smoke"
    if run_name == smoke_name:
        expected = {
            "mode": "relational",
            "max_steps": 2,
            "warmup_steps": 1,
            "eval_steps": 1,
            "save_steps": 1,
            "logging_steps": 1,
            "resume": False,
            "max_train_subjects": 16,
            "free_running_final": False,
            "output_basename": smoke_name,
        }
    else:
        if mode not in MATCHED_MODES:
            raise ValueError("formal V2 profile mode is invalid")
        expected_name = f"t4x2_multires_event_v2_{mode}"
        if run_name != expected_name:
            raise ValueError(
                "formal V2 run_name must exactly identify its declared mode: "
                f"expected {expected_name!r}"
            )
        expected = {
            "mode": mode,
            "max_steps": 4000,
            "warmup_steps": 400,
            "eval_steps": 250,
            "save_steps": 500,
            "initial_checkpoint_step": 2,
            "logging_steps": 100,
            "resume": True,
            "max_train_subjects": None,
            "free_running_final": True,
            "output_basename": expected_name,
        }
    if mode != expected["mode"]:
        raise ValueError(f"V2 profile mode must equal {expected['mode']!r}")
    for key in ("max_steps", "warmup_steps", "eval_steps", "save_steps", "logging_steps"):
        observed = training.get(key)
        if isinstance(observed, bool) or not isinstance(observed, int) or observed != expected[key]:
            raise ValueError(f"training.{key} must equal {expected[key]!r} for {run_name}")
    if run_name != smoke_name:
        observed_initial = training.get("initial_checkpoint_step")
        if (
            isinstance(observed_initial, bool)
            or not isinstance(observed_initial, int)
            or observed_initial != expected["initial_checkpoint_step"]
        ):
            raise ValueError(
                "training.initial_checkpoint_step must equal 2 for the primary run"
            )
    if training.get("resume") is not expected["resume"]:
        raise ValueError(f"training.resume must equal {expected['resume']!r} for {run_name}")
    if training.get("max_train_subjects") != expected["max_train_subjects"]:
        raise ValueError(
            "training.max_train_subjects must equal "
            f"{expected['max_train_subjects']!r} for {run_name}"
        )
    if evaluation.get("free_running_final") is not expected["free_running_final"]:
        raise ValueError(
            "evaluation.free_running_final must equal "
            f"{expected['free_running_final']!r} for {run_name}"
        )
    outputs = _mapping(train.get("outputs"), "train.outputs")
    output_dir = Path(str(outputs.get("output_dir") or ""))
    metrics_path = Path(str(outputs.get("metrics_jsonl") or ""))
    if output_dir.name != expected["output_basename"]:
        raise ValueError(
            f"outputs.output_dir basename must equal {expected['output_basename']!r}"
        )
    if metrics_path.name != "metrics.jsonl" or metrics_path.parent.name != output_dir.name:
        raise ValueError("outputs.metrics_jsonl must be metrics.jsonl inside the declared run root")


def _validate_optimizer_contract(training: Mapping[str, Any]) -> None:
    if "max_grad_norm" in training:
        raise ValueError(
            "training.max_grad_norm is forbidden: the V2 optimizer contract disables "
            "all gradient clipping"
        )
    for key, expected in EXPECTED_OPTIMIZER_CONTRACT.items():
        observed = training.get(key)
        if isinstance(expected, bool):
            valid = isinstance(observed, bool) and observed is expected
        elif isinstance(expected, float):
            valid = (
                isinstance(observed, (int, float))
                and not isinstance(observed, bool)
                and float(observed) == expected
            )
        elif isinstance(expected, list):
            valid = (
                isinstance(observed, list)
                and len(observed) == len(expected)
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and float(value) == target
                    for value, target in zip(observed, expected, strict=True)
                )
            )
        else:
            valid = observed == expected
        if not valid:
            raise ValueError(f"training.{key} must equal {expected!r}")
    learning_rate = float(training.get("learning_rate", math.nan))
    weight_decay = float(training.get("weight_decay", math.nan))
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("training.learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("training.weight_decay must be finite and nonnegative")


def matched_design_signature(
    train: Mapping[str, Any],
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
) -> str:
    """Hash every matched-run factor except the declared access/bias mode and paths."""

    payload = copy.deepcopy(dict(train))
    payload.pop("mode", None)
    payload.pop("run_name", None)
    payload.pop("outputs", None)
    return sha256_payload({"train": payload, "dataset": dataset, "model": model})


def build_multires_event_v2_model(model: Mapping[str, Any], *, mode: str) -> MultiResolutionEventV2Model:
    architecture = dict(_mapping(model.get("architecture"), "model.architecture"))
    architecture.update(
        mode=mode,
        primitive_head_dims=V2_PRIMITIVE_HEAD_DIMS,
        primitive_feedback_dims=V2_PRIMITIVE_FEEDBACK_DIMS,
    )
    built = MultiResolutionEventV2Model(MultiResolutionEventV2Config.from_mapping(architecture))
    if _is_formal_model_architecture(architecture):
        validate_formal_model_parameter_count(built)
    return built


def build_multires_event_v2_optimizer(
    model: torch.nn.Module,
    training: Mapping[str, Any],
) -> torch.optim.AdamW:
    """Build the one frozen AdamW implementation used by probe and formal runs."""

    _validate_optimizer_contract(training)
    parameters = [
        parameter
        for parameter in _unwrapped_model(model).parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("V2 optimizer requires at least one trainable parameter")
    betas = tuple(float(value) for value in training["adamw_betas"])
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        betas=betas,
        eps=float(training["adamw_eps"]),
        weight_decay=float(training["weight_decay"]),
        amsgrad=bool(training["adamw_amsgrad"]),
        maximize=bool(training["adamw_maximize"]),
        foreach=bool(training["adamw_foreach"]),
        capturable=False,
        differentiable=False,
        fused=bool(training["adamw_fused"]),
    )
    _validate_built_optimizer_contract(optimizer)
    return optimizer


def _validate_built_optimizer_contract(
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("V2 optimizer must be torch.optim.AdamW")
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("V2 optimizer contract requires exactly one parameter group")
    group = optimizer.param_groups[0]
    defaults = optimizer.defaults
    expected = EXPECTED_OPTIMIZER_CONTRACT
    observed = {
        "learning_rate": float(defaults.get("lr", math.nan)),
        "weight_decay": float(defaults.get("weight_decay", math.nan)),
        "adamw_betas": [float(value) for value in defaults.get("betas", ())],
        "adamw_eps": float(defaults.get("eps", math.nan)),
        "adamw_amsgrad": defaults.get("amsgrad"),
        "adamw_maximize": defaults.get("maximize"),
        "adamw_foreach": defaults.get("foreach"),
        "adamw_fused": defaults.get("fused"),
    }
    for key, value in observed.items():
        if value != expected[key]:
            raise RuntimeError(f"built V2 AdamW {key} differs from the frozen contract")
    group_contract = {
        "weight_decay": float(group.get("weight_decay", math.nan)),
        "adamw_betas": [float(value) for value in group.get("betas", ())],
        "adamw_eps": float(group.get("eps", math.nan)),
        "adamw_amsgrad": group.get("amsgrad"),
        "adamw_maximize": group.get("maximize"),
        "adamw_foreach": group.get("foreach"),
        "adamw_fused": group.get("fused"),
    }
    for key, value in group_contract.items():
        if value != expected[key]:
            raise RuntimeError(f"V2 AdamW parameter group {key} differs from the frozen contract")
    initial_lr = float(group.get("initial_lr", expected["learning_rate"]))
    if initial_lr != expected["learning_rate"]:
        raise RuntimeError("V2 AdamW parameter-group initial LR differs from the frozen contract")
    current_lr = float(group.get("lr", math.nan))
    if not math.isfinite(current_lr) or current_lr <= 0.0:
        raise RuntimeError("V2 AdamW current learning rate must be finite and positive")
    return {
        "optimizer": "AdamW",
        "parameter_group_count": 1,
        "base_learning_rate": float(expected["learning_rate"]),
        "current_learning_rate": current_lr,
        "weight_decay": float(expected["weight_decay"]),
        "adamw_betas": list(expected["adamw_betas"]),
        "adamw_eps": float(expected["adamw_eps"]),
        "adamw_amsgrad": bool(expected["adamw_amsgrad"]),
        "adamw_maximize": bool(expected["adamw_maximize"]),
        "adamw_foreach": bool(expected["adamw_foreach"]),
        "adamw_fused": bool(expected["adamw_fused"]),
    }


def _optimizer_step_health_payload(
    optimizer: torch.optim.Optimizer,
    gradient_health: Mapping[str, Any],
    optimizer_state_health: Mapping[str, Any],
    *,
    training: Mapping[str, Any],
) -> dict[str, Any]:
    optimizer_configuration = _validate_built_optimizer_contract(optimizer)
    gradient_audit_seconds = float(gradient_health.get("audit_wall_seconds", math.nan))
    state_audit_seconds = float(optimizer_state_health.get("audit_wall_seconds", math.nan))
    audit_wall_seconds = gradient_audit_seconds + state_audit_seconds
    expected_optimizer_step = int(optimizer_state_health.get("expected_optimizer_step", -1))
    observed_step_min = float(
        optimizer_state_health.get("observed_optimizer_step_min", math.nan)
    )
    observed_step_max = float(
        optimizer_state_health.get("observed_optimizer_step_max", math.nan)
    )
    expected_learning_rate = _frozen_optimizer_learning_rate(
        training, completed_optimizer_steps=expected_optimizer_step - 1
    )
    learning_rate_used = float(optimizer_configuration["current_learning_rate"])
    if (
        not math.isfinite(gradient_audit_seconds)
        or gradient_audit_seconds <= 0.0
        or not math.isfinite(state_audit_seconds)
        or state_audit_seconds <= 0.0
    ):
        raise RuntimeError("V2 optimizer audit wall time must be finite and positive")
    if (
        expected_optimizer_step < 1
        or observed_step_min != float(expected_optimizer_step)
        or observed_step_max != float(expected_optimizer_step)
        or learning_rate_used != expected_learning_rate
    ):
        raise RuntimeError("V2 optimizer step/LR health differs from the frozen schedule")
    return {
        "optimizer_contract_version": OPTIMIZER_CONTRACT_VERSION,
        "loss_reduction": RAW_JOINT_NLL_REDUCTION,
        "expected_optimizer_step": expected_optimizer_step,
        "observed_optimizer_step_min": observed_step_min,
        "observed_optimizer_step_max": observed_step_max,
        "expected_learning_rate_used": expected_learning_rate,
        "learning_rate_used": learning_rate_used,
        "optimizer_audit_wall_seconds": audit_wall_seconds,
        "gradient_health": dict(gradient_health),
        "optimizer_state_health": dict(optimizer_state_health),
    }


def _frozen_optimizer_learning_rate(
    training: Mapping[str, Any],
    *,
    completed_optimizer_steps: int,
) -> float:
    if (
        isinstance(completed_optimizer_steps, bool)
        or not isinstance(completed_optimizer_steps, int)
        or completed_optimizer_steps < 0
    ):
        raise ValueError("completed_optimizer_steps must be a nonnegative integer")
    warmup = int(training["warmup_steps"])
    total = int(training["max_steps"])
    if completed_optimizer_steps > total:
        raise ValueError("completed optimizer step exceeds the frozen schedule")
    if warmup > 0 and completed_optimizer_steps < warmup:
        factor = float(completed_optimizer_steps + 1) / float(warmup)
    else:
        remaining = max(0, total - completed_optimizer_steps)
        factor = float(remaining) / float(max(1, total - warmup))
    return float(training["learning_rate"]) * factor


def _validate_resume_optimizer_alignment(
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    training: Mapping[str, Any],
    *,
    global_step: int,
) -> dict[str, Any]:
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise ValueError("resume global_step must be a nonnegative integer")
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("V2 resume requires exactly one AdamW parameter group")
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    ]
    observed_steps: list[float] = []
    if global_step == 0:
        if len(optimizer.state) != 0:
            raise RuntimeError("fresh V2 step zero requires empty AdamW state")
    else:
        if len(optimizer.state) != len(parameters):
            raise RuntimeError("resumed V2 AdamW state is incomplete")
        for parameter in parameters:
            state = optimizer.state.get(parameter)
            step = state.get("step") if isinstance(state, Mapping) else None
            if not isinstance(step, torch.Tensor) or step.numel() != 1:
                raise RuntimeError("resumed V2 AdamW state lacks a scalar step")
            value = float(step.detach().cpu().item())
            if not math.isfinite(value) or value != float(global_step):
                raise RuntimeError("resumed V2 AdamW state steps must exactly equal global_step")
            observed_steps.append(value)
    if int(getattr(scheduler, "last_epoch", -1)) != global_step:
        raise RuntimeError("resumed V2 scheduler.last_epoch must exactly equal global_step")
    expected_lr = _frozen_optimizer_learning_rate(
        training, completed_optimizer_steps=global_step
    )
    observed_lr = float(optimizer.param_groups[0].get("lr", math.nan))
    if observed_lr != expected_lr:
        raise RuntimeError("resumed V2 optimizer LR differs from the frozen lambda schedule")
    return {
        "global_step": global_step,
        "optimizer_state_entries": len(optimizer.state),
        "expected_optimizer_step": global_step,
        "observed_optimizer_step_min": min(observed_steps) if observed_steps else None,
        "observed_optimizer_step_max": max(observed_steps) if observed_steps else None,
        "scheduler_last_epoch": int(scheduler.last_epoch),
        "expected_learning_rate": expected_lr,
        "observed_learning_rate": observed_lr,
    }


def raw_414_factor_joint_nll_batch_mean(
    loss_result: Mapping[str, Any],
) -> torch.Tensor:
    """Reconstruct the optimizer scalar directly from all 414 raw log factors."""

    if int(loss_result.get("primitive_count", -1)) != 414:
        raise ValueError("V2 optimizer loss requires exactly 414 primitive factors")
    primitive_log_prob = loss_result.get("primitive_log_prob")
    if (
        not isinstance(primitive_log_prob, torch.Tensor)
        or primitive_log_prob.ndim != 2
        or int(primitive_log_prob.shape[0]) < 1
        or int(primitive_log_prob.shape[1]) != 414
    ):
        raise ValueError("V2 optimizer loss requires a nonempty [batch,414] log-probability bank")
    return -primitive_log_prob.sum(dim=-1).mean()


def _validated_optimizer_loss(
    loss_result: Mapping[str, Any],
    *,
    expected_local_batch: int,
) -> torch.Tensor:
    per_sample = loss_result.get("per_sample_nll")
    if (
        not isinstance(per_sample, torch.Tensor)
        or per_sample.ndim != 1
        or int(per_sample.numel()) != int(expected_local_batch)
        or int(expected_local_batch) < 1
    ):
        raise ValueError("V2 per-sample NLL count must equal the nonempty local batch")
    declared_loss = loss_result.get("loss")
    primitive_log_prob = loss_result.get("primitive_log_prob")
    if not isinstance(declared_loss, torch.Tensor) or declared_loss.numel() != 1:
        raise ValueError("V2 optimizer loss result must expose one scalar loss")
    loss = raw_414_factor_joint_nll_batch_mean(loss_result)
    if not isinstance(primitive_log_prob, torch.Tensor):
        raise AssertionError("validated primitive bank unexpectedly disappeared")
    if int(primitive_log_prob.shape[0]) != int(expected_local_batch):
        raise ValueError(
            "V2 primitive log-probability batch must equal the exact local batch"
        )
    reconstructed_per_sample = -primitive_log_prob.sum(dim=-1)
    finite_and_equal = torch.stack(
        (
            torch.isfinite(loss),
            torch.isfinite(declared_loss).all(),
            torch.isfinite(per_sample).all(),
            per_sample.eq(reconstructed_per_sample).all(),
            declared_loss.reshape(()).eq(per_sample.mean()),
            loss.eq(per_sample.mean()),
        )
    ).all()
    if not bool(finite_and_equal.detach().cpu().item()):
        raise FloatingPointError(
            "V2 optimizer loss/per-sample NLL must be finite and algebraically "
            "identical to the raw 414-factor joint NLL"
        )
    return loss


def _validate_optimizer_health_event(
    row: Mapping[str, Any],
    *,
    training: Mapping[str, Any],
) -> int:
    step = row.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("V2 optimizer health step must be a positive integer")
    expected_lr = _frozen_optimizer_learning_rate(
        training, completed_optimizer_steps=step - 1
    )
    gradient = _mapping(row.get("gradient_health"), "optimizer gradient health")
    state = _mapping(row.get("optimizer_state_health"), "optimizer state health")
    configuration = _mapping(
        state.get("optimizer_configuration"), "optimizer configuration health"
    )
    trainable = int(gradient.get("trainable_parameter_tensors", -1))
    audit_seconds = float(row.get("optimizer_audit_wall_seconds", math.nan))
    gradient_audit = float(gradient.get("audit_wall_seconds", math.nan))
    state_audit = float(state.get("audit_wall_seconds", math.nan))
    probe_before = float(state.get("probe_value_before", math.nan))
    probe_after = float(state.get("probe_value_after", math.nan))
    probe_changed = state.get("probe_parameter_changed")
    expected = EXPECTED_OPTIMIZER_CONTRACT
    valid = (
        row.get("event") == "v2_optimizer_health"
        and row.get("optimizer_contract_version") == OPTIMIZER_CONTRACT_VERSION
        and row.get("loss_reduction") == RAW_JOINT_NLL_REDUCTION
        and int(row.get("local_anchors", -1)) == 32
        and int(row.get("world_size", -1)) == 2
        and int(row.get("global_anchors", -1)) == 64
        and int(row.get("expected_optimizer_step", -1)) == step
        and float(row.get("observed_optimizer_step_min", math.nan)) == float(step)
        and float(row.get("observed_optimizer_step_max", math.nan)) == float(step)
        and float(row.get("expected_learning_rate_used", math.nan)) == expected_lr
        and float(row.get("learning_rate_used", math.nan)) == expected_lr
        and expected_lr > 0.0
        and row.get("scaler_scale_before") == 32.0
        and row.get("scaler_scale_after") == 32.0
        and int(row.get("scaler_skipped_steps", -1)) == 0
        and "max_grad_norm" not in row
        and math.isfinite(audit_seconds)
        and audit_seconds > 0.0
        and math.isfinite(gradient_audit)
        and gradient_audit > 0.0
        and math.isfinite(state_audit)
        and state_audit > 0.0
        and math.isclose(
            audit_seconds,
            gradient_audit + state_audit,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        and gradient.get("optimizer_contract_version") == OPTIMIZER_CONTRACT_VERSION
        and trainable > 0
        and int(gradient.get("gradient_tensors", -1)) == trainable
        and int(gradient.get("missing_gradient_tensors", -1)) == 0
        and gradient.get("all_gradients_finite") is True
        and math.isfinite(float(gradient.get("global_l2_norm", math.nan)))
        and float(gradient.get("global_l2_norm", math.nan)) > 0.0
        and gradient.get("global_l2_positive") is True
        and gradient.get("gradient_clipping") == "disabled"
        and gradient.get("gradient_modified_after_unscale") is False
        and math.isfinite(float(gradient.get("probe_gradient_abs", math.nan)))
        and float(gradient.get("probe_gradient_abs", math.nan)) > 0.0
        and state.get("optimizer_contract_version") == OPTIMIZER_CONTRACT_VERSION
        and int(state.get("trainable_parameter_tensors", -1)) == trainable
        and int(state.get("optimizer_state_entries", -1)) == trainable
        and state.get("state_complete") is True
        and int(state.get("expected_optimizer_step", -1)) == step
        and float(state.get("observed_optimizer_step_min", math.nan)) == float(step)
        and float(state.get("observed_optimizer_step_max", math.nan)) == float(step)
        and state.get("state_steps_complete_equal_expected") is True
        and state.get("parameters_finite") is True
        and state.get("exp_avg_finite") is True
        and state.get("exp_avg_sq_finite") is True
        and state.get("exp_avg_sq_nonnegative") is True
        and math.isfinite(float(state.get("exp_avg_sq_minimum", math.nan)))
        and float(state.get("exp_avg_sq_minimum", math.nan)) >= 0.0
        and isinstance(probe_changed, bool)
        and math.isfinite(probe_before)
        and math.isfinite(probe_after)
        and probe_changed is (probe_before != probe_after)
        and state.get("optimizer_updated") is True
        and configuration.get("optimizer") == "AdamW"
        and int(configuration.get("parameter_group_count", -1)) == 1
        and float(configuration.get("base_learning_rate", math.nan))
        == float(expected["learning_rate"])
        and float(configuration.get("current_learning_rate", math.nan)) == expected_lr
        and float(configuration.get("weight_decay", math.nan))
        == float(expected["weight_decay"])
        and list(configuration.get("adamw_betas", ())) == expected["adamw_betas"]
        and float(configuration.get("adamw_eps", math.nan))
        == float(expected["adamw_eps"])
        and configuration.get("adamw_amsgrad") is expected["adamw_amsgrad"]
        and configuration.get("adamw_maximize") is expected["adamw_maximize"]
        and configuration.get("adamw_foreach") is expected["adamw_foreach"]
        and configuration.get("adamw_fused") is expected["adamw_fused"]
    )
    if not valid:
        raise ValueError(f"V2 optimizer health contract failed at step {step}")
    return step


def summarize_optimizer_health_metrics(
    metrics_path: str | Path,
    *,
    training: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(metrics_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"V2 optimizer metrics file is absent: {path}")
    expected_steps = int(training["max_steps"])
    raw_rows = 0
    skip_events = 0
    canonical: dict[int, Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"V2 metrics JSON is invalid at line {line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ValueError("V2 metrics rows must be JSON objects")
            if row.get("event") == "v2_grad_scaler_skip":
                skip_events += 1
            if row.get("event") != "v2_optimizer_health":
                continue
            step = _validate_optimizer_health_event(row, training=training)
            raw_rows += 1
            canonical[step] = row
    required_steps = list(range(1, expected_steps + 1))
    if skip_events != 0:
        raise ValueError("V2 optimizer metrics contain a forbidden GradScaler skip")
    if sorted(canonical) != required_steps:
        raise ValueError("V2 optimizer health canonical steps are incomplete")
    canonical_rows = [canonical[step] for step in required_steps]
    return {
        "schema_version": OPTIMIZER_HEALTH_SUMMARY_SCHEMA,
        "status": "COMPLETE",
        "metrics_path": "metrics.jsonl",
        "metrics_sha256": sha256_file(path),
        "optimizer_contract_version": OPTIMIZER_CONTRACT_VERSION,
        "loss_reduction": RAW_JOINT_NLL_REDUCTION,
        "gradient_clipping": "disabled",
        "expected_steps": expected_steps,
        "raw_health_rows": raw_rows,
        "canonical_steps": len(canonical_rows),
        "replayed_rows": raw_rows - len(canonical_rows),
        "first_step": required_steps[0],
        "last_step": required_steps[-1],
        "canonical_step_sequence_sha256": sha256_payload(required_steps),
        "canonical_health_rows_sha256": sha256_payload(canonical_rows),
        "scaler_skipped_events": 0,
    }


def validate_optimizer_health_summary(
    summary_path: str | Path,
    metrics_path: str | Path,
    *,
    training: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(summary_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"V2 optimizer health summary is absent: {path}")
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = summarize_optimizer_health_metrics(metrics_path, training=training)
    if observed != expected:
        raise ValueError("V2 optimizer health summary differs from metrics truth")
    return dict(observed)


def validate_formal_target_field_order(
    model: Mapping[str, Any],
    contract: MultiresEventV2Contract,
) -> None:
    """Bind the formal decoder field axis to the mounted full_r9 contract."""

    architecture = _mapping(model.get("architecture"), "model.architecture")
    configured = tuple(int(value) for value in architecture.get("target_field_ids", ()))
    registered = tuple(int(value) for value in contract.registered_core_field_ids)
    if configured != registered:
        raise ValueError(
            "formal model target_field_ids must exactly match full_r9 "
            "contract.registered_core_field_ids in order: "
            f"configured={configured}, registered={registered}"
        )


def validate_formal_model_parameter_count(model: torch.nn.Module) -> int:
    """Reject any formal model whose total parameterization drifted from the freeze."""

    observed = sum(parameter.numel() for parameter in model.parameters())
    if observed != EXPECTED_FORMAL_MODEL_PARAMETER_COUNT:
        raise ValueError(
            "formal V2 model parameter count differs from the frozen matched design: "
            f"{observed:,} != {EXPECTED_FORMAL_MODEL_PARAMETER_COUNT:,}"
        )
    return observed


def _is_formal_model_architecture(architecture: Mapping[str, Any]) -> bool:
    try:
        return all(
            int(architecture.get(key, -1)) == expected
            for key, expected in EXPECTED_FORMAL_ARCHITECTURE.items()
        )
    except (TypeError, ValueError):
        return False


def build_multires_event_v2_runtime(
    train: Mapping[str, Any],
    dataset: Mapping[str, Any],
    *,
    repo_root: str | Path,
    rank: int,
    world_size: int,
    phase: str,
) -> MultiresEventV2Runtime:
    """Join immutable V1 inputs to frozen full_r9 targets without rebuilding either split."""

    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("V2 runtime requires torch") from exc
    if phase not in {"interval", "final"}:
        raise ValueError("V2 runtime phase must be interval or final")
    root = Path(repo_root).resolve()
    base_config = _mapping(dataset.get("base"), "dataset.base")
    target_config = _mapping(dataset.get("target"), "dataset.target")
    base_root = resolve_repo_path(str(base_config["root"]), root)
    target_root = resolve_repo_path(str(target_config["root"]), root)
    supervision_path = resolve_repo_path(str(dataset["supervision_path"]), root)
    normalization_path = resolve_repo_path(str(dataset["normalization"]["path"]), root)
    _verify_artifact_files(base_root, target_root, supervision_path, dataset)

    evaluation = _mapping(train.get("evaluation"), "train.evaluation")
    training = _mapping(train.get("training"), "train.training")
    base_runtime_training = dict(training)
    # The V1 runtime is used only to recover the immutable datasets and fitted
    # input normalization; its legacy evaluator insists on batch size one.
    # V2 replaces that loader below with a row-preserving batch-32 evaluator.
    base_runtime_training["per_device_eval_batch_size"] = 1
    base_runtime_config = {
        "seed": int(train["seed"]),
        "data": {
            "dataset_id": EXPECTED_BASE_DATASET_ID,
            "dataset_fingerprint": EXPECTED_BASE_FINGERPRINT,
            "source_fingerprint": str(base_config["source_fingerprint"]),
            "expected_counts": EXPECTED_COUNTS,
        },
        "loader": dict(_mapping(dataset.get("loader"), "dataset.loader")),
        "normalization": dict(_mapping(dataset.get("normalization"), "dataset.normalization")),
        "preflight": dict(_mapping(dataset.get("preflight"), "dataset.preflight")),
        "training": base_runtime_training,
        "evaluation": {
            # V1 interval semantics are one fixed anchor per subject. V2 owns
            # checkpoint sampling and requires all 6,309 validation anchors,
            # so request the immutable V1 eval dataset through its final route
            # and replace only its sampler below.
            "phase": "final",
            "final_expected_samples": EXPECTED_COUNTS["val"],
        },
    }
    base_runtime = build_v1_runtime(
        base_runtime_config,
        base_root,
        supervision_path,
        normalization_path,
        rank,
        world_size,
    )
    if not isinstance(base_runtime.train_dataset, MultiresEventDataset):
        raise AssertionError("V2 runtime did not receive the immutable V1 map dataset")
    contract = MultiresEventV2Contract.from_dataset_root(target_root)
    runtime_model_path = resolve_repo_path(
        str(_mapping(train.get("model"), "train.model")["config_path"]),
        root,
    )
    runtime_model_config = load_yaml_config(runtime_model_path)
    validate_formal_target_field_order(runtime_model_config, contract)
    validate_emission_registry_head_contract(contract.emission_registry)
    lab_scale_path = resolve_repo_path(str(train.get("lab_scale_artifact", "")), root)
    lab_scale_metadata = load_lab_scale_artifact(
        lab_scale_path,
        expected_content_sha256=str(train.get("lab_scale_artifact_hash", "")),
        contract=contract,
    )
    standardized_scale_path = resolve_repo_path(
        str(train.get("standardized_primitive_scale_artifact", "")), root
    )
    load_standardized_primitive_scale_artifact(
        standardized_scale_path,
        expected_content_sha256=str(
            train.get("standardized_primitive_scale_artifact_hash", "")
        ),
        contract=contract,
        expected_lab_scale_artifact_hash=str(train.get("lab_scale_artifact_hash", "")),
    )
    promotion_contract_path = resolve_repo_path(
        str(train.get("promotion_metric_contract", "")), root
    )
    load_promotion_metric_contract(
        promotion_contract_path,
        expected_sha256=str(train.get("promotion_metric_contract_hash", "")),
        data_contract=contract,
    )
    train_dataset = MultiresEventV2Dataset(
        base_runtime.train_dataset,
        target_root,
        contract=contract,
        cache_shards=int(dataset["loader"].get("cache_shards", 1)),
        strict=True,
        verify_shard_hashes=bool(dataset["preflight"].get("verify_target_shard_sha256", False)),
    )
    eval_dataset = MultiresEventV2Dataset(
        base_runtime.eval_dataset,
        target_root,
        contract=contract,
        cache_shards=int(dataset["loader"].get("cache_shards", 1)),
        strict=True,
        verify_shard_hashes=bool(dataset["preflight"].get("verify_target_shard_sha256", False)),
    )
    if (len(train_dataset), len(eval_dataset)) != (EXPECTED_COUNTS["train"], EXPECTED_COUNTS["val"]):
        raise ValueError("V1/V2 joined split counts differ from the persisted manifest")
    # Force strict content-hash joins before a hosted optimizer is created.
    for view in (train_dataset, eval_dataset):
        view[0]
        view[len(view) - 1]

    seed = int(train["seed"])
    max_train_subjects = _optional_positive_int(training.get("max_train_subjects"))
    max_eval_subjects = _optional_positive_int(training.get("max_eval_subjects"))
    train_sampler = SubjectAnchorDistributedSampler(
        train_dataset,
        rank=rank,
        world_size=world_size,
        seed=seed,
        mode="subject_uniform_replacement",
        shuffle=True,
        pad_to_world_size=False,
        require_even_divisible=True,
        max_subjects=max_train_subjects,
        max_samples=int(training["train_samples_per_epoch"]),
    )
    eval_mode = "anchor_uniform"
    expected_eval = int(
        evaluation["interval_expected_samples"]
        if phase == "interval"
        else evaluation["final_expected_samples"]
    )
    eval_sampler = SubjectAnchorDistributedSampler(
        eval_dataset,
        rank=rank,
        world_size=world_size,
        seed=seed,
        mode=eval_mode,
        shuffle=False,
        pad_to_world_size=False,
        require_even_divisible=False,
        max_subjects=max_eval_subjects,
        max_samples=expected_eval,
    )
    if eval_sampler.global_sample_count != expected_eval:
        raise ValueError(
            f"V2 {phase} sampler resolved {eval_sampler.global_sample_count}, expected {expected_eval}"
        )
    supervision = SupervisionContract.from_json(supervision_path)
    collator = _LabScaleBoundCollator(
        MultiresEventV2Collator(
            contract=contract,
            supervision=supervision,
            templates=base_runtime.train_dataset.templates,
            normalization=base_runtime.normalization,
        ),
        lab_scale_metadata,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["per_device_train_batch_size"]),
        sampler=train_sampler,
        num_workers=0,
        pin_memory=bool(dataset["loader"].get("pin_memory", True)),
        persistent_workers=False,
        drop_last=False,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(training["per_device_eval_batch_size"]),
        sampler=eval_sampler,
        num_workers=0,
        pin_memory=bool(dataset["loader"].get("pin_memory", True)),
        persistent_workers=False,
        drop_last=False,
        collate_fn=collator,
    )
    identity = {
        "base_dataset_id": EXPECTED_BASE_DATASET_ID,
        "base_fingerprint": EXPECTED_BASE_FINGERPRINT,
        "base_dataset_manifest_sha256": EXPECTED_BASE_MANIFEST_SHA256,
        "target_dataset_id": EXPECTED_TARGET_DATASET_ID,
        "dataset_id": EXPECTED_TARGET_DATASET_ID,
        "target_dataset_manifest_sha256": EXPECTED_TARGET_MANIFEST_SHA256,
        "contract_bundle_hash": contract.contract_bundle_hash,
        "process_contract_sha256": contract.contract_hashes["process"],
        "emission_contract_sha256": contract.contract_hashes["emission"],
        "projection_contract_sha256": contract.contract_hashes["projection"],
        "relation_contract_sha256": contract.contract_hashes["relation"],
        "sidecar_schema_sha256": contract.contract_hashes["sidecar_schema"],
        "counts": dict(EXPECTED_COUNTS),
        "phase": phase,
        "train_subjects": len(set(train_dataset.subject_ids)),
        "validation_subjects": len(set(eval_dataset.subject_ids)),
        "enabled_factors": 414,
        "normalization_artifact": str(normalization_path),
        "normalization_artifact_sha256": sha256_file(normalization_path),
        "input_normalization_sha256": sha256_file(normalization_path),
        "lab_scale_artifact": str(lab_scale_path),
        "lab_scale_artifact_hash": str(train["lab_scale_artifact_hash"]),
        "lab_scale_artifact_sha256": str(train["lab_scale_artifact_hash"]),
        "standardized_primitive_scale_artifact": str(standardized_scale_path),
        "standardized_primitive_scale_artifact_hash": str(
            train["standardized_primitive_scale_artifact_hash"]
        ),
        "standardized_primitive_scale_sha256": str(
            train["standardized_primitive_scale_artifact_hash"]
        ),
        "promotion_metric_contract": str(promotion_contract_path),
        "promotion_metric_contract_sha256": str(
            train["promotion_metric_contract_hash"]
        ),
    }
    return MultiresEventV2Runtime(
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_sampler=train_sampler,
        eval_sampler=eval_sampler,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        contract=contract,
        normalization=base_runtime.normalization,
        identity=identity,
    )


def _evaluation_contract_identity(
    runtime: MultiresEventV2Runtime,
    *,
    source_identity: Mapping[str, Any] | None = None,
    identity_hashes: Mapping[str, str] | None = None,
    selected_model_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the row-level identity shared by teacher and free trajectories."""

    identity = runtime.identity
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
    )
    result = {key: str(identity.get(key) or "") for key in keys}
    missing = [key for key, value in result.items() if not value]
    if missing:
        raise ValueError(f"V2 evaluation identity is incomplete: {missing}")
    semantic_runtime_sha = str(identity.get("semantic_runtime_identity_sha256") or "")
    if semantic_runtime_sha:
        if not _is_sha256(semantic_runtime_sha):
            raise ValueError("V2 semantic runtime identity is not SHA-256")
        result["semantic_runtime_identity_sha256"] = semantic_runtime_sha
    final_parts = (source_identity, identity_hashes, selected_model_identity)
    if any(part is not None for part in final_parts):
        if any(part is None for part in final_parts):
            raise ValueError(
                "V2 final evaluation identity requires source, run, and selected-model identities"
            )
        assert source_identity is not None
        assert identity_hashes is not None
        assert selected_model_identity is not None
        source_sha = sha256_payload(source_identity)
        expected_source = {
            "source_tree": str(source_identity.get("source_tree_sha256") or ""),
            "source_identity": source_sha,
            "git_commit": str(source_identity.get("git_commit") or ""),
            "git_head_tree": str(source_identity.get("git_head_tree") or ""),
        }
        mismatched = {
            key: {"source": value, "run": identity_hashes.get(key)}
            for key, value in expected_source.items()
            if not value or str(identity_hashes.get(key) or "") != value
        }
        if mismatched:
            raise ValueError(
                f"V2 source identity is not bound to the frozen run identity: {mismatched}"
            )
        selected_step = selected_model_identity.get("selected_checkpoint_step")
        selected_sha = str(
            selected_model_identity.get("selected_checkpoint_model_sha256") or ""
        )
        if (
            isinstance(selected_step, bool)
            or not isinstance(selected_step, int)
            or selected_step < 1
            or not _is_sha256(selected_sha)
        ):
            raise ValueError("V2 selected-model identity is incomplete")
        matched_design = str(identity_hashes.get("matched_design") or "")
        if not _is_sha256(matched_design):
            raise ValueError("V2 matched-design identity is not SHA-256")
        result.update(
            {
                "source_tree_sha256": expected_source["source_tree"],
                "source_identity_sha256": source_sha,
                "git_commit": expected_source["git_commit"],
                "git_head_tree": expected_source["git_head_tree"],
                "matched_design_signature": matched_design,
                "selected_checkpoint_step": selected_step,
                "selected_checkpoint_model_sha256": selected_sha,
            }
        )
    return result


def project_multires_event_v2_capacity_runtime(
    training: Mapping[str, Any],
    *,
    optimizer_step_seconds: tuple[float, ...],
    teacher_probe_seconds: float,
    free_running_probe_seconds: float,
) -> dict[str, Any]:
    """Conservatively project the complete formal mode from the frozen probe.

    The slower of the two real optimizer steps is extrapolated.  Teacher-forced
    time includes every 250-step checkpoint evaluation plus the final selected-
    checkpoint pass; free-running time remains exactly one 6,309-anchor pass.
    """

    if len(optimizer_step_seconds) != CAPACITY_PROBE_OPTIMIZER_STEPS:
        raise ValueError("capacity projection requires exactly two optimizer timings")
    timings = tuple(float(value) for value in optimizer_step_seconds)
    scalar_timings = (*timings, float(teacher_probe_seconds), float(free_running_probe_seconds))
    if any(not math.isfinite(value) or value <= 0.0 for value in scalar_timings):
        raise ValueError("capacity projection timings must be finite and positive")
    max_steps = int(training.get("max_steps", 0))
    eval_steps = int(training.get("eval_steps", 0))
    if max_steps < 1 or eval_steps < 1:
        raise ValueError("capacity projection requires positive formal max_steps/eval_steps")
    interval_teacher_passes = math.ceil(max_steps / eval_steps)
    final_teacher_passes = 1
    teacher_passes = interval_teacher_passes + final_teacher_passes
    optimizer_seconds_per_step = max(timings)
    teacher_seconds_per_anchor = (
        float(teacher_probe_seconds) / CAPACITY_PROBE_VALIDATION_ANCHORS
    )
    free_seconds_per_anchor = (
        float(free_running_probe_seconds) / CAPACITY_PROBE_VALIDATION_ANCHORS
    )
    components = {
        "optimizer": optimizer_seconds_per_step * max_steps,
        "teacher_forced": (
            teacher_seconds_per_anchor * EXPECTED_COUNTS["val"] * teacher_passes
        ),
        "free_running": free_seconds_per_anchor * EXPECTED_COUNTS["val"],
    }
    return {
        "formal_max_steps": max_steps,
        "formal_eval_steps": eval_steps,
        "interval_teacher_passes": interval_teacher_passes,
        "final_teacher_passes": final_teacher_passes,
        "total_teacher_passes": teacher_passes,
        "optimizer_seconds_per_step": optimizer_seconds_per_step,
        "teacher_seconds_per_anchor": teacher_seconds_per_anchor,
        "free_running_seconds_per_anchor": free_seconds_per_anchor,
        "components_seconds": components,
        "projected_formal_runtime_seconds": sum(components.values()),
    }


def require_multires_event_v2_training_authorization(
    train: Mapping[str, Any] | None,
) -> None:
    if not TRAINING_AUTHORIZED:
        raise RuntimeError(
            "V2 training is source-gated in the training core; only read-only dry "
            f"preflight is authorized. Reason: {TRAINING_AUTHORIZATION_REASON}."
        )
    if not isinstance(train, Mapping):
        raise RuntimeError("V2 source authorization must be bound to one loaded train config")
    run_name = str(train.get("run_name") or "")
    if run_name not in AUTHORIZED_TRAINING_RUN_NAMES:
        raise RuntimeError(
            f"V2 training is not authorized for run_name={run_name!r}; "
            f"authorized={AUTHORIZED_TRAINING_RUN_NAMES!r}. Reason: "
            f"{TRAINING_AUTHORIZATION_REASON}."
        )


def require_multires_event_v2_verification_authorization(
    train: Mapping[str, Any] | None,
) -> None:
    if not VERIFICATION_AUTHORIZED:
        raise RuntimeError(
            "V2 verification is source-gated. Reason: "
            f"{VERIFICATION_AUTHORIZATION_REASON}."
        )
    if not isinstance(train, Mapping):
        raise RuntimeError("V2 verification authorization requires one train config")
    run_name = str(train.get("run_name") or "")
    if run_name not in AUTHORIZED_VERIFICATION_RUN_NAMES:
        raise RuntimeError(
            f"V2 verification is not authorized for run_name={run_name!r}; "
            f"authorized={AUTHORIZED_VERIFICATION_RUN_NAMES!r}. Reason: "
            f"{VERIFICATION_AUTHORIZATION_REASON}."
        )


def _capacity_authorization_from_report(
    train_config_path: str | Path,
    train: Mapping[str, Any],
    report: Mapping[str, Any],
) -> _CapacityAuthorization:
    config_path = Path(train_config_path).resolve()
    report_path = Path(str(report.get("report_path") or "")).resolve()
    if report.get("status") != "PASSED" or not report_path.is_file():
        raise RuntimeError("internal capacity authorization requires one persisted PASS report")
    return _CapacityAuthorization(
        guard=_CAPACITY_AUTHORIZATION_GUARD,
        config_path=str(config_path),
        config_file_sha256=sha256_file(config_path),
        train_config_sha256=sha256_payload(train),
        mode=str(train["mode"]),
        capacity_report_path=str(report_path),
        capacity_report_sha256=sha256_file(report_path),
    )


def _validate_capacity_authorization(
    authorization: _CapacityAuthorization | None,
    *,
    train_config_path: str | Path,
    train: Mapping[str, Any],
) -> None:
    if str(train.get("run_name")) == "t4x2_multires_event_v2_smoke":
        return
    config_path = Path(train_config_path).resolve()
    if (
        not isinstance(authorization, _CapacityAuthorization)
        or authorization.guard is not _CAPACITY_AUTHORIZATION_GUARD
        or authorization.config_path != str(config_path)
        or authorization.config_file_sha256 != sha256_file(config_path)
        or authorization.train_config_sha256 != sha256_payload(train)
        or authorization.mode != str(train.get("mode"))
    ):
        raise RuntimeError(
            "formal V2 training requires an internal same-process capacity PASS "
            "authorization bound to this config and mode"
        )
    report_path = Path(authorization.capacity_report_path)
    if (
        not report_path.is_file()
        or sha256_file(report_path) != authorization.capacity_report_sha256
    ):
        raise RuntimeError("formal V2 capacity authorization report bytes changed")


def run_multires_event_v2_capacity_gated_training(
    train_config_path: str | Path,
    *,
    repo_root: str | Path,
    capacity_output_dir: str | Path,
    elapsed_before_capacity_seconds: float,
) -> dict[str, Any]:
    """Run one attempt-local capacity gate and then one formal mode.

    Both phases live inside the same torchrun process group.  The probe model is
    destroyed before the formal runner re-seeds and rebuilds the model, so probe
    RNG consumption and optimizer state cannot alter the matched experiment.
    """

    train, _, _, _, _ = load_multires_event_v2_configs(
        train_config_path, repo_root=repo_root
    )
    require_multires_event_v2_training_authorization(train)
    completed = False
    try:
        report = run_multires_event_v2_capacity_probe(
            train_config_path,
            repo_root=repo_root,
            output_dir=capacity_output_dir,
            elapsed_before_capacity_seconds=elapsed_before_capacity_seconds,
        )
        if report.get("status") != "PASSED":
            raise RuntimeError("V2 capacity probe did not authorize formal training")
        if is_rank_zero():
            print(
                "MULTIRES_EVENT_V2_CAPACITY_PROBE_OK "
                f"mode={report['mode']} path={report['report_path']}",
                flush=True,
            )
            print(
                "MULTIRES_EVENT_V2_HOSTED_SMOKE_OK "
                f"mode={report['mode']} optimizer_steps="
                f"{CAPACITY_PROBE_OPTIMIZER_STEPS} trajectories="
                f"{CAPACITY_PROBE_VALIDATION_ANCHORS}x"
                f"{CAPACITY_PROBE_TRAJECTORIES_PER_ANCHOR}",
                flush=True,
            )
        result = run_multires_event_v2_training(
            train_config_path,
            repo_root=repo_root,
        )
        completed = True
        return result
    finally:
        # On failure the worker must exit immediately so torchelastic can kill
        # a peer blocked in a collective.  A best-effort "graceful" destroy on
        # the failing rank can itself wait for the peer until the process-group
        # timeout, which caused the r3 failure to waste an additional 600 s.
        if (
            completed
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.destroy_process_group()


def run_multires_event_v2_rank_artifact_preflight_only(
    *,
    output_dir: str | Path,
    mode: str,
) -> dict[str, Any]:
    """Run exact distributed artifact paths before Dataset loading.

    This route accepts no model or data config and performs no Dataset scan. It
    exercises both rank-local writer/hash/gather and the production
    best-checkpoint save/load boundary.  The checkpoint canary uses a
    zero-parameter Identity module; it is an I/O/collective contract check, not
    a capacity model or optimization attempt.  A hosted attempt can therefore
    reject deterministic collective-order defects before materializing the
    50,350-anchor runtime.
    """

    if mode not in MATCHED_MODES:
        raise ValueError("early rank artifact preflight mode is invalid")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 2:
        raise RuntimeError(
            "early V2 rank artifact preflight requires torchrun --nproc_per_node=2"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("early V2 rank artifact preflight requires two visible GPUs")
    output_root = Path(output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("early rank artifact preflight output is not empty")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(
                seconds=V2_EARLY_CANARY_PROCESS_GROUP_TIMEOUT_SECONDS
            ),
            device_id=device,
        )
    completed = False
    try:
        result = verify_rank_local_artifact_preflight(
            output_dir=output_root,
            mode=mode,
        )
        if int(result.get("world_size", -1)) != world_size:
            raise RuntimeError("early rank artifact preflight world size changed")
        _run_v2_best_checkpoint_collective_canary(
            output_root=output_root / "best-checkpoint-collective-canary",
            mode=mode,
            world_size=world_size,
        )
        if is_rank_zero():
            print(
                "MULTIRES_EVENT_V2_RANK_ARTIFACT_CANARY_OK "
                f"phase=predata world_size={world_size} best_checkpoint=verified "
                f"sha256={result['manifest_sha256']}",
                flush=True,
            )
        completed = True
        return result
    finally:
        # Match the formal failure policy: do not enter a graceful destroy after
        # an asymmetric failure because it can hide the original traceback.
        if completed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def run_multires_event_v2_verification_probe(
    train_config_path: str | Path,
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    elapsed_before_capacity_seconds: float,
) -> dict[str, Any]:
    """Run the full capacity/failure-path probe and stop before formal training."""

    train, _, _, _, _ = load_multires_event_v2_configs(
        train_config_path,
        repo_root=repo_root,
    )
    require_multires_event_v2_verification_authorization(train)
    completed = False
    try:
        report = run_multires_event_v2_capacity_probe(
            train_config_path,
            repo_root=repo_root,
            output_dir=output_dir,
            elapsed_before_capacity_seconds=elapsed_before_capacity_seconds,
            _verification_guard=_VERIFICATION_AUTHORIZATION_GUARD,
        )
        if report.get("status") != "PASSED":
            raise RuntimeError("V2 verification probe did not produce a PASS report")
        output_root = Path(output_dir).resolve()
        completion_path = output_root / "verification_complete.json"
        if is_rank_zero():
            atomic_write_json(
                completion_path,
                {
                    "schema_version": (
                        "trauma_predict.multires_event_v2_verification_complete.v1"
                    ),
                    "created_at": utc_now(),
                    "status": "PASSED_STOPPED_BEFORE_FORMAL_TRAINING",
                    "formal_training_authorized": False,
                    "formal_optimizer_steps": 0,
                    "mode": str(train["mode"]),
                    "run_name": str(train["run_name"]),
                    "capacity_report_path": str(report["report_path"]),
                    "capacity_report_sha256": sha256_file(
                        Path(str(report["report_path"]))
                    ),
                },
            )
        _barrier()
        result = {
            **report,
            "verification_complete_path": str(completion_path),
            "verification_complete_sha256": sha256_file(completion_path),
            "formal_optimizer_steps": 0,
        }
        if is_rank_zero():
            print(
                "MULTIRES_EVENT_V2_VERIFICATION_ONLY_COMPLETE "
                f"mode={train['mode']} path={completion_path}",
                flush=True,
            )
        completed = True
        return result
    finally:
        if completed and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def run_multires_event_v2_capacity_probe(
    train_config_path: str | Path,
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    elapsed_before_capacity_seconds: float,
    _verification_guard: object | None = None,
) -> dict[str, Any]:
    """Exercise the exact hosted path without writing into the formal run root."""

    root = Path(repo_root).resolve()
    train, dataset, model_config, _, _ = load_multires_event_v2_configs(
        train_config_path,
        repo_root=root,
    )
    if _verification_guard is None:
        require_multires_event_v2_training_authorization(train)
    else:
        if _verification_guard is not _VERIFICATION_AUTHORIZATION_GUARD:
            raise RuntimeError("invalid V2 verification authorization guard")
        require_multires_event_v2_verification_authorization(train)
    mode = str(train["mode"])
    if mode not in MATCHED_MODES or str(train.get("run_name")) == "t4x2_multires_event_v2_smoke":
        raise ValueError("capacity probe is defined only for a formal matched mode")
    elapsed_before = float(elapsed_before_capacity_seconds)
    if not math.isfinite(elapsed_before) or elapsed_before < 0.0:
        raise ValueError("elapsed_before_capacity_seconds must be finite and nonnegative")
    probe_root = Path(output_dir).resolve()
    formal_root = resolve_repo_path(str(train["outputs"]["output_dir"]), root)
    if _paths_overlap(probe_root, formal_root):
        raise ValueError("capacity probe output must not overlap the formal run root")
    if probe_root.exists() and any(probe_root.iterdir()):
        raise FileExistsError("capacity probe output directory is not empty")

    rank, world_size, local_rank, device = _initialize_v2_distributed(train)
    probe_started = time.monotonic()
    _seed_everything(int(train["seed"]), rank)
    if is_rank_zero():
        probe_root.mkdir(parents=True, exist_ok=True)
    _barrier()

    rank_artifact_canary = verify_rank_local_artifact_preflight(
        output_dir=probe_root / "ddp_rank_artifact_canary",
        mode=mode,
    )
    if is_rank_zero():
        print(
            "MULTIRES_EVENT_V2_RANK_ARTIFACT_CANARY_OK "
            f"world_size={rank_artifact_canary['world_size']} "
            f"sha256={rank_artifact_canary['manifest_sha256']}",
            flush=True,
        )

    runtime = build_multires_event_v2_runtime(
        train,
        dataset,
        repo_root=root,
        rank=rank,
        world_size=world_size,
        phase="final",
    )
    validate_formal_target_field_order(model_config, runtime.contract)
    expected_sample_ids = tuple(
        str(value)
        for value in runtime.eval_dataset.sample_ids[:CAPACITY_PROBE_VALIDATION_ANCHORS]
    )
    if (
        len(expected_sample_ids) != CAPACITY_PROBE_VALIDATION_ANCHORS
        or len(set(expected_sample_ids)) != CAPACITY_PROBE_VALIDATION_ANCHORS
    ):
        raise RuntimeError("capacity probe cannot resolve the first 100 persisted val anchors")
    probe_loader = _capacity_probe_eval_loader(
        runtime,
        rank=rank,
        world_size=world_size,
    )
    semantic_canary_loader = _capacity_probe_eval_loader(
        runtime,
        rank=rank,
        world_size=world_size,
        validation_anchors=CAPACITY_SEMANTIC_CANARY_ANCHORS,
    )

    torch.cuda.reset_peak_memory_stats(device)
    model = build_multires_event_v2_model(model_config, mode=mode).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
    training = _mapping(train["training"], "train.training")
    promotion_contract = load_promotion_metric_contract(
        resolve_repo_path(str(train["promotion_metric_contract"]), root),
        expected_sha256=str(train["promotion_metric_contract_hash"]),
        data_contract=runtime.contract,
    )

    semantic_canary_root = probe_root / "ddp_semantic_canary"
    _barrier()
    torch.cuda.synchronize(device)
    semantic_canary_started = time.monotonic()
    semantic_canary_result = evaluate_free_running_v2(
        model=model,
        loader=semantic_canary_loader,
        contract=runtime.contract,
        device=device,
        mode=mode,
        expected_samples=CAPACITY_SEMANTIC_CANARY_ANCHORS,
        step=0,
        output_dir=semantic_canary_root,
        expected_lab_scale_artifact_hash=str(train["lab_scale_artifact_hash"]),
        standardized_primitive_scale_path=resolve_repo_path(
            str(train["standardized_primitive_scale_artifact"]), root
        ),
        expected_standardized_primitive_scale_hash=str(
            train["standardized_primitive_scale_artifact_hash"]
        ),
        input_normalization_sha256=str(runtime.identity["input_normalization_sha256"]),
        promotion_metric_contract=promotion_contract,
        trajectories_per_anchor=CAPACITY_SEMANTIC_CANARY_TRAJECTORIES_PER_ANCHOR,
        trajectory_batch_size=CAPACITY_SEMANTIC_CANARY_TRAJECTORIES_PER_ANCHOR,
        crn_seed=int(train["evaluation"]["free_running_crn_seed"]),
        metrics_path=None,
        precision=str(training["precision"]),
    )
    torch.cuda.synchronize(device)
    _barrier()
    semantic_canary_seconds = _distributed_max_float(
        time.monotonic() - semantic_canary_started,
        device,
    )
    semantic_coherence = _mapping(
        semantic_canary_result.get("coherence"),
        "semantic canary coherence",
    )
    semantic_shards = semantic_canary_result.get("shards")
    if (
        int(semantic_canary_result.get("anchors", -1))
        != CAPACITY_SEMANTIC_CANARY_ANCHORS
        or int(semantic_canary_result.get("trajectories_per_anchor", -1))
        != CAPACITY_SEMANTIC_CANARY_TRAJECTORIES_PER_ANCHOR
        or float(semantic_coherence.get("rate", -1.0)) != 1.0
        or int(semantic_coherence.get("coherent_trajectories", -1))
        != CAPACITY_SEMANTIC_CANARY_ANCHORS
        * CAPACITY_SEMANTIC_CANARY_TRAJECTORIES_PER_ANCHOR
        or not isinstance(semantic_shards, list)
        or len(semantic_shards) != world_size
        or sorted(int(row.get("rank", -1)) for row in semantic_shards) != [0, 1]
        or any(int(row.get("anchors", -1)) != 1 for row in semantic_shards)
    ):
        raise RuntimeError("V2 semantic DDP canary did not close the exact 2x100 path")
    semantic_canary = {
        "status": "PASSED",
        "anchors": CAPACITY_SEMANTIC_CANARY_ANCHORS,
        "trajectories_per_anchor": (
            CAPACITY_SEMANTIC_CANARY_TRAJECTORIES_PER_ANCHOR
        ),
        "world_size": world_size,
        "wall_seconds": semantic_canary_seconds,
        "coherence_rate": float(semantic_coherence["rate"]),
        "manifest_path": str(
            (semantic_canary_root / str(semantic_canary_result["manifest_path"])).resolve()
        ),
        "manifest_sha256": str(semantic_canary_result["manifest_sha256"]),
    }
    if is_rank_zero():
        print(
            "MULTIRES_EVENT_V2_SEMANTIC_CANARY_OK "
            f"anchors={semantic_canary['anchors']} "
            f"trajectories={semantic_canary['trajectories_per_anchor']} "
            f"elapsed={semantic_canary_seconds:.3f}s",
            flush=True,
        )

    optimizer = build_multires_event_v2_optimizer(model, training)
    scheduler = _build_scheduler(optimizer, training)
    scaler = _build_grad_scaler(torch, device, training)
    optimizer_rows = _run_capacity_optimizer_steps(
        model=model,
        runtime=runtime,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        train=train,
        device=device,
        world_size=world_size,
    )

    checkpoint_canary_root = probe_root / "checkpoint_canary"
    checkpoint_identity = {
        "hosted_smoke_identity": sha256_payload(
            {
                "dataset_id": runtime.identity["dataset_id"],
                "contract_bundle_hash": runtime.identity["contract_bundle_hash"],
                "mode": mode,
                "optimizer_steps": CAPACITY_PROBE_OPTIMIZER_STEPS,
            }
        )
    }
    _save_v2_checkpoint(
        output_dir=checkpoint_canary_root,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        trainer_state={
            "global_step": CAPACITY_PROBE_OPTIMIZER_STEPS,
            "epoch": 0,
            "batches_in_epoch": CAPACITY_PROBE_OPTIMIZER_STEPS,
            "micro_in_accum": 0,
            "best_metric": None,
            "best_step": None,
            "scaler_skipped_steps": 0,
        },
        identity_hashes=checkpoint_identity,
        runtime=runtime,
        rank=rank,
        keep_last=1,
    )
    checkpoint_path = (
        checkpoint_canary_root
        / "checkpoints"
        / f"checkpoint-{CAPACITY_PROBE_OPTIMIZER_STEPS:08d}"
    )
    checkpoint_manifest = _validate_v2_checkpoint_directory(
        checkpoint_path,
        expected_world_size=world_size,
        expected_step=CAPACITY_PROBE_OPTIMIZER_STEPS,
    )
    resume_config = copy.deepcopy(train)
    resume_config["training"]["resume"] = True
    resumed_state, _ = _maybe_resume(
        output_dir=checkpoint_canary_root,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        identity_hashes=checkpoint_identity,
        device=device,
        rank=rank,
        config=resume_config,
        runtime=runtime,
    )
    resume_alignment = _validate_resume_optimizer_alignment(
        optimizer,
        scheduler,
        training,
        global_step=int(resumed_state["global_step"]),
    )
    if int(resumed_state["global_step"]) != CAPACITY_PROBE_OPTIMIZER_STEPS:
        raise RuntimeError("hosted smoke checkpoint did not restore exact optimizer step two")

    metrics_path = probe_root / "metrics.jsonl"
    _barrier()
    torch.cuda.synchronize(device)
    started = time.monotonic()
    teacher = evaluate_teacher_forced(
        model=model,
        loader=probe_loader,
        registry=runtime.contract.process_registry,
        device=device,
        mode=mode,
        expected_samples=CAPACITY_PROBE_VALIDATION_ANCHORS,
        phase="final",
        step=CAPACITY_PROBE_OPTIMIZER_STEPS,
        precision=str(training["precision"]),
        metrics_path=metrics_path,
        expected_lab_scale_artifact_hash=train.get("lab_scale_artifact_hash"),
        evaluation_identity=_evaluation_contract_identity(runtime),
    )
    torch.cuda.synchronize(device)
    _barrier()
    teacher_seconds = _distributed_max_float(time.monotonic() - started, device)

    free_root = probe_root / "free_running"
    _barrier()
    torch.cuda.synchronize(device)
    started = time.monotonic()
    free_running = evaluate_free_running_v2(
        model=model,
        loader=probe_loader,
        contract=runtime.contract,
        device=device,
        mode=mode,
        expected_samples=CAPACITY_PROBE_VALIDATION_ANCHORS,
        step=CAPACITY_PROBE_OPTIMIZER_STEPS,
        output_dir=free_root,
        expected_lab_scale_artifact_hash=str(train["lab_scale_artifact_hash"]),
        standardized_primitive_scale_path=resolve_repo_path(
            str(train["standardized_primitive_scale_artifact"]), root
        ),
        expected_standardized_primitive_scale_hash=str(
            train["standardized_primitive_scale_artifact_hash"]
        ),
        input_normalization_sha256=str(runtime.identity["input_normalization_sha256"]),
        promotion_metric_contract=promotion_contract,
        trajectories_per_anchor=CAPACITY_PROBE_TRAJECTORIES_PER_ANCHOR,
        trajectory_batch_size=CAPACITY_PROBE_TRAJECTORIES_PER_ANCHOR,
        crn_seed=int(train["evaluation"]["free_running_crn_seed"]),
        metrics_path=metrics_path,
        precision=str(training["precision"]),
    )
    torch.cuda.synchronize(device)
    _barrier()
    free_seconds = _distributed_max_float(time.monotonic() - started, device)

    device_properties = torch.cuda.get_device_properties(device)
    local_hardware = {
        "rank": rank,
        "local_rank": local_rank,
        "device_name": str(device_properties.name),
        "compute_capability": [
            int(device_properties.major),
            int(device_properties.minor),
        ],
        "total_memory_bytes": int(device_properties.total_memory),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    hardware = _all_gather_objects(local_hardware)
    probe_elapsed = _distributed_max_float(time.monotonic() - probe_started, device)
    projection = project_multires_event_v2_capacity_runtime(
        training,
        optimizer_step_seconds=tuple(float(row["wall_seconds"]) for row in optimizer_rows),
        teacher_probe_seconds=teacher_seconds,
        free_running_probe_seconds=free_seconds,
    )

    report: dict[str, Any] | None = None
    if is_rank_zero():
        observed_sample_ids = _capacity_free_running_sample_ids(free_root, free_running)
        structural_metrics = {
            key: float(_mapping(free_running.get(key), key).get("subject_macro", math.nan))
            for key in CAPACITY_STRUCTURAL_METRICS
        }
        coherence = _mapping(free_running.get("coherence"), "free_running.coherence")
        projected_background_runtime_seconds = (
            elapsed_before
            + probe_elapsed
            + float(projection["projected_formal_runtime_seconds"])
        )
        failures: list[str] = []
        if len(hardware) != 2 or any(
            "T4" not in str(row.get("device_name", "")).upper() for row in hardware
        ):
            failures.append("capacity probe requires exactly two T4 devices")
        if len(optimizer_rows) != CAPACITY_PROBE_OPTIMIZER_STEPS or any(
            int(row.get("global_anchors", -1)) != 64
            or not math.isfinite(float(row.get("joint_nll_anchor_mean", math.nan)))
            or row.get("optimizer_updated") is not True
            for row in optimizer_rows
        ):
            failures.append("two exact B32/GPU optimizer updates were not completed")
        if int(teacher.get("samples", -1)) != CAPACITY_PROBE_VALIDATION_ANCHORS:
            failures.append("teacher probe did not cover exactly 100 anchors")
        if (
            int(free_running.get("anchors", -1)) != CAPACITY_PROBE_VALIDATION_ANCHORS
            or int(free_running.get("trajectories_per_anchor", -1))
            != CAPACITY_PROBE_TRAJECTORIES_PER_ANCHOR
        ):
            failures.append("free-running probe did not preserve the frozen 100x100 contract")
        if tuple(sorted(observed_sample_ids)) != tuple(sorted(expected_sample_ids)):
            failures.append("free-running probe did not use the first 100 persisted val anchors")
        if any(not math.isfinite(value) for value in structural_metrics.values()):
            failures.append("one or more structural capacity metrics are non-finite")
        if (
            float(coherence.get("rate", -1.0)) != 1.0
            or int(coherence.get("coherent_trajectories", -1))
            != CAPACITY_PROBE_VALIDATION_ANCHORS
            * CAPACITY_PROBE_TRAJECTORIES_PER_ANCHOR
        ):
            failures.append("capacity trajectories are not 100% coherent")
        if any(
            int(row["peak_allocated_bytes"]) <= 0
            or int(row["peak_reserved_bytes"]) > int(row["total_memory_bytes"])
            for row in hardware
        ):
            failures.append("capacity probe GPU memory accounting is invalid")
        if list(probe_root.rglob("SUCCESS")):
            failures.append("capacity probe emitted a forbidden formal SUCCESS marker")
        status = "PASSED" if not failures else "FAILED"
        report_path = probe_root / "capacity_probe.json"
        report = {
            "schema_version": CAPACITY_PROBE_SCHEMA,
            "created_at": utc_now(),
            "status": status,
            "mode": mode,
            "report_path": str(report_path),
            "contract": {
                "optimizer_steps": CAPACITY_PROBE_OPTIMIZER_STEPS,
                "per_device_train_batch_size": 32,
                "world_size": 2,
                "precision": "fp16",
                "validation_selection": "persisted_val_manifest_prefix",
                "validation_anchors": CAPACITY_PROBE_VALIDATION_ANCHORS,
                "trajectories_per_anchor": CAPACITY_PROBE_TRAJECTORIES_PER_ANCHOR,
                "formal_validation_anchors": EXPECTED_COUNTS["val"],
                "formal_trajectories_per_anchor": 100,
            },
            "identity": {
                "dataset_id": runtime.identity["dataset_id"],
                "contract_bundle_hash": runtime.identity["contract_bundle_hash"],
                "relation_contract_sha256": runtime.identity[
                    "relation_contract_sha256"
                ],
                "sidecar_schema_sha256": runtime.identity[
                    "sidecar_schema_sha256"
                ],
                "input_normalization_sha256": runtime.identity[
                    "input_normalization_sha256"
                ],
                "promotion_metric_contract_sha256": train[
                    "promotion_metric_contract_hash"
                ],
                "first_100_sample_ids_sha256": sha256_payload(expected_sample_ids),
                "first_100_sample_id_set_sha256": sha256_payload(
                    tuple(sorted(expected_sample_ids))
                ),
            },
            "hardware": hardware,
            "distributed_canaries": {
                "rank_artifact": rank_artifact_canary,
                "semantic_rollout": semantic_canary,
            },
            "optimizer": {
                "optimizer_contract_version": OPTIMIZER_CONTRACT_VERSION,
                "loss_reduction": RAW_JOINT_NLL_REDUCTION,
                "gradient_clipping": "disabled",
                "configured_contract": copy.deepcopy(EXPECTED_OPTIMIZER_CONTRACT),
                "steps": optimizer_rows,
                "scaler_skipped_steps": 0,
            },
            "checkpoint_resume_canary": {
                "schema_version": V2_CHECKPOINT_SCHEMA,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_manifest_sha256": sha256_file(
                    checkpoint_path / "checkpoint_manifest.json"
                ),
                "manifest_file_count": len(checkpoint_manifest["files"]),
                "restored_global_step": int(resumed_state["global_step"]),
                "resume_alignment": resume_alignment,
            },
            "teacher_probe": {
                "anchors": int(teacher["samples"]),
                "subjects": int(teacher["subjects"]),
                "wall_seconds": teacher_seconds,
                "joint_nll_subject_macro": float(teacher["joint_nll_subject_macro"]),
            },
            "free_running_probe": {
                "anchors": int(free_running["anchors"]),
                "trajectories_per_anchor": int(
                    free_running["trajectories_per_anchor"]
                ),
                "wall_seconds": free_seconds,
                "structural_subject_macro": structural_metrics,
                "coherence_rate": float(coherence["rate"]),
                "coherent_trajectories": int(coherence["coherent_trajectories"]),
                "observed_sample_ids_sha256": sha256_payload(
                    tuple(sorted(observed_sample_ids))
                ),
                "selection_verified": tuple(sorted(observed_sample_ids))
                == tuple(sorted(expected_sample_ids)),
            },
            "projection": projection,
            "runtime_projection": {
                "policy": CAPACITY_RUNTIME_POLICY,
                "hard_limit_seconds": None,
                "gates_capacity_status": False,
                "elapsed_before_capacity_seconds": elapsed_before,
                "capacity_probe_elapsed_seconds": probe_elapsed,
                "projected_formal_runtime_seconds": float(
                    projection["projected_formal_runtime_seconds"]
                ),
                "projected_background_runtime_seconds": (
                    projected_background_runtime_seconds
                ),
            },
            "failures": failures,
        }
        atomic_write_json(report_path, report)
    report = _broadcast_object(report)
    if not isinstance(report, Mapping):
        raise RuntimeError("capacity probe report broadcast failed")

    del (
        probe_loader,
        semantic_canary_loader,
        optimizer,
        scheduler,
        scaler,
        model,
        runtime,
    )
    gc.collect()
    torch.cuda.empty_cache()
    _barrier()
    if report.get("status") != "PASSED":
        raise RuntimeError(
            "V2 capacity probe blocked formal training: "
            + "; ".join(str(value) for value in report.get("failures", ()))
        )
    return dict(report)


def _capacity_probe_eval_loader(
    runtime: MultiresEventV2Runtime,
    *,
    rank: int,
    world_size: int,
    validation_anchors: int = CAPACITY_PROBE_VALIDATION_ANCHORS,
) -> Any:
    from torch.utils.data import DataLoader

    if world_size != 2 or rank not in {0, 1}:
        raise ValueError("capacity probe requires two DDP ranks")
    validation_anchors = int(validation_anchors)
    if validation_anchors < world_size or validation_anchors % world_size != 0:
        raise ValueError("capacity probe anchors must be positive and evenly split")
    indices = tuple(range(rank, validation_anchors, world_size))
    if len(indices) != validation_anchors // world_size:
        raise AssertionError("capacity probe did not split anchors evenly")
    return DataLoader(
        runtime.eval_dataset,
        batch_size=32,
        sampler=indices,
        num_workers=0,
        pin_memory=bool(getattr(runtime.eval_loader, "pin_memory", True)),
        persistent_workers=False,
        drop_last=False,
        collate_fn=runtime.eval_loader.collate_fn,
    )


def _run_capacity_optimizer_steps(
    *,
    model: Any,
    runtime: MultiresEventV2Runtime,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    train: Mapping[str, Any],
    device: torch.device,
    world_size: int,
) -> list[dict[str, Any]]:
    training = _mapping(train["training"], "train.training")
    iterator = iter(runtime.train_loader)
    autocast = _autocast_factory(device, str(training["precision"]))
    optimizer.zero_grad(set_to_none=True)
    model.train()
    rows: list[dict[str, Any]] = []
    for step in range(1, CAPACITY_PROBE_OPTIMIZER_STEPS + 1):
        _barrier()
        torch.cuda.synchronize(device)
        started = time.monotonic()
        try:
            raw_batch = next(iterator)
        except StopIteration as exc:
            raise RuntimeError("capacity probe train loader ended before two steps") from exc
        local_anchors = len(raw_batch.get("sample_id") or ())
        if local_anchors != 32:
            raise RuntimeError("capacity probe requires B32 on every rank and optimizer step")
        batch = move_to_device(raw_batch, device)
        _, loss_result = exact_teacher_forced_loss(
            model,
            batch,
            runtime.contract.process_registry,
            mode=str(train["mode"]),
            expected_lab_scale_artifact_hash=train.get("lab_scale_artifact_hash"),
            autocast=autocast,
        )
        loss = _validated_optimizer_loss(
            loss_result, expected_local_batch=local_anchors
        )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        (
            optimizer_updated,
            scale_before,
            scale_after,
            gradient_health,
            optimizer_state_health,
        ) = _audited_optimizer_step(
            model,
            optimizer,
            scaler,
            expected_optimizer_step=step,
        )
        if not optimizer_updated:
            raise FloatingPointError(
                "capacity probe FP16 overflow blocks the matched formal run"
            )
        if optimizer_state_health is None:
            raise AssertionError("successful capacity optimizer step lacks state health")
        step_health_payload = _optimizer_step_health_payload(
            optimizer,
            gradient_health,
            optimizer_state_health,
            training=training,
        )
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        torch.cuda.synchronize(device)
        _barrier()
        wall_seconds = _distributed_max_float(time.monotonic() - started, device)
        total_nll, total_anchors = _distributed_sum_count(
            float(loss_result["per_sample_nll"].detach().sum().cpu().item()),
            local_anchors,
            device,
        )
        if int(total_anchors) != 32 * world_size:
            raise RuntimeError("capacity probe global optimizer batch is not exactly 64")
        rows.append(
            {
                "event": "v2_optimizer_health",
                "step": step,
                "local_anchors": local_anchors,
                "world_size": world_size,
                "global_anchors": int(total_anchors),
                "wall_seconds": wall_seconds,
                "joint_nll_anchor_mean": total_nll / total_anchors,
                **step_health_payload,
                "scaler_scale_before": scale_before,
                "scaler_scale_after": scale_after,
                "optimizer_updated": True,
            }
        )
    return rows


def _capacity_free_running_sample_ids(
    root: Path,
    result: Mapping[str, Any],
) -> list[str]:
    values: list[str] = []
    shards = result.get("shards")
    if not isinstance(shards, list) or len(shards) != 2:
        raise ValueError("capacity free-running result must expose two rank shards")
    for shard in shards:
        row = _mapping(shard, "capacity free-running shard")
        path = root / str(row.get("per_anchor_score_path") or "")
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    values.append(str(json.loads(line)["sample_id"]))
    if len(values) != CAPACITY_PROBE_VALIDATION_ANCHORS or len(set(values)) != len(values):
        raise RuntimeError("capacity free-running shards do not contain 100 unique anchors")
    return sorted(values)


def _distributed_max_float(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return float(tensor.item())


def _all_gather_objects(value: Any) -> list[Any]:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return [value]
    result: list[Any] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(result, value)
    return result


def _broadcast_object(value: Any) -> Any:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return value
    payload = [value]
    torch.distributed.broadcast_object_list(payload, src=0)
    return payload[0]


def _paths_overlap(left: Path, right: Path) -> bool:
    for candidate, parent in ((left, right), (right, left)):
        try:
            candidate.relative_to(parent)
            return True
        except ValueError:
            pass
    return False


def run_multires_event_v2_training(
    train_config_path: str | Path,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Run the authorized relational primary, including its formal step-2 checkpoint."""

    root = Path(repo_root).resolve()
    train, dataset, model_config, dataset_path, model_path = load_multires_event_v2_configs(
        train_config_path,
        repo_root=root,
    )
    require_multires_event_v2_training_authorization(train)
    rank, world_size, local_rank, device = _initialize_v2_distributed(train)
    _seed_everything(int(train["seed"]), rank)
    output_dir = resolve_repo_path(str(train["outputs"]["output_dir"]), root)
    metrics_path = resolve_repo_path(str(train["outputs"]["metrics_jsonl"]), root)
    _collect_distributed_phase(
        "formal output-root materialization",
        lambda: output_dir.mkdir(parents=True, exist_ok=True)
        if is_rank_zero()
        else None,
    )
    _run_v2_best_checkpoint_collective_canary(
        output_root=output_dir / "preflight/best-checkpoint-collective-canary",
        mode=str(train["mode"]),
        world_size=world_size,
    )
    runtime = build_multires_event_v2_runtime(
        train,
        dataset,
        repo_root=root,
        rank=rank,
        world_size=world_size,
        phase="interval",
    )
    validate_formal_target_field_order(model_config, runtime.contract)
    _collect_distributed_phase(
        "formal run-artifact materialization",
        lambda: _materialize_run_artifacts(
            output_dir,
            runtime=runtime,
            train=train,
            repo_root=root,
            train_path=Path(train_config_path).resolve(),
            dataset_path=dataset_path,
            model_path=model_path,
        )
        if is_rank_zero()
        else None,
    )
    runtime = _bind_runtime_to_run_artifacts(output_dir, runtime)
    model = build_multires_event_v2_model(model_config, mode=str(train["mode"])).to(device)
    source_identity = _source_tree_identity(root)
    identity_hashes = {
        "train_config": sha256_payload(train),
        "dataset_config": sha256_payload(dataset),
        "model_config": sha256_payload(model_config),
        "runtime": sha256_payload(runtime.identity),
        "semantic_runtime": str(runtime.identity["semantic_runtime_identity_sha256"]),
        "contract_bundle": runtime.contract.contract_bundle_hash,
        "normalization": str(runtime.identity["normalization_artifact_sha256"]),
        "source_tree": source_identity["source_tree_sha256"],
        "source_identity": sha256_payload(source_identity),
        "git_commit": str(source_identity.get("git_commit") or "unavailable"),
        "git_head_tree": str(source_identity.get("git_head_tree") or "unavailable"),
        "matched_design": matched_design_signature(train, dataset, model_config),
    }
    _collect_distributed_phase(
        "formal identity materialization",
        lambda: _write_identity(
            output_dir,
            train=train,
            dataset=dataset,
            model=model_config,
            runtime=runtime,
            identity_hashes=identity_hashes,
            train_path=Path(train_config_path).resolve(),
            dataset_path=dataset_path,
            model_path=model_path,
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            source_identity=source_identity,
        )
        if is_rank_zero()
        else None,
    )
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
    training = train["training"]
    optimizer = build_multires_event_v2_optimizer(model, training)
    scheduler = _build_scheduler(optimizer, training)
    scaler = _build_grad_scaler(torch, device, training)
    _validate_v2_checkpoint_integrity(output_dir, expected_world_size=world_size)
    state, deferred_rng = _maybe_resume(
        output_dir=output_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        identity_hashes=identity_hashes,
        device=device,
        rank=rank,
        config=train,
        runtime=runtime,
    )
    resume_alignment = _validate_resume_optimizer_alignment(
        optimizer,
        scheduler,
        training,
        global_step=int(state.get("global_step", 0)),
    )
    _collect_distributed_phase(
        "formal resume-alignment logging",
        lambda: append_jsonl(
            metrics_path,
            {
                "event": "v2_resume_optimizer_alignment",
                "created_at": utc_now(),
                **resume_alignment,
            },
        )
        if is_rank_zero()
        else None,
    )
    result = _train_loop(
        model=model,
        runtime=runtime,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        train=train,
        output_dir=output_dir,
        metrics_path=metrics_path,
        identity_hashes=identity_hashes,
        state=state,
        deferred_rng=deferred_rng,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    if isinstance(result.get("hosted_verification_stop_after_step"), int):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return result
    selected_model_identity = _load_v2_best_model(
        output_dir,
        model,
        device,
        expected_identity_hashes=identity_hashes,
        expected_best_step=result.get("best_step"),
    )
    result = _completed_training_result(result, selected_model_identity)
    final_runtime = build_multires_event_v2_runtime(
        train,
        dataset,
        repo_root=root,
        rank=rank,
        world_size=world_size,
        phase="final",
    )
    final_runtime = _bind_runtime_to_run_artifacts(output_dir, final_runtime)
    _assert_runtime_identity_stable(runtime.identity, final_runtime.identity)
    final_evaluation_identity = _evaluation_contract_identity(
        final_runtime,
        source_identity=source_identity,
        identity_hashes=identity_hashes,
        selected_model_identity=selected_model_identity,
    )
    selected_step = int(selected_model_identity["selected_checkpoint_step"])
    final_evaluation = evaluate_teacher_forced(
        model=model,
        loader=final_runtime.eval_loader,
        registry=final_runtime.contract.process_registry,
        device=device,
        mode=str(train["mode"]),
        expected_samples=int(train["evaluation"]["final_expected_samples"]),
        phase="final",
        step=selected_step,
        precision=str(training["precision"]),
        metrics_path=metrics_path,
        expected_lab_scale_artifact_hash=train.get("lab_scale_artifact_hash"),
        per_anchor_output_path=output_dir / "val_per_anchor_joint_nll.jsonl",
        evaluation_identity=final_evaluation_identity,
    )
    free_running_evaluation: dict[str, Any] | None = None
    if bool(train["evaluation"]["free_running_final"]):
        promotion_metric_contract = load_promotion_metric_contract(
            output_dir / RUN_ARTIFACT_PATHS["promotion_metric_contract"],
            expected_sha256=str(train["promotion_metric_contract_hash"]),
            data_contract=final_runtime.contract,
        )
        free_running_evaluation = evaluate_free_running_v2(
            model=model,
            loader=final_runtime.eval_loader,
            contract=final_runtime.contract,
            device=device,
            mode=str(train["mode"]),
            expected_samples=int(train["evaluation"]["final_expected_samples"]),
            step=selected_step,
            output_dir=output_dir / "free_running",
            expected_lab_scale_artifact_hash=str(train["lab_scale_artifact_hash"]),
            standardized_primitive_scale_path=(
                output_dir / RUN_ARTIFACT_PATHS["standardized_primitive_scale"]
            ),
            expected_standardized_primitive_scale_hash=str(
                train["standardized_primitive_scale_artifact_hash"]
            ),
            input_normalization_sha256=str(
                final_runtime.identity["input_normalization_sha256"]
            ),
            promotion_metric_contract=promotion_metric_contract,
            evaluation_identity=final_evaluation_identity,
            trajectories_per_anchor=int(
                train["evaluation"]["free_running_trajectories_per_anchor"]
            ),
            trajectory_batch_size=int(
                train["evaluation"]["free_running_trajectory_batch_size"]
            ),
            crn_seed=int(train["evaluation"]["free_running_crn_seed"]),
            metrics_path=metrics_path,
            precision=str(training["precision"]),
        )
    _collect_distributed_phase(
        "formal final export",
        lambda: _export_run(
            output_dir,
            model,
            train,
            identity_hashes,
            result,
            final_evaluation,
            free_running_evaluation,
            selected_model_identity,
        )
        if is_rank_zero()
        else None,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return {
        **result,
        "final_evaluation": final_evaluation,
        "free_running_evaluation": free_running_evaluation,
    }


def _train_loop(
    *,
    model: Any,
    runtime: MultiresEventV2Runtime,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    train: Mapping[str, Any],
    output_dir: Path,
    metrics_path: Path,
    identity_hashes: Mapping[str, str],
    state: dict[str, Any],
    deferred_rng: Mapping[str, Any] | None,
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    training = train["training"]
    max_steps = int(training["max_steps"])
    accumulation = int(training["gradient_accumulation_steps"])
    expected_local_batch = int(training["per_device_train_batch_size"])
    expected_world_size = int(training["required_world_size"])
    if (
        expected_local_batch != 32
        or expected_world_size != 2
        or world_size != expected_world_size
        or expected_local_batch * expected_world_size * accumulation != 64
    ):
        raise RuntimeError("formal V2 optimizer loop requires exact B32/rank, world size 2, B64")
    global_step = int(state.get("global_step", 0))
    verification_stop_step = _hosted_verification_stop_step(
        starting_global_step=global_step
    )
    epoch = int(state.get("epoch", 0))
    batches_in_epoch = int(state.get("batches_in_epoch", 0))
    micro_in_accum = int(state.get("micro_in_accum", 0))
    best_metric = state.get("best_metric")
    best_step = state.get("best_step")
    scaler_skipped_steps = int(state.get("scaler_skipped_steps", 0))
    consecutive_scaler_skips = 0
    _set_sampler_epoch(runtime.train_sampler, epoch)
    iterator = iter(runtime.train_loader)
    for _ in range(batches_in_epoch):
        try:
            next(iterator)
        except StopIteration as exc:
            raise RuntimeError("V2 resume cursor exceeds deterministic subject-uniform epoch") from exc
    if deferred_rng is not None:
        _restore_rng_state(deferred_rng)
    optimizer.zero_grad(set_to_none=True)
    interval_sum = 0.0
    interval_count = 0
    pending_interval_sum = 0.0
    pending_interval_count = 0
    model.train()
    autocast = _autocast_factory(device, str(training["precision"]))
    while global_step < max_steps:
        try:
            raw_batch = next(iterator)
            batches_in_epoch += 1
        except StopIteration:
            epoch += 1
            batches_in_epoch = 0
            _set_sampler_epoch(runtime.train_sampler, epoch)
            iterator = iter(runtime.train_loader)
            continue
        batch = move_to_device(raw_batch, device)
        local_anchors = len(raw_batch.get("sample_id") or ())
        if local_anchors != expected_local_batch:
            raise RuntimeError("formal V2 training requires exactly 32 local anchors per rank")
        micro_in_accum += 1
        synchronize = micro_in_accum == accumulation
        sync_context = (
            model.no_sync()
            if world_size > 1 and not synchronize and hasattr(model, "no_sync")
            else nullcontext()
        )
        with sync_context:
            _, loss_result = exact_teacher_forced_loss(
                model,
                batch,
                runtime.contract.process_registry,
                mode=str(train["mode"]),
                expected_lab_scale_artifact_hash=train.get("lab_scale_artifact_hash"),
                autocast=autocast,
            )
            loss = _validated_optimizer_loss(
                loss_result, expected_local_batch=local_anchors
            )
            scaler.scale(loss).backward()
        per_sample = loss_result["per_sample_nll"].detach()
        pending_interval_sum += float(per_sample.sum().cpu().item())
        pending_interval_count += int(per_sample.numel())
        if not synchronize:
            continue
        scaler.unscale_(optimizer)
        (
            optimizer_updated,
            scale_before,
            scale_after,
            gradient_health,
            optimizer_state_health,
        ) = _audited_optimizer_step(
            model,
            optimizer,
            scaler,
            expected_optimizer_step=global_step + 1,
        )
        micro_in_accum = 0
        if not optimizer_updated:
            optimizer.zero_grad(set_to_none=True)
            pending_interval_sum = 0.0
            pending_interval_count = 0
            scaler_skipped_steps += 1
            consecutive_scaler_skips += 1
            if is_rank_zero():
                append_jsonl(
                    metrics_path,
                    {
                        "event": "v2_grad_scaler_skip",
                        "created_at": utc_now(),
                        "successful_optimizer_updates": global_step,
                        "attempted_scale": scale_before,
                        "next_scale": scale_after,
                        "consecutive_skips": consecutive_scaler_skips,
                        "total_skips": scaler_skipped_steps,
                    },
                )
            if consecutive_scaler_skips > int(training["max_consecutive_scaler_skips"]):
                raise FloatingPointError(
                    "V2 FP16 gradient overflow invalidates the matched run: retrying "
                    "would consume a different stochastic row/RNG schedule across modes; "
                    "no nominal training step was counted"
                )
            continue
        if optimizer_state_health is None:
            raise AssertionError("successful formal optimizer step lacks state health")
        step_health_payload = _optimizer_step_health_payload(
            optimizer,
            gradient_health,
            optimizer_state_health,
            training=training,
        )
        optimizer.zero_grad(set_to_none=True)
        consecutive_scaler_skips = 0
        scheduler.step()
        interval_sum += pending_interval_sum
        interval_count += pending_interval_count
        pending_interval_sum = 0.0
        pending_interval_count = 0
        global_step += 1
        if is_rank_zero():
            append_jsonl(
                metrics_path,
                {
                    "event": "v2_optimizer_health",
                    "created_at": utc_now(),
                    "step": global_step,
                    "local_anchors": local_anchors,
                    "world_size": world_size,
                    "global_anchors": local_anchors * world_size,
                    **step_health_payload,
                    "scaler_scale_before": scale_before,
                    "scaler_scale_after": scale_after,
                    "scaler_skipped_steps": scaler_skipped_steps,
                },
            )

        if global_step % int(training["logging_steps"]) == 0 or global_step == max_steps:
            total, count = _distributed_sum_count(interval_sum, interval_count, device)
            if is_rank_zero():
                value = total / count
                print(f"V2_TRAIN_NLL step={global_step} anchor_mean={value:.6f}", flush=True)
                append_jsonl(metrics_path, {
                    "event": "v2_train_joint_nll",
                    "created_at": utc_now(),
                    "step": global_step,
                    "joint_nll_anchor_mean": value,
                    "anchors": int(count),
                    "primitive_factors_per_anchor": 414,
                })
            interval_sum, interval_count = 0.0, 0

        if global_step % int(training["eval_steps"]) == 0 or global_step == max_steps:
            evaluation = evaluate_teacher_forced(
                model=model,
                loader=runtime.eval_loader,
                registry=runtime.contract.process_registry,
                device=device,
                mode=str(train["mode"]),
                expected_samples=int(train["evaluation"]["interval_expected_samples"]),
                phase="interval",
                step=global_step,
                precision=str(training["precision"]),
                metrics_path=metrics_path,
                expected_lab_scale_artifact_hash=train.get("lab_scale_artifact_hash"),
                evaluation_identity=_evaluation_contract_identity(runtime),
            )
            candidate = float(evaluation["joint_nll_subject_macro"])
            if best_metric is None or candidate < float(best_metric):
                best_metric, best_step = candidate, global_step
                _materialize_v2_best_model(
                    output_dir=output_dir,
                    model=model,
                    identity_hashes=identity_hashes,
                    step=global_step,
                    metric=candidate,
                )
            model.train()

        initial_checkpoint_step = training.get("initial_checkpoint_step")
        if (
            global_step % int(training["save_steps"]) == 0
            or global_step == max_steps
            or (
                isinstance(initial_checkpoint_step, int)
                and not isinstance(initial_checkpoint_step, bool)
                and global_step == initial_checkpoint_step
            )
            or global_step == verification_stop_step
        ):
            _save_v2_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                trainer_state={
                    "global_step": global_step,
                    "epoch": epoch,
                    "batches_in_epoch": batches_in_epoch,
                    "micro_in_accum": micro_in_accum,
                    "best_metric": best_metric,
                    "best_step": best_step,
                    "scaler_skipped_steps": scaler_skipped_steps,
                },
                identity_hashes=identity_hashes,
                runtime=runtime,
                rank=rank,
                keep_last=int(training["keep_last_checkpoints"]),
            )
            if global_step == initial_checkpoint_step and is_rank_zero():
                checkpoint = output_dir / "checkpoints" / f"checkpoint-{global_step:08d}"
                manifest_path = checkpoint / "checkpoint_manifest.json"
                readiness = {
                    "schema_version": "trauma_predict.multires_event_v2_formal_step2_readiness.v1",
                    "status": "PASSED",
                    "created_at": utc_now(),
                    "mode": str(train["mode"]),
                    "run_name": str(train["run_name"]),
                    "global_step": global_step,
                    "checkpoint": str(checkpoint.relative_to(output_dir)),
                    "checkpoint_manifest_sha256": sha256_file(manifest_path),
                    "model_parameter_count": EXPECTED_FORMAL_MODEL_PARAMETER_COUNT,
                    "target_dataset_id": EXPECTED_TARGET_DATASET_ID,
                    "contract_bundle_hash": EXPECTED_CONTRACT_BUNDLE_HASH,
                    "identity_hashes": dict(identity_hashes),
                }
                atomic_write_json(output_dir / "formal_step2_readiness.json", readiness)
                print(
                    "MULTIRES_EVENT_V2_FORMAL_STEP2_CHECKPOINT_OK "
                    f"mode={train['mode']} parameters={EXPECTED_FORMAL_MODEL_PARAMETER_COUNT} "
                    f"path={checkpoint}",
                    flush=True,
                )
            if global_step == 3 and verification_stop_step == 3 and is_rank_zero():
                checkpoint = output_dir / "checkpoints/checkpoint-00000003"
                manifest_path = checkpoint / "checkpoint_manifest.json"
                atomic_write_json(
                    output_dir / "formal_resume_step3_readiness.json",
                    {
                        "schema_version": (
                            "trauma_predict.multires_event_v2_formal_resume_step3_readiness.v1"
                        ),
                        "status": "PASSED",
                        "created_at": utc_now(),
                        "mode": str(train["mode"]),
                        "run_name": str(train["run_name"]),
                        "restored_from_step": 2,
                        "global_step": 3,
                        "checkpoint": str(checkpoint.relative_to(output_dir)),
                        "checkpoint_manifest_sha256": sha256_file(manifest_path),
                        "model_parameter_count": EXPECTED_FORMAL_MODEL_PARAMETER_COUNT,
                        "target_dataset_id": EXPECTED_TARGET_DATASET_ID,
                        "contract_bundle_hash": EXPECTED_CONTRACT_BUNDLE_HASH,
                        "identity_hashes": dict(identity_hashes),
                    },
                )
                print(
                    "MULTIRES_EVENT_V2_FORMAL_RESUME_STEP3_CHECKPOINT_OK "
                    f"mode={train['mode']} parameters={EXPECTED_FORMAL_MODEL_PARAMETER_COUNT} "
                    f"path={checkpoint}",
                    flush=True,
                )
            if verification_stop_step is not None and global_step == verification_stop_step:
                _collect_distributed_phase(
                    f"formal hosted-verification step-{global_step} close",
                    lambda: (
                        _validate_formal_step2_readiness(
                            output_dir,
                            expected_step=global_step,
                            expected_mode=str(train["mode"]),
                        )
                        if global_step == 2
                        else _validate_formal_resume_step3_readiness(
                            output_dir,
                            expected_mode=str(train["mode"]),
                        )
                    )
                    if is_rank_zero()
                    else None,
                )
                return {
                    "global_step": global_step,
                    "epochs_started": epoch + 1,
                    "best_metric": best_metric,
                    "best_step": best_step,
                    "max_steps": max_steps,
                    "scaler_skipped_steps": scaler_skipped_steps,
                    "hosted_verification_stop_after_step": global_step,
                }
    return {
        "global_step": global_step,
        "epochs_started": epoch + 1,
        "best_metric": best_metric,
        "best_step": best_step,
        "max_steps": max_steps,
        "scaler_skipped_steps": scaler_skipped_steps,
    }


def _verification_stop_after_formal_step2_requested() -> bool:
    value = os.environ.get(
        "TRAUMA_PREDICT_V2_HOSTED_VERIFY_STOP_AFTER_FORMAL_STEP2", "0"
    ).strip()
    if value not in {"0", "1"}:
        raise ValueError(
            "TRAUMA_PREDICT_V2_HOSTED_VERIFY_STOP_AFTER_FORMAL_STEP2 must be 0 or 1"
        )
    return value == "1"


def _verification_stop_after_resume_step3_requested() -> bool:
    value = os.environ.get(
        "TRAUMA_PREDICT_V2_HOSTED_VERIFY_STOP_AFTER_RESUME_STEP3", "0"
    ).strip()
    if value not in {"0", "1"}:
        raise ValueError(
            "TRAUMA_PREDICT_V2_HOSTED_VERIFY_STOP_AFTER_RESUME_STEP3 must be 0 or 1"
        )
    return value == "1"


def _hosted_verification_stop_step(*, starting_global_step: int) -> int | None:
    stop_at_2 = _verification_stop_after_formal_step2_requested()
    stop_at_3 = _verification_stop_after_resume_step3_requested()
    if stop_at_2 and stop_at_3:
        raise ValueError("hosted verification stop modes are mutually exclusive")
    if stop_at_2:
        if starting_global_step != 0:
            raise ValueError("formal step-2 verification must start from optimizer step 0")
        return 2
    if stop_at_3:
        if starting_global_step != 2:
            raise ValueError("resume step-3 verification must restore optimizer step 2")
        return 3
    return None


def _validate_formal_step2_readiness(
    output_dir: Path,
    *,
    expected_step: int,
    expected_mode: str,
) -> dict[str, Any]:
    readiness_path = output_dir / "formal_step2_readiness.json"
    if readiness_path.is_symlink() or not readiness_path.is_file():
        raise FileNotFoundError("formal step-2 readiness evidence is absent")
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    checkpoint = output_dir / str(readiness.get("checkpoint") or "")
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = _validate_v2_checkpoint_directory(
        checkpoint,
        expected_world_size=2,
        expected_step=expected_step,
    )
    if (
        readiness.get("schema_version")
        != "trauma_predict.multires_event_v2_formal_step2_readiness.v1"
        or readiness.get("status") != "PASSED"
        or readiness.get("mode") != expected_mode
        or readiness.get("run_name") != "t4x2_multires_event_v2_relational"
        or int(readiness.get("global_step", -1)) != expected_step
        or int(readiness.get("model_parameter_count", -1))
        != EXPECTED_FORMAL_MODEL_PARAMETER_COUNT
        or readiness.get("target_dataset_id") != EXPECTED_TARGET_DATASET_ID
        or readiness.get("contract_bundle_hash") != EXPECTED_CONTRACT_BUNDLE_HASH
        or readiness.get("checkpoint_manifest_sha256") != sha256_file(manifest_path)
        or readiness.get("identity_hashes") != manifest.get("identity_hashes")
    ):
        raise ValueError("formal step-2 readiness evidence does not bind its checkpoint")
    return readiness


def _validate_formal_resume_step3_readiness(
    output_dir: Path,
    *,
    expected_mode: str,
) -> dict[str, Any]:
    readiness_path = output_dir / "formal_resume_step3_readiness.json"
    if readiness_path.is_symlink() or not readiness_path.is_file():
        raise FileNotFoundError("formal resume step-3 readiness evidence is absent")
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    checkpoint = output_dir / str(readiness.get("checkpoint") or "")
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = _validate_v2_checkpoint_directory(
        checkpoint,
        expected_world_size=2,
        expected_step=3,
    )
    if (
        readiness.get("schema_version")
        != "trauma_predict.multires_event_v2_formal_resume_step3_readiness.v1"
        or readiness.get("status") != "PASSED"
        or readiness.get("mode") != expected_mode
        or readiness.get("run_name") != "t4x2_multires_event_v2_relational"
        or int(readiness.get("restored_from_step", -1)) != 2
        or int(readiness.get("global_step", -1)) != 3
        or int(readiness.get("model_parameter_count", -1))
        != EXPECTED_FORMAL_MODEL_PARAMETER_COUNT
        or readiness.get("target_dataset_id") != EXPECTED_TARGET_DATASET_ID
        or readiness.get("contract_bundle_hash") != EXPECTED_CONTRACT_BUNDLE_HASH
        or readiness.get("checkpoint_manifest_sha256") != sha256_file(manifest_path)
        or readiness.get("identity_hashes") != manifest.get("identity_hashes")
    ):
        raise ValueError("formal resume step-3 evidence does not bind its checkpoint")
    return readiness


def _save_v2_checkpoint(
    *,
    output_dir: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    trainer_state: Mapping[str, Any],
    identity_hashes: Mapping[str, str],
    runtime: Any,
    rank: int,
    keep_last: int,
) -> None:
    """Persist one V2 checkpoint without asymmetric rank/barrier deadlocks."""

    step = int(trainer_state["global_step"])
    world_size = _world_size()
    checkpoint_root = output_dir / "checkpoints"
    checkpoint = checkpoint_root / f"checkpoint-{step:08d}"
    partial = checkpoint_root / f".checkpoint-{step:08d}.partial"

    def prepare_partial() -> None:
        if not is_rank_zero():
            return
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        if checkpoint.exists():
            raise FileExistsError(
                f"refusing to overwrite completed V2 checkpoint {checkpoint}"
            )
        if partial.exists():
            abandoned_root = checkpoint_root / "incomplete"
            abandoned_root.mkdir(parents=True, exist_ok=True)
            timestamp = utc_now().replace(":", "-")
            partial.rename(
                abandoned_root / f"{partial.name}-{timestamp}-pid{os.getpid()}"
            )
        partial.mkdir(parents=False, exist_ok=False)

    _collect_distributed_phase("V2 checkpoint partial preparation", prepare_partial)

    def save_rank_state() -> dict[str, str]:
        rng_name = f"rng-rank-{rank:04d}.pt"
        sampler_name = f"sampler-rank-{rank:04d}.pt"
        torch.save(_capture_rng_state(), partial / rng_name)
        sampler_state = (
            runtime.train_sampler.state_dict()
            if hasattr(runtime.train_sampler, "state_dict")
            else None
        )
        torch.save(sampler_state, partial / sampler_name)
        return {
            rng_name: sha256_file(partial / rng_name),
            sampler_name: sha256_file(partial / sampler_name),
        }

    rank_hashes = _collect_distributed_phase(
        "V2 checkpoint rank-local state",
        save_rank_state,
    )

    def finalize_checkpoint() -> None:
        if not is_rank_zero():
            return
        shared_writers = {
            "model.pt": lambda path: torch.save(
                _unwrapped_model(model).state_dict(), path
            ),
            "optimizer.pt": lambda path: torch.save(optimizer.state_dict(), path),
            "scheduler.pt": lambda path: torch.save(scheduler.state_dict(), path),
            "scaler.pt": lambda path: torch.save(scaler.state_dict(), path),
            "trainer_state.json": lambda path: atomic_write_json(
                path, dict(trainer_state)
            ),
            "identity_hashes.json": lambda path: atomic_write_json(
                path, dict(identity_hashes)
            ),
        }
        hashes: dict[str, str] = {}
        for name, writer in shared_writers.items():
            writer(partial / name)
            hashes[name] = sha256_file(partial / name)
        for payload in rank_hashes:
            hashes.update({str(name): str(digest) for name, digest in payload.items()})
        files = tuple(sorted(hashes))
        atomic_write_json(
            partial / "checkpoint_manifest.json",
            {
                "schema_version": V2_CHECKPOINT_SCHEMA,
                "created_at": utc_now(),
                "global_step": step,
                "world_size": world_size,
                "identity_hashes": dict(identity_hashes),
                "files": list(files),
                "sha256": {name: hashes[name] for name in files},
            },
        )
        _validate_v2_checkpoint_directory(
            partial,
            expected_world_size=world_size,
            expected_step=step,
        )
        partial.replace(checkpoint)

    _collect_distributed_phase("V2 checkpoint shared finalization", finalize_checkpoint)
    _collect_distributed_phase(
        "V2 checkpoint pruning",
        lambda: _prune_checkpoints(checkpoint_root, keep_last)
        if is_rank_zero()
        else None,
    )


def _validate_v2_checkpoint_directory(
    checkpoint: Path,
    *,
    expected_world_size: int,
    expected_step: int,
) -> dict[str, Any]:
    manifest_path = checkpoint / "checkpoint_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(f"V2 checkpoint manifest is absent: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    hashes = manifest.get("sha256")
    expected_rank_files = {
        *(f"rng-rank-{rank:04d}.pt" for rank in range(expected_world_size)),
        *(f"sampler-rank-{rank:04d}.pt" for rank in range(expected_world_size)),
    }
    expected_files = {
        "model.pt",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
        "trainer_state.json",
        "identity_hashes.json",
        *expected_rank_files,
    }
    if (
        manifest.get("schema_version") != V2_CHECKPOINT_SCHEMA
        or int(manifest.get("global_step", -1)) != expected_step
        or int(manifest.get("world_size", -1)) != expected_world_size
        or not isinstance(files, list)
        or set(str(name) for name in files) != expected_files
        or not isinstance(hashes, Mapping)
        or set(str(name) for name in hashes) != expected_files
    ):
        raise ValueError(f"V2 checkpoint manifest contract failed: {checkpoint}")
    for name in sorted(expected_files):
        path = checkpoint / name
        digest = str(hashes[name])
        if (
            Path(name).name != name
            or path.is_symlink()
            or not path.is_file()
            or not _is_sha256(digest)
            or sha256_file(path) != digest
        ):
            raise ValueError(f"V2 checkpoint file/hash failed: {path}")
    return manifest


def _validate_v2_checkpoint_integrity(
    output_dir: Path,
    *,
    expected_world_size: int,
) -> None:
    checkpoint_root = output_dir / "checkpoints"
    if not checkpoint_root.is_dir():
        return
    for checkpoint in sorted(checkpoint_root.glob("checkpoint-*")):
        if not checkpoint.is_dir():
            raise ValueError(f"V2 checkpoint entry is not a directory: {checkpoint}")
        try:
            step = int(checkpoint.name.removeprefix("checkpoint-"))
        except ValueError as error:
            raise ValueError(f"invalid V2 checkpoint directory name: {checkpoint}") from error
        _validate_v2_checkpoint_directory(
            checkpoint,
            expected_world_size=expected_world_size,
            expected_step=step,
        )


def _audit_unscaled_gradients(
    model: torch.nn.Module,
) -> tuple[dict[str, Any], _OptimizerUpdateProbe]:
    """Read-only audit of the exact unscaled gradient set before AdamW."""

    named_parameters = tuple(
        (name, parameter)
        for name, parameter in _unwrapped_model(model).named_parameters()
        if parameter.requires_grad
    )
    if not named_parameters:
        raise RuntimeError("V2 gradient audit found no trainable parameters")
    missing = [name for name, parameter in named_parameters if parameter.grad is None]
    if missing:
        raise RuntimeError(
            "V2 gradient audit requires every trainable parameter gradient; missing: "
            + ", ".join(missing[:8])
        )
    parameters = [parameter for _, parameter in named_parameters]
    audit_device = parameters[0].device
    if any(parameter.device != audit_device for parameter in parameters):
        raise RuntimeError("V2 optimizer audit requires one parameter device per rank")
    gradients = [parameter.grad for parameter in parameters]
    if any(gradient is None or gradient.is_sparse for gradient in gradients):
        raise RuntimeError("V2 optimizer audit requires dense gradients for every parameter")
    dense_gradients = [gradient for gradient in gradients if gradient is not None]
    finite_flags = torch.stack(
        [torch.isfinite(gradient.detach()).all() for gradient in dense_gradients]
    )
    squared_norms = torch.stack(
        [gradient.detach().float().square().sum() for gradient in dense_gradients]
    )
    maximums = torch.stack(
        [gradient.detach().float().abs().amax() for gradient in dense_gradients]
    )
    selected_parameter_index = maximums.argmax()
    summary = torch.stack(
        (
            finite_flags.all().to(dtype=torch.float32),
            squared_norms.sum().sqrt(),
            maximums.max(),
            selected_parameter_index.to(dtype=torch.float32),
        )
    ).detach().cpu().tolist()
    all_finite = bool(summary[0])
    global_l2_norm = float(summary[1])
    maximum_gradient = float(summary[2])
    parameter_index = int(summary[3])
    if not all_finite or not math.isfinite(global_l2_norm):
        raise FloatingPointError("V2 unscaled gradients or their global L2 norm are non-finite")
    if global_l2_norm <= 0.0 or maximum_gradient <= 0.0:
        raise FloatingPointError("V2 unscaled global gradient L2 norm must be positive")

    probe_name, probe_parameter = named_parameters[parameter_index]
    probe_gradient = probe_parameter.grad
    if probe_gradient is None:
        raise AssertionError("selected optimizer probe unexpectedly lacks a gradient")
    probe_max, probe_flat_index = probe_gradient.detach().float().abs().reshape(-1).max(dim=0)
    probe_gradient_abs = float(probe_max.cpu().item())
    flat_index = int(probe_flat_index.cpu().item())
    if not math.isfinite(probe_gradient_abs) or probe_gradient_abs <= 0.0:
        raise FloatingPointError("V2 optimizer probe gradient must be finite and positive")
    probe = _OptimizerUpdateProbe(
        parameter_name=probe_name,
        parameter=probe_parameter,
        flat_index=flat_index,
        value_before=probe_parameter.detach().reshape(-1)[flat_index].clone(),
    )
    return {
        "optimizer_contract_version": OPTIMIZER_CONTRACT_VERSION,
        "trainable_parameter_tensors": len(named_parameters),
        "gradient_tensors": len(dense_gradients),
        "missing_gradient_tensors": 0,
        "all_gradients_finite": True,
        "global_l2_norm": global_l2_norm,
        "global_l2_positive": True,
        "gradient_clipping": "disabled",
        "gradient_modified_after_unscale": False,
        "probe_parameter": probe_name,
        "probe_flat_index": flat_index,
        "probe_gradient_abs": probe_gradient_abs,
    }, probe


def _audit_optimizer_state_after_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    probe: _OptimizerUpdateProbe,
    *,
    expected_optimizer_step: int,
) -> dict[str, Any]:
    """Verify AdamW parameters, moments, and exact optimizer-step state."""

    if (
        isinstance(expected_optimizer_step, bool)
        or not isinstance(expected_optimizer_step, int)
        or expected_optimizer_step < 1
    ):
        raise ValueError("expected_optimizer_step must be a positive integer")

    optimizer_configuration = _validate_built_optimizer_contract(optimizer)
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in _unwrapped_model(model).named_parameters()
        if parameter.requires_grad
    )
    parameters = [parameter for _, parameter in named_parameters]
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    ]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    parameter_ids = [id(parameter) for parameter in parameters]
    if len(set(optimizer_ids)) != len(optimizer_ids) or set(optimizer_ids) != set(parameter_ids):
        raise RuntimeError("V2 AdamW parameter set does not exactly equal the trainable model set")
    if len(optimizer.state) != len(parameters):
        raise RuntimeError("V2 AdamW state is incomplete after the optimizer step")
    probe_matches = [
        name == probe.parameter_name and parameter is probe.parameter
        for name, parameter in named_parameters
    ]
    if sum(probe_matches) != 1:
        raise RuntimeError("V2 optimizer update probe no longer identifies one parameter")

    parameter_finite: list[torch.Tensor] = []
    exp_avg_finite: list[torch.Tensor] = []
    exp_avg_sq_finite: list[torch.Tensor] = []
    exp_avg_sq_nonnegative: list[torch.Tensor] = []
    exp_avg_sq_minimums: list[torch.Tensor] = []
    cuda_step_checks: list[torch.Tensor] = []
    cuda_step_values: list[torch.Tensor] = []
    cpu_step_values: list[float] = []
    cpu_steps_valid = True
    for name, parameter in named_parameters:
        state = optimizer.state.get(parameter)
        if not isinstance(state, Mapping) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
            raise RuntimeError(f"V2 AdamW state is incomplete for {name}")
        step = state["step"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        if not isinstance(step, torch.Tensor) or step.numel() != 1:
            raise RuntimeError(f"V2 AdamW step state is invalid for {name}")
        if not isinstance(exp_avg, torch.Tensor) or exp_avg.shape != parameter.shape:
            raise RuntimeError(f"V2 AdamW exp_avg shape is invalid for {name}")
        if not isinstance(exp_avg_sq, torch.Tensor) or exp_avg_sq.shape != parameter.shape:
            raise RuntimeError(f"V2 AdamW exp_avg_sq shape is invalid for {name}")
        step_check = torch.isfinite(step.detach()).all() & (
            step.detach() == expected_optimizer_step
        ).all()
        if step.device.type == "cuda":
            cuda_step_checks.append(step_check.to(device=parameter.device))
            cuda_step_values.append(step.detach().float().to(device=parameter.device))
        else:
            step_value = float(step.detach().item())
            cpu_step_values.append(step_value)
            cpu_steps_valid = cpu_steps_valid and bool(step_check.item())
        parameter_finite.append(torch.isfinite(parameter.detach()).all())
        exp_avg_finite.append(torch.isfinite(exp_avg.detach()).all())
        exp_avg_sq_finite.append(torch.isfinite(exp_avg_sq.detach()).all())
        exp_avg_sq_nonnegative.append((exp_avg_sq.detach() >= 0).all())
        exp_avg_sq_minimums.append(exp_avg_sq.detach().float().amin())

    audit_device = parameters[0].device
    cuda_steps_valid = (
        torch.stack(cuda_step_checks).all()
        if cuda_step_checks
        else torch.ones((), dtype=torch.bool, device=audit_device)
    )
    observed_step_values = list(cpu_step_values)
    if cuda_step_values:
        cuda_values = torch.stack(cuda_step_values)
        observed_step_values.extend(
            float(value)
            for value in torch.stack((cuda_values.amin(), cuda_values.amax()))
            .detach()
            .cpu()
            .tolist()
        )
    observed_step_min = min(observed_step_values)
    observed_step_max = max(observed_step_values)
    state_steps_match = (
        cpu_steps_valid
        and bool(cuda_steps_valid.detach().cpu().item())
        and observed_step_min == float(expected_optimizer_step)
        and observed_step_max == float(expected_optimizer_step)
    )
    probe_after = probe.parameter.detach().reshape(-1)[probe.flat_index]
    flags = torch.stack(
        (
            torch.stack(parameter_finite).all(),
            torch.stack(exp_avg_finite).all(),
            torch.stack(exp_avg_sq_finite).all(),
            torch.stack(exp_avg_sq_nonnegative).all(),
            cuda_steps_valid,
            probe_after.ne(probe.value_before),
        )
    )
    summary = torch.cat(
        (
            flags.to(dtype=torch.float32),
            torch.stack(exp_avg_sq_minimums).amin().reshape(1),
            probe.value_before.float().reshape(1),
            probe_after.float().reshape(1),
        )
    ).detach().cpu().tolist()
    health = {
        "optimizer_contract_version": OPTIMIZER_CONTRACT_VERSION,
        "trainable_parameter_tensors": len(parameters),
        "optimizer_state_entries": len(optimizer.state),
        "state_complete": True,
        "expected_optimizer_step": expected_optimizer_step,
        "observed_optimizer_step_min": observed_step_min,
        "observed_optimizer_step_max": observed_step_max,
        "state_steps_complete_equal_expected": state_steps_match and bool(summary[4]),
        "parameters_finite": bool(summary[0]),
        "exp_avg_finite": bool(summary[1]),
        "exp_avg_sq_finite": bool(summary[2]),
        "exp_avg_sq_nonnegative": bool(summary[3]),
        "exp_avg_sq_minimum": float(summary[6]),
        "probe_parameter": probe.parameter_name,
        "probe_flat_index": probe.flat_index,
        "probe_value_before": float(summary[7]),
        "probe_value_after": float(summary[8]),
        "probe_parameter_changed": bool(summary[5]),
        "optimizer_updated": True,
        "optimizer_configuration": optimizer_configuration,
    }
    required = (
        "state_steps_complete_equal_expected",
        "parameters_finite",
        "exp_avg_finite",
        "exp_avg_sq_finite",
        "exp_avg_sq_nonnegative",
    )
    failed = [key for key in required if health[key] is not True]
    if failed or not math.isfinite(float(health["exp_avg_sq_minimum"])):
        raise FloatingPointError(
            "V2 post-step AdamW health audit failed: " + ", ".join(failed or ["moment_minimum"])
        )
    return health


def _audited_optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    *,
    expected_optimizer_step: int,
) -> tuple[bool, float, float, dict[str, Any], dict[str, Any] | None]:
    audit_started = time.monotonic()
    gradient_health, probe = _audit_unscaled_gradients(model)
    gradient_health["audit_wall_seconds"] = time.monotonic() - audit_started
    updated, scale_before, scale_after = _step_grad_scaler(scaler, optimizer)
    if scale_before != 32.0 or scale_after != 32.0:
        raise FloatingPointError(
            "V2 GradScaler must remain exactly 32.0 before and after every optimizer step"
        )
    state_health = None
    if updated:
        audit_started = time.monotonic()
        state_health = _audit_optimizer_state_after_step(
            model,
            optimizer,
            probe,
            expected_optimizer_step=expected_optimizer_step,
        )
        state_health["audit_wall_seconds"] = time.monotonic() - audit_started
    return updated, scale_before, scale_after, gradient_health, state_health


def _step_grad_scaler(scaler: Any, optimizer: Any) -> tuple[bool, float, float]:
    """Step AMP once and report whether an optimizer update actually occurred.

    PyTorch lowers the scale only when ``GradScaler.step`` skipped the optimizer
    because an Inf/NaN gradient was found.  V2 uses successful optimizer
    updates—not attempted FP16 batches—as its frozen 4,000-step budget.
    """

    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    return scale_after >= scale_before, scale_before, scale_after


def _verify_artifact_files(
    base_root: Path,
    target_root: Path,
    supervision_path: Path,
    dataset: Mapping[str, Any],
) -> None:
    checks = (
        (base_root / "dataset_manifest.json", EXPECTED_BASE_MANIFEST_SHA256, "V1 manifest"),
        (base_root / "sample_manifest.csv", EXPECTED_BASE_SAMPLE_MANIFEST_SHA256, "V1 sample manifest"),
        (base_root / "subject_split.csv", EXPECTED_SUBJECT_SPLIT_SHA256, "patient split"),
        (target_root / "dataset_manifest.json", EXPECTED_TARGET_MANIFEST_SHA256, "full_r9 manifest"),
        (target_root / "sample_manifest.csv", EXPECTED_TARGET_SAMPLE_MANIFEST_SHA256, "full_r9 sample manifest"),
        (
            target_root / "contracts/target_process_registry_v2.json",
            EXPECTED_PROCESS_CONTRACT_SHA256,
            "V2 process contract",
        ),
        (
            target_root / "contracts/target_emission_registry_v2.json",
            EXPECTED_EMISSION_CONTRACT_SHA256,
            "V2 emission contract",
        ),
        (
            target_root / "contracts/target_projection_registry_v2.json",
            EXPECTED_PROJECTION_CONTRACT_SHA256,
            "V2 projection contract",
        ),
        (supervision_path, str(dataset["supervision_sha256"]), "V1 input overlay"),
    )
    for path, expected, label in checks:
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"{label} hash mismatch: {observed} != {expected}")


def load_lab_scale_artifact(
    path: str | Path,
    *,
    expected_content_sha256: str,
    contract: MultiresEventV2Contract,
) -> dict[str, Any]:
    """Validate the train-only affine artifact and return loss-facing metadata."""

    artifact_path = Path(path).resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"missing configured V2 lab scale artifact: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("V2 lab scale artifact must be a JSON object")
    canonical = json.dumps(
        {key: value for key, value in payload.items() if key != "content_sha256"},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    observed_hash = hashlib.sha256(canonical).hexdigest()
    if payload.get("content_sha256") != observed_hash:
        raise ValueError("V2 lab scale self hash differs from its canonical JSON content")
    if observed_hash != expected_content_sha256:
        raise ValueError("V2 lab scale differs from the training-config run identity")
    required = {
        "schema": "multires_event_v2_lab_affine_scale_v1",
        "version": "2026-07-13-train-target-windows-v1",
        "status": "frozen_train_only_fit",
        "fit_split": "train",
        "coordinate_contract": "lab_shared_affine_canonical_v1",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"V2 lab scale {key} must equal {expected!r}")
    transform = _mapping(payload.get("transform"), "lab_scale.transform")
    expected_transform = {
        "forward": "z=(x-center)/scale",
        "inverse": "x=center+scale*z",
        "clipping": "forbidden",
        "center": "linear_interpolation_median_of_fit_multiset",
        "scale": "q75_minus_q25_of_fit_multiset",
        "scale_fallback": "none_fail_if_nonpositive",
        "shared_coordinates": ["last", "min", "max"],
    }
    if dict(transform) != expected_transform:
        raise ValueError("V2 lab scale transform contract changed")
    population = _mapping(payload.get("fit_population"), "lab_scale.fit_population")
    population_contract = {
        "authority": "persisted_full_sidecar_train_target_shards",
        "physical_window_key": [
            "subject_id", "stay_id", "absolute_start_hour", "absolute_end_hour", "field"
        ],
        "duplicate_truth_policy": "require_exact_canonical_json_then_count_once",
        "coordinate_multiset_per_active_unique_window": ["last", "min", "max"],
    }
    for key, expected in population_contract.items():
        if population.get(key) != expected:
            raise ValueError(f"V2 lab scale fit_population.{key} changed")
    for key in ("train_samples", "train_subjects", "unique_physical_field_windows"):
        if int(population.get(key, 0)) < 1:
            raise ValueError(f"V2 lab scale fit_population.{key} must be positive")
    if int(population.get("collapsed_duplicate_field_windows", -1)) < 0:
        raise ValueError("V2 lab scale duplicate-window audit count is invalid")
    for key in ("train_subject_ids_sha256", "window_truth_ledger_sha256"):
        if not _is_sha256(population.get(key)):
            raise ValueError(f"V2 lab scale fit_population.{key} is not a SHA-256")

    source = _mapping(payload.get("source"), "lab_scale.source")
    source_expected = {
        "sidecar_dataset_id": str(contract.manifest["dataset_id"]),
        "sidecar_dataset_manifest_sha256": sha256_file(contract.dataset_root / "dataset_manifest.json"),
        "sidecar_sample_manifest_sha256": str(
            contract.manifest["files"]["sample_manifest"]["sha256"]
        ),
        "sidecar_contract_bundle_hash": contract.contract_bundle_hash,
        "sidecar_process_contract_sha256": contract.contract_hashes["process"],
        "sidecar_emission_contract_sha256": contract.contract_hashes["emission"],
        "process_registry_sha256": contract.contract_hashes["process"],
    }
    for key, expected in source_expected.items():
        if str(source.get(key)) != expected:
            raise ValueError(f"V2 lab scale source.{key} differs from attached sidecar")
    if not _is_sha256(source.get("v1_element_registry_sha256")):
        raise ValueError("V2 lab scale lacks a valid V1 element registry hash")

    field_order = tuple(str(value) for value in payload.get("field_order", ()))
    if field_order != tuple(contract.lab_fields):
        raise ValueError("V2 lab scale field order differs from the 13 registered labs")
    fields = _mapping(payload.get("fields"), "lab_scale.fields")
    if set(fields) != set(field_order):
        raise ValueError("V2 lab scale field rows do not match field_order")
    ids = dict(zip(contract.core_fields, contract.registered_core_field_ids, strict=True))
    supports = _mapping(
        _mapping(contract.emission_registry.get("field_supports"), "emission.field_supports").get(
            "intermittent_labs"
        ),
        "emission.field_supports.intermittent_labs",
    )
    compact_fields: dict[str, dict[str, Any]] = {}
    for field in field_order:
        row = _mapping(fields[field], f"lab_scale.fields.{field}")
        if row.get("field") != field or int(row.get("field_id", -1)) != ids[field]:
            raise ValueError(f"V2 lab scale identity mismatch for {field}")
        unit = str(_mapping(supports.get(field), f"lab support {field}").get("unit") or "")
        if str(row.get("unit") or "") != unit:
            raise ValueError(f"V2 lab scale unit mismatch for {field}")
        center = float(row.get("center"))
        scale = float(row.get("scale"))
        q25 = float(row.get("q25"))
        q75 = float(row.get("q75"))
        if not all(math.isfinite(value) for value in (center, scale, q25, q75)):
            raise ValueError(f"V2 lab scale contains a non-finite statistic for {field}")
        if scale <= 0 or q75 <= q25 or not math.isclose(scale, q75 - q25, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"V2 lab scale IQR contract is invalid for {field}")
        if int(row.get("coordinate_count", 0)) < 1 or int(row.get("unique_window_count", 0)) < 1:
            raise ValueError(f"V2 lab scale has no fit support for {field}")
        compact_fields[field] = {"unit": unit, "center": center, "scale": scale}
    return {
        "schema": "multires_event_v2_lab_affine_scale_v1",
        "version": "2026-07-13-train-target-windows-v1",
        "coordinate_contract": "lab_shared_affine_canonical_v1",
        "content_sha256": observed_hash,
        "fields": compact_fields,
    }


def _initialize_v2_distributed(
    train: Mapping[str, Any],
) -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    training = train["training"]
    required_world_size = int(training["required_world_size"])
    required_devices = int(training["required_cuda_devices"])
    if world_size != required_world_size:
        raise RuntimeError(
            f"launch V2 with torchrun --nproc_per_node={required_world_size}; "
            f"observed WORLD_SIZE={world_size}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < required_devices:
        raise RuntimeError(
            f"V2 matched run requires {required_devices} visible CUDA devices; "
            f"found {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=V2_DISTRIBUTED_TIMEOUT_SECONDS),
            device_id=device,
        )
    return rank, world_size, local_rank, device


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_git_object_id(value: Any) -> bool:
    text = str(value or "")
    return len(text) in {40, 64} and all(
        character in "0123456789abcdef" for character in text
    )


def _copy_verified_file_atomic(source: Path, destination: Path) -> str:
    """Copy an immutable run input without ever replacing a conflicting prior copy."""

    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"V2 run artifact source is absent: {source}")
    source_sha256 = sha256_file(source)
    if destination.exists():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or sha256_file(destination) != source_sha256
        ):
            raise RuntimeError(
                f"V2 run artifact conflicts with an existing frozen copy: {destination}"
            )
        return source_sha256
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with source.open("rb") as source_handle, temporary.open("xb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        if sha256_file(temporary) != source_sha256:
            raise IOError(f"V2 atomic artifact copy failed hash verification: {source}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    if sha256_file(destination) != source_sha256:
        raise IOError(f"V2 persisted artifact failed hash verification: {destination}")
    return source_sha256


def _runtime_environment_artifact(
    repo_root: Path,
    train: Mapping[str, Any],
    *,
    world_size: int,
) -> dict[str, Any]:
    """Capture diagnostics while hashing only path/time-independent runtime facts."""

    required_devices = int(train["training"]["required_cuda_devices"])
    if world_size != 2 or required_devices != 2:
        raise ValueError("formal V2 runtime identity requires world_size=2 and two devices")
    if not torch.cuda.is_available() or torch.cuda.device_count() < required_devices:
        raise RuntimeError("formal V2 runtime identity requires two visible CUDA devices")
    devices: list[dict[str, Any]] = []
    diagnostic_devices: list[dict[str, Any]] = []
    for index in range(required_devices):
        properties = torch.cuda.get_device_properties(index)
        semantic_device = {
            "name": str(properties.name),
            "compute_capability": [
                int(properties.major),
                int(properties.minor),
            ],
        }
        devices.append(semantic_device)
        diagnostic_devices.append(
            {
                "index": index,
                **semantic_device,
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    requirements_path = repo_root / "requirements-multires-kaggle.txt"
    lock_path = repo_root / "uv.lock"
    if not requirements_path.is_file():
        raise FileNotFoundError("formal V2 runtime lacks requirements-multires-kaggle.txt")
    dependency_versions: dict[str, str] = {}
    for package in ("numpy", "PyYAML", "safetensors"):
        try:
            dependency_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"formal V2 runtime lacks required dependency metadata: {package}"
            ) from exc
    semantic = {
        "schema_version": "trauma_predict.multires_event_v2_semantic_runtime.v1",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda or ""),
        "cudnn": int(torch.backends.cudnn.version() or 0),
        "devices": devices,
        "world_size": world_size,
        "precision": str(train["training"]["precision"]),
        "requirements_sha256": sha256_file(requirements_path),
        "lock_sha256": sha256_file(lock_path) if lock_path.is_file() else None,
        "dependency_versions": dependency_versions,
    }
    if not semantic["cuda_runtime"] or int(semantic["cudnn"]) <= 0:
        raise RuntimeError("formal V2 runtime lacks CUDA/cuDNN identity")
    package_versions: dict[str, str | None] = {}
    for package in ("numpy", "PyYAML", "safetensors", "torch"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    return {
        "schema_version": "trauma_predict.multires_event_v2_runtime_environment.v1",
        "captured_at": utc_now(),
        "semantic_runtime_identity": semantic,
        "semantic_runtime_identity_sha256": sha256_payload(semantic),
        "diagnostics": {
            "sys_version": sys.version,
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "devices": diagnostic_devices,
            "packages": package_versions,
        },
    }


def _materialize_run_artifacts(
    output_dir: Path,
    *,
    runtime: MultiresEventV2Runtime,
    train: Mapping[str, Any],
    repo_root: Path,
    train_path: Path,
    dataset_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    """Freeze all external model/evaluation inputs inside the portable run."""

    sources = {
        "input_normalization": Path(str(runtime.identity["normalization_artifact"])),
        "lab_affine_scale": resolve_repo_path(str(train["lab_scale_artifact"]), repo_root),
        "standardized_primitive_scale": resolve_repo_path(
            str(train["standardized_primitive_scale_artifact"]), repo_root
        ),
        "promotion_metric_contract": resolve_repo_path(
            str(train["promotion_metric_contract"]), repo_root
        ),
        "train_config": train_path,
        "dataset_config": dataset_path,
        "model_config": model_path,
    }
    entries: dict[str, Any] = {}
    for name, source in sources.items():
        relative = RUN_ARTIFACT_PATHS[name]
        file_sha256 = _copy_verified_file_atomic(source, output_dir / relative)
        entries[name] = {"path": relative, "file_sha256": file_sha256}
    runtime_environment = _runtime_environment_artifact(
        repo_root,
        train,
        world_size=int(train["training"]["required_world_size"]),
    )
    runtime_environment_path = output_dir / RUN_ARTIFACT_PATHS["runtime_environment"]
    if runtime_environment_path.is_file():
        existing_runtime = json.loads(runtime_environment_path.read_text(encoding="utf-8"))
        if existing_runtime.get("semantic_runtime_identity_sha256") != runtime_environment.get(
            "semantic_runtime_identity_sha256"
        ):
            raise RuntimeError(
                "V2 resume runtime environment differs from the original formal run"
            )
    else:
        atomic_write_json(runtime_environment_path, runtime_environment)
    entries["runtime_environment"] = {
        "path": RUN_ARTIFACT_PATHS["runtime_environment"],
        "file_sha256": sha256_file(runtime_environment_path),
        "semantic_sha256": runtime_environment[
            "semantic_runtime_identity_sha256"
        ],
    }
    if entries["input_normalization"]["file_sha256"] != str(
        runtime.identity["normalization_artifact_sha256"]
    ):
        raise ValueError("V2 copied normalization differs from the runtime identity")
    entries["lab_affine_scale"]["semantic_sha256"] = str(
        runtime.identity["lab_scale_artifact_sha256"]
    )
    entries["standardized_primitive_scale"]["semantic_sha256"] = str(
        runtime.identity["standardized_primitive_scale_sha256"]
    )
    entries["promotion_metric_contract"]["semantic_sha256"] = str(
        runtime.identity["promotion_metric_contract_sha256"]
    )
    payload = {
        "schema_version": "trauma_predict.multires_event_v2_run_artifacts.v1",
        "artifacts": entries,
    }
    manifest_path = output_dir / "artifacts/manifest.json"
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("V2 existing run artifact manifest is unreadable") from exc
        if existing != payload:
            raise RuntimeError("V2 existing run artifact manifest conflicts with source inputs")
    atomic_write_json(manifest_path, payload)
    return payload


def _bind_runtime_to_run_artifacts(
    output_dir: Path,
    runtime: MultiresEventV2Runtime,
) -> MultiresEventV2Runtime:
    """Replace build-machine artifact paths with verified run-relative pointers."""

    manifest_path = output_dir / "artifacts/manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("V2 run artifact manifest is absent")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "trauma_predict.multires_event_v2_run_artifacts.v1":
        raise ValueError("V2 run artifact manifest schema is invalid")
    entries = manifest.get("artifacts")
    if not isinstance(entries, Mapping) or set(entries) != set(RUN_ARTIFACT_PATHS):
        raise ValueError("V2 run artifact manifest has an incomplete artifact set")
    for name, expected_path in RUN_ARTIFACT_PATHS.items():
        entry = entries.get(name)
        if not isinstance(entry, Mapping) or entry.get("path") != expected_path:
            raise ValueError(f"V2 run artifact pointer is invalid for {name}")
        expected_sha = str(entry.get("file_sha256") or "")
        if not _is_sha256(expected_sha):
            raise ValueError(f"V2 run artifact hash is invalid for {name}")
        artifact_path = output_dir / expected_path
        if artifact_path.is_symlink() or sha256_file(artifact_path) != expected_sha:
            raise ValueError(f"V2 run artifact hash mismatch for {name}")
    identity = dict(runtime.identity)
    normalization_sha = str(identity["normalization_artifact_sha256"])
    if entries["input_normalization"]["file_sha256"] != normalization_sha:
        raise ValueError("V2 portable normalization does not match the runtime")
    semantic_checks = {
        "lab_affine_scale": str(identity["lab_scale_artifact_sha256"]),
        "standardized_primitive_scale": str(
            identity["standardized_primitive_scale_sha256"]
        ),
        "promotion_metric_contract": str(
            identity["promotion_metric_contract_sha256"]
        ),
    }
    for name, expected in semantic_checks.items():
        if entries[name].get("semantic_sha256") != expected:
            raise ValueError(f"V2 portable artifact semantic identity mismatch for {name}")
    runtime_environment_path = output_dir / RUN_ARTIFACT_PATHS["runtime_environment"]
    runtime_environment = json.loads(runtime_environment_path.read_text(encoding="utf-8"))
    semantic_runtime_identity = runtime_environment.get("semantic_runtime_identity")
    semantic_runtime_sha = str(
        runtime_environment.get("semantic_runtime_identity_sha256") or ""
    )
    if (
        not isinstance(semantic_runtime_identity, Mapping)
        or sha256_payload(semantic_runtime_identity) != semantic_runtime_sha
        or entries["runtime_environment"].get("semantic_sha256")
        != semantic_runtime_sha
    ):
        raise ValueError("V2 portable runtime environment semantic identity is invalid")
    identity.update(
        {
            "normalization_artifact": RUN_ARTIFACT_PATHS["input_normalization"],
            "normalization_artifact_file_sha256": entries["input_normalization"][
                "file_sha256"
            ],
            "lab_scale_artifact": RUN_ARTIFACT_PATHS["lab_affine_scale"],
            "lab_scale_artifact_file_sha256": entries["lab_affine_scale"][
                "file_sha256"
            ],
            "standardized_primitive_scale_artifact": RUN_ARTIFACT_PATHS[
                "standardized_primitive_scale"
            ],
            "standardized_primitive_scale_artifact_file_sha256": entries[
                "standardized_primitive_scale"
            ]["file_sha256"],
            "promotion_metric_contract": RUN_ARTIFACT_PATHS[
                "promotion_metric_contract"
            ],
            "promotion_metric_contract_file_sha256": entries[
                "promotion_metric_contract"
            ]["file_sha256"],
            "runtime_environment_artifact": RUN_ARTIFACT_PATHS[
                "runtime_environment"
            ],
            "runtime_environment_artifact_file_sha256": entries[
                "runtime_environment"
            ]["file_sha256"],
            "semantic_runtime_identity_sha256": semantic_runtime_sha,
        }
    )
    return replace(runtime, identity=identity)


def _completed_training_result(
    training: Mapping[str, Any],
    selected_model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep optimizer completion separate from the checkpoint selected for inference."""

    completed_step = training.get("global_step")
    best_step = training.get("best_step")
    selected_step = selected_model_identity.get("selected_checkpoint_step")
    if (
        isinstance(completed_step, bool)
        or not isinstance(completed_step, int)
        or completed_step < 1
        or isinstance(best_step, bool)
        or not isinstance(best_step, int)
        or best_step < 1
        or best_step != selected_step
        or best_step > completed_step
    ):
        raise ValueError("V2 completed and selected checkpoint steps are inconsistent")
    result = dict(training)
    result.update(
        {
            "training_completed_step": completed_step,
            "selected_checkpoint_step": best_step,
            "selected_checkpoint_model_sha256": selected_model_identity[
                "selected_checkpoint_model_sha256"
            ],
        }
    )
    return result


def _save_v2_best_model(
    *,
    output_dir: Path,
    model: Any,
    identity_hashes: Mapping[str, str],
    step: int,
    metric: float,
) -> None:
    """Write best-model files on rank zero without entering a collective.

    Every rank must call :func:`_materialize_v2_best_model`; this filesystem
    writer is deliberately collective-free so the enclosing distributed phase
    cannot diverge into barrier versus all-gather ordering.
    """

    if not is_rank_zero():
        raise RuntimeError("V2 best-checkpoint writer is rank-zero only")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("V2 best checkpoint step must be a positive integer")
    if not math.isfinite(float(metric)):
        raise ValueError("V2 best checkpoint metric must be finite")
    if not identity_hashes or any(
        not isinstance(key, str) or not isinstance(value, str) or not value
        for key, value in identity_hashes.items()
    ):
        raise ValueError("V2 best checkpoint requires a complete run identity")
    best_dir = output_dir / "best_checkpoint"
    best_dir.mkdir(parents=True, exist_ok=True)
    temporary = best_dir / f".model.pt.tmp-{os.getpid()}"
    torch.save(_unwrapped_model(model).state_dict(), temporary)
    temporary.replace(best_dir / "model.pt")
    atomic_write_json(best_dir / "identity_hashes.json", dict(identity_hashes))
    atomic_write_json(output_dir / "best_checkpoint.json", {
        "schema_version": BEST_CHECKPOINT_SCHEMA,
        "updated_at": utc_now(),
        "step": int(step),
        "joint_nll_subject_macro": float(metric),
        "path": "best_checkpoint",
        "model_sha256": sha256_file(best_dir / "model.pt"),
        "identity_hashes": dict(identity_hashes),
    })


def _materialize_v2_best_model(
    *,
    output_dir: Path,
    model: Any,
    identity_hashes: Mapping[str, str],
    step: int,
    metric: float,
) -> None:
    """Synchronize one rank-zero best-checkpoint write across every rank."""

    _collect_distributed_phase(
        "formal best-checkpoint materialization",
        lambda: _save_v2_best_model(
            output_dir=output_dir,
            model=model,
            identity_hashes=identity_hashes,
            step=step,
            metric=metric,
        )
        if is_rank_zero()
        else None,
    )


def _run_v2_best_checkpoint_collective_canary(
    *,
    output_root: Path,
    mode: str,
    world_size: int,
) -> dict[str, Any]:
    """Exercise the production best-checkpoint save/load order before data loading."""

    checkpoint_canary_identity = {
        "canary": sha256_payload(
            {
                "schema": "multires_event_v2_best_checkpoint_collective_canary_v1",
                "mode": mode,
                "world_size": world_size,
            }
        )
    }
    checkpoint_canary_model = torch.nn.Identity()
    _materialize_v2_best_model(
        output_dir=output_root,
        model=checkpoint_canary_model,
        identity_hashes=checkpoint_canary_identity,
        step=1,
        metric=0.0,
    )
    selected = _load_v2_best_model(
        output_root,
        checkpoint_canary_model,
        torch.device("cpu"),
        expected_identity_hashes=checkpoint_canary_identity,
        expected_best_step=1,
    )
    if is_rank_zero():
        print(
            "MULTIRES_EVENT_V2_BEST_CHECKPOINT_COLLECTIVE_CANARY_OK "
            f"phase=predata world_size={world_size} "
            f"model_sha256={selected['selected_checkpoint_model_sha256']}",
            flush=True,
        )
    return selected


def _load_v2_best_model(
    output_dir: Path,
    model: Any,
    device: torch.device,
    *,
    expected_identity_hashes: Mapping[str, str],
    expected_best_step: Any,
) -> dict[str, Any]:
    """Load only a checkpoint fully bound by its pointer, model bytes, and run identity."""

    _barrier()
    pointer_path = output_dir / "best_checkpoint.json"
    if pointer_path.is_symlink() or not pointer_path.is_file():
        raise FileNotFoundError("V2 training completed without a best-checkpoint pointer")
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("V2 best-checkpoint pointer is unreadable") from exc
    required_keys = {
        "schema_version",
        "updated_at",
        "step",
        "joint_nll_subject_macro",
        "path",
        "model_sha256",
        "identity_hashes",
    }
    if not isinstance(pointer, Mapping) or set(pointer) != required_keys:
        raise ValueError("V2 best-checkpoint pointer schema fields are invalid")
    if pointer.get("schema_version") != BEST_CHECKPOINT_SCHEMA:
        raise ValueError("V2 best-checkpoint pointer schema version is invalid")
    updated_at = pointer.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at.endswith("Z"):
        raise ValueError("V2 best-checkpoint pointer timestamp is invalid")
    if pointer.get("path") != "best_checkpoint":
        raise ValueError("V2 best-checkpoint path must be run-relative best_checkpoint")
    step = pointer.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("V2 best-checkpoint pointer step is invalid")
    if (
        isinstance(expected_best_step, bool)
        or not isinstance(expected_best_step, int)
        or step != expected_best_step
    ):
        raise ValueError("V2 best-checkpoint pointer differs from trainer best_step")
    try:
        metric = float(pointer.get("joint_nll_subject_macro"))
    except (TypeError, ValueError) as exc:
        raise ValueError("V2 best-checkpoint metric is invalid") from exc
    if not math.isfinite(metric):
        raise ValueError("V2 best-checkpoint metric is non-finite")
    if pointer.get("identity_hashes") != dict(expected_identity_hashes):
        raise RuntimeError("V2 best-checkpoint pointer has a different run identity")
    best_dir = output_dir / "best_checkpoint"
    persisted_identity_path = best_dir / "identity_hashes.json"
    if persisted_identity_path.is_symlink() or not persisted_identity_path.is_file():
        raise FileNotFoundError("V2 best checkpoint lacks its identity hash file")
    try:
        persisted_identity = json.loads(persisted_identity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("V2 best-checkpoint identity file is unreadable") from exc
    if (
        persisted_identity != dict(expected_identity_hashes)
        or persisted_identity != pointer["identity_hashes"]
    ):
        raise RuntimeError("V2 best-checkpoint identity files disagree")
    model_sha256 = str(pointer.get("model_sha256") or "")
    if not _is_sha256(model_sha256):
        raise ValueError("V2 best-checkpoint model hash is invalid")
    path = best_dir / "model.pt"
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("V2 training completed without a subject-macro checkpoint")
    if sha256_file(path) != model_sha256:
        raise ValueError("V2 best-checkpoint model bytes fail SHA-256 verification")
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - older supported torch
        state = torch.load(path, map_location=device)
    if not isinstance(state, Mapping):
        raise ValueError("V2 best-checkpoint payload is not a state dictionary")
    _unwrapped_model(model).load_state_dict(state, strict=True)
    _barrier()
    return {
        "schema_version": SELECTED_MODEL_SCHEMA,
        "selected_checkpoint_step": step,
        "selected_checkpoint_model_sha256": model_sha256,
        "selected_checkpoint_path": "best_checkpoint/model.pt",
        "best_checkpoint_manifest_path": "best_checkpoint.json",
        "best_checkpoint_manifest_sha256": sha256_file(pointer_path),
    }


def _write_identity(
    output_dir: Path,
    *,
    train: Mapping[str, Any],
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
    runtime: MultiresEventV2Runtime,
    identity_hashes: Mapping[str, str],
    train_path: Path,
    dataset_path: Path,
    model_path: Path,
    parameter_count: int,
    source_identity: Mapping[str, Any],
) -> None:
    existing = output_dir / "identity_hashes.json"
    if existing.is_file() and json.loads(existing.read_text(encoding="utf-8")) != dict(identity_hashes):
        raise RuntimeError("V2 output directory contains a different frozen run identity")
    atomic_write_json(output_dir / "resolved_config.json", {
        "schema_version": "trauma_predict.multires_event_v2_resolved_config.v1",
        "train_config_path": RUN_ARTIFACT_PATHS["train_config"],
        "dataset_config_path": RUN_ARTIFACT_PATHS["dataset_config"],
        "model_config_path": RUN_ARTIFACT_PATHS["model_config"],
        "train": dict(train),
        "dataset": dict(dataset),
        "model": dict(model),
    })
    atomic_write_json(output_dir / "dataset_identity.json", dict(runtime.identity))
    atomic_write_json(output_dir / "objective_contract.json", {
        "objective": dict(train["objective"]),
        "process_contract": runtime.contract.process_registry,
        "contract_bundle_hash": runtime.contract.contract_bundle_hash,
        "contract_hashes": dict(runtime.contract.contract_hashes),
    })
    atomic_write_json(output_dir / "model_identity.json", {
        "mode": train["mode"],
        "parameter_count": int(parameter_count),
        "matched_design_signature": identity_hashes["matched_design"],
        "initialization": "from_scratch",
        "text_backbone": None,
        "tokenizer": None,
    })
    atomic_write_json(
        output_dir / "normalization_identity.json",
        {
            **_normalization_identity(runtime.normalization),
            "artifact_path": runtime.identity["normalization_artifact"],
            "artifact_sha256": runtime.identity["normalization_artifact_sha256"],
            "artifact_file_sha256": runtime.identity[
                "normalization_artifact_file_sha256"
            ],
        },
    )
    atomic_write_json(output_dir / "source_identity.json", dict(source_identity))
    atomic_write_json(output_dir / "identity_hashes.json", dict(identity_hashes))


def _export_run(
    output_dir: Path,
    model: Any,
    train: Mapping[str, Any],
    identity_hashes: Mapping[str, str],
    training: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    free_running_evaluation: Mapping[str, Any] | None,
    selected_model_identity: Mapping[str, Any],
) -> None:
    training_config = _mapping(train.get("training"), "train.training")
    configured_max_steps = int(training_config["max_steps"])
    if (
        int(training.get("global_step", -1)) != configured_max_steps
        or int(training.get("max_steps", -1)) != configured_max_steps
        or int(training.get("training_completed_step", -1)) != configured_max_steps
        or int(training.get("scaler_skipped_steps", -1)) != 0
    ):
        raise ValueError("V2 export requires complete skip-free training at configured max_steps")
    selected_step = selected_model_identity.get("selected_checkpoint_step")
    selected_sha = str(
        selected_model_identity.get("selected_checkpoint_model_sha256") or ""
    )
    if (
        selected_model_identity.get("schema_version") != SELECTED_MODEL_SCHEMA
        or isinstance(selected_step, bool)
        or not isinstance(selected_step, int)
        or selected_step < 1
        or not _is_sha256(selected_sha)
    ):
        raise ValueError("V2 export received an invalid selected-model identity")
    if (
        training.get("selected_checkpoint_step") != selected_step
        or training.get("selected_checkpoint_model_sha256") != selected_sha
        or evaluation.get("step") != selected_step
    ):
        raise ValueError("V2 final evaluation is not bound to the selected checkpoint")
    required_row_identity = (
        "source_tree_sha256",
        "source_identity_sha256",
        "git_commit",
        "git_head_tree",
        "matched_design_signature",
        "selected_checkpoint_step",
        "selected_checkpoint_model_sha256",
    )
    teacher_identity = evaluation.get("identity")
    if not isinstance(teacher_identity, Mapping) or any(
        teacher_identity.get(key) in (None, "") for key in required_row_identity
    ):
        raise ValueError("V2 final teacher rows lack the selected source/model identity")
    if (
        teacher_identity.get("selected_checkpoint_step") != selected_step
        or teacher_identity.get("selected_checkpoint_model_sha256") != selected_sha
    ):
        raise ValueError("V2 final teacher row identity differs from the selected checkpoint")
    portable_evaluation = _portable_local_pointer_payload(output_dir, evaluation)
    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model_path = final_dir / "model.pt"
    selected_model_path = _resolve_run_pointer(
        output_dir,
        selected_model_identity.get("selected_checkpoint_path"),
        relative_base=output_dir,
        label="selected checkpoint model",
    )
    copied_model_sha = _copy_verified_file_atomic(selected_model_path, model_path)
    if copied_model_sha != selected_sha:
        raise ValueError("V2 final model bytes differ from the selected best checkpoint")
    model_manifest_path = final_dir / "model_manifest.json"
    atomic_write_json(model_manifest_path, {
        "schema_version": "trauma_predict.multires_event_v2_model_manifest.v1",
        "created_at": utc_now(),
        "mode": train["mode"],
        "model_file": "final_model/model.pt",
        "model_sha256": copied_model_sha,
        "selected_checkpoint_step": selected_step,
        "selected_checkpoint_model_sha256": selected_sha,
        "training_completed_step": training["training_completed_step"],
        "joint_nll_subject_macro": evaluation["joint_nll_subject_macro"],
        "identity_hashes": dict(identity_hashes),
    })
    atomic_write_json(output_dir / "evaluation.json", portable_evaluation)
    free_running_pointer: dict[str, Any] | None = None
    if free_running_evaluation is not None:
        free_root = output_dir / "free_running"
        free_path = free_root / "evaluation.json"
        if not free_path.is_file():
            raise FileNotFoundError("V2 free-running evaluation summary was not persisted")
        if int(free_running_evaluation.get("anchors", -1)) != EXPECTED_COUNTS["val"]:
            raise ValueError("V2 free-running evaluation did not cover all validation anchors")
        if free_running_evaluation.get("step") != selected_step:
            raise ValueError("V2 free-running evaluation did not use the selected checkpoint")
        if free_running_evaluation.get("identity") != teacher_identity:
            raise ValueError("V2 teacher and free-running row identities differ")
        manifest_path = _resolve_run_pointer(
            output_dir,
            free_running_evaluation.get("manifest_path"),
            relative_base=free_root,
            label="free-running manifest",
        )
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        portable_manifest = _portable_local_pointer_payload(free_root, manifest_payload)
        atomic_write_json(manifest_path, portable_manifest)
        portable_free = _portable_local_pointer_payload(
            free_root, free_running_evaluation
        )
        portable_free["manifest_sha256"] = sha256_file(manifest_path)
        atomic_write_json(free_path, portable_free)
        free_running_pointer = {
            "path": _run_relative_path(output_dir, free_path),
            "sha256": sha256_file(free_path),
            "manifest_path": _run_relative_path(output_dir, manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "trajectories_per_anchor": int(
                free_running_evaluation["trajectories_per_anchor"]
            ),
            "coherence_rate": float(
                _mapping(
                    free_running_evaluation["coherence"],
                    "free_running_evaluation.coherence",
                )["rate"]
            ),
        }
    artifact_manifest_path = output_dir / "artifacts/manifest.json"
    if artifact_manifest_path.is_symlink() or not artifact_manifest_path.is_file():
        raise FileNotFoundError("V2 portable artifact manifest is absent at export")
    metrics_path = output_dir / "metrics.jsonl"
    optimizer_health_summary_path = output_dir / "optimizer_health_summary.json"
    optimizer_health_summary = summarize_optimizer_health_metrics(
        metrics_path, training=training_config
    )
    atomic_write_json(optimizer_health_summary_path, optimizer_health_summary)
    optimizer_health_pointer = {
        "path": "optimizer_health_summary.json",
        "sha256": sha256_file(optimizer_health_summary_path),
        "metrics_path": "metrics.jsonl",
        "metrics_sha256": sha256_file(metrics_path),
    }
    run_manifest_path = output_dir / "run_manifest.json"
    atomic_write_json(run_manifest_path, {
        "schema_version": "trauma_predict.multires_event_v2_run_manifest.v1",
        "completed_at": utc_now(),
        "status": "SUCCEEDED",
        "route": ROUTE,
        "run_name": train["run_name"],
        "mode": train["mode"],
        "training": dict(training),
        "evaluation": portable_evaluation,
        "free_running_evaluation": free_running_pointer,
        "selected_model_identity": dict(selected_model_identity),
        "final_model": {
            "path": _run_relative_path(output_dir, model_path),
            "sha256": copied_model_sha,
            "manifest_path": _run_relative_path(output_dir, model_manifest_path),
            "manifest_sha256": sha256_file(model_manifest_path),
        },
        "artifact_manifest": {
            "path": _run_relative_path(output_dir, artifact_manifest_path),
            "sha256": sha256_file(artifact_manifest_path),
        },
        "optimizer_health_summary": optimizer_health_pointer,
        "identity_hashes": dict(identity_hashes),
    })
    atomic_write_json(output_dir / "SUCCESS", {
        "schema_version": "trauma_predict.multires_event_v2_success.v1",
        "completed_at": utc_now(),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "optimizer_health_summary_sha256": optimizer_health_pointer["sha256"],
        "metrics_jsonl_sha256": optimizer_health_pointer["metrics_sha256"],
    })


def _run_relative_path(output_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"V2 export pointer escapes the run directory: {path}") from exc


def _resolve_run_pointer(
    output_dir: Path,
    value: Any,
    *,
    relative_base: Path,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"V2 {label} pointer is absent")
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (relative_base / raw).resolve()
    _run_relative_path(output_dir, path)
    if not path.is_file():
        raise FileNotFoundError(f"V2 {label} pointer is not a file: {path}")
    return path


def _portable_local_pointer_payload(
    local_root: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Make file pointers local-root relative while rejecting path traversal."""

    def convert(value: Any, key: str | None = None) -> Any:
        if isinstance(value, Mapping):
            return {str(name): convert(item, str(name)) for name, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if (
            isinstance(value, str)
            and key is not None
            and (key == "path" or key.endswith("_path"))
        ):
            raw = Path(value)
            path = raw.resolve() if raw.is_absolute() else (local_root / raw).resolve()
            try:
                relative = path.relative_to(local_root.resolve()).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"V2 exported pointer escapes its portable artifact root: {value}"
                ) from exc
            if not path.is_file():
                raise FileNotFoundError(f"V2 exported pointer is not a file: {path}")
            return relative
        return value

    return convert(payload)


def _distributed_sum_count(total: float, count: int, device: torch.device) -> tuple[float, float]:
    values = torch.tensor([total, float(count)], dtype=torch.float64, device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    return float(values[0].item()), float(values[1].item())


def _autocast_factory(device: torch.device, precision: str) -> Any:
    if device.type != "cuda" or precision != "fp16":
        return nullcontext

    def factory() -> Any:
        try:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        except AttributeError:  # pragma: no cover
            return torch.cuda.amp.autocast(dtype=torch.float16)

    return factory


def _normalization_identity(normalization: Any) -> dict[str, Any]:
    return {
        "dataset_fingerprint": str(normalization.dataset_fingerprint),
        "supervision_sha256": str(normalization.supervision_sha256),
        "fit_split": str(normalization.fit_split),
        "subject_count": int(normalization.subject_count),
        "subject_ids_sha256": str(normalization.subject_ids_sha256),
        "clip_value": float(normalization.clip_value),
        "epsilon": float(normalization.epsilon),
    }


def _source_tree_identity(repo_root: Path) -> dict[str, Any]:
    """Hash the executable source bytes independently of Git cleanliness."""

    candidates = list((repo_root / "src/trauma_predict").rglob("*.py"))
    candidates.extend(
        repo_root / relative
        for relative in (
            "notebooks/kaggle/train_relational_primary.py",
            "requirements-multires-kaggle.txt",
            "pyproject.toml",
        )
    )
    files: dict[str, str] = {}
    for path in sorted(set(candidates)):
        if not path.is_file():
            raise FileNotFoundError(f"V2 source identity file is absent: {path}")
        relative = path.relative_to(repo_root).as_posix()
        files[relative] = sha256_file(path)
    source_tree_sha256 = sha256_payload(
        {
            "schema_version": "trauma_predict.multires_event_v2_source_tree.v1",
            "files": files,
        }
    )

    def git_text(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = git_text("status", "--porcelain=v1", "--untracked-files=all")
    release_identity: Mapping[str, Any] | None = None
    release_path = repo_root / "SOURCE_RELEASE.json"
    if release_path.is_file() and not release_path.is_symlink():
        candidate = json.loads(release_path.read_text(encoding="utf-8"))
        if not isinstance(candidate, Mapping):
            raise ValueError("SOURCE_RELEASE.json must contain one object")
        if (
            candidate.get("schema_version")
            != "trauma_predict.multires_event_v2_source_release.v1"
            or candidate.get("source_tree_sha256") != source_tree_sha256
            or not _is_git_object_id(candidate.get("git_commit"))
            or not _is_git_object_id(candidate.get("git_head_tree"))
        ):
            raise ValueError("SOURCE_RELEASE.json does not bind the executable source tree")
        release_identity = candidate
    return {
        "schema_version": "trauma_predict.multires_event_v2_source_identity.v1",
        "git_commit": (
            git_text("rev-parse", "HEAD")
            or (str(release_identity["git_commit"]) if release_identity else None)
        ),
        "git_head_tree": (
            git_text("rev-parse", "HEAD^{tree}")
            or (str(release_identity["git_head_tree"]) if release_identity else None)
        ),
        "git_clean": (
            status == ""
            if status is not None
            else (True if release_identity is not None else None)
        ),
        "git_status_sha256": sha256_payload(status) if status is not None else None,
        "release_identity_sha256": (
            sha256_file(release_path) if release_identity is not None else None
        ),
        "source_tree_sha256": source_tree_sha256,
        "source_file_count": len(files),
        "source_files": files,
    }


def _assert_runtime_identity_stable(
    training_identity: Mapping[str, Any],
    final_identity: Mapping[str, Any],
) -> None:
    training = dict(training_identity)
    final = dict(final_identity)
    training.pop("phase", None)
    final.pop("phase", None)
    if training != final:
        keys = sorted(set(training) | set(final))
        mismatch = {
            key: {"training": training.get(key), "final": final.get(key)}
            for key in keys
            if training.get(key) != final.get(key)
        }
        raise RuntimeError(
            "V2 final runtime differs from the optimizer runtime; refusing evaluation: "
            f"{mismatch}"
        )


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result < 1:
        raise ValueError("configured sampler cap must be positive")
    return result


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


__all__ = [
    "CAPACITY_PROBE_OPTIMIZER_STEPS",
    "CAPACITY_PROBE_SCHEMA",
    "CAPACITY_PROBE_TRAJECTORIES_PER_ANCHOR",
    "CAPACITY_PROBE_VALIDATION_ANCHORS",
    "CAPACITY_SEMANTIC_CANARY_ANCHORS",
    "CAPACITY_SEMANTIC_CANARY_TRAJECTORIES_PER_ANCHOR",
    "CAPACITY_RUNTIME_POLICY",
    "CAPACITY_STRUCTURAL_METRICS",
    "AUTHORIZED_TRAINING_RUN_NAMES",
    "AUTHORIZED_VERIFICATION_RUN_NAMES",
    "EXPECTED_OPTIMIZER_CONTRACT",
    "MATCHED_MODES",
    "EXPECTED_FORMAL_MODEL_PARAMETER_COUNT",
    "MultiresEventV2Runtime",
    "OPTIMIZER_CONTRACT_VERSION",
    "OPTIMIZER_HEALTH_SUMMARY_SCHEMA",
    "RAW_JOINT_NLL_REDUCTION",
    "ROUTE",
    "TRAINING_AUTHORIZED",
    "TRAINING_AUTHORIZATION_REASON",
    "VERIFICATION_AUTHORIZED",
    "VERIFICATION_AUTHORIZATION_REASON",
    "V2_DISTRIBUTED_TIMEOUT_SECONDS",
    "V2_EARLY_CANARY_PROCESS_GROUP_TIMEOUT_SECONDS",
    "V2_NCCL_MONITOR_HEARTBEAT_TIMEOUT_SECONDS",
    "build_multires_event_v2_model",
    "build_multires_event_v2_optimizer",
    "build_multires_event_v2_runtime",
    "load_multires_event_v2_configs",
    "load_lab_scale_artifact",
    "matched_design_signature",
    "project_multires_event_v2_capacity_runtime",
    "raw_414_factor_joint_nll_batch_mean",
    "require_multires_event_v2_training_authorization",
    "run_multires_event_v2_capacity_gated_training",
    "run_multires_event_v2_capacity_probe",
    "run_multires_event_v2_rank_artifact_preflight_only",
    "run_multires_event_v2_training",
    "run_multires_event_v2_verification_probe",
    "validate_formal_model_parameter_count",
    "validate_formal_target_field_order",
    "validate_multires_event_v2_configs",
    "validate_optimizer_health_summary",
    "require_multires_event_v2_verification_authorization",
    "summarize_optimizer_health_metrics",
]
