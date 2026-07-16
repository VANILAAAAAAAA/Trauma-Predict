from __future__ import annotations

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
from typing import Any, Mapping, Sequence

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
    MultiresEventV2RelationContract,
)
from trauma_predict.data.multires_event_v2.relation_contract import (
    EXPECTED_RELATION_FILE_HASHES,
    RELATION_CONTRACT_VERSION,
)
from trauma_predict.eval.multires_event_v2 import (
    evaluate_teacher_forced,
    exact_teacher_forced_loss,
    move_to_device,
)
from trauma_predict.eval.multires_event_v2_free_running import (
    _collect_distributed_phase,
    evaluate_free_running_v2,
    probe_free_running_v2_capacity,
    verify_rank_local_artifact_preflight,
)
from trauma_predict.eval.multires_event_v2_metric_contract import (
    load_trajectory_metric_contract,
)
from trauma_predict.eval.multires_event_v2_projections import (
    load_standardized_primitive_scale_artifact,
)
from trauma_predict.modeling.multires_event_v2.config import MultiResolutionEventV2Config
from trauma_predict.modeling.multires_event_v2.model import MultiResolutionEventV2Model
from trauma_predict.training.config import (
    expand_env,
    load_yaml_config,
    load_yaml_config_unexpanded,
)
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
    _torch_load,
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


ROUTE = "multires_event_v2_m4_relation_v2"
TRAINING_AUTHORIZED = True
AUTHORIZED_TRAINING_RUN_NAMES: tuple[str, ...] = (
    "p100_multires_event_v2_relation_v2",
)
TRAINING_AUTHORIZATION_REASON = (
    "the active six-M4 route is frozen to relation contract V2 and starts from scratch"
)
EXPECTED_BASE_DATASET_ID = "multires_event_v1_c4_full_20260712"
EXPECTED_BASE_FINGERPRINT = "d58d003b6a9b2dd7c1f8d269a1867b534ea475a91118d7d4d44804bee69f9e47"
EXPECTED_BASE_SOURCE_FINGERPRINT = "ed578cf6b6e82c96f3aef71d58d6c176c794c9e8fbd37a468a709d64e94739b9"
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
EXPECTED_EMBEDDED_HISTORICAL_RELATION_CONTRACT_SHA256 = (
    "65286cd9fb7e1038270de39ea17daafffb160cf9c5ab7bb3beb2556a9aa8eea0"
)
EXPECTED_SIDECAR_SCHEMA_SHA256 = "58d3673b6e232344709555a7bff2186047b08ea7932b6685553ee1b526d7e0dd"
EXPECTED_RELATION_CONFIG_DIR = "configs/contracts/multires_event_v2"
EXPECTED_RELATION_BUNDLE_SHA256 = (
    "0331ec0d552e47790d1dc4f8bae3520062c9e6f5fa62cf62e87c187f6783c033"
)
EXPECTED_P100_RUNTIME_CONTRACT_SHA256 = (
    "aada1dee4ee21e02fd5c81ae97d441c38e72d770eec5398932ee295d08f8f2cc"
)
EXPECTED_P100_TORCH_VERSION = "2.10.0+cu126"
EXPECTED_P100_CUDA_RUNTIME = "12.6"
EXPECTED_SUPERVISION_SHA256 = "722cae631ca2b7cf801514cccdfd6cf18be9742f0cb94c9a1e89fb3696d095f6"
EXPECTED_INPUT_NORMALIZATION_SHA256 = (
    "4f54dbeaab4b2becd349d1d8fcaac7b6bdea2567a20874ee7d29338c1f930add"
)
EXPECTED_LAB_SCALE_ARTIFACT = "configs/dataset/multires_event_v2_c4_lab_affine_scale_r9.json"
EXPECTED_LAB_SCALE_ARTIFACT_HASH = "cae827b1f8b1c6a156da4bad340af1b9b0411ca2f5fbe0b9aa8d36ed06cb87bb"
EXPECTED_STANDARDIZED_PRIMITIVE_SCALE_ARTIFACT = (
    "configs/dataset/multires_event_v2_c4_standardized_primitive_scale_r9.json"
)
EXPECTED_STANDARDIZED_PRIMITIVE_SCALE_ARTIFACT_HASH = (
    "f075a9d2d415028845026b06e746cecf382102dfc0ed2c31631000506030665f"
)
EXPECTED_TRAJECTORY_METRIC_CONTRACT = (
    "configs/evaluation/multires_event_v2_relation_v2_metrics.json"
)
EXPECTED_TRAJECTORY_METRIC_CONTRACT_HASH = (
    "2d8f9c2c421e9d69e18470a37b3c6ae7696fff0664c6083df8a7e2de1745123e"
)
EXPECTED_COUNTS = {"samples": 50350, "train": 37734, "val": 6309, "test": 6307, "shards": 52}
EXPECTED_FORMAL_MODEL_PARAMETER_COUNT = 48_728_439
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
V2_DISTRIBUTED_TIMEOUT_SECONDS = 600
V2_NCCL_MONITOR_HEARTBEAT_TIMEOUT_SECONDS = 120
V2_EARLY_CANARY_PROCESS_GROUP_TIMEOUT_SECONDS = 60
EXPECTED_FORMAL_ARCHITECTURE = {
    "hidden_size": 480,
    "num_attention_heads": 8,
    "trajectory_encoder_layers": 6,
    "target_decoder_layers": 6,
    "block_compressor_layers": 1,
    "block_latent_count": 8,
    "future_block_count": 6,
    "target_field_count": 29,
}
BEST_CHECKPOINT_SCHEMA = "trauma_predict.multires_event_v2_best_checkpoint.v1"
V2_CHECKPOINT_SCHEMA = "trauma_predict.multires_event_v2_checkpoint.v2"
SELECTED_MODEL_SCHEMA = "trauma_predict.multires_event_v2_selected_model.v1"
HOSTED_STOP_READINESS_SCHEMA = (
    "trauma_predict.multires_event_v2_hosted_stop_readiness.v1"
)
FINAL_TEACHER_CACHE_SCHEMA = (
    "trauma_predict.multires_event_v2_final_teacher_cache.v1"
)
AUTHORIZED_HOSTED_STOP_STEPS = (2, 250, 1500, 2750, 4000)
RUN_ARTIFACT_PATHS = {
    "input_normalization": "artifacts/input_normalization.json",
    "lab_affine_scale": "artifacts/lab_affine_scale.json",
    "standardized_primitive_scale": "artifacts/standardized_primitive_scale.json",
    "trajectory_metric_contract": "artifacts/trajectory_metric_contract.json",
    "runtime_environment": "artifacts/runtime_environment.json",
    "train_config": "artifacts/config/train.yaml",
    "dataset_config": "artifacts/config/dataset.yaml",
    "model_config": "artifacts/config/model.yaml",
    "relation_field_registry": "artifacts/relation_contract/field_category_matrix_v1.csv",
    "relation_target_target": "artifacts/relation_contract/target_target_relation_edges_v2.csv",
    "relation_input_target": "artifacts/relation_contract/input_target_relation_edges_v2.csv",
    "relation_evidence_registry": "artifacts/relation_contract/relation_evidence_registry_v2.json",
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
    relation_contract: MultiresEventV2RelationContract
    normalization: Any
    identity: Mapping[str, Any]


@dataclass(frozen=True)
class _OptimizerUpdateProbe:
    parameter_name: str
    parameter: torch.nn.Parameter
    flat_index: int
    value_before: torch.Tensor


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
    authored_train = load_yaml_config_unexpanded(train_path)
    dataset_path = resolve_repo_path(authored_train["dataset"]["config_path"], root)
    model_path = resolve_repo_path(authored_train["model"]["config_path"], root)
    authored_dataset = load_yaml_config_unexpanded(dataset_path)
    authored_model = load_yaml_config_unexpanded(model_path)
    validate_multires_event_v2_configs(
        authored_train,
        authored_dataset,
        authored_model,
    )
    train = expand_env(authored_train)
    dataset = expand_env(authored_dataset)
    model = expand_env(authored_model)
    return train, dataset, model, dataset_path, model_path


def validate_multires_event_v2_configs(
    train: Mapping[str, Any],
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        train,
        {
            "schema_version", "route", "run_name", "seed", "lab_scale_artifact",
            "lab_scale_artifact_hash", "standardized_primitive_scale_artifact",
            "standardized_primitive_scale_artifact_hash", "trajectory_metric_contract",
            "trajectory_metric_contract_hash", "dataset", "model", "objective",
            "evaluation", "training", "outputs",
        },
        "train",
    )
    _require_exact_keys(
        dataset,
        {
            "schema_version", "route", "base", "target", "expected_counts",
            "split_authority", "split_key", "join_key", "join_guards",
            "supervision_path", "supervision_sha256", "normalization", "loader",
            "preflight",
        },
        "dataset",
    )
    _require_exact_keys(
        model,
        {
            "schema_version", "route", "role", "initialization", "text_backbone",
            "tokenizer", "architecture", "relation_contract", "primitive_contract",
            "formal_contract",
        },
        "model",
    )
    for label, payload in (("train", train), ("dataset", dataset), ("model", model)):
        if payload.get("route") != ROUTE:
            raise ValueError(f"{label} route must be {ROUTE!r}")
    if train.get("schema_version") != "trauma_predict.multires_event_v2_train_config.v2":
        raise ValueError("train schema_version differs from relation V2")
    if dataset.get("schema_version") != "trauma_predict.multires_event_v2_dataset_config.v2":
        raise ValueError("dataset schema_version differs from relation V2")
    if model.get("schema_version") != "trauma_predict.multires_event_v2_model_config.v2":
        raise ValueError("model schema_version differs from relation V2")
    if int(train.get("seed", -1)) != 20260713:
        raise ValueError("relation V2 seed must equal 20260713")
    train_dataset = _mapping(train.get("dataset"), "train.dataset")
    train_model = _mapping(train.get("model"), "train.model")
    _require_exact_keys(train_dataset, {"config_path"}, "train.dataset")
    _require_exact_keys(train_model, {"config_path"}, "train.model")
    if (
        train_dataset.get("config_path")
        != "configs/dataset/multires_event_v2_relation_v2_c4.yaml"
    ):
        raise ValueError("train.dataset.config_path differs from the frozen relation V2 route")
    if (
        train_model.get("config_path")
        != "configs/model/multires_event_v2_relation_v2.yaml"
    ):
        raise ValueError("train.model.config_path differs from the frozen relation V2 route")
    _require_exact_keys(
        _mapping(train.get("outputs"), "train.outputs"),
        {"output_dir", "metrics_jsonl"},
        "train.outputs",
    )
    if {"mode", "comparison"}.intersection(train):
        raise ValueError("relation V2 train config cannot contain mode/comparison switches")
    architecture_payload = _mapping(model.get("architecture"), "model.architecture")
    if {"mode", "relation_type_count", "relation_types"}.intersection(
        architecture_payload
    ):
        raise ValueError("relation V2 architecture cannot contain relation-off switches")
    _validate_relation_config(_mapping(model.get("relation_contract"), "model.relation_contract"))
    primitive_contract = _mapping(model.get("primitive_contract"), "model.primitive_contract")
    expected_primitive_contract = {
        "parameter_dims_source": (
            "trauma_predict.training.multires_event_v2_loss.V2_PRIMITIVE_HEAD_DIMS"
        ),
        "feedback_dims_source": (
            "trauma_predict.training.multires_event_v2_loss.V2_PRIMITIVE_FEEDBACK_DIMS"
        ),
        "stochastic_factors_per_anchor": 414,
        "output_resolution": "M4",
        "output_blocks": 6,
        "registered_fields": 29,
    }
    _require_exact_keys(
        primitive_contract,
        set(expected_primitive_contract),
        "model.primitive_contract",
    )
    if dict(primitive_contract) != expected_primitive_contract:
        raise ValueError("model.primitive_contract differs from the 414-factor contract")
    formal_contract = _mapping(model.get("formal_contract"), "model.formal_contract")
    expected_formal_contract = {
        "exact_parameter_count": EXPECTED_FORMAL_MODEL_PARAMETER_COUNT,
        "relation_contract_required": True,
        "runtime_relation_override": "forbidden",
        "causal_cross_block_attention": True,
        "target_target_registered_bias": True,
        "input_target_registered_bias": True,
        "silent_capacity_fallback": "forbidden",
        "legacy_checkpoint_loading": "forbidden",
        "h1_head": False,
        "f24_head": False,
    }
    _require_exact_keys(
        formal_contract,
        set(expected_formal_contract),
        "model.formal_contract",
    )
    if dict(formal_contract) != expected_formal_contract:
        raise ValueError("model.formal_contract differs from strict relation V2")

    base = _mapping(dataset.get("base"), "dataset.base")
    target = _mapping(dataset.get("target"), "dataset.target")
    expected_base = {
        "root": "${TRAUMA_PREDICT_DATA_ROOT}",
        "dataset_id": EXPECTED_BASE_DATASET_ID,
        "fingerprint": EXPECTED_BASE_FINGERPRINT,
        "source_fingerprint": EXPECTED_BASE_SOURCE_FINGERPRINT,
        "dataset_manifest_sha256": EXPECTED_BASE_MANIFEST_SHA256,
        "sample_manifest_sha256": EXPECTED_BASE_SAMPLE_MANIFEST_SHA256,
        "subject_split_sha256": EXPECTED_SUBJECT_SPLIT_SHA256,
    }
    _require_exact_keys(
        base,
        {
            "root", "dataset_id", "fingerprint", "source_fingerprint",
            "dataset_manifest_sha256", "sample_manifest_sha256",
            "subject_split_sha256",
        },
        "dataset.base",
    )
    _require_exact_keys(
        target,
        {
            "root", "dataset_id", "dataset_manifest_sha256", "sample_manifest_sha256",
            "contract_bundle_hash", "process_contract_sha256",
            "emission_contract_sha256", "projection_contract_sha256",
            "embedded_historical_relation_contract_sha256", "sidecar_schema_sha256",
        },
        "dataset.target",
    )
    expected_target = {
        "root": "${TRAUMA_PREDICT_V2_TARGET_ROOT}",
        "dataset_id": EXPECTED_TARGET_DATASET_ID,
        "dataset_manifest_sha256": EXPECTED_TARGET_MANIFEST_SHA256,
        "sample_manifest_sha256": EXPECTED_TARGET_SAMPLE_MANIFEST_SHA256,
        "contract_bundle_hash": EXPECTED_CONTRACT_BUNDLE_HASH,
        "process_contract_sha256": EXPECTED_PROCESS_CONTRACT_SHA256,
        "emission_contract_sha256": EXPECTED_EMISSION_CONTRACT_SHA256,
        "projection_contract_sha256": EXPECTED_PROJECTION_CONTRACT_SHA256,
        "embedded_historical_relation_contract_sha256": (
            EXPECTED_EMBEDDED_HISTORICAL_RELATION_CONTRACT_SHA256
        ),
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
    _require_exact_keys(
        _mapping(dataset.get("expected_counts"), "dataset.expected_counts"),
        set(EXPECTED_COUNTS),
        "dataset.expected_counts",
    )
    expected_dataset_scalars = {
        "split_authority": "base/sample_manifest.csv",
        "split_key": "subject_id",
        "join_key": "sample_id",
        "join_guards": ["base_content_hash", "target_content_hash"],
        "supervision_path": "configs/model/multires_event_v1_supervision.json",
        "supervision_sha256": EXPECTED_SUPERVISION_SHA256,
    }
    for key, expected in expected_dataset_scalars.items():
        if dataset.get(key) != expected:
            raise ValueError(f"dataset.{key} differs from the frozen input contract")
    normalization = _mapping(dataset.get("normalization"), "dataset.normalization")
    expected_normalization = {
        "path": "${TRAUMA_PREDICT_OUTPUT_ROOT}/contracts/multires_event_v1_input_normalization.json",
        "artifact_sha256": EXPECTED_INPUT_NORMALIZATION_SHA256,
        "fit_if_missing": False,
        "fit_split": "train",
        "fit_by_subject_only": True,
        "clip_value": 10.0,
        "epsilon": 1.0e-6,
        "max_values_per_key": 200000,
        "seed": 20260713,
    }
    _require_exact_keys(normalization, set(expected_normalization), "dataset.normalization")
    if dict(normalization) != expected_normalization:
        raise ValueError("dataset.normalization differs from the frozen V1 input contract")
    loader = _mapping(dataset.get("loader"), "dataset.loader")
    expected_loader = {
        "cache_shards": 1,
        "num_workers": 0,
        "pin_memory": True,
        "persistent_workers": False,
    }
    _require_exact_keys(loader, set(expected_loader), "dataset.loader")
    if dict(loader) != expected_loader:
        raise ValueError("dataset.loader differs from the formal route")
    preflight = _mapping(dataset.get("preflight"), "dataset.preflight")
    expected_preflight = {
        "verify_all_shard_headers": True,
        "verify_shard_sha256": False,
        "verify_target_shard_sha256": False,
    }
    _require_exact_keys(preflight, set(expected_preflight), "dataset.preflight")
    if dict(preflight) != expected_preflight:
        raise ValueError("dataset.preflight differs from the formal route")
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
    if str(train.get("trajectory_metric_contract")) != EXPECTED_TRAJECTORY_METRIC_CONTRACT:
        raise ValueError("V2 evaluation must use the frozen trajectory metric contract")
    if (
        str(train.get("trajectory_metric_contract_hash"))
        != EXPECTED_TRAJECTORY_METRIC_CONTRACT_HASH
    ):
        raise ValueError("V2 trajectory metric contract hash differs from the frozen identity")

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
    _require_exact_keys(objective, set(required_objective), "train.objective")
    for key, expected in required_objective.items():
        if objective.get(key) != expected:
            raise ValueError(f"objective.{key} must equal {expected!r}")

    if model.get("initialization") != "from_scratch":
        raise ValueError("V2 must initialize from scratch")
    if model.get("text_backbone") is not None or model.get("tokenizer") is not None:
        raise ValueError("V2 structured process model cannot use a text backbone/tokenizer")
    if model.get("role") != "relation_v2":
        raise ValueError("V2 formal model config must declare role=relation_v2")
    architecture = architecture_payload
    _require_exact_keys(
        architecture,
        {
            "hidden_size", "num_attention_heads", "trajectory_encoder_layers",
            "target_decoder_layers", "block_compressor_layers", "block_latent_count",
            "dropout", "field_vocab_size", "operator_vocab_size",
            "condition_vocab_size", "role_vocab_size", "resolution_vocab_size",
            "static_numeric_fields", "static_categorical_fields",
            "static_categorical_vocab_size", "study_slot_vocab_size",
            "time_scale_hours", "input_only_temporal_fusion",
            "future_block_count", "target_field_count",
            "target_field_ids",
        },
        "model.architecture",
    )
    expected_architecture = {
        "hidden_size": 480,
        "num_attention_heads": 8,
        "trajectory_encoder_layers": 6,
        "target_decoder_layers": 6,
        "block_compressor_layers": 1,
        "block_latent_count": 8,
        "dropout": 0.1,
        "field_vocab_size": 38,
        "operator_vocab_size": 11,
        "condition_vocab_size": 64,
        "role_vocab_size": 8,
        "resolution_vocab_size": 4,
        "static_numeric_fields": 4,
        "static_categorical_fields": 5,
        "static_categorical_vocab_size": 32,
        "study_slot_vocab_size": 16,
        "time_scale_hours": 24.0,
        "input_only_temporal_fusion": "block_geometry_softmax_v1",
        "future_block_count": 6,
        "target_field_count": 29,
        "target_field_ids": list(REGISTERED_CORE_FIELD_IDS),
    }
    if dict(architecture) != expected_architecture:
        raise ValueError("model.architecture differs from the frozen relation V2 model")
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
    _require_exact_keys(
        evaluation,
        {
            "checkpoint_metric", "interval_anchor_policy", "interval_expected_samples",
            "final_anchor_policy", "final_expected_samples", "subject_macro",
            "no_ddp_padding_duplicates", "free_running_final",
            "free_running_trajectories_per_anchor",
            "free_running_trajectory_batch_size", "free_running_crn_seed",
        },
        "train.evaluation",
    )
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
    if evaluation.get("free_running_final") is not True:
        raise ValueError("relation V2 formal evaluation requires free-running trajectories")
    required_free_running = {
        "free_running_trajectories_per_anchor": 100,
        "free_running_trajectory_batch_size": 100,
        "free_running_crn_seed": 20260713,
    }
    for key, expected in required_free_running.items():
        if int(evaluation.get(key, -1)) != expected:
            raise ValueError(f"evaluation.{key} must equal {expected}")
    expected_evaluation = {
        "checkpoint_metric": "joint_nll_subject_macro",
        "interval_anchor_policy": "all_validation_anchors",
        "interval_expected_samples": 6309,
        "final_anchor_policy": "all_validation_anchors",
        "final_expected_samples": 6309,
        "subject_macro": True,
        "no_ddp_padding_duplicates": True,
        "free_running_final": True,
        "free_running_trajectories_per_anchor": 100,
        "free_running_trajectory_batch_size": 100,
        "free_running_crn_seed": 20260713,
    }
    if dict(evaluation) != expected_evaluation:
        raise ValueError("train.evaluation differs from the frozen relation V2 evaluator")
    training = _mapping(train.get("training"), "train.training")
    _require_exact_keys(
        training,
        {
            "required_cuda_devices", "required_world_size",
            "required_device_name_substring", "precision",
            "per_device_train_batch_size", "per_device_eval_batch_size",
            "gradient_accumulation_steps", "grad_scaler_initial_scale",
            "grad_scaler_growth_factor", "grad_scaler_backoff_factor",
            "grad_scaler_growth_interval", "max_consecutive_scaler_skips",
            "grad_scaler_overflow_policy", "train_samples_per_epoch", "max_steps",
            "optimizer_contract_version", "loss_reduction", "optimizer",
            "learning_rate", "warmup_steps", "weight_decay", "adamw_betas",
            "adamw_eps", "adamw_amsgrad", "adamw_maximize", "adamw_foreach",
            "adamw_fused", "gradient_clipping", "dataloader_num_workers",
            "logging_steps", "eval_steps", "save_steps", "initial_checkpoint_step",
            "keep_last_checkpoints", "max_train_subjects", "max_eval_subjects",
            "resume", "ddp_find_unused_parameters",
        },
        "train.training",
    )
    expected_training = {
        "required_cuda_devices": 1,
        "required_world_size": 1,
        "required_device_name_substring": "P100",
        "precision": "fp16",
        "per_device_train_batch_size": 64,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "grad_scaler_initial_scale": 32.0,
        "grad_scaler_growth_factor": 2.0,
        "grad_scaler_backoff_factor": 0.5,
        "grad_scaler_growth_interval": 1_000_000,
        "max_consecutive_scaler_skips": 0,
        "grad_scaler_overflow_policy": "fail_run_preserve_rows",
        "train_samples_per_epoch": 3072,
        "max_steps": 4000,
        "optimizer_contract_version": OPTIMIZER_CONTRACT_VERSION,
        "loss_reduction": RAW_JOINT_NLL_REDUCTION,
        "optimizer": "AdamW",
        "learning_rate": 2.0e-4,
        "warmup_steps": 400,
        "weight_decay": 0.01,
        "adamw_betas": [0.9, 0.999],
        "adamw_eps": 1.0e-8,
        "adamw_amsgrad": False,
        "adamw_maximize": False,
        "adamw_foreach": False,
        "adamw_fused": False,
        "gradient_clipping": "disabled",
        "dataloader_num_workers": 0,
        "logging_steps": 100,
        "eval_steps": 250,
        "save_steps": 500,
        "initial_checkpoint_step": 2,
        "keep_last_checkpoints": 3,
        "max_train_subjects": None,
        "max_eval_subjects": None,
        "resume": True,
        "ddp_find_unused_parameters": False,
    }
    if dict(training) != expected_training:
        raise ValueError("train.training differs from the frozen relation V2 optimizer route")
    if int(training.get("required_world_size", -1)) != 1:
        raise ValueError("relation V2 P100 hosted training requires one process")
    if int(training.get("required_cuda_devices", -1)) != 1:
        raise ValueError("relation V2 P100 hosted training requires one CUDA device")
    if training.get("required_device_name_substring") != "P100":
        raise ValueError("relation V2 hosted hardware must be a P100")
    if training.get("precision") != "fp16":
        raise ValueError("relation V2 P100 training requires fp16 neural forward")
    if int(training.get("per_device_train_batch_size", -1)) != 64:
        raise ValueError("relation V2 freezes the single-P100 train batch size at 64")
    if int(training.get("gradient_accumulation_steps", -1)) != 1:
        raise ValueError("relation V2 freezes gradient accumulation at one")
    if int(training.get("train_samples_per_epoch", -1)) != 3072:
        raise ValueError(
            "relation V2 freezes 3,072 uniform-subject replacement draws per epoch"
        )
    if (
        int(training["per_device_train_batch_size"])
        * int(training["required_world_size"])
        * int(training["gradient_accumulation_steps"])
        != 64
    ):
        raise ValueError("relation V2 requires effective batch size 64")
    if int(training.get("per_device_eval_batch_size", -1)) != 32:
        raise ValueError(
            "relation V2 freezes eval batch size at 32; subject macro is row-wise"
        )
    required_scaler = {
        "grad_scaler_initial_scale": 32.0,
        "grad_scaler_growth_factor": 2.0,
        "grad_scaler_backoff_factor": 0.5,
        "grad_scaler_growth_interval": 1_000_000,
        "max_consecutive_scaler_skips": 0,
        "grad_scaler_overflow_policy": "fail_run_preserve_rows",
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
        raise ValueError("relation V2 requires every registered relation path to be trainable")


def _validate_relation_config(configuration: Mapping[str, Any]) -> None:
    expected = {
        "config_dir": EXPECTED_RELATION_CONFIG_DIR,
        "version": RELATION_CONTRACT_VERSION,
        "bundle_sha256": EXPECTED_RELATION_BUNDLE_SHA256,
        "files": dict(EXPECTED_RELATION_FILE_HASHES),
        "runtime_override": "forbidden",
        "target_target_edges": 52,
        "input_target_edges": 39,
    }
    if set(configuration) != set(expected):
        raise ValueError("relation V2 model config keys differ from the frozen contract")
    for key, value in expected.items():
        if configuration.get(key) != value:
            raise ValueError(f"model.relation_contract.{key} differs from relation V2")


def _validate_run_profile(
    train: Mapping[str, Any],
    *,
    training: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> None:
    run_name = str(train.get("run_name") or "")
    expected_name = "p100_multires_event_v2_relation_v2"
    if run_name != expected_name:
        raise ValueError(f"relation V2 run_name must equal {expected_name!r}")
    expected = {
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
    for key in ("max_steps", "warmup_steps", "eval_steps", "save_steps", "logging_steps"):
        observed = training.get(key)
        if isinstance(observed, bool) or not isinstance(observed, int) or observed != expected[key]:
            raise ValueError(f"training.{key} must equal {expected[key]!r} for {run_name}")
    observed_initial = training.get("initial_checkpoint_step")
    if (
        isinstance(observed_initial, bool)
        or not isinstance(observed_initial, int)
        or observed_initial != expected["initial_checkpoint_step"]
    ):
        raise ValueError("training.initial_checkpoint_step must equal 2")
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


def run_contract_signature(
    train: Mapping[str, Any],
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
) -> str:
    """Hash the complete single-model run contract."""

    return sha256_payload({"train": train, "dataset": dataset, "model": model})


def build_multires_event_v2_model(
    model: Mapping[str, Any],
    *,
    relation_contract: MultiresEventV2RelationContract,
) -> MultiResolutionEventV2Model:
    if (
        relation_contract.version != RELATION_CONTRACT_VERSION
        or relation_contract.bundle_hash != EXPECTED_RELATION_BUNDLE_SHA256
        or dict(relation_contract.file_hashes) != dict(EXPECTED_RELATION_FILE_HASHES)
    ):
        raise ValueError("formal model builder requires the exact relation V2 bundle")
    architecture = dict(_mapping(model.get("architecture"), "model.architecture"))
    architecture.update(
        primitive_head_dims=V2_PRIMITIVE_HEAD_DIMS,
        primitive_feedback_dims=V2_PRIMITIVE_FEEDBACK_DIMS,
    )
    built = MultiResolutionEventV2Model(
        MultiResolutionEventV2Config.from_mapping(architecture),
        relation_contract,
    )
    if _is_formal_model_architecture(architecture):
        validate_formal_model_parameter_count(built)
    return built


def build_multires_event_v2_optimizer(
    model: torch.nn.Module,
    training: Mapping[str, Any],
) -> torch.optim.AdamW:
    """Build the one frozen AdamW implementation used by the formal route."""

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
    expected_local_anchors = int(training["per_device_train_batch_size"])
    expected_world_size = int(training["required_world_size"])
    expected_global_anchors = (
        expected_local_anchors
        * expected_world_size
        * int(training["gradient_accumulation_steps"])
    )
    valid = (
        row.get("event") == "v2_optimizer_health"
        and row.get("optimizer_contract_version") == OPTIMIZER_CONTRACT_VERSION
        and row.get("loss_reduction") == RAW_JOINT_NLL_REDUCTION
        and int(row.get("local_anchors", -1)) == expected_local_anchors
        and int(row.get("world_size", -1)) == expected_world_size
        and int(row.get("global_anchors", -1)) == expected_global_anchors
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


def validate_relation_runtime_axes(
    relation_contract: MultiresEventV2RelationContract,
    target_contract: MultiresEventV2Contract,
    templates: Any,
) -> None:
    """Bind relation target/history axes to the mounted r9 and V1 registries."""

    relation_contract.assert_target_field_order(target_contract.core_fields)
    by_id: dict[int, str] = {}
    for template in templates.by_key.values():
        field_id = int(template.field_id)
        field = str(template.field)
        existing = by_id.get(field_id)
        if existing is not None and existing != field:
            raise ValueError(f"V1 template field_id={field_id} has conflicting names")
        by_id[field_id] = field
    observed = tuple((field_id, by_id.get(field_id)) for field_id in range(1, 38))
    expected = tuple((field.field_id, field.field) for field in relation_contract.fields)
    if observed != expected:
        raise ValueError(
            "relation V2 37-field history axis differs from the V1 event-template registry"
        )


def validate_formal_model_parameter_count(model: torch.nn.Module) -> int:
    """Reject any formal model whose total parameterization drifted from the freeze."""

    observed = sum(parameter.numel() for parameter in model.parameters())
    if observed != EXPECTED_FORMAL_MODEL_PARAMETER_COUNT:
        raise ValueError(
            "formal V2 model parameter count differs from the frozen relation V2 design: "
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
    expected_normalization_sha256 = str(
        dataset["normalization"]["artifact_sha256"]
    )
    if not normalization_path.is_file():
        raise FileNotFoundError(
            "formal relation V2 requires the frozen input normalization artifact; "
            f"refusing to refit missing {normalization_path}"
        )
    observed_normalization_sha256 = sha256_file(normalization_path)
    if observed_normalization_sha256 != expected_normalization_sha256:
        raise ValueError(
            "formal relation V2 input normalization hash mismatch: "
            f"{observed_normalization_sha256} != {expected_normalization_sha256}"
        )

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
    relation_configuration = _mapping(
        runtime_model_config.get("relation_contract"),
        "model.relation_contract",
    )
    _validate_relation_config(relation_configuration)
    relation_root = resolve_repo_path(
        str(relation_configuration["config_dir"]),
        root,
    )
    relation_contract = MultiresEventV2RelationContract.from_config_dir(relation_root)
    if (
        relation_contract.version != RELATION_CONTRACT_VERSION
        or relation_contract.bundle_hash != EXPECTED_RELATION_BUNDLE_SHA256
        or dict(relation_contract.file_hashes) != dict(EXPECTED_RELATION_FILE_HASHES)
    ):
        raise ValueError("loaded relation V2 bundle differs from the model contract")
    validate_relation_runtime_axes(
        relation_contract,
        contract,
        base_runtime.train_dataset.templates,
    )
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
    trajectory_metric_contract_path = resolve_repo_path(
        str(train.get("trajectory_metric_contract", "")), root
    )
    load_trajectory_metric_contract(
        trajectory_metric_contract_path,
        expected_sha256=str(train.get("trajectory_metric_contract_hash", "")),
        relation_contract=relation_contract,
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
        "target_sidecar_relation_contract_sha256": contract.contract_hashes["relation"],
        "relation_contract_version": relation_contract.version,
        "relation_contract_sha256": relation_contract.bundle_hash,
        "relation_contract_bundle_sha256": relation_contract.bundle_hash,
        "relation_contract_file_sha256": dict(relation_contract.file_hashes),
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
        "trajectory_metric_contract": str(trajectory_metric_contract_path),
        "trajectory_metric_contract_sha256": str(
            train["trajectory_metric_contract_hash"]
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
        relation_contract=relation_contract,
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
        "trajectory_metric_contract_sha256",
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
        run_contract = str(identity_hashes.get("run_contract") or "")
        if not _is_sha256(run_contract):
            raise ValueError("V2 run-contract identity is not SHA-256")
        result.update(
            {
                "source_tree_sha256": expected_source["source_tree"],
                "source_identity_sha256": source_sha,
                "git_commit": expected_source["git_commit"],
                "git_head_tree": expected_source["git_head_tree"],
                "run_contract_signature": run_contract,
                "selected_checkpoint_step": selected_step,
                "selected_checkpoint_model_sha256": selected_sha,
            }
        )
    return result


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


def run_multires_event_v2_rank_artifact_preflight_only(
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run exact single-P100 artifact paths before Dataset loading.

    This route accepts no model or data config and performs no Dataset scan. It
    exercises the rank-local writer/hash path and the production best-checkpoint
    save/load boundary.  The checkpoint canary uses a
    zero-parameter Identity module; it is an I/O/collective contract check, not
    a model or optimization attempt.  A hosted attempt can therefore
    reject deterministic collective-order defects before materializing the
    50,350-anchor runtime.
    """

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 1 or local_rank != 0:
        raise RuntimeError("early V2 P100 artifact preflight requires one process")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("early V2 P100 artifact preflight requires one visible GPU")
    if "P100" not in torch.cuda.get_device_name(0):
        raise RuntimeError(
            "early V2 hosted preflight requires a P100; observed "
            f"{torch.cuda.get_device_name(0)!r}"
        )
    output_root = Path(output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("early rank artifact preflight output is not empty")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    completed = False
    try:
        result = verify_rank_local_artifact_preflight(
            output_dir=output_root,
        )
        if int(result.get("world_size", -1)) != world_size:
            raise RuntimeError("early rank artifact preflight world size changed")
        _run_v2_best_checkpoint_collective_canary(
            output_root=output_root / "best-checkpoint-collective-canary",
            world_size=world_size,
        )
        if is_rank_zero():
            print(
                "MULTIRES_EVENT_V2_P100_ARTIFACT_CANARY_OK "
                f"phase=predata world_size={world_size} best_checkpoint=verified "
                f"sha256={result['manifest_sha256']}",
                flush=True,
            )
        completed = True
        return result
    finally:
        # Keep the cleanup compatible with the historical distributed helper,
        # although the active P100 route intentionally has no process group.
        if completed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def run_multires_event_v2_training(
    train_config_path: str | Path,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Run the authorized relation V2 model from a fresh initialization."""

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
    model = build_multires_event_v2_model(
        model_config,
        relation_contract=runtime.relation_contract,
    ).to(device)
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
        "run_contract": run_contract_signature(train, dataset, model_config),
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
    _load_or_run_free_running_capacity_probe(
        output_dir=output_dir,
        model=model,
        runtime=runtime,
        device=device,
        train=train,
        identity_hashes=identity_hashes,
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
    if isinstance(result.get("hosted_stop_after_step"), int):
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
    final_evaluation = _load_or_evaluate_final_teacher(
        output_dir=output_dir,
        model=model,
        runtime=final_runtime,
        device=device,
        step=selected_step,
        precision=str(training["precision"]),
        metrics_path=metrics_path,
        expected_lab_scale_artifact_hash=str(train["lab_scale_artifact_hash"]),
        evaluation_identity=final_evaluation_identity,
        expected_samples=int(train["evaluation"]["final_expected_samples"]),
    )
    free_running_evaluation: dict[str, Any] | None = None
    if bool(train["evaluation"]["free_running_final"]):
        trajectory_metric_contract = load_trajectory_metric_contract(
            output_dir / RUN_ARTIFACT_PATHS["trajectory_metric_contract"],
            expected_sha256=str(train["trajectory_metric_contract_hash"]),
            relation_contract=final_runtime.relation_contract,
        )
        free_running_evaluation = evaluate_free_running_v2(
            model=model,
            loader=final_runtime.eval_loader,
            contract=final_runtime.contract,
            relation_contract=final_runtime.relation_contract,
            device=device,
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
            trajectory_metric_contract=trajectory_metric_contract,
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
            max_new_anchors=_hosted_free_running_max_new_anchors(),
        )
        if free_running_evaluation.get("status") == "INCOMPLETE":
            completed = int(free_running_evaluation.get("anchors", -1))
            expected = int(
                free_running_evaluation.get("expected_anchors", -1)
            )
            new_anchors = int(
                free_running_evaluation.get("new_anchors_this_invocation", -1)
            )
            if (
                completed < 1
                or completed >= expected
                or expected != int(train["evaluation"]["final_expected_samples"])
                or new_anchors < 1
            ):
                raise ValueError("free-running hosted partial result is invalid")
            print(
                "MULTIRES_EVENT_V2_FREE_RUNNING_HOSTED_STOP_OK "
                f"completed={completed} expected={expected} new={new_anchors}",
                flush=True,
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            return {
                **result,
                "final_evaluation": final_evaluation,
                "free_running_evaluation": free_running_evaluation,
                "hosted_free_running_incomplete": True,
            }
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
        expected_local_batch != 64
        or expected_world_size != 1
        or world_size != expected_world_size
        or accumulation != 1
        or expected_local_batch * expected_world_size * accumulation != 64
    ):
        raise RuntimeError(
            "formal V2 optimizer loop requires exact single-P100 B64, "
            "world size 1, accumulation 1"
        )
    global_step = int(state.get("global_step", 0))
    hosted_stop_step = _hosted_verification_stop_step(
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
            raise RuntimeError("formal V2 training requires exactly 64 local anchors")
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
                    "V2 FP16 gradient overflow invalidates the formal run: retrying "
                    "would consume a different stochastic row/RNG schedule; "
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

        interval_evaluation: dict[str, Any] | None = None
        if global_step % int(training["eval_steps"]) == 0 or global_step == max_steps:
            interval_evaluation = evaluate_teacher_forced(
                model=model,
                loader=runtime.eval_loader,
                registry=runtime.contract.process_registry,
                device=device,
                expected_samples=int(train["evaluation"]["interval_expected_samples"]),
                phase="interval",
                step=global_step,
                precision=str(training["precision"]),
                metrics_path=metrics_path,
                expected_lab_scale_artifact_hash=train.get("lab_scale_artifact_hash"),
                evaluation_identity=_evaluation_contract_identity(runtime),
            )
            candidate = float(interval_evaluation["joint_nll_subject_macro"])
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
            or global_step == hosted_stop_step
        ):
            checkpoint_trainer_state = {
                "global_step": global_step,
                "epoch": epoch,
                "batches_in_epoch": batches_in_epoch,
                "micro_in_accum": micro_in_accum,
                "best_metric": best_metric,
                "best_step": best_step,
                "scaler_skipped_steps": scaler_skipped_steps,
            }
            _save_v2_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                trainer_state=checkpoint_trainer_state,
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
                    "model_contract": "relation_v2",
                    "run_name": str(train["run_name"]),
                    "global_step": global_step,
                    "checkpoint": str(checkpoint.relative_to(output_dir)),
                    "checkpoint_manifest_sha256": sha256_file(manifest_path),
                    "model_parameter_count": EXPECTED_FORMAL_MODEL_PARAMETER_COUNT,
                    "target_dataset_id": EXPECTED_TARGET_DATASET_ID,
                    "contract_bundle_hash": EXPECTED_CONTRACT_BUNDLE_HASH,
                    "relation_contract_sha256": EXPECTED_RELATION_BUNDLE_SHA256,
                    "identity_hashes": dict(identity_hashes),
                }
                atomic_write_json(output_dir / "formal_step2_readiness.json", readiness)
                print(
                    "MULTIRES_EVENT_V2_FORMAL_STEP2_CHECKPOINT_OK "
                    f"model_contract=relation_v2 parameters={EXPECTED_FORMAL_MODEL_PARAMETER_COUNT} "
                    f"path={checkpoint}",
                    flush=True,
                )
            if global_step == initial_checkpoint_step:
                _reopen_v2_checkpoint_in_place(
                    output_dir=output_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    runtime=runtime,
                    device=device,
                    rank=rank,
                    expected_trainer_state=checkpoint_trainer_state,
                    expected_identity_hashes=identity_hashes,
                    training=training,
                )
            if hosted_stop_step is not None and global_step == hosted_stop_step:
                readiness = _materialize_hosted_stop_readiness(
                    output_dir=output_dir,
                    model=model,
                    device=device,
                    identity_hashes=identity_hashes,
                    stop_step=global_step,
                    best_step=best_step,
                    interval_evaluation=interval_evaluation,
                )
                stopped = {
                    "global_step": global_step,
                    "epochs_started": epoch + 1,
                    "best_metric": best_metric,
                    "best_step": best_step,
                    "max_steps": max_steps,
                    "scaler_skipped_steps": scaler_skipped_steps,
                    "hosted_stop_after_step": global_step,
                    "hosted_stop_readiness": readiness,
                }
                if global_step == 2:
                    stopped["hosted_verification_stop_after_step"] = global_step
                return stopped
    return {
        "global_step": global_step,
        "epochs_started": epoch + 1,
        "best_metric": best_metric,
        "best_step": best_step,
        "max_steps": max_steps,
        "scaler_skipped_steps": scaler_skipped_steps,
    }


def _load_or_run_free_running_capacity_probe(
    *,
    output_dir: Path,
    model: Any,
    runtime: MultiresEventV2Runtime,
    device: torch.device,
    train: Mapping[str, Any],
    identity_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Prove the production 100-trajectory path before creating AdamW state."""

    evidence_path = output_dir / "formal_free_running_capacity_probe.json"
    expected_device_name = torch.cuda.get_device_name(device)
    if evidence_path.is_symlink():
        raise ValueError("formal capacity evidence cannot be a symlink")
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if (
            evidence.get("schema_version")
            != "trauma_predict.multires_event_v2_free_running_capacity_probe.v1"
            or evidence.get("status") != "PASSED"
            or evidence.get("identity_hashes") != dict(identity_hashes)
            or int(evidence.get("model_parameter_count", -1))
            != EXPECTED_FORMAL_MODEL_PARAMETER_COUNT
            or evidence.get("device_name") != expected_device_name
            or int(evidence.get("trajectories_per_anchor", -1)) != 100
            or int(evidence.get("trajectory_batch_size", -1)) != 100
            or int(evidence.get("encode_calls", -1)) != 1
            or evidence.get("parameter_state_unchanged") is not True
            or evidence.get("parameter_state_sha256_before")
            != evidence.get("parameter_state_sha256_after")
        ):
            raise ValueError("formal free-running capacity evidence is invalid")
        print(
            "MULTIRES_EVENT_V2_FREE_RUNNING_CAPACITY_REUSED "
            f"sample_id={evidence['sample_id']} device={expected_device_name}",
            flush=True,
        )
        return evidence

    checkpoint_root = output_dir / "checkpoints"
    if checkpoint_root.is_dir() and any(checkpoint_root.glob("checkpoint-*")):
        raise FileNotFoundError(
            "resumed training lacks the pre-optimizer free-running capacity evidence"
        )
    try:
        validation_batch = next(iter(runtime.eval_loader))
    except StopIteration as exc:  # pragma: no cover - frozen val set is nonempty
        raise RuntimeError("formal capacity probe found no validation batch") from exc
    evaluation = _mapping(train.get("evaluation"), "train.evaluation")
    probe = probe_free_running_v2_capacity(
        model=model,
        validation_batch=validation_batch,
        contract=runtime.contract,
        device=device,
        expected_lab_scale_artifact_hash=str(train["lab_scale_artifact_hash"]),
        crn_seed=int(evaluation["free_running_crn_seed"]),
        precision=str(train["training"]["precision"]),
        trajectories_per_anchor=int(
            evaluation["free_running_trajectories_per_anchor"]
        ),
        trajectory_batch_size=int(
            evaluation["free_running_trajectory_batch_size"]
        ),
    )
    evidence = {
        **probe,
        "created_at": utc_now(),
        "device_name": expected_device_name,
        "model_parameter_count": EXPECTED_FORMAL_MODEL_PARAMETER_COUNT,
        "identity_hashes": dict(identity_hashes),
    }
    atomic_write_json(evidence_path, evidence)
    persisted = json.loads(evidence_path.read_text(encoding="utf-8"))
    if persisted != evidence:
        raise RuntimeError("formal capacity evidence persistence failed")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        "MULTIRES_EVENT_V2_FREE_RUNNING_CAPACITY_OK "
        f"sample_id={evidence['sample_id']} device={expected_device_name} "
        f"peak_allocated={evidence['cuda_memory']['peak_allocated_bytes']}",
        flush=True,
    )
    return evidence


def _load_or_evaluate_final_teacher(
    *,
    output_dir: Path,
    model: Any,
    runtime: MultiresEventV2Runtime,
    device: torch.device,
    step: int,
    precision: str,
    metrics_path: Path,
    expected_lab_scale_artifact_hash: str,
    evaluation_identity: Mapping[str, Any],
    expected_samples: int,
) -> dict[str, Any]:
    """Persist final teacher evaluation before the resumable free-running phase."""

    cache_dir = output_dir / "teacher_forced_final"
    cache_path = cache_dir / "evaluation_cache.json"
    rows_path = output_dir / "val_per_anchor_joint_nll.jsonl"
    expected_sample_ids = tuple(str(item) for item in runtime.eval_dataset.sample_ids)
    if len(expected_sample_ids) != expected_samples:
        raise ValueError("final teacher runtime does not expose the frozen validation set")

    if cache_path.is_symlink():
        raise ValueError("final teacher cache cannot be a symlink")
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        evaluation = cache.get("evaluation")
        if (
            cache.get("schema_version") != FINAL_TEACHER_CACHE_SCHEMA
            or cache.get("status") != "COMPLETE"
            or cache.get("per_anchor_path") != "val_per_anchor_joint_nll.jsonl"
            or not isinstance(evaluation, Mapping)
            or evaluation.get("phase") != "final"
            or int(evaluation.get("step", -1)) != step
            or int(evaluation.get("samples", -1)) != expected_samples
            or evaluation.get("identity") != dict(evaluation_identity)
        ):
            raise ValueError("final teacher cache contract is invalid")
        row_evidence = _validate_final_teacher_rows(
            rows_path,
            expected_sample_ids=expected_sample_ids,
            step=step,
            evaluation_identity=evaluation_identity,
        )
        if (
            cache.get("per_anchor_sha256") != row_evidence["sha256"]
            or cache.get("sample_ids_sha256")
            != row_evidence["sample_ids_sha256"]
            or int(cache.get("row_count", -1)) != expected_samples
        ):
            raise ValueError("final teacher cache rows differ from their frozen evidence")
        reused = dict(evaluation)
        reused["per_anchor_output_path"] = str(rows_path)
        reused["per_anchor_output_sha256"] = row_evidence["sha256"]
        append_jsonl(
            metrics_path,
            {
                "event": "v2_final_evaluation_reused",
                "created_at": utc_now(),
                "step": step,
                "samples": expected_samples,
                "cache_sha256": sha256_file(cache_path),
            },
        )
        print(
            "MULTIRES_EVENT_V2_FINAL_TEACHER_REUSED "
            f"step={step} samples={expected_samples} sha256={row_evidence['sha256']}",
            flush=True,
        )
        return reused

    if rows_path.exists():
        if rows_path.is_symlink() or not rows_path.is_file():
            raise ValueError("incomplete final teacher rows are not a regular file")
        incomplete_dir = cache_dir / "incomplete"
        incomplete_dir.mkdir(parents=True, exist_ok=True)
        rows_path.replace(
            incomplete_dir
            / f"val_per_anchor_joint_nll-{utc_now().replace(':', '-')}.jsonl"
        )

    evaluation = evaluate_teacher_forced(
        model=model,
        loader=runtime.eval_loader,
        registry=runtime.contract.process_registry,
        device=device,
        expected_samples=expected_samples,
        phase="final",
        step=step,
        precision=precision,
        metrics_path=metrics_path,
        expected_lab_scale_artifact_hash=expected_lab_scale_artifact_hash,
        per_anchor_output_path=rows_path,
        evaluation_identity=evaluation_identity,
    )
    row_evidence = _validate_final_teacher_rows(
        rows_path,
        expected_sample_ids=expected_sample_ids,
        step=step,
        evaluation_identity=evaluation_identity,
    )
    portable_evaluation = dict(evaluation)
    portable_evaluation["per_anchor_output_path"] = "val_per_anchor_joint_nll.jsonl"
    portable_evaluation["per_anchor_output_sha256"] = row_evidence["sha256"]
    cache = {
        "schema_version": FINAL_TEACHER_CACHE_SCHEMA,
        "status": "COMPLETE",
        "created_at": utc_now(),
        "evaluation": portable_evaluation,
        "per_anchor_path": "val_per_anchor_joint_nll.jsonl",
        "per_anchor_sha256": row_evidence["sha256"],
        "row_count": row_evidence["row_count"],
        "sample_ids_sha256": row_evidence["sample_ids_sha256"],
    }
    atomic_write_json(cache_path, cache)
    if sha256_file(cache_path) == "":  # pragma: no cover - defensive filesystem guard
        raise RuntimeError("final teacher cache hash is empty")
    return evaluation


def _validate_final_teacher_rows(
    path: Path,
    *,
    expected_sample_ids: Sequence[str],
    step: int,
    evaluation_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("final teacher per-anchor rows are absent")
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"blank final teacher row at line {line_number}")
            row = json.loads(raw)
            if not isinstance(row, Mapping):
                raise ValueError("final teacher row must be an object")
            if (
                not str(row.get("sample_id") or "")
                or not str(row.get("subject_id") or "")
                or int(row.get("step", -1)) != step
                or int(row.get("primitive_factors", -1)) != 414
                or row.get("model_contract") != "relation_v2"
                or row.get("identity") != dict(evaluation_identity)
            ):
                raise ValueError("final teacher row identity is invalid")
            try:
                nll = float(row.get("joint_nll"))
            except (TypeError, ValueError) as exc:
                raise ValueError("final teacher row NLL is invalid") from exc
            if not math.isfinite(nll):
                raise ValueError("final teacher row NLL is non-finite")
            rows.append(row)
    sample_ids = tuple(str(row["sample_id"]) for row in rows)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("final teacher cache contains duplicate sample IDs")
    if set(sample_ids) != set(str(item) for item in expected_sample_ids):
        raise ValueError("final teacher cache does not match the persisted validation anchors")
    return {
        "row_count": len(rows),
        "sha256": sha256_file(path),
        "sample_ids_sha256": sha256_payload(sorted(sample_ids)),
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


def _hosted_free_running_max_new_anchors() -> int | None:
    raw = os.environ.get(
        "TRAUMA_PREDICT_V2_FREE_RUNNING_MAX_NEW_ANCHORS", "0"
    ).strip()
    try:
        value = int(raw or "0")
    except ValueError as exc:
        raise ValueError(
            "TRAUMA_PREDICT_V2_FREE_RUNNING_MAX_NEW_ANCHORS must be an integer"
        ) from exc
    if value < 0:
        raise ValueError(
            "TRAUMA_PREDICT_V2_FREE_RUNNING_MAX_NEW_ANCHORS cannot be negative"
        )
    return value or None


def _hosted_verification_stop_step(*, starting_global_step: int) -> int | None:
    stop_at_2 = _verification_stop_after_formal_step2_requested()
    raw_stop_step = os.environ.get("TRAUMA_PREDICT_V2_HOSTED_STOP_STEP", "0").strip()
    try:
        requested_step = int(raw_stop_step or "0")
    except ValueError as exc:
        raise ValueError(
            "TRAUMA_PREDICT_V2_HOSTED_STOP_STEP must be an integer"
        ) from exc
    if requested_step not in {0, *AUTHORIZED_HOSTED_STOP_STEPS}:
        raise ValueError(
            "TRAUMA_PREDICT_V2_HOSTED_STOP_STEP must be 0 or one of "
            f"{AUTHORIZED_HOSTED_STOP_STEPS}"
        )
    if stop_at_2:
        if requested_step not in {0, 2}:
            raise ValueError(
                "formal step-2 verification conflicts with the hosted stop step"
            )
        requested_step = 2
    if requested_step == 0:
        return None
    if requested_step <= starting_global_step:
        raise ValueError(
            "hosted stop step must be strictly greater than the restored optimizer step"
        )
    return requested_step


def _materialize_hosted_stop_readiness(
    *,
    output_dir: Path,
    model: Any,
    device: torch.device,
    identity_hashes: Mapping[str, str],
    stop_step: int,
    best_step: Any,
    interval_evaluation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Verify and persist one atomic hosted training-stage boundary."""

    if stop_step not in AUTHORIZED_HOSTED_STOP_STEPS:
        raise ValueError("hosted readiness received an unauthorized stop step")
    checkpoint = output_dir / "checkpoints" / f"checkpoint-{stop_step:08d}"
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = _validate_v2_checkpoint_directory(
        checkpoint,
        expected_world_size=1,
        expected_step=stop_step,
    )
    checkpoint_model_sha256 = str(manifest["sha256"]["model.pt"])
    if stop_step == 2:
        _validate_formal_step2_readiness(output_dir, expected_step=stop_step)
        if best_step is not None or interval_evaluation is not None:
            raise ValueError("formal step-2 stop cannot contain evaluation selection state")
        best_model_sha256: str | None = None
    else:
        if (
            isinstance(best_step, bool)
            or not isinstance(best_step, int)
            or best_step < 1
            or best_step > stop_step
        ):
            raise ValueError("hosted post-evaluation stop requires a valid best step")
        if (
            not isinstance(interval_evaluation, Mapping)
            or interval_evaluation.get("phase") != "interval"
            or int(interval_evaluation.get("step", -1)) != stop_step
            or int(interval_evaluation.get("samples", -1)) != EXPECTED_COUNTS["val"]
        ):
            raise ValueError(
                "hosted post-evaluation stop requires the complete 6,309-anchor interval result"
            )
        if stop_step == 250 and best_step != 250:
            raise ValueError("the first full validation must select the step-250 model")
        if stop_step == 250:
            selected = _load_v2_best_model(
                output_dir,
                model,
                device,
                expected_identity_hashes=identity_hashes,
                expected_best_step=best_step,
            )
            best_model_sha256 = str(
                selected["selected_checkpoint_model_sha256"]
            )
        else:
            pointer_path = output_dir / "best_checkpoint.json"
            if pointer_path.is_symlink() or not pointer_path.is_file():
                raise FileNotFoundError("hosted stop lacks a best-checkpoint pointer")
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            best_path = output_dir / "best_checkpoint/model.pt"
            best_model_sha256 = str(pointer.get("model_sha256") or "")
            if (
                pointer.get("schema_version") != BEST_CHECKPOINT_SCHEMA
                or pointer.get("step") != best_step
                or pointer.get("identity_hashes") != dict(identity_hashes)
                or not _is_sha256(best_model_sha256)
                or best_path.is_symlink()
                or not best_path.is_file()
                or sha256_file(best_path) != best_model_sha256
            ):
                raise ValueError("hosted stop best-checkpoint identity is invalid")
    readiness = {
        "schema_version": HOSTED_STOP_READINESS_SCHEMA,
        "status": "PASSED",
        "created_at": utc_now(),
        "run_name": "p100_multires_event_v2_relation_v2",
        "model_contract": "relation_v2",
        "stop_step": stop_step,
        "global_step": stop_step,
        "model_parameter_count": EXPECTED_FORMAL_MODEL_PARAMETER_COUNT,
        "checkpoint": str(checkpoint.relative_to(output_dir)),
        "checkpoint_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_model_sha256": checkpoint_model_sha256,
        "best_step": best_step,
        "best_model_sha256": best_model_sha256,
        "identity_hashes": dict(identity_hashes),
        "interval_evaluation": (
            dict(interval_evaluation) if interval_evaluation is not None else None
        ),
    }
    stable_path = output_dir / "formal_hosted_stop_readiness.json"
    history_path = output_dir / "hosted_stages" / f"step-{stop_step:08d}.json"
    atomic_write_json(stable_path, readiness)
    atomic_write_json(history_path, readiness)
    persisted = json.loads(stable_path.read_text(encoding="utf-8"))
    if persisted != readiness or sha256_file(stable_path) != sha256_file(history_path):
        raise RuntimeError("hosted stop readiness persistence failed")
    marker = (
        "MULTIRES_EVENT_V2_FORMAL_STEP250_OK"
        if stop_step == 250
        else "MULTIRES_EVENT_V2_HOSTED_STOP_OK"
    )
    print(
        f"{marker} step={stop_step} checkpoint_model_sha256={checkpoint_model_sha256} "
        f"best_model_sha256={best_model_sha256}",
        flush=True,
    )
    return readiness


def _validate_formal_step2_readiness(
    output_dir: Path,
    *,
    expected_step: int,
) -> dict[str, Any]:
    readiness_path = output_dir / "formal_step2_readiness.json"
    if readiness_path.is_symlink() or not readiness_path.is_file():
        raise FileNotFoundError("formal step-2 readiness evidence is absent")
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    checkpoint = output_dir / str(readiness.get("checkpoint") or "")
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = _validate_v2_checkpoint_directory(
        checkpoint,
        expected_world_size=1,
        expected_step=expected_step,
    )
    if (
        readiness.get("schema_version")
        != "trauma_predict.multires_event_v2_formal_step2_readiness.v1"
        or readiness.get("status") != "PASSED"
        or readiness.get("model_contract") != "relation_v2"
        or readiness.get("run_name") != "p100_multires_event_v2_relation_v2"
        or int(readiness.get("global_step", -1)) != expected_step
        or int(readiness.get("model_parameter_count", -1))
        != EXPECTED_FORMAL_MODEL_PARAMETER_COUNT
        or readiness.get("target_dataset_id") != EXPECTED_TARGET_DATASET_ID
        or readiness.get("contract_bundle_hash") != EXPECTED_CONTRACT_BUNDLE_HASH
        or readiness.get("relation_contract_sha256")
        != EXPECTED_RELATION_BUNDLE_SHA256
        or readiness.get("checkpoint_manifest_sha256") != sha256_file(manifest_path)
        or readiness.get("identity_hashes") != manifest.get("identity_hashes")
    ):
        raise ValueError("formal step-2 readiness evidence does not bind its checkpoint")
    return readiness


def _reopen_v2_checkpoint_in_place(
    *,
    output_dir: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    runtime: MultiresEventV2Runtime,
    device: torch.device,
    rank: int,
    expected_trainer_state: Mapping[str, Any],
    expected_identity_hashes: Mapping[str, str],
    training: Mapping[str, Any],
) -> None:
    """Load the step-2 checkpoint immediately and prove every resume payload."""

    expected_step = int(expected_trainer_state["global_step"])
    checkpoint = output_dir / "checkpoints" / f"checkpoint-{expected_step:08d}"
    manifest = _validate_v2_checkpoint_directory(
        checkpoint,
        expected_world_size=1,
        expected_step=expected_step,
    )
    persisted_identity = json.loads(
        (checkpoint / "identity_hashes.json").read_text(encoding="utf-8")
    )
    if (
        persisted_identity != dict(expected_identity_hashes)
        or manifest.get("identity_hashes") != dict(expected_identity_hashes)
    ):
        raise RuntimeError("formal step-2 checkpoint identity cannot be reopened")
    _unwrapped_model(model).load_state_dict(
        _torch_load(checkpoint / "model.pt", map_location=device, weights_only=True),
        strict=True,
    )
    optimizer.load_state_dict(
        _torch_load(
            checkpoint / "optimizer.pt", map_location=device, weights_only=True
        )
    )
    scheduler.load_state_dict(
        _torch_load(
            checkpoint / "scheduler.pt", map_location=device, weights_only=True
        )
    )
    scaler.load_state_dict(
        _torch_load(checkpoint / "scaler.pt", map_location=device, weights_only=True)
    )
    trainer_state = json.loads(
        (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )
    if trainer_state != dict(expected_trainer_state):
        raise RuntimeError("formal step-2 trainer state changed during reopen")
    sampler_state = _torch_load(
        checkpoint / f"sampler-rank-{rank:04d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    if sampler_state is not None:
        if not hasattr(runtime.train_sampler, "load_state_dict"):
            raise RuntimeError("formal step-2 sampler state cannot be reopened")
        runtime.train_sampler.load_state_dict(sampler_state)
    rng_state = _torch_load(
        checkpoint / f"rng-rank-{rank:04d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(rng_state, Mapping):
        raise ValueError("formal step-2 RNG state cannot be reopened")
    _restore_rng_state(rng_state)
    _validate_resume_optimizer_alignment(
        optimizer,
        scheduler,
        training,
        global_step=expected_step,
    )
    _validate_formal_step2_readiness(output_dir, expected_step=expected_step)
    print(
        "MULTIRES_EVENT_V2_FORMAL_STEP2_REOPEN_OK "
        f"step={expected_step} checkpoint_manifest_sha256="
        f"{sha256_file(checkpoint / 'checkpoint_manifest.json')}",
        flush=True,
    )


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
    required_name = str(training["required_device_name_substring"])
    observed_name = torch.cuda.get_device_name(local_rank)
    if required_name not in observed_name:
        raise RuntimeError(
            f"V2 requires CUDA device containing {required_name!r}; "
            f"observed {observed_name!r}"
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
    required_name = str(train["training"]["required_device_name_substring"])
    if world_size != 1 or required_devices != 1:
        raise ValueError("formal V2 runtime identity requires one P100 process/device")
    if not torch.cuda.is_available() or torch.cuda.device_count() < required_devices:
        raise RuntimeError("formal V2 runtime identity requires one visible CUDA device")
    runtime_contract_sha256 = os.environ.get(
        "TRAUMA_PREDICT_RUNTIME_LOCK_SHA256", ""
    )
    if runtime_contract_sha256 != EXPECTED_P100_RUNTIME_CONTRACT_SHA256:
        raise RuntimeError("formal V2 runtime lacks the frozen P100 cu126 lock identity")
    if str(torch.__version__) != EXPECTED_P100_TORCH_VERSION:
        raise RuntimeError(f"formal V2 runtime has unexpected torch: {torch.__version__}")
    if str(torch.version.cuda or "") != EXPECTED_P100_CUDA_RUNTIME:
        raise RuntimeError(
            f"formal V2 runtime has unexpected CUDA runtime: {torch.version.cuda}"
        )
    if "sm_60" not in set(torch.cuda.get_arch_list()):
        raise RuntimeError("formal V2 PyTorch runtime does not contain sm_60 kernels")
    devices: list[dict[str, Any]] = []
    diagnostic_devices: list[dict[str, Any]] = []
    for index in range(required_devices):
        properties = torch.cuda.get_device_properties(index)
        if required_name not in str(properties.name):
            raise RuntimeError(
                f"formal V2 runtime requires device containing {required_name!r}; "
                f"observed {str(properties.name)!r}"
            )
        semantic_device = {
            "name": str(properties.name),
            "compute_capability": [
                int(properties.major),
                int(properties.minor),
            ],
        }
        if semantic_device["compute_capability"] != [6, 0]:
            raise RuntimeError("formal V2 runtime requires Pascal compute capability 6.0")
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
        "p100_runtime_contract_sha256": runtime_contract_sha256,
        "cuda_arch_list": list(torch.cuda.get_arch_list()),
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
        "trajectory_metric_contract": resolve_repo_path(
            str(train["trajectory_metric_contract"]), repo_root
        ),
        "train_config": train_path,
        "dataset_config": dataset_path,
        "model_config": model_path,
        "relation_field_registry": (
            runtime.relation_contract.config_dir / "field_category_matrix_v1.csv"
        ),
        "relation_target_target": (
            runtime.relation_contract.config_dir / "target_target_relation_edges_v2.csv"
        ),
        "relation_input_target": (
            runtime.relation_contract.config_dir / "input_target_relation_edges_v2.csv"
        ),
        "relation_evidence_registry": (
            runtime.relation_contract.config_dir / "relation_evidence_registry_v2.json"
        ),
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
    entries["trajectory_metric_contract"]["semantic_sha256"] = str(
        runtime.identity["trajectory_metric_contract_sha256"]
    )
    relation_entry_files = {
        "relation_field_registry": "field_category_matrix_v1.csv",
        "relation_target_target": "target_target_relation_edges_v2.csv",
        "relation_input_target": "input_target_relation_edges_v2.csv",
        "relation_evidence_registry": "relation_evidence_registry_v2.json",
    }
    for entry_name, filename in relation_entry_files.items():
        expected = runtime.relation_contract.file_hashes[filename]
        if entries[entry_name]["file_sha256"] != expected:
            raise ValueError(f"copied relation V2 artifact differs for {filename}")
        entries[entry_name]["semantic_sha256"] = expected
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
        "trajectory_metric_contract": str(
            identity["trajectory_metric_contract_sha256"]
        ),
    }
    for name, expected in semantic_checks.items():
        if entries[name].get("semantic_sha256") != expected:
            raise ValueError(f"V2 portable artifact semantic identity mismatch for {name}")
    portable_relation_contract = MultiresEventV2RelationContract.from_config_dir(
        output_dir / "artifacts/relation_contract"
    )
    if (
        portable_relation_contract.bundle_hash != runtime.relation_contract.bundle_hash
        or dict(portable_relation_contract.file_hashes)
        != dict(runtime.relation_contract.file_hashes)
    ):
        raise ValueError("V2 portable relation bundle differs from the runtime contract")
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
            "trajectory_metric_contract": RUN_ARTIFACT_PATHS[
                "trajectory_metric_contract"
            ],
            "trajectory_metric_contract_file_sha256": entries[
                "trajectory_metric_contract"
            ]["file_sha256"],
            "runtime_environment_artifact": RUN_ARTIFACT_PATHS[
                "runtime_environment"
            ],
            "runtime_environment_artifact_file_sha256": entries[
                "runtime_environment"
            ]["file_sha256"],
            "semantic_runtime_identity_sha256": semantic_runtime_sha,
            "relation_contract_artifact_dir": "artifacts/relation_contract",
            "relation_contract_bundle_sha256": portable_relation_contract.bundle_hash,
            "relation_contract_file_sha256": dict(portable_relation_contract.file_hashes),
        }
    )
    return replace(
        runtime,
        identity=identity,
        relation_contract=portable_relation_contract,
    )


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
    world_size: int,
) -> dict[str, Any]:
    """Exercise the production best-checkpoint save/load order before data loading."""

    checkpoint_canary_identity = {
        "canary": sha256_payload(
            {
                "schema": "multires_event_v2_best_checkpoint_collective_canary_v1",
                "model_contract": "relation_v2",
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
        "relation_contract_version": runtime.relation_contract.version,
        "relation_contract_bundle_sha256": runtime.relation_contract.bundle_hash,
        "relation_contract_file_sha256": dict(runtime.relation_contract.file_hashes),
    })
    atomic_write_json(output_dir / "model_identity.json", {
        "model_contract": "relation_v2",
        "parameter_count": int(parameter_count),
        "run_contract_signature": identity_hashes["run_contract"],
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
        "run_contract_signature",
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
        "model_contract": "relation_v2",
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
        "model_contract": "relation_v2",
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
    """Hash every executable byte and require a clean Git or release identity."""

    candidates = list((repo_root / "src/trauma_predict").rglob("*.py"))
    candidates.extend(
        repo_root / relative
        for relative in (
            "configs/runtime/p100_torch_2_10_cu126_cp312.json",
            "notebooks/kaggle/train_relation_v2_p100.py",
            "notebooks/kaggle/run_relation_v2_p100_bundle.py",
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
    if status is not None and status != "":
        raise RuntimeError(
            "formal relation V2 training requires a clean committed source tree"
        )
    if status is None and release_identity is None:
        raise RuntimeError(
            "formal relation V2 training outside Git requires a valid SOURCE_RELEASE.json"
        )
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


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    observed = {str(key) for key in value}
    if observed != expected:
        raise ValueError(
            f"{label} keys differ from strict relation V2: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


__all__ = [
    "AUTHORIZED_TRAINING_RUN_NAMES",
    "EXPECTED_OPTIMIZER_CONTRACT",
    "EXPECTED_FORMAL_MODEL_PARAMETER_COUNT",
    "MultiresEventV2Runtime",
    "OPTIMIZER_CONTRACT_VERSION",
    "OPTIMIZER_HEALTH_SUMMARY_SCHEMA",
    "RAW_JOINT_NLL_REDUCTION",
    "ROUTE",
    "TRAINING_AUTHORIZED",
    "TRAINING_AUTHORIZATION_REASON",
    "V2_DISTRIBUTED_TIMEOUT_SECONDS",
    "V2_EARLY_CANARY_PROCESS_GROUP_TIMEOUT_SECONDS",
    "V2_NCCL_MONITOR_HEARTBEAT_TIMEOUT_SECONDS",
    "build_multires_event_v2_model",
    "build_multires_event_v2_optimizer",
    "build_multires_event_v2_runtime",
    "load_multires_event_v2_configs",
    "load_lab_scale_artifact",
    "run_contract_signature",
    "raw_414_factor_joint_nll_batch_mean",
    "require_multires_event_v2_training_authorization",
    "run_multires_event_v2_rank_artifact_preflight_only",
    "run_multires_event_v2_training",
    "validate_formal_model_parameter_count",
    "validate_formal_target_field_order",
    "validate_multires_event_v2_configs",
    "validate_optimizer_health_summary",
    "summarize_optimizer_health_metrics",
]
