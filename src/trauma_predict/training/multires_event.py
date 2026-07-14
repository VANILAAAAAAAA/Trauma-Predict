from __future__ import annotations

import copy
import csv
import dataclasses
import gzip
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping

from trauma_predict.training.config import load_yaml_config
from trauma_predict.training.observability import (
    LossAccumulator,
    append_jsonl,
    atomic_write_json,
    emit_loss,
    is_rank_zero,
    sha256_file,
    sha256_payload,
    utc_now,
)


ROUTE = "multires_event_v1_baseline"
EXPECTED_DATASET_ID = "multires_event_v1_c4_full_20260712"
EXPECTED_DATASET_FINGERPRINT = "d58d003b6a9b2dd7c1f8d269a1867b534ea475a91118d7d4d44804bee69f9e47"
EXPECTED_SOURCE_FINGERPRINT = "ed578cf6b6e82c96f3aef71d58d6c176c794c9e8fbd37a468a709d64e94739b9"
EXPECTED_COUNTS = {"samples": 50350, "train": 37734, "val": 6309, "test": 6307, "shards": 52}
EXPECTED_TARGET_COUNTS = {
    "canonical_rows": 1314,
    "primary_direct_queries": 986,
    "h1_queries": 92,
    "m4_blocks": 6,
    "m4_queries_per_block": 149,
    "f24_evaluation_queries": 149,
    "auxiliary_direct_queries": 105,
}

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
    "query_field_ids",
    "query_operator_ids",
    "query_condition_ids",
    "query_resolution_ids",
    "query_time_index",
    "query_span",
    "query_mask",
)


def resolve_repo_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    if "${" in str(path):
        raise ValueError(f"unexpanded environment variable in path: {path}")
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def validate_multires_event_config(config: Mapping[str, Any], model_config: Mapping[str, Any]) -> None:
    if config.get("route") != ROUTE:
        raise ValueError(f"route must be {ROUTE!r}")
    if model_config.get("route") != ROUTE:
        raise ValueError(f"model route must be {ROUTE!r}")
    if model_config.get("initialization") != "from_scratch":
        raise ValueError("multires_event_v1 baseline must initialize from scratch")
    if model_config.get("text_backbone") is not None or model_config.get("tokenizer") is not None:
        raise ValueError("multires_event_v1 must not use a text backbone or tokenizer")

    data = config.get("data", {})
    if data.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError(f"unexpected dataset_id: {data.get('dataset_id')!r}")
    if data.get("dataset_fingerprint") != EXPECTED_DATASET_FINGERPRINT:
        raise ValueError("dataset fingerprint does not match the frozen C4 artifact")
    if data.get("source_fingerprint") != EXPECTED_SOURCE_FINGERPRINT:
        raise ValueError("source fingerprint does not match the frozen C4 artifact")
    if dict(data.get("expected_counts", {})) != EXPECTED_COUNTS:
        raise ValueError(f"expected_counts must equal {EXPECTED_COUNTS}")

    target = config.get("target", {})
    for key, expected in EXPECTED_TARGET_COUNTS.items():
        if int(target.get(key, -1)) != expected:
            raise ValueError(f"target.{key} must be {expected}")
    if target.get("f24_training_loss") is not False:
        raise ValueError("F24 is derived evaluation only and must not receive training loss")
    if target.get("auxiliary_training_loss") is not False:
        raise ValueError("the baseline must keep auxiliary direct losses disabled")
    if target.get("predict_target_mask") is not False:
        raise ValueError("target_mask is censoring truth, not a prediction head")
    if EXPECTED_TARGET_COUNTS["h1_queries"] + (
        EXPECTED_TARGET_COUNTS["m4_blocks"] * EXPECTED_TARGET_COUNTS["m4_queries_per_block"]
    ) != EXPECTED_TARGET_COUNTS["primary_direct_queries"]:
        raise AssertionError("internal target arithmetic is inconsistent")

    pooling = model_config.get("block_pooling", {})
    if pooling.get("type") != "learned_latent_cross_attention":
        raise ValueError("block pooling must use learned latent cross-attention")
    if int(pooling.get("latent_tokens_per_block", -1)) != 8:
        raise ValueError("block pooling must use eight learned latent tokens")
    if int(model_config.get("encoder", {}).get("trajectory_layers", -1)) != 4:
        raise ValueError("trajectory encoder must use four layers")
    if int(model_config.get("decoder", {}).get("query_layers", -1)) != 3:
        raise ValueError("query decoder must use three layers")

    training = config.get("training", {})
    if int(training.get("required_world_size", -1)) != 2:
        raise ValueError("the hosted baseline contract requires torchrun world size 2")
    if int(training.get("required_cuda_devices", -1)) != 2:
        raise ValueError("the hosted baseline contract requires two visible CUDA devices")
    if training.get("precision") != "fp16":
        raise ValueError("the first hosted baseline is frozen to fp16")
    if int(training.get("per_device_eval_batch_size", -1)) != 1:
        raise ValueError("subject-macro evaluation currently requires eval batch size 1")

    evaluation = config.get("evaluation", {})
    if evaluation.get("interval_anchor_policy") != "one_fixed_anchor_per_subject":
        raise ValueError("interval evaluation must use one fixed anchor per subject")
    if evaluation.get("final_anchor_policy") != "all_validation_anchors":
        raise ValueError("final evaluation must use all validation anchors")
    if evaluation.get("subject_macro") is not True:
        raise ValueError("validation must be subject-macro")
    if evaluation.get("no_ddp_padding_duplicates") is not True:
        raise ValueError("validation DDP sampler must not introduce padding duplicates")


def preflight_dataset_root(config: Mapping[str, Any], dataset_root: Path) -> dict[str, Any]:
    """Validate exact identity and inventory without copying or rewriting the artifact."""
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    data = config["data"]
    manifest_path = dataset_root / str(data.get("dataset_manifest", "dataset_manifest.json"))
    sample_manifest_path = dataset_root / str(data.get("sample_manifest", "sample_manifest.csv"))
    subject_split_path = dataset_root / str(data.get("subject_split", "subject_split.csv"))
    for path in (manifest_path, sample_manifest_path, subject_split_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != data["dataset_id"]:
        raise ValueError("attached dataset_id mismatch")
    if manifest.get("fingerprint") != data["dataset_fingerprint"]:
        raise ValueError("attached dataset fingerprint mismatch")
    if manifest.get("source", {}).get("source_fingerprint") != data["source_fingerprint"]:
        raise ValueError("attached dataset source fingerprint mismatch")
    if manifest.get("status") != "SUCCEEDED":
        raise ValueError(f"dataset status is not SUCCEEDED: {manifest.get('status')!r}")

    counts = manifest.get("counts", {})
    observed_manifest_counts = {
        "samples": int(counts.get("samples", -1)),
        "train": int(counts.get("selected_by_split", {}).get("train", -1)),
        "val": int(counts.get("selected_by_split", {}).get("val", -1)),
        "test": int(counts.get("selected_by_split", {}).get("test", -1)),
        "shards": int(counts.get("completed_shards", -1)),
    }
    if observed_manifest_counts != EXPECTED_COUNTS:
        raise ValueError(
            f"dataset manifest count mismatch: expected {EXPECTED_COUNTS}, got {observed_manifest_counts}"
        )

    csv_counts = defaultdict(int)
    validation_subjects: set[str] = set()
    with sample_manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "subject_id", "prediction_hour", "split"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"sample_manifest lacks columns: {sorted(required - set(reader.fieldnames or []))}")
        seen_sample_ids: set[str] = set()
        for row in reader:
            sample_id = row["sample_id"]
            if sample_id in seen_sample_ids:
                raise ValueError(f"duplicate sample_id in sample_manifest: {sample_id}")
            seen_sample_ids.add(sample_id)
            split = row["split"]
            csv_counts[split] += 1
            if split == "val":
                validation_subjects.add(row["subject_id"])
    observed_csv_counts = {
        "samples": sum(csv_counts.values()),
        "train": csv_counts["train"],
        "val": csv_counts["val"],
        "test": csv_counts["test"],
    }
    if observed_csv_counts != {key: EXPECTED_COUNTS[key] for key in observed_csv_counts}:
        raise ValueError(f"sample_manifest count mismatch: {observed_csv_counts}")

    shard_paths = []
    for split in ("train", "val", "test"):
        matches = sorted(dataset_root.glob(str(data[f"{split}_shards"])))
        if not matches:
            raise FileNotFoundError(f"no {split} shards match {data[f'{split}_shards']!r}")
        shard_paths.extend(matches)
    if len(shard_paths) != EXPECTED_COUNTS["shards"]:
        raise ValueError(f"expected 52 shard files, found {len(shard_paths)}")
    manifest_file_entries = manifest.get("files", {}).get("shards", {})
    if len(manifest_file_entries) != EXPECTED_COUNTS["shards"]:
        raise ValueError("dataset manifest does not enumerate exactly 52 shards")
    missing_manifest_shards = [
        entry.get("sample_path")
        for entry in manifest_file_entries.values()
        if not (dataset_root / str(entry.get("sample_path", ""))).is_file()
    ]
    if missing_manifest_shards:
        raise FileNotFoundError(f"manifest shard files are missing: {missing_manifest_shards[:3]}")
    if len(validation_subjects) != 505:
        raise ValueError(
            f"expected 505 eligible validation subjects with persisted anchors, "
            f"found {len(validation_subjects)}"
        )

    return {
        "schema_version": "trauma_predict.multires_dataset_preflight.v1",
        "checked_at": utc_now(),
        "dataset_root": str(dataset_root),
        "read_only_source": True,
        "dataset_id": manifest["dataset_id"],
        "dataset_fingerprint": manifest["fingerprint"],
        "source_fingerprint": manifest["source"]["source_fingerprint"],
        "counts": observed_manifest_counts,
        "validation_subjects": len(validation_subjects),
        "validation_subject_denominator": "eligible subjects with persisted anchors",
        "manifest_sha256": sha256_file(manifest_path),
        "sample_manifest_sha256": sha256_file(sample_manifest_path),
        "subject_split_sha256": sha256_file(subject_split_path),
    }


def load_and_preflight(
    config_path: Path,
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path, dict[str, Any]]:
    config = load_yaml_config(config_path)
    model_path = resolve_repo_path(config["model"]["config_path"], repo_root)
    model_config = load_yaml_config(model_path)
    validate_multires_event_config(config, model_config)
    dataset_root = resolve_repo_path(config["data"]["dataset_root"], repo_root)
    supervision_path = resolve_repo_path(config["target"]["supervision_path"], repo_root)
    if not supervision_path.is_file():
        raise FileNotFoundError(supervision_path)
    observed_supervision_sha256 = sha256_file(supervision_path)
    if observed_supervision_sha256 != config["target"].get("supervision_file_sha256"):
        raise ValueError(
            "supervision overlay file hash differs from the frozen training target contract"
        )
    supervision_payload = json.loads(supervision_path.read_text(encoding="utf-8"))
    canonical_layout = supervision_payload.get("base_registry", {}).get(
        "canonical_target_layout_sha256"
    )
    if canonical_layout != config["target"].get("canonical_layout_sha256"):
        raise ValueError("supervision canonical target layout hash mismatch")
    preflight = preflight_dataset_root(config, dataset_root)
    return config, model_config, dataset_root, supervision_path, model_path, preflight


def run_multires_event_training(config_path: Path, *, repo_root: Path) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    config, model_config, dataset_root, supervision_path, model_path, preflight = load_and_preflight(
        config_path, repo_root=repo_root
    )
    rank, world_size, local_rank, device = _initialize_distributed(config)
    _seed_everything(int(config["seed"]), rank)

    output_dir = resolve_repo_path(config["outputs"]["output_dir"], repo_root)
    metrics_path = resolve_repo_path(config["outputs"]["metrics_jsonl"], repo_root)
    if is_rank_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
    _barrier()

    normalization_value = config.get("normalization", {}).get("path")
    normalization_path = (
        resolve_repo_path(normalization_value, repo_root)
        if normalization_value
        else output_dir / "normalization_stats.json"
    )

    runtime_config = copy.deepcopy(config)
    runtime_config.setdefault("evaluation", {})["phase"] = "interval"
    build_runtime, build_model, compute_loss = _load_components()
    runtime = build_runtime(
        runtime_config,
        dataset_root,
        supervision_path,
        normalization_path,
        rank,
        world_size,
    )
    _validate_runtime_identity(runtime, config)
    target_contract = runtime.target_contract
    normalization = runtime.normalization
    model = build_model(model_config, target_contract).to(device)

    source_identity = _source_identity(repo_root)
    dataset_identity = _jsonable(runtime.dataset_fingerprint)
    target_payload = _jsonable(target_contract)
    normalization_payload = _jsonable(normalization)
    runtime_environment = _runtime_environment(repo_root, config, world_size)
    identity_hashes = {
        "source": sha256_payload(source_identity),
        "dataset": sha256_payload(dataset_identity),
        "resolved_config": sha256_payload(config),
        "target_contract": sha256_payload(target_payload),
        "normalization": sha256_payload(normalization_payload),
        "model_config": sha256_payload(model_config),
    }
    if is_rank_zero():
        existing_identity_path = output_dir / "identity_hashes.json"
        if existing_identity_path.is_file():
            observed = json.loads(existing_identity_path.read_text(encoding="utf-8"))
            assert_resume_identity(identity_hashes, observed)
        _write_run_identity_artifacts(
            output_dir=output_dir,
            config=config,
            model_config=model_config,
            source_identity=source_identity,
            dataset_identity=dataset_identity,
            target_payload=target_payload,
            normalization_payload=normalization_payload,
            preflight=preflight,
            config_path=config_path,
            model_path=model_path,
            supervision_path=supervision_path,
            identity_hashes=identity_hashes,
            runtime_environment=runtime_environment,
        )
    _barrier()

    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(config["training"].get("ddp_find_unused_parameters", False)),
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = _build_scheduler(optimizer, config["training"])
    scaler = _build_grad_scaler(torch, device, config["training"])
    trainer_state, deferred_rng = _maybe_resume(
        output_dir=output_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        identity_hashes=identity_hashes,
        device=device,
        rank=rank,
        config=config,
        runtime=runtime,
    )

    result = _train_loop(
        model=model,
        runtime=runtime,
        compute_loss=compute_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        output_dir=output_dir,
        metrics_path=metrics_path,
        identity_hashes=identity_hashes,
        trainer_state=trainer_state,
        deferred_rng=deferred_rng,
        device=device,
        rank=rank,
        world_size=world_size,
    )

    _load_best_model(output_dir, model, device)
    final_config = copy.deepcopy(config)
    final_config.setdefault("evaluation", {})["phase"] = "final"
    final_runtime = build_runtime(
        final_config,
        dataset_root,
        supervision_path,
        normalization_path,
        rank,
        world_size,
    )
    _validate_runtime_identity(final_runtime, config)
    if sha256_payload(_jsonable(final_runtime.target_contract)) != identity_hashes["target_contract"]:
        raise RuntimeError("final evaluation target contract differs from training")
    if sha256_payload(_jsonable(final_runtime.normalization)) != identity_hashes["normalization"]:
        raise RuntimeError("final evaluation normalization differs from training")
    final_evaluation = evaluate_model(
        model=model,
        loader=final_runtime.eval_loader,
        compute_loss=compute_loss,
        target_contract=final_runtime.target_contract,
        normalizer=final_runtime.normalization,
        device=device,
        metrics_path=metrics_path,
        step=int(result["global_step"]),
        expected_samples=int(config["evaluation"]["final_expected_samples"]),
        prediction_path=output_dir / "val_predictions.jsonl.gz",
        output_dir=output_dir,
        phase="final",
    )
    if is_rank_zero():
        _export_final_model(output_dir, model, identity_hashes, final_evaluation, result)
        atomic_write_json(output_dir / "evaluation.json", final_evaluation)
        run_manifest = _write_run_manifest(
            output_dir=output_dir,
            config=config,
            identity_hashes=identity_hashes,
            train_result=result,
            final_evaluation=final_evaluation,
        )
        atomic_write_json(output_dir / "SUCCESS", {
            "schema_version": "trauma_predict.multires_run_success.v1",
            "completed_at": utc_now(),
            "run_manifest_sha256": sha256_file(output_dir / "run_manifest.json"),
            "global_step": result["global_step"],
        })
        print(f"MULTIRES_EVENT_TRAINING_COMPLETE step={result['global_step']}", flush=True)
    _barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    return {**result, "final_evaluation": final_evaluation if is_rank_zero() else {}}


def _initialize_distributed(config: Mapping[str, Any]) -> tuple[int, int, int, Any]:
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    required_world_size = int(config["training"]["required_world_size"])
    required_cuda = int(config["training"]["required_cuda_devices"])
    if world_size != required_world_size:
        raise RuntimeError(
            f"launch with torchrun --nproc_per_node={required_world_size}; observed WORLD_SIZE={world_size}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < required_cuda:
        raise RuntimeError(
            f"the hosted baseline requires {required_cuda} visible CUDA devices; "
            f"found {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def _load_components() -> tuple[Any, Any, Any]:
    from trauma_predict.data.multires_event import build_runtime
    from trauma_predict.modeling.multires_event import build_multires_model
    from trauma_predict.training.multires_event_loss import compute_multires_loss

    return build_runtime, build_multires_model, compute_multires_loss


def _validate_runtime_identity(runtime: Any, config: Mapping[str, Any]) -> None:
    observed = _jsonable(runtime.dataset_fingerprint)
    if isinstance(observed, str):
        fingerprint = observed
    elif isinstance(observed, Mapping):
        fingerprint = observed.get("fingerprint") or observed.get("dataset_fingerprint")
    else:
        fingerprint = getattr(runtime.dataset_fingerprint, "fingerprint", None)
    if fingerprint != config["data"]["dataset_fingerprint"]:
        raise ValueError(
            f"runtime dataset fingerprint mismatch: expected {config['data']['dataset_fingerprint']}, "
            f"got {fingerprint}"
        )


def _source_identity(repo_root: Path) -> dict[str, Any]:
    def command(*parts: str) -> str | None:
        result = subprocess.run(parts, cwd=repo_root, text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "schema_version": "trauma_predict.source_identity.v1",
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_branch": command("git", "branch", "--show-current"),
        "required_git_ref": os.environ.get("REQUIRED_GIT_REF"),
        "git_remote": command("git", "config", "--get", "remote.origin.url"),
        "route": ROUTE,
    }


def _runtime_environment(
    repo_root: Path,
    config: Mapping[str, Any],
    world_size: int,
) -> dict[str, Any]:
    import torch

    requirements_path = repo_root / "requirements-multires-kaggle.txt"
    package_names = (
        "numpy",
        "pandas",
        "pyyaml",
        "safetensors",
        "torch",
    )
    package_versions: dict[str, str | None] = {}
    for package in package_names:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append({
            "index": index,
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
        })
    return {
        "schema_version": "trauma_predict.runtime_environment.v1",
        "captured_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "devices": devices,
        "world_size": int(world_size),
        "precision": config["training"]["precision"],
        "packages": package_versions,
        "requirements_path": str(requirements_path),
        "requirements_sha256": (
            sha256_file(requirements_path) if requirements_path.is_file() else None
        ),
    }


def _write_run_identity_artifacts(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    dataset_identity: Any,
    target_payload: Any,
    normalization_payload: Any,
    preflight: Mapping[str, Any],
    config_path: Path,
    model_path: Path,
    supervision_path: Path,
    identity_hashes: Mapping[str, str],
    runtime_environment: Mapping[str, Any],
) -> None:
    atomic_write_json(output_dir / "resolved_config.json", {
        "schema_version": "trauma_predict.resolved_multires_config.v1",
        "resolved_at": utc_now(),
        "config_path": str(config_path.resolve()),
        "config": dict(config),
        "config_sha256": identity_hashes["resolved_config"],
    })
    atomic_write_json(output_dir / "source_identity.json", dict(source_identity))
    atomic_write_json(output_dir / "dataset_fingerprint.json", {
        "runtime": dataset_identity,
        "preflight": dict(preflight),
        "identity_sha256": identity_hashes["dataset"],
    })
    atomic_write_json(output_dir / "target_contract.json", {
        "contract": target_payload,
        "contract_sha256": identity_hashes["target_contract"],
        "supervision_path": str(supervision_path),
        "supervision_file_sha256": sha256_file(supervision_path),
    })
    atomic_write_json(output_dir / "normalization.json", {
        "normalization": normalization_payload,
        "normalization_sha256": identity_hashes["normalization"],
    })
    atomic_write_json(output_dir / "model_identity.json", {
        "model_config_path": str(model_path),
        "model_config_file_sha256": sha256_file(model_path),
        "model_config_sha256": identity_hashes["model_config"],
        "model_config": dict(model_config),
        "initialization": "from_scratch",
        "text_backbone": None,
        "tokenizer": None,
    })
    atomic_write_json(output_dir / "identity_hashes.json", dict(identity_hashes))
    atomic_write_json(output_dir / "runtime_environment.json", dict(runtime_environment))


def _build_scheduler(optimizer: Any, training: Mapping[str, Any]) -> Any:
    import torch

    warmup = int(training["warmup_steps"])
    total = int(training["max_steps"])

    def factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        remaining = max(0, total - step)
        denominator = max(1, total - warmup)
        return float(remaining) / float(denominator)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _build_grad_scaler(torch_module: Any, device: Any, training: Mapping[str, Any]) -> Any:
    enabled = training.get("precision") == "fp16" and device.type == "cuda"
    scaler_kwargs = {
        "enabled": enabled,
        "init_scale": float(training.get("grad_scaler_initial_scale", 65536.0)),
        "growth_factor": float(training.get("grad_scaler_growth_factor", 2.0)),
        "backoff_factor": float(training.get("grad_scaler_backoff_factor", 0.5)),
        "growth_interval": int(training.get("grad_scaler_growth_interval", 2000)),
    }
    try:
        return torch_module.amp.GradScaler("cuda", **scaler_kwargs)
    except (AttributeError, TypeError):
        return torch_module.cuda.amp.GradScaler(**scaler_kwargs)


def _autocast_context(device: Any, training: Mapping[str, Any]) -> Any:
    import torch

    if training.get("precision") != "fp16" or device.type != "cuda":
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    except AttributeError:
        return torch.cuda.amp.autocast(dtype=torch.float16)


def _train_loop(
    *,
    model: Any,
    runtime: Any,
    compute_loss: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    config: Mapping[str, Any],
    output_dir: Path,
    metrics_path: Path,
    identity_hashes: Mapping[str, str],
    trainer_state: dict[str, Any],
    deferred_rng: Mapping[str, Any] | None,
    device: Any,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    import torch

    training = config["training"]
    max_steps = int(training["max_steps"])
    accumulation = int(training["gradient_accumulation_steps"])
    logging_steps = int(training["logging_steps"])
    eval_steps = int(training["eval_steps"])
    save_steps = int(training["save_steps"])
    max_grad_norm = float(training["max_grad_norm"])
    global_step = int(trainer_state.get("global_step", 0))
    epoch = int(trainer_state.get("epoch", 0))
    batches_in_epoch = int(trainer_state.get("batches_in_epoch", 0))
    micro_in_accum = int(trainer_state.get("micro_in_accum", 0))
    best_metric = trainer_state.get("best_metric")
    best_step = trainer_state.get("best_step")
    train_accumulator = LossAccumulator()
    if trainer_state.get("train_accumulator"):
        train_accumulator.load_state(trainer_state["train_accumulator"])

    _set_sampler_epoch(runtime.train_sampler, epoch)
    train_iterator = iter(runtime.train_loader)
    for _ in range(batches_in_epoch):
        try:
            next(train_iterator)
        except StopIteration as exc:
            raise RuntimeError("resume cursor exceeds the deterministic train epoch") from exc
    if deferred_rng is not None:
        _restore_rng_state(deferred_rng)

    optimizer.zero_grad(set_to_none=True)
    model.train()
    while global_step < max_steps:
        try:
            batch = next(train_iterator)
            batches_in_epoch += 1
        except StopIteration:
            epoch += 1
            batches_in_epoch = 0
            _set_sampler_epoch(runtime.train_sampler, epoch)
            train_iterator = iter(runtime.train_loader)
            continue

        batch = _move_to_device(batch, device)
        micro_in_accum += 1
        synchronize_now = micro_in_accum == accumulation
        sync_context = (
            model.no_sync()
            if world_size > 1 and not synchronize_now and hasattr(model, "no_sync")
            else nullcontext()
        )
        with sync_context:
            with _autocast_context(device, training):
                outputs = model(**_model_inputs(batch))
                loss_result = compute_loss(
                    outputs,
                    batch,
                    runtime.target_contract,
                    normalizer=None,
                )
                loss = loss_result["loss"] / accumulation
            scaler.scale(loss).backward()
        numerator, denominator, parts = _loss_aggregates(loss_result)
        train_accumulator.update_aggregates(numerator, denominator, parts)
        if not synchronize_now:
            continue

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        micro_in_accum = 0
        global_step += 1

        if global_step % logging_steps == 0 or global_step == max_steps:
            summary = _distributed_accumulator_summary(train_accumulator)
            emit_loss("TRAIN", global_step, summary, metrics_path)
            train_accumulator.reset()

        evaluation = None
        if global_step % eval_steps == 0 or global_step == max_steps:
            evaluation = evaluate_model(
                model=model,
                loader=runtime.eval_loader,
                compute_loss=compute_loss,
                target_contract=runtime.target_contract,
                normalizer=runtime.normalization,
                device=device,
                metrics_path=metrics_path,
                step=global_step,
                expected_samples=int(config["evaluation"]["interval_expected_subjects"]),
                prediction_path=None,
                output_dir=output_dir,
                phase="interval",
            )
            candidate = float(evaluation["eval_primary_loss_subject_macro"])
            if best_metric is None or candidate < float(best_metric):
                best_metric = candidate
                best_step = global_step
                _save_best_model(
                    output_dir=output_dir,
                    model=model,
                    identity_hashes=identity_hashes,
                    step=global_step,
                    metric=candidate,
                )
            model.train()

        if global_step % save_steps == 0 or global_step == max_steps:
            state = {
                "global_step": global_step,
                "epoch": epoch,
                "batches_in_epoch": batches_in_epoch,
                "micro_in_accum": micro_in_accum,
                "best_metric": best_metric,
                "best_step": best_step,
                "train_accumulator": train_accumulator.state(),
            }
            _save_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                trainer_state=state,
                identity_hashes=identity_hashes,
                runtime=runtime,
                rank=rank,
                keep_last=int(training["keep_last_checkpoints"]),
            )

    return {
        "global_step": global_step,
        "epochs_started": epoch + 1,
        "best_metric": best_metric,
        "best_step": best_step,
        "max_steps": max_steps,
    }


def evaluate_model(
    *,
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    compute_loss: Any,
    target_contract: Any,
    normalizer: Any,
    device: Any,
    metrics_path: Path,
    step: int,
    expected_samples: int,
    prediction_path: Path | None,
    output_dir: Path,
    phase: str,
) -> dict[str, Any]:
    import torch

    if phase not in {"interval", "final"}:
        raise ValueError(f"unknown evaluation phase: {phase}")
    model.eval()
    accumulator = LossAccumulator()
    f24_accumulator = LossAccumulator()
    f24_batches = 0
    f24_status_counts: dict[str, int] = defaultdict(int)
    subject_totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    local_sample_ids: set[str] = set()
    local_rows = 0
    prediction_tmp = output_dir / "prediction-parts" / f"{phase}-rank-{_rank():04d}.jsonl.gz"
    writer = None
    if prediction_path is not None:
        prediction_tmp.parent.mkdir(parents=True, exist_ok=True)
        writer = gzip.open(prediction_tmp, "wt", encoding="utf-8")
    try:
        with torch.no_grad():
            for batch in loader:
                batch = _move_to_device(batch, device)
                outputs = model(**_model_inputs(batch))
                loss_result = compute_loss(
                    outputs,
                    batch,
                    target_contract,
                    normalizer=normalizer,
                )
                numerator, denominator, parts = _loss_aggregates(loss_result)
                accumulator.update_aggregates(numerator, denominator, parts)
                if loss_result.get("f24_parts") is not None:
                    f24_accumulator.update_aggregates(
                        0.0,
                        0.0,
                        _parts_aggregates(loss_result["f24_parts"], label="f24_parts"),
                    )
                    f24_batches += 1
                    f24_status_counts[str(loss_result.get("f24_status", "unknown"))] += 1
                sample_ids = _string_batch(batch.get("sample_id"))
                subject_ids = _string_batch(batch.get("subject_id"))
                if len(sample_ids) != 1 or len(subject_ids) != 1:
                    raise ValueError("subject-macro evaluator requires exactly one sample per eval batch")
                sample_id = sample_ids[0]
                if sample_id in local_sample_ids:
                    raise ValueError(f"duplicate validation sample on rank {_rank()}: {sample_id}")
                local_sample_ids.add(sample_id)
                subject_totals[subject_ids[0]][0] += numerator
                subject_totals[subject_ids[0]][1] += denominator
                local_rows += 1
                if writer is not None:
                    row = _prediction_row(loss_result, batch, sample_id, subject_ids[0])
                    writer.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    finally:
        if writer is not None:
            writer.close()

    summary = _distributed_accumulator_summary(accumulator)
    f24_summary = _distributed_part_summary(f24_accumulator)
    gathered_subjects = _gather_objects(dict(subject_totals))
    merged_subjects: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for rank_subjects in gathered_subjects:
        for subject_id, totals in rank_subjects.items():
            merged_subjects[str(subject_id)][0] += float(totals[0])
            merged_subjects[str(subject_id)][1] += float(totals[1])
    valid_subject_losses = [
        numerator / denominator
        for numerator, denominator in merged_subjects.values()
        if denominator > 0
    ]
    if not valid_subject_losses:
        raise RuntimeError("evaluation produced no subjects with observable direct targets")
    subject_macro = sum(valid_subject_losses) / len(valid_subject_losses)
    gathered_ids = _gather_objects(sorted(local_sample_ids))
    all_sample_ids = [sample_id for ids in gathered_ids for sample_id in ids]
    if len(all_sample_ids) != len(set(all_sample_ids)):
        raise RuntimeError("validation sampler introduced cross-rank padding duplicates")
    if len(all_sample_ids) != expected_samples:
        raise RuntimeError(
            f"{phase} evaluation expected {expected_samples} unique samples, got {len(all_sample_ids)}"
        )
    gathered_f24_batches = sum(int(value) for value in _gather_objects(f24_batches))
    merged_f24_status: dict[str, int] = defaultdict(int)
    for counts in _gather_objects(dict(f24_status_counts)):
        for status, count in counts.items():
            merged_f24_status[str(status)] += int(count)
    if phase == "final" and gathered_f24_batches != len(all_sample_ids):
        raise RuntimeError(
            "final evaluation requires raw-unit F24 diagnostic parts for every validation sample"
        )
    if phase == "final" and "total" not in f24_summary:
        raise RuntimeError("final evaluation produced no observable derived F24 metric")
    summary["subject_macro"] = subject_macro
    emit_loss("EVAL", step, summary, metrics_path)
    result = {
        "schema_version": "trauma_predict.multires_evaluation.v1",
        "phase": phase,
        "evaluated_at": utc_now(),
        "step": int(step),
        "samples": len(all_sample_ids),
        "subjects": len(merged_subjects),
        "eval_primary_loss": float(summary["total"]),
        "eval_primary_loss_subject_macro": float(subject_macro),
        "loss_parts": {
            key: float(value)
            for key, value in summary.items()
            if key not in {"total", "subject_macro"}
        },
        "aggregation": "within_subject_then_subject_macro",
        "f24": {
            "status": "derived_from_m4_predictions_without_truth_mask_or_source_count",
            "evaluation_status_counts": dict(sorted(merged_f24_status.items())),
            "raw_unit_diagnostics": f24_summary,
            "no_cross_field_raw_mae": True,
        },
    }
    if is_rank_zero():
        append_jsonl(metrics_path, {
            "created_at": utc_now(),
            "event": f"{phase}_evaluation",
            **result,
        })
    if prediction_path is not None:
        _barrier()
        if is_rank_zero():
            _merge_prediction_parts(
                output_dir / "prediction-parts",
                prediction_path,
                phase=phase,
                expected_samples=expected_samples,
            )
            result["prediction_path"] = str(prediction_path)
            result["prediction_sha256"] = sha256_file(prediction_path)
        _barrier()
    return result if is_rank_zero() else {
        "eval_primary_loss_subject_macro": float(subject_macro),
        "eval_primary_loss": float(summary["total"]),
        "samples": len(all_sample_ids),
    }


def _prediction_row(
    loss_result: Mapping[str, Any],
    batch: Mapping[str, Any],
    sample_id: str,
    subject_id: str,
) -> dict[str, Any]:
    predictions = loss_result.get("predictions")
    prediction_mask = loss_result.get("prediction_mask")
    if predictions is None or prediction_mask is None:
        raise ValueError("loss result must provide predictions and prediction_mask for final export")
    prediction_values = _first_row_values(predictions)
    mask_values = [bool(value) for value in _first_row_values(prediction_mask)]
    target_values = _first_row_values(batch.get("target_raw_values"))
    target_mask = [bool(value) for value in _first_row_values(batch.get("target_mask"))]
    if len(prediction_values) != EXPECTED_TARGET_COUNTS["primary_direct_queries"]:
        raise ValueError(
            f"expected 986 direct predictions, got {len(prediction_values)} for {sample_id}"
        )
    if len(mask_values) != len(prediction_values):
        raise ValueError("prediction_mask length differs from direct predictions")
    if len(target_values) != len(prediction_values) or len(target_mask) != len(prediction_values):
        raise ValueError("direct target truth must align with all 986 active queries")
    summary = loss_result.get("prediction_summary")
    if not isinstance(summary, Mapping):
        raise ValueError("final export requires typed prediction_summary in raw units")
    required_summary = {
        "conditional_raw_value",
        "expected_raw_value",
        "presence_probability",
    }
    missing_summary = required_summary - set(summary)
    if missing_summary:
        raise ValueError(f"typed prediction_summary lacks {sorted(missing_summary)}")
    conditional = _first_row_values(summary["conditional_raw_value"])
    expected = _first_row_values(summary["expected_raw_value"])
    presence = _first_row_values(summary["presence_probability"])
    if not all(
        len(values) == len(prediction_values)
        for values in (conditional, expected, presence)
    ):
        raise ValueError("typed raw prediction banks must align with all 986 active queries")
    binary = (
        _first_row_values(summary["binary_probability"])
        if summary.get("binary_probability") is not None
        else [None] * len(prediction_values)
    )
    ordinal = _optional_ordinal_rows(summary, len(prediction_values))
    typed_predictions = []
    for index in range(len(prediction_values)):
        item = {
            "conditional_raw_value": _finite_or_none(conditional[index]),
            "expected_raw_value": _finite_or_none(expected[index]),
            "presence_probability": _finite_or_none(presence[index]),
        }
        binary_value = _finite_or_none(binary[index])
        if binary_value is not None:
            item["binary_probability"] = binary_value
        if ordinal[index] is not None:
            item["ordinal_probabilities"] = ordinal[index]
        typed_predictions.append(item)
    prediction_hour = _first_scalar(batch.get("prediction_hour"))
    row = {
        "sample_id": sample_id,
        "subject_id": subject_id,
        "prediction_hour": prediction_hour,
        "active_query_predictions": typed_predictions,
        "query_mask": mask_values,
        "target_raw_values": [_finite_or_none(value) for value in target_values],
        "target_mask": target_mask,
    }
    derived = loss_result.get("derived_f24_prediction_summary")
    if derived is None:
        derived = loss_result.get("derived_f24_predictions")
    if derived is None:
        raise ValueError("final export requires 149 raw-unit F24 predictions composed from M4")
    if isinstance(derived, Mapping):
        row["derived_f24_predictions"] = _jsonable_prediction_mapping(derived)
        derived_length = len(next(iter(row["derived_f24_predictions"].values()), []))
    else:
        derived_values = [_finite_or_none(value) for value in _first_row_values(derived)]
        row["derived_f24_predictions"] = {"expected_raw_value": derived_values}
        derived_length = len(derived_values)
    if derived_length != EXPECTED_TARGET_COUNTS["f24_evaluation_queries"]:
        raise ValueError("derived F24 prediction count must be 149")
    f24_target_values = _first_row_values(batch.get("f24_target_raw_values"))
    f24_target_mask = [bool(value) for value in _first_row_values(batch.get("f24_target_mask"))]
    if (
        len(f24_target_values) != EXPECTED_TARGET_COUNTS["f24_evaluation_queries"]
        or len(f24_target_mask) != EXPECTED_TARGET_COUNTS["f24_evaluation_queries"]
    ):
        raise ValueError("F24 target truth must align with all 149 derived queries")
    row["f24_target_raw_values"] = [_finite_or_none(value) for value in f24_target_values]
    row["f24_target_mask"] = f24_target_mask
    return row


def _merge_prediction_parts(
    parts_dir: Path,
    destination: Path,
    *,
    phase: str,
    expected_samples: int,
) -> None:
    part_paths = sorted(parts_dir.glob(f"{phase}-rank-*.jsonl.gz"))
    if not part_paths:
        raise FileNotFoundError("no rank-local validation prediction parts")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    seen: set[str] = set()
    count = 0
    with gzip.open(temporary, "wt", encoding="utf-8") as output:
        for part_path in part_paths:
            with gzip.open(part_path, "rt", encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    sample_id = str(row["sample_id"])
                    if sample_id in seen:
                        raise RuntimeError(f"duplicate prediction sample_id: {sample_id}")
                    seen.add(sample_id)
                    output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                    count += 1
    if count != expected_samples:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"prediction export expected {expected_samples} rows, got {count}")
    temporary.replace(destination)
    shutil.rmtree(parts_dir, ignore_errors=True)


def _loss_aggregates(
    loss_result: Mapping[str, Any],
) -> tuple[float, float, dict[str, tuple[float, float]]]:
    required = {"loss", "loss_numerator", "loss_denominator", "parts"}
    missing = required - set(loss_result)
    if missing:
        raise ValueError(f"loss result lacks true aggregation fields: {sorted(missing)}")
    numerator = _scalar(loss_result["loss_numerator"])
    denominator = _scalar(loss_result["loss_denominator"])
    parts = _parts_aggregates(loss_result["parts"], label="parts")
    return numerator, denominator, parts


def _parts_aggregates(
    payloads: Mapping[str, Any], *, label: str
) -> dict[str, tuple[float, float]]:
    parts: dict[str, tuple[float, float]] = {}
    for name, payload in payloads.items():
        if not isinstance(payload, Mapping) or "numerator" not in payload or "denominator" not in payload:
            raise ValueError(f"{label} entry {name!r} lacks additive numerator/denominator")
        parts[str(name)] = (_scalar(payload["numerator"]), _scalar(payload["denominator"]))
    return parts


def _distributed_accumulator_summary(accumulator: LossAccumulator) -> dict[str, float]:
    gathered = _gather_objects(accumulator.state())
    merged = LossAccumulator()
    sums: dict[str, float] = defaultdict(float)
    weights: dict[str, float] = defaultdict(float)
    for state in gathered:
        for name, value in state["sums"].items():
            sums[str(name)] += float(value)
        for name, value in state["weights"].items():
            weights[str(name)] += float(value)
    merged.load_state({"sums": sums, "weights": weights})
    summary = merged.summary()
    if "total" not in summary:
        raise RuntimeError("loss interval has no observable targets")
    return summary


def _distributed_part_summary(accumulator: LossAccumulator) -> dict[str, float]:
    gathered = _gather_objects(accumulator.state())
    sums: dict[str, float] = defaultdict(float)
    weights: dict[str, float] = defaultdict(float)
    for state in gathered:
        for name, value in state["sums"].items():
            sums[str(name)] += float(value)
        for name, value in state["weights"].items():
            weights[str(name)] += float(value)
    return {
        name: sums[name] / weights[name]
        for name in sorted(sums)
        if weights[name] > 0
    }


def _gather_objects(value: Any) -> list[Any]:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return [value]
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def _save_checkpoint(
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
    import torch

    step = int(trainer_state["global_step"])
    checkpoint_root = output_dir / "checkpoints"
    checkpoint = checkpoint_root / f"checkpoint-{step:08d}"
    partial = checkpoint_root / f".checkpoint-{step:08d}.partial"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite completed checkpoint {checkpoint}")
    if is_rank_zero():
        if partial.exists():
            abandoned_root = checkpoint_root / "incomplete"
            abandoned_root.mkdir(parents=True, exist_ok=True)
            timestamp = utc_now().replace(":", "-")
            partial.rename(
                abandoned_root / f"{partial.name}-{timestamp}-pid{os.getpid()}"
            )
        partial.mkdir(parents=False, exist_ok=False)
    _barrier()
    torch.save(_capture_rng_state(), partial / f"rng-rank-{rank:04d}.pt")
    sampler_state = runtime.train_sampler.state_dict() if hasattr(runtime.train_sampler, "state_dict") else None
    torch.save(sampler_state, partial / f"sampler-rank-{rank:04d}.pt")
    _barrier()
    if is_rank_zero():
        torch.save(_unwrapped_model(model).state_dict(), partial / "model.pt")
        torch.save(optimizer.state_dict(), partial / "optimizer.pt")
        torch.save(scheduler.state_dict(), partial / "scheduler.pt")
        torch.save(scaler.state_dict(), partial / "scaler.pt")
        atomic_write_json(partial / "trainer_state.json", dict(trainer_state))
        atomic_write_json(partial / "identity_hashes.json", dict(identity_hashes))
        atomic_write_json(partial / "checkpoint_manifest.json", {
            "schema_version": "trauma_predict.multires_checkpoint.v1",
            "created_at": utc_now(),
            "global_step": step,
            "world_size": _world_size(),
            "identity_hashes": dict(identity_hashes),
            "files": [
                "model.pt",
                "optimizer.pt",
                "scheduler.pt",
                "scaler.pt",
                "trainer_state.json",
                "identity_hashes.json",
                *[f"rng-rank-{index:04d}.pt" for index in range(_world_size())],
                *[f"sampler-rank-{index:04d}.pt" for index in range(_world_size())],
            ],
        })
        partial.replace(checkpoint)
    _barrier()
    if is_rank_zero():
        _prune_checkpoints(output_dir / "checkpoints", keep_last)
    _barrier()


def _maybe_resume(
    *,
    output_dir: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    identity_hashes: Mapping[str, str],
    device: Any,
    rank: int,
    config: Mapping[str, Any],
    runtime: Any,
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    resume = bool(config["training"].get("resume", False))
    all_checkpoints = _sorted_checkpoints(output_dir / "checkpoints")
    checkpoints = [path for path in all_checkpoints if _checkpoint_is_complete(path)]
    incomplete = [path for path in all_checkpoints if path not in checkpoints]
    if incomplete and is_rank_zero():
        print(
            "IGNORING_INCOMPLETE_CHECKPOINTS "
            + json.dumps([str(path) for path in incomplete]),
            flush=True,
        )
    if all_checkpoints and not checkpoints:
        raise RuntimeError(
            "checkpoint directory contains no complete recoverable checkpoint: "
            f"{all_checkpoints}"
        )
    if not checkpoints:
        return {
            "global_step": 0,
            "epoch": 0,
            "batches_in_epoch": 0,
            "micro_in_accum": 0,
            "best_metric": None,
            "best_step": None,
        }, None
    if not resume:
        raise RuntimeError(
            f"existing checkpoints found under {output_dir}; set training.resume=true or use a new output root"
        )
    checkpoint = checkpoints[-1]
    observed_hashes = json.loads((checkpoint / "identity_hashes.json").read_text(encoding="utf-8"))
    assert_resume_identity(identity_hashes, observed_hashes)
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    for name in manifest["files"]:
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"incomplete checkpoint {checkpoint}: missing {name}")
    _unwrapped_model(model).load_state_dict(_torch_load(checkpoint / "model.pt", map_location=device, weights_only=True))
    optimizer.load_state_dict(_torch_load(checkpoint / "optimizer.pt", map_location=device, weights_only=True))
    scheduler.load_state_dict(_torch_load(checkpoint / "scheduler.pt", map_location=device, weights_only=True))
    scaler.load_state_dict(_torch_load(checkpoint / "scaler.pt", map_location=device, weights_only=True))
    sampler_state = _torch_load(
        checkpoint / f"sampler-rank-{rank:04d}.pt", map_location="cpu", weights_only=False
    )
    if sampler_state is not None:
        if not hasattr(runtime.train_sampler, "load_state_dict"):
            raise RuntimeError("checkpoint has sampler state but runtime sampler cannot restore it")
        runtime.train_sampler.load_state_dict(sampler_state)
    deferred_rng = _torch_load(
        checkpoint / f"rng-rank-{rank:04d}.pt", map_location="cpu", weights_only=False
    )
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if is_rank_zero():
        print(f"RESUME_CHECKPOINT path={checkpoint} step={trainer_state['global_step']}", flush=True)
        append_jsonl(output_dir / "metrics.jsonl", {
            "created_at": utc_now(),
            "event": "resume_checkpoint",
            "checkpoint": str(checkpoint),
            "global_step": int(trainer_state["global_step"]),
            "identity_hashes": dict(identity_hashes),
        })
    return trainer_state, deferred_rng


def assert_resume_identity(
    expected: Mapping[str, str], observed: Mapping[str, str]
) -> None:
    if dict(expected) == dict(observed):
        return
    keys = sorted(set(expected) | set(observed))
    mismatches = {
        key: {"expected": expected.get(key), "observed": observed.get(key)}
        for key in keys
        if expected.get(key) != observed.get(key)
    }
    raise RuntimeError(f"checkpoint identity mismatch; refusing resume: {mismatches}")


def _save_best_model(
    *,
    output_dir: Path,
    model: Any,
    identity_hashes: Mapping[str, str],
    step: int,
    metric: float,
) -> None:
    import torch

    if is_rank_zero():
        best_dir = output_dir / "best_checkpoint"
        best_dir.mkdir(parents=True, exist_ok=True)
        temporary = best_dir / f".model.pt.tmp-{os.getpid()}"
        torch.save(_unwrapped_model(model).state_dict(), temporary)
        temporary.replace(best_dir / "model.pt")
        atomic_write_json(best_dir / "identity_hashes.json", dict(identity_hashes))
        atomic_write_json(output_dir / "best_checkpoint.json", {
            "schema_version": "trauma_predict.multires_best_checkpoint.v1",
            "updated_at": utc_now(),
            "step": int(step),
            "eval_primary_loss_subject_macro": float(metric),
            "path": str(best_dir),
            "model_sha256": sha256_file(best_dir / "model.pt"),
            "identity_hashes": dict(identity_hashes),
        })
    _barrier()


def _load_best_model(output_dir: Path, model: Any, device: Any) -> None:
    _barrier()
    path = output_dir / "best_checkpoint" / "model.pt"
    if not path.is_file():
        raise FileNotFoundError("training completed without a best validation checkpoint")
    _unwrapped_model(model).load_state_dict(_torch_load(path, map_location=device, weights_only=True))
    _barrier()


def _export_final_model(
    output_dir: Path,
    model: Any,
    identity_hashes: Mapping[str, str],
    final_evaluation: Mapping[str, Any],
    train_result: Mapping[str, Any],
) -> None:
    import torch

    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in _unwrapped_model(model).state_dict().items()}
    model_path = final_dir / "model.safetensors"
    try:
        from safetensors.torch import save_file

        save_file(state, str(model_path), metadata={
            "route": ROUTE,
            "global_step": str(train_result["global_step"]),
        })
    except ImportError:
        model_path = final_dir / "model.pt"
        torch.save(state, model_path)
    atomic_write_json(final_dir / "model_manifest.json", {
        "schema_version": "trauma_predict.multires_final_model.v1",
        "created_at": utc_now(),
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
        "selected_checkpoint_step": train_result["best_step"],
        "eval_primary_loss_subject_macro": final_evaluation["eval_primary_loss_subject_macro"],
        "identity_hashes": dict(identity_hashes),
    })


def _write_run_manifest(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
    identity_hashes: Mapping[str, str],
    train_result: Mapping[str, Any],
    final_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    required = [
        "resolved_config.json",
        "source_identity.json",
        "dataset_fingerprint.json",
        "target_contract.json",
        "normalization.json",
        "model_identity.json",
        "runtime_environment.json",
        "identity_hashes.json",
        "metrics.jsonl",
        "best_checkpoint.json",
        "final_model/model_manifest.json",
        "val_predictions.jsonl.gz",
        "evaluation.json",
    ]
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot complete run; required outputs missing: {missing}")
    artifacts = {
        name: {"path": name, "sha256": sha256_file(output_dir / name)}
        for name in required
    }
    manifest = {
        "schema_version": "trauma_predict.multires_run_manifest.v1",
        "completed_at": utc_now(),
        "status": "SUCCEEDED",
        "route": ROUTE,
        "run_name": config["run_name"],
        "identity_hashes": dict(identity_hashes),
        "training": dict(train_result),
        "evaluation": dict(final_evaluation),
        "evaluation_contract": {
            "interval": "one fixed anchor per validation subject",
            "final": "all validation anchors; within-subject mean then subject macro",
            "checkpoint_metric": config["evaluation"]["checkpoint_metric"],
        },
        "artifacts": artifacts,
    }
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def _capture_rng_state() -> dict[str, Any]:
    import torch

    payload: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    try:
        import numpy as np

        payload["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return payload


def _restore_rng_state(payload: Mapping[str, Any]) -> None:
    import torch

    random.setstate(payload["python"])
    torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available() and payload.get("torch_cuda"):
        torch.cuda.set_rng_state_all(payload["torch_cuda"])
    if "numpy" in payload:
        import numpy as np

        np.random.set_state(payload["numpy"])


def _torch_load(path: Path, *, map_location: Any, weights_only: bool) -> Any:
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _sorted_checkpoints(checkpoint_root: Path) -> list[Path]:
    def step(path: Path) -> int:
        try:
            return int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            return -1

    return sorted(
        [path for path in checkpoint_root.glob("checkpoint-*") if path.is_dir()],
        key=step,
    )


def _checkpoint_is_complete(path: Path) -> bool:
    manifest_path = path / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        return False
    return all(
        isinstance(name, str)
        and name
        and not Path(name).is_absolute()
        and ".." not in Path(name).parts
        and (path / name).is_file()
        for name in files
    )


def _prune_checkpoints(checkpoint_root: Path, keep_last: int) -> None:
    if keep_last < 1:
        raise ValueError("keep_last_checkpoints must be >= 1")
    complete = [
        path for path in _sorted_checkpoints(checkpoint_root)
        if _checkpoint_is_complete(path)
    ]
    for path in complete[:-keep_last]:
        shutil.rmtree(path)


def _seed_everything(seed: int, rank: int) -> None:
    import torch

    rank_seed = seed + rank
    random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)
    try:
        import numpy as np

        np.random.seed(rank_seed)
    except ImportError:
        pass


def _set_sampler_epoch(sampler: Any, epoch: int) -> None:
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(int(epoch))


def _model_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in MODEL_INPUT_KEYS if key not in batch]
    if missing:
        raise KeyError(f"batch lacks model input fields: {missing}")
    return {key: batch[key] for key in MODEL_INPUT_KEYS}


def _move_to_device(value: Any, device: Any) -> Any:
    if hasattr(value, "to") and callable(value.to):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        value = value.item()
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"non-finite loss aggregate: {result}")
    return result


def _first_row_values(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise TypeError(f"prediction payload is not list-like: {type(value).__name__}")
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("prediction writer requires eval batch size 1")
        value = value[0]
    return [_jsonable(item) for item in value]


def _optional_ordinal_rows(summary: Mapping[str, Any], query_count: int) -> list[list[float] | None]:
    probabilities = summary.get("ordinal_probabilities")
    if probabilities is None:
        probabilities = summary.get("ordinal_probability")
    if probabilities is None:
        return [None] * query_count
    values = probabilities.detach().cpu().tolist() if hasattr(probabilities, "detach") else probabilities
    if values and isinstance(values[0], list) and values[0] and isinstance(values[0][0], list):
        if len(values) != 1:
            raise ValueError("ordinal export requires eval batch size 1")
        values = values[0]
    if len(values) != query_count:
        raise ValueError("ordinal probability bank must align with active queries")
    class_mask = summary.get("ordinal_class_mask")
    masks = None
    if class_mask is not None:
        masks = class_mask.detach().cpu().tolist() if hasattr(class_mask, "detach") else class_mask
        if masks and isinstance(masks[0], list) and masks[0] and isinstance(masks[0][0], list):
            masks = masks[0]
    result: list[list[float] | None] = []
    for index, row in enumerate(values):
        selected = []
        for class_index, value in enumerate(row):
            if masks is not None and not bool(masks[index][class_index]):
                continue
            converted = _finite_or_none(value)
            if converted is not None:
                selected.append(converted)
        result.append(selected or None)
    return result


def _jsonable_prediction_mapping(payload: Mapping[str, Any]) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, list):
            raise TypeError(
                f"derived prediction bank {key!r} is not list-like: {type(value).__name__}"
            )
        # Prediction banks carry a leading eval-batch dimension; static class
        # masks are [query, class] and therefore must not be stripped.
        if len(value) == 1 and value and isinstance(value[0], list):
            value = value[0]
        result[str(key)] = _finite_nested(value)
    return result


def _finite_nested(value: Any) -> Any:
    if isinstance(value, list):
        return [_finite_nested(item) for item in value]
    if isinstance(value, bool):
        return value
    return _finite_or_none(value)


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _first_scalar(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("metadata writer requires eval batch size 1")
        return _jsonable(value[0])
    return _jsonable(value)


def _string_batch(value: Any) -> list[str]:
    if value is None:
        raise KeyError("evaluation batch lacks string metadata")
    if isinstance(value, str):
        return [value]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return [str(value)]
    return [str(item) for item in value]


def _unwrapped_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
