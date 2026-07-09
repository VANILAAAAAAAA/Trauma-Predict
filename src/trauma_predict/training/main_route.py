from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trauma_predict.data.main_route import (
    HourValueNormalizer,
    MainRouteBatchCollator,
    MainRouteRecordDataset,
    decode_next24_predictions,
    load_main_route_records,
)
from trauma_predict.data.main_route_contract import (
    HOUR_SPECIAL_TOKENS,
    HOUR_VALUE_ORDER,
    MAIN_ROUTE,
    STATE_TOKEN,
)
from trauma_predict.data.preflight import ArtifactPreflightResult
from trauma_predict.data.records import resolve_shard_paths
from trauma_predict.training.runtime import (
    append_jsonl,
    latest_checkpoint,
    maybe_cap_records,
    quarantine_rng_state_files,
    sanitize_json,
    utc_now,
    write_environment_snapshot,
)
from trauma_predict.training.stages import (
    TrainingStageContract,
    is_next24_active,
    is_next_hour_active,
    labels_for_active_losses,
    resolve_training_stage_contract,
)


@dataclass(frozen=True)
class MainRouteRunResult:
    output_dir: str
    train_samples: int
    eval_samples: int
    checkpoint: str | None
    resume_checkpoint: str | None
    final_model: str
    final_model_stage_metadata: str
    metrics_path: str
    prediction_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "train_samples": self.train_samples,
            "eval_samples": self.eval_samples,
            "checkpoint": self.checkpoint,
            "resume_checkpoint": self.resume_checkpoint,
            "final_model": self.final_model,
            "final_model_stage_metadata": self.final_model_stage_metadata,
            "metrics_path": self.metrics_path,
            "prediction_path": self.prediction_path,
        }


def run_main_route_training(
    train_config: dict[str, Any],
    dataset_config: dict[str, Any],
    output_dir: Path,
    preflight: ArtifactPreflightResult,
) -> MainRouteRunResult:
    validate_main_route_config(train_config)
    stage_contract = resolve_training_stage_contract(train_config, require_implemented=True)

    import torch
    from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

    from trauma_predict.modeling.main_route import MainRouteModel

    model_config = train_config["model"]
    training_config = train_config["training"]
    eval_config = train_config.get("evaluation", {})
    output_config = train_config.get("outputs", {})
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(output_config.get("metrics_jsonl", output_dir / "metrics.jsonl"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "validation_predictions.jsonl"

    set_seed(int(train_config.get("seed", 0)))
    required_cuda_devices = int(training_config.get("required_cuda_devices", 0) or 0)
    if required_cuda_devices and torch.cuda.device_count() < required_cuda_devices:
        raise RuntimeError(
            f"training requires {required_cuda_devices} CUDA devices, "
            f"but torch sees {torch.cuda.device_count()}"
        )
    base_model = str(model_config["base_model"])
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens({"additional_special_tokens": [*HOUR_SPECIAL_TOKENS, STATE_TOKEN]})
    tokenizer.padding_side = "right"

    normalizer = HourValueNormalizer.from_config(model_config.get("value_normalization"))
    next_hour_value_mode = str(model_config.get("next_hour_value_mode") or "absolute")
    next_hour_delta_loss_weight = float(training_config.get("next_hour_delta_loss_weight", 0.0) or 0.0)
    model = MainRouteModel(
        base_model=base_model,
        tokenizer_length=len(tokenizer),
        adapter_hidden_size=int(model_config.get("hour_adapter_hidden", 256)),
        hour_field_hidden_size=int(model_config.get("hour_field_hidden", 64)),
        dropout=float(model_config.get("dropout", 0.1)),
        loss_weights=stage_contract.loss_weights,
        active_losses=stage_contract.active_losses,
        next_hour_value_mode=next_hour_value_mode,
        next_hour_delta_loss_weight=next_hour_delta_loss_weight,
    )
    warm_start_report: dict[str, Any] | None = None
    warm_start_checkpoint = training_config.get("warm_start_checkpoint")
    if warm_start_checkpoint:
        warm_start_report = load_warm_start_checkpoint(
            model=model,
            checkpoint=Path(str(warm_start_checkpoint)),
            reset_prefixes=[
                str(item)
                for item in training_config.get("warm_start_reset_heads", [])
            ],
        )
    max_position_embeddings = int(getattr(model.encoder.config, "max_position_embeddings", 0) or 0)
    max_input_tokens = int(model_config.get("max_input_tokens", 4096))
    if max_position_embeddings and max_input_tokens > max_position_embeddings:
        raise ValueError(
            f"model.max_input_tokens={max_input_tokens} exceeds encoder max_position_embeddings="
            f"{max_position_embeddings}"
        )
    if bool(training_config.get("gradient_checkpointing", False)):
        model.enable_gradient_checkpointing()

    required_fields = list(dataset_config.get("required_sample_fields") or [])
    train_records = load_main_route_records(resolve_shard_paths(dataset_config, "train"), required_fields, split="train")
    eval_records = load_main_route_records(resolve_shard_paths(dataset_config, "val"), required_fields, split="val")
    train_records = maybe_cap_records(train_records, training_config.get("max_train_samples"))
    eval_records = maybe_cap_records(eval_records, training_config.get("max_eval_samples"))

    train_dataset = MainRouteRecordDataset(train_records)
    eval_dataset = MainRouteRecordDataset(eval_records)

    precision = str(training_config.get("precision", "fp16")).lower()
    fp16 = precision == "fp16"
    bf16 = precision == "bf16"
    collator = MainRouteBatchCollator(
        tokenizer=tokenizer,
        max_input_tokens=max_input_tokens,
        normalizer=normalizer,
        pad_to_multiple_of=8 if fp16 or bf16 else None,
        active_losses=stage_contract.active_losses,
    )

    training_args_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "run_name": str(train_config.get("run_name", "trauma_predict_main_route")),
        "seed": int(train_config.get("seed", 0)),
        "per_device_train_batch_size": int(training_config.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(training_config.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(training_config.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(training_config.get("learning_rate", 2e-5)),
        "weight_decay": float(training_config.get("weight_decay", 0.01)),
        "max_grad_norm": float(training_config.get("max_grad_norm", 1.0)),
        "warmup_steps": int(training_config.get("warmup_steps", 0)),
        "max_steps": int(training_config.get("max_steps", 1000)),
        "eval_steps": int(training_config.get("eval_steps", 250)),
        "save_steps": int(training_config.get("save_steps", 500)),
        "logging_steps": int(training_config.get("logging_steps", 25)),
        "save_total_limit": int(training_config.get("max_checkpoints", 3)),
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "fp16": fp16,
        "bf16": bf16,
        "dataloader_num_workers": int(training_config.get("dataloader_num_workers", 2)),
        "report_to": [],
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": bool(training_config.get("ddp_find_unused_parameters", True)),
        "prediction_loss_only": True,
        "label_names": labels_for_active_losses(stage_contract.active_losses),
    }
    argument_names = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in argument_names:
        training_args_kwargs["eval_strategy"] = "steps"
    else:
        training_args_kwargs["evaluation_strategy"] = "steps"
    if "disable_tqdm" in argument_names:
        training_args_kwargs["disable_tqdm"] = bool(training_config.get("disable_tqdm", True))
    if "log_level" in argument_names:
        training_args_kwargs["log_level"] = str(training_config.get("log_level", "warning"))
    if "log_level_replica" in argument_names:
        training_args_kwargs["log_level_replica"] = str(training_config.get("log_level_replica", "error"))
    args = TrainingArguments(**training_args_kwargs)

    def stage_metadata() -> dict[str, Any]:
        return stage_contract.to_metadata()

    class MainRouteTrainer(Trainer):
        latest_loss_parts: dict[str, float]

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[no-untyped-def]
            outputs = model(**inputs)
            loss = outputs["loss"]
            loss_parts = outputs.get("loss_parts") or {}
            self.latest_loss_parts = {
                f"train_{name}_loss": float(value.detach().cpu().item())
                for name, value in loss_parts.items()
            }
            return (loss, outputs) if return_outputs else loss

    trainer: MainRouteTrainer

    class MetricsJsonlCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
            if not logs or int(getattr(args, "process_index", 0)) != 0:
                return
            logs.update(getattr(trainer, "latest_loss_parts", {}))
            append_jsonl(metrics_path, {
                "created_at": utc_now(),
                "event": "trainer_log",
                "step": int(state.global_step),
                "logs": sanitize_json(logs),
            })

    class CheckpointMetadataCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            if int(getattr(args, "process_index", 0)) != 0:
                return
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "created_at": utc_now(),
                "route": MAIN_ROUTE,
                **stage_metadata(),
            }
            (checkpoint_dir / "training_stage_metadata.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    trainer = MainRouteTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[MetricsJsonlCallback(), CheckpointMetadataCallback()],
    )

    checkpoint = latest_checkpoint(output_dir) if bool(training_config.get("resume", True)) else None
    validate_resume_checkpoint_stage(checkpoint, stage_contract)
    resume_checkpoint = checkpoint
    quarantined_rng_files = quarantine_rng_state_files(checkpoint)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if trainer.is_world_process_zero():
        write_environment_snapshot(output_dir / "environment_snapshot.json", preflight)
        if warm_start_report is not None:
            (output_dir / "warm_start_report.json").write_text(
                json.dumps(sanitize_json(warm_start_report), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        append_jsonl(metrics_path, {
            "created_at": utc_now(),
            "event": "training_start",
            "route": MAIN_ROUTE,
            **stage_metadata(),
            "base_model": base_model,
            "train_samples": len(train_records),
            "eval_samples": len(eval_records),
            "preflight": preflight.to_dict(),
            "resume_checkpoint": resume_checkpoint,
            "warm_start_report": warm_start_report,
            "quarantined_rng_state_files": quarantined_rng_files,
            "torch": torch.__version__,
        })

    trainer.train(resume_from_checkpoint=checkpoint)
    eval_metrics = trainer.evaluate()
    produced_checkpoint = latest_checkpoint(output_dir)
    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    final_model_stage_metadata = final_model_dir / "training_stage_metadata.json"

    prediction_output: str | None = None
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(final_model_dir))
        model.save_main_route(final_model_dir)
        prediction_records = select_prediction_records(
            eval_records,
            eval_config.get("max_prediction_samples", 32),
        )
        append_jsonl(metrics_path, {
            "created_at": utc_now(),
            "event": "final_eval",
            "metrics": sanitize_json(eval_metrics),
            "prediction_samples": len(prediction_records),
        })
        write_validation_predictions(
            prediction_path,
            model,
            collator,
            normalizer,
            prediction_records,
            stage_contract.active_losses,
        )
        final_model_stage_metadata.write_text(
            json.dumps({
                "created_at": utc_now(),
                "route": MAIN_ROUTE,
                **stage_metadata(),
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        prediction_output = str(prediction_path)

    return MainRouteRunResult(
        output_dir=str(output_dir),
        train_samples=len(train_records),
        eval_samples=len(eval_records),
        checkpoint=produced_checkpoint,
        resume_checkpoint=resume_checkpoint,
        final_model=str(final_model_dir),
        final_model_stage_metadata=str(final_model_stage_metadata),
        metrics_path=str(metrics_path),
        prediction_path=prediction_output,
    )


def validate_main_route_config(config: dict[str, Any]) -> None:
    resolve_training_stage_contract(config, require_implemented=True)
    if config.get("schema_version") != "trauma_predict.train_config.v1":
        raise ValueError("train config schema_version mismatch")
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("train config model must be an object")
    if model.get("task") != MAIN_ROUTE:
        raise ValueError(f"model.task must be {MAIN_ROUTE}")
    if not str(model.get("base_model") or ""):
        raise ValueError("model.base_model is required")
    if int(model.get("max_input_tokens", 1)) < 512:
        raise ValueError("model.max_input_tokens must be >= 512")
    if int(model.get("hour_adapter_hidden", 1)) < 32:
        raise ValueError("model.hour_adapter_hidden must be >= 32")
    if int(model.get("hour_field_hidden", 64)) < 8:
        raise ValueError("model.hour_field_hidden must be >= 8")
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("train config training must be an object")
    precision = str(training.get("precision", "fp16")).lower()
    if precision not in {"fp32", "no", "none", "fp16", "bf16"}:
        raise ValueError("training.precision must be one of fp32/no/none/fp16/bf16")
    if float(training.get("learning_rate", 0.0)) <= 0:
        raise ValueError("training.learning_rate must be positive")
    if int(training.get("max_steps", 0)) < 1:
        raise ValueError("training.max_steps must be positive")
    value_mode = str(model.get("next_hour_value_mode") or "absolute")
    if value_mode not in {"absolute", "h0_residual"}:
        raise ValueError("model.next_hour_value_mode must be absolute or h0_residual")
    if float(training.get("next_hour_delta_loss_weight", 0.0) or 0.0) < 0.0:
        raise ValueError("training.next_hour_delta_loss_weight must be >= 0")


def load_warm_start_checkpoint(
    *,
    model: Any,
    checkpoint: Path,
    reset_prefixes: list[str] | None = None,
) -> dict[str, Any]:
    import torch

    weight_path = _resolve_warm_start_weight_path(checkpoint)
    reset_prefix_tuple = tuple(prefix.rstrip(".") for prefix in (reset_prefixes or []) if prefix)
    if weight_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        source_state = load_file(str(weight_path), device="cpu")
    else:
        source_state = torch.load(weight_path, map_location="cpu")
    if not isinstance(source_state, dict):
        raise ValueError(f"warm-start checkpoint did not contain a state dict: {weight_path}")
    if "state_dict" in source_state and isinstance(source_state["state_dict"], dict):
        source_state = source_state["state_dict"]

    target_state = model.state_dict()
    compatible_state = {}
    skipped_reset_keys: list[str] = []
    skipped_shape_keys: list[str] = []
    skipped_non_tensor_keys: list[str] = []
    for key, value in source_state.items():
        if reset_prefix_tuple and any(key == prefix or key.startswith(f"{prefix}.") for prefix in reset_prefix_tuple):
            skipped_reset_keys.append(str(key))
            continue
        if not hasattr(value, "shape"):
            skipped_non_tensor_keys.append(str(key))
            continue
        if key in target_state and tuple(target_state[key].shape) != tuple(value.shape):
            skipped_shape_keys.append(str(key))
            continue
        compatible_state[str(key)] = value

    incompatible = model.load_state_dict(compatible_state, strict=False)
    metadata = _read_warm_start_metadata(checkpoint)
    return {
        "checkpoint": str(checkpoint),
        "weight_path": str(weight_path),
        "source_key_count": len(source_state),
        "loaded_key_count": len(compatible_state) - len(incompatible.unexpected_keys),
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_key_count": len(incompatible.unexpected_keys),
        "reset_prefixes": list(reset_prefix_tuple),
        "skipped_reset_keys": sorted(skipped_reset_keys),
        "skipped_shape_keys": sorted(skipped_shape_keys),
        "skipped_non_tensor_keys": sorted(skipped_non_tensor_keys),
        "missing_keys": sorted(str(item) for item in incompatible.missing_keys),
        "unexpected_keys": sorted(str(item) for item in incompatible.unexpected_keys),
        "source_metadata": metadata,
    }


def _resolve_warm_start_weight_path(checkpoint: Path) -> Path:
    if checkpoint.is_file():
        return checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"warm-start checkpoint does not exist: {checkpoint}")
    for name in ("model.safetensors", "pytorch_model.bin", "main_route_model.pt"):
        candidate = checkpoint / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"warm-start checkpoint has no supported weight file: {checkpoint}")


def _read_warm_start_metadata(checkpoint: Path) -> dict[str, Any] | None:
    metadata_path = checkpoint / "training_stage_metadata.json" if checkpoint.is_dir() else checkpoint.parent / "training_stage_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"metadata_error": f"invalid JSON: {metadata_path}"}


def validate_resume_checkpoint_stage(
    checkpoint: str | None,
    stage_contract: TrainingStageContract,
) -> None:
    if not checkpoint:
        return
    metadata_path = Path(checkpoint) / "training_stage_metadata.json"
    if not metadata_path.exists():
        raise ValueError(
            f"resume checkpoint lacks training_stage_metadata.json: {metadata_path}"
        )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("route") != MAIN_ROUTE:
        raise ValueError(f"resume checkpoint route mismatch: {payload.get('route')}")
    expected = stage_contract.to_metadata()
    keys = ["training_stage", "active_losses", "active_loss_names"]
    if expected.get("training_stage") == "stage_a1_residual":
        keys.extend(["next_hour_value_mode", "next_hour_delta_loss_weight"])
    for key in keys:
        if payload.get(key) != expected[key]:
            raise ValueError(
                f"resume checkpoint {key} mismatch: expected {expected[key]}, got {payload.get(key)}"
            )
    observed_weights = {
        key: float(value)
        for key, value in dict(payload.get("loss_weights") or {}).items()
    }
    expected_weights = {
        key: float(value)
        for key, value in expected["loss_weights"].items()
    }
    if observed_weights != expected_weights:
        raise ValueError(
            f"resume checkpoint loss_weights mismatch: expected {expected_weights}, got {observed_weights}"
        )


def write_validation_predictions(
    path: Path,
    model: Any,
    collator: MainRouteBatchCollator,
    normalizer: HourValueNormalizer,
    records: list[dict[str, Any]],
    active_losses: dict[str, bool],
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            batch = collator([record])
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                output = model(**batch)
            prediction_payload: dict[str, Any] = {}
            target_payload: dict[str, Any] = {}
            if is_next_hour_active(active_losses):
                next_hour_values = output["next_hour_value_logits"][0].detach().cpu().tolist()
                prediction_payload["next_hour"] = {
                    "label": "NEXT_HOUR",
                    "relative_hour": "H+1",
                    "value_order": list(HOUR_VALUE_ORDER),
                    "values": normalizer.denormalize_row(next_hour_values),
                }
                if active_losses.get("next_hour_vent", False):
                    next_hour_vent_probability = float(torch.sigmoid(output["next_hour_vent_logits"][0]).detach().cpu().item())
                    prediction_payload["next_hour"]["vent_probability"] = next_hour_vent_probability
                target_payload["next_hour"] = _next_hour_target_for_active_losses(
                    record["targets"]["next_hour"],
                    active_losses,
                )
            if is_next24_active(active_losses):
                domain_scores = torch.sigmoid(output["next24_domain_logits"][0]).detach().cpu().tolist()
                binary_scores = torch.sigmoid(output["next24_binary_logits"][0]).detach().cpu().tolist()
                multiclass_indices = [
                    int(logits[0].detach().cpu().argmax().item())
                    for logits in output["next24_multiclass_logits"]
                ]
                prediction_payload["next24h"] = decode_next24_predictions(
                    domain_scores=domain_scores,
                    binary_scores=binary_scores,
                    multiclass_indices=multiclass_indices,
                )
                target_payload["next24h"] = record["targets"]["next24h"]
            prediction = {
                "sample_id": record["sample_id"],
                "hadm_id": record["hadm_id"],
                "stay_id": record["stay_id"],
                "prediction_hour": record["prediction_hour"],
                "input": {
                    "hour": _hour_input_context(record),
                },
                "prediction": prediction_payload,
                "target": target_payload,
            }
            handle.write(json.dumps(prediction, sort_keys=True) + "\n")


def select_prediction_records(
    records: list[dict[str, Any]],
    max_prediction_samples: Any,
) -> list[dict[str, Any]]:
    if max_prediction_samples is None:
        return records
    if isinstance(max_prediction_samples, str) and max_prediction_samples.lower() in {"all", "full"}:
        return records
    limit = int(max_prediction_samples)
    if limit < 1:
        raise ValueError("evaluation.max_prediction_samples must be positive, null, all, or full")
    return records[:limit]


def _hour_input_context(record: dict[str, Any]) -> dict[str, Any]:
    hour_values = record["hour_values"]
    hour_mask = record["hour_mask"]
    hour_vent = record["hour_vent"]
    hour_placeholders = record["hour_placeholders"]
    return {
        "value_order": record["hour_value_order"],
        "hour_placeholders": hour_placeholders,
        "hour_values": hour_values,
        "hour_mask": hour_mask,
        "hour_vent": hour_vent,
        "h0": {
            "placeholder": hour_placeholders[-1],
            "hour_values": hour_values[-1],
            "hour_mask": hour_mask[-1],
            "hour_vent": hour_vent[-1],
        },
    }


def _next_hour_target_for_active_losses(
    target: dict[str, Any],
    active_losses: dict[str, bool],
) -> dict[str, Any]:
    payload = {
        "label": target["label"],
        "relative_hour": target["relative_hour"],
        "value_order": target["value_order"],
        "values": target["values"],
        "mask": target["mask"],
        "hour_values": target["hour_values"],
        "hour_mask": target["hour_mask"],
    }
    if active_losses.get("next_hour_vent", False):
        payload["vent_on"] = target["vent_on"]
        payload["hour_vent"] = target["hour_vent"]
    return payload
