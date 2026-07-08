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


@dataclass(frozen=True)
class MainRouteRunResult:
    output_dir: str
    train_samples: int
    eval_samples: int
    checkpoint: str | None
    metrics_path: str
    prediction_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "train_samples": self.train_samples,
            "eval_samples": self.eval_samples,
            "checkpoint": self.checkpoint,
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
    model = MainRouteModel(
        base_model=base_model,
        tokenizer_length=len(tokenizer),
        adapter_hidden_size=int(model_config.get("hour_adapter_hidden", 256)),
        dropout=float(model_config.get("dropout", 0.1)),
        loss_weights=dict(training_config.get("loss_weights") or {}),
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
        "ddp_find_unused_parameters": False,
        "prediction_loss_only": True,
        "label_names": [
            "next_hour_values",
            "next_hour_mask",
            "next_hour_vent",
            "next24_domain_labels",
            "next24_binary_labels",
            "next24_multiclass_labels",
        ],
    }
    argument_names = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in argument_names:
        training_args_kwargs["eval_strategy"] = "steps"
    else:
        training_args_kwargs["evaluation_strategy"] = "steps"
    args = TrainingArguments(**training_args_kwargs)

    class MetricsJsonlCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
            if not logs or int(getattr(args, "process_index", 0)) != 0:
                return
            append_jsonl(metrics_path, {
                "created_at": utc_now(),
                "event": "trainer_log",
                "step": int(state.global_step),
                "logs": sanitize_json(logs),
            })

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[MetricsJsonlCallback()],
    )

    checkpoint = latest_checkpoint(output_dir) if bool(training_config.get("resume", True)) else None
    quarantined_rng_files = quarantine_rng_state_files(checkpoint)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if trainer.is_world_process_zero():
        write_environment_snapshot(output_dir / "environment_snapshot.json", preflight)
        append_jsonl(metrics_path, {
            "created_at": utc_now(),
            "event": "training_start",
            "route": MAIN_ROUTE,
            "base_model": base_model,
            "train_samples": len(train_records),
            "eval_samples": len(eval_records),
            "preflight": preflight.to_dict(),
            "resume_checkpoint": checkpoint,
            "quarantined_rng_state_files": quarantined_rng_files,
            "torch": torch.__version__,
        })

    trainer.train(resume_from_checkpoint=checkpoint)
    eval_metrics = trainer.evaluate()
    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))

    prediction_output: str | None = None
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(final_model_dir))
        model.save_main_route(final_model_dir)
        append_jsonl(metrics_path, {
            "created_at": utc_now(),
            "event": "final_eval",
            "metrics": sanitize_json(eval_metrics),
        })
        max_prediction_samples = int(eval_config.get("max_prediction_samples", 32))
        write_validation_predictions(
            prediction_path,
            model,
            collator,
            normalizer,
            eval_records[:max_prediction_samples],
        )
        prediction_output = str(prediction_path)

    return MainRouteRunResult(
        output_dir=str(output_dir),
        train_samples=len(train_records),
        eval_samples=len(eval_records),
        checkpoint=checkpoint,
        metrics_path=str(metrics_path),
        prediction_path=prediction_output,
    )


def validate_main_route_config(config: dict[str, Any]) -> None:
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


def write_validation_predictions(
    path: Path,
    model: Any,
    collator: MainRouteBatchCollator,
    normalizer: HourValueNormalizer,
    records: list[dict[str, Any]],
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
            next_hour_values = output["next_hour_value_logits"][0].detach().cpu().tolist()
            next_hour_vent_probability = float(torch.sigmoid(output["next_hour_vent_logits"][0]).detach().cpu().item())
            domain_scores = torch.sigmoid(output["next24_domain_logits"][0]).detach().cpu().tolist()
            binary_scores = torch.sigmoid(output["next24_binary_logits"][0]).detach().cpu().tolist()
            multiclass_indices = [
                int(logits[0].detach().cpu().argmax().item())
                for logits in output["next24_multiclass_logits"]
            ]
            prediction = {
                "sample_id": record["sample_id"],
                "hadm_id": record["hadm_id"],
                "stay_id": record["stay_id"],
                "prediction_hour": record["prediction_hour"],
                "prediction": {
                    "next_hour": {
                        "label": "NEXT_HOUR",
                        "relative_hour": "H+1",
                        "value_order": list(HOUR_VALUE_ORDER),
                        "values": normalizer.denormalize_row(next_hour_values),
                        "vent_probability": next_hour_vent_probability,
                    },
                    "next24h": decode_next24_predictions(
                        domain_scores=domain_scores,
                        binary_scores=binary_scores,
                        multiclass_indices=multiclass_indices,
                    ),
                },
                "target": record["targets"],
            }
            handle.write(json.dumps(prediction, sort_keys=True) + "\n")
