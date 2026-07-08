from __future__ import annotations

import json
import platform
import inspect
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trauma_predict.data.preflight import ArtifactPreflightResult
from trauma_predict.data.records import load_text_records, resolve_shard_paths
from trauma_predict.training.checkpoints import sorted_checkpoints


@dataclass(frozen=True)
class Seq2SeqRunResult:
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


class TextPairDataset:
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        max_input_tokens: int,
        max_target_tokens: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.max_target_tokens = max_target_tokens

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        item = self.tokenizer(
            row["input_text"],
            max_length=self.max_input_tokens,
            truncation=True,
        )
        labels = self.tokenizer(
            text_target=row["target_text"],
            max_length=self.max_target_tokens,
            truncation=True,
        )
        item["labels"] = labels["input_ids"]
        return item


def run_seq2seq_training(
    train_config: dict[str, Any],
    dataset_config: dict[str, Any],
    output_dir: Path,
    preflight: ArtifactPreflightResult,
) -> Seq2SeqRunResult:
    validate_seq2seq_config(train_config)

    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        set_seed,
    )

    import torch

    model_config = train_config["model"]
    training_config = train_config["training"]
    eval_config = train_config.get("evaluation", {})
    output_config = train_config.get("outputs", {})

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(output_config.get("metrics_jsonl", output_dir / "metrics.jsonl"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "validation_predictions.jsonl"

    set_seed(int(train_config.get("seed", 0)))
    base_model = str(model_config["base_model"])
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    if bool(training_config.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    required_fields = list(dataset_config.get("required_sample_fields") or [])
    train_records = load_text_records(resolve_shard_paths(dataset_config, "train"), required_fields)
    eval_records = load_text_records(resolve_shard_paths(dataset_config, "val"), required_fields)
    train_records = maybe_cap_records(train_records, training_config.get("max_train_samples"))
    eval_records = maybe_cap_records(eval_records, training_config.get("max_eval_samples"))

    train_dataset = TextPairDataset(
        train_records,
        tokenizer=tokenizer,
        max_input_tokens=int(model_config.get("max_input_tokens", 1024)),
        max_target_tokens=int(model_config.get("max_target_tokens", 256)),
    )
    eval_dataset = TextPairDataset(
        eval_records,
        tokenizer=tokenizer,
        max_input_tokens=int(model_config.get("max_input_tokens", 1024)),
        max_target_tokens=int(model_config.get("max_target_tokens", 256)),
    )

    precision = str(training_config.get("precision", "fp16")).lower()
    fp16 = precision == "fp16"
    bf16 = precision == "bf16"
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if fp16 or bf16 else None,
    )

    training_args_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "run_name": str(train_config.get("run_name", "trauma_predict_run")),
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
        "predict_with_generate": False,
        "dataloader_num_workers": int(training_config.get("dataloader_num_workers", 2)),
        "report_to": [],
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
    }
    argument_names = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "eval_strategy" in argument_names:
        training_args_kwargs["eval_strategy"] = "steps"
    else:
        training_args_kwargs["evaluation_strategy"] = "steps"
    args = Seq2SeqTrainingArguments(**training_args_kwargs)

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

    trainer = Seq2SeqTrainer(
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
    trainer.save_model(str(output_dir / "final_model"))

    prediction_output: str | None = None
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(output_dir / "final_model"))
        append_jsonl(metrics_path, {
            "created_at": utc_now(),
            "event": "final_eval",
            "metrics": sanitize_json(eval_metrics),
        })
        max_prediction_samples = int(eval_config.get("max_prediction_samples", 32))
        write_validation_predictions(
            prediction_path,
            trainer.model,
            tokenizer,
            eval_records[:max_prediction_samples],
            max_input_tokens=int(model_config.get("max_input_tokens", 1024)),
            max_new_tokens=int(model_config.get("generation_max_new_tokens", model_config.get("max_target_tokens", 256))),
        )
        prediction_output = str(prediction_path)

    return Seq2SeqRunResult(
        output_dir=str(output_dir),
        train_samples=len(train_records),
        eval_samples=len(eval_records),
        checkpoint=checkpoint,
        metrics_path=str(metrics_path),
        prediction_path=prediction_output,
    )


def validate_seq2seq_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "trauma_predict.train_config.v1":
        raise ValueError("train config schema_version mismatch")
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("train config model must be an object")
    if model.get("task") != "next24_text_generation":
        raise ValueError("model.task must be next24_text_generation")
    base_model = str(model.get("base_model") or "")
    if not base_model:
        raise ValueError("model.base_model is required")
    if "distilbert" in base_model.lower():
        raise ValueError("DistilBERT is encoder-only and cannot run this seq2seq text-generation task")
    if int(model.get("max_input_tokens", 1)) < 128:
        raise ValueError("model.max_input_tokens must be >= 128")
    if int(model.get("max_target_tokens", 1)) < 32:
        raise ValueError("model.max_target_tokens must be >= 32")
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("train config training must be an object")
    precision = str(training.get("precision", "fp32")).lower()
    if precision == "fp16":
        raise ValueError("fp16 is disabled for this T5 text-generation run because it produced NaN gradients on T4")
    if precision not in {"fp32", "no", "none", "bf16"}:
        raise ValueError("training.precision must be one of fp32/no/none/bf16")
    if float(training.get("learning_rate", 0.0)) > 1e-5:
        raise ValueError("training.learning_rate must be <= 1e-5 for the first stable T5 run")


def maybe_cap_records(records: list[dict[str, Any]], cap: Any) -> list[dict[str, Any]]:
    if cap in (None, "", 0):
        return records
    cap_int = int(cap)
    if cap_int < 1:
        raise ValueError("sample caps must be positive")
    return records[:cap_int]


def latest_checkpoint(output_dir: Path) -> str | None:
    checkpoints = sorted_checkpoints(output_dir)
    return str(checkpoints[-1]) if checkpoints else None


def quarantine_rng_state_files(checkpoint: str | None) -> list[str]:
    if not checkpoint:
        return []
    checkpoint_path = Path(checkpoint)
    paths = []
    for name in ("rng_state.pth",):
        paths.append(checkpoint_path / name)
    paths.extend(sorted(checkpoint_path.glob("rng_state_*.pth")))

    quarantined: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        target = _unused_quarantine_path(path)
        try:
            path.rename(target)
        except FileNotFoundError:
            continue
        quarantined.append(str(target))
    return quarantined


def _unused_quarantine_path(path: Path) -> Path:
    base = path.with_name(f"{path.name}.ignored_for_torch_weights_only")
    if not base.exists():
        return base
    index = 1
    while True:
        candidate = path.with_name(f"{path.name}.ignored_for_torch_weights_only.{index}")
        if not candidate.exists():
            return candidate
        index += 1


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_environment_snapshot(path: Path, preflight: ArtifactPreflightResult) -> None:
    payload = {
        "created_at": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "preflight": preflight.to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_validation_predictions(
    path: Path,
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    max_input_tokens: int,
    max_new_tokens: int,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            inputs = tokenizer(
                row["input_text"],
                max_length=max_input_tokens,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            handle.write(json.dumps({
                "sample_id": row["sample_id"],
                "hadm_id": row["hadm_id"],
                "stay_id": row["stay_id"],
                "prediction_hour": row["prediction_hour"],
                "prediction_text": prediction,
                "target_text": row["target_text"],
            }, sort_keys=True) + "\n")


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
