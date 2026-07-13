from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .collator import MultiresEventCollator
from .contract import (
    EventTemplate,
    EventTemplateRegistry,
    SupervisionContract,
    TargetLayout,
    TargetSlot,
)
from .dataset import MultiresEventDataset
from .normalization import RobustNormalizer
from .preflight import MultiresPreflightResult, preflight_multires_event
from .sampler import SubjectAnchorDistributedSampler


@dataclass(frozen=True)
class MultiresEventRuntime:
    train_loader: Any
    eval_loader: Any
    train_sampler: SubjectAnchorDistributedSampler
    eval_sampler: SubjectAnchorDistributedSampler
    dataset_fingerprint: Mapping[str, Any]
    target_contract: TargetLayout
    normalization: RobustNormalizer
    train_dataset: MultiresEventDataset
    eval_dataset: MultiresEventDataset
    preflight: MultiresPreflightResult


def build_runtime(
    config: Mapping[str, Any],
    dataset_root: str | Path,
    supervision_path: str | Path,
    normalization_path: str | Path | None,
    rank: int,
    world_size: int,
) -> MultiresEventRuntime:
    """Build the frozen C4 loaders without reconstructing samples or splits."""
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - torch is a train extra
        raise RuntimeError("build_runtime requires the train extra with torch") from exc

    if not isinstance(config, Mapping):
        raise ValueError("multires runtime config must be a mapping")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"invalid runtime rank/world_size: {rank}/{world_size}")

    preflight = preflight_multires_event(config, dataset_root, supervision_path)
    supervision = SupervisionContract.from_json(supervision_path)
    loader_config = _mapping_or_empty(config.get("loader"))
    training_config = _mapping_or_empty(config.get("training"))
    cache_shards = int(loader_config.get("cache_shards", 1))
    num_workers = int(
        training_config.get("dataloader_num_workers", loader_config.get("num_workers", 0))
    )
    if num_workers != 0:
        raise ValueError(
            "multires_event_v1 gzip runtime requires dataloader_num_workers=0 so its "
            "one-shard cache can guarantee one decompression per selected shard per epoch"
        )
    if cache_shards != 1:
        raise ValueError("multires_event_v1 baseline requires cache_shards=1")

    train_dataset = MultiresEventDataset(
        dataset_root,
        "train",
        supervision,
        cache_shards=cache_shards,
        strict=True,
    )
    eval_dataset = MultiresEventDataset(
        dataset_root,
        "val",
        supervision,
        cache_shards=cache_shards,
        strict=True,
    )
    if train_dataset.target_layout != eval_dataset.target_layout:
        raise ValueError("train and validation target layouts differ")
    target_layout = train_dataset.target_layout

    sampler_config = _mapping_or_empty(config.get("sampler"))
    seed = int(config.get("seed", sampler_config.get("seed", 20260712)))
    max_train_subjects = _optional_positive_int(training_config.get("max_train_subjects"))
    max_eval_subjects = _optional_positive_int(training_config.get("max_eval_subjects"))
    train_sampler = SubjectAnchorDistributedSampler(
        train_dataset,
        rank=rank,
        world_size=world_size,
        seed=seed,
        mode="subject_uniform",
        shuffle=True,
        pad_to_world_size=False,
        require_even_divisible=True,
        max_subjects=max_train_subjects,
    )

    evaluation = _mapping_or_empty(config.get("evaluation"))
    phase = str(evaluation.get("phase", "interval"))
    if phase == "interval":
        eval_mode = "one_fixed_per_subject"
        eval_max_samples = _optional_positive_int(evaluation.get("interval_expected_subjects"))
    elif phase == "final":
        eval_mode = "anchor_uniform"
        eval_max_samples = _optional_positive_int(evaluation.get("final_expected_samples"))
    else:
        raise ValueError(f"evaluation.phase must be interval or final, got {phase!r}")
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
        max_samples=eval_max_samples,
    )
    if eval_max_samples is not None and eval_sampler.global_sample_count != eval_max_samples:
        raise ValueError(
            f"{phase} evaluation contract expects {eval_max_samples} persisted samples, "
            f"but the eligible sampler resolves to {eval_sampler.global_sample_count}"
        )

    normalization = _load_or_fit_normalization(
        config=config,
        path=normalization_path,
        train_dataset=train_dataset,
        supervision=supervision,
        target_layout=target_layout,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )
    missing_f24_stats = []
    for source_index in target_layout.derived_primary_f24_indices:
        slot = target_layout.slots[source_index]
        template = train_dataset.templates.get(
            slot.field_id, slot.operator_id, slot.condition_id
        )
        if slot.loss_family in {"ordinal", "count"}:
            continue
        if not normalization.has_event_stat(template, "F24"):
            missing_f24_stats.append(slot.slot_id)
    if missing_f24_stats:
        raise ValueError(
            "normalization cannot invert every derived F24 target; missing "
            f"{missing_f24_stats[:5]}"
        )
    preflight = replace(
        preflight,
        normalization_fallback_key_count=len(normalization.fallback_event_keys),
        normalization_fallback_level_counts=dict(normalization.fallback_level_counts),
    )
    collator = MultiresEventCollator(
        supervision=supervision,
        templates=train_dataset.templates,
        target_layout=target_layout,
        normalization=normalization,
    )
    train_batch_size = int(
        training_config.get("per_device_train_batch_size", loader_config.get("train_batch_size", 1))
    )
    eval_batch_size = int(
        training_config.get("per_device_eval_batch_size", loader_config.get("eval_batch_size", 1))
    )
    if train_batch_size < 1 or eval_batch_size != 1:
        raise ValueError(
            "train batch size must be positive and eval batch size must be exactly one "
            "for subject-macro evaluation"
        )
    pin_memory = bool(loader_config.get("pin_memory", True))
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=pin_memory,
        persistent_workers=False,
        drop_last=False,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        sampler=eval_sampler,
        num_workers=0,
        pin_memory=pin_memory,
        persistent_workers=False,
        drop_last=False,
        collate_fn=collator,
    )
    identity = {
        "dataset_id": preflight.dataset_id,
        "fingerprint": preflight.dataset_fingerprint,
        "dataset_fingerprint": preflight.dataset_fingerprint,
        "source_fingerprint": preflight.source_fingerprint,
        "sample_count": preflight.sample_count,
        "split_counts": dict(preflight.split_counts),
        "subject_counts": dict(preflight.subject_counts),
        "shard_count": preflight.shard_count,
        "normalization_fallback_key_count": preflight.normalization_fallback_key_count,
        "normalization_fallback_level_counts": dict(
            preflight.normalization_fallback_level_counts or {}
        ),
    }
    return MultiresEventRuntime(
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_sampler=train_sampler,
        eval_sampler=eval_sampler,
        dataset_fingerprint=identity,
        target_contract=target_layout,
        normalization=normalization,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        preflight=preflight,
    )


def _load_or_fit_normalization(
    *,
    config: Mapping[str, Any],
    path: str | Path | None,
    train_dataset: MultiresEventDataset,
    supervision: SupervisionContract,
    target_layout: TargetLayout,
    seed: int,
    rank: int,
    world_size: int,
) -> RobustNormalizer:
    normalization_config = _mapping_or_empty(config.get("normalization"))
    selected_path = path if path is not None else normalization_config.get("path")
    if selected_path in (None, ""):
        raise ValueError("a persisted normalization_path is required")
    destination = Path(str(selected_path))
    if "${" in str(destination):
        raise ValueError(f"normalization_path contains an unexpanded variable: {destination}")
    destination = destination.resolve()

    def load() -> RobustNormalizer:
        return RobustNormalizer.from_json(
            destination,
            expected_dataset_fingerprint=train_dataset.dataset_fingerprint,
            expected_supervision_sha256=supervision.source_sha256,
        )

    if destination.is_file():
        return load()

    distributed = _distributed_ready(world_size)
    if world_size > 1 and not distributed:
        raise RuntimeError(
            "normalization fitting for world_size>1 requires an initialized torch.distributed "
            "group so rank zero can fit once and synchronize the persisted artifact"
        )
    if rank == 0:
        fit_sampler = SubjectAnchorDistributedSampler(
            train_dataset,
            rank=0,
            world_size=1,
            seed=seed,
            mode="one_fixed_per_subject",
            shuffle=True,
            pad_to_world_size=False,
            require_even_divisible=False,
            max_subjects=None,
        )
        fitted = RobustNormalizer.fit(
            train_dataset.iter_indices(list(fit_sampler)),
            templates=train_dataset.templates,
            target_layout=target_layout,
            supervision=supervision,
            dataset_fingerprint=train_dataset.dataset_fingerprint,
            clip_value=float(normalization_config.get("clip_value", 10.0)),
            epsilon=float(normalization_config.get("epsilon", 1e-6)),
            max_values_per_key=int(normalization_config.get("max_values_per_key", 200_000)),
            seed=int(normalization_config.get("seed", seed)),
        )
        fitted.save_json(destination)
    if distributed:
        import torch.distributed as dist

        dist.barrier()
    if not destination.is_file():
        raise FileNotFoundError(f"normalization fit did not create {destination}")
    return load()


def _distributed_ready(world_size: int) -> bool:
    if world_size == 1:
        return False
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover - torch is a train extra
        return False
    return bool(dist.is_available() and dist.is_initialized())


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result < 1:
        raise ValueError("configured sampler cap must be positive")
    return result


__all__ = [
    "EventTemplate",
    "EventTemplateRegistry",
    "MultiresEventCollator",
    "MultiresEventDataset",
    "MultiresEventRuntime",
    "MultiresPreflightResult",
    "RobustNormalizer",
    "SubjectAnchorDistributedSampler",
    "SupervisionContract",
    "TargetLayout",
    "TargetSlot",
    "build_runtime",
    "preflight_multires_event",
]
