from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Iterator
from typing import Any, Sequence

try:
    from torch.utils.data import Sampler
except ImportError:  # pragma: no cover - torch is a train extra
    class Sampler:  # type: ignore[no-redef]
        pass


class SubjectAnchorDistributedSampler(Sampler[int]):
    """Deterministic subject/anchor sampling with gzip-shard locality.

    ``subject_uniform`` visits each subject once and chooses one anchor.
    ``subject_uniform_replacement`` makes an exact number of independent
    uniform-subject then uniform-anchor draws, which is useful for fixed-size
    unbiased optimizer batches. Selected indices are emitted in a randomized
    shard order and randomized within each shard, so a one-shard LRU cache does
    not repeatedly decompress gzip files.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 20260712,
        mode: str = "subject_uniform",
        shuffle: bool = True,
        pad_to_world_size: bool = True,
        require_even_divisible: bool = False,
        max_subjects: int | None = None,
        max_samples: int | None = None,
    ) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"invalid distributed sampler rank/world_size: {rank}/{world_size}")
        if mode not in {
            "subject_uniform",
            "subject_uniform_replacement",
            "one_fixed_per_subject",
            "anchor_uniform",
        }:
            raise ValueError(f"invalid multires sampler mode: {mode}")
        if mode == "subject_uniform_replacement" and max_samples is None:
            raise ValueError(
                "subject_uniform_replacement requires an exact max_samples draw count"
            )
        if len(dataset.subject_ids) != len(dataset) or len(dataset.shard_keys) != len(dataset):
            raise ValueError("dataset must expose aligned subject_ids and shard_keys")
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.mode = mode
        self.shuffle = bool(shuffle)
        self.pad_to_world_size = bool(pad_to_world_size)
        self.require_even_divisible = bool(require_even_divisible)
        self.max_subjects = max_subjects
        self.max_samples = max_samples
        self.epoch = 0
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, subject_id in enumerate(dataset.subject_ids):
            grouped[str(subject_id)].append(index)
        subjects = sorted(grouped, key=lambda value: _stable_key(self.seed, value))
        if max_subjects is not None:
            if max_subjects < 1:
                raise ValueError("max_subjects must be positive")
            subjects = subjects[:max_subjects]
        self.subject_to_indices = {subject: tuple(grouped[subject]) for subject in subjects}
        self.eligible_indices = tuple(
            index for subject in subjects for index in self.subject_to_indices[subject]
        )
        if not self.eligible_indices:
            raise ValueError("sampler has no eligible persisted anchors")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("sampler epoch must be nonnegative")
        self.epoch = int(epoch)

    def state_dict(self) -> dict[str, int | str]:
        return {
            "epoch": self.epoch,
            "seed": self.seed,
            "mode": self.mode,
            "rank": self.rank,
            "world_size": self.world_size,
        }

    def load_state_dict(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise ValueError("sampler state must be a mapping")
        immutable = {
            "seed": self.seed,
            "mode": self.mode,
            "rank": self.rank,
            "world_size": self.world_size,
        }
        for key, expected in immutable.items():
            if state.get(key) != expected:
                raise ValueError(
                    f"sampler resume identity mismatch for {key}: "
                    f"{state.get(key)!r} != {expected!r}"
                )
        self.set_epoch(int(state.get("epoch", 0)))

    @property
    def active_mode(self) -> str:
        return self.mode

    @property
    def global_sample_count(self) -> int:
        return len(self._selected_indices())

    def __iter__(self) -> Iterator[int]:
        order = self._global_order()
        if self.require_even_divisible and len(order) % self.world_size:
            raise ValueError(
                f"{len(order)} selected anchors are not divisible by world_size={self.world_size}; "
                "refusing duplicate DDP padding"
            )
        local = order[self.rank :: self.world_size]
        if self.pad_to_world_size:
            target = math.ceil(len(order) / self.world_size)
            if local and len(local) < target:
                local.extend([local[-1]] * (target - len(local)))
        return iter(local)

    def __len__(self) -> int:
        count = len(self._selected_indices())
        if self.require_even_divisible and count % self.world_size:
            raise ValueError(
                f"{count} selected anchors are not divisible by world_size={self.world_size}"
            )
        if self.pad_to_world_size:
            return math.ceil(count / self.world_size)
        return len(range(self.rank, count, self.world_size))

    def _selected_indices(self) -> list[int]:
        mode = self.active_mode
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        if mode == "subject_uniform_replacement":
            if self.max_samples is None:  # guarded in __init__; keeps the type explicit
                raise AssertionError("replacement sampler lacks its exact draw count")
            subjects = tuple(self.subject_to_indices)
            selected = []
            for _ in range(self.max_samples):
                subject = subjects[rng.randrange(len(subjects))]
                values = self.subject_to_indices[subject]
                selected.append(values[rng.randrange(len(values))])
            return selected
        if mode in {"subject_uniform", "one_fixed_per_subject"}:
            selected = []
            for subject, values in self.subject_to_indices.items():
                if mode == "one_fixed_per_subject":
                    position = _stable_key(self.seed, subject) % len(values)
                else:
                    position = rng.randrange(len(values))
                selected.append(values[position])
        else:
            selected = list(self.eligible_indices)
        if self.max_samples is not None and len(selected) > self.max_samples:
            if self.max_samples < 1:
                raise ValueError("max_samples must be positive")
            rng.shuffle(selected)
            selected = selected[: self.max_samples]
        return selected

    def _global_order(self) -> list[int]:
        selected = self._selected_indices()
        by_shard: dict[str, list[int]] = defaultdict(list)
        for index in selected:
            by_shard[str(self.dataset.shard_keys[index])].append(index)
        shard_keys = sorted(by_shard)
        rng = random.Random(self.seed + self.epoch * 1_000_003 + 97)
        if self.shuffle:
            rng.shuffle(shard_keys)
        order: list[int] = []
        for shard_key in shard_keys:
            values = by_shard[shard_key]
            if self.shuffle:
                rng.shuffle(values)
            else:
                values.sort()
            order.extend(values)
        return order


def _stable_key(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
