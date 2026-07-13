from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contract import (
    EventTemplate,
    EventTemplateRegistry,
    NORMALIZATION_SCHEMA,
    SupervisionContract,
    TargetLayout,
)


@dataclass(frozen=True)
class RobustStat:
    median: float
    iqr: float
    count: int
    sampled_count: int
    log1p: bool


class _Reservoir:
    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.values: list[float] = []
        self.count = 0
        self.rng = random.Random(seed)

    def add(self, value: float) -> None:
        self.count += 1
        if len(self.values) < self.capacity:
            self.values.append(value)
            return
        position = self.rng.randrange(self.count)
        if position < self.capacity:
            self.values[position] = value


class RobustNormalizer:
    def __init__(
        self,
        *,
        event_stats: Mapping[str, RobustStat],
        fallback_event_stats: Mapping[str, RobustStat],
        fallback_event_keys: Mapping[str, str],
        static_stats: Mapping[str, RobustStat],
        dataset_fingerprint: str,
        supervision_sha256: str,
        fit_split: str,
        subject_count: int,
        subject_ids_sha256: str,
        clip_value: float = 10.0,
        epsilon: float = 1e-6,
    ) -> None:
        if fit_split != "train":
            raise ValueError("normalization statistics must be fit on the train split")
        if clip_value <= 0 or epsilon <= 0:
            raise ValueError("normalization clip_value and epsilon must be positive")
        self.event_stats = dict(event_stats)
        self.fallback_event_stats = dict(fallback_event_stats)
        self.fallback_event_keys = dict(fallback_event_keys)
        self.static_stats = dict(static_stats)
        self.dataset_fingerprint = dataset_fingerprint
        self.supervision_sha256 = supervision_sha256
        self.fit_split = fit_split
        self.subject_count = int(subject_count)
        self.subject_ids_sha256 = subject_ids_sha256
        self.clip_value = float(clip_value)
        self.epsilon = float(epsilon)

    @classmethod
    def fit(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        templates: EventTemplateRegistry,
        target_layout: TargetLayout,
        supervision: SupervisionContract,
        dataset_fingerprint: str,
        clip_value: float = 10.0,
        epsilon: float = 1e-6,
        max_values_per_key: int = 200_000,
        seed: int = 20260712,
        max_samples: int | None = None,
    ) -> "RobustNormalizer":
        if max_values_per_key < 100:
            raise ValueError("max_values_per_key must be at least 100")
        event_values: dict[str, _Reservoir] = {}
        fallback_event_values: dict[str, _Reservoir] = {}
        static_values: dict[str, _Reservoir] = {}
        subjects: set[str] = set()
        static_subjects: set[str] = set()
        sample_count = 0

        def reservoir(store: dict[str, _Reservoir], key: str) -> _Reservoir:
            value = store.get(key)
            if value is None:
                key_seed = int.from_bytes(
                    hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()[:8], "big"
                )
                value = _Reservoir(max_values_per_key, key_seed)
                store[key] = value
            return value

        for record in records:
            if str(record.get("split")) != "train":
                raise ValueError(
                    "normalization fitting received a non-train record; validation/test leakage blocked"
                )
            subject_id = str(record.get("subject_id") or "")
            if not subject_id:
                raise ValueError("normalization fitting record is missing subject_id")
            subjects.add(subject_id)
            blocks = {int(item["block_id"]): item for item in record["block_table"]}
            for event in record["input_events"]:
                template = templates.get(int(event[0]), int(event[1]), int(event[2]))
                block = blocks[int(event[4])]
                _add_event_value(
                    reservoir,
                    event_values,
                    fallback_event_values,
                    template,
                    str(block["resolution"]),
                    event[3],
                )
            normalization_target_indices = (
                target_layout.active_direct_indices
                + target_layout.derived_primary_f24_indices
            )
            for canonical_index in normalization_target_indices:
                if int(record["target_mask"][canonical_index]) == 0:
                    continue
                event = record["target_events"][canonical_index]
                template = templates.get(int(event[0]), int(event[1]), int(event[2]))
                block = blocks[int(event[4])]
                _add_event_value(
                    reservoir,
                    event_values,
                    fallback_event_values,
                    template,
                    str(block["resolution"]),
                    event[3],
                )
            if subject_id not in static_subjects:
                static_subjects.add(subject_id)
                static = record.get("static") or {}
                for field in supervision.static_numeric_fields:
                    raw = static.get(field)
                    if raw is None:
                        continue
                    value = float(raw)
                    if not math.isfinite(value):
                        raise ValueError(f"static field {field} contains a non-finite value")
                    reservoir(static_values, field).add(value)
            sample_count += 1
            if max_samples is not None and sample_count >= max_samples:
                break

        if not subjects:
            raise ValueError("normalization fit received no train records")
        event_stats = {
            key: _finalize_reservoir(value, log1p=key.endswith("|log1p"))
            for key, value in event_values.items()
        }
        event_stats = {
            key.removesuffix("|log1p"): stat for key, stat in event_stats.items()
        }
        fallback_event_stats = {
            key: _finalize_reservoir(value, log1p=key.endswith(":log1p"))
            for key, value in fallback_event_values.items()
        }
        fallback_event_keys = _resolve_fallback_event_keys(
            templates=templates,
            event_stats=event_stats,
            fallback_event_stats=fallback_event_stats,
        )
        static_stats = {
            key: _finalize_reservoir(value, log1p=False)
            for key, value in static_values.items()
        }
        missing_required_static = [field for field in ("age",) if field not in static_stats]
        if missing_required_static:
            raise ValueError(f"missing required static normalization stats: {missing_required_static}")
        subject_ids_sha256 = hashlib.sha256(
            "\n".join(sorted(subjects)).encode("utf-8")
        ).hexdigest()
        return cls(
            event_stats=event_stats,
            fallback_event_stats=fallback_event_stats,
            fallback_event_keys=fallback_event_keys,
            static_stats=static_stats,
            dataset_fingerprint=dataset_fingerprint,
            supervision_sha256=supervision.source_sha256,
            fit_split="train",
            subject_count=len(subjects),
            subject_ids_sha256=subject_ids_sha256,
            clip_value=clip_value,
            epsilon=epsilon,
        )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        expected_dataset_fingerprint: str | None = None,
        expected_supervision_sha256: str | None = None,
    ) -> "RobustNormalizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != NORMALIZATION_SCHEMA:
            raise ValueError(f"normalization schema must be {NORMALIZATION_SCHEMA}")
        if payload.get("fit_split") != "train":
            raise ValueError("normalization fit_split must be train")
        if (
            expected_dataset_fingerprint is not None
            and payload.get("dataset_fingerprint") != expected_dataset_fingerprint
        ):
            raise ValueError("normalization dataset fingerprint mismatch")
        if (
            expected_supervision_sha256 is not None
            and payload.get("supervision_sha256") != expected_supervision_sha256
        ):
            raise ValueError("normalization supervision hash mismatch")
        return cls(
            event_stats={key: _stat_from_payload(value) for key, value in payload["event_stats"].items()},
            fallback_event_stats={
                key: _stat_from_payload(value)
                for key, value in payload["fallback_event_stats"].items()
            },
            fallback_event_keys={
                str(key): str(value)
                for key, value in payload["fallback_event_keys"].items()
            },
            static_stats={key: _stat_from_payload(value) for key, value in payload["static_stats"].items()},
            dataset_fingerprint=str(payload["dataset_fingerprint"]),
            supervision_sha256=str(payload["supervision_sha256"]),
            fit_split=str(payload["fit_split"]),
            subject_count=int(payload["subject_count"]),
            subject_ids_sha256=str(payload["subject_ids_sha256"]),
            clip_value=float(payload["clip_value"]),
            epsilon=float(payload["epsilon"]),
        )

    def save_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": NORMALIZATION_SCHEMA,
            "version": 1,
            "dataset_fingerprint": self.dataset_fingerprint,
            "supervision_sha256": self.supervision_sha256,
            "fit_split": self.fit_split,
            "subject_count": self.subject_count,
            "subject_ids_sha256": self.subject_ids_sha256,
            "clip_value": self.clip_value,
            "epsilon": self.epsilon,
            "event_stats": {
                key: _stat_payload(value) for key, value in sorted(self.event_stats.items())
            },
            "fallback_event_stats": {
                key: _stat_payload(value)
                for key, value in sorted(self.fallback_event_stats.items())
            },
            "fallback_event_keys": dict(sorted(self.fallback_event_keys.items())),
            "fallback_level_counts": self.fallback_level_counts,
            "static_stats": {
                key: _stat_payload(value) for key, value in sorted(self.static_stats.items())
            },
        }
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)

    def transform_event(
        self,
        value: Any,
        *,
        template: EventTemplate,
        resolution: str,
        span_hours: float,
        loss_family: str | None = None,
        for_target: bool = False,
    ) -> tuple[float, bool]:
        if value is None:
            return 0.0, False
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("event value is non-finite")
        if loss_family == "ordinal":
            return number, True
        if template.value_type == "duration":
            if span_hours <= 0:
                raise ValueError("duration transform requires positive span_hours")
            return min(max(number / span_hours, 0.0), 1.0), True
        if template.value_type == "binary":
            return number, True
        if for_target and loss_family == "count":
            return number, True
        if template.value_type == "study_slot":
            return number, True
        key = event_stat_key(template, resolution)
        stat = self._event_stat(key, template)
        transformed = math.log1p(number) if stat.log1p else number
        scale = max(stat.iqr, self.epsilon)
        normalized = (transformed - stat.median) / scale
        return min(max(normalized, -self.clip_value), self.clip_value), True

    def transform_static(self, field: str, value: Any) -> tuple[float, bool]:
        if value is None:
            return 0.0, False
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"static field {field} is non-finite")
        stat = self.static_stats.get(field)
        if stat is None:
            raise ValueError(f"normalization statistics missing static field {field}")
        scale = max(stat.iqr, self.epsilon)
        normalized = (number - stat.median) / scale
        return min(max(normalized, -self.clip_value), self.clip_value), True

    def inverse_event(
        self,
        value: float,
        *,
        template: EventTemplate,
        resolution: str,
        span_hours: float,
        loss_family: str | None = None,
        for_target: bool = False,
    ) -> float:
        """Invert a model-view value before cross-resolution composition."""
        number = float(value)
        if loss_family == "ordinal":
            return number
        if template.value_type == "duration":
            return number * span_hours
        if template.value_type == "binary":
            return number
        if for_target and loss_family == "count":
            return number
        if template.value_type == "study_slot":
            return number
        key = event_stat_key(template, resolution)
        stat = self._event_stat(key, template)
        raw = number * max(stat.iqr, self.epsilon) + stat.median
        if stat.log1p:
            return max(math.expm1(raw), 0.0)
        return raw

    def has_event_stat(self, template: EventTemplate, resolution: str) -> bool:
        if template.value_type in {"duration", "binary", "study_slot"}:
            return True
        exact_key = event_stat_key(template, resolution)
        if exact_key in self.event_stats:
            return True
        fallback_key = self.fallback_event_keys.get(exact_key)
        if fallback_key in self.fallback_event_stats:
            return True
        return any(
            key in self.fallback_event_stats for key in _fallback_stat_keys(template)
        )

    @property
    def fallback_level_counts(self) -> Mapping[str, int]:
        counts = {"template": 0, "field": 0, "global": 0}
        for fallback_key in self.fallback_event_keys.values():
            level = fallback_key.split(":", 1)[0]
            counts[level] = counts.get(level, 0) + 1
        return counts

    def _event_stat(self, exact_key: str, template: EventTemplate) -> RobustStat:
        stat = self.event_stats.get(exact_key)
        if stat is not None:
            return stat
        fallback_key = self.fallback_event_keys.get(exact_key)
        if fallback_key is None:
            candidates = _fallback_stat_keys(template)
            fallback_key = next(
                (key for key in candidates if key in self.fallback_event_stats), None
            )
        if fallback_key is None:
            raise ValueError(
                f"normalization has neither an exact nor fallback statistic for {exact_key}"
            )
        try:
            return self.fallback_event_stats[fallback_key]
        except KeyError as exc:
            raise ValueError(
                f"normalization fallback map for {exact_key} references missing {fallback_key}"
            ) from exc


def event_stat_key(template: EventTemplate, resolution: str) -> str:
    return f"{template.field_id}:{template.operator_id}:{template.condition_id}:{resolution}"


def _add_event_value(
    reservoir_factory: Any,
    store: dict[str, _Reservoir],
    fallback_store: dict[str, _Reservoir],
    template: EventTemplate,
    resolution: str,
    raw: Any,
) -> None:
    if raw is None or template.value_type in {"duration", "binary", "study_slot"}:
        return
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"event template {template.key} contains a non-finite value")
    use_log1p = template.value_type in {"nonnegative", "count"}
    if use_log1p:
        if value < 0:
            raise ValueError(f"event template {template.key} requires a nonnegative value")
        value = math.log1p(value)
    key = event_stat_key(template, resolution) + ("|log1p" if use_log1p else "")
    reservoir_factory(store, key).add(value)
    for fallback_key in _fallback_stat_keys(template):
        reservoir_factory(fallback_store, fallback_key).add(value)


def _fallback_stat_keys(template: EventTemplate) -> tuple[str, str, str]:
    mode = "log1p" if template.value_type in {"nonnegative", "count"} else "linear"
    return (
        f"template:{template.field_id}:{template.operator_id}:{template.condition_id}:{mode}",
        f"field:{template.field_id}:{mode}",
        f"global:{template.value_type}:{mode}",
    )


def _resolve_fallback_event_keys(
    *,
    templates: EventTemplateRegistry,
    event_stats: Mapping[str, RobustStat],
    fallback_event_stats: Mapping[str, RobustStat],
) -> dict[str, str]:
    result: dict[str, str] = {}
    resolutions = ("H1", "M4", "F24")
    for template in templates.by_key.values():
        if template.value_type in {"duration", "binary", "study_slot"}:
            continue
        candidates = _fallback_stat_keys(template)
        fallback_key = next(
            (key for key in candidates if key in fallback_event_stats), None
        )
        for resolution in resolutions:
            exact_key = event_stat_key(template, resolution)
            if exact_key in event_stats:
                continue
            if fallback_key is None:
                raise ValueError(
                    "deterministic train-subject normalization anchor did not observe "
                    f"template {template.key} at any resolution; no registered fallback exists"
                )
            result[exact_key] = fallback_key
    return result


def _finalize_reservoir(reservoir: _Reservoir, *, log1p: bool) -> RobustStat:
    if not reservoir.values:
        raise ValueError("cannot finalize empty robust-stat reservoir")
    values = sorted(reservoir.values)
    median = _percentile(values, 0.5)
    q1 = _percentile(values, 0.25)
    q3 = _percentile(values, 0.75)
    return RobustStat(
        median=median,
        iqr=max(q3 - q1, 0.0),
        count=reservoir.count,
        sampled_count=len(values),
        log1p=log1p,
    )


def _percentile(values: list[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _stat_payload(value: RobustStat) -> dict[str, Any]:
    return {
        "median": value.median,
        "iqr": value.iqr,
        "count": value.count,
        "sampled_count": value.sampled_count,
        "log1p": value.log1p,
    }


def _stat_from_payload(value: Mapping[str, Any]) -> RobustStat:
    return RobustStat(
        median=float(value["median"]),
        iqr=float(value["iqr"]),
        count=int(value["count"]),
        sampled_count=int(value["sampled_count"]),
        log1p=bool(value["log1p"]),
    )
