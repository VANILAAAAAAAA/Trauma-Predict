from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .aggregation import (
    aggregate_cxr_template,
    aggregate_template,
    should_emit_input,
)
from .allocation import MAX_HISTORY_HOURS, TimeBlock, allocate_h1_input_blocks
from .io import StayData, load_stay
from .registry import EventRegistry, Template


EventArray = list[int | float | None]
TUPLE_ORDER = ["field_id", "operator_id", "condition_id", "value", "block_id"]


class GRUDH1SampleBuilder:
    """Build the input-only H1 view for the classic GRU-D baseline.

    The builder reuses the frozen clinical field, aggregation, unit, missingness,
    and availability rules.  It changes only the input time partition: every
    visible history hour is represented by one H1 block.
    """

    def __init__(
        self,
        registry: EventRegistry,
        *,
        max_history_hours: int = MAX_HISTORY_HOURS,
    ) -> None:
        self.registry = registry
        self.max_history_hours = int(max_history_hours)
        self.h1_templates = tuple(registry.allowed_templates("H1", "input"))
        if len(self.h1_templates) != 118:
            raise ValueError(
                f"Frozen H1 input registry must expose 118 templates, found {len(self.h1_templates)}"
            )

    def build_from_dir(
        self,
        field_ready_dir: Path,
        *,
        prediction_hour: int,
        split: str,
        base_content_hash: str,
        target_content_hash: str,
        target_shard_key: str,
        target_line_index: int,
    ) -> dict[str, Any]:
        return self.build(
            load_stay(field_ready_dir),
            prediction_hour=prediction_hour,
            split=split,
            base_content_hash=base_content_hash,
            target_content_hash=target_content_hash,
            target_shard_key=target_shard_key,
            target_line_index=target_line_index,
        )

    def build(
        self,
        stay: StayData,
        *,
        prediction_hour: int,
        split: str,
        base_content_hash: str,
        target_content_hash: str,
        target_shard_key: str,
        target_line_index: int,
    ) -> dict[str, Any]:
        prediction_hour = int(prediction_hour)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"invalid patient split: {split!r}")
        if prediction_hour + 24 > stay.available_until_hour + 1e-9:
            raise ValueError(
                f"anchor no longer has the frozen complete future: "
                f"target_end={prediction_hour + 24} available_until={stay.available_until_hour:.3f}"
            )
        for label, digest in (
            ("base_content_hash", base_content_hash),
            ("target_content_hash", target_content_hash),
        ):
            if not _is_sha256(digest):
                raise ValueError(f"{label} is not a SHA-256 digest")
        if not target_shard_key:
            raise ValueError("target_shard_key must be non-empty")
        if isinstance(target_line_index, bool) or int(target_line_index) < 0:
            raise ValueError("target_line_index must be a non-negative integer")
        self._validate_source_contract(stay)
        allocation = allocate_h1_input_blocks(
            prediction_hour,
            max_history_hours=self.max_history_hours,
        )

        input_events: list[EventArray] = []
        input_source_count: list[int] = []
        for block_id, block in enumerate(allocation.blocks):
            events, counts = self._compile_input_block(stay, block, block_id)
            input_events.extend(events)
            input_source_count.extend(counts)

        sample_id = f"hadm_{stay.hadm_id}_stay_{stay.stay_id}_h{prediction_hour}"
        sample: dict[str, Any] = {
            "schema": "grud_h1_baseline_input_sample_v1",
            "sample_id": sample_id,
            "subject_id": stay.subject_id,
            "hadm_id": stay.hadm_id,
            "stay_id": stay.stay_id,
            "sample_key": stay.sample_key,
            "split": split,
            "prediction_hour": prediction_hour,
            "static": self.registry.compile_static(stay.static),
            "input_geometry": allocation.to_dict(),
            "input_events": input_events,
            "input_source_count": input_source_count,
            "source_reference": {
                "base_content_hash": base_content_hash,
                "target_content_hash": target_content_hash,
            },
            "target_reference": {
                "sample_id": sample_id,
                "contract": "multires_event_m4_target_v2_c4_full_20260714_r9",
                "future_blocks": 6,
                "resolution": "M4",
                "stochastic_factors": 414,
                "target_content_hash": target_content_hash,
                "target_shard_key": str(target_shard_key),
                "target_line_index": int(target_line_index),
            },
            "registry": {
                "registry_version": self.registry.version,
                "resolution": "H1",
                "h1_template_count": len(self.h1_templates),
                "tuple_order": TUPLE_ORDER,
                "padding_id": self.registry.padding_id,
            },
        }
        sample["content_hash"] = content_hash(sample)
        assert_valid_h1_sample(sample, self.registry)
        return sample

    def _compile_input_block(
        self,
        stay: StayData,
        block: TimeBlock,
        block_id: int,
    ) -> tuple[list[EventArray], list[int]]:
        rows: list[tuple[EventArray, int]] = []
        for template in self.h1_templates:
            results = (
                aggregate_cxr_template(stay, block, template)
                if template.field == "cxr"
                else [aggregate_template(stay, block, template)]
            )
            for result in results:
                if should_emit_input(template, result):
                    rows.append(
                        (
                            self._event(template, result.value, block_id),
                            int(result.source_count),
                        )
                    )
        rows.sort(key=self._input_sort_key)
        return [row[0] for row in rows], [row[1] for row in rows]

    def _event(
        self,
        template: Template,
        value: float | None,
        block_id: int,
    ) -> EventArray:
        return [
            self.registry.field_id(template.field),
            self.registry.operator_id(template.operator),
            self.registry.condition_id(template.condition),
            None if value is None else float(value),
            int(block_id),
        ]

    def _input_sort_key(self, row: tuple[EventArray, int]) -> tuple[float | int, ...]:
        field_id, operator_id, condition_id, value, block_id = row[0]
        if (
            field_id == self.registry.field_id("cxr")
            and operator_id == self.registry.operator_id("OBS")
        ):
            return (int(block_id), int(field_id), 0, float(value or 0), int(condition_id))
        return (int(block_id), int(field_id), int(operator_id), int(condition_id), 0)

    def _validate_source_contract(self, stay: StayData) -> None:
        point_kinds = {"point", "point_amount"}
        interval_kinds = {
            "state_interval",
            "amount_interval",
            "rate_interval",
            "interval_start",
        }
        unknown = (set(stay.points) | set(stay.intervals)) - set(self.registry.fields)
        if unknown:
            raise ValueError(f"Unregistered field-ready fields: {sorted(unknown)}")
        for field, events in stay.points.items():
            meta = self.registry.fields[field]
            if meta["source_kind"] not in point_kinds:
                raise ValueError(f"Field-ready source kind mismatch: {field} arrived as point")
            expected_unit = str(meta.get("unit") or "")
            if any(event.unit and event.unit != expected_unit for event in events):
                raise ValueError(f"Non-canonical unit in point field {field}; expected {expected_unit}")
        filtered_conditions = {
            field: {
                template.condition
                for template in self.registry.templates
                if template.field == field
                and (template.condition_spec or {}).get("kind")
                in {"event_subtype", "antibiotic_source"}
            }
            for field in ("respiratory_support", "vasopressor_support", "antibiotics")
        }
        for field, events in stay.intervals.items():
            meta = self.registry.fields[field]
            if meta["source_kind"] not in interval_kinds:
                raise ValueError(f"Field-ready source kind mismatch: {field} arrived as interval")
            expected_unit = str(meta.get("unit") or "")
            if any(event.unit and event.unit != expected_unit for event in events):
                raise ValueError(f"Non-canonical unit in interval field {field}; expected {expected_unit}")
            if field in filtered_conditions:
                illegal = sorted(
                    {event.condition for event in events} - filtered_conditions[field]
                )
                if illegal:
                    raise ValueError(f"Unregistered {field} conditions: {illegal}")
        cxr_labels = {
            template.condition
            for template in self.registry.templates
            if template.field == "cxr" and template.operator == "OBS"
        }
        illegal_cxr = sorted(
            {label for event in stay.cxr_events for label in event.labels} - cxr_labels
        )
        if illegal_cxr:
            raise ValueError(f"Unregistered CXR labels: {illegal_cxr}")


def content_hash(sample: dict[str, Any]) -> str:
    payload = {key: value for key, value in sample.items() if key != "content_hash"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_h1_sample(sample: dict[str, Any], registry: EventRegistry) -> list[str]:
    errors: list[str] = []
    if sample.get("schema") != "grud_h1_baseline_input_sample_v1":
        errors.append("wrong sample schema")
    prediction_hour = sample.get("prediction_hour")
    if isinstance(prediction_hour, bool) or not isinstance(prediction_hour, int):
        errors.append("prediction_hour must be an integer")
        prediction_hour = -1
    geometry = sample.get("input_geometry") or {}
    start = geometry.get("history_start_hour")
    block_count = geometry.get("block_count")
    if (
        geometry.get("resolution") != "H1"
        or geometry.get("history_end_hour") != prediction_hour
        or not isinstance(start, int)
        or not isinstance(block_count, int)
        or block_count != prediction_hour - start
        or block_count < 1
        or block_count > 312
    ):
        errors.append("input geometry differs from the H1 history contract")
    events = sample.get("input_events") or []
    counts = sample.get("input_source_count") or []
    if len(events) != len(counts):
        errors.append("input event/source-count arrays have different lengths")
    previous_key: tuple[float | int, ...] | None = None
    cxr_field = registry.field_id("cxr")
    cxr_obs = registry.operator_id("OBS")
    for index, event in enumerate(events):
        if not isinstance(event, list) or len(event) != 5:
            errors.append(f"input_events[{index}] is not a five-element tuple")
            continue
        field_id, operator_id, condition_id, value, block_id = event
        if any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in (field_id, operator_id, condition_id, block_id)
        ):
            errors.append(f"input_events[{index}] contains non-integer IDs")
            continue
        if not isinstance(block_count, int) or not 0 <= block_id < block_count:
            errors.append(f"input_events[{index}] references invalid H1 block {block_id}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append(f"input_events[{index}] lacks a finite exact value")
            continue
        try:
            field, operator, condition = registry.decode_ids(
                field_id,
                operator_id,
                condition_id,
            )
        except KeyError as exc:
            errors.append(str(exc))
            continue
        if not registry.is_legal(field, operator, condition, "H1", "input"):
            errors.append(f"illegal H1 tuple: {field}/{operator}/{condition}")
        key = (
            (block_id, field_id, 0, float(value), condition_id)
            if field_id == cxr_field and operator_id == cxr_obs
            else (block_id, field_id, operator_id, condition_id, 0)
        )
        if previous_key is not None and key < previous_key:
            errors.append("input events are not in canonical order")
        previous_key = key
        if index < len(counts) and (
            isinstance(counts[index], bool)
            or not isinstance(counts[index], int)
            or counts[index] < 0
        ):
            errors.append(f"invalid input_source_count[{index}]")
    if sample.get("content_hash") != content_hash(sample):
        errors.append("content_hash mismatch")
    reference = sample.get("source_reference") or {}
    if not _is_sha256(reference.get("base_content_hash")):
        errors.append("base content reference is invalid")
    if not _is_sha256(reference.get("target_content_hash")):
        errors.append("target content reference is invalid")
    if (sample.get("target_reference") or {}).get("sample_id") != sample.get("sample_id"):
        errors.append("target sidecar sample_id differs from the H1 sample")
    target_reference = sample.get("target_reference") or {}
    if target_reference.get("target_content_hash") != reference.get("target_content_hash"):
        errors.append("target content reference differs between source and target metadata")
    if not target_reference.get("target_shard_key"):
        errors.append("target shard reference is absent")
    target_line_index = target_reference.get("target_line_index")
    if (
        isinstance(target_line_index, bool)
        or not isinstance(target_line_index, int)
        or target_line_index < 0
    ):
        errors.append("target line reference is invalid")
    return errors


def assert_valid_h1_sample(sample: dict[str, Any], registry: EventRegistry) -> None:
    errors = validate_h1_sample(sample, registry)
    if errors:
        raise ValueError(f"Invalid GRU-D H1 sample {sample.get('sample_id')}: {errors}")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "GRUDH1SampleBuilder",
    "TUPLE_ORDER",
    "assert_valid_h1_sample",
    "content_hash",
    "validate_h1_sample",
]
