from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SAMPLE_SCHEMA = "multires_event_sample_v1"
SUPERVISION_SCHEMA = "trauma_predict.multires_event_supervision.v1"
NORMALIZATION_SCHEMA = "trauma_predict.multires_event_normalization.v1"
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class EventTemplate:
    field_id: int
    operator_id: int
    condition_id: int
    field: str
    operator: str
    condition: str
    source_kind: str
    value_type: str
    target_head_family: str | None
    cross_resolution_compose: str

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.field_id, self.operator_id, self.condition_id)


class EventTemplateRegistry:
    def __init__(self, templates: Iterable[EventTemplate]) -> None:
        by_key: dict[tuple[int, int, int], EventTemplate] = {}
        for template in templates:
            existing = by_key.get(template.key)
            if existing is not None and existing != template:
                raise ValueError(f"conflicting event template for IDs {template.key}")
            by_key[template.key] = template
        if not by_key:
            raise ValueError("event template registry is empty")
        self.by_key = by_key

    @classmethod
    def from_json(cls, path: str | Path) -> "EventTemplateRegistry":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "multires_event_registry_v1":
            raise ValueError("event_templates.json schema mismatch")
        values = payload.get("templates")
        if not isinstance(values, list):
            raise ValueError("event_templates.json templates must be an array")
        return cls(
            EventTemplate(
                field_id=int(item["field_id"]),
                operator_id=int(item["operator_id"]),
                condition_id=int(item["condition_id"]),
                field=str(item["field"]),
                operator=str(item["operator"]),
                condition=str(item["condition"]),
                source_kind=str(item["source_kind"]),
                value_type=str(item["value_type"]),
                target_head_family=(
                    None if item.get("target_head_family") is None
                    else str(item["target_head_family"])
                ),
                cross_resolution_compose=str(
                    item.get("cross_resolution_compose") or "not_cross_resolution"
                ),
            )
            for item in values
        )

    def get(self, field_id: int, operator_id: int, condition_id: int) -> EventTemplate:
        key = (int(field_id), int(operator_id), int(condition_id))
        try:
            return self.by_key[key]
        except KeyError as exc:
            raise ValueError(f"event tuple IDs are absent from event_templates.json: {key}") from exc


@dataclass(frozen=True)
class TargetSlot:
    canonical_index: int
    slot_id: str
    time_slot: str
    time_index: int
    resolution: str
    resolution_id: int
    relative_start_hour: int
    relative_end_hour: int
    span_hours: int
    field_id: int
    operator_id: int
    condition_id: int
    field: str
    operator: str
    condition: str
    value_type: str
    loss_family: str
    loss_family_id: int
    semantic_component: str
    semantic_component_id: int
    duration_kind: str | None
    coverage_query_position: int
    ordinal_classes: int
    task_group: str
    prediction_mode: str
    enabled_in_baseline: bool
    compose_rule: str
    reason: str | None

    @property
    def source_index(self) -> int:
        return self.canonical_index

    @property
    def span(self) -> int:
        return self.span_hours

    @property
    def primary(self) -> bool:
        return self.task_group == "primary"

    @property
    def active(self) -> bool:
        return self.enabled_in_baseline


@dataclass(frozen=True)
class TargetLayout:
    slots: tuple[TargetSlot, ...]
    canonical_layout_sha256: str
    supervision_sha256: str
    static_numeric_fields: tuple[str, ...]
    static_categorical_fields: tuple[str, ...]
    vocab_sizes: Mapping[str, int]
    active_direct_indices: tuple[int, ...]
    derived_primary_f24_indices: tuple[int, ...]
    auxiliary_direct_indices: tuple[int, ...]
    semantic_holdout_indices: tuple[int, ...]
    f24_to_m4_indices: Mapping[int, tuple[int, ...]]

    @property
    def queries(self) -> tuple[TargetSlot, ...]:
        return self.active_slots

    @property
    def active_slots(self) -> tuple[TargetSlot, ...]:
        return tuple(self.slots[index] for index in self.active_direct_indices)

    @property
    def active_query_count(self) -> int:
        return len(self.active_direct_indices)


class SupervisionContract:
    def __init__(self, payload: Mapping[str, Any], source_sha256: str) -> None:
        if payload.get("schema") != SUPERVISION_SCHEMA:
            raise ValueError(f"supervision schema must be {SUPERVISION_SCHEMA}")
        self.payload = dict(payload)
        self.source_sha256 = source_sha256
        base = _mapping(payload.get("base_registry"), "base_registry")
        target = _mapping(payload.get("target"), "target")
        input_payload = _mapping(payload.get("input"), "input")
        static = _mapping(payload.get("static"), "static")

        self.version = str(payload.get("version") or "")
        self.registry_version = str(base.get("version") or "")
        self.tuple_order = tuple(str(value) for value in base.get("tuple_order", []))
        self.padding_id = int(base.get("padding_id", -1))
        self.expected_layout_sha256 = str(base.get("canonical_target_layout_sha256") or "")
        self.expected_rows = int(target.get("canonical_rows", -1))
        self.direct_resolutions = frozenset(str(value) for value in target.get("direct_resolutions", []))
        self.derived_eval_resolutions = frozenset(
            str(value) for value in target.get("derived_eval_resolutions", [])
        )
        self.excluded_input_field_ids = frozenset(
            int(item["field_id"])
            for item in input_payload.get("excluded_fields", [])
        )
        self.auxiliary_field_ids = frozenset(
            int(item["field_id"])
            for item in target.get("auxiliary_fields", [])
        )
        self.semantic_exclusions = tuple(target.get("semantic_exclusions", []))
        self.loss_overrides = tuple(target.get("loss_family_overrides", []))
        self.expected_counts = _mapping(target.get("expected_counts"), "target.expected_counts")
        self.loss_family_ids = {
            str(key): int(value)
            for key, value in _mapping(payload.get("loss_family_ids"), "loss_family_ids").items()
        }
        components = _mapping(payload.get("semantic_component_ids"), "semantic_component_ids")
        self.semantic_component_ids = {str(key): int(value) for key, value in components.items()}
        self.static_numeric_fields = tuple(str(value) for value in static.get("numeric_fields", []))
        self.static_categorical_fields = tuple(
            str(value) for value in static.get("categorical_fields", [])
        )
        categories = _mapping(static.get("categorical_values"), "static.categorical_values")
        self.static_category_ids = {
            field: {str(value): index + 1 for index, value in enumerate(categories.get(field, []))}
            for field in self.static_categorical_fields
        }
        self.static_unknown_id = int(static.get("unknown_id", 0))
        self._validate_definition()

    @classmethod
    def from_json(cls, path: str | Path) -> "SupervisionContract":
        raw = Path(path).read_bytes()
        return cls(json.loads(raw), hashlib.sha256(raw).hexdigest())

    def _validate_definition(self) -> None:
        if self.version != "multires_event_training_target_v1":
            raise ValueError("supervision contract version mismatch")
        if self.registry_version != "2026-07-12-full-field-v1":
            raise ValueError("supervision base registry version mismatch")
        if self.tuple_order != (
            "field_id", "operator_id", "condition_id", "value", "block_id"
        ):
            raise ValueError("supervision tuple_order mismatch")
        if self.padding_id != 0:
            raise ValueError("supervision padding_id must be zero")
        if self.expected_rows != 1314:
            raise ValueError("supervision canonical target must retain all 1,314 rows")
        if self.direct_resolutions != {"H1", "M4"}:
            raise ValueError("baseline direct resolutions must be H1 and M4")
        if self.derived_eval_resolutions != {"F24"}:
            raise ValueError("F24 must be the only derived evaluation resolution")
        if self.excluded_input_field_ids != {9}:
            raise ValueError("gcs_verbal field_id=9 must be the sole model-side input exclusion")
        if len(self.auxiliary_field_ids) != 7:
            raise ValueError("supervision contract must declare seven auxiliary fields")
        required_families = {"continuous", "ordinal", "duration", "count", "binary", "nonnegative"}
        if set(self.loss_family_ids) != required_families:
            raise ValueError("loss_family_ids must cover the six frozen loss families")
        required_components = {
            "value_summary", "observation_coverage", "abnormal_duration",
            "observation_count", "support_duration", "state", "start", "amount_sum",
        }
        if set(self.semantic_component_ids) != required_components:
            raise ValueError("semantic_component_ids do not match the frozen macro-loss groups")

    def filter_input_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        validate_record_shape(record, expected_target_rows=self.expected_rows)
        events = record["input_events"]
        source_counts = record["input_source_count"]
        kept = [
            (event, source_count)
            for event, source_count in zip(events, source_counts, strict=True)
            if int(event[0]) not in self.excluded_input_field_ids
        ]
        filtered = copy.copy(dict(record))
        filtered["input_events"] = [event for event, _ in kept]
        filtered["input_source_count"] = [value for _, value in kept]
        if any(int(event[0]) in self.excluded_input_field_ids for event in filtered["input_events"]):
            raise AssertionError("excluded input field survived model-side filtering")
        return filtered

    def compile_target_layout(
        self,
        record: Mapping[str, Any],
        templates: EventTemplateRegistry,
        *,
        strict: bool = True,
    ) -> TargetLayout:
        validate_record_shape(record, expected_target_rows=self.expected_rows if strict else None)
        blocks = {int(item["block_id"]): item for item in record["block_table"]}
        hash_rows: list[list[Any]] = []
        slots: list[TargetSlot] = []
        for index, event in enumerate(record["target_events"]):
            field_id, operator_id, condition_id, _, block_id = event
            block = blocks.get(int(block_id))
            if block is None or block.get("side") != "target":
                raise ValueError(f"target event {index} references a non-target block_id={block_id}")
            template = templates.get(int(field_id), int(operator_id), int(condition_id))
            resolution = str(block["resolution"])
            relative_start = int(block["relative_start_hour"])
            relative_end = int(block["relative_end_hour"])
            span = int(block["span_hours"])
            time_slot, time_index = _time_slot(resolution, relative_start, relative_end)
            hash_rows.append([
                resolution,
                relative_start,
                relative_end,
                int(field_id),
                int(operator_id),
                int(condition_id),
            ])
            task_group, prediction_mode, enabled, reason = self._classify(
                template, resolution
            )
            loss_family = self._loss_family(template)
            component = _semantic_component(template)
            slots.append(TargetSlot(
                canonical_index=index,
                slot_id=f"{time_slot}::{template.field}::{template.operator}::{template.condition}",
                time_slot=time_slot,
                time_index=time_index,
                resolution=resolution,
                resolution_id=int(block["resolution_id"]),
                relative_start_hour=relative_start,
                relative_end_hour=relative_end,
                span_hours=span,
                field_id=int(field_id),
                operator_id=int(operator_id),
                condition_id=int(condition_id),
                field=template.field,
                operator=template.operator,
                condition=template.condition,
                value_type=template.value_type,
                loss_family=loss_family,
                loss_family_id=self.loss_family_ids[loss_family],
                semantic_component=component,
                semantic_component_id=self.semantic_component_ids[component],
                duration_kind=_duration_kind(template),
                coverage_query_position=-1,
                ordinal_classes={8: 4, 10: 6}.get(template.field_id, 0),
                task_group=task_group,
                prediction_mode=prediction_mode,
                enabled_in_baseline=enabled,
                compose_rule=template.cross_resolution_compose,
                reason=reason,
            ))

        layout_sha256 = hashlib.sha256(
            json.dumps(hash_rows, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        active = tuple(slot.canonical_index for slot in slots if slot.enabled_in_baseline)
        query_position = {canonical_index: index for index, canonical_index in enumerate(active)}
        coverage_by_key = {
            (slot.time_slot, slot.field_id): query_position[slot.canonical_index]
            for slot in slots
            if (
                slot.enabled_in_baseline
                and slot.operator == "DURATION"
                and slot.condition == "OBSERVED"
            )
        }
        slots = [
            replace(
                slot,
                coverage_query_position=coverage_by_key.get(
                    (slot.time_slot, slot.field_id), -1
                ),
            )
            if (
                slot.enabled_in_baseline
                and slot.duration_kind == "point_binomial"
                and slot.condition != "OBSERVED"
            )
            else slot
            for slot in slots
        ]
        derived = tuple(
            slot.canonical_index
            for slot in slots
            if slot.task_group == "primary" and slot.prediction_mode == "derived_eval"
        )
        auxiliary = tuple(
            slot.canonical_index
            for slot in slots
            if slot.task_group == "auxiliary" and slot.prediction_mode == "direct"
        )
        holdout = tuple(
            slot.canonical_index for slot in slots if slot.task_group == "semantic_holdout"
        )
        composition = _build_f24_composition(slots, derived)
        vocab_sizes = {
            "field": max(template.field_id for template in templates.by_key.values()) + 1,
            "operator": max(template.operator_id for template in templates.by_key.values()) + 1,
            "condition": max(template.condition_id for template in templates.by_key.values()) + 1,
            "role": max(int(item["role_id"]) for item in record["block_table"]) + 1,
            "resolution": max(
                int(item["resolution_id"]) for item in record["block_table"]
            ) + 1,
            "study_slot": 9,
            "static_categorical": max(
                (
                    max(vocabulary.values(), default=0)
                    for vocabulary in self.static_category_ids.values()
                ),
                default=0,
            ) + 1,
        }
        layout = TargetLayout(
            slots=tuple(slots),
            canonical_layout_sha256=layout_sha256,
            supervision_sha256=self.source_sha256,
            static_numeric_fields=self.static_numeric_fields,
            static_categorical_fields=self.static_categorical_fields,
            vocab_sizes=vocab_sizes,
            active_direct_indices=active,
            derived_primary_f24_indices=derived,
            auxiliary_direct_indices=auxiliary,
            semantic_holdout_indices=holdout,
            f24_to_m4_indices=composition,
        )
        if strict:
            self.validate_target_layout(layout)
        return layout

    def validate_target_layout(self, layout: TargetLayout) -> None:
        if len(layout.slots) != self.expected_rows:
            raise ValueError(
                f"canonical target row count mismatch: {len(layout.slots)} != {self.expected_rows}"
            )
        if layout.canonical_layout_sha256 != self.expected_layout_sha256:
            raise ValueError(
                "canonical target layout hash mismatch: "
                f"{layout.canonical_layout_sha256} != {self.expected_layout_sha256}"
            )
        by_resolution = Counter(slot.resolution for slot in layout.slots)
        expected_canonical = _mapping(self.expected_counts.get("canonical"), "canonical counts")
        for resolution in ("H1", "M4", "F24"):
            if by_resolution[resolution] != int(expected_canonical[resolution]):
                raise ValueError(f"canonical {resolution} target count mismatch")
        active_slots = layout.active_slots
        active_by_resolution = Counter(slot.resolution for slot in active_slots)
        expected_primary = _mapping(
            self.expected_counts.get("primary_direct"), "primary_direct counts"
        )
        if len(active_slots) != int(expected_primary["total"]):
            raise ValueError("active primary direct target count must be 986")
        for resolution in ("H1", "M4"):
            if active_by_resolution[resolution] != int(expected_primary[resolution]):
                raise ValueError(f"active primary {resolution} target count mismatch")
        if len(layout.derived_primary_f24_indices) != int(
            self.expected_counts["primary_f24_derived_eval"]
        ):
            raise ValueError("derived primary F24 target count must be 149")
        expected_aux = _mapping(self.expected_counts.get("auxiliary_direct"), "auxiliary counts")
        if len(layout.auxiliary_direct_indices) != int(expected_aux["total"]):
            raise ValueError("auxiliary direct target count must be 105")
        holdout_direct = sum(
            slot.prediction_mode == "excluded" and slot.resolution in self.direct_resolutions
            for slot in layout.slots
        )
        expected_holdout = _mapping(
            self.expected_counts.get("semantic_holdout"), "semantic_holdout counts"
        )
        if holdout_direct != int(expected_holdout["direct_total"]):
            raise ValueError("semantic holdout direct target count must be 51")
        families = Counter(slot.loss_family for slot in active_slots)
        expected_families = _mapping(
            self.expected_counts.get("active_loss_families"), "active loss family counts"
        )
        for family in self.loss_family_ids:
            if families[family] != int(expected_families[family]):
                raise ValueError(
                    f"active loss family {family} mismatch: "
                    f"{families[family]} != {expected_families[family]}"
                )
        if len(layout.f24_to_m4_indices) != len(layout.derived_primary_f24_indices):
            raise ValueError("every derived primary F24 slot must map to six M4 slots")
        expected_vocab = {
            "field": 38,
            "operator": 11,
            "condition": 34,
            "role": 7,
            "resolution": 4,
            "study_slot": 9,
            "static_categorical": 4,
        }
        if dict(layout.vocab_sizes) != expected_vocab:
            raise ValueError(
                f"compiled vocabulary sizes differ from the frozen registry: "
                f"{dict(layout.vocab_sizes)} != {expected_vocab}"
            )

    def assert_record_layout(self, record: Mapping[str, Any], layout: TargetLayout) -> None:
        blocks = {int(item["block_id"]): item for item in record["block_table"]}
        hash_rows: list[list[Any]] = []
        for event in record["target_events"]:
            block = blocks[int(event[4])]
            hash_rows.append([
                str(block["resolution"]),
                int(block["relative_start_hour"]),
                int(block["relative_end_hour"]),
                int(event[0]),
                int(event[1]),
                int(event[2]),
            ])
        observed = hashlib.sha256(
            json.dumps(hash_rows, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if observed != layout.canonical_layout_sha256:
            raise ValueError("sample target layout differs from the frozen canonical target order")

    def _classify(
        self, template: EventTemplate, resolution: str
    ) -> tuple[str, str, bool, str | None]:
        for rule in self.semantic_exclusions:
            match = _mapping(rule.get("match"), "semantic exclusion match")
            if _template_matches(template, match):
                return "semantic_holdout", "excluded", False, str(rule.get("reason") or "")
        task_group = "auxiliary" if template.field_id in self.auxiliary_field_ids else "primary"
        if resolution in self.derived_eval_resolutions:
            return task_group, "derived_eval", False, None
        if resolution not in self.direct_resolutions:
            raise ValueError(f"unsupported target resolution: {resolution}")
        return task_group, "direct", task_group == "primary", None

    def _loss_family(self, template: EventTemplate) -> str:
        for rule in self.loss_overrides:
            if _template_matches(template, _mapping(rule.get("match"), "loss override match")):
                return str(rule["loss_family"])
        mapping = {
            "continuous_regression": "continuous",
            "duration_regression": "duration",
            "count_regression": "count",
            "binary_classification": "binary",
            "nonnegative_regression": "nonnegative",
        }
        try:
            return mapping[str(template.target_head_family)]
        except KeyError as exc:
            raise ValueError(
                f"target template {template.key} has unsupported head family "
                f"{template.target_head_family}"
            ) from exc


def validate_record_shape(
    record: Mapping[str, Any], *, expected_target_rows: int | None = 1314
) -> None:
    if record.get("schema") != SAMPLE_SCHEMA:
        raise ValueError(f"sample schema must be {SAMPLE_SCHEMA}")
    if str(record.get("split")) not in SPLITS:
        raise ValueError(f"sample has invalid split: {record.get('split')}")
    registry = _mapping(record.get("registry"), "sample.registry")
    if registry.get("registry_version") != "2026-07-12-full-field-v1":
        raise ValueError("sample registry version mismatch")
    arrays = ("input_events", "input_source_count", "target_events", "target_mask", "target_source_count")
    for key in arrays:
        if not isinstance(record.get(key), list):
            raise ValueError(f"sample {key} must be an array")
    if len(record["input_events"]) != len(record["input_source_count"]):
        raise ValueError("input_events and input_source_count length mismatch")
    target_length = len(record["target_events"])
    if target_length != len(record["target_mask"]) or target_length != len(record["target_source_count"]):
        raise ValueError("target arrays must have identical lengths")
    if expected_target_rows is not None and target_length != expected_target_rows:
        raise ValueError(f"target row count must be {expected_target_rows}, got {target_length}")
    for label in ("input_events", "target_events"):
        for index, event in enumerate(record[label]):
            if not isinstance(event, list) or len(event) != 5:
                raise ValueError(f"{label}[{index}] must be a five-element array")
            if any(int(event[position]) < 1 for position in range(3)):
                raise ValueError(f"{label}[{index}] contains a non-positive registry ID")
            value = event[3]
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{label}[{index}] contains a non-finite value")


def _semantic_component(template: EventTemplate) -> str:
    if template.operator in {"OBS", "LAST", "MIN", "MAX", "MEAN"}:
        return "value_summary"
    if template.operator == "COUNT":
        return "observation_count"
    if template.operator == "DURATION":
        if template.condition == "OBSERVED":
            return "observation_coverage"
        if template.field_id in {11, 27, 34}:
            return "support_duration"
        return "abnormal_duration"
    if template.operator == "STATE":
        return "state"
    if template.operator == "START":
        return "start"
    if template.operator == "SUM":
        return "amount_sum"
    raise ValueError(f"no semantic component for target template {template.key}")


def _duration_kind(template: EventTemplate) -> str | None:
    if template.operator != "DURATION":
        return None
    if template.field_id == 11:
        return "respiratory_grouped"
    if template.source_kind == "point":
        return "point_binomial"
    return "interval_zoib"


def _build_f24_composition(
    slots: Sequence[TargetSlot], derived_indices: Sequence[int]
) -> Mapping[int, tuple[int, ...]]:
    m4_by_key: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for slot in slots:
        if (
            slot.resolution == "M4"
            and slot.task_group == "primary"
            and slot.prediction_mode == "direct"
        ):
            m4_by_key[(slot.field_id, slot.operator_id, slot.condition_id)].append(
                slot.canonical_index
            )
    result: dict[int, tuple[int, ...]] = {}
    for f24_index in derived_indices:
        slot = slots[f24_index]
        values = tuple(m4_by_key[(slot.field_id, slot.operator_id, slot.condition_id)])
        if len(values) != 6:
            raise ValueError(
                f"derived F24 slot {slot.slot_id} maps to {len(values)} M4 slots, expected 6"
            )
        result[f24_index] = values
    return result


def _time_slot(resolution: str, start: int, end: int) -> tuple[str, int]:
    if resolution == "H1" and (start, end) == (0, 1):
        return "H1_01", 0
    if resolution == "M4" and end - start == 4 and 0 <= start <= 20 and start % 4 == 0:
        index = start // 4 + 1
        return f"M4_{index:02d}", index
    if resolution == "F24" and (start, end) == (0, 24):
        return "F24_01", 7
    raise ValueError(f"invalid target block geometry: {resolution} ({start}, {end}]" )


def _template_matches(template: EventTemplate, match: Mapping[str, Any]) -> bool:
    scalar_fields = {
        "field_id": template.field_id,
        "operator_id": template.operator_id,
        "condition_id": template.condition_id,
    }
    plural_fields = {
        "field_ids": template.field_id,
        "operator_ids": template.operator_id,
        "condition_ids": template.condition_id,
    }
    for key, observed in scalar_fields.items():
        if key in match and int(match[key]) != observed:
            return False
    for key, observed in plural_fields.items():
        if key in match and observed not in {int(value) for value in match[key]}:
            return False
    return True


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value
