from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contract import EventTemplateRegistry, SupervisionContract, TargetLayout
from .normalization import RobustNormalizer


DURATION_KIND_IDS = {
    None: 0,
    "point_binomial": 1,
    "interval_zoib": 2,
    "respiratory_grouped": 3,
}
COMPOSE_RULE_IDS = {
    "last": 1,
    "min": 2,
    "max": 3,
    "weighted_mean": 4,
    "block_weighted_mean": 5,
    "sum": 6,
}


class MultiresEventCollator:
    def __init__(
        self,
        *,
        supervision: SupervisionContract,
        templates: EventTemplateRegistry,
        target_layout: TargetLayout,
        normalization: RobustNormalizer,
    ) -> None:
        if target_layout.active_query_count != 986:
            raise ValueError("baseline collator requires exactly 986 active direct queries")
        self.supervision = supervision
        self.templates = templates
        self.target_layout = target_layout
        self.normalization = normalization

    def __call__(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not records:
            raise ValueError("multires event batch is empty")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is a train extra
            raise RuntimeError("MultiresEventCollator requires the train extra with torch") from exc

        prepared = [self._prepare_record(record) for record in records]
        max_events = max(len(item["event_field_ids"]) for item in prepared)
        max_blocks = max(len(item["block_role_ids"]) for item in prepared)

        batch: dict[str, Any] = {
            "event_field_ids": torch.tensor(
                _pad([item["event_field_ids"] for item in prepared], max_events, 0),
                dtype=torch.long,
            ),
            "event_operator_ids": torch.tensor(
                _pad([item["event_operator_ids"] for item in prepared], max_events, 0),
                dtype=torch.long,
            ),
            "event_condition_ids": torch.tensor(
                _pad([item["event_condition_ids"] for item in prepared], max_events, 0),
                dtype=torch.long,
            ),
            "event_values": torch.tensor(
                _pad([item["event_values"] for item in prepared], max_events, 0.0),
                dtype=torch.float32,
            ),
            "event_value_mask": torch.tensor(
                _pad([item["event_value_mask"] for item in prepared], max_events, False),
                dtype=torch.bool,
            ),
            "event_study_slot_ids": torch.tensor(
                _pad([item["event_study_slot_ids"] for item in prepared], max_events, 0),
                dtype=torch.long,
            ),
            "event_block_index": torch.tensor(
                _pad([item["event_block_index"] for item in prepared], max_events, 0),
                dtype=torch.long,
            ),
            "event_mask": torch.tensor(
                _pad([[True] * len(item["event_field_ids"]) for item in prepared], max_events, False),
                dtype=torch.bool,
            ),
            "block_role_ids": torch.tensor(
                _pad([item["block_role_ids"] for item in prepared], max_blocks, 0),
                dtype=torch.long,
            ),
            "block_resolution_ids": torch.tensor(
                _pad([item["block_resolution_ids"] for item in prepared], max_blocks, 0),
                dtype=torch.long,
            ),
            "block_relative_start": torch.tensor(
                _pad([item["block_relative_start"] for item in prepared], max_blocks, 0.0),
                dtype=torch.float32,
            ),
            "block_relative_end": torch.tensor(
                _pad([item["block_relative_end"] for item in prepared], max_blocks, 0.0),
                dtype=torch.float32,
            ),
            "block_span": torch.tensor(
                _pad([item["block_span"] for item in prepared], max_blocks, 0.0),
                dtype=torch.float32,
            ),
            "block_mask": torch.tensor(
                _pad([[True] * len(item["block_role_ids"]) for item in prepared], max_blocks, False),
                dtype=torch.bool,
            ),
            "static_numeric": torch.tensor(
                [item["static_numeric"] for item in prepared], dtype=torch.float32
            ),
            "static_numeric_mask": torch.tensor(
                [item["static_numeric_mask"] for item in prepared], dtype=torch.bool
            ),
            "static_categorical": torch.tensor(
                [item["static_categorical"] for item in prepared], dtype=torch.long
            ),
            "static_categorical_mask": torch.tensor(
                [item["static_categorical_mask"] for item in prepared], dtype=torch.bool
            ),
            "target_values": torch.tensor(
                [item["target_values"] for item in prepared], dtype=torch.float32
            ),
            "target_raw_values": torch.tensor(
                [item["target_raw_values"] for item in prepared], dtype=torch.float32
            ),
            "target_mask": torch.tensor(
                [item["target_mask"] for item in prepared], dtype=torch.bool
            ),
            "f24_target_raw_values": torch.tensor(
                [item["f24_target_raw_values"] for item in prepared], dtype=torch.float32
            ),
            "f24_target_mask": torch.tensor(
                [item["f24_target_mask"] for item in prepared], dtype=torch.bool
            ),
            "prediction_hour": torch.tensor(
                [int(item["prediction_hour"]) for item in prepared], dtype=torch.long
            ),
            "sample_id": [str(item["sample_id"]) for item in prepared],
            "subject_id": [str(item["subject_id"]) for item in prepared],
        }
        batch.update(self._query_tensors(torch, len(prepared)))
        batch["operator_ids"] = batch["event_operator_ids"]
        batch["condition_ids"] = batch["event_condition_ids"]
        batch["values"] = batch["event_values"]
        batch["block_index"] = batch["event_block_index"]
        batch["resolution_ids"] = batch["block_resolution_ids"]
        batch["relative_start"] = batch["block_relative_start"]
        batch["relative_end"] = batch["block_relative_end"]
        batch["span"] = batch["block_span"]
        return batch

    def _prepare_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        self.supervision.assert_record_layout(record, self.target_layout)
        input_blocks = sorted(
            (item for item in record["block_table"] if item["side"] == "input"),
            key=lambda item: int(item["block_id"]),
        )
        block_position = {
            int(block["block_id"]): index for index, block in enumerate(input_blocks)
        }
        blocks_by_id = {int(item["block_id"]): item for item in record["block_table"]}
        event_field_ids: list[int] = []
        event_operator_ids: list[int] = []
        event_condition_ids: list[int] = []
        event_values: list[float] = []
        event_value_mask: list[bool] = []
        event_study_slot_ids: list[int] = []
        event_block_index: list[int] = []
        for event in record["input_events"]:
            field_id, operator_id, condition_id, raw_value, block_id = event
            if int(field_id) in self.supervision.excluded_input_field_ids:
                raise ValueError("gcs_verbal field_id=9 reached the collator")
            block = blocks_by_id[int(block_id)]
            if block.get("side") != "input":
                raise ValueError("input event references a target block")
            template = self.templates.get(int(field_id), int(operator_id), int(condition_id))
            study_slot_id = 0
            if template.value_type == "study_slot":
                study_slot_id = int(raw_value)
                if not 1 <= study_slot_id <= 8:
                    raise ValueError(
                        f"CXR study_slot must be in 1..8, got {study_slot_id}"
                    )
                transformed, observed = 0.0, False
            else:
                transformed, observed = self.normalization.transform_event(
                    raw_value,
                    template=template,
                    resolution=str(block["resolution"]),
                    span_hours=float(block["span_hours"]),
                )
            event_field_ids.append(int(field_id))
            event_operator_ids.append(int(operator_id))
            event_condition_ids.append(int(condition_id))
            event_values.append(transformed)
            event_value_mask.append(observed)
            event_study_slot_ids.append(study_slot_id)
            event_block_index.append(block_position[int(block_id)])

        static = record.get("static") or {}
        static_numeric: list[float] = []
        static_numeric_mask: list[bool] = []
        for field in self.supervision.static_numeric_fields:
            transformed, observed = self.normalization.transform_static(field, static.get(field))
            static_numeric.append(transformed)
            static_numeric_mask.append(observed)
        static_categorical: list[int] = []
        static_categorical_mask: list[bool] = []
        for field in self.supervision.static_categorical_fields:
            value = static.get(field)
            vocabulary = self.supervision.static_category_ids[field]
            category_id = vocabulary.get(str(value), self.supervision.static_unknown_id)
            if value is not None and category_id == self.supervision.static_unknown_id:
                raise ValueError(f"static field {field} has unregistered category {value}")
            static_categorical.append(category_id)
            static_categorical_mask.append(category_id != self.supervision.static_unknown_id)

        target_values: list[float] = []
        target_raw_values: list[float] = []
        target_mask: list[bool] = []
        for slot in self.target_layout.queries:
            event = record["target_events"][slot.source_index]
            raw_value = event[3]
            observed = int(record["target_mask"][slot.source_index]) == 1
            template = self.templates.get(slot.field_id, slot.operator_id, slot.condition_id)
            if observed and raw_value is None:
                raise ValueError(f"observed target {slot.slot_id} has a null value")
            transformed, transform_observed = self.normalization.transform_event(
                raw_value if observed else None,
                template=template,
                resolution=slot.resolution,
                span_hours=slot.span_hours,
                loss_family=slot.loss_family,
                for_target=True,
            )
            target_values.append(transformed)
            target_raw_values.append(float(raw_value) if observed else 0.0)
            target_mask.append(observed and transform_observed)

        f24_target_raw_values: list[float] = []
        f24_target_mask: list[bool] = []
        for canonical_index in self.target_layout.derived_primary_f24_indices:
            event = record["target_events"][canonical_index]
            observed = int(record["target_mask"][canonical_index]) == 1
            if observed and event[3] is None:
                raise ValueError("observed derived F24 target has a null value")
            f24_target_raw_values.append(float(event[3]) if observed else 0.0)
            f24_target_mask.append(observed)

        return {
            "event_field_ids": event_field_ids,
            "event_operator_ids": event_operator_ids,
            "event_condition_ids": event_condition_ids,
            "event_values": event_values,
            "event_value_mask": event_value_mask,
            "event_study_slot_ids": event_study_slot_ids,
            "event_block_index": event_block_index,
            "block_role_ids": [int(item["role_id"]) for item in input_blocks],
            "block_resolution_ids": [int(item["resolution_id"]) for item in input_blocks],
            "block_relative_start": [float(item["relative_start_hour"]) for item in input_blocks],
            "block_relative_end": [float(item["relative_end_hour"]) for item in input_blocks],
            "block_span": [float(item["span_hours"]) for item in input_blocks],
            "static_numeric": static_numeric,
            "static_numeric_mask": static_numeric_mask,
            "static_categorical": static_categorical,
            "static_categorical_mask": static_categorical_mask,
            "target_values": target_values,
            "target_raw_values": target_raw_values,
            "target_mask": target_mask,
            "f24_target_raw_values": f24_target_raw_values,
            "f24_target_mask": f24_target_mask,
            "sample_id": record["sample_id"],
            "subject_id": record["subject_id"],
            "prediction_hour": record["prediction_hour"],
        }

    def _query_tensors(self, torch: Any, batch_size: int) -> dict[str, Any]:
        queries = self.target_layout.queries
        query_position = {
            slot.source_index: index for index, slot in enumerate(queries)
        }
        f24_indices = self.target_layout.derived_primary_f24_indices
        values = {
            "query_source_indices": torch.tensor(
                [slot.source_index for slot in queries], dtype=torch.long
            ),
            "query_field_ids": torch.tensor([slot.field_id for slot in queries], dtype=torch.long),
            "query_operator_ids": torch.tensor(
                [slot.operator_id for slot in queries], dtype=torch.long
            ),
            "query_condition_ids": torch.tensor(
                [slot.condition_id for slot in queries], dtype=torch.long
            ),
            "query_resolution_ids": torch.tensor(
                [slot.resolution_id for slot in queries], dtype=torch.long
            ),
            "query_time_index": torch.tensor(
                [slot.time_index for slot in queries], dtype=torch.long
            ),
            "query_span": torch.tensor([slot.span_hours for slot in queries], dtype=torch.float32),
            "target_loss_family_ids": torch.tensor(
                [slot.loss_family_id for slot in queries], dtype=torch.long
            ),
            "target_semantic_component_ids": torch.tensor(
                [slot.semantic_component_id for slot in queries], dtype=torch.long
            ),
            "target_duration_kind_ids": torch.tensor(
                [DURATION_KIND_IDS[slot.duration_kind] for slot in queries], dtype=torch.long
            ),
            "target_coverage_query_position": torch.tensor(
                [slot.coverage_query_position for slot in queries], dtype=torch.long
            ),
            "target_ordinal_classes": torch.tensor(
                [slot.ordinal_classes for slot in queries], dtype=torch.long
            ),
            "f24_source_indices": torch.tensor(f24_indices, dtype=torch.long),
            "f24_m4_query_positions": torch.tensor(
                [
                    [query_position[index] for index in self.target_layout.f24_to_m4_indices[f24]]
                    for f24 in f24_indices
                ],
                dtype=torch.long,
            ),
            "f24_compose_rule_ids": torch.tensor(
                [COMPOSE_RULE_IDS[self.target_layout.slots[index].compose_rule] for index in f24_indices],
                dtype=torch.long,
            ),
            "f24_compose_rules": [
                self.target_layout.slots[index].compose_rule for index in f24_indices
            ],
        }
        query_keys = {
            "query_source_indices", "query_field_ids", "query_operator_ids",
            "query_condition_ids", "query_resolution_ids", "query_time_index", "query_span",
        }
        for key in query_keys:
            values[key] = values[key].unsqueeze(0).expand(batch_size, -1).clone()
        values["query_mask"] = torch.ones(
            (batch_size, len(queries)), dtype=torch.bool
        )
        return values


def _pad(rows: Sequence[Sequence[Any]], width: int, value: Any) -> list[list[Any]]:
    return [list(row) + [value] * (width - len(row)) for row in rows]
