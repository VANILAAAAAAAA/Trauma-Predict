from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from trauma_predict.data.multires_event.contract import (
    EventTemplate,
    EventTemplateRegistry,
    SupervisionContract,
)
from trauma_predict.data.multires_event.normalization import RobustNormalizer
from trauma_predict.data.multires_event_v2.collator import MultiresEventV2Collator
from trauma_predict.data.multires_event_v2.contract import MultiresEventV2Contract


H1_CHANNEL_REGISTRY_SCHEMA = "grud_h1_channel_registry_v1"


@dataclass(frozen=True)
class H1Channel:
    channel_id: int
    template: EventTemplate

    @property
    def key(self) -> tuple[int, int, int]:
        return self.template.key


class H1ChannelRegistry:
    """Closed 118-channel ordering persisted beside the H1 sidecar."""

    def __init__(self, channels: Sequence[H1Channel], *, source_sha256: str) -> None:
        ordered = tuple(sorted(channels, key=lambda item: item.channel_id))
        if tuple(item.channel_id for item in ordered) != tuple(range(118)):
            raise ValueError("H1 channel IDs must be exactly 0..117")
        keys = tuple(item.key for item in ordered)
        if len(set(keys)) != 118:
            raise ValueError("H1 channel semantic keys must be unique")
        self.channels = ordered
        self.source_sha256 = str(source_sha256)
        self.by_key = {item.key: item for item in ordered}
        self.templates = EventTemplateRegistry(item.template for item in ordered)

    @classmethod
    def from_json(cls, path: str | Path) -> "H1ChannelRegistry":
        source = Path(path)
        raw = source.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, Mapping) or payload.get("schema") != H1_CHANNEL_REGISTRY_SCHEMA:
            raise ValueError("H1 channel registry schema mismatch")
        rows = payload.get("channels")
        if not isinstance(rows, list) or len(rows) != 118:
            raise ValueError("H1 channel registry must contain exactly 118 rows")
        channels: list[H1Channel] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("H1 channel row must be an object")
            if "H1" not in row.get("input_resolutions", ()):
                raise ValueError("H1 channel row is not registered for H1 input")
            channels.append(
                H1Channel(
                    channel_id=int(row["channel_id"]),
                    template=EventTemplate(
                        field_id=int(row["field_id"]),
                        operator_id=int(row["operator_id"]),
                        condition_id=int(row["condition_id"]),
                        field=str(row["field"]),
                        operator=str(row["operator"]),
                        condition=str(row["condition"]),
                        source_kind=str(row["source_kind"]),
                        value_type=str(row["value_type"]),
                        target_head_family=(
                            None
                            if row.get("target_head_family") is None
                            else str(row["target_head_family"])
                        ),
                        cross_resolution_compose=str(
                            row.get("cross_resolution_compose")
                            or "not_cross_resolution"
                        ),
                    ),
                )
            )
        return cls(channels, source_sha256=hashlib.sha256(raw).hexdigest())

    @property
    def channel_count(self) -> int:
        return len(self.channels)


def load_frozen_h1_normalizer(
    path: str | Path,
    *,
    expected_dataset_fingerprint: str | None = None,
    expected_supervision_sha256: str | None = None,
) -> RobustNormalizer:
    """Load an already fitted train-subject normalizer; fitting is deliberately absent."""

    return RobustNormalizer.from_json(
        path,
        expected_dataset_fingerprint=expected_dataset_fingerprint,
        expected_supervision_sha256=expected_supervision_sha256,
    )


class GRUDH1V2Collator(MultiresEventV2Collator):
    """Densify H1 history while reusing the exact V2 target primitive packer."""

    def __init__(
        self,
        *,
        contract: MultiresEventV2Contract,
        supervision: SupervisionContract,
        templates: EventTemplateRegistry,
        normalization: RobustNormalizer | Any,
        channel_registry: H1ChannelRegistry,
    ) -> None:
        if channel_registry.channel_count != 118:
            raise ValueError("GRU-D H1 collator requires exactly 118 channels")
        for channel in channel_registry.channels:
            registered = templates.get(*channel.key)
            if registered != channel.template:
                raise ValueError(f"H1 channel/template contract mismatch for {channel.key}")
        self.channel_registry = channel_registry
        super().__init__(
            contract=contract,
            supervision=supervision,
            templates=templates,
            normalization=normalization,
        )

    def _validate_joined_record(self, record: Mapping[str, Any]) -> None:
        required = {
            "sample_id",
            "subject_id",
            "hadm_id",
            "stay_id",
            "prediction_hour",
            "split",
            "base_content_hash",
            "target_content_hash",
            "h1_content_hash",
            "input_record",
            "target_record",
        }
        if set(record) != required:
            raise ValueError("GRU-D H1/V2 joined record keys differ from the frozen contract")
        input_record = record["input_record"]
        target_record = record["target_record"]
        if not isinstance(input_record, Mapping) or not isinstance(target_record, Mapping):
            raise ValueError("GRU-D H1/V2 joined records must be mappings")
        if input_record.get("schema") != "grud_h1_baseline_input_sample_v1":
            raise ValueError("GRU-D input record schema mismatch")
        self.contract.validate_target_record(target_record, verify_content_hash=True)
        for key in ("sample_id", "subject_id", "hadm_id", "stay_id", "prediction_hour", "split"):
            if str(input_record.get(key)) != str(record[key]):
                raise ValueError(f"GRU-D input identity mismatch for {key}")
            if str(target_record.get(key)) != str(record[key]):
                raise ValueError(f"V2 target identity mismatch for {key}")
        if str(input_record.get("content_hash") or "") != str(record["h1_content_hash"]):
            raise ValueError("GRU-D H1 content hash differs from joined authority")

    def _collate_input(
        self,
        torch: Any,
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        lengths = [self._history_length(record["input_record"]) for record in records]
        batch_size = len(records)
        max_length = max(lengths)
        channel_count = self.channel_registry.channel_count
        values = torch.zeros((batch_size, max_length, channel_count), dtype=torch.float32)
        observed = torch.zeros((batch_size, max_length, channel_count), dtype=torch.bool)
        sequence_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)

        static_numeric_rows: list[list[float]] = []
        static_numeric_mask_rows: list[list[bool]] = []
        static_categorical_rows: list[list[int]] = []
        static_categorical_mask_rows: list[list[bool]] = []
        history_start: list[int] = []

        for batch_index, joined in enumerate(records):
            record = joined["input_record"]
            length = lengths[batch_index]
            sequence_mask[batch_index, :length] = True
            history_start.append(int(record["input_geometry"]["history_start_hour"]))
            cells: dict[tuple[int, int], list[float]] = {}
            input_events = record.get("input_events") or []
            source_counts = record.get("input_source_count") or []
            if len(input_events) != len(source_counts):
                raise ValueError("H1 input events and source counts are misaligned")
            for event in input_events:
                if not isinstance(event, list) or len(event) != 5:
                    raise ValueError("H1 input event must be a five-element tuple")
                field_id, operator_id, condition_id, raw_value, block_id = event
                channel = self.channel_registry.by_key.get(
                    (int(field_id), int(operator_id), int(condition_id))
                )
                if channel is None:
                    raise ValueError(
                        "H1 input tuple is absent from the frozen 118-channel registry: "
                        f"{(field_id, operator_id, condition_id)}"
                    )
                block_index = int(block_id)
                if not 0 <= block_index < length:
                    raise ValueError("H1 input event references a padded/invalid hour")
                number = float(raw_value)
                if not math.isfinite(number):
                    raise ValueError("H1 input event contains a non-finite value")
                cells.setdefault((block_index, channel.channel_id), []).append(number)

            for (block_index, channel_id), raw_values in cells.items():
                template = self.channel_registry.channels[channel_id].template
                if template.value_type == "study_slot":
                    raw_value = float(len(raw_values))
                else:
                    if len(raw_values) != 1:
                        raise ValueError(
                            "multiple non-CXR events occupy one H1 channel/hour"
                        )
                    raw_value = raw_values[0]
                transformed, is_observed = self.normalization.transform_event(
                    raw_value,
                    template=template,
                    resolution="H1",
                    span_hours=1.0,
                )
                if not is_observed or not math.isfinite(float(transformed)):
                    raise ValueError("persisted H1 event did not normalize to an observed value")
                values[batch_index, block_index, channel_id] = float(transformed)
                observed[batch_index, block_index, channel_id] = True

            (
                static_numeric,
                static_numeric_mask,
                static_categorical,
                static_categorical_mask,
            ) = self._prepare_static(record.get("static") or {})
            static_numeric_rows.append(static_numeric)
            static_numeric_mask_rows.append(static_numeric_mask)
            static_categorical_rows.append(static_categorical)
            static_categorical_mask_rows.append(static_categorical_mask)

        delta = self._elapsed_since_previous_observation(torch, observed, sequence_mask)
        return {
            "h1_values": values,
            "h1_observed_mask": observed,
            "h1_delta_hours": delta,
            "h1_sequence_mask": sequence_mask,
            "h1_history_start_hour": torch.tensor(history_start, dtype=torch.long),
            "h1_lengths": torch.tensor(lengths, dtype=torch.long),
            "static_numeric": torch.tensor(static_numeric_rows, dtype=torch.float32),
            "static_numeric_mask": torch.tensor(static_numeric_mask_rows, dtype=torch.bool),
            "static_categorical": torch.tensor(static_categorical_rows, dtype=torch.long),
            "static_categorical_mask": torch.tensor(
                static_categorical_mask_rows, dtype=torch.bool
            ),
            "prediction_hour": torch.tensor(
                [int(record["prediction_hour"]) for record in records], dtype=torch.long
            ),
            "sample_id": [str(record["sample_id"]) for record in records],
            "subject_id": [str(record["subject_id"]) for record in records],
        }

    def _history_length(self, record: Mapping[str, Any]) -> int:
        geometry = record.get("input_geometry")
        if not isinstance(geometry, Mapping):
            raise ValueError("H1 input record lacks input_geometry")
        length = int(geometry.get("block_count", -1))
        start = int(geometry.get("history_start_hour", -1))
        end = int(geometry.get("history_end_hour", -1))
        prediction_hour = int(record.get("prediction_hour", -1))
        if (
            geometry.get("resolution") != "H1"
            or not 1 <= length <= 312
            or end != prediction_hour
            or end - start != length
        ):
            raise ValueError("H1 input geometry differs from the frozen hourly contract")
        return length

    def _prepare_static(
        self,
        static: Mapping[str, Any],
    ) -> tuple[list[float], list[bool], list[int], list[bool]]:
        numeric: list[float] = []
        numeric_mask: list[bool] = []
        for field in self.supervision.static_numeric_fields:
            transformed, observed = self.normalization.transform_static(field, static.get(field))
            numeric.append(float(transformed))
            numeric_mask.append(bool(observed))
        categorical: list[int] = []
        categorical_mask: list[bool] = []
        for field in self.supervision.static_categorical_fields:
            value = static.get(field)
            category_id = self.supervision.static_category_ids[field].get(
                str(value), self.supervision.static_unknown_id
            )
            if value is not None and category_id == self.supervision.static_unknown_id:
                raise ValueError(f"static field {field} has an unregistered category {value}")
            categorical.append(int(category_id))
            categorical_mask.append(category_id != self.supervision.static_unknown_id)
        return numeric, numeric_mask, categorical, categorical_mask

    @staticmethod
    def _elapsed_since_previous_observation(
        torch: Any,
        observed: Any,
        sequence_mask: Any,
    ) -> Any:
        batch_size, time_steps, channel_count = observed.shape
        if sequence_mask.shape != (batch_size, time_steps):
            raise ValueError("H1 sequence mask shape mismatch")
        delta = torch.zeros((batch_size, time_steps, channel_count), dtype=torch.float32)
        if time_steps < 1:
            raise ValueError("H1 batch contains no history hours")
        delta[:, 0] = sequence_mask[:, 0].unsqueeze(-1).float()
        for time_index in range(1, time_steps):
            elapsed = 1.0 + (~observed[:, time_index - 1]).float() * delta[:, time_index - 1]
            delta[:, time_index] = torch.where(
                sequence_mask[:, time_index].unsqueeze(-1),
                elapsed,
                torch.zeros_like(elapsed),
            )
        return delta


__all__ = [
    "GRUDH1V2Collator",
    "H1Channel",
    "H1ChannelRegistry",
    "load_frozen_h1_normalizer",
]
