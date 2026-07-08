from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trauma_predict.data.main_route_contract import (
    BINARY_NEXT24_FIELD_SPECS,
    DEFAULT_HOUR_NORMALIZATION,
    HOUR_SPECIAL_TOKENS,
    HOUR_VALUE_ORDER,
    MULTICLASS_NEXT24_FIELD_SPECS,
    NEXT24_FIELD_SPECS,
    STATE_TOKEN,
    TARGET_DOMAINS,
    hour_token_to_time_index,
    validate_main_route_record,
)


@dataclass(frozen=True)
class HourValueNormalizer:
    stats: dict[str, dict[str, float]]

    @classmethod
    def from_config(cls, value: Any) -> "HourValueNormalizer":
        stats = DEFAULT_HOUR_NORMALIZATION.copy()
        if isinstance(value, dict):
            for field, payload in value.items():
                if field not in HOUR_VALUE_ORDER or not isinstance(payload, dict):
                    continue
                mean = float(payload.get("mean", stats[field]["mean"]))
                std = float(payload.get("std", stats[field]["std"]))
                if std <= 0:
                    raise ValueError(f"value_normalization.{field}.std must be positive")
                stats[field] = {"mean": mean, "std": std}
        return cls(stats=stats)

    def normalize_row(self, values: list[Any], mask: list[Any]) -> list[float]:
        if len(values) != len(HOUR_VALUE_ORDER) or len(mask) != len(HOUR_VALUE_ORDER):
            raise ValueError("HOUR row shape mismatch")
        out: list[float] = []
        for index, field in enumerate(HOUR_VALUE_ORDER):
            if int(mask[index]) == 0:
                out.append(0.0)
                continue
            value = float(values[index])
            stats = self.stats[field]
            out.append((value - stats["mean"]) / stats["std"])
        return out

    def denormalize_row(self, values: list[float]) -> dict[str, float]:
        if len(values) != len(HOUR_VALUE_ORDER):
            raise ValueError("normalized HOUR row shape mismatch")
        out: dict[str, float] = {}
        for index, field in enumerate(HOUR_VALUE_ORDER):
            stats = self.stats[field]
            out[field] = float(values[index]) * stats["std"] + stats["mean"]
        return out


class MainRouteRecordDataset:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        if not records:
            raise ValueError("main-route dataset is empty")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class MainRouteBatchCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_input_tokens: int,
        normalizer: HourValueNormalizer,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.normalizer = normalizer
        self.pad_to_multiple_of = pad_to_multiple_of
        if getattr(tokenizer, "padding_side", "right") != "right":
            raise ValueError("main-route collator requires tokenizer.padding_side='right'")
        self.state_token_id = tokenizer.convert_tokens_to_ids(STATE_TOKEN)
        self.hour_token_ids = {
            token: tokenizer.convert_tokens_to_ids(token)
            for token in HOUR_SPECIAL_TOKENS
        }
        if self._is_unknown_token_id(self.state_token_id):
            raise ValueError(f"tokenizer does not know {STATE_TOKEN}")
        unknown_hour_tokens = [
            token for token, token_id in self.hour_token_ids.items()
            if self._is_unknown_token_id(token_id)
        ]
        if unknown_hour_tokens:
            raise ValueError(f"tokenizer does not know HOUR tokens: {unknown_hour_tokens[:5]}")

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        encoded_items: list[dict[str, Any]] = []
        hour_values: list[list[list[float]]] = []
        hour_masks: list[list[list[float]]] = []
        hour_vents: list[list[list[float]]] = []
        hour_positions: list[list[int]] = []
        hour_position_masks: list[list[int]] = []
        hour_time_indices: list[list[int]] = []
        next_hour_values: list[list[float]] = []
        next_hour_masks: list[list[float]] = []
        next_hour_vents: list[list[float]] = []
        domain_labels: list[list[float]] = []
        binary_labels: list[list[float]] = []
        multiclass_labels: list[list[int]] = []

        for record in records:
            encoded = self.tokenizer(
                str(record["input_text"]),
                add_special_tokens=True,
                truncation=False,
            )
            input_ids = list(encoded["input_ids"])
            if len(input_ids) > self.max_input_tokens:
                raise ValueError(
                    f"sample {record.get('sample_id')} has {len(input_ids)} input tokens, "
                    f"exceeding max_input_tokens={self.max_input_tokens}; refusing to truncate "
                    "because HOUR placeholders or <STATE> would become untrustworthy"
                )
            encoded_items.append(encoded)

            placeholders = [str(token) for token in record["hour_placeholders"]]
            positions = [self._single_token_position(input_ids, self.hour_token_ids[token], token, record) for token in placeholders]
            hour_positions.append(positions)
            hour_position_masks.append([1] * len(positions))
            hour_time_indices.append([hour_token_to_time_index(token) for token in placeholders])
            self._single_token_position(input_ids, self.state_token_id, STATE_TOKEN, record)

            raw_values = record["hour_values"]
            raw_masks = record["hour_mask"]
            hour_values.append([
                self.normalizer.normalize_row(list(value_row), list(mask_row))
                for value_row, mask_row in zip(raw_values, raw_masks, strict=True)
            ])
            hour_masks.append([[float(item) for item in row] for row in raw_masks])
            hour_vents.append([[float(row[0])] for row in record["hour_vent"]])

            target = record["targets"]["next_hour"]
            next_hour_values.append(
                self.normalizer.normalize_row(list(target["hour_values"]), list(target["hour_mask"]))
            )
            next_hour_masks.append([float(item) for item in target["hour_mask"]])
            next_hour_vents.append([float(target["hour_vent"][0])])

            labels = encode_next24_labels(record["targets"]["next24h"])
            domain_labels.append(labels["domains"])
            binary_labels.append(labels["binary_fields"])
            multiclass_labels.append(labels["multiclass_fields"])

        batch = self.tokenizer.pad(
            encoded_items,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        input_ids_tensor = batch["input_ids"]
        state_positions = [
            self._single_token_position(list(input_ids_tensor[index].tolist()), self.state_token_id, STATE_TOKEN, records[index])
            for index in range(len(records))
        ]

        max_hours = max(len(item) for item in hour_values)
        return {
            "input_ids": input_ids_tensor,
            "attention_mask": batch["attention_mask"],
            "hour_values": torch.tensor(_pad_3d(hour_values, max_hours, len(HOUR_VALUE_ORDER), 0.0), dtype=torch.float32),
            "hour_mask": torch.tensor(_pad_3d(hour_masks, max_hours, len(HOUR_VALUE_ORDER), 0.0), dtype=torch.float32),
            "hour_vent": torch.tensor(_pad_3d(hour_vents, max_hours, 1, 0.0), dtype=torch.float32),
            "hour_positions": torch.tensor(_pad_2d(hour_positions, max_hours, -1), dtype=torch.long),
            "hour_position_mask": torch.tensor(_pad_2d(hour_position_masks, max_hours, 0), dtype=torch.bool),
            "hour_time_indices": torch.tensor(_pad_2d(hour_time_indices, max_hours, 0), dtype=torch.long),
            "state_position": torch.tensor(state_positions, dtype=torch.long),
            "next_hour_values": torch.tensor(next_hour_values, dtype=torch.float32),
            "next_hour_mask": torch.tensor(next_hour_masks, dtype=torch.float32),
            "next_hour_vent": torch.tensor(next_hour_vents, dtype=torch.float32),
            "next24_domain_labels": torch.tensor(domain_labels, dtype=torch.float32),
            "next24_binary_labels": torch.tensor(binary_labels, dtype=torch.float32),
            "next24_multiclass_labels": torch.tensor(multiclass_labels, dtype=torch.long),
        }

    def _single_token_position(
        self,
        input_ids: list[int],
        token_id: int,
        token: str,
        record: dict[str, Any],
    ) -> int:
        positions = [index for index, value in enumerate(input_ids) if int(value) == int(token_id)]
        if len(positions) != 1:
            raise ValueError(
                f"sample {record.get('sample_id')} must contain token {token} exactly once after tokenization; "
                f"found {len(positions)}"
            )
        return positions[0]

    def _is_unknown_token_id(self, token_id: Any) -> bool:
        if token_id in (None, -1):
            return True
        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        return unk_token_id is not None and int(token_id) == int(unk_token_id)


def load_main_route_records(
    paths: list[Any],
    required_fields: list[str],
    split: str | None = None,
) -> list[dict[str, Any]]:
    from trauma_predict.data.records import read_jsonl

    records: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            validate_main_route_record(row, required_fields, split=split, label=str(path))
            records.append(row)
    if not records:
        raise ValueError("no main-route records loaded")
    return records


def encode_next24_labels(target: dict[str, Any]) -> dict[str, list[Any]]:
    sections = target.get("sections", {})
    if not isinstance(sections, dict):
        raise ValueError("NEXT_24H sections must be an object")

    domain_labels = [1.0 if _domain_has_target_fields(domain, sections.get(domain)) else 0.0 for domain in TARGET_DOMAINS]
    binary_labels: list[float] = []
    for spec in BINARY_NEXT24_FIELD_SPECS:
        section = sections.get(spec.domain)
        value = section.get(spec.name) if isinstance(section, dict) else None
        binary_labels.append(1.0 if value == spec.values[0] else 0.0)

    multiclass_labels: list[int] = []
    for spec in MULTICLASS_NEXT24_FIELD_SPECS:
        section = sections.get(spec.domain)
        value = section.get(spec.name) if isinstance(section, dict) else None
        if value is None:
            multiclass_labels.append(0)
            continue
        try:
            multiclass_labels.append(spec.values.index(str(value)) + 1)
        except ValueError as exc:
            raise ValueError(f"invalid NEXT_24H value for {spec.key}: {value}") from exc
    return {
        "domains": domain_labels,
        "binary_fields": binary_labels,
        "multiclass_fields": multiclass_labels,
    }


def decode_next24_predictions(
    domain_scores: list[float],
    binary_scores: list[float],
    multiclass_indices: list[int],
    threshold: float = 0.5,
) -> dict[str, Any]:
    sections: dict[str, dict[str, str]] = {}
    active_domains = {
        domain
        for domain, score in zip(TARGET_DOMAINS, domain_scores, strict=True)
        if float(score) >= threshold
    }
    for spec, score in zip(BINARY_NEXT24_FIELD_SPECS, binary_scores, strict=True):
        if spec.domain in active_domains and float(score) >= threshold:
            sections.setdefault(spec.domain, {})[spec.name] = spec.values[0]
    for spec, class_index in zip(MULTICLASS_NEXT24_FIELD_SPECS, multiclass_indices, strict=True):
        index = int(class_index)
        if spec.domain in active_domains and index > 0:
            sections.setdefault(spec.domain, {})[spec.name] = spec.values[index - 1]
    return {
        "label": "NEXT_24H",
        "len_hours": 24,
        "sections": sections,
    }


def _domain_has_target_fields(domain: str, section: Any) -> bool:
    if not isinstance(section, dict):
        return False
    allowed = {spec.name for spec in NEXT24_FIELD_SPECS if spec.domain == domain}
    return any(field in allowed and section[field] is not None for field in section)


def _pad_2d(rows: list[list[int]], width: int, value: int) -> list[list[int]]:
    return [row + [value] * (width - len(row)) for row in rows]


def _pad_3d(rows: list[list[list[float]]], width: int, inner_width: int, value: float) -> list[list[list[float]]]:
    padded = []
    for row in rows:
        missing = [[value] * inner_width for _ in range(width - len(row))]
        padded.append(row + missing)
    return padded
