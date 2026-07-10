from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import re
from typing import Any, Iterable


MAIN_RECORD_SCHEMA = "standard_textual_v1_main_record_v2"
MAIN_ROUTE = "main_hour_adapter_structured_heads"
STATE_TOKEN = "<STATE>"
HOUR_VALUE_ORDER = ("hr", "sbp", "dbp", "map", "rr", "temp", "spo2")
HOUR_SPECIAL_TOKENS = tuple(f"<H-{index:02d}>" for index in range(23, 0, -1)) + ("<H0>",)
HOUR_TOKENIZATION_HOUR = "hour"
HOUR_TOKENIZATION_FIELD_HOUR = "field_hour"
HOUR_TOKENIZATION_MODES = (HOUR_TOKENIZATION_HOUR, HOUR_TOKENIZATION_FIELD_HOUR)
TARGET_DOMAINS = ("shock", "resp", "renal", "heme", "tx")
FORBIDDEN_HOUR_TEXT = ("hr=", "sbp=", "dbp=", "map=", "rr=", "temp=", "spo2=", "vent_on=")
FORBIDDEN_LEGACY_TEXT = ("fio2", "fio2_max", "high_support", "very_high_support")
HOUR_BLOCK_PATTERN = re.compile(r"HOUR len=(?P<length>\d+):(?P<body>.*?)<STATE>", re.DOTALL)
HOUR_TOKEN_PATTERN = re.compile(r"<H(?:-\d{2}|0)>")


@dataclass(frozen=True)
class Next24FieldSpec:
    domain: str
    name: str
    values: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.domain}.{self.name}"

    @property
    def safe_key(self) -> str:
        return f"{self.domain}__{self.name}"

    @property
    def is_binary(self) -> bool:
        return len(self.values) == 1


NEXT24_FIELD_SPECS = (
    Next24FieldSpec("shock", "map_low_hours", ("brief", "intermittent", "prolonged", "persistent")),
    Next24FieldSpec("shock", "systolic_bp_min", ("hypotension", "low", "geriatric_low")),
    Next24FieldSpec("shock", "heart_rate_max", ("extreme_tachycardia",)),
    Next24FieldSpec("resp", "vent_hours", ("partial_window", "most_window", "full_window")),
    Next24FieldSpec("resp", "vent_course", ("first_day", "early", "prolonged")),
    Next24FieldSpec("resp", "spo2_min", ("borderline_low", "low", "critical_low")),
    Next24FieldSpec("resp", "respiratory_rate_high_hours", ("brief", "intermediate", "prolonged")),
    Next24FieldSpec("renal", "creatinine_change", ("kdigo_delta",)),
    Next24FieldSpec("renal", "creatinine_ratio", ("kdigo_ratio",)),
    Next24FieldSpec("renal", "bicarbonate_min", ("low",)),
    Next24FieldSpec("renal", "bun_creatinine_ratio", ("prerenal_pattern",)),
    Next24FieldSpec("renal", "uop_status", ("kdigo_low",)),
    Next24FieldSpec("heme", "wbc", ("high", "low")),
    Next24FieldSpec("tx", "rbc", ("present",)),
    Next24FieldSpec("tx", "surg", ("present",)),
    Next24FieldSpec("tx", "crystalloid", ("low", "moderate", "high", "very_high")),
    Next24FieldSpec("tx", "antibiotics", ("present",)),
)
NEXT24_FIELD_BY_DOMAIN = {
    domain: tuple(spec for spec in NEXT24_FIELD_SPECS if spec.domain == domain)
    for domain in TARGET_DOMAINS
}
BINARY_NEXT24_FIELD_SPECS = tuple(spec for spec in NEXT24_FIELD_SPECS if spec.is_binary)
MULTICLASS_NEXT24_FIELD_SPECS = tuple(spec for spec in NEXT24_FIELD_SPECS if not spec.is_binary)

DEFAULT_HOUR_NORMALIZATION = {
    "hr": {"mean": 90.0, "std": 30.0},
    "sbp": {"mean": 120.0, "std": 30.0},
    "dbp": {"mean": 70.0, "std": 20.0},
    "map": {"mean": 80.0, "std": 20.0},
    "rr": {"mean": 20.0, "std": 8.0},
    "temp": {"mean": 37.0, "std": 1.5},
    "spo2": {"mean": 96.0, "std": 4.0},
}


def expected_hour_placeholders(length: int) -> list[str]:
    if length < 1 or length > len(HOUR_SPECIAL_TOKENS):
        raise ValueError(f"HOUR length must be 1..24, got {length}")
    return list(HOUR_SPECIAL_TOKENS[-length:])


def hour_token_to_time_index(token: str) -> int:
    try:
        return HOUR_SPECIAL_TOKENS.index(token)
    except ValueError as exc:
        raise ValueError(f"unknown HOUR placeholder: {token}") from exc


def resolve_hour_tokenization(value: Any) -> str:
    mode = str(value or HOUR_TOKENIZATION_HOUR)
    if mode not in HOUR_TOKENIZATION_MODES:
        allowed = ", ".join(HOUR_TOKENIZATION_MODES)
        raise ValueError(f"model.hour_tokenization must be one of: {allowed}")
    return mode


def effective_input_token_count(base_token_count: int, hour_count: int, mode: str) -> int:
    resolved = resolve_hour_tokenization(mode)
    if hour_count < 1 or hour_count > len(HOUR_SPECIAL_TOKENS):
        raise ValueError(f"HOUR length must be 1..24, got {hour_count}")
    if resolved == HOUR_TOKENIZATION_FIELD_HOUR:
        return base_token_count + hour_count * (len(HOUR_VALUE_ORDER) - 1)
    return base_token_count


def validate_main_route_record(
    row: dict[str, Any],
    required_fields: Iterable[str],
    split: str | None = None,
    label: str = "record",
) -> None:
    for field in required_fields:
        if row.get(field) in ("", None):
            raise ValueError(f"{label} missing required field {field}")
    if row.get("schema") != MAIN_RECORD_SCHEMA:
        raise ValueError(f"{label} schema must be {MAIN_RECORD_SCHEMA}")
    if row.get("route") != MAIN_ROUTE:
        raise ValueError(f"{label} route must be {MAIN_ROUTE}")
    if split is not None and str(row.get("split")) != split:
        raise ValueError(f"{label} contains split={row.get('split')} inside {split} shard")
    if str(row.get("split") or "") not in {"train", "val", "test"}:
        raise ValueError(f"{label} has invalid split: {row.get('split')}")
    prediction_hour = float(row.get("prediction_hour", -1))
    if not 0 <= prediction_hour < 336:
        raise ValueError(f"{label} prediction_hour out of range: {prediction_hour}")

    input_text = str(row.get("input_text") or "")
    if not input_text.strip():
        raise ValueError(f"{label} has empty input_text")
    _validate_input_text(input_text, row, label)
    if tuple(row.get("hour_value_order") or ()) != HOUR_VALUE_ORDER:
        raise ValueError(f"{label} hour_value_order mismatch")

    placeholders = row.get("hour_placeholders")
    hour_values = row.get("hour_values")
    hour_mask = row.get("hour_mask")
    hour_vent = row.get("hour_vent")
    validate_hour_side_tensors(placeholders, hour_values, hour_mask, hour_vent, label)

    targets = row.get("targets")
    if not isinstance(targets, dict):
        raise ValueError(f"{label} targets must be an object")
    validate_next_hour_target(targets.get("next_hour"), label)
    validate_next24h_target(targets.get("next24h"), label)

    target_text = row.get("target_text")
    if target_text not in (None, ""):
        text = str(target_text)
        _reject_forbidden_text(text, f"{label}.target_text")
        if "NEXT_HOUR" not in text or "NEXT_24H" not in text:
            raise ValueError(f"{label} target_text must contain NEXT_HOUR and NEXT_24H")


def validate_hour_side_tensors(
    placeholders: Any,
    hour_values: Any,
    hour_mask: Any,
    hour_vent: Any,
    label: str,
) -> None:
    if not isinstance(placeholders, list) or not placeholders:
        raise ValueError(f"{label} hour_placeholders must be a non-empty list")
    if not isinstance(hour_values, list) or not isinstance(hour_mask, list) or not isinstance(hour_vent, list):
        raise ValueError(f"{label} hour_values/hour_mask/hour_vent must be lists")
    length = len(placeholders)
    if list(placeholders) != expected_hour_placeholders(length):
        raise ValueError(
            f"{label} hour_placeholders mismatch: expected {expected_hour_placeholders(length)}, got {placeholders}"
        )
    if len(hour_values) != length or len(hour_mask) != length or len(hour_vent) != length:
        raise ValueError(f"{label} HOUR side tensor length mismatch")

    observed = 0
    for row_index in range(length):
        values = hour_values[row_index]
        mask = hour_mask[row_index]
        vent = hour_vent[row_index]
        if not isinstance(values, list) or len(values) != len(HOUR_VALUE_ORDER):
            raise ValueError(f"{label} hour_values row {row_index} shape mismatch")
        if not isinstance(mask, list) or len(mask) != len(HOUR_VALUE_ORDER):
            raise ValueError(f"{label} hour_mask row {row_index} shape mismatch")
        if not isinstance(vent, list) or len(vent) != 1 or int(vent[0]) not in {0, 1}:
            raise ValueError(f"{label} hour_vent row {row_index} shape/value mismatch")
        for col_index, value in enumerate(values):
            mask_value = int(mask[col_index])
            if mask_value not in {0, 1}:
                raise ValueError(f"{label} hour_mask row {row_index} has non-binary value")
            if mask_value == 0 and value is not None:
                raise ValueError(f"{label} hour_values row {row_index} has value where mask=0")
            if mask_value == 1:
                _require_finite_number(value, f"{label} hour_values row {row_index}")
                observed += 1
    if observed == 0:
        raise ValueError(f"{label} HOUR side tensor has no observed vital values")


def validate_next_hour_target(target: Any, label: str) -> None:
    if not isinstance(target, dict):
        raise ValueError(f"{label} targets.next_hour must be an object")
    if target.get("label") != "NEXT_HOUR":
        raise ValueError(f"{label} target.next_hour label must be NEXT_HOUR")
    if target.get("relative_hour") != "H+1":
        raise ValueError(f"{label} target.next_hour relative_hour must be H+1")
    if tuple(target.get("value_order") or ()) != HOUR_VALUE_ORDER:
        raise ValueError(f"{label} target.next_hour value_order mismatch")
    values = target.get("values")
    mask = target.get("mask")
    row_values = target.get("hour_values")
    row_mask = target.get("hour_mask")
    row_vent = target.get("hour_vent")
    if not isinstance(values, dict) or not isinstance(mask, dict):
        raise ValueError(f"{label} target.next_hour values/mask must be objects")
    if set(values) != set(HOUR_VALUE_ORDER) or set(mask) != set(HOUR_VALUE_ORDER):
        raise ValueError(f"{label} target.next_hour values/mask fields mismatch")
    if not isinstance(row_values, list) or len(row_values) != len(HOUR_VALUE_ORDER):
        raise ValueError(f"{label} target.next_hour hour_values shape mismatch")
    if not isinstance(row_mask, list) or len(row_mask) != len(HOUR_VALUE_ORDER):
        raise ValueError(f"{label} target.next_hour hour_mask shape mismatch")
    if not isinstance(row_vent, list) or len(row_vent) != 1 or int(row_vent[0]) not in {0, 1}:
        raise ValueError(f"{label} target.next_hour hour_vent shape/value mismatch")
    if int(target.get("vent_on", 0) or 0) != int(row_vent[0]):
        raise ValueError(f"{label} target.next_hour vent_on and hour_vent disagree")
    observed_count = 0
    for index, field in enumerate(HOUR_VALUE_ORDER):
        observed = int(mask[field])
        if observed not in {0, 1}:
            raise ValueError(f"{label} target.next_hour mask for {field} is not binary")
        if int(row_mask[index]) != observed:
            raise ValueError(f"{label} target.next_hour hour_mask disagrees for {field}")
        if observed == 0 and values[field] is not None:
            raise ValueError(f"{label} target.next_hour has value for unobserved {field}")
        if row_values[index] != values[field]:
            raise ValueError(f"{label} target.next_hour hour_values disagrees for {field}")
        if observed == 1:
            _require_finite_number(values[field], f"{label} target.next_hour {field}")
            observed_count += 1
    if observed_count == 0:
        raise ValueError(f"{label} target.next_hour has no observed vital values")


def validate_next24h_target(target: Any, label: str) -> None:
    if not isinstance(target, dict):
        raise ValueError(f"{label} targets.next24h must be an object")
    if target.get("label") != "NEXT_24H":
        raise ValueError(f"{label} target.next24h label must be NEXT_24H")
    if int(target.get("len_hours", 0)) != 24:
        raise ValueError(f"{label} target.next24h len_hours must be 24")
    sections = target.get("sections")
    if not isinstance(sections, dict):
        raise ValueError(f"{label} target.next24h sections must be an object")
    invalid_domains = sorted(set(sections) - set(TARGET_DOMAINS))
    if invalid_domains:
        raise ValueError(f"{label} target.next24h contains non-target domains: {invalid_domains}")
    allowed_fields = {
        domain: {item.name: item for item in specs}
        for domain, specs in NEXT24_FIELD_BY_DOMAIN.items()
    }
    for domain, section in sections.items():
        if not isinstance(section, dict):
            raise ValueError(f"{label} target.next24h section {domain} must be an object")
        if "trend" in section or domain in {"cxr", "dq"}:
            raise ValueError(f"{label} target.next24h contains input-only target content")
        domain_specs = allowed_fields.get(domain, {})
        unknown_fields = sorted(set(section) - set(domain_specs))
        if unknown_fields:
            raise ValueError(f"{label} target.next24h section {domain} has unknown fields: {unknown_fields}")
        for field, value in section.items():
            spec = domain_specs[field]
            if str(value) not in spec.values:
                raise ValueError(f"{label} target.next24h invalid value for {domain}.{field}: {value}")


def _validate_input_text(input_text: str, row: dict[str, Any], label: str) -> None:
    _reject_forbidden_text(input_text, f"{label}.input_text")
    if input_text.count(STATE_TOKEN) != 1:
        raise ValueError(f"{label} input_text must contain exactly one {STATE_TOKEN}")
    if input_text.count("HOUR len=") != 1:
        raise ValueError(f"{label} input_text must contain exactly one HOUR len block")
    placeholders = row.get("hour_placeholders")
    if not isinstance(placeholders, list):
        raise ValueError(f"{label} hour_placeholders must be a list")
    placeholders = [str(token) for token in placeholders]
    match = HOUR_BLOCK_PATTERN.search(input_text)
    if not match:
        raise ValueError(f"{label} input_text must contain a HOUR len block before {STATE_TOKEN}")
    declared_length = int(match.group("length"))
    if declared_length != len(placeholders):
        raise ValueError(
            f"{label} HOUR len does not match hour_placeholders: "
            f"len={declared_length} placeholders={len(placeholders)}"
        )
    if placeholders != expected_hour_placeholders(declared_length):
        raise ValueError(
            f"{label} hour_placeholders mismatch: "
            f"expected {expected_hour_placeholders(declared_length)}, got {placeholders}"
        )
    hour_tokens = HOUR_TOKEN_PATTERN.findall(match.group("body"))
    if hour_tokens != placeholders:
        raise ValueError(f"{label} HOUR text tokens do not match hour_placeholders")
    non_token_text = HOUR_TOKEN_PATTERN.sub("", match.group("body")).strip()
    if non_token_text:
        raise ValueError(f"{label} HOUR block must contain only HOUR placeholders")
    all_hour_tokens = HOUR_TOKEN_PATTERN.findall(input_text)
    if all_hour_tokens != placeholders:
        raise ValueError(f"{label} input_text contains HOUR tokens outside the declared HOUR block")
    hour_block = match.group("body").lower()
    for forbidden in FORBIDDEN_HOUR_TEXT:
        if forbidden in hour_block:
            raise ValueError(f"{label} input_text HOUR block expands numeric field: {forbidden}")


def _reject_forbidden_text(text: str, label: str) -> None:
    lowered = text.lower()
    for forbidden in FORBIDDEN_LEGACY_TEXT:
        if forbidden in lowered:
            raise ValueError(f"{label} contains forbidden legacy FiO2 text: {forbidden}")


def _require_finite_number(value: Any, label: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
