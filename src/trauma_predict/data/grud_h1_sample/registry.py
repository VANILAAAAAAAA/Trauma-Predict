from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Template:
    template_id: str
    field: str
    operator: str
    condition: str
    domain: str
    source_kind: str
    unit: str
    value_type: str
    aggregation: str
    input_resolutions: tuple[str, ...]
    target_resolutions: tuple[str, ...]
    input_emit: str
    valid_min: float | None = None
    valid_max: float | None = None
    quality: str | None = None
    missingness_policy: str | None = None
    cross_resolution_compose: str | None = None
    target_head_family: str | None = None
    condition_spec: dict[str, Any] | None = None

    def allows(self, resolution: str, side: str) -> bool:
        allowed = self.input_resolutions if side == "input" else self.target_resolutions
        return resolution in allowed


class EventRegistry:
    def __init__(self, payload: dict[str, Any], source_path: Path | None = None) -> None:
        self.payload = payload
        self.source_path = source_path
        self.schema = str(payload.get("schema") or "")
        if self.schema != "multires_event_registry_v1":
            raise ValueError(f"Unsupported event registry schema: {self.schema}")
        self.version = str(payload.get("version") or "")
        self.padding_id = int(payload.get("padding_id", 0))
        self.sides: dict[str, dict[str, Any]] = dict(payload.get("sides") or {})
        self.roles: dict[str, dict[str, Any]] = dict(payload.get("roles") or {})
        self.resolutions: dict[str, dict[str, Any]] = dict(payload["resolutions"])
        self.operator_definitions = dict(payload["operators"]) if isinstance(payload["operators"], dict) else {}
        self.condition_definitions = dict(payload["conditions"]) if isinstance(payload["conditions"], dict) else {}
        self.operators = set(payload["operators"])
        self.conditions = set(payload["conditions"])
        self.fields: dict[str, dict[str, Any]] = dict(payload["fields"])
        self.field_ids, self.fields_by_id = self._id_maps("field", self.fields)
        self.operator_ids, self.operators_by_id = self._id_maps("operator", self.operator_definitions)
        self.condition_ids, self.conditions_by_id = self._id_maps("condition", self.condition_definitions)
        self.static_fields: dict[str, dict[str, Any]] = dict(payload.get("static_fields") or {})
        self.forbidden_static_keys = set(payload.get("forbidden_static_keys") or ())
        self.templates = self._expand_templates()
        self.templates_by_id = {template.template_id: template for template in self.templates}
        if len(self.templates_by_id) != len(self.templates):
            raise ValueError("Duplicate template_id in event registry")
        self._validate()

    @classmethod
    def load(cls, path: Path) -> "EventRegistry":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "multires_event_registry_manifest_v1":
            return cls(payload, source_path=path)
        root = path.parent
        files = payload["authoritative_files"]
        elements = json.loads((root / files["event_elements"]).read_text(encoding="utf-8"))
        combinations = json.loads((root / files["event_combinations"]).read_text(encoding="utf-8"))
        static = json.loads((root / files["static"]).read_text(encoding="utf-8"))
        for label, child in (("event_elements", elements), ("event_combinations", combinations), ("static", static)):
            if child.get("version") != payload.get("version"):
                raise ValueError(
                    f"Registry version mismatch: manifest={payload.get('version')} "
                    f"{label}={child.get('version')}"
                )
        merged = {
            "schema": "multires_event_registry_v1",
            "version": payload["version"],
            "padding_id": elements.get("padding_id", 0),
            "sides": elements.get("sides") or {},
            "roles": elements.get("roles") or {},
            "resolutions": elements["resolutions"],
            "operators": elements["operators"],
            "conditions": elements["conditions"],
            "fields": elements["fields"],
            "templates": combinations["templates"],
            "static_fields": static["fields"],
            "forbidden_static_keys": static["forbidden_source_keys"],
        }
        return cls(merged, source_path=path)

    @staticmethod
    def make_template_id(field: str, operator: str, condition: str) -> str:
        return f"{field}::{operator}::{condition}"

    def resolution_span(self, resolution: str) -> float:
        return float(self.resolutions[resolution]["span_hours"])

    def field_id(self, field: str) -> int:
        return self.field_ids[field]

    def operator_id(self, operator: str) -> int:
        return self.operator_ids[operator]

    def condition_id(self, condition: str) -> int:
        return self.condition_ids[condition]

    def side_id(self, side: str) -> int:
        return int(self.sides[side]["id"])

    def role_id(self, role: str) -> int:
        return int(self.roles[role]["id"])

    def resolution_id(self, resolution: str) -> int:
        return int(self.resolutions[resolution]["id"])

    def decode_ids(self, field_id: int, operator_id: int, condition_id: int) -> tuple[str, str, str]:
        try:
            return (
                self.fields_by_id[int(field_id)],
                self.operators_by_id[int(operator_id)],
                self.conditions_by_id[int(condition_id)],
            )
        except KeyError as exc:
            raise KeyError(f"Unknown event element ID: {field_id}/{operator_id}/{condition_id}") from exc

    def allowed_templates(self, resolution: str, side: str) -> list[Template]:
        return [template for template in self.templates if template.allows(resolution, side)]

    def get(self, field: str, operator: str, condition: str) -> Template:
        template_id = self.make_template_id(field, operator, condition)
        try:
            return self.templates_by_id[template_id]
        except KeyError as exc:
            raise KeyError(f"Unregistered event template: {template_id}") from exc

    def is_legal(self, field: str, operator: str, condition: str, resolution: str, side: str) -> bool:
        try:
            return self.get(field, operator, condition).allows(resolution, side)
        except KeyError:
            return False

    def expanded_contract(self) -> list[dict[str, Any]]:
        return [
            {
                "template_id": template.template_id,
                "field_id": self.field_id(template.field),
                "operator_id": self.operator_id(template.operator),
                "condition_id": self.condition_id(template.condition),
                "field": template.field,
                "operator": template.operator,
                "condition": template.condition,
                "domain": template.domain,
                "source_kind": template.source_kind,
                "unit": template.unit,
                "value_type": template.value_type,
                "aggregation": template.aggregation,
                "input_resolutions": list(template.input_resolutions),
                "target_resolutions": list(template.target_resolutions),
                "input_emit": template.input_emit,
                "valid_range": [template.valid_min, template.valid_max]
                if template.valid_min is not None and template.valid_max is not None
                else None,
                "quality": template.quality,
                "missingness_policy": template.missingness_policy,
                "cross_resolution_compose": template.cross_resolution_compose,
                "target_head_family": template.target_head_family,
            }
            for template in self.templates
        ]

    def compile_static(self, source: dict[str, Any]) -> dict[str, Any]:
        if not self.static_fields:
            out = {key: value for key, value in source.items() if value is not None}
            out.pop("early48", None)
            return out
        out: dict[str, Any] = {}
        for model_key, spec in self.static_fields.items():
            source_key = str(spec["source_key"])
            value = source.get(source_key)
            if value is None or value == "":
                if spec.get("required"):
                    raise ValueError(f"Missing required STATIC field: {source_key}")
                continue
            value_type = str(spec["type"])
            if value_type in {"categorical", "binary_category"}:
                if value not in spec["values"]:
                    raise ValueError(f"Illegal STATIC value {source_key}={value!r}")
                out[model_key] = value
                continue
            if value_type != "continuous" or isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Illegal STATIC type {source_key}={value!r}")
            numeric = float(value)
            valid_range = spec.get("valid_range")
            if valid_range and not (float(valid_range[0]) <= numeric <= float(valid_range[1])):
                if spec.get("required"):
                    raise ValueError(f"STATIC value outside range {source_key}={numeric}")
                continue
            out[model_key] = numeric
        return out

    def _expand_templates(self) -> list[Template]:
        if "templates" in self.payload:
            return sorted(
                [self._template(str(rule["field"]), self.fields[str(rule["field"])], rule) for rule in self.payload["templates"]],
                key=lambda item: (
                    self.field_ids[item.field],
                    self.operator_ids[item.operator],
                    self.condition_ids[item.condition],
                ),
            )
        templates: list[Template] = []
        profiles = self.payload["profiles"]
        for field, field_meta in self.fields.items():
            profile_name = field_meta["profile"]
            for rule in profiles[profile_name]:
                templates.append(self._template(field, field_meta, rule))
        for rule in self.payload.get("extra_templates", []):
            field = str(rule["field"])
            templates.append(self._template(field, self.fields[field], rule))
        return sorted(
            templates,
            key=lambda item: (
                self.field_ids[item.field],
                self.operator_ids[item.operator],
                self.condition_ids[item.condition],
            ),
        )

    def _template(self, field: str, field_meta: dict[str, Any], rule: dict[str, Any]) -> Template:
        operator = str(rule["operator"])
        condition = str(rule["condition"])
        unit = str(rule.get("unit") or field_meta.get("unit") or "")
        valid_range = rule.get("valid_range") or field_meta.get("valid_range")
        input_resolutions = rule.get("input_resolutions", rule.get("input", []))
        target_resolutions = rule.get("target_resolutions", rule.get("target", []))
        return Template(
            template_id=str(rule.get("template_id") or self.make_template_id(field, operator, condition)),
            field=field,
            operator=operator,
            condition=condition,
            domain=str(field_meta["domain"]),
            source_kind=str(field_meta["source_kind"]),
            unit=unit,
            value_type=str(rule["value_type"]),
            aggregation=str(rule["aggregation"]),
            input_resolutions=tuple(str(value) for value in input_resolutions),
            target_resolutions=tuple(str(value) for value in target_resolutions),
            input_emit=str(rule.get("input_emit") or "observed_only"),
            valid_min=float(valid_range[0]) if valid_range else None,
            valid_max=float(valid_range[1]) if valid_range else None,
            quality=str(rule["quality"]) if rule.get("quality") else None,
            missingness_policy=str(rule["missingness_policy"]) if rule.get("missingness_policy") else None,
            cross_resolution_compose=str(rule["cross_resolution_compose"])
            if rule.get("cross_resolution_compose")
            else None,
            target_head_family=str(rule["target_head_family"]) if rule.get("target_head_family") else None,
            condition_spec=dict(self.condition_definitions.get(condition) or {}),
        )

    def _validate(self) -> None:
        self._validate_named_ids("side", self.sides)
        self._validate_named_ids("role", self.roles)
        self._validate_named_ids("resolution", self.resolutions)
        for name, meta in self.resolutions.items():
            if float(meta["span_hours"]) <= 0:
                raise ValueError(f"Resolution span must be positive: {name}")
        for template in self.templates:
            if template.field not in self.fields:
                raise ValueError(f"Unknown field: {template.field}")
            if template.operator not in self.operators:
                raise ValueError(f"Unknown operator: {template.operator}")
            if template.condition not in self.conditions:
                raise ValueError(f"Unknown condition: {template.condition}")
            for resolution in (*template.input_resolutions, *template.target_resolutions):
                if resolution not in self.resolutions:
                    raise ValueError(f"Unknown resolution {resolution} in {template.template_id}")
            if template.input_emit not in {"always", "positive_only", "observed_only", "observed_required", "known_state"}:
                raise ValueError(f"Unknown input_emit {template.input_emit} in {template.template_id}")
            if (template.valid_min is None) != (template.valid_max is None):
                raise ValueError(f"Incomplete valid range in {template.template_id}")
            if template.valid_min is not None and template.valid_min >= template.valid_max:
                raise ValueError(f"Invalid valid range in {template.template_id}")
            if template.field == "cxr" and template.target_resolutions:
                raise ValueError("CXR is input-only and cannot register target resolutions")

    def _validate_named_ids(self, namespace: str, definitions: dict[str, dict[str, Any]]) -> None:
        seen: dict[int, str] = {}
        for name, spec in definitions.items():
            identifier = int(spec.get("id", -1))
            if identifier <= self.padding_id:
                raise ValueError(f"Invalid {namespace} ID: {name}={identifier}")
            if identifier in seen:
                raise ValueError(f"Duplicate {namespace} ID {identifier}: {seen[identifier]} and {name}")
            seen[identifier] = name

    def _id_maps(
        self, namespace: str, definitions: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, int], dict[int, str]]:
        by_name: dict[str, int] = {}
        by_id: dict[int, str] = {}
        for name, spec in definitions.items():
            if "id" not in spec:
                raise ValueError(f"Missing stable {namespace} ID: {name}")
            identifier = int(spec["id"])
            if identifier <= self.padding_id:
                raise ValueError(f"{namespace} ID must be greater than padding ID: {name}={identifier}")
            if identifier in by_id:
                raise ValueError(f"Duplicate {namespace} ID {identifier}: {by_id[identifier]} and {name}")
            by_name[name] = identifier
            by_id[identifier] = name
        return by_name, by_id


def template_map(templates: Iterable[Template]) -> dict[tuple[str, str, str], Template]:
    return {(item.field, item.operator, item.condition): item for item in templates}
