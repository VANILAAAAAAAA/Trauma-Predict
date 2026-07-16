from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


RELATION_CONTRACT_VERSION = "2026-07-16-v2"
RELATION_FIELD_COUNT = 37
TARGET_FIELD_COUNT = 29
TARGET_RELATION_EDGE_COUNT = 52
INPUT_TARGET_RELATION_EDGE_COUNT = 39

FIELD_REGISTRY_FILENAME = "field_category_matrix_v1.csv"
TARGET_RELATION_FILENAME = "target_target_relation_edges_v2.csv"
INPUT_TARGET_RELATION_FILENAME = "input_target_relation_edges_v2.csv"
EVIDENCE_REGISTRY_FILENAME = "relation_evidence_registry_v2.json"

EXPECTED_RELATION_FILE_HASHES: Mapping[str, str] = {
    FIELD_REGISTRY_FILENAME: "33fd240c99b5d991f5412ebb7c6799054debaa11b884fd3b1b57334720ba32f5",
    TARGET_RELATION_FILENAME: "3232dacc91e0b772cb1667fa24ec6285f9b82dcd6566d34add5d8f38687ae77a",
    INPUT_TARGET_RELATION_FILENAME: "27facf2703231fb8bed843731c803e97d36e87e14047e2ec9254c1e5a641d91e",
    EVIDENCE_REGISTRY_FILENAME: "9e0c07c50fc582e19ed726bb4e609c1ffe51d81b67eccd521baad92eb362763e",
}

FIELD_REGISTRY_HEADER = (
    "field_id",
    "field",
    "clinical_groups",
    "data_process",
    "observation_regime",
    "emission_family",
    "v2_target_role",
    "status",
    "evidence_ids",
)
RELATION_HEADER = (
    "edge_id",
    "attention_path",
    "source_scope",
    "source_field",
    "source_channel",
    "target_scope",
    "target_field",
    "target_channel",
    "relation_type",
    "runtime_direction",
    "time_scope",
    "parameter_key",
    "evidence_id",
)

TARGET_TIME_SCOPE_IDS: Mapping[str, int] = {
    "same_future_block_registered_order": 0,
    "adjacent_future_blocks": 1,
}
INPUT_TARGET_TIME_SCOPE_IDS: Mapping[str, int] = {
    "latest_visible_history_block_to_first_future_block": 0,
    "all_visible_history_blocks_to_each_future_block": 1,
}
ALLOWED_CHANNELS = frozenset(
    {
        "value",
        "observation_and_value",
        "support",
        "finding_and_observation",
        "treatment_amount",
        "treatment_state",
    }
)

# This is the frozen r9 target process order, not numeric field-id order.  In
# particular SpO2 follows PEEP and urine output is the final registered target.
TARGET_FIELD_ORDER = (
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "mean_arterial_pressure",
    "respiratory_rate",
    "temperature",
    "gcs_eye",
    "gcs_verbal",
    "gcs_motor",
    "respiratory_support",
    "fio2",
    "peep",
    "spo2",
    "lactate",
    "base_excess",
    "bicarbonate",
    "creatinine",
    "bun",
    "wbc",
    "hemoglobin",
    "platelet_count",
    "inr",
    "sodium",
    "potassium",
    "chloride",
    "glucose",
    "vasopressor_support",
    "norepinephrine_equivalent_dose",
    "urine_output",
)


@dataclass(frozen=True)
class RelationField:
    field_id: int
    field: str
    clinical_groups: str
    data_process: str
    observation_regime: str
    emission_family: str
    v2_target_role: str
    status: str
    evidence_ids: str


@dataclass(frozen=True)
class RegisteredRelationEdge:
    edge_index: int
    edge_id: str
    attention_path: str
    source_scope: str
    source_field: str
    source_channel: str
    target_scope: str
    target_field: str
    target_channel: str
    relation_type: str
    runtime_direction: str
    time_scope: str
    time_scope_id: int
    parameter_key: str
    evidence_id: str


@dataclass(frozen=True)
class RelationEvidence:
    evidence_id: str
    evidence_class: str
    source_ids: tuple[str, ...]
    source_locators: tuple[str, ...]
    supports: str


@dataclass(frozen=True)
class MultiresEventV2RelationContract:
    """Hash-bound, model-global relation contract.

    Each CSV row is one independently parameterized directed edge.  The first
    tensor axis therefore indexes registered edges/parameter keys, not a
    relation-type group.  Matrices use ``[edge, query=target, key=source]``.
    """

    config_dir: Path
    version: str
    file_hashes: Mapping[str, str]
    bundle_hash: str
    fields: tuple[RelationField, ...]
    history_fields: tuple[str, ...]
    target_fields: tuple[str, ...]
    target_edges: tuple[RegisteredRelationEdge, ...]
    input_target_edges: tuple[RegisteredRelationEdge, ...]
    evidence_registry: tuple[RelationEvidence, ...]
    target_relation_adjacency: tuple[tuple[tuple[int, ...], ...], ...]
    target_time_scope_ids: tuple[int, ...]
    input_target_relation_adjacency: tuple[tuple[tuple[int, ...], ...], ...]
    input_target_time_scope_ids: tuple[int, ...]
    target_parameter_keys: tuple[str, ...]
    input_target_parameter_keys: tuple[str, ...]

    @classmethod
    def from_default_config(cls) -> "MultiresEventV2RelationContract":
        repo_root = Path(__file__).resolve().parents[4]
        return cls.from_config_dir(repo_root / "configs/contracts/multires_event_v2")

    @classmethod
    def from_config_dir(cls, config_dir: str | Path) -> "MultiresEventV2RelationContract":
        root = Path(config_dir).resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        observed_files = {path.name for path in root.iterdir() if path.is_file()}
        expected_files = set(EXPECTED_RELATION_FILE_HASHES)
        if observed_files != expected_files:
            raise ValueError(
                "relation V2 config directory must contain exactly the frozen four-file bundle: "
                f"missing={sorted(expected_files - observed_files)}, "
                f"extra={sorted(observed_files - expected_files)}"
            )
        observed_hashes = {
            filename: _sha256_file(root / filename)
            for filename in EXPECTED_RELATION_FILE_HASHES
        }
        if observed_hashes != dict(EXPECTED_RELATION_FILE_HASHES):
            mismatches = {
                filename: {
                    "observed": observed_hashes[filename],
                    "expected": expected,
                }
                for filename, expected in EXPECTED_RELATION_FILE_HASHES.items()
                if observed_hashes[filename] != expected
            }
            raise ValueError(f"relation V2 file hash mismatch: {mismatches}")

        fields = _read_fields(root / FIELD_REGISTRY_FILENAME)
        history_fields = tuple(field.field for field in fields)
        if len(history_fields) != RELATION_FIELD_COUNT:
            raise ValueError(f"relation V2 requires exactly {RELATION_FIELD_COUNT} history fields")
        if not set(TARGET_FIELD_ORDER).issubset(history_fields):
            raise ValueError("relation V2 target field order is not covered by the 37-field registry")

        target_rows = _read_relation_rows(
            root / TARGET_RELATION_FILENAME,
            expected_count=TARGET_RELATION_EDGE_COUNT,
        )
        input_rows = _read_relation_rows(
            root / INPUT_TARGET_RELATION_FILENAME,
            expected_count=INPUT_TARGET_RELATION_EDGE_COUNT,
        )
        _validate_unique_relation_identities(target_rows, input_rows)
        target_edges = _compile_target_edges(target_rows, history_fields)
        input_edges = _compile_input_target_edges(input_rows, history_fields)
        evidence_registry = _read_evidence_registry(root / EVIDENCE_REGISTRY_FILENAME)
        _validate_evidence_resolution(target_edges, input_edges, evidence_registry)

        target_adjacency = _one_hot_adjacency(
            target_edges,
            source_fields=TARGET_FIELD_ORDER,
            target_fields=TARGET_FIELD_ORDER,
        )
        input_adjacency = _one_hot_adjacency(
            input_edges,
            source_fields=history_fields,
            target_fields=TARGET_FIELD_ORDER,
        )
        file_hashes = dict(observed_hashes)
        return cls(
            config_dir=root,
            version=RELATION_CONTRACT_VERSION,
            file_hashes=file_hashes,
            bundle_hash=_sha256_canonical_json(file_hashes),
            fields=fields,
            history_fields=history_fields,
            target_fields=TARGET_FIELD_ORDER,
            target_edges=target_edges,
            input_target_edges=input_edges,
            evidence_registry=evidence_registry,
            target_relation_adjacency=target_adjacency,
            target_time_scope_ids=tuple(edge.time_scope_id for edge in target_edges),
            input_target_relation_adjacency=input_adjacency,
            input_target_time_scope_ids=tuple(
                edge.time_scope_id for edge in input_edges
            ),
            target_parameter_keys=tuple(edge.parameter_key for edge in target_edges),
            input_target_parameter_keys=tuple(
                edge.parameter_key for edge in input_edges
            ),
        )

    def assert_target_field_order(self, fields: Sequence[str]) -> None:
        observed = tuple(str(field) for field in fields)
        if observed != self.target_fields:
            raise ValueError(
                "relation V2 target field order differs from the stochastic-process contract"
            )

    @property
    def target_source_channels(self) -> tuple[str, ...]:
        return tuple(edge.source_channel for edge in self.target_edges)

    @property
    def target_target_channels(self) -> tuple[str, ...]:
        return tuple(edge.target_channel for edge in self.target_edges)

    @property
    def input_target_source_channels(self) -> tuple[str, ...]:
        return tuple(edge.source_channel for edge in self.input_target_edges)

    @property
    def input_target_target_channels(self) -> tuple[str, ...]:
        return tuple(edge.target_channel for edge in self.input_target_edges)


def _read_fields(path: Path) -> tuple[RelationField, ...]:
    rows = _read_csv(path, FIELD_REGISTRY_HEADER)
    if len(rows) != RELATION_FIELD_COUNT:
        raise ValueError(f"field registry must contain exactly {RELATION_FIELD_COUNT} rows")
    fields: list[RelationField] = []
    for expected_id, row in enumerate(rows, start=1):
        try:
            field_id = int(row["field_id"])
        except ValueError as exc:
            raise ValueError("field registry field_id must be an integer") from exc
        if field_id != expected_id:
            raise ValueError("field registry ids must be unique and contiguous from 1 through 37")
        fields.append(RelationField(field_id=field_id, **{key: row[key] for key in FIELD_REGISTRY_HEADER[1:]}))
    names = tuple(field.field for field in fields)
    if len(set(names)) != len(names):
        raise ValueError("field registry contains duplicate field names")
    return tuple(fields)


def _read_relation_rows(
    path: Path,
    *,
    expected_count: int,
) -> tuple[Mapping[str, str], ...]:
    rows = _read_csv(path, RELATION_HEADER)
    if len(rows) != expected_count:
        raise ValueError(f"{path.name} must contain exactly {expected_count} rows")
    return rows


def _read_evidence_registry(path: Path) -> tuple[RelationEvidence, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("relation evidence registry is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "version",
        "scope",
        "evidence",
    }:
        raise ValueError("relation evidence registry top-level keys differ from contract")
    if payload["schema"] != "multires_event_relation_evidence_registry_v2":
        raise ValueError("relation evidence registry schema mismatch")
    if payload["version"] != RELATION_CONTRACT_VERSION:
        raise ValueError("relation evidence registry version mismatch")
    scope = payload["scope"]
    expected_scope_keys = {
        "claim",
        "hard_constraints",
        "signed_effects",
        "causal_claims",
        "target_target_edge_count",
        "input_target_edge_count",
        "evidence_interpretation",
    }
    if not isinstance(scope, dict) or set(scope) != expected_scope_keys:
        raise ValueError("relation evidence registry scope differs from contract")
    if (
        scope["claim"] != "soft_typed_context_only"
        or scope["hard_constraints"] is not False
        or scope["signed_effects"] is not False
        or scope["causal_claims"] is not False
        or scope["target_target_edge_count"] != TARGET_RELATION_EDGE_COUNT
        or scope["input_target_edge_count"] != INPUT_TARGET_RELATION_EDGE_COUNT
        or not isinstance(scope["evidence_interpretation"], str)
        or not scope["evidence_interpretation"]
    ):
        raise ValueError("relation evidence registry scope semantics changed")
    raw_evidence = payload["evidence"]
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise ValueError("relation evidence registry must contain evidence entries")
    entries: list[RelationEvidence] = []
    expected_entry_keys = {
        "evidence_id",
        "evidence_class",
        "source_ids",
        "source_locators",
        "supports",
    }
    for index, raw_entry in enumerate(raw_evidence):
        if not isinstance(raw_entry, dict) or set(raw_entry) != expected_entry_keys:
            raise ValueError(f"relation evidence entry {index} keys differ from contract")
        scalar_values = (
            raw_entry["evidence_id"],
            raw_entry["evidence_class"],
            raw_entry["supports"],
        )
        if any(not isinstance(value, str) or not value or value != value.strip() for value in scalar_values):
            raise ValueError(f"relation evidence entry {index} has an invalid scalar value")
        source_ids = _nonempty_string_tuple(raw_entry["source_ids"], "source_ids", index)
        source_locators = _nonempty_string_tuple(
            raw_entry["source_locators"], "source_locators", index
        )
        entries.append(
            RelationEvidence(
                evidence_id=raw_entry["evidence_id"],
                evidence_class=raw_entry["evidence_class"],
                source_ids=source_ids,
                source_locators=source_locators,
                supports=raw_entry["supports"],
            )
        )
    evidence_ids = tuple(entry.evidence_id for entry in entries)
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("relation evidence registry contains duplicate evidence_id")
    return tuple(entries)


def _nonempty_string_tuple(value: object, label: str, index: int) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"relation evidence entry {index} {label} must be a nonempty array")
    result = tuple(value)
    if any(not isinstance(item, str) or not item or item != item.strip() for item in result):
        raise ValueError(f"relation evidence entry {index} {label} contains an invalid value")
    if len(set(result)) != len(result):
        raise ValueError(f"relation evidence entry {index} {label} contains duplicates")
    return result


def _validate_evidence_resolution(
    target_edges: Sequence[RegisteredRelationEdge],
    input_edges: Sequence[RegisteredRelationEdge],
    evidence_registry: Sequence[RelationEvidence],
) -> None:
    edges = tuple(target_edges) + tuple(input_edges)
    if len(edges) != TARGET_RELATION_EDGE_COUNT + INPUT_TARGET_RELATION_EDGE_COUNT:
        raise AssertionError("relation evidence resolution did not receive all 91 edges")
    registered = {entry.evidence_id for entry in evidence_registry}
    referenced = {edge.evidence_id for edge in edges}
    missing = referenced - registered
    unused = registered - referenced
    if missing or unused:
        raise ValueError(
            "relation evidence ids must resolve exactly: "
            f"missing={sorted(missing)}, unused={sorted(unused)}"
        )


def _read_csv(path: Path, expected_header: Sequence[str]) -> tuple[Mapping[str, str], ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(expected_header):
            raise ValueError(f"{path.name} header differs from the frozen contract")
        rows = tuple(dict(row) for row in reader)
    for row_index, row in enumerate(rows):
        if set(row) != set(expected_header):
            raise ValueError(f"{path.name} row {row_index} has missing or extra columns")
        invalid = {
            key: value
            for key, value in row.items()
            if value is None or not value or value != value.strip()
        }
        if invalid:
            raise ValueError(f"{path.name} row {row_index} has empty or padded values: {invalid}")
    return rows


def _validate_unique_relation_identities(
    target_rows: Sequence[Mapping[str, str]],
    input_rows: Sequence[Mapping[str, str]],
) -> None:
    rows = tuple(target_rows) + tuple(input_rows)
    for column in ("edge_id", "parameter_key"):
        values = tuple(row[column] for row in rows)
        if len(set(values)) != len(values):
            raise ValueError(f"relation V2 contains duplicate {column}")


def _compile_target_edges(
    rows: Sequence[Mapping[str, str]],
    history_fields: Sequence[str],
) -> tuple[RegisteredRelationEdge, ...]:
    known_fields = set(history_fields)
    target_fields = set(TARGET_FIELD_ORDER)
    edges: list[RegisteredRelationEdge] = []
    for edge_index, row in enumerate(rows):
        _validate_common_relation_row(
            row,
            known_fields=known_fields,
            target_fields=target_fields,
            expected_attention_path="target_self_attention",
            expected_source_scope="future_output",
            allowed_time_scopes=TARGET_TIME_SCOPE_IDS,
        )
        edges.append(
            _edge_from_row(edge_index, row, TARGET_TIME_SCOPE_IDS[row["time_scope"]])
        )

    self_edges = tuple(edge for edge in edges if edge.relation_type == "self_transition")
    cross_edges = tuple(edge for edge in edges if edge.relation_type != "self_transition")
    if len(self_edges) != TARGET_FIELD_COUNT or len(cross_edges) != 23:
        raise ValueError("target relation table must contain 29 self and 23 cross-output edges")
    if tuple(edge.source_field for edge in self_edges) != TARGET_FIELD_ORDER:
        raise ValueError("target self-transition rows must exactly follow the 29-field order")
    for edge in self_edges:
        if (
            edge.source_field != edge.target_field
            or edge.source_channel != edge.target_channel
            or edge.time_scope != "adjacent_future_blocks"
        ):
            raise ValueError("each target self-transition must be same-field/channel at lag one")
    target_index = {field: index for index, field in enumerate(TARGET_FIELD_ORDER)}
    cross_pairs: set[tuple[str, str]] = set()
    for edge in cross_edges:
        if edge.time_scope != "same_future_block_registered_order":
            raise ValueError("cross-output relations must stay within one future block")
        if target_index[edge.source_field] >= target_index[edge.target_field]:
            raise ValueError(
                "cross-output relation source must precede target in registered generation order"
            )
        pair = (edge.source_field, edge.target_field)
        if pair in cross_pairs:
            raise ValueError("target relation table contains a duplicate directed field pair")
        cross_pairs.add(pair)
    return tuple(edges)


def _compile_input_target_edges(
    rows: Sequence[Mapping[str, str]],
    history_fields: Sequence[str],
) -> tuple[RegisteredRelationEdge, ...]:
    known_fields = set(history_fields)
    target_fields = set(TARGET_FIELD_ORDER)
    edges: list[RegisteredRelationEdge] = []
    for edge_index, row in enumerate(rows):
        _validate_common_relation_row(
            row,
            known_fields=known_fields,
            target_fields=target_fields,
            expected_attention_path="input_target_cross_attention",
            expected_source_scope="history_input",
            allowed_time_scopes=INPUT_TARGET_TIME_SCOPE_IDS,
        )
        edges.append(
            _edge_from_row(
                edge_index,
                row,
                INPUT_TARGET_TIME_SCOPE_IDS[row["time_scope"]],
            )
        )

    bridges = tuple(edge for edge in edges if edge.relation_type == "self_transition_bridge")
    clinical = tuple(edge for edge in edges if edge.relation_type != "self_transition_bridge")
    if len(bridges) != TARGET_FIELD_COUNT or len(clinical) != 10:
        raise ValueError("input-target table must contain 29 history bridges and 10 clinical edges")
    if tuple(edge.source_field for edge in bridges) != TARGET_FIELD_ORDER:
        raise ValueError("history bridge rows must exactly follow the 29-field target order")
    for edge in bridges:
        if (
            edge.source_field != edge.target_field
            or edge.source_channel != edge.target_channel
            or edge.time_scope
            != "latest_visible_history_block_to_first_future_block"
        ):
            raise ValueError(
                "history bridges must connect the same field/channel from latest input to M4_01"
            )
    for edge in clinical:
        if edge.source_field in target_fields:
            raise ValueError("the ten retained clinical input edges must originate outside outputs")
        if edge.time_scope != "all_visible_history_blocks_to_each_future_block":
            raise ValueError("clinical input edges must be visible from all history to every M4 block")
    return tuple(edges)


def _validate_common_relation_row(
    row: Mapping[str, str],
    *,
    known_fields: set[str],
    target_fields: set[str],
    expected_attention_path: str,
    expected_source_scope: str,
    allowed_time_scopes: Mapping[str, int],
) -> None:
    if row["attention_path"] != expected_attention_path:
        raise ValueError(f"relation edge {row['edge_id']} has the wrong attention path")
    if row["source_scope"] != expected_source_scope:
        raise ValueError(f"relation edge {row['edge_id']} has the wrong source scope")
    if row["target_scope"] != "future_output":
        raise ValueError(f"relation edge {row['edge_id']} must target future output")
    if row["runtime_direction"] != "source_to_target":
        raise ValueError(f"relation edge {row['edge_id']} must be directed source-to-target")
    if row["source_channel"] not in ALLOWED_CHANNELS:
        raise ValueError(f"relation edge {row['edge_id']} has an unknown source channel")
    if row["target_channel"] not in ALLOWED_CHANNELS:
        raise ValueError(f"relation edge {row['edge_id']} has an unknown target channel")
    if row["time_scope"] not in allowed_time_scopes:
        raise ValueError(f"relation edge {row['edge_id']} has an invalid time scope")
    if row["source_field"] not in known_fields:
        raise ValueError(f"relation edge {row['edge_id']} has an unknown source field")
    if row["target_field"] not in target_fields:
        raise ValueError(f"relation edge {row['edge_id']} has a non-output target field")


def _edge_from_row(
    edge_index: int,
    row: Mapping[str, str],
    time_scope_id: int,
) -> RegisteredRelationEdge:
    return RegisteredRelationEdge(
        edge_index=edge_index,
        edge_id=row["edge_id"],
        attention_path=row["attention_path"],
        source_scope=row["source_scope"],
        source_field=row["source_field"],
        source_channel=row["source_channel"],
        target_scope=row["target_scope"],
        target_field=row["target_field"],
        target_channel=row["target_channel"],
        relation_type=row["relation_type"],
        runtime_direction=row["runtime_direction"],
        time_scope=row["time_scope"],
        time_scope_id=time_scope_id,
        parameter_key=row["parameter_key"],
        evidence_id=row["evidence_id"],
    )


def _one_hot_adjacency(
    edges: Sequence[RegisteredRelationEdge],
    *,
    source_fields: Sequence[str],
    target_fields: Sequence[str],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    source_index = {field: index for index, field in enumerate(source_fields)}
    target_index = {field: index for index, field in enumerate(target_fields)}
    matrices: list[tuple[tuple[int, ...], ...]] = []
    for expected_edge_index, edge in enumerate(edges):
        if edge.edge_index != expected_edge_index:
            raise ValueError("relation edge indices must be contiguous from zero")
        matrix = [
            [0 for _ in range(len(source_fields))]
            for _ in range(len(target_fields))
        ]
        matrix[target_index[edge.target_field]][source_index[edge.source_field]] = 1
        if sum(sum(row) for row in matrix) != 1:
            raise AssertionError("each registered relation edge must compile to one matrix entry")
        matrices.append(tuple(tuple(row) for row in matrix))
    return tuple(matrices)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_canonical_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
