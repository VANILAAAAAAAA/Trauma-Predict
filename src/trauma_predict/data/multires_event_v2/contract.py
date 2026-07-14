from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


TARGET_SCHEMA = "multires_event_m4_target_sidecar_v2"
DATASET_MANIFEST_SCHEMA = "multires_event_m4_target_dataset_manifest_v2"
PROCESS_REGISTRY_SCHEMA = "multires_event_target_process_registry_v2"
PROCESS_REGISTRY_VERSION = "2026-07-14-r9"
EMISSION_REGISTRY_SCHEMA = "multires_event_target_emission_registry_v2"
EMISSION_REGISTRY_VERSION = "2026-07-14-r9"
ENABLED_CORE_LAYOUTS_SHA256 = "006d99e3a44f0250755c6235f5f486459fb80199a1341aec41238a3e60d870b1"
EXPECTED_RETAINED_AUXILIARY_WITHOUT_INITIAL_HEADS = (
    "treatment_amount_onset_joint",
    "antibiotic_joint_process",
    "surgery_six_block_event_sequence",
)
PROJECTION_REGISTRY_SCHEMA = "multires_event_target_projection_registry_v2"
PROJECTION_REGISTRY_VERSION = "2026-07-14-r9"

TARGET_ARITHMETIC_SCHEMA = "multires_event_m4_target_arithmetic_canonicalization_v2"
TARGET_ARITHMETIC_POLICY = "exact_support_arithmetic_r9"
RESPIRATORY_ARITHMETIC_TYPE = "respiratory_negative_uncovered_roundoff_to_zero"
TARGET_ARITHMETIC_TYPES = (
    "dense_zero_range_mean_to_min",
    "dense_positive_range_mean_lower_outside_one_ulp",
    "dense_positive_range_mean_lower_inside_one_ulp",
    "dense_positive_range_mean_upper_inside_one_ulp",
    "dense_positive_range_mean_upper_outside_one_ulp",
    RESPIRATORY_ARITHMETIC_TYPE,
)
EXPECTED_TARGET_ARITHMETIC_BY_TYPE = {
    "dense_zero_range_mean_to_min": 774,
    "dense_positive_range_mean_lower_outside_one_ulp": 14,
    "dense_positive_range_mean_lower_inside_one_ulp": 19,
    "dense_positive_range_mean_upper_inside_one_ulp": 11,
    "dense_positive_range_mean_upper_outside_one_ulp": 6,
    RESPIRATORY_ARITHMETIC_TYPE: 8,
}
EXPECTED_TARGET_ARITHMETIC_RECORDS = 832
EXPECTED_TARGET_ARITHMETIC_SAMPLES = 812

BLOCK_IDS = tuple(f"M4_{index:02d}" for index in range(1, 7))
BLOCK_BOUNDS = tuple((4 * index, 4 * (index + 1)) for index in range(6))
EXPECTED_CORE_FIELD_COUNT = 29
EXPECTED_ENABLED_FACTOR_COUNT = 414
DETERMINISTIC_PROJECTIONS_PER_BLOCK = 155
PHYSICAL_ARITHMETIC_ATOL = 1e-12

CONTRACT_FILES = {
    "process": "target_process_registry_v2.json",
    "emission": "target_emission_registry_v2.json",
    "projection": "target_projection_registry_v2.json",
    "category": "field_category_matrix_v1.csv",
    "relation": "field_relation_edges_v1.csv",
    "element_extension": "event_element_extension_v2.json",
    "sidecar_schema": "target_sidecar_schema_v2.json",
}

CORE_FIELD_GROUPS = (
    "dense_continuous",
    "gcs_ordinal_enabled",
    "gcs_verbal_reaggregated",
    "intermittent_labs",
    "respiratory_support",
    "vasopressor_support",
    "ned",
    "uop",
)


@dataclass(frozen=True)
class CoreRelationEdge:
    """One canonical registry row whose endpoints are both V2 core targets."""

    edge_id: str
    source_field: str
    target_field: str
    relation_type: str
    direction: str
    lag_blocks: int


@dataclass(frozen=True)
class MultiresEventV2Contract:
    dataset_root: Path
    manifest: Mapping[str, Any]
    process_registry: Mapping[str, Any]
    emission_registry: Mapping[str, Any]
    projection_registry: Mapping[str, Any]
    contract_hashes: Mapping[str, str]
    contract_bundle_hash: str
    dense_fields: tuple[str, ...]
    ordinal_fields: tuple[str, ...]
    verbal_field: str
    lab_fields: tuple[str, ...]
    respiratory_field: str
    vasopressor_field: str
    ned_field: str
    uop_field: str
    dense_abnormal_conditions: Mapping[str, tuple[str, ...]]
    respiratory_modalities: tuple[str, ...]
    vasopressor_agents: tuple[str, ...]
    ordinal_max: Mapping[str, int]
    registered_core_fields: tuple[str, ...]
    registered_core_field_ids: tuple[int, ...]
    relation_types: tuple[str, ...]
    relation_type_lags: tuple[int, ...]
    relation_adjacency: tuple[tuple[tuple[int, ...], ...], ...]
    active_core_relation_edges: tuple[CoreRelationEdge, ...]
    relation_total_edges: int
    relation_active_core_edges: int
    relation_deferred_edges: int

    @classmethod
    def from_dataset_root(
        cls,
        dataset_root: str | Path,
        *,
        verify_contract_hashes: bool = True,
    ) -> "MultiresEventV2Contract":
        root = Path(dataset_root).resolve()
        manifest = _read_json(root / "dataset_manifest.json")
        contract_dir = root / "contracts"
        declared_hashes = {
            str(key): str(value)
            for key, value in _mapping(
                manifest.get("contract_hashes"), "dataset_manifest.contract_hashes"
            ).items()
        }
        if set(declared_hashes) != set(CONTRACT_FILES):
            raise ValueError(
                "dataset manifest contract_hashes must exactly cover the V2 contract bundle"
            )
        if verify_contract_hashes:
            for key, filename in CONTRACT_FILES.items():
                path = contract_dir / filename
                observed = sha256_file(path)
                if observed != declared_hashes[key]:
                    raise ValueError(
                        f"V2 contract hash mismatch for {key}: {observed} != "
                        f"{declared_hashes[key]}"
                    )
        observed_bundle = sha256_canonical_json(declared_hashes)
        declared_bundle = str(manifest.get("contract_bundle_hash") or "")
        if observed_bundle != declared_bundle:
            raise ValueError(
                f"V2 contract bundle hash mismatch: {observed_bundle} != {declared_bundle}"
            )

        process = _read_json(contract_dir / CONTRACT_FILES["process"])
        emission = _read_json(contract_dir / CONTRACT_FILES["emission"])
        projection = _read_json(contract_dir / CONTRACT_FILES["projection"])
        field_sets = _mapping(process.get("field_sets"), "process_registry.field_sets")
        conditions = _mapping(
            process.get("condition_sets"), "process_registry.condition_sets"
        )
        parameters = _mapping(
            process.get("field_parameters"), "process_registry.field_parameters"
        )
        dense_fields = _string_tuple(field_sets.get("dense_continuous"), "dense_continuous")
        ordinal_fields = _string_tuple(
            field_sets.get("gcs_ordinal_enabled"), "gcs_ordinal_enabled"
        )
        verbal_fields = _string_tuple(
            field_sets.get("gcs_verbal_reaggregated"), "gcs_verbal_reaggregated"
        )
        lab_fields = _string_tuple(
            field_sets.get("intermittent_labs"), "intermittent_labs"
        )
        respiratory_fields = _string_tuple(
            field_sets.get("respiratory_support"), "respiratory_support"
        )
        vasopressor_fields = _string_tuple(
            field_sets.get("vasopressor_support"), "vasopressor_support"
        )
        ned_fields = _string_tuple(field_sets.get("ned"), "ned")
        uop_fields = _string_tuple(field_sets.get("uop"), "uop")
        singleton_groups = {
            "gcs_verbal_reaggregated": verbal_fields,
            "respiratory_support": respiratory_fields,
            "vasopressor_support": vasopressor_fields,
            "ned": ned_fields,
            "uop": uop_fields,
        }
        for label, values in singleton_groups.items():
            if len(values) != 1:
                raise ValueError(f"process registry {label} must contain exactly one field")

        abnormal = _mapping(conditions.get("dense_abnormal"), "dense_abnormal")
        abnormal_conditions = {
            str(field): _string_tuple(values, f"dense_abnormal.{field}")
            for field, values in abnormal.items()
        }
        ordinal_max = {
            field: int(_mapping(parameters.get(field), f"field_parameters.{field}")["ordinal_max"])
            for field in ordinal_fields + verbal_fields
        }
        group_core_fields = set(
            dense_fields
            + ordinal_fields
            + verbal_fields
            + lab_fields
            + respiratory_fields
            + vasopressor_fields
            + ned_fields
            + uop_fields
        )
        category_rows = _read_category_rows(contract_dir / CONTRACT_FILES["category"])
        category_core = tuple(
            (field_id, field)
            for field_id, field in category_rows
            if field in group_core_fields
        )
        order_payload = process.get("registered_core_field_order")
        if not isinstance(order_payload, list) or len(order_payload) != EXPECTED_CORE_FIELD_COUNT:
            raise ValueError(
                "V2 process registry must explicitly declare registered_core_field_order"
            )
        registered_core_items: list[tuple[int, str]] = []
        for expected_position, raw_item in enumerate(order_payload):
            item = _mapping(raw_item, f"registered_core_field_order[{expected_position}]")
            _assert_exact_keys(
                item,
                {"position", "field_id", "field"},
                f"registered_core_field_order[{expected_position}]",
            )
            if _integer(item["position"], "position") != expected_position:
                raise ValueError("registered_core_field_order positions must be contiguous from zero")
            registered_core_items.append(
                (_integer(item["field_id"], "field_id"), str(item["field"]))
            )
        registered_core = tuple(registered_core_items)
        if set(registered_core) != set(category_core):
            raise ValueError(
                "registered_core_field_order must be a permutation of the category core fields"
            )
        if len(set(registered_core)) != EXPECTED_CORE_FIELD_COUNT:
            raise ValueError("registered_core_field_order contains duplicate field identities")
        registered_core_ids = tuple(field_id for field_id, _ in registered_core)
        registered_core_fields = tuple(field for _, field in registered_core)
        (
            relation_types,
            relation_lags,
            relation_adjacency,
            active_core_relation_edges,
            relation_total_edges,
            relation_active_core_edges,
            relation_deferred_edges,
        ) = _read_relation_contract(
            contract_dir / CONTRACT_FILES["relation"],
            registered_core_fields,
        )
        result = cls(
            dataset_root=root,
            manifest=manifest,
            process_registry=process,
            emission_registry=emission,
            projection_registry=projection,
            contract_hashes=declared_hashes,
            contract_bundle_hash=declared_bundle,
            dense_fields=dense_fields,
            ordinal_fields=ordinal_fields,
            verbal_field=verbal_fields[0],
            lab_fields=lab_fields,
            respiratory_field=respiratory_fields[0],
            vasopressor_field=vasopressor_fields[0],
            ned_field=ned_fields[0],
            uop_field=uop_fields[0],
            dense_abnormal_conditions=abnormal_conditions,
            respiratory_modalities=_string_tuple(
                conditions.get("respiratory_modalities"), "respiratory_modalities"
            ),
            vasopressor_agents=_string_tuple(
                conditions.get("vasopressor_agents"), "vasopressor_agents"
            ),
            ordinal_max=ordinal_max,
            registered_core_fields=registered_core_fields,
            registered_core_field_ids=registered_core_ids,
            relation_types=relation_types,
            relation_type_lags=relation_lags,
            relation_adjacency=relation_adjacency,
            active_core_relation_edges=active_core_relation_edges,
            relation_total_edges=relation_total_edges,
            relation_active_core_edges=relation_active_core_edges,
            relation_deferred_edges=relation_deferred_edges,
        )
        result._validate_definition(field_sets)
        result.validate_dataset_manifest(manifest)
        return result

    @property
    def core_fields(self) -> tuple[str, ...]:
        return self.registered_core_fields

    @property
    def abnormal_keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (field, condition)
            for field in self.dense_fields
            for condition in self.dense_abnormal_conditions.get(field, ())
        )

    @property
    def deterministic_projections_are_supervision(self) -> bool:
        return False

    def _validate_definition(self, field_sets: Mapping[str, Any]) -> None:
        if self.process_registry.get("schema") != PROCESS_REGISTRY_SCHEMA:
            raise ValueError("V2 process registry schema mismatch")
        if self.process_registry.get("version") != PROCESS_REGISTRY_VERSION:
            raise ValueError(
                "V2 process registry must use the frozen r9 explicit topological field order"
            )
        if self.emission_registry.get("schema") != EMISSION_REGISTRY_SCHEMA:
            raise ValueError("V2 emission registry schema mismatch")
        if self.emission_registry.get("version") != EMISSION_REGISTRY_VERSION:
            raise ValueError("V2 emission registry must use the frozen r9 head contract")
        head_contract = _mapping(
            self.emission_registry.get("enabled_core_head_contract"),
            "emission_registry.enabled_core_head_contract",
        )
        layouts = _mapping(
            head_contract.get("layouts"),
            "emission_registry.enabled_core_head_contract.layouts",
        )
        if len(layouts) != 19:
            raise ValueError("V2 emission registry must declare exactly 19 enabled head layouts")
        for likelihood_id, row_value in layouts.items():
            row = _mapping(
                row_value,
                f"emission_registry.enabled_core_head_contract.layouts.{likelihood_id}",
            )
            _assert_exact_keys(
                row,
                {"width", "layout"},
                f"emission_registry.enabled_core_head_contract.layouts.{likelihood_id}",
            )
            _bounded_int(
                row["width"],
                1,
                2**31 - 1,
                f"emission_registry.enabled_core_head_contract.layouts.{likelihood_id}.width",
            )
            if not isinstance(row["layout"], str) or not row["layout"]:
                raise ValueError(f"V2 emission layout {likelihood_id!r} must be a nonempty string")
        if sha256_canonical_json(layouts) != ENABLED_CORE_LAYOUTS_SHA256:
            raise ValueError("V2 emission registry head widths/layouts differ from frozen r9")
        retained_auxiliary = head_contract.get("retained_auxiliary_without_initial_heads")
        if tuple(retained_auxiliary or ()) != EXPECTED_RETAINED_AUXILIARY_WITHOUT_INITIAL_HEADS:
            raise ValueError("V2 retained auxiliary no-head policy differs from the frozen contract")
        if self.projection_registry.get("schema") != PROJECTION_REGISTRY_SCHEMA:
            raise ValueError("V2 projection registry schema mismatch")
        if self.projection_registry.get("version") != PROJECTION_REGISTRY_VERSION:
            raise ValueError("V2 projection registry must use the frozen r9 contract")
        scope = _mapping(self.process_registry.get("scope"), "process_registry.scope")
        if tuple(scope.get("future_blocks") or ()) != BLOCK_IDS:
            raise ValueError("V2 target contract must use exactly M4_01 through M4_06")
        if str(scope.get("resolution")) != "M4":
            raise ValueError("V2 target contract resolution must be M4")
        if int(scope.get("core_field_count", -1)) != EXPECTED_CORE_FIELD_COUNT:
            raise ValueError("V2 process registry must declare 29 core fields")
        if int(scope.get("expanded_enabled_core_primitives", -1)) != EXPECTED_ENABLED_FACTOR_COUNT:
            raise ValueError("V2 process registry must declare 414 enabled primitive factors")
        if str(scope.get("input_contract")) != "multires_event_v1_unchanged":
            raise ValueError("V2 process registry must preserve the V1 input contract")
        if scope.get("legacy_tuple_authority") is not False:
            raise ValueError("legacy five-tuples cannot be V2 target authority")

        fields_from_registry: list[str] = []
        for group in CORE_FIELD_GROUPS:
            fields_from_registry.extend(_string_tuple(field_sets.get(group), group))
        if len(fields_from_registry) != EXPECTED_CORE_FIELD_COUNT:
            raise ValueError("V2 core field groups do not expand to 29 fields")
        if len(set(fields_from_registry)) != EXPECTED_CORE_FIELD_COUNT:
            raise ValueError("V2 core field groups contain duplicate fields")
        if set(fields_from_registry) != set(self.core_fields):
            raise ValueError("compiled V2 core fields differ from process registry")
        if len(self.relation_types) != 14:
            raise ValueError("V2 relation registry must declare exactly 14 relation types")
        if len(self.relation_type_lags) != len(self.relation_types):
            raise ValueError("V2 relation type lag metadata is misaligned")
        if any(lag not in {0, 1} for lag in self.relation_type_lags):
            raise ValueError("V2 relation type lag must be zero or one block")
        if (
            self.relation_total_edges,
            self.relation_active_core_edges,
            self.relation_deferred_edges,
        ) != (68, 50, 18):
            raise ValueError(
                "V2 relation subset must retain 50 core edges and defer 18 input/aux edges"
            )
        temporal_edges = tuple(
            edge for edge in self.active_core_relation_edges if edge.lag_blocks == 1
        )
        structural_edges = tuple(
            edge
            for edge in self.active_core_relation_edges
            if edge.lag_blocks == 0 and edge.source_field != edge.target_field
        )
        if (
            len(temporal_edges) != EXPECTED_CORE_FIELD_COUNT
            or {edge.source_field for edge in temporal_edges} != set(self.core_fields)
            or any(
                edge.source_field != edge.target_field
                or edge.relation_type != "self_transition"
                for edge in temporal_edges
            )
        ):
            raise ValueError("V2 core relation registry must contain 29 exact lag-1 self edges")
        if len(structural_edges) != 21 or len(
            {edge.edge_id for edge in structural_edges}
        ) != 21:
            raise ValueError("V2 promotion relation cover must contain 21 canonical lag-0 edges")
        if not set(self.dense_abnormal_conditions).issubset(self.dense_fields):
            raise ValueError("dense abnormal condition map references a non-dense field")

        objective = _mapping(
            self.process_registry.get("objective_contract"),
            "process_registry.objective_contract",
        )
        if objective.get("legacy_tuple_likelihood") != "forbidden":
            raise ValueError("V2 objective must forbid legacy tuple likelihood")
        global_emission = _mapping(
            self.emission_registry.get("global_contract"),
            "emission_registry.global_contract",
        )
        if global_emission.get("legacy_tuple_terms") != "none":
            raise ValueError("V2 emission contract must contain no tuple loss terms")
        if global_emission.get("relation_terms") != "none":
            raise ValueError("V2 emission contract must contain no relation loss terms")
        arithmetic = _mapping(
            global_emission.get("target_arithmetic_canonicalization"),
            "emission_registry.global_contract.target_arithmetic_canonicalization",
        )
        if (
            arithmetic.get("policy") != TARGET_ARITHMETIC_POLICY
            or arithmetic.get("scope")
            != "dense_continuous_MEAN_and_negative_binary64_respiratory_simplex_residual_only"
            or arithmetic.get("clinical_clipping") is not False
        ):
            raise ValueError("V2 target arithmetic canonicalization contract drift")
        legacy_projection = _mapping(
            self.projection_registry.get("legacy_contract"),
            "projection_registry.legacy_contract",
        )
        if legacy_projection.get("direct_tuple_likelihood") != "forbidden":
            raise ValueError("deterministic five-tuple projections cannot be direct losses")

    def validate_dataset_manifest(self, manifest: Mapping[str, Any]) -> None:
        if manifest.get("schema") != DATASET_MANIFEST_SCHEMA:
            raise ValueError("V2 dataset manifest schema mismatch")
        if manifest.get("status") != "SUCCEEDED":
            raise ValueError("V2 target sidecar is not a completed artifact")
        if not str(manifest.get("dataset_id") or ""):
            raise ValueError("V2 dataset manifest has an empty dataset_id")
        hashes = _mapping(manifest.get("contract_hashes"), "contract_hashes")
        if dict(hashes) != dict(self.contract_hashes):
            raise ValueError("V2 dataset manifest contract hashes changed after compilation")
        if str(manifest.get("contract_bundle_hash") or "") != self.contract_bundle_hash:
            raise ValueError("V2 dataset manifest contract bundle hash mismatch")
        base = _mapping(manifest.get("base_dataset"), "base_dataset")
        required_base = {
            "dataset_id",
            "dataset_manifest_sha256",
            "fingerprint",
            "root",
            "sample_manifest_sha256",
            "subject_split_sha256",
        }
        if not required_base.issubset(base):
            raise ValueError("V2 dataset manifest base authority is incomplete")
        for key in (
            "dataset_manifest_sha256",
            "fingerprint",
            "sample_manifest_sha256",
            "subject_split_sha256",
        ):
            _assert_sha256(str(base[key]), f"base_dataset.{key}")
        counts = _mapping(manifest.get("counts"), "counts")
        samples = _nonnegative_int(counts.get("samples"), "counts.samples")
        valid = _nonnegative_int(counts.get("valid_samples"), "counts.valid_samples")
        errors = _nonnegative_int(counts.get("errors"), "counts.errors")
        if valid != samples or errors != 0:
            raise ValueError("V2 sidecar must have every sample valid and zero errors")
        by_split = _mapping(counts.get("by_split"), "counts.by_split")
        if sum(_nonnegative_int(value, f"counts.by_split.{key}") for key, value in by_split.items()) != samples:
            raise ValueError("V2 sidecar split counts do not sum to sample count")
        if (
            _nonnegative_int(
                counts.get("target_arithmetic_canonicalizations"),
                "counts.target_arithmetic_canonicalizations",
            )
            != EXPECTED_TARGET_ARITHMETIC_RECORDS
            or _nonnegative_int(
                counts.get("samples_with_target_arithmetic_canonicalization"),
                "counts.samples_with_target_arithmetic_canonicalization",
            )
            != EXPECTED_TARGET_ARITHMETIC_SAMPLES
        ):
            raise ValueError("V2 r9 arithmetic canonicalization manifest counts drifted")
        arithmetic_by_type = _mapping(
            counts.get("target_arithmetic_canonicalization_by_type"),
            "counts.target_arithmetic_canonicalization_by_type",
        )
        if dict(arithmetic_by_type) != EXPECTED_TARGET_ARITHMETIC_BY_TYPE:
            raise ValueError("V2 r9 arithmetic canonicalization type ledger drifted")
        files = _mapping(manifest.get("files"), "files")
        for name in ("sample_manifest", "subject_split", "target_shards"):
            if name not in files:
                raise ValueError(f"V2 dataset manifest is missing files.{name}")

    def validate_target_record(
        self,
        record: Mapping[str, Any],
        *,
        verify_content_hash: bool = True,
    ) -> None:
        required = {
            "schema",
            "sample_id",
            "subject_id",
            "hadm_id",
            "stay_id",
            "prediction_hour",
            "split",
            "base_dataset_id",
            "base_dataset_fingerprint",
            "base_content_hash",
            "contract_hashes",
            "target_arithmetic_canonicalization",
            "source_evidence",
            "blocks",
            "target_content_hash",
        }
        _assert_exact_keys(record, required, "V2 target row")
        if record.get("schema") != TARGET_SCHEMA:
            raise ValueError("V2 target row schema mismatch")
        for key in ("sample_id", "subject_id", "hadm_id", "stay_id"):
            if not str(record.get(key) or ""):
                raise ValueError(f"V2 target row has empty {key}")
        _integer(record.get("prediction_hour"), "prediction_hour")
        if record.get("split") not in {"train", "val", "test"}:
            raise ValueError("V2 target row split is invalid")
        base = _mapping(self.manifest.get("base_dataset"), "base_dataset")
        if str(record.get("base_dataset_id")) != str(base["dataset_id"]):
            raise ValueError("V2 target row base_dataset_id mismatch")
        if str(record.get("base_dataset_fingerprint")) != str(base["fingerprint"]):
            raise ValueError("V2 target row base fingerprint mismatch")
        _assert_sha256(str(record.get("base_content_hash") or ""), "base_content_hash")
        if dict(_mapping(record.get("contract_hashes"), "record.contract_hashes")) != dict(
            self.contract_hashes
        ):
            raise ValueError("V2 target row contract hashes differ from dataset manifest")
        evidence = _mapping(record.get("source_evidence"), "source_evidence")
        _assert_exact_keys(
            evidence,
            {"field_ready_contract_hash", "point_events_sha256", "gcs_verbal_reaggregated"},
            "source_evidence",
        )
        _assert_sha256(str(evidence["field_ready_contract_hash"]), "field_ready_contract_hash")
        _assert_sha256(str(evidence["point_events_sha256"]), "point_events_sha256")
        if evidence["gcs_verbal_reaggregated"] is not True:
            raise ValueError("V2 gcs_verbal target must be reaggregated from field-ready evidence")

        if verify_content_hash:
            target_hash = str(record.get("target_content_hash") or "")
            _assert_sha256(target_hash, "target_content_hash")
            payload = copy.deepcopy(dict(record))
            payload.pop("target_content_hash", None)
            observed_hash = sha256_canonical_json(payload)
            if observed_hash != target_hash:
                raise ValueError(
                    f"V2 target_content_hash mismatch for {record.get('sample_id')}: "
                    f"{observed_hash} != {target_hash}"
                )

        blocks = record.get("blocks")
        if not isinstance(blocks, list) or len(blocks) != len(BLOCK_IDS):
            raise ValueError("V2 target row must contain exactly six M4 blocks")
        for index, block in enumerate(blocks):
            self._validate_block(block, index)
        self._validate_target_arithmetic_canonicalization(
            record["target_arithmetic_canonicalization"], blocks
        )

    def _validate_target_arithmetic_canonicalization(
        self,
        value: Any,
        blocks: Sequence[Mapping[str, Any]],
    ) -> None:
        evidence = _mapping(value, "target_arithmetic_canonicalization")
        _assert_exact_keys(
            evidence,
            {"schema", "policy", "count", "by_type", "records"},
            "target_arithmetic_canonicalization",
        )
        if evidence["schema"] != TARGET_ARITHMETIC_SCHEMA:
            raise ValueError("V2 target arithmetic evidence schema mismatch")
        if evidence["policy"] != TARGET_ARITHMETIC_POLICY:
            raise ValueError("V2 target arithmetic evidence policy mismatch")
        records = evidence["records"]
        if not isinstance(records, list):
            raise ValueError("V2 target arithmetic records must be an array")
        declared_count = _nonnegative_int(evidence["count"], "target arithmetic count")
        if declared_count != len(records):
            raise ValueError("V2 target arithmetic record count mismatch")
        declared_by_type = _mapping(evidence["by_type"], "target arithmetic by_type")
        if not set(declared_by_type).issubset(TARGET_ARITHMETIC_TYPES):
            raise ValueError("V2 target arithmetic by_type contains an unknown type")
        for key, count in declared_by_type.items():
            if _nonnegative_int(count, f"target arithmetic by_type.{key}") < 1:
                raise ValueError("V2 target arithmetic by_type counts must be positive")

        block_map = {str(block["block_id"]): block for block in blocks}
        seen: set[tuple[str, str]] = set()
        observed_by_type: dict[str, int] = {}
        dense_record_keys = {
            "block_id",
            "field",
            "type",
            "observed_hours",
            "original_mean",
            "canonical_mean",
            "support_boundary",
            "boundary_value",
            "ulp_rule",
        }
        respiratory_record_keys = {
            "block_id",
            "field",
            "type",
            "adjusted_component",
            "original_component_duration",
            "canonical_component_duration",
            "original_uncovered",
            "canonical_uncovered",
            "block_span_hours",
            "original_documented_duration_sum",
            "canonical_documented_duration_sum",
            "ulp_at_block_span",
            "max_negative_roundoff_ulps",
            "rule",
        }
        for index, raw_record in enumerate(records):
            row = _mapping(raw_record, f"target arithmetic record {index}")
            block_id = str(row["block_id"])
            field = str(row["field"])
            record_type = str(row["type"])
            key = (block_id, field)
            if key in seen:
                raise ValueError("V2 target arithmetic records duplicate block/field")
            seen.add(key)
            if record_type not in TARGET_ARITHMETIC_TYPES:
                raise ValueError("V2 target arithmetic record type is invalid")
            if record_type == RESPIRATORY_ARITHMETIC_TYPE:
                _assert_exact_keys(
                    row,
                    respiratory_record_keys,
                    f"target arithmetic record {index}",
                )
                if block_id not in block_map or field != self.respiratory_field:
                    raise ValueError(
                        "V2 respiratory arithmetic record references an illegal target"
                    )
                process = _mapping(
                    _mapping(
                        block_map[block_id].get("processes"),
                        f"{block_id}.processes",
                    ).get(field),
                    f"{block_id}.{field}",
                )
                durations = _numeric_vector(
                    process.get("documented_duration"),
                    self.respiratory_modalities,
                    f"{block_id}.{field}.duration",
                )
                component = str(row["adjusted_component"])
                if component not in self.respiratory_modalities:
                    raise ValueError("V2 respiratory arithmetic component is invalid")
                original_component = _number(
                    row["original_component_duration"],
                    "respiratory arithmetic original component",
                )
                canonical_component = _number(
                    row["canonical_component_duration"],
                    "respiratory arithmetic canonical component",
                )
                original_uncovered = _number(
                    row["original_uncovered"],
                    "respiratory arithmetic original uncovered",
                )
                canonical_uncovered = _number(
                    row["canonical_uncovered"],
                    "respiratory arithmetic canonical uncovered",
                )
                span = _number(row["block_span_hours"], "respiratory arithmetic span")
                original_sum = _number(
                    row["original_documented_duration_sum"],
                    "respiratory arithmetic original sum",
                )
                canonical_sum = _number(
                    row["canonical_documented_duration_sum"],
                    "respiratory arithmetic canonical sum",
                )
                restored = dict(durations)
                restored[component] = original_component
                valid = (
                    _float_exact(durations[component], canonical_component)
                    and _float_exact(sum(restored[name] for name in self.respiratory_modalities), original_sum)
                    and _float_exact(span - original_sum, original_uncovered)
                    and original_uncovered < 0.0
                    and _float_exact(sum(durations[name] for name in self.respiratory_modalities), span)
                    and _float_exact(canonical_sum, span)
                    and _float_exact(
                        _number(process.get("uncovered_duration"), "respiratory uncovered"),
                        canonical_uncovered,
                    )
                    and _float_exact(canonical_uncovered, 0.0)
                    and _float_exact(
                        _number(row["ulp_at_block_span"], "respiratory arithmetic ULP"),
                        math.ulp(span),
                    )
                    and _integer(
                        row["max_negative_roundoff_ulps"],
                        "respiratory arithmetic ULP limit",
                    )
                    == 32
                    and -original_uncovered <= 32 * math.ulp(span)
                    and row["rule"]
                    == "negative_residual_within_registered_binary64_ulp_bound_to_zero"
                )
                if not valid:
                    raise ValueError("V2 respiratory arithmetic evidence violates closure rule")
                observed_by_type[record_type] = observed_by_type.get(record_type, 0) + 1
                continue
            _assert_exact_keys(
                row, dense_record_keys, f"target arithmetic record {index}"
            )
            if block_id not in block_map or field not in self.dense_fields:
                raise ValueError("V2 target arithmetic record references a non-dense target")
            process = _mapping(
                _mapping(block_map[block_id].get("processes"), f"{block_id}.processes").get(field),
                f"{block_id}.{field}",
            )
            state = _numeric_state(
                process.get("value_state"),
                ("last", "min", "max", "mean"),
                f"{block_id}.{field}.value_state",
            )
            observed = _bounded_int(
                process.get("observed_hours"), 1, 4, f"{block_id}.{field}.observed_hours"
            )
            if _integer(row["observed_hours"], "target arithmetic observed_hours") != observed:
                raise ValueError("V2 target arithmetic observed-hours mismatch")
            original = _number(row["original_mean"], "target arithmetic original_mean")
            canonical = _number(row["canonical_mean"], "target arithmetic canonical_mean")
            boundary = _number(row["boundary_value"], "target arithmetic boundary_value")
            if not _float_exact(state["mean"], canonical):
                raise ValueError("V2 target arithmetic canonical MEAN is not persisted")
            if _float_exact(original, canonical):
                raise ValueError("V2 target arithmetic evidence records no value change")
            lower, upper = _dense_mean_bounds(
                observed, state["last"], state["min"], state["max"]
            )
            if not _valid_target_arithmetic_record(
                row,
                record_type=record_type,
                original=original,
                canonical=canonical,
                boundary=boundary,
                minimum=state["min"],
                maximum=state["max"],
                lower=lower,
                upper=upper,
            ):
                raise ValueError("V2 target arithmetic evidence violates its one-ULP rule")
            observed_by_type[record_type] = observed_by_type.get(record_type, 0) + 1
        if dict(sorted(observed_by_type.items())) != dict(sorted(declared_by_type.items())):
            raise ValueError("V2 target arithmetic evidence by_type accounting mismatch")

    def _validate_block(self, value: Any, index: int) -> None:
        block = _mapping(value, f"blocks[{index}]")
        _assert_exact_keys(
            block,
            {"block_id", "block_index", "relative_start_hour", "relative_end_hour", "processes"},
            f"blocks[{index}]",
        )
        expected_start, expected_end = BLOCK_BOUNDS[index]
        if block["block_id"] != BLOCK_IDS[index]:
            raise ValueError(f"blocks[{index}] block_id is out of canonical order")
        if _integer(block["block_index"], "block_index") != index + 1:
            raise ValueError(f"blocks[{index}] block_index mismatch")
        if _integer(block["relative_start_hour"], "relative_start_hour") != expected_start:
            raise ValueError(f"blocks[{index}] relative_start_hour mismatch")
        if _integer(block["relative_end_hour"], "relative_end_hour") != expected_end:
            raise ValueError(f"blocks[{index}] relative_end_hour mismatch")
        processes = _mapping(block.get("processes"), f"blocks[{index}].processes")
        if set(processes) != set(self.core_fields):
            missing = sorted(set(self.core_fields) - set(processes))
            extra = sorted(set(processes) - set(self.core_fields))
            raise ValueError(
                f"{BLOCK_IDS[index]} must contain exactly 29 core fields; "
                f"missing={missing}, extra={extra}"
            )
        for field in self.dense_fields:
            self._validate_dense(field, processes[field], BLOCK_IDS[index])
        for field in self.ordinal_fields:
            self._validate_ordinal(field, processes[field], BLOCK_IDS[index])
        self._validate_verbal(processes[self.verbal_field], BLOCK_IDS[index])
        for field in self.lab_fields:
            self._validate_lab(field, processes[field], BLOCK_IDS[index])
        self._validate_respiratory(processes[self.respiratory_field], BLOCK_IDS[index])
        self._validate_vasopressor(processes[self.vasopressor_field], BLOCK_IDS[index])
        self._validate_ned(processes[self.ned_field], BLOCK_IDS[index])
        self._validate_uop(processes[self.uop_field], BLOCK_IDS[index])
        ned_state = processes[self.ned_field]["value_state"]
        compatible_agents = self.vasopressor_agents[:-1]
        compatible_active = any(
            float(processes[self.vasopressor_field]["duration"][agent]) > 0.0
            for agent in compatible_agents
        )
        if float(ned_state["max"]) > 0.0 and not compatible_active:
            raise ValueError(
                f"{BLOCK_IDS[index]} positive NED requires a compatible vasopressor duration"
            )

    def _validate_dense(self, field: str, value: Any, block: str) -> None:
        process = _mapping(value, f"{block}.{field}")
        _assert_exact_keys(
            process,
            {"family", "observed_hours", "value_state", "abnormal_occupancy"},
            f"{block}.{field}",
        )
        if process["family"] != "dense_point":
            raise ValueError(f"{block}.{field} family must be dense_point")
        observed = _bounded_int(process["observed_hours"], 0, 4, f"{block}.{field}.observed_hours")
        state = process["value_state"]
        if (state is None) != (observed == 0):
            raise ValueError(f"{block}.{field} value_state activation disagrees with observed_hours")
        if state is not None:
            values = _numeric_state(state, ("last", "min", "max", "mean"), f"{block}.{field}")
            minimum, last, maximum, mean = (
                values["min"],
                values["last"],
                values["max"],
                values["mean"],
            )
            if not minimum <= last <= maximum or not minimum <= mean <= maximum:
                raise ValueError(f"{block}.{field} dense value state violates ordering")
            lower, upper = _dense_mean_bounds(observed, last, minimum, maximum)
            if not lower <= mean <= upper:
                raise ValueError(f"{block}.{field} mean is incompatible with H/MIN/LAST/MAX")
            if (
                minimum < maximum
                and not _float_exact(mean, lower)
                and not _float_exact(mean, upper)
                and (
                    _float_exact(mean, math.nextafter(lower, math.inf))
                    or _float_exact(mean, math.nextafter(upper, -math.inf))
                )
            ):
                raise ValueError(
                    f"{block}.{field} retains a forbidden pre-r8 one-ULP interior MEAN"
                )
        occupancy = _mapping(process["abnormal_occupancy"], f"{block}.{field}.abnormal")
        expected_conditions = (
            set(self.dense_abnormal_conditions.get(field, ())) if observed > 0 else set()
        )
        if set(occupancy) != expected_conditions:
            raise ValueError(f"{block}.{field} abnormal occupancy channels mismatch")
        for condition, count in occupancy.items():
            _bounded_int(count, 0, observed, f"{block}.{field}.{condition}")

    def _validate_ordinal(self, field: str, value: Any, block: str) -> None:
        process = _mapping(value, f"{block}.{field}")
        _assert_exact_keys(
            process,
            {"family", "observed_hours", "ordinal_state"},
            f"{block}.{field}",
        )
        if process["family"] != "ordinal_point":
            raise ValueError(f"{block}.{field} family must be ordinal_point")
        observed = _bounded_int(process["observed_hours"], 0, 4, f"{block}.{field}.observed_hours")
        state = process["ordinal_state"]
        if (state is None) != (observed == 0):
            raise ValueError(f"{block}.{field} ordinal activation disagrees with observed_hours")
        if state is not None:
            values = _integer_state(state, ("last", "min", "max"), f"{block}.{field}")
            if not 1 <= values["min"] <= values["last"] <= values["max"] <= self.ordinal_max[field]:
                raise ValueError(f"{block}.{field} ordinal state is outside legal support")

    def _validate_verbal(self, value: Any, block: str) -> None:
        field = self.verbal_field
        process = _mapping(value, f"{block}.{field}")
        _assert_exact_keys(
            process,
            {
                "family",
                "observed_hours",
                "ungradable_hours",
                "gradable_hours",
                "last_observation_status",
                "gradable_state",
            },
            f"{block}.{field}",
        )
        if process["family"] != "gcs_verbal_testability":
            raise ValueError(f"{block}.{field} family mismatch")
        observed = _bounded_int(process["observed_hours"], 0, 4, f"{block}.{field}.observed")
        ungradable = _bounded_int(
            process["ungradable_hours"], 0, observed, f"{block}.{field}.ungradable"
        )
        gradable = _bounded_int(
            process["gradable_hours"], 0, observed, f"{block}.{field}.gradable"
        )
        if gradable != observed - ungradable:
            raise ValueError(f"{block}.{field} violates H_gradable = H_observed - H_ungradable")
        status = process["last_observation_status"]
        if status not in {"UNOBSERVED", "GRADABLE", "UNGRADABLE"}:
            raise ValueError(f"{block}.{field} has invalid latest observation status")
        if observed == 0 and status != "UNOBSERVED":
            raise ValueError(f"{block}.{field} unobserved block must use UNOBSERVED status")
        if observed > 0 and status == "UNOBSERVED":
            raise ValueError(f"{block}.{field} observed block cannot use UNOBSERVED status")
        if ungradable == 0 and observed > 0 and status != "GRADABLE":
            raise ValueError(f"{block}.{field} status must be GRADABLE")
        if gradable == 0 and observed > 0 and status != "UNGRADABLE":
            raise ValueError(f"{block}.{field} status must be UNGRADABLE")
        state = process["gradable_state"]
        if (state is None) != (gradable == 0):
            raise ValueError(f"{block}.{field} gradable state activation mismatch")
        if state is not None:
            values = _integer_state(state, ("last", "min", "max"), f"{block}.{field}")
            if not 1 <= values["min"] <= values["last"] <= values["max"] <= self.ordinal_max[field]:
                raise ValueError(f"{block}.{field} gradable ordinal state is illegal")

    def _validate_lab(self, field: str, value: Any, block: str) -> None:
        process = _mapping(value, f"{block}.{field}")
        _assert_exact_keys(
            process,
            {"family", "observation_count", "value_state"},
            f"{block}.{field}",
        )
        if process["family"] != "intermittent_lab":
            raise ValueError(f"{block}.{field} family must be intermittent_lab")
        count = _nonnegative_int(process["observation_count"], f"{block}.{field}.count")
        state = process["value_state"]
        if (state is None) != (count == 0):
            raise ValueError(f"{block}.{field} value state activation disagrees with count")
        if state is not None:
            values = _numeric_state(state, ("last", "min", "max"), f"{block}.{field}")
            if (
                values["last"] < values["min"] - PHYSICAL_ARITHMETIC_ATOL
                or values["last"] > values["max"] + PHYSICAL_ARITHMETIC_ATOL
            ):
                raise ValueError(f"{block}.{field} lab value state violates ordering")

    def _validate_respiratory(self, value: Any, block: str) -> None:
        field = self.respiratory_field
        process = _mapping(value, f"{block}.{field}")
        _assert_exact_keys(
            process,
            {
                "family",
                "block_evidence",
                "edge_evidence",
                "documented_duration",
                "uncovered_duration",
                "edge_category",
                "onset_count",
            },
            f"{block}.{field}",
        )
        if process["family"] != "respiratory_support":
            raise ValueError(f"{block}.{field} family mismatch")
        block_evidence = _boolean(process["block_evidence"], f"{block}.{field}.block_evidence")
        edge_evidence = _boolean(process["edge_evidence"], f"{block}.{field}.edge_evidence")
        if edge_evidence and not block_evidence:
            raise ValueError(f"{block}.{field} edge evidence requires block evidence")
        nullable_keys = ("documented_duration", "uncovered_duration", "onset_count")
        if not block_evidence:
            if any(process[key] is not None for key in nullable_keys) or process["edge_category"] is not None:
                raise ValueError(f"{block}.{field} inactive respiratory branch must be null")
            return
        durations = _numeric_vector(
            process["documented_duration"], self.respiratory_modalities, f"{block}.{field}.duration"
        )
        uncovered = _number(process["uncovered_duration"], f"{block}.{field}.uncovered")
        if any(
            number < 0.0
            or number > 4.0 + PHYSICAL_ARITHMETIC_ATOL
            for number in durations.values()
        ) or (
            uncovered < 0.0
            or uncovered > 4.0 + PHYSICAL_ARITHMETIC_ATOL
        ):
            raise ValueError(f"{block}.{field} respiratory duration is outside [0,4]")
        if not math.isclose(
            sum(durations.values()) + uncovered,
            4.0,
            rel_tol=0.0,
            abs_tol=PHYSICAL_ARITHMETIC_ATOL,
        ):
            raise ValueError(f"{block}.{field} respiratory occupancy does not close to four hours")
        onsets = _integer_vector(
            process["onset_count"], self.respiratory_modalities, f"{block}.{field}.onset"
        )
        if any(number < 0 for number in onsets.values()):
            raise ValueError(f"{block}.{field} respiratory onset count is negative")
        category = process["edge_category"]
        if edge_evidence and category not in self.respiratory_modalities:
            raise ValueError(f"{block}.{field} edge category is invalid")
        if not edge_evidence and category is not None:
            raise ValueError(f"{block}.{field} edge category requires edge evidence")

    def _validate_vasopressor(self, value: Any, block: str) -> None:
        field = self.vasopressor_field
        process = _mapping(value, f"{block}.{field}")
        _assert_exact_keys(
            process,
            {"family", "duration", "edge_state", "onset_count"},
            f"{block}.{field}",
        )
        if process["family"] != "vasopressor_support":
            raise ValueError(f"{block}.{field} family mismatch")
        durations = _numeric_vector(
            process["duration"], self.vasopressor_agents, f"{block}.{field}.duration"
        )
        if any(
            number < -PHYSICAL_ARITHMETIC_ATOL
            or number > 4.0 + PHYSICAL_ARITHMETIC_ATOL
            for number in durations.values()
        ):
            raise ValueError(f"{block}.{field} agent duration is outside [0,4]")
        edge = _integer_vector(
            process["edge_state"], self.vasopressor_agents, f"{block}.{field}.edge_state"
        )
        if any(number not in {0, 1} for number in edge.values()):
            raise ValueError(f"{block}.{field} edge state must be binary")
        onsets = _integer_vector(
            process["onset_count"], self.vasopressor_agents, f"{block}.{field}.onset"
        )
        if any(number < 0 for number in onsets.values()):
            raise ValueError(f"{block}.{field} onset count is negative")

    def _validate_ned(self, value: Any, block: str) -> None:
        field = self.ned_field
        process = _mapping(value, f"{block}.{field}")
        _assert_exact_keys(process, {"family", "value_state"}, f"{block}.{field}")
        if process["family"] != "norepinephrine_equivalent_dose":
            raise ValueError(f"{block}.{field} family mismatch")
        state = _numeric_state(
            process["value_state"], ("last", "max", "mean"), f"{block}.{field}"
        )
        if any(number < 0 for number in state.values()):
            raise ValueError(f"{block}.{field} NED values must be nonnegative")
        if (
            state["last"] > state["max"] + PHYSICAL_ARITHMETIC_ATOL
            or state["mean"] > state["max"] + PHYSICAL_ARITHMETIC_ATOL
        ):
            raise ValueError(f"{block}.{field} NED state exceeds its maximum")

    def _validate_uop(self, value: Any, block: str) -> None:
        field = self.uop_field
        process = _mapping(value, f"{block}.{field}")
        _assert_exact_keys(process, {"family", "observation_count", "sum"}, f"{block}.{field}")
        if process["family"] != "urine_output":
            raise ValueError(f"{block}.{field} family mismatch")
        count = _nonnegative_int(process["observation_count"], f"{block}.{field}.count")
        amount = process["sum"]
        if (amount is None) != (count == 0):
            raise ValueError(f"{block}.{field} sum activation disagrees with count")
        if amount is not None and _number(amount, f"{block}.{field}.sum") < 0:
            raise ValueError(f"{block}.{field} urine output sum must be nonnegative")


def _float_exact(left: float, right: float) -> bool:
    return float(left).hex() == float(right).hex()


def _dense_mean_bounds(
    observed_hours: int,
    last: float,
    minimum: float,
    maximum: float,
) -> tuple[float, float]:
    if not 1 <= observed_hours <= 4 or not minimum <= last <= maximum:
        raise ValueError("dense reducer state is outside its exact arithmetic support")
    if observed_hours == 1:
        return last, last
    lower = (
        minimum
        if _float_exact(last, minimum)
        else (last + (observed_hours - 1) * minimum) / observed_hours
    )
    upper = (
        maximum
        if _float_exact(last, maximum)
        else (last + (observed_hours - 1) * maximum) / observed_hours
    )
    lower = min(max(lower, minimum), maximum)
    upper = min(max(upper, minimum), maximum)
    if lower > upper:
        raise ValueError("dense binary64 reducer bounds crossed")
    return lower, upper


def _valid_target_arithmetic_record(
    row: Mapping[str, Any],
    *,
    record_type: str,
    original: float,
    canonical: float,
    boundary: float,
    minimum: float,
    maximum: float,
    lower: float,
    upper: float,
) -> bool:
    if record_type == "dense_zero_range_mean_to_min":
        below = row.get("ulp_rule") == "original_MEAN_equals_nextafter(MIN,-infinity)"
        above = row.get("ulp_rule") == "original_MEAN_equals_nextafter(MIN,+infinity)"
        return bool(
            _float_exact(minimum, maximum)
            and row.get("support_boundary") == "MIN"
            and (below or above)
            and _float_exact(boundary, minimum)
            and _float_exact(canonical, minimum)
            and _float_exact(
                original,
                math.nextafter(minimum, -math.inf if below else math.inf),
            )
        )
    if not minimum < maximum:
        return False
    cases = {
        "dense_positive_range_mean_lower_outside_one_ulp": (
            "LOWER",
            lower,
            -math.inf,
            "original_MEAN_equals_nextafter(lower,-infinity)",
            True,
        ),
        "dense_positive_range_mean_lower_inside_one_ulp": (
            "LOWER",
            lower,
            math.inf,
            "original_MEAN_equals_nextafter(lower,+infinity)",
            original < upper,
        ),
        "dense_positive_range_mean_upper_inside_one_ulp": (
            "UPPER",
            upper,
            -math.inf,
            "original_MEAN_equals_nextafter(upper,-infinity)",
            original > lower,
        ),
        "dense_positive_range_mean_upper_outside_one_ulp": (
            "UPPER",
            upper,
            math.inf,
            "original_MEAN_equals_nextafter(upper,+infinity)",
            True,
        ),
    }
    expected = cases.get(record_type)
    if expected is None:
        return False
    support_boundary, endpoint, direction, ulp_rule, nonambiguous = expected
    return bool(
        nonambiguous
        and row.get("support_boundary") == support_boundary
        and row.get("ulp_rule") == ulp_rule
        and _float_exact(boundary, endpoint)
        and _float_exact(canonical, endpoint)
        and _float_exact(original, math.nextafter(endpoint, direction))
    )


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_canonical_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_category_rows(path: Path) -> tuple[tuple[int, str], ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"field_id", "field"}.issubset(reader.fieldnames or ()):
            raise ValueError("field_category_matrix_v1.csv is missing field_id/field")
        rows = tuple(
            (_bounded_int(int(row["field_id"]), 1, 2**31 - 1, "field_id"), str(row["field"]))
            for row in reader
        )
    if not rows or any(not field for _, field in rows):
        raise ValueError("field category matrix has empty rows")
    if len({field_id for field_id, _ in rows}) != len(rows):
        raise ValueError("field category matrix contains duplicate field_id")
    if len({field for _, field in rows}) != len(rows):
        raise ValueError("field category matrix contains duplicate field")
    if tuple(sorted(rows)) != rows:
        raise ValueError("field category matrix must be ordered by field_id")
    return rows


def _read_relation_contract(
    path: Path,
    core_fields: Sequence[str],
) -> tuple[
    tuple[str, ...],
    tuple[int, ...],
    tuple[tuple[tuple[int, ...], ...], ...],
    tuple[CoreRelationEdge, ...],
    int,
    int,
    int,
]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "edge_id",
            "source_field",
            "target_field",
            "relation_type",
            "direction",
            "lag_blocks",
            "hard",
            "sign",
            "causal",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("field_relation_edges_v1.csv header is incomplete")
        rows = tuple(dict(row) for row in reader)
    if not rows:
        raise ValueError("field relation registry is empty")
    edge_ids = [row["edge_id"] for row in rows]
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("field relation registry contains duplicate edge_id")
    relation_types = tuple(sorted({row["relation_type"] for row in rows}))
    lags_by_type: dict[str, set[int]] = {relation_type: set() for relation_type in relation_types}
    for row in rows:
        if row["hard"] != "false" or row["sign"] != "null" or row["causal"] != "false":
            raise ValueError("V2 relations must remain soft, unsigned, and non-causal")
        if row["direction"] not in {"directed", "undirected"}:
            raise ValueError(f"invalid relation direction: {row['direction']}")
        lag = int(row["lag_blocks"])
        if lag not in {0, 1}:
            raise ValueError("V2 relation lag must be zero or one block")
        lags_by_type[row["relation_type"]].add(lag)
    ambiguous = {
        relation_type: sorted(lags)
        for relation_type, lags in lags_by_type.items()
        if len(lags) != 1
    }
    if ambiguous:
        raise ValueError(f"each relation type must have one frozen lag: {ambiguous}")
    relation_lags = tuple(next(iter(lags_by_type[name])) for name in relation_types)

    field_index = {field: index for index, field in enumerate(core_fields)}
    width = len(core_fields)
    adjacency = [
        [[0 for _ in range(width)] for _ in range(width)]
        for _ in relation_types
    ]
    relation_index = {name: index for index, name in enumerate(relation_types)}
    active_core_edges = 0
    canonical_core_edges: list[CoreRelationEdge] = []
    for row in rows:
        source = row["source_field"]
        target = row["target_field"]
        if source not in field_index or target not in field_index:
            continue
        active_core_edges += 1
        canonical_core_edges.append(
            CoreRelationEdge(
                edge_id=str(row["edge_id"]),
                source_field=source,
                target_field=target,
                relation_type=str(row["relation_type"]),
                direction=str(row["direction"]),
                lag_blocks=int(row["lag_blocks"]),
            )
        )
        relation_id = relation_index[row["relation_type"]]
        # Frozen orientation: an edge source -> target is an attention bias at
        # [relation_type, query=target, key=source].
        adjacency[relation_id][field_index[target]][field_index[source]] = 1
        if row["direction"] == "undirected":
            adjacency[relation_id][field_index[source]][field_index[target]] = 1
    return (
        relation_types,
        relation_lags,
        tuple(tuple(tuple(row) for row in matrix) for matrix in adjacency),
        tuple(canonical_core_edges),
        len(rows),
        active_core_edges,
        len(rows) - active_core_edges,
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array")
    result = tuple(str(item) for item in value)
    if not result or any(not item for item in result) or len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique nonempty strings")
    return result


def _assert_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def _assert_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    result = _integer(value, label)
    if result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _bounded_int(value: Any, minimum: int, maximum: int, label: str) -> int:
    result = _integer(value, label)
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} must be in [{minimum},{maximum}]")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _numeric_state(value: Any, keys: Sequence[str], label: str) -> dict[str, float]:
    state = _mapping(value, label)
    _assert_exact_keys(state, set(keys), label)
    return {key: _number(state[key], f"{label}.{key}") for key in keys}


def _integer_state(value: Any, keys: Sequence[str], label: str) -> dict[str, int]:
    state = _mapping(value, label)
    _assert_exact_keys(state, set(keys), label)
    return {key: _integer(state[key], f"{label}.{key}") for key in keys}


def _numeric_vector(value: Any, keys: Sequence[str], label: str) -> dict[str, float]:
    vector = _mapping(value, label)
    _assert_exact_keys(vector, set(keys), label)
    return {key: _number(vector[key], f"{label}.{key}") for key in keys}


def _integer_vector(value: Any, keys: Sequence[str], label: str) -> dict[str, int]:
    vector = _mapping(value, label)
    _assert_exact_keys(vector, set(keys), label)
    return {key: _integer(vector[key], f"{label}.{key}") for key in keys}
