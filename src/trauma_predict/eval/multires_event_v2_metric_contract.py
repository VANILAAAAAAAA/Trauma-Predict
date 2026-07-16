from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from trauma_predict.data.multires_event_v2 import MultiresEventV2RelationContract
from trauma_predict.training.observability import sha256_file


METRIC_CONTRACT_SCHEMA = (
    "trauma_predict.multires_event_v2_relation_v2_metric_contract.v1"
)
METRIC_CONTRACT_VERSION = "2026-07-16-relation-v2-reporting-v1"


def load_trajectory_metric_contract(
    path: str | Path,
    *,
    expected_sha256: str,
    relation_contract: MultiresEventV2RelationContract,
) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"missing V2 trajectory metric contract: {source}")
    if sha256_file(source) != expected_sha256:
        raise ValueError("V2 trajectory metric contract SHA-256 mismatch")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("V2 trajectory metric contract must be a JSON object")
    if payload.get("schema_version") != METRIC_CONTRACT_SCHEMA:
        raise ValueError("V2 trajectory metric contract schema mismatch")
    if payload.get("contract_version") != METRIC_CONTRACT_VERSION:
        raise ValueError("V2 trajectory metric contract version mismatch")
    if payload.get("decision_authority") != "report_only":
        raise ValueError("V2 trajectory metrics cannot make a model-selection decision")
    forbidden = {"bootstrap", "gates", "winner_rule", "comparison"}.intersection(payload)
    if forbidden:
        raise ValueError(f"V2 trajectory metric contract contains decision keys: {forbidden}")

    population = _mapping(payload.get("population"), "population")
    if (
        int(population.get("anchors", -1)) != 6309
        or int(population.get("subjects", -1)) != 505
        or population.get("aggregation")
        != "anchor_mean_within_subject_then_subject_macro"
    ):
        raise ValueError("V2 trajectory metric population drift")
    ensemble = _mapping(payload.get("ancestral_ensemble"), "ancestral_ensemble")
    if (
        int(ensemble.get("trajectories_per_anchor", -1)) != 100
        or int(ensemble.get("common_random_seed", -1)) != 20260713
    ):
        raise ValueError("V2 trajectory ensemble drift")
    vector = _mapping(
        payload.get("standardized_primitive_vector"),
        "standardized_primitive_vector",
    )
    if (
        int(vector.get("blocks", -1)) != 6
        or int(vector.get("coordinates_per_block", -1)) != 160
        or int(vector.get("fields", -1)) != 29
        or float(vector.get("variogram_order", -1.0)) != 0.5
    ):
        raise ValueError("V2 trajectory vector contract drift")
    edge_cover = _mapping(payload.get("relation_edge_cover"), "relation_edge_cover")
    structural_edges = tuple(
        edge
        for edge in relation_contract.target_edges
        if edge.time_scope == "same_future_block_registered_order"
        and edge.source_field != edge.target_field
    )
    if (
        edge_cover.get("source") != "target_target_relation_edges_v2.csv"
        or int(edge_cover.get("expected_edges", -1)) != 23
        or len(structural_edges) != 23
        or len({edge.edge_id for edge in structural_edges}) != 23
    ):
        raise ValueError("V2 trajectory metric relation coverage must be the 23 V2 edges")
    partitions = marginal_encoding_partitions(payload)
    if partitions != {
        "state": frozenset({"binary", "one_hot", "positive_gate"}),
        "value": frozenset(
            {
                "bounded_ratio",
                "bounded_unit",
                "lab_shared_affine_asinh",
                "natural_asinh_nonnegative_integer",
                "ordered_unit",
                "positive_log_robust_affine",
                "robust_affine_asinh",
            }
        ),
    }:
        raise ValueError("V2 trajectory marginal partitions drift")
    return payload


def marginal_encoding_partitions(
    contract: Mapping[str, Any],
) -> dict[str, frozenset[str]]:
    partitions = _mapping(contract.get("marginal_partitions"), "marginal_partitions")
    return {
        name: frozenset(str(value) for value in _mapping(row, name)["encodings"])
        for name, row in partitions.items()
    }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"V2 trajectory metric {label} must be an object")
    return value


__all__ = [
    "METRIC_CONTRACT_SCHEMA",
    "METRIC_CONTRACT_VERSION",
    "load_trajectory_metric_contract",
    "marginal_encoding_partitions",
]
