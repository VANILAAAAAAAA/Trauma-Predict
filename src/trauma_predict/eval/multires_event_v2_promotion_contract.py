from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from trauma_predict.data.multires_event_v2 import MultiresEventV2Contract
from trauma_predict.training.observability import sha256_file


PROMOTION_CONTRACT_SCHEMA = (
    "trauma_predict.multires_event_v2_promotion_metric_contract.v2"
)
PROMOTION_CONTRACT_VERSION = "2026-07-13-structural-promotion-v2"


def load_promotion_metric_contract(
    path: str | Path,
    *,
    expected_sha256: str,
    data_contract: MultiresEventV2Contract | None = None,
) -> dict[str, Any]:
    """Load the byte-hashed decision contract and fail closed on semantic drift."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"missing V2 promotion metric contract: {source}")
    if sha256_file(source) != expected_sha256:
        raise ValueError("V2 promotion metric contract SHA-256 mismatch")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("V2 promotion metric contract must be a JSON object")
    if payload.get("schema_version") != PROMOTION_CONTRACT_SCHEMA:
        raise ValueError("V2 promotion metric contract schema mismatch")
    if payload.get("contract_version") != PROMOTION_CONTRACT_VERSION:
        raise ValueError("V2 promotion metric contract version mismatch")

    population = _mapping(payload.get("population"), "population")
    if (
        int(population.get("anchors", -1)) != 6309
        or int(population.get("subjects", -1)) != 505
        or population.get("aggregation")
        != "anchor_mean_within_subject_then_subject_macro"
    ):
        raise ValueError("V2 promotion population contract drift")
    ensemble = _mapping(payload.get("ancestral_ensemble"), "ancestral_ensemble")
    if (
        int(ensemble.get("trajectories_per_anchor", -1)) != 100
        or int(ensemble.get("common_random_seed", -1)) != 20260713
    ):
        raise ValueError("V2 promotion ancestral ensemble contract drift")
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
        raise ValueError("V2 promotion standardized vector contract drift")

    edge_cover = _mapping(payload.get("relation_edge_cover"), "relation_edge_cover")
    if int(edge_cover.get("expected_edges", -1)) != 21:
        raise ValueError("V2 promotion relation edge count must be 21")
    partitions = _mapping(payload.get("marginal_partitions"), "marginal_partitions")
    expected_partitions = {
        "state": (
            {"binary", "one_hot", "positive_gate"},
            16,
        ),
        "value": (
            {
                "bounded_ratio",
                "bounded_unit",
                "lab_shared_affine_asinh",
                "natural_asinh_nonnegative_integer",
                "ordered_unit",
                "positive_log_robust_affine",
                "robust_affine_asinh",
            },
            144,
        ),
    }
    observed_encodings: set[str] = set()
    for name, (encodings, expected_count) in expected_partitions.items():
        row = _mapping(partitions.get(name), f"marginal_partitions.{name}")
        observed = {str(value) for value in row.get("encodings", ())}
        if observed != encodings or int(
            row.get("expected_coordinates_per_block", -1)
        ) != expected_count:
            raise ValueError(f"V2 promotion marginal partition {name!r} drift")
        if observed_encodings & observed:
            raise ValueError("V2 promotion marginal partitions overlap")
        observed_encodings.update(observed)

    bootstrap = _mapping(payload.get("bootstrap"), "bootstrap")
    if (
        bootstrap.get("unit") != "subject_id"
        or int(bootstrap.get("repetitions", -1)) != 2000
        or int(bootstrap.get("seed", -1)) != 20260713
        or bootstrap.get("shared_subject_index_schedule") is not True
        or float(bootstrap.get("lower_quantile", -1.0)) != 0.025
        or float(bootstrap.get("upper_quantile", -1.0)) != 0.975
    ):
        raise ValueError("V2 promotion bootstrap contract drift")
    gates = _mapping(payload.get("gates"), "gates")
    trajectory = _mapping(gates.get("trajectory_over_block"), "trajectory gate")
    relational = _mapping(
        gates.get("relational_over_trajectory"), "relational gate"
    )
    expected_trajectory = {
        "teacher_joint_nll_delta_ci95_upper_lt": 0.0,
        "temporal_score_observed_ratio_lte": 0.98,
        "temporal_score_ci95_upper_lt": 1.0,
        "value_marginal_score_ci95_upper_lt": 1.01,
        "state_marginal_score_ci95_upper_lt": 1.01,
    }
    expected_relational = {
        "teacher_joint_nll_delta_ci95_upper_lt": 0.0,
        "relation_score_observed_ratio_lte": 0.99,
        "relation_score_ci95_upper_lt": 1.0,
        "value_marginal_score_ci95_upper_lt": 1.01,
        "state_marginal_score_ci95_upper_lt": 1.01,
    }
    if dict(trajectory) != expected_trajectory or dict(relational) != expected_relational:
        raise ValueError("V2 promotion decision thresholds drift")
    hard = _mapping(gates.get("hard"), "hard gate")
    if (
        float(hard.get("coherence_rate", -1.0)) != 1.0
        or hard.get("missing_primary_endpoint") != "invalid"
        or hard.get("nonpositive_ratio_denominator") != "invalid"
        or hard.get("physical_conditional_coverage_is_veto") is not False
    ):
        raise ValueError("V2 promotion hard-gate contract drift")

    if data_contract is not None:
        structural_edges = tuple(
            edge
            for edge in data_contract.active_core_relation_edges
            if edge.lag_blocks == 0 and edge.source_field != edge.target_field
        )
        if len(structural_edges) != int(edge_cover["expected_edges"]):
            raise ValueError("V2 promotion contract does not cover exact core relation rows")
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
        raise ValueError(f"V2 promotion {label} must be an object")
    return value


__all__ = [
    "PROMOTION_CONTRACT_SCHEMA",
    "PROMOTION_CONTRACT_VERSION",
    "load_promotion_metric_contract",
    "marginal_encoding_partitions",
]
