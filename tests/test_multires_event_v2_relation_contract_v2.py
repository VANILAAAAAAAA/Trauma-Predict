from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from trauma_predict.data.multires_event_v2.relation_contract import (
    INPUT_TARGET_RELATION_EDGE_COUNT,
    RELATION_HEADER,
    RELATION_FIELD_COUNT,
    TARGET_FIELD_COUNT,
    TARGET_RELATION_EDGE_COUNT,
    MultiresEventV2RelationContract,
    _read_evidence_registry,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs/contracts/multires_event_v2"


def _sum_matrix(matrix: tuple[tuple[int, ...], ...]) -> int:
    return sum(sum(row) for row in matrix)


def test_relation_v2_contract_loads_all_rows_as_independent_parameters() -> None:
    contract = MultiresEventV2RelationContract.from_default_config()

    assert len(contract.history_fields) == RELATION_FIELD_COUNT == 37
    assert len(contract.target_fields) == TARGET_FIELD_COUNT == 29
    assert len(contract.target_edges) == TARGET_RELATION_EDGE_COUNT == 52
    assert len(contract.input_target_edges) == INPUT_TARGET_RELATION_EDGE_COUNT == 39
    assert len(contract.evidence_registry) == 20
    assert len(contract.target_relation_adjacency) == 52
    assert len(contract.input_target_relation_adjacency) == 39
    assert all(_sum_matrix(matrix) == 1 for matrix in contract.target_relation_adjacency)
    assert all(_sum_matrix(matrix) == 1 for matrix in contract.input_target_relation_adjacency)
    assert len(set(contract.target_parameter_keys)) == 52
    assert len(set(contract.input_target_parameter_keys)) == 39
    assert set(contract.target_parameter_keys).isdisjoint(contract.input_target_parameter_keys)
    edge_evidence_ids = {
        edge.evidence_id for edge in contract.target_edges + contract.input_target_edges
    }
    assert len(contract.target_edges + contract.input_target_edges) == 91
    assert edge_evidence_ids == {
        evidence.evidence_id for evidence in contract.evidence_registry
    }


def test_relation_v2_time_scopes_encode_the_frozen_visibility_rules() -> None:
    contract = MultiresEventV2RelationContract.from_default_config()

    assert contract.target_time_scope_ids.count(1) == 29
    assert contract.target_time_scope_ids.count(0) == 23
    assert contract.input_target_time_scope_ids.count(0) == 29
    assert contract.input_target_time_scope_ids.count(1) == 10
    bridges = contract.input_target_edges[:29]
    assert tuple(edge.source_field for edge in bridges) == contract.target_fields
    assert all(edge.source_field == edge.target_field for edge in bridges)
    assert all(
        edge.time_scope == "latest_visible_history_block_to_first_future_block"
        for edge in bridges
    )
    assert all(
        edge.time_scope == "all_visible_history_blocks_to_each_future_block"
        for edge in contract.input_target_edges[29:]
    )


def test_relation_v2_orientation_and_new_edges_are_not_transposed() -> None:
    contract = MultiresEventV2RelationContract.from_default_config()
    target_index = {field: index for index, field in enumerate(contract.target_fields)}
    history_index = {field: index for index, field in enumerate(contract.history_fields)}

    target_edges = {edge.edge_id: edge for edge in contract.target_edges}
    for edge_id in ("tt_resp_support_fio2", "tt_fio2_peep", "tt_map_ned"):
        edge = target_edges[edge_id]
        matrix = contract.target_relation_adjacency[edge.edge_index]
        assert matrix[target_index[edge.target_field]][target_index[edge.source_field]] == 1
        assert _sum_matrix(matrix) == 1

    input_edges = {edge.edge_id: edge for edge in contract.input_target_edges}
    for edge_id in ("it_cxr_resp_support", "it_rbc_hemoglobin", "it_antibiotics_wbc"):
        edge = input_edges[edge_id]
        matrix = contract.input_target_relation_adjacency[edge.edge_index]
        assert matrix[target_index[edge.target_field]][history_index[edge.source_field]] == 1
        assert _sum_matrix(matrix) == 1


def test_relation_v2_preserves_channels_and_parameter_identity() -> None:
    contract = MultiresEventV2RelationContract.from_default_config()
    edge = next(edge for edge in contract.input_target_edges if edge.edge_id == "it_cxr_spo2")
    assert edge.source_channel == "finding_and_observation"
    assert edge.target_channel == "value"
    assert edge.parameter_key == "it.edge.cxr_spo2"
    assert edge.evidence_id == "project_cxr_contract"
    assert "status" not in RELATION_HEADER
    with unittest.TestCase().assertRaisesRegex(ValueError, "target field order"):
        contract.assert_target_field_order(tuple(reversed(contract.target_fields)))


def test_evidence_registry_rejects_duplicate_ids_after_schema_validation() -> None:
    source = CONFIG_DIR / "relation_evidence_registry_v2.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["evidence"][1]["evidence_id"] = payload["evidence"][0]["evidence_id"]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / source.name
        path.write_text(json.dumps(payload), encoding="utf-8")
        with unittest.TestCase().assertRaisesRegex(ValueError, "duplicate evidence_id"):
            _read_evidence_registry(path)


def test_relation_v2_rejects_any_table_mutation_before_compilation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        copied = Path(directory)
        for source in CONFIG_DIR.iterdir():
            shutil.copy2(source, copied / source.name)
        target_path = copied / "target_target_relation_edges_v2.csv"
        lines = target_path.read_text(encoding="utf-8").splitlines()
        target_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        with unittest.TestCase().assertRaisesRegex(ValueError, "file hash mismatch"):
            MultiresEventV2RelationContract.from_config_dir(copied)


def test_relation_v2_rejects_extra_unregistered_config_files() -> None:
    with tempfile.TemporaryDirectory() as directory:
        copied = Path(directory)
        for source in CONFIG_DIR.iterdir():
            shutil.copy2(source, copied / source.name)
        (copied / "unregistered.csv").write_text("x\n", encoding="utf-8")
        with unittest.TestCase().assertRaisesRegex(
            ValueError,
            "exactly the frozen four-file bundle",
        ):
            MultiresEventV2RelationContract.from_config_dir(copied)


class MultiresEventV2RelationContractV2Test(unittest.TestCase):
    test_load = staticmethod(
        test_relation_v2_contract_loads_all_rows_as_independent_parameters
    )
    test_scopes = staticmethod(
        test_relation_v2_time_scopes_encode_the_frozen_visibility_rules
    )
    test_orientation = staticmethod(
        test_relation_v2_orientation_and_new_edges_are_not_transposed
    )
    test_channels = staticmethod(
        test_relation_v2_preserves_channels_and_parameter_identity
    )
    test_evidence = staticmethod(
        test_evidence_registry_rejects_duplicate_ids_after_schema_validation
    )
    test_mutation = staticmethod(
        test_relation_v2_rejects_any_table_mutation_before_compilation
    )
    test_extra_file = staticmethod(
        test_relation_v2_rejects_extra_unregistered_config_files
    )
