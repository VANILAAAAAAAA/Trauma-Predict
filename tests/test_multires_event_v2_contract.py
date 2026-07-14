from __future__ import annotations

import copy
import gzip
import json
import math
import os
import unittest
from pathlib import Path

from trauma_predict.data.multires_event_v2 import (
    BLOCK_IDS,
    EXPECTED_ENABLED_FACTOR_COUNT,
    MultiresEventV2Contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_ROOT = Path(
    os.environ.get(
        "TRAUMA_PREDICT_V2_TARGET_ROOT",
        "/mnt/d/Data/trauma_predict_work/"
        "multires_event_m4_target_v2_c4_20260713/full_r8",
    )
)
OLD_R2_ROOT = Path(
    "/mnt/d/Data/trauma_predict_work/"
    "multires_event_m4_target_v2_c4_20260713/full_r2"
)


def _is_succeeded(root: Path) -> bool:
    path = root / "dataset_manifest.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "SUCCEEDED"
    except (OSError, json.JSONDecodeError):
        return False


def _first_target(root: Path) -> dict[str, object]:
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    shard = next(iter(manifest["files"]["target_shards"].values()))
    with gzip.open(root / shard["path"], "rt", encoding="utf-8") as handle:
        return json.loads(next(line for line in handle if line.strip()))


def _first_target_with_arithmetic_type(
    root: Path,
    accepted_types: set[str] | None = None,
) -> dict[str, object]:
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    for shard in manifest["files"]["target_shards"].values():
        with gzip.open(root / shard["path"], "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                target = json.loads(line)
                records = target["target_arithmetic_canonicalization"]["records"]
                if records and (
                    accepted_types is None
                    or any(record["type"] in accepted_types for record in records)
                ):
                    return target
    raise AssertionError(f"no r8 arithmetic evidence found for types={accepted_types}")


@unittest.skipUnless(_is_succeeded(TARGET_ROOT), "formal V2 full_r8 sidecar is not ready")
class MultiresEventV2ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = MultiresEventV2Contract.from_dataset_root(TARGET_ROOT)
        cls.first_target = _first_target(TARGET_ROOT)
        cls.arithmetic_target = _first_target_with_arithmetic_type(TARGET_ROOT)
        cls.interior_arithmetic_target = _first_target_with_arithmetic_type(
            TARGET_ROOT,
            {
                "dense_positive_range_mean_lower_inside_one_ulp",
                "dense_positive_range_mean_upper_inside_one_ulp",
            },
        )

    def test_contract_hashes_explicit_topological_order_and_factor_count(self) -> None:
        contract = self.contract
        self.assertEqual(
            contract.process_registry["version"],
            "2026-07-13-r8",
        )
        self.assertEqual(
            contract.registered_core_field_ids,
            (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 7)
            + tuple(range(14, 29))
            + (35,),
        )
        self.assertEqual(len(contract.core_fields), 29)
        self.assertEqual(
            contract.process_registry["scope"]["expanded_enabled_core_primitives"],
            EXPECTED_ENABLED_FACTOR_COUNT,
        )
        self.assertFalse(contract.deterministic_projections_are_supervision)
        self.assertEqual(
            contract.projection_registry["legacy_contract"]["direct_tuple_likelihood"],
            "forbidden",
        )

    def test_target_row_has_exact_six_blocks_and_29_fields(self) -> None:
        self.contract.validate_target_record(self.first_target)
        self.assertEqual(
            tuple(block["block_id"] for block in self.first_target["blocks"]),
            BLOCK_IDS,
        )
        self.assertTrue(
            all(
                set(block["processes"]) == set(self.contract.core_fields)
                for block in self.first_target["blocks"]
            )
        )

    def test_semantic_mutations_are_rejected_independently_of_content_hash(self) -> None:
        missing_block = copy.deepcopy(self.first_target)
        missing_block["blocks"].pop()
        with self.assertRaisesRegex(ValueError, "exactly six"):
            self.contract.validate_target_record(missing_block, verify_content_hash=False)

        missing_field = copy.deepcopy(self.first_target)
        del missing_field["blocks"][0]["processes"]["heart_rate"]
        with self.assertRaisesRegex(ValueError, "exactly 29 core fields"):
            self.contract.validate_target_record(missing_field, verify_content_hash=False)

        changed_contract = copy.deepcopy(self.first_target)
        changed_contract["contract_hashes"]["process"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "contract hashes"):
            self.contract.validate_target_record(changed_contract, verify_content_hash=False)

    def test_content_hash_covers_primitive_truth(self) -> None:
        changed = copy.deepcopy(self.first_target)
        changed["blocks"][0]["processes"]["heart_rate"]["observed_hours"] = 0
        with self.assertRaisesRegex(ValueError, "target_content_hash mismatch"):
            self.contract.validate_target_record(changed, verify_content_hash=True)

    def test_real_r8_arithmetic_evidence_and_semantic_tampering(self) -> None:
        self.contract.validate_target_record(self.arithmetic_target)

        changed_original = copy.deepcopy(self.arithmetic_target)
        changed_record = changed_original["target_arithmetic_canonicalization"]["records"][0]
        changed_record["original_mean"] = changed_record["canonical_mean"]
        with self.assertRaisesRegex(ValueError, "records no value change"):
            self.contract.validate_target_record(changed_original, verify_content_hash=False)

        changed_canonical = copy.deepcopy(self.arithmetic_target)
        changed_record = changed_canonical["target_arithmetic_canonicalization"]["records"][0]
        changed_record["canonical_mean"] = changed_record["original_mean"]
        with self.assertRaisesRegex(ValueError, "canonical MEAN is not persisted"):
            self.contract.validate_target_record(changed_canonical, verify_content_hash=False)

        changed_type = copy.deepcopy(self.arithmetic_target)
        arithmetic = changed_type["target_arithmetic_canonicalization"]
        changed_record = arithmetic["records"][0]
        replacement = (
            "dense_positive_range_mean_lower_inside_one_ulp"
            if changed_record["type"] != "dense_positive_range_mean_lower_inside_one_ulp"
            else "dense_positive_range_mean_upper_inside_one_ulp"
        )
        changed_record["type"] = replacement
        arithmetic["by_type"] = {replacement: arithmetic["count"]}
        with self.assertRaisesRegex(ValueError, "violates its one-ULP rule"):
            self.contract.validate_target_record(changed_type, verify_content_hash=False)

    def test_pre_r8_interior_one_ulp_mean_is_rejected_without_evidence(self) -> None:
        changed = copy.deepcopy(self.interior_arithmetic_target)
        arithmetic = changed["target_arithmetic_canonicalization"]
        evidence = next(
            record
            for record in arithmetic["records"]
            if record["type"]
            in {
                "dense_positive_range_mean_lower_inside_one_ulp",
                "dense_positive_range_mean_upper_inside_one_ulp",
            }
        )
        block = next(
            item for item in changed["blocks"] if item["block_id"] == evidence["block_id"]
        )
        state = block["processes"][evidence["field"]]["value_state"]
        state["mean"] = evidence["original_mean"]
        self.assertTrue(math.isfinite(state["mean"]))
        arithmetic["count"] = 0
        arithmetic["by_type"] = {}
        arithmetic["records"] = []
        with self.assertRaisesRegex(ValueError, "forbidden pre-r8 one-ULP interior MEAN"):
            self.contract.validate_target_record(changed, verify_content_hash=False)

    def test_relation_orientation_and_type_lags_are_not_transposed(self) -> None:
        contract = self.contract
        self.assertEqual(len(contract.relation_types), 14)
        self.assertEqual(
            (
                contract.relation_total_edges,
                contract.relation_active_core_edges,
                contract.relation_deferred_edges,
            ),
            (68, 50, 18),
        )
        relation_id = contract.relation_types.index("support_context")
        source = contract.core_fields.index("respiratory_support")
        target = contract.core_fields.index("fio2")
        self.assertEqual(contract.relation_adjacency[relation_id][target][source], 1)
        self.assertEqual(contract.relation_adjacency[relation_id][source][target], 0)
        self_transition_id = contract.relation_types.index("self_transition")
        self.assertEqual(contract.relation_type_lags[self_transition_id], 1)
        self.assertTrue(
            all(
                lag == 0
                for index, lag in enumerate(contract.relation_type_lags)
                if index != self_transition_id
            )
        )
        structural = tuple(
            edge
            for edge in contract.active_core_relation_edges
            if edge.lag_blocks == 0 and edge.source_field != edge.target_field
        )
        self.assertEqual(len(structural), 21)
        self.assertEqual(len({edge.edge_id for edge in structural}), 21)
        self.assertFalse(any(edge.relation_type == "self_transition" for edge in structural))

    def test_repo_json_schemas_validate_formal_manifest_and_target(self) -> None:
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            self.skipTest("jsonschema dev dependency is unavailable")
        manifest_schema = json.loads(
            (REPO_ROOT / "schemas/multires_event_v2_dataset_manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        target_schema = json.loads(
            (REPO_ROOT / "schemas/multires_event_v2_target.schema.json").read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator.check_schema(manifest_schema)
        Draft202012Validator.check_schema(target_schema)
        manifest = json.loads((TARGET_ROOT / "dataset_manifest.json").read_text(encoding="utf-8"))
        Draft202012Validator(manifest_schema).validate(manifest)
        Draft202012Validator(target_schema).validate(self.first_target)

    @unittest.skipUnless(OLD_R2_ROOT.is_dir(), "superseded full_r2 is not mounted")
    def test_superseded_non_topological_process_contract_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "r8 explicit topological"):
            MultiresEventV2Contract.from_dataset_root(OLD_R2_ROOT)


if __name__ == "__main__":
    unittest.main()
