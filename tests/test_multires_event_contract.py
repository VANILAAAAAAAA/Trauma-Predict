from __future__ import annotations

import json
import os
import unittest
from collections import Counter
from pathlib import Path

from trauma_predict.data.multires_event import (
    MultiresEventDataset,
    SupervisionContract,
    preflight_multires_event,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVISION_PATH = REPO_ROOT / "configs/model/multires_event_v1_supervision.json"
CANONICAL_ROOT = Path(
    os.environ.get(
        "TRAUMA_PREDICT_DATA_ROOT",
        "/mnt/d/Data/trauma_predict_work/multires_event_v1_c4_full_20260712/full",
    )
)


class MultiresEventContractTest(unittest.TestCase):
    def test_supervision_overlay_freezes_the_baseline_counts(self) -> None:
        contract = SupervisionContract.from_json(SUPERVISION_PATH)
        expected = contract.expected_counts
        self.assertEqual(contract.expected_rows, 1314)
        self.assertEqual(expected["primary_direct"], {"H1": 92, "M4": 894, "total": 986})
        self.assertEqual(expected["primary_f24_derived_eval"], 149)
        self.assertEqual(expected["auxiliary_direct"]["total"], 105)
        self.assertEqual(expected["semantic_holdout"]["direct_total"], 51)
        self.assertEqual(contract.excluded_input_field_ids, {9})
        self.assertEqual(contract.auxiliary_field_ids, {29, 30, 31, 32, 33, 34, 36})

    @unittest.skipUnless(CANONICAL_ROOT.is_dir(), "canonical C4 artifact is not mounted")
    def test_canonical_layout_compiles_to_exact_direct_and_derived_contract(self) -> None:
        contract = SupervisionContract.from_json(SUPERVISION_PATH)
        dataset = MultiresEventDataset(CANONICAL_ROOT, "train", contract)
        layout = dataset.target_layout

        self.assertEqual(len(layout.slots), 1314)
        self.assertEqual(len(layout.queries), 986)
        self.assertEqual(len(layout.derived_primary_f24_indices), 149)
        self.assertEqual(len(layout.auxiliary_direct_indices), 105)
        self.assertEqual(len(layout.semantic_holdout_indices), 59)
        self.assertEqual(Counter(slot.resolution for slot in layout.queries), {"M4": 894, "H1": 92})
        self.assertEqual(
            Counter(slot.loss_family for slot in layout.queries),
            {
                "continuous": 472,
                "duration": 210,
                "count": 168,
                "binary": 70,
                "ordinal": 38,
                "nonnegative": 28,
            },
        )
        self.assertEqual(
            dict(layout.vocab_sizes),
            {
                "field": 38,
                "operator": 11,
                "condition": 34,
                "role": 7,
                "resolution": 4,
                "study_slot": 9,
                "static_categorical": 4,
            },
        )
        self.assertEqual(len(layout.static_numeric_fields), 4)
        self.assertEqual(len(layout.static_categorical_fields), 5)
        self.assertEqual(layout.supervision_sha256, contract.source_sha256)
        self.assertTrue(all(slot.primary and slot.active for slot in layout.queries))
        self.assertTrue(all(len(layout.f24_to_m4_indices[index]) == 6 for index in layout.derived_primary_f24_indices))
        point_durations = [
            slot for slot in layout.queries
            if slot.duration_kind == "point_binomial" and slot.condition != "OBSERVED"
        ]
        self.assertTrue(point_durations)
        self.assertTrue(all(slot.coverage_query_position >= 0 for slot in point_durations))

    @unittest.skipUnless(CANONICAL_ROOT.is_dir(), "canonical C4 artifact is not mounted")
    def test_preflight_uses_persisted_eligible_subject_denominators(self) -> None:
        config = json.loads(json.dumps({
            "data": {
                "dataset_id": "multires_event_v1_c4_full_20260712",
                "dataset_fingerprint": "d58d003b6a9b2dd7c1f8d269a1867b534ea475a91118d7d4d44804bee69f9e47",
                "source_fingerprint": "ed578cf6b6e82c96f3aef71d58d6c176c794c9e8fbd37a468a709d64e94739b9",
                "expected_counts": {
                    "samples": 50350,
                    "train": 37734,
                    "val": 6309,
                    "test": 6307,
                    "shards": 52,
                },
            },
            "preflight": {"verify_all_shard_headers": True, "verify_shard_sha256": False},
        }))
        result = preflight_multires_event(config, CANONICAL_ROOT, SUPERVISION_PATH)
        self.assertEqual(result.sample_count, 50350)
        self.assertEqual(dict(result.split_counts), {"train": 37734, "val": 6309, "test": 6307})
        self.assertEqual(dict(result.subject_counts), {"train": 3022, "val": 505, "test": 502})
        self.assertEqual(result.shard_count, 52)


if __name__ == "__main__":
    unittest.main()
