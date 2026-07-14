from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from trauma_predict.data.multires_event import MultiresEventDataset, SupervisionContract
from trauma_predict.data.multires_event_v2 import (
    LIKELIHOOD_SPECS,
    MultiresEventV2Collator,
    MultiresEventV2Dataset,
    preflight_multires_event_v2,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVISION_PATH = REPO_ROOT / "configs/model/multires_event_v1_supervision.json"
BASE_ROOT = Path(
    os.environ.get(
        "TRAUMA_PREDICT_DATA_ROOT",
        "/mnt/d/Data/trauma_predict_work/multires_event_v1_c4_full_20260712/full",
    )
)
TARGET_ROOT = Path(
    os.environ.get(
        "TRAUMA_PREDICT_V2_TARGET_ROOT",
        "/mnt/d/Data/trauma_predict_work/"
        "multires_event_m4_target_v2_c4_20260713/full_r8",
    )
)


def _is_succeeded(root: Path) -> bool:
    path = root / "dataset_manifest.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "SUCCEEDED"
    except (OSError, json.JSONDecodeError):
        return False


class _IdentityInputNormalizer:
    def transform_event(self, value: object, **_: object) -> tuple[float, bool]:
        if value is None:
            return 0.0, False
        return float(value), True

    def transform_static(self, _: str, value: object) -> tuple[float, bool]:
        if value is None:
            return 0.0, False
        return float(value), True


class _DelegatingBaseDataset:
    def __init__(self, delegate: MultiresEventDataset, root: Path) -> None:
        self._delegate = delegate
        self.root = root

    def __len__(self) -> int:
        return len(self._delegate)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._delegate[index]

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


class _TamperedBaseDataset(_DelegatingBaseDataset):
    def __getitem__(self, index: int) -> dict[str, object]:
        record = copy.deepcopy(self._delegate[index])
        record["content_hash"] = "0" * 64
        return record


@unittest.skipUnless(
    BASE_ROOT.is_dir() and _is_succeeded(TARGET_ROOT),
    "formal V1 input or V2 full_r8 sidecar is not mounted",
)
class MultiresEventV2DataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.supervision = SupervisionContract.from_json(SUPERVISION_PATH)
        cls.base_dataset = MultiresEventDataset(BASE_ROOT, "train", cls.supervision)
        cls.dataset = MultiresEventV2Dataset(cls.base_dataset, TARGET_ROOT)

    def test_full_r8_join_uses_sample_and_both_content_hashes(self) -> None:
        self.assertEqual(len(self.dataset), 37734)
        for index in (0, len(self.dataset) - 1):
            joined = self.dataset[index]
            self.assertEqual(joined["sample_id"], joined["input_record"]["sample_id"])
            self.assertEqual(joined["sample_id"], joined["target_record"]["sample_id"])
            self.assertEqual(
                joined["base_content_hash"], joined["input_record"]["content_hash"]
            )
            self.assertEqual(
                joined["base_content_hash"], joined["target_record"]["base_content_hash"]
            )
            self.assertEqual(
                joined["target_content_hash"], joined["target_record"]["target_content_hash"]
            )
            self.assertEqual(len(joined["target_record"]["blocks"]), 6)
            self.assertTrue(
                all(len(block["processes"]) == 29 for block in joined["target_record"]["blocks"])
            )

    def test_v1_target_supervision_is_not_exposed(self) -> None:
        input_record = self.dataset[0]["input_record"]
        forbidden = {"target_events", "target_mask", "target_source_count", "target_contract"}
        self.assertFalse(forbidden.intersection(input_record))
        self.assertTrue(all(block["side"] == "input" for block in input_record["block_table"]))

    def test_hash_mismatch_fails_the_join(self) -> None:
        tampered_base = _TamperedBaseDataset(self.base_dataset, self.base_dataset.root)
        dataset = MultiresEventV2Dataset(tampered_base, TARGET_ROOT)
        with self.assertRaisesRegex(ValueError, "base_content_hash join mismatch"):
            dataset[0]

    def test_relocated_base_root_uses_hash_identity_not_build_machine_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            relocated_root = Path(directory)
            for filename in ("dataset_manifest.json", "sample_manifest.csv", "subject_split.csv"):
                shutil.copy2(BASE_ROOT / filename, relocated_root / filename)
            relocated = _DelegatingBaseDataset(self.base_dataset, relocated_root)
            dataset = MultiresEventV2Dataset(relocated, TARGET_ROOT)
            self.assertEqual(dataset[0]["sample_id"], self.dataset[0]["sample_id"])

    def test_collator_emits_19_full_grid_primitives_and_unchanged_input_batch(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        collator = MultiresEventV2Collator(
            contract=self.dataset.contract,
            supervision=self.supervision,
            templates=self.base_dataset.templates,
            normalization=_IdentityInputNormalizer(),
        )
        batch = collator([self.dataset[0], self.dataset[1]])
        self.assertEqual(set(batch["target_primitives"]), set(LIKELIHOOD_SPECS))
        self.assertEqual(set(batch["target_primitive_masks"]), set(LIKELIHOOD_SPECS))
        for likelihood_id, tensor in batch["target_primitives"].items():
            self.assertEqual(tuple(tensor.shape[:3]), (2, 6, 29), likelihood_id)
            expected_dtype = (
                torch.float64
                if LIKELIHOOD_SPECS[likelihood_id][0] == "float"
                else torch.long
            )
            self.assertEqual(tensor.dtype, expected_dtype, likelihood_id)
            self.assertEqual(
                tuple(batch["target_primitive_masks"][likelihood_id].shape),
                (2, 6, 29),
                likelihood_id,
            )
        metadata = batch["target_primitive_metadata"]
        self.assertEqual(tuple(metadata["field_order"]), self.dataset.contract.core_fields)
        self.assertEqual(len(metadata["factor_order"]), 414)
        self.assertFalse(metadata["deterministic_projections_have_direct_loss"])
        self.assertEqual(batch["relation_adjacency"].shape, (14, 29, 29))
        self.assertEqual(tuple(batch["relation_type_lags"].shape), (14,))
        self.assertEqual(
            metadata["relation_edge_counts"],
            {"total_registry": 68, "active_core": 50, "deferred_input_or_aux": 18},
        )
        self.assertIn("observed_hours", batch["target_primitive_gates"]["dense_joint_value_state"])
        self.assertIn("compatible_active", batch["target_primitive_gates"]["ned_joint_value_state"])

        input_batch = batch["input_batch"]
        self.assertEqual(tuple(input_batch["event_field_ids"].shape[:1]), (2,))
        forbidden = {
            "target_values",
            "target_raw_values",
            "target_mask",
            "f24_target_raw_values",
            "f24_target_mask",
            "query_field_ids",
        }
        self.assertFalse(forbidden.intersection(input_batch))
        self.assertFalse(forbidden.intersection(batch))

    def test_relation_direction_and_lag_survive_collation(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        collator = MultiresEventV2Collator(
            contract=self.dataset.contract,
            supervision=self.supervision,
            templates=self.base_dataset.templates,
            normalization=_IdentityInputNormalizer(),
        )
        batch = collator([self.dataset[0]])
        metadata = batch["target_primitive_metadata"]
        relation_id = metadata["relation_type_ids"]["support_context"]
        source = metadata["field_index"]["respiratory_support"]
        target = metadata["field_index"]["fio2"]
        self.assertTrue(batch["relation_adjacency"][relation_id, target, source].item())
        self.assertFalse(batch["relation_adjacency"][relation_id, source, target].item())
        self_id = metadata["relation_type_ids"]["self_transition"]
        self.assertEqual(batch["relation_type_lags"][self_id].item(), 1)
        self.assertTrue(
            batch["relation_type_lags"]
            .index_select(
                0,
                batch["relation_type_lags"].new_tensor(
                    [index for index in range(14) if index != self_id]
                ),
            )
            .eq(0)
            .all()
            .item()
        )

    def test_preflight_validates_full_r8_boundary_records(self) -> None:
        result = preflight_multires_event_v2(
            self.base_dataset,
            TARGET_ROOT,
            verify_target_shard_hashes=False,
            verify_all_records=False,
        )
        self.assertEqual(result.dataset_id, "multires_event_m4_target_v2_c4_full_20260713_r8")
        self.assertEqual(result.sample_count, 37734)
        self.assertEqual(result.block_count, 6)
        self.assertEqual(result.core_field_count, 29)
        self.assertEqual(result.enabled_factor_count, 414)
        self.assertEqual(result.relation_total_edges, 68)
        self.assertEqual(result.relation_active_core_edges, 50)
        self.assertEqual(result.relation_deferred_edges, 18)
        self.assertEqual(result.validated_record_count, 2)


if __name__ == "__main__":
    unittest.main()
