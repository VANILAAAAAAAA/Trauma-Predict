from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from trauma_predict.data.multires_event import (
    MultiresEventCollator,
    MultiresEventDataset,
    RobustNormalizer,
    SubjectAnchorDistributedSampler,
    SupervisionContract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVISION_PATH = REPO_ROOT / "configs/model/multires_event_v1_supervision.json"
CANONICAL_ROOT = Path(
    os.environ.get(
        "TRAUMA_PREDICT_DATA_ROOT",
        "/mnt/d/Data/trauma_predict_work/multires_event_v1_c4_full_20260712/full",
    )
)


class _FakeDataset:
    subject_ids = ("s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4")
    shard_keys = ("a", "b", "a", "b", "c", "c", "b", "c")

    def __len__(self) -> int:
        return len(self.subject_ids)


class MultiresEventSamplerTest(unittest.TestCase):
    def test_training_selects_one_anchor_per_subject_every_epoch(self) -> None:
        sampler = SubjectAnchorDistributedSampler(
            _FakeDataset(), seed=17, mode="subject_uniform", pad_to_world_size=False
        )
        for epoch in (0, 1, 9):
            sampler.set_epoch(epoch)
            indices = list(sampler)
            self.assertEqual(len(indices), 4)
            self.assertEqual(len({_FakeDataset.subject_ids[index] for index in indices}), 4)
            shard_runs = []
            for index in indices:
                shard = _FakeDataset.shard_keys[index]
                if not shard_runs or shard_runs[-1] != shard:
                    shard_runs.append(shard)
            self.assertEqual(len(shard_runs), len(set(shard_runs)))
        self.assertEqual(sampler.active_mode, "subject_uniform")

    def test_ddp_partition_has_no_padding_or_cross_rank_duplicates(self) -> None:
        left = SubjectAnchorDistributedSampler(
            _FakeDataset(), rank=0, world_size=2, seed=17,
            mode="subject_uniform", pad_to_world_size=False, require_even_divisible=True,
        )
        right = SubjectAnchorDistributedSampler(
            _FakeDataset(), rank=1, world_size=2, seed=17,
            mode="subject_uniform", pad_to_world_size=False, require_even_divisible=True,
        )
        left_indices, right_indices = list(left), list(right)
        self.assertEqual(len(left_indices), 2)
        self.assertEqual(len(right_indices), 2)
        self.assertFalse(set(left_indices) & set(right_indices))
        self.assertEqual(len(set(left_indices + right_indices)), 4)

    def test_fixed_eval_anchor_does_not_change_with_epoch(self) -> None:
        sampler = SubjectAnchorDistributedSampler(
            _FakeDataset(), seed=17, mode="one_fixed_per_subject",
            shuffle=False, pad_to_world_size=False,
        )
        initial = list(sampler)
        sampler.set_epoch(12)
        self.assertEqual(initial, list(sampler))
        state = sampler.state_dict()
        restored = SubjectAnchorDistributedSampler(
            _FakeDataset(), seed=17, mode="one_fixed_per_subject",
            shuffle=False, pad_to_world_size=False,
        )
        restored.load_state_dict(state)
        self.assertEqual(list(restored), initial)

    def test_odd_training_denominator_is_rejected_instead_of_padded(self) -> None:
        sampler = SubjectAnchorDistributedSampler(
            _FakeDataset(), rank=0, world_size=2, seed=17,
            mode="subject_uniform", pad_to_world_size=False,
            require_even_divisible=True, max_subjects=3,
        )
        with self.assertRaisesRegex(ValueError, "not divisible"):
            list(sampler)


@unittest.skipUnless(CANONICAL_ROOT.is_dir(), "canonical C4 artifact is not mounted")
class MultiresEventDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.supervision = SupervisionContract.from_json(SUPERVISION_PATH)
        cls.dataset = MultiresEventDataset(CANONICAL_ROOT, "train", cls.supervision)

    def test_dataset_filters_gcs_verbal_and_preserves_parallel_arrays(self) -> None:
        record = self.dataset[0]
        self.assertFalse(any(int(event[0]) == 9 for event in record["input_events"]))
        self.assertEqual(len(record["input_events"]), len(record["input_source_count"]))
        self.assertEqual(len(record["target_events"]), 1314)
        self.assertEqual(len(record["target_mask"]), 1314)

    def test_train_only_fit_blocks_validation_leakage(self) -> None:
        record = dict(self.dataset[0])
        record["split"] = "val"
        with self.assertRaisesRegex(ValueError, "leakage blocked"):
            RobustNormalizer.fit(
                [record],
                templates=self.dataset.templates,
                target_layout=self.dataset.target_layout,
                supervision=self.supervision,
                dataset_fingerprint=self.dataset.dataset_fingerprint,
                max_values_per_key=100,
            )

    def test_collator_emits_model_queries_active_targets_and_f24_eval_metadata(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        record = self.dataset[0]
        normalizer = RobustNormalizer.fit(
            [record],
            templates=self.dataset.templates,
            target_layout=self.dataset.target_layout,
            supervision=self.supervision,
            dataset_fingerprint=self.dataset.dataset_fingerprint,
            max_values_per_key=100,
        )
        collator = MultiresEventCollator(
            supervision=self.supervision,
            templates=self.dataset.templates,
            target_layout=self.dataset.target_layout,
            normalization=normalizer,
        )
        batch = collator([record])
        self.assertEqual(tuple(batch["query_field_ids"].shape), (1, 986))
        self.assertEqual(tuple(batch["query_operator_ids"].shape), (1, 986))
        self.assertEqual(tuple(batch["query_condition_ids"].shape), (1, 986))
        self.assertEqual(tuple(batch["target_values"].shape), (1, 986))
        self.assertEqual(tuple(batch["target_raw_values"].shape), (1, 986))
        self.assertEqual(tuple(batch["target_mask"].shape), (1, 986))
        self.assertEqual(tuple(batch["f24_target_raw_values"].shape), (1, 149))
        self.assertEqual(tuple(batch["f24_target_mask"].shape), (1, 149))
        self.assertEqual(tuple(batch["f24_m4_query_positions"].shape), (149, 6))
        self.assertFalse(batch["event_field_ids"].eq(9).any().item())
        self.assertEqual(tuple(batch["static_numeric"].shape), (1, 4))
        self.assertEqual(tuple(batch["static_categorical"].shape), (1, 5))
        self.assertEqual(batch["target_loss_family_ids"].numel(), 986)
        self.assertEqual(batch["target_semantic_component_ids"].numel(), 986)
        self.assertEqual(len(normalizer.fallback_event_keys), sum(normalizer.fallback_level_counts.values()))
        for source_index in self.dataset.target_layout.derived_primary_f24_indices:
            slot = self.dataset.target_layout.slots[source_index]
            template = self.dataset.templates.get(
                slot.field_id, slot.operator_id, slot.condition_id
            )
            if slot.loss_family not in {"ordinal", "count"}:
                self.assertTrue(normalizer.has_event_stat(template, "F24"), slot.slot_id)

    def test_normalization_round_trip_and_atomic_persistence(self) -> None:
        record = self.dataset[0]
        normalizer = RobustNormalizer.fit(
            [record],
            templates=self.dataset.templates,
            target_layout=self.dataset.target_layout,
            supervision=self.supervision,
            dataset_fingerprint=self.dataset.dataset_fingerprint,
            max_values_per_key=100,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normalization.json"
            normalizer.save_json(path)
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_suffix(".json.tmp").exists())
            loaded = RobustNormalizer.from_json(
                path,
                expected_dataset_fingerprint=self.dataset.dataset_fingerprint,
                expected_supervision_sha256=self.supervision.source_sha256,
            )
            self.assertEqual(loaded.subject_ids_sha256, normalizer.subject_ids_sha256)
            self.assertEqual(loaded.fallback_event_keys, normalizer.fallback_event_keys)
            self.assertEqual(loaded.fallback_level_counts, normalizer.fallback_level_counts)


if __name__ == "__main__":
    unittest.main()
