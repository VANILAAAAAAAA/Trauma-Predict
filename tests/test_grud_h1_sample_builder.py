from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from trauma_predict.data.grud_h1_sample import (
    EventRegistry,
    GRUDH1SampleBuilder,
    allocate_h1_input_blocks,
    load_joined_authority,
    validate_h1_sample,
)
from trauma_predict.data.grud_h1_sample.io import PointEvent, StayData


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = (
    ROOT / "configs" / "contracts" / "grud_h1_baseline" / "registry_manifest_v1.json"
)
SCHEMA_PATH = ROOT / "schemas" / "grud_h1_baseline_input_sample.schema.json"


class GRUDH1SampleBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = EventRegistry.load(REGISTRY_PATH)

    def test_registry_is_exactly_the_frozen_h1_view(self) -> None:
        templates = self.registry.allowed_templates("H1", "input")
        self.assertEqual(len(self.registry.fields), 37)
        self.assertEqual(len(templates), 118)
        self.assertEqual(len({template.template_id for template in templates}), 118)

    def test_h1_partition_is_hourly_and_ends_at_anchor(self) -> None:
        for prediction_hour in (18, 72, 202, 312):
            allocation = allocate_h1_input_blocks(prediction_hour)
            self.assertEqual(allocation.history_start_hour, 0)
            self.assertEqual(len(allocation.blocks), prediction_hour)
            self.assertEqual(allocation.blocks[0].start_hour, 0)
            self.assertEqual(allocation.blocks[-1].end_hour, prediction_hour)
            self.assertTrue(all(block.span_hours == 1 for block in allocation.blocks))

    def test_builder_keeps_visible_value_and_rejects_delayed_value(self) -> None:
        stay = _synthetic_stay()
        stay.points["heart_rate"] = [
            PointEvent("heart_rate", 1.25, 1.5, 100.0, "beats/min", "chartevents", "1", ""),
            PointEvent("heart_rate", 1.50, 19.0, 200.0, "beats/min", "chartevents", "2", ""),
        ]
        sample = GRUDH1SampleBuilder(self.registry).build(
            stay,
            prediction_hour=18,
            split="train",
            base_content_hash="a" * 64,
            target_content_hash="b" * 64,
            target_shard_key="train-00000",
            target_line_index=0,
        )
        decoded = [
            (*self.registry.decode_ids(event[0], event[1], event[2]), event[3], event[4])
            for event in sample["input_events"]
        ]
        heart_rate_values = [row[3] for row in decoded if row[0] == "heart_rate"]
        self.assertIn(100.0, heart_rate_values)
        self.assertNotIn(200.0, heart_rate_values)
        self.assertEqual(validate_h1_sample(sample, self.registry), [])
        self.assertEqual(sample["input_geometry"]["block_count"], 18)
        self.assertEqual(sample["target_reference"]["target_content_hash"], "b" * 64)

    def test_sample_validates_against_json_schema(self) -> None:
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema optional dependency is absent")
        sample = GRUDH1SampleBuilder(self.registry).build(
            _synthetic_stay(),
            prediction_hour=18,
            split="val",
            base_content_hash="a" * 64,
            target_content_hash="b" * 64,
            target_shard_key="val-00000",
            target_line_index=4,
        )
        jsonschema.Draft202012Validator(
            json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        ).validate(sample)

    def test_base_and_target_manifests_join_in_persisted_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.csv"
            target = root / "target.csv"
            identity = {
                "sample_id": "hadm_2_stay_3_h18",
                "subject_id": "1",
                "hadm_id": "2",
                "stay_id": "3",
                "prediction_hour": "18",
                "split": "train",
            }
            _write_csv(
                base,
                [
                    *identity,
                    "content_hash",
                    "shard_key",
                    "trajectory_path",
                ],
                {**identity, "content_hash": "a" * 64, "shard_key": "train-00000", "trajectory_path": ""},
            )
            _write_csv(
                target,
                [
                    *identity,
                    "base_content_hash",
                    "target_content_hash",
                    "target_shard_key",
                    "target_line_index",
                ],
                {
                    **identity,
                    "base_content_hash": "a" * 64,
                    "target_content_hash": "b" * 64,
                    "target_shard_key": "train-00000",
                    "target_line_index": "0",
                },
            )
            rows = load_joined_authority(base, target)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].base_content_hash, "a" * 64)
            self.assertEqual(rows[0].target_content_hash, "b" * 64)


def _synthetic_stay() -> StayData:
    return StayData(
        source_dir=Path("/synthetic"),
        subject_id="1",
        hadm_id="2",
        stay_id="3",
        sample_key="hadm_2_stay_3",
        icu_intime="2020-01-01 00:00:00",
        icu_outtime="2020-01-05 00:00:00",
        available_until_hour=96.0,
        static={
            "age": 40.0,
            "sex": "M",
            "mechanism": "blunt",
            "transfer": "direct",
            "ed": "yes",
            "head_injury": "no",
        },
        points={},
        intervals={},
        cxr_events=[],
        source_counts={},
    )


def _write_csv(path: Path, fieldnames: list[str], row: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    unittest.main()
