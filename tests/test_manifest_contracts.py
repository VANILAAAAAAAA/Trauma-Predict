from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from trauma_predict.data.manifest import validate_dataset_manifest
from trauma_predict.data.splits import assert_patient_level_split
from trauma_predict.training.config import expand_env


class ManifestContractTest(unittest.TestCase):
    def test_dataset_manifest_contract_accepts_expected_shape(self) -> None:
        payload = {
            "schema_version": "trauma_predict.dataset_manifest.v1",
            "dataset_id": "first_train_001",
            "sample_unit": "icu_stay_anchor",
            "split_key": "subject_id",
            "created_at": "2026-07-07T00:00:00Z",
            "source": {
                "ehr_predict_commit": "abcdef1",
                "field_adapter_version": "field_adapter_textual_v1",
                "sample_builder_version": "standard_textual_v1",
            },
            "counts": {"subjects": 10, "hadm": 11, "stays": 12, "samples": 100},
            "shards": {"train": ["train/shard-00000.jsonl.zst"], "val": [], "test": []},
        }
        validate_dataset_manifest(payload)

    def test_patient_level_split_rejects_cross_split_subject(self) -> None:
        rows = [
            {"subject_id": "1", "split": "train"},
            {"subject_id": "1", "split": "val"},
        ]
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            assert_patient_level_split(rows)

    def test_environment_expansion_keeps_unknown_variables(self) -> None:
        os.environ.pop("TRAUMA_PREDICT_UNKNOWN", None)
        expanded = expand_env({"path": "${TRAUMA_PREDICT_UNKNOWN}/x"})
        self.assertEqual(expanded, {"path": "${TRAUMA_PREDICT_UNKNOWN}/x"})

    def test_schema_files_are_valid_json(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for path in (root / "schemas").glob("*.schema.json"):
            self.assertIn("$schema", json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
