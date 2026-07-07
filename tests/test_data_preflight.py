from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from trauma_predict.data.preflight import preflight_training_artifact


class DataPreflightTest(unittest.TestCase):
    def test_preflight_accepts_valid_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_artifact(root)

            result = preflight_training_artifact(self._dataset_config(root))

            self.assertEqual(result.dataset_id, "synthetic-first-train")
            self.assertEqual(result.manifest_samples, 3)
            self.assertEqual(result.split_counts, {"train": 1, "val": 1, "test": 1})

    def test_preflight_rejects_cross_split_subject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                self._record("s1", "101", "201", "301", 48, "train"),
                self._record("s2", "101", "202", "302", 56, "val"),
                self._record("s3", "103", "203", "303", 64, "test"),
            ]
            self._write_artifact(root, rows=rows)

            with self.assertRaisesRegex(ValueError, "multiple splits"):
                preflight_training_artifact(self._dataset_config(root))

    def test_preflight_rejects_manifest_sample_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_artifact(root, manifest_samples=4)

            with self.assertRaisesRegex(ValueError, "sample count mismatch"):
                preflight_training_artifact(self._dataset_config(root))

    def _dataset_config(self, root: Path) -> dict[str, object]:
        return {
            "dataset_manifest": str(root / "dataset_manifest.json"),
            "sample_manifest": str(root / "sample_manifest.csv"),
            "train_shards": str(root / "train" / "*.jsonl.gz"),
            "val_shards": str(root / "val" / "*.jsonl.gz"),
            "test_shards": str(root / "test" / "*.jsonl.gz"),
            "required_sample_fields": [
                "sample_id",
                "subject_id",
                "hadm_id",
                "stay_id",
                "prediction_hour",
                "input_text",
                "target_text",
            ],
        }

    def _write_artifact(
        self,
        root: Path,
        rows: list[dict[str, object]] | None = None,
        manifest_samples: int | None = None,
    ) -> None:
        rows = rows or [
            self._record("s1", "101", "201", "301", 48, "train"),
            self._record("s2", "102", "202", "302", 56, "val"),
            self._record("s3", "103", "203", "303", 64, "test"),
        ]
        shard_paths: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        for split in ("train", "val", "test"):
            split_rows = [row for row in rows if row["split"] == split]
            split_dir = root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            shard_path = split_dir / "shard-00000.jsonl.gz"
            with gzip.open(shard_path, "wt", encoding="utf-8") as handle:
                for row in split_rows:
                    handle.write(json.dumps(row) + "\n")
            shard_paths[split].append(f"{split}/shard-00000.jsonl.gz")

        sample_manifest_path = root / "sample_manifest.csv"
        with sample_manifest_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "sample_id",
                "subject_id",
                "hadm_id",
                "stay_id",
                "prediction_hour",
                "split",
                "shard_path",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row[field] for field in fieldnames})

        manifest = {
            "schema_version": "trauma_predict.dataset_manifest.v1",
            "dataset_id": "synthetic-first-train",
            "sample_unit": "icu_stay_anchor",
            "split_key": "subject_id",
            "created_at": "2026-07-07T00:00:00",
            "source": {
                "ehr_predict_commit": "abcdef1",
                "field_adapter_version": "field_adapter_textual_v1",
                "sample_builder_version": "standard_textual_v1",
            },
            "counts": {
                "subjects": len({row["subject_id"] for row in rows}),
                "hadm": len({row["hadm_id"] for row in rows}),
                "stays": len({row["stay_id"] for row in rows}),
                "samples": manifest_samples if manifest_samples is not None else len(rows),
                "by_split": {
                    split: sum(1 for row in rows if row["split"] == split)
                    for split in ("train", "val", "test")
                },
            },
            "shards": shard_paths,
        }
        (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def _record(
        self,
        sample_id: str,
        subject_id: str,
        hadm_id: str,
        stay_id: str,
        prediction_hour: int,
        split: str,
    ) -> dict[str, object]:
        return {
            "schema": "standard_textual_v1_input_record_v1",
            "dataset_id": "synthetic-first-train",
            "sample_id": sample_id,
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "stay_id": stay_id,
            "prediction_hour": prediction_hour,
            "split": split,
            "input_text": f"input {sample_id}",
            "target_text": f"NEXT_24H target {sample_id}",
            "shard_path": f"{split}/shard-00000.jsonl.gz",
        }


if __name__ == "__main__":
    unittest.main()
