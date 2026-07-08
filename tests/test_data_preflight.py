from __future__ import annotations

import csv
import copy
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

    def test_preflight_rejects_cross_split_subject_in_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_rows = [
                self._record("s1", "101", "201", "301", 48, "train"),
                self._record("s2", "101", "202", "302", 56, "val"),
                self._record("s3", "103", "203", "303", 64, "test"),
            ]
            manifest_rows = copy.deepcopy(shard_rows)
            manifest_rows[1]["subject_id"] = "102"
            self._write_artifact(root, rows=shard_rows, manifest_rows=manifest_rows)

            with self.assertRaisesRegex(ValueError, "multiple splits"):
                preflight_training_artifact(self._dataset_config(root))

    def test_preflight_rejects_manifest_shard_metadata_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                self._record("s1", "101", "201", "301", 48, "train"),
                self._record("s2", "102", "202", "302", 56, "val"),
                self._record("s3", "103", "203", "303", 64, "test"),
            ]
            manifest_rows = copy.deepcopy(rows)
            manifest_rows[0]["hadm_id"] = "999"
            self._write_artifact(root, rows=rows, manifest_rows=manifest_rows)

            with self.assertRaisesRegex(ValueError, "sample_manifest and shards disagree on sample metadata"):
                preflight_training_artifact(self._dataset_config(root))

    def test_preflight_rejects_shard_path_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                self._record("s1", "101", "201", "301", 48, "train"),
                self._record("s2", "102", "202", "302", 56, "val"),
                self._record("s3", "103", "203", "303", 64, "test"),
            ]
            manifest_rows = copy.deepcopy(rows)
            manifest_rows[0]["shard_path"] = "train/wrong-shard.jsonl.gz"
            self._write_artifact(root, rows=rows, manifest_rows=manifest_rows)

            with self.assertRaisesRegex(ValueError, "sample_manifest and shards disagree on sample metadata"):
                preflight_training_artifact(self._dataset_config(root))

    def test_preflight_rejects_duplicate_clinical_primary_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                self._record("s1", "101", "201", "301", 48, "train"),
                self._record("s2", "101", "201", "301", 48, "train"),
                self._record("s3", "102", "202", "302", 56, "val"),
                self._record("s4", "103", "203", "303", 64, "test"),
            ]
            self._write_artifact(root, rows=rows)

            with self.assertRaisesRegex(ValueError, "duplicate clinical primary keys"):
                preflight_training_artifact(self._dataset_config(root))

    def _dataset_config(self, root: Path) -> dict[str, object]:
        return {
            "dataset_manifest": str(root / "dataset_manifest.json"),
            "sample_manifest": str(root / "sample_manifest.csv"),
            "train_shards": str(root / "train" / "*.jsonl.gz"),
            "val_shards": str(root / "val" / "*.jsonl.gz"),
            "test_shards": str(root / "test" / "*.jsonl.gz"),
            "required_sample_fields": [
                "schema",
                "route",
                "sample_id",
                "subject_id",
                "hadm_id",
                "stay_id",
                "prediction_hour",
                "split",
                "input_text",
                "hour_value_order",
                "hour_placeholders",
                "hour_values",
                "hour_mask",
                "hour_vent",
                "targets",
                "target_text",
            ],
        }

    def _write_artifact(
        self,
        root: Path,
        rows: list[dict[str, object]] | None = None,
        manifest_rows: list[dict[str, object]] | None = None,
        manifest_samples: int | None = None,
    ) -> None:
        rows = rows or [
            self._record("s1", "101", "201", "301", 48, "train"),
            self._record("s2", "102", "202", "302", 56, "val"),
            self._record("s3", "103", "203", "303", 64, "test"),
        ]
        manifest_rows = manifest_rows or rows
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
            for row in manifest_rows:
                writer.writerow({
                    field: (row.get(field) or f"{row['split']}/shard-00000.jsonl.gz")
                    for field in fieldnames
                })

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
            "schema": "standard_textual_v1_main_record_v2",
            "route": "main_hour_adapter_structured_heads",
            "dataset_id": "synthetic-first-train",
            "sample_id": sample_id,
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "stay_id": stay_id,
            "prediction_hour": prediction_hour,
            "split": split,
            "input_text": (
                "<SAMPLE>\n"
                "schema=icu_state_major_textual_v1\n"
                f"sample_id={sample_id}\n"
                "\nSTATIC:\n"
                "static{age=70;sex=F;early48{lactate=high}}\n"
                "\nDAY:\n"
                "D0 i=0 len=24 resp{spo2_min=low} dq{vital=dense;lab=drawn;uop=measured}\n"
                "\nHOUR len=2:\n"
                "<H-01> <H0>\n"
                "\n<STATE>\n"
                "</SAMPLE>"
            ),
            "hour_value_order": ["hr", "sbp", "dbp", "map", "rr", "temp", "spo2"],
            "hour_placeholders": ["<H-01>", "<H0>"],
            "hour_values": [
                [90.0, 120.0, None, 78.0, 22.0, 37.2, 95.0],
                [92.0, 118.0, 65.0, 80.0, 24.0, 37.4, 94.0],
            ],
            "hour_mask": [
                [1, 1, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ],
            "hour_vent": [[0], [1]],
            "targets": {
                "next_hour": {
                    "label": "NEXT_HOUR",
                    "relative_hour": "H+1",
                    "value_order": ["hr", "sbp", "dbp", "map", "rr", "temp", "spo2"],
                    "values": {
                        "hr": 93.0,
                        "sbp": 116.0,
                        "dbp": 64.0,
                        "map": 79.0,
                        "rr": 25.0,
                        "temp": 37.5,
                        "spo2": 93.0,
                    },
                    "mask": {"hr": 1, "sbp": 1, "dbp": 1, "map": 1, "rr": 1, "temp": 1, "spo2": 1},
                    "hour_values": [93.0, 116.0, 64.0, 79.0, 25.0, 37.5, 93.0],
                    "hour_mask": [1, 1, 1, 1, 1, 1, 1],
                    "vent_on": 1,
                    "hour_vent": [1],
                },
                "next24h": {
                    "label": "NEXT_24H",
                    "len_hours": 24,
                    "sections": {
                        "shock": {"map_low_hours": "brief"},
                        "resp": {"vent_hours": "partial_window", "spo2_min": "low"},
                        "tx": {"antibiotics": "present"},
                    },
                },
            },
            "target_text": (
                "<FORECAST_REPORT>\n"
                "NEXT_HOUR:\n"
                "H+1|HR=93|HR_obs=1|SBP=116|SBP_obs=1|DBP=64|DBP_obs=1|MAP=79|MAP_obs=1|RR=25|RR_obs=1|TEMP=37.5|TEMP_obs=1|SpO2=93|SpO2_obs=1|VENT=1\n"
                "\nNEXT_24H:\n"
                "NEXT_24H len=24 shock{map_low_hours=brief} resp{vent_hours=partial_window;spo2_min=low} tx{antibiotics=present}\n"
                "</FORECAST_REPORT>"
            ),
            "shard_path": f"{split}/shard-00000.jsonl.gz",
        }


if __name__ == "__main__":
    unittest.main()
