from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from trauma_predict.data.records import load_text_records, resolve_shard_paths
from trauma_predict.training.seq2seq import validate_seq2seq_config


class TrainingSeq2SeqTest(unittest.TestCase):
    def test_text_records_load_from_gzip_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "train"
            split_dir.mkdir()
            shard = split_dir / "shard-00000.jsonl.gz"
            row = {
                "sample_id": "s1",
                "subject_id": "p1",
                "hadm_id": "h1",
                "stay_id": "st1",
                "prediction_hour": 48,
                "input_text": "<SAMPLE> input",
                "target_text": "NEXT_24H target",
            }
            with gzip.open(shard, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")

            config = {"train_shards": str(split_dir / "*.jsonl.gz")}
            paths = resolve_shard_paths(config, "train")
            records = load_text_records(paths, ["sample_id", "input_text", "target_text"])

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["sample_id"], "s1")
        self.assertEqual(records[0]["target_text"], "NEXT_24H target")

    def test_seq2seq_config_rejects_encoder_only_model(self) -> None:
        config = {
            "schema_version": "trauma_predict.train_config.v1",
            "model": {
                "base_model": "distilbert-base-uncased",
                "task": "next24_text_generation",
                "max_input_tokens": 1024,
                "max_target_tokens": 256,
            },
            "training": {},
        }

        with self.assertRaisesRegex(ValueError, "encoder-only"):
            validate_seq2seq_config(config)

    def test_seq2seq_config_accepts_flan_t5(self) -> None:
        config = {
            "schema_version": "trauma_predict.train_config.v1",
            "model": {
                "base_model": "google/flan-t5-base",
                "task": "next24_text_generation",
                "max_input_tokens": 1024,
                "max_target_tokens": 256,
            },
            "training": {},
        }

        validate_seq2seq_config(config)


if __name__ == "__main__":
    unittest.main()
