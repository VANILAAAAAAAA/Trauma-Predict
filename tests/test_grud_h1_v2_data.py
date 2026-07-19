from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from trauma_predict.data.grud_h1_sample.builder import content_hash
from trauma_predict.data.grud_h1_v2 import (
    GRUDH1V2Collator,
    GRUDH1V2Dataset,
    H1ChannelRegistry,
)
from trauma_predict.data.multires_event import SupervisionContract
from trauma_predict.data.multires_event_v2 import LIKELIHOOD_SPECS


ROOT = Path(__file__).resolve().parents[1]
SUPERVISION_PATH = ROOT / "configs/model/multires_event_v1_supervision.json"
H1_ROOT = Path(
    os.environ.get(
        "TRAUMA_PREDICT_GRUD_H1_ROOT",
        "/mnt/d/Data/trauma_predict_work/grud_h1_baseline_c4_20260717/full_v1",
    )
)
TARGET_ROOT = Path(
    os.environ.get(
        "TRAUMA_PREDICT_V2_TARGET_ROOT",
        "/mnt/d/Data/trauma_predict_work/"
        "multires_event_m4_target_v2_c4_20260714/full_r9",
    )
)


class _FakeContract:
    def __init__(self, root: Path, manifest: Mapping[str, Any]) -> None:
        self.dataset_root = root.resolve()
        self.manifest = dict(manifest)
        self.contract_bundle_hash = "c" * 64

    def validate_target_record(
        self, record: Mapping[str, Any], *, verify_content_hash: bool
    ) -> None:
        del verify_content_hash
        if record.get("schema") != "synthetic_full_r9":
            raise ValueError("synthetic target schema mismatch")


class _IdentityNormalizer:
    def transform_event(self, value: object, **_: object) -> tuple[float, bool]:
        if value is None:
            return 0.0, False
        return float(value), True

    def transform_static(self, _: str, value: object) -> tuple[float, bool]:
        if value is None:
            return 0.0, False
        return float(value), True


class GRUDH1V2ManifestJoinTest(unittest.TestCase):
    def test_sample_id_and_hash_join_loads_both_cached_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            h1_root, target_root, contract = _write_synthetic_dataset(Path(directory))
            dataset = GRUDH1V2Dataset(
                h1_root,
                target_root,
                split="train",
                contract=contract,
                cache_shards=1,
                strict=True,
                verify_shard_hashes=True,
            )
            self.assertEqual(dataset.sample_ids, ("hadm_2_stay_3_h2",))
            joined = dataset[0]
            self.assertEqual(joined["sample_id"], joined["input_record"]["sample_id"])
            self.assertEqual(joined["sample_id"], joined["target_record"]["sample_id"])
            self.assertEqual(joined["h1_content_hash"], joined["input_record"]["content_hash"])
            self.assertEqual(list(dataset._h1_cache), ["train-00000"])
            self.assertEqual(list(dataset._target_cache), ["train-00000"])
            self.assertEqual(dataset.indices_by_subject(), {"1": (0,)})

    def test_manifest_hash_disagreement_is_rejected_before_shard_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            h1_root, target_root, contract = _write_synthetic_dataset(Path(directory))
            manifest_path = h1_root / "sample_manifest.csv"
            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["target_content_hash"] = "d" * 64
            _write_csv(manifest_path, list(rows[0]), rows)
            dataset_manifest_path = h1_root / "dataset_manifest.json"
            payload = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
            payload["files"]["sample_manifest"]["sha256"] = _sha256_file(manifest_path)
            dataset_manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "H1/full_r9 manifest mismatch"):
                GRUDH1V2Dataset(
                    h1_root,
                    target_root,
                    split="train",
                    contract=contract,
                )


@unittest.skipUnless(
    (H1_ROOT / "dataset_manifest.json").is_file()
    and (TARGET_ROOT / "dataset_manifest.json").is_file(),
    "formal H1 or full_r9 artifact is not mounted",
)
class GRUDH1V2CollatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = GRUDH1V2Dataset(H1_ROOT, TARGET_ROOT, split="train")
        cls.supervision = SupervisionContract.from_json(SUPERVISION_PATH)
        cls.channels = H1ChannelRegistry.from_json(H1_ROOT / "h1_event_templates.json")
        cls.collator = GRUDH1V2Collator(
            contract=cls.dataset.contract,
            supervision=cls.supervision,
            templates=cls.channels.templates,
            normalization=_IdentityNormalizer(),
            channel_registry=cls.channels,
        )

    def test_real_batch_has_hourly_118_channel_input_and_exact_v2_targets(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        batch = self.collator([self.dataset[0], self.dataset[1]])
        inputs = batch["input_batch"]
        self.assertEqual(tuple(inputs["h1_values"].shape), (2, 26, 118))
        self.assertEqual(inputs["h1_observed_mask"].dtype, torch.bool)
        self.assertEqual(tuple(inputs["h1_delta_hours"].shape), (2, 26, 118))
        self.assertEqual(inputs["h1_sequence_mask"][0].sum().item(), 18)
        self.assertFalse(inputs["h1_sequence_mask"][0, 18:].any())
        self.assertFalse(inputs["h1_delta_hours"][0, 18:].any())
        self.assertEqual(set(batch["target_primitives"]), set(LIKELIHOOD_SPECS))
        self.assertEqual(len(batch["target_primitive_metadata"]["factor_order"]), 414)
        for likelihood_id, value in batch["target_primitives"].items():
            self.assertEqual(tuple(value.shape[:3]), (2, 6, 29), likelihood_id)

    def test_unknown_zero_and_cxr_count_have_distinct_masks(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        joined = copy.deepcopy(self.dataset[0])
        regular = next(
            channel
            for channel in self.channels.channels
            if channel.template.value_type != "study_slot"
        )
        unknown = next(
            channel
            for channel in self.channels.channels
            if channel.channel_id != regular.channel_id
            and channel.template.value_type != "study_slot"
        )
        cxr = next(
            channel
            for channel in self.channels.channels
            if channel.template.value_type == "study_slot"
        )
        joined["input_record"]["input_events"] = [
            [*regular.key, 0.0, 0],
            [*cxr.key, 1.0, 0],
            [*cxr.key, 2.0, 0],
        ]
        joined["input_record"]["input_source_count"] = [1, 1, 1]
        batch = self.collator([joined])["input_batch"]
        self.assertTrue(batch["h1_observed_mask"][0, 0, regular.channel_id])
        self.assertEqual(batch["h1_values"][0, 0, regular.channel_id].item(), 0.0)
        self.assertFalse(batch["h1_observed_mask"][0, 0, unknown.channel_id])
        self.assertEqual(batch["h1_values"][0, 0, unknown.channel_id].item(), 0.0)
        self.assertTrue(batch["h1_observed_mask"][0, 0, cxr.channel_id])
        self.assertEqual(batch["h1_values"][0, 0, cxr.channel_id].item(), 2.0)
        self.assertEqual(
            batch["h1_delta_hours"][0, :3, unknown.channel_id].tolist(),
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(
            batch["h1_delta_hours"][0, :3, regular.channel_id].tolist(),
            [1.0, 1.0, 2.0],
        )


def _write_synthetic_dataset(root: Path) -> tuple[Path, Path, _FakeContract]:
    h1_root = root / "h1"
    target_root = root / "target"
    h1_root.mkdir()
    target_root.mkdir()
    identity = {
        "sample_id": "hadm_2_stay_3_h2",
        "subject_id": "1",
        "hadm_id": "2",
        "stay_id": "3",
        "prediction_hour": 2,
        "split": "train",
    }
    base_hash = "a" * 64
    target_hash = "b" * 64
    h1_record: dict[str, Any] = {
        "schema": "grud_h1_baseline_input_sample_v1",
        **identity,
        "sample_key": "hadm_2_stay_3",
        "static": {},
        "input_geometry": {
            "resolution": "H1",
            "prediction_hour": 2,
            "history_start_hour": 0,
            "history_end_hour": 2,
            "history_hours": 2,
            "max_history_hours": 312,
            "block_count": 2,
            "block_id_semantics": "zero_based_chronological_H1_from_history_start",
        },
        "input_events": [],
        "input_source_count": [],
        "source_reference": {
            "base_content_hash": base_hash,
            "target_content_hash": target_hash,
        },
        "target_reference": {
            "sample_id": identity["sample_id"],
            "contract": "multires_event_m4_target_v2_c4_full_20260714_r9",
            "future_blocks": 6,
            "resolution": "M4",
            "stochastic_factors": 414,
            "target_content_hash": target_hash,
            "target_shard_key": "train-00000",
            "target_line_index": 0,
        },
        "registry": {
            "registry_version": "2026-07-12-full-field-v1",
            "resolution": "H1",
            "h1_template_count": 118,
            "tuple_order": ["field_id", "operator_id", "condition_id", "value", "block_id"],
            "padding_id": 0,
        },
    }
    h1_record["content_hash"] = content_hash(h1_record)
    target_record = {
        "schema": "synthetic_full_r9",
        **identity,
        "base_content_hash": base_hash,
        "target_content_hash": target_hash,
    }
    h1_shard = h1_root / "h1_shards/train/train-00000.jsonl.gz"
    target_shard = target_root / "target_shards/train/train-00000.jsonl.gz"
    _write_gzip_rows(h1_shard, [h1_record])
    _write_gzip_rows(target_shard, [target_record])

    target_manifest_path = target_root / "sample_manifest.csv"
    target_row = {
        **{key: str(value) for key, value in identity.items()},
        "base_content_hash": base_hash,
        "target_content_hash": target_hash,
        "target_shard_key": "train-00000",
        "target_line_index": "0",
    }
    _write_csv(target_manifest_path, list(target_row), [target_row])
    target_manifest = {
        "dataset_id": "multires_event_m4_target_v2_c4_full_20260714_r9",
        "files": {
            "sample_manifest": {
                "path": "sample_manifest.csv",
                "sha256": _sha256_file(target_manifest_path),
            },
            "target_shards": {
                "train-00000": {
                    "path": "target_shards/train/train-00000.jsonl.gz",
                    "samples": 1,
                    "sha256": _sha256_file(target_shard),
                }
            },
        },
    }
    (target_root / "dataset_manifest.json").write_text(
        json.dumps(target_manifest), encoding="utf-8"
    )

    h1_manifest_path = h1_root / "sample_manifest.csv"
    h1_row = {
        **{key: str(value) for key, value in identity.items()},
        "base_content_hash": base_hash,
        "target_content_hash": target_hash,
        "h1_content_hash": h1_record["content_hash"],
        "h1_shard_key": "train-00000",
        "h1_line_index": "0",
        "target_shard_key": "train-00000",
        "target_line_index": "0",
    }
    _write_csv(h1_manifest_path, list(h1_row), [h1_row])
    templates_path = h1_root / "h1_event_templates.json"
    templates_path.write_text("{}", encoding="utf-8")
    h1_manifest = {
        "schema": "grud_h1_baseline_dataset_manifest_v1",
        "dataset_id": "grud_h1_baseline_c4_20260717_v1",
        "status": "SUCCEEDED",
        "full_authority_build": True,
        "fingerprint": "e" * 64,
        "input_contract": {
            "resolution": "H1",
            "registered_channels": 118,
            "max_history_hours": 312,
        },
        "target_contract": {
            "dataset_id": "multires_event_m4_target_v2_c4_full_20260714_r9"
        },
        "authority": {
            "target_sample_manifest_sha256": _sha256_file(target_manifest_path)
        },
        "counts": {
            "samples": 1,
            "by_split": {"train": 1},
            "shards": 1,
        },
        "files": {
            "sample_manifest": {
                "path": "sample_manifest.csv",
                "sha256": _sha256_file(h1_manifest_path),
            },
            "h1_event_templates": {
                "path": "h1_event_templates.json",
                "sha256": _sha256_file(templates_path),
            },
            "h1_shards": {
                "train-00000": {
                    "shard_key": "train-00000",
                    "split": "train",
                    "sample_path": "h1_shards/train/train-00000.jsonl.gz",
                    "sample_sha256": _sha256_file(h1_shard),
                    "samples": 1,
                }
            },
        },
    }
    (h1_root / "dataset_manifest.json").write_text(
        json.dumps(h1_manifest), encoding="utf-8"
    )
    return h1_root, target_root, _FakeContract(target_root, target_manifest)


def _write_csv(path: Path, fieldnames: list[str], rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_gzip_rows(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
