from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from trauma_predict.training.grud_h1_v2 import (
    EXPECTED_RESUME_CHECKPOINT_SHA256,
    EXPECTED_RESUME_RNG_SEED,
    EXPECTED_RESUME_STEP,
    RESUME_TRAIN_SCHEMA,
    _capture_rng_state,
    _import_resume_metrics_prefix,
    _position_train_iterator_for_step,
    _save_checkpoint,
    validate_grud_h1_v2_configs,
)
from trauma_predict.training.observability import sha256_file


ROOT = Path(__file__).resolve().parents[1]
RESUME_CONFIG = ROOT / "configs/train/p100_grud_h1_joint_m4_v2_resume_2500.yaml"
DATASET_CONFIG = ROOT / "configs/dataset/grud_h1_joint_m4_v2_c4.yaml"
MODEL_CONFIG = ROOT / "configs/model/grud_h1_joint_m4_v2.yaml"


class _Sampler:
    def __init__(self) -> None:
        self.epoch = -1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class _Scaler:
    def state_dict(self) -> dict:
        return {"scale": 32.0, "_growth_tracker": 3000}


class GRUDH1V2ResumeTest(unittest.TestCase):
    def test_resume_config_freezes_global_schedule_and_recovery_identity(self) -> None:
        train = yaml.safe_load(RESUME_CONFIG.read_text(encoding="utf-8"))
        dataset = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
        model = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
        validate_grud_h1_v2_configs(train, dataset, model)
        self.assertEqual(train["schema_version"], RESUME_TRAIN_SCHEMA)
        self.assertFalse(train["training"]["fresh_start"])
        self.assertTrue(train["training"]["resume"])
        self.assertEqual(train["training"]["max_steps"], 4000)
        self.assertEqual(train["training"]["keep_last_checkpoints"], 1)
        self.assertEqual(train["resume_state"]["checkpoint_step"], EXPECTED_RESUME_STEP)
        self.assertEqual(
            train["resume_state"]["checkpoint_sha256"],
            EXPECTED_RESUME_CHECKPOINT_SHA256,
        )
        self.assertEqual(train["resume_state"]["rng_seed"], EXPECTED_RESUME_RNG_SEED)

    def test_step_2500_cursor_is_epoch_52_after_eight_microbatches(self) -> None:
        sampler = _Sampler()
        runtime = SimpleNamespace(train_loader=list(range(96)), train_sampler=sampler)
        epoch, iterator, skipped, steps_per_epoch = _position_train_iterator_for_step(
            runtime,
            global_step=2500,
            accumulation_steps=2,
        )
        self.assertEqual(epoch, 52)
        self.assertEqual(sampler.epoch, 52)
        self.assertEqual(skipped, 8)
        self.assertEqual(steps_per_epoch, 48)
        self.assertEqual(next(iterator), 8)

    def test_cancelled_metrics_import_stops_at_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_metrics = root / "source.jsonl"
            rows = [
                {"event": "training_start", "step": 0},
                {"event": "train_nll", "step": 2400},
                {"event": "interval_validation", "step": 2500},
                {"event": "train_nll", "step": 2600},
                {"event": "interval_validation", "step": 2750},
                {"event": "train_nll", "step": 2900},
            ]
            source_metrics.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            source_manifest = root / "training_manifest.json"
            source_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "trauma_predict.grud_h1_v2_training_manifest.v1",
                        "route": "grud_h1_to_joint_m4_v2",
                        "status": "RUNNING",
                    }
                ),
                encoding="utf-8",
            )
            destination = root / "active.jsonl"
            result = _import_resume_metrics_prefix(
                source_metrics_path=source_metrics,
                source_training_manifest_path=source_manifest,
                expected_metrics_sha256=sha256_file(source_metrics),
                expected_training_manifest_sha256=sha256_file(source_manifest),
                metrics_path=destination,
                resume_step=2500,
            )
            imported = [json.loads(line) for line in destination.read_text().splitlines()]
            self.assertEqual([row["step"] for row in imported], [0, 2400, 2500])
            self.assertEqual(result["discarded_first_step"], 2600)
            self.assertEqual(result["discarded_last_step"], 2900)

    def test_new_checkpoint_contains_exact_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            checkpoint = _save_checkpoint(
                output_dir=output,
                step=3000,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=_Scaler(),
                train_config={"route": "grud_h1_to_joint_m4_v2"},
                dataset_config={"dataset": "frozen"},
                model_config={"model": "frozen"},
                runtime_identity={"identity": "frozen"},
                validation={"joint_nll_subject_macro": 230.0},
                trainer_state={
                    "global_step": 3000,
                    "epoch": 62,
                    "microbatches_consumed_in_epoch": 48,
                    "best_step": 3000,
                    "best_metric": 230.0,
                },
                sampler_state={"epoch": 62, "seed": 20260713},
                rng_state=_capture_rng_state(),
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            manifest = json.loads(checkpoint.with_name("manifest.json").read_text())
            self.assertEqual(payload["schema_version"], "trauma_predict.grud_h1_v2_checkpoint.v2")
            self.assertEqual(payload["trainer_state"]["global_step"], 3000)
            self.assertIn("torch_cpu", payload["rng_state"])
            self.assertTrue(manifest["exact_resume_state"])


if __name__ == "__main__":
    unittest.main()
