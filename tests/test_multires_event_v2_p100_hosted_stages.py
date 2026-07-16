from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from trauma_predict.training.multires_event_v2 import (
    HOSTED_STOP_READINESS_SCHEMA,
    _hosted_free_running_max_new_anchors,
    _hosted_verification_stop_step,
    _materialize_hosted_stop_readiness,
    _validate_final_teacher_rows,
)


class P100HostedStageContractTest(unittest.TestCase):
    def test_hosted_training_stop_schedule_is_closed_and_monotone(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TRAUMA_PREDICT_V2_HOSTED_STOP_STEP": "250",
                "TRAUMA_PREDICT_V2_HOSTED_VERIFY_STOP_AFTER_FORMAL_STEP2": "0",
            },
            clear=False,
        ):
            self.assertEqual(_hosted_verification_stop_step(starting_global_step=0), 250)
        with patch.dict(
            os.environ,
            {
                "TRAUMA_PREDICT_V2_HOSTED_STOP_STEP": "1500",
                "TRAUMA_PREDICT_V2_HOSTED_VERIFY_STOP_AFTER_FORMAL_STEP2": "0",
            },
            clear=False,
        ):
            self.assertEqual(
                _hosted_verification_stop_step(starting_global_step=250), 1500
            )
        for value in ("251", "-1", "bad"):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {
                    "TRAUMA_PREDICT_V2_HOSTED_STOP_STEP": value,
                    "TRAUMA_PREDICT_V2_HOSTED_VERIFY_STOP_AFTER_FORMAL_STEP2": "0",
                },
                clear=False,
            ), self.assertRaises(ValueError):
                _hosted_verification_stop_step(starting_global_step=0)
        with patch.dict(
            os.environ,
            {
                "TRAUMA_PREDICT_V2_HOSTED_STOP_STEP": "250",
                "TRAUMA_PREDICT_V2_HOSTED_VERIFY_STOP_AFTER_FORMAL_STEP2": "0",
            },
            clear=False,
        ), self.assertRaises(ValueError):
            _hosted_verification_stop_step(starting_global_step=250)

    def test_free_running_invocation_limit_is_orchestration_only(self) -> None:
        for raw, expected in (("0", None), ("2000", 2000), ("", None)):
            with self.subTest(raw=raw), patch.dict(
                os.environ,
                {"TRAUMA_PREDICT_V2_FREE_RUNNING_MAX_NEW_ANCHORS": raw},
                clear=False,
            ):
                self.assertEqual(_hosted_free_running_max_new_anchors(), expected)
        for raw in ("-1", "nope"):
            with self.subTest(raw=raw), patch.dict(
                os.environ,
                {"TRAUMA_PREDICT_V2_FREE_RUNNING_MAX_NEW_ANCHORS": raw},
                clear=False,
            ), self.assertRaises(ValueError):
                _hosted_free_running_max_new_anchors()

    def test_step250_readiness_reopens_best_and_binds_checkpoint(self) -> None:
        identity = {"source_tree": "a" * 64}
        interval = {
            "phase": "interval",
            "step": 250,
            "samples": 6309,
            "joint_nll_subject_macro": 1.25,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoints/checkpoint-00000250"
            checkpoint.mkdir(parents=True)
            manifest_path = checkpoint / "checkpoint_manifest.json"
            manifest_path.write_text("{}\n", encoding="utf-8")
            with (
                patch(
                    "trauma_predict.training.multires_event_v2._validate_v2_checkpoint_directory",
                    return_value={"sha256": {"model.pt": "b" * 64}},
                ),
                patch(
                    "trauma_predict.training.multires_event_v2._load_v2_best_model",
                    return_value={
                        "selected_checkpoint_model_sha256": "c" * 64,
                    },
                ) as reopen,
            ):
                readiness = _materialize_hosted_stop_readiness(
                    output_dir=root,
                    model=torch.nn.Linear(1, 1),
                    device=torch.device("cpu"),
                    identity_hashes=identity,
                    stop_step=250,
                    best_step=250,
                    interval_evaluation=interval,
                )
            self.assertEqual(readiness["schema_version"], HOSTED_STOP_READINESS_SCHEMA)
            self.assertEqual(readiness["status"], "PASSED")
            self.assertEqual(readiness["checkpoint_model_sha256"], "b" * 64)
            self.assertEqual(readiness["best_model_sha256"], "c" * 64)
            reopen.assert_called_once()
            stable = root / "formal_hosted_stop_readiness.json"
            history = root / "hosted_stages/step-00000250.json"
            self.assertEqual(stable.read_bytes(), history.read_bytes())

    def test_final_teacher_rows_are_exactly_identity_and_sample_bound(self) -> None:
        identity = {"selected_checkpoint_model_sha256": "d" * 64}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            rows = [
                {
                    "sample_id": sample_id,
                    "subject_id": f"subject-{index}",
                    "joint_nll": 1.0 + index,
                    "primitive_factors": 414,
                    "model_contract": "relation_v2",
                    "step": 4000,
                    "identity": identity,
                }
                for index, sample_id in enumerate(("sample-a", "sample-b"))
            ]
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            evidence = _validate_final_teacher_rows(
                path,
                expected_sample_ids=("sample-b", "sample-a"),
                step=4000,
                evaluation_identity=identity,
            )
            self.assertEqual(evidence["row_count"], 2)
            rows[1]["sample_id"] = "sample-a"
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _validate_final_teacher_rows(
                    path,
                    expected_sample_ids=("sample-a", "sample-b"),
                    step=4000,
                    evaluation_identity=identity,
                )


if __name__ == "__main__":
    unittest.main()
