from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from trauma_predict.training.multires_event import (
    MODEL_INPUT_KEYS,
    _loss_aggregates,
    _prediction_row,
    assert_resume_identity,
    evaluate_model,
    validate_multires_event_config,
)
from trauma_predict.training.observability import LossAccumulator


REPO_ROOT = Path(__file__).resolve().parents[1]


class MultiresEventTrainingContractTest(unittest.TestCase):
    def load_configs(self) -> tuple[dict, dict]:
        train = yaml.safe_load(
            (REPO_ROOT / "configs/train/t4x2_multires_event_v1_full.yaml").read_text()
        )
        model = yaml.safe_load(
            (REPO_ROOT / "configs/model/multires_event_v1.yaml").read_text()
        )
        return train, model

    def test_frozen_route_has_986_direct_queries_and_no_text_backbone(self) -> None:
        train, model = self.load_configs()
        validate_multires_event_config(train, model)
        self.assertEqual(train["target"]["primary_direct_queries"], 986)
        self.assertEqual(train["target"]["h1_queries"], 92)
        self.assertEqual(train["target"]["m4_queries_per_block"], 149)
        self.assertFalse(train["target"]["f24_training_loss"])
        self.assertEqual(train["evaluation"]["interval_expected_subjects"], 505)
        self.assertEqual(train["evaluation"]["final_expected_samples"], 6309)
        self.assertIsNone(model["text_backbone"])
        self.assertIsNone(model["tokenizer"])
        self.assertEqual(model["block_pooling"]["latent_tokens_per_block"], 8)
        self.assertEqual(model["decoder"]["query_layers"], 3)

    def test_resume_identity_is_exact(self) -> None:
        identity = {
            "source": "a",
            "dataset": "b",
            "resolved_config": "c",
            "target_contract": "d",
            "normalization": "e",
            "model_config": "f",
        }
        assert_resume_identity(identity, dict(identity))
        changed = dict(identity, normalization="changed")
        with self.assertRaisesRegex(RuntimeError, "normalization"):
            assert_resume_identity(identity, changed)

    def test_accumulator_uses_additive_numerator_denominator(self) -> None:
        accumulator = LossAccumulator()
        accumulator.update_aggregates(2.0, 1.0, {"family/x": (4.0, 2.0)})
        accumulator.update_aggregates(9.0, 3.0, {"family/x": (3.0, 1.0)})
        self.assertAlmostEqual(accumulator.summary()["total"], 11.0 / 4.0)
        self.assertAlmostEqual(accumulator.summary()["family/x"], 7.0 / 3.0)

    def test_loss_result_requires_true_aggregates(self) -> None:
        with self.assertRaisesRegex(ValueError, "aggregation fields"):
            _loss_aggregates({"loss": 1.0})
        numerator, denominator, parts = _loss_aggregates({
            "loss": 1.0,
            "loss_numerator": 3.0,
            "loss_denominator": 2.0,
            "parts": {"resolution/H1": {"numerator": 2.0, "denominator": 1.0}},
        })
        self.assertEqual((numerator, denominator), (3.0, 2.0))
        self.assertEqual(parts["resolution/H1"], (2.0, 1.0))

    def test_prediction_export_uses_raw_typed_summary_and_f24(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        query_count = 986
        summary = {
            "conditional_raw_value": torch.ones((1, query_count)),
            "expected_raw_value": torch.full((1, query_count), 0.5),
            "presence_probability": torch.full((1, query_count), 0.5),
            "binary_probability": torch.full((1, query_count), float("nan")),
            "ordinal_probability": torch.zeros((1, query_count, 6)),
            "ordinal_class_mask": torch.zeros((query_count, 6), dtype=torch.bool),
        }
        result = {
            "predictions": summary["expected_raw_value"],
            "prediction_mask": torch.ones((1, query_count), dtype=torch.bool),
            "prediction_summary": summary,
            "derived_f24_prediction_summary": {
                "conditional_raw_value": torch.ones((1, 149)),
                "expected_raw_value": torch.zeros((1, 149)),
                "presence_probability": torch.full((1, 149), 0.5),
                "binary_probability": torch.full((1, 149), float("nan")),
                "ordinal_probability": torch.full((1, 149, 6), 1.0 / 6.0),
                "ordinal_class_mask": torch.ones((149, 6), dtype=torch.bool),
            },
        }
        row = _prediction_row(
            result,
            {
                "prediction_hour": torch.tensor([18]),
                "target_raw_values": torch.arange(query_count).unsqueeze(0),
                "target_mask": torch.ones((1, query_count), dtype=torch.bool),
                "f24_target_raw_values": torch.arange(149).unsqueeze(0),
                "f24_target_mask": torch.ones((1, 149), dtype=torch.bool),
            },
            "sample-1",
            "subject-1",
        )
        self.assertEqual(len(row["active_query_predictions"]), 986)
        self.assertEqual(
            set(row["active_query_predictions"][0]),
            {"conditional_raw_value", "expected_raw_value", "presence_probability"},
        )
        self.assertEqual(len(row["derived_f24_predictions"]["expected_raw_value"]), 149)
        self.assertEqual(len(row["derived_f24_predictions"]["ordinal_probability"]), 149)
        self.assertEqual(
            len(row["derived_f24_predictions"]["ordinal_probability"][0]), 6
        )
        self.assertEqual(len(row["derived_f24_predictions"]["ordinal_class_mask"]), 149)
        self.assertEqual(len(row["target_raw_values"]), 986)
        self.assertEqual(len(row["target_mask"]), 986)
        self.assertEqual(len(row["f24_target_raw_values"]), 149)
        self.assertEqual(len(row["f24_target_mask"]), 149)
        self.assertNotIn("prediction_mask", row)
        self.assertEqual(len(row["query_mask"]), 986)

    def test_eval_reports_true_global_and_subject_macro_losses(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class Model:
            def eval(self):
                return self

            def __call__(self, **kwargs):
                return {"token": kwargs["event_values"].sum()}

        def batch(sample: str, subject: str, numerator: float, denominator: float) -> dict:
            payload = {key: torch.zeros((1, 1)) for key in MODEL_INPUT_KEYS}
            payload.update({
                "sample_id": [sample],
                "subject_id": [subject],
                "prediction_hour": torch.tensor([18]),
                "numerator": torch.tensor(numerator),
                "denominator": torch.tensor(denominator),
            })
            return payload

        def compute_loss(outputs, payload, contract, *, normalizer):
            numerator = payload["numerator"]
            denominator = payload["denominator"]
            return {
                "loss": numerator / denominator,
                "loss_numerator": numerator,
                "loss_denominator": denominator,
                "parts": {
                    "resolution/H1": {
                        "numerator": numerator,
                        "denominator": denominator,
                    }
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = evaluate_model(
                model=Model(),
                loader=[batch("s1", "p1", 2.0, 1.0), batch("s2", "p2", 6.0, 2.0)],
                compute_loss=compute_loss,
                target_contract={},
                normalizer=None,
                device=torch.device("cpu"),
                metrics_path=root / "metrics.jsonl",
                step=250,
                expected_samples=2,
                prediction_path=None,
                output_dir=root,
                phase="interval",
            )
            self.assertAlmostEqual(result["eval_primary_loss"], 8.0 / 3.0)
            self.assertAlmostEqual(result["eval_primary_loss_subject_macro"], 2.5)
            events = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(events[0]["event"], "eval_loss")


if __name__ == "__main__":
    unittest.main()
