from __future__ import annotations

import unittest

import torch

from trauma_predict.modeling.multires_event.heads import TypedPredictionHeads
from trauma_predict.training.multires_event_loss import compute_multires_loss


def _duration_contract() -> dict:
    queries = []
    for resolution, time_index in (("H1", 0), *(("M4", value) for value in range(1, 7))):
        coverage_position = len(queries)
        common = {
            "field_id": 1,
            "operator_id": 6,
            "field": "heart_rate",
            "operator": "DURATION",
            "resolution": resolution,
            "resolution_id": 1 if resolution == "H1" else 2,
            "time_index": time_index,
            "span_hours": 1 if resolution == "H1" else 4,
            "value_type": "duration",
            "loss_family": "duration",
            "duration_kind": "point_binomial",
        }
        queries.append({
            **common,
            "condition_id": 1,
            "condition": "OBSERVED",
            "semantic_component": "observation_coverage",
            "coverage_query_position": -1,
        })
        queries.append({
            **common,
            "condition_id": 2,
            "condition": "HR_GT120",
            "semantic_component": "abnormal_duration",
            "coverage_query_position": coverage_position,
        })
    return {"queries": queries}


class MultiresEventLossTest(unittest.TestCase):
    def test_zero_coverage_excludes_conditional_abnormal_duration(self) -> None:
        contract = _duration_contract()
        query_count = len(contract["queries"])
        hidden = torch.randn(2, query_count, 16, requires_grad=True)
        outputs = TypedPredictionHeads(16, 0.0)(hidden)
        outputs["query_mask"] = torch.ones((2, query_count), dtype=torch.bool)
        batch = {
            "target_values": torch.zeros((2, query_count)),
            "target_raw_values": torch.zeros((2, query_count)),
            "target_mask": torch.ones((2, query_count), dtype=torch.bool),
        }
        result = compute_multires_loss(outputs, batch, contract)
        abnormal = torch.tensor(list(range(1, query_count, 2)))
        coverage = torch.tensor(list(range(0, query_count, 2)))
        self.assertFalse(result["per_query_valid"].index_select(1, abnormal).any())
        self.assertTrue(result["per_query_valid"].index_select(1, coverage).all())
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_half_precision_head_banks_are_promoted_before_index_writes(self) -> None:
        contract = _duration_contract()
        query_count = len(contract["queries"])
        outputs = TypedPredictionHeads(16, 0.0)(
            torch.randn(1, query_count, 16)
        )
        outputs = {
            key: value.half() if torch.is_floating_point(value) else value
            for key, value in outputs.items()
        }
        outputs["query_mask"] = torch.ones((1, query_count), dtype=torch.bool)
        batch = {
            "target_values": torch.zeros((1, query_count), dtype=torch.float32),
            "target_raw_values": torch.zeros((1, query_count), dtype=torch.float32),
            "target_mask": torch.ones((1, query_count), dtype=torch.bool),
        }
        result = compute_multires_loss(outputs, batch, contract)
        self.assertEqual(result["predictions"].dtype, torch.float32)
        self.assertEqual(result["per_query_loss"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(result["loss"]))


if __name__ == "__main__":
    unittest.main()
