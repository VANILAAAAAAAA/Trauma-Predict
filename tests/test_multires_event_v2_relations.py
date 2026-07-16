from __future__ import annotations

import inspect
import unittest

import torch

import trauma_predict.modeling.multires_event_v2.relation_bias as relation_bias_module
from trauma_predict.modeling.multires_event_v2.relation_bias import (
    RegisteredRelationBias,
)
from trauma_predict.modeling.multires_event_v2.trajectory import (
    build_joint_target_access_mask,
)


class MultiresEventV2RelationTest(unittest.TestCase):
    def test_joint_access_is_strictly_causal_across_fields_and_blocks(self) -> None:
        access = build_joint_target_access_mask(2, 3)
        self.assertTrue(access[1, 0])
        self.assertTrue(access[3, 2])
        self.assertFalse(access[0].any())
        self.assertFalse(access.triu().any())

    def test_parameter_key_bias_is_additive_and_nonedges_remain_zero(self) -> None:
        module = RegisteredRelationBias(("edge.a", "edge.b"), num_attention_heads=2)
        with torch.no_grad():
            module.edge_head_bias.copy_(
                torch.tensor([[1.25, -0.75], [3.0, 2.0]])
            )
        adjacency = torch.zeros(2, 2, 3)
        adjacency[0, 0, 1] = 1.0
        adjacency[1, 1, 2] = 1.0

        bias = module(
            adjacency,
            query_field_indices=torch.tensor([0, 1]),
            key_field_indices=torch.tensor([0, 1, 2]),
        )
        self.assertEqual(bias.shape, (1, 2, 2, 3))
        torch.testing.assert_close(bias[0, :, 0, 1], torch.tensor([1.25, -0.75]))
        torch.testing.assert_close(bias[0, :, 1, 2], torch.tensor([3.0, 2.0]))
        torch.testing.assert_close(bias[0, :, 0, 0], torch.zeros(2))
        self.assertTrue(torch.isfinite(bias).all())

    def test_each_parameter_key_has_an_independent_zero_initialized_row(self) -> None:
        module = RegisteredRelationBias(
            ("tt.self.hr", "tt.edge.hr_sbp", "tt.edge.map_ned"),
            num_attention_heads=4,
        )
        self.assertEqual(module.edge_head_bias.shape, (3, 4))
        torch.testing.assert_close(
            module.edge_head_bias,
            torch.zeros_like(module.edge_head_bias),
        )
        self.assertEqual(module.relation_count, 3)

    def test_explicit_scope_mask_controls_each_registered_edge(self) -> None:
        module = RegisteredRelationBias(("same", "adjacent"), num_attention_heads=1)
        with torch.no_grad():
            module.edge_head_bias[:, 0] = torch.tensor([1.0, 10.0])
        adjacency = torch.zeros(2, 2, 2)
        adjacency[0, 1, 0] = 1.0
        adjacency[1, 1, 1] = 1.0
        scope = torch.tensor(
            [
                [[True, False, False, False]],
                [[False, False, False, True]],
            ]
        )
        bias = module(
            adjacency,
            query_field_indices=torch.tensor([1]),
            key_field_indices=torch.tensor([0, 0, 1, 1]),
            relation_scope_mask=scope,
        )
        torch.testing.assert_close(
            bias[0, 0, 0],
            torch.tensor([1.0, 0.0, 0.0, 10.0]),
        )

    def test_checkpoint_parameter_key_order_is_part_of_state_identity(self) -> None:
        source = RegisteredRelationBias(("edge.a", "edge.b"), 2)
        reordered = RegisteredRelationBias(("edge.b", "edge.a"), 2)
        with self.assertRaisesRegex(RuntimeError, "parameter_key order"):
            reordered.load_state_dict(source.state_dict())

    def test_legacy_relation_type_api_is_absent(self) -> None:
        parameters = inspect.signature(RegisteredRelationBias).parameters
        self.assertNotIn("relation_type_count", parameters)
        forward_parameters = inspect.signature(RegisteredRelationBias.forward).parameters
        self.assertNotIn("relation_type_lags", forward_parameters)
        self.assertFalse(hasattr(relation_bias_module, "TypedRelationBias"))


if __name__ == "__main__":
    unittest.main()
