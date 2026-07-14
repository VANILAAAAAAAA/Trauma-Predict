from __future__ import annotations

import unittest

import torch

from trauma_predict.modeling.multires_event_v2.relation_bias import TypedRelationBias
from trauma_predict.modeling.multires_event_v2.trajectory import (
    FieldStateTrajectoryDecoder,
    build_target_access_mask,
)


def test_target_access_masks_match_the_three_ablation_contracts() -> None:
    block = build_target_access_mask("block", 6, 29)
    trajectory = build_target_access_mask("trajectory", 6, 29)
    relational = build_target_access_mask("relational", 6, 29)

    block_0_field_1 = 1
    block_1_field_0 = 29
    assert block[block_0_field_1, 0]
    assert not block[block_0_field_1, block_0_field_1]
    assert not block[block_1_field_0, 28]
    assert trajectory[block_1_field_0, 28]
    assert torch.equal(trajectory, relational)
    assert not trajectory[0].any()
    assert not trajectory.triu().any()


def test_typed_relations_are_additive_and_nonedges_are_zero_not_masks() -> None:
    module = TypedRelationBias(relation_type_count=14, num_attention_heads=2)
    with torch.no_grad():
        module.type_head_bias.zero_()
        module.type_head_bias[3] = torch.tensor([1.25, -0.75])
    adjacency = torch.zeros(14, 29, 29)
    adjacency[3, 2, 1] = 1.0

    bias = module(
        adjacency,
        query_field_indices=torch.tensor([2, 4]),
        key_field_indices=torch.tensor([1, 5]),
    )
    assert bias.shape == (1, 2, 2, 2)
    torch.testing.assert_close(bias[0, :, 0, 0], torch.tensor([1.25, -0.75]))
    torch.testing.assert_close(bias[0, :, 1, 1], torch.zeros(2))
    assert torch.isfinite(bias).all()


def test_relation_bias_is_an_exact_zero_residual_at_initialization() -> None:
    module = TypedRelationBias(relation_type_count=14, num_attention_heads=8)
    torch.testing.assert_close(
        module.type_head_bias,
        torch.zeros_like(module.type_head_bias),
    )


def test_relational_mode_changes_only_the_static_attention_bias() -> None:
    torch.manual_seed(7)
    decoder = FieldStateTrajectoryDecoder(
        hidden_size=8,
        num_heads=2,
        layers=1,
        dropout=0.0,
        block_count=6,
        field_count=29,
        relation_type_count=14,
    ).eval()
    with torch.no_grad():
        decoder.relation_bias.type_head_bias.zero_()
        decoder.relation_bias.type_head_bias[2] = torch.tensor([4.0, -3.0])
    queries = torch.randn(1, 6, 29, 8)
    truth_context = torch.randn_like(queries)
    truth_mask = torch.ones(1, 6, 29, dtype=torch.bool)
    memory = torch.randn(1, 3, 8)
    memory_mask = torch.ones(1, 3, dtype=torch.bool)
    zero_graph = torch.zeros(14, 29, 29)
    graph = zero_graph.clone()
    graph[2, 1, 0] = 1.0
    relation_type_lags = torch.zeros(14, dtype=torch.long)

    trajectory = decoder(
        queries,
        memory,
        memory_mask,
        mode="trajectory",
        context_states=truth_context,
        context_mask=truth_mask,
    )
    relational_zero = decoder(
        queries,
        memory,
        memory_mask,
        mode="relational",
        context_states=truth_context,
        context_mask=truth_mask,
        relation_adjacency=zero_graph,
        relation_type_lags=relation_type_lags,
    )
    relational_graph = decoder(
        queries,
        memory,
        memory_mask,
        mode="relational",
        context_states=truth_context,
        context_mask=truth_mask,
        relation_adjacency=graph,
        relation_type_lags=relation_type_lags,
    )
    torch.testing.assert_close(trajectory, relational_zero)
    assert not torch.allclose(trajectory[:, 1], relational_graph[:, 1])


def test_relation_parameters_have_zero_not_missing_gradients_in_neutral_modes() -> None:
    torch.manual_seed(11)
    decoder = FieldStateTrajectoryDecoder(
        hidden_size=8,
        num_heads=2,
        layers=1,
        dropout=0.0,
        block_count=6,
        field_count=29,
        relation_type_count=14,
    )
    queries = torch.randn(1, 6, 29, 8)
    context = torch.randn_like(queries)
    memory = torch.randn(1, 2, 8)
    memory_mask = torch.ones(1, 2, dtype=torch.bool)
    context_mask = torch.ones(1, 6, 29, dtype=torch.bool)

    for mode in ("block", "trajectory"):
        decoder.zero_grad(set_to_none=True)
        output = decoder(
            queries,
            memory,
            memory_mask,
            mode=mode,
            context_states=context,
            context_mask=context_mask,
        )
        output.square().mean().backward()
        gradient = decoder.relation_bias.type_head_bias.grad
        assert gradient is not None
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))


def test_relation_bias_respects_registered_block_lag() -> None:
    module = TypedRelationBias(relation_type_count=2, num_attention_heads=1)
    with torch.no_grad():
        module.type_head_bias[:, 0] = torch.tensor([1.0, 10.0])
    adjacency = torch.zeros(2, 2, 2)
    adjacency[0, 1, 0] = 1.0  # cross-field lag 0
    adjacency[1, 1, 1] = 1.0  # self-transition lag 1
    bias = module(
        adjacency,
        query_field_indices=torch.tensor([1]),
        key_field_indices=torch.tensor([0, 0, 1, 1]),
        relation_type_lags=torch.tensor([0, 1]),
        query_block_indices=torch.tensor([1]),
        key_block_indices=torch.tensor([1, 0, 1, 0]),
    )
    torch.testing.assert_close(
        bias[0, 0, 0],
        torch.tensor([1.0, 0.0, 0.0, 10.0]),
    )


class MultiresEventV2RelationTest(unittest.TestCase):
    test_access_masks = staticmethod(test_target_access_masks_match_the_three_ablation_contracts)
    test_additive_nonedge = staticmethod(
        test_typed_relations_are_additive_and_nonedges_are_zero_not_masks
    )
    test_zero_residual_initialization = staticmethod(
        test_relation_bias_is_an_exact_zero_residual_at_initialization
    )
    test_relation_delta = staticmethod(test_relational_mode_changes_only_the_static_attention_bias)
    test_neutral_gradient = staticmethod(
        test_relation_parameters_have_zero_not_missing_gradients_in_neutral_modes
    )
    test_registered_lag = staticmethod(test_relation_bias_respects_registered_block_lag)
