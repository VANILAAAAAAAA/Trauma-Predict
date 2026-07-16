from __future__ import annotations

import unittest

import torch

from trauma_predict.data.multires_event_v2.relation_contract import (
    MultiresEventV2RelationContract,
)
from trauma_predict.modeling.multires_event_v2.input_field_memory import (
    INPUT_ONLY_FIELD_IDS,
    InputFieldMemoryEncoder,
)
from trauma_predict.modeling.multires_event_v2.relation_bias import (
    RegisteredRelationBias,
)
from trauma_predict.modeling.multires_event_v2.trajectory import (
    FieldStateTrajectoryDecoder,
    build_joint_target_access_mask,
)


def _small_decoder() -> FieldStateTrajectoryDecoder:
    decoder = FieldStateTrajectoryDecoder(
        hidden_size=8,
        num_heads=2,
        layers=2,
        dropout=0.0,
        block_count=2,
        field_count=2,
        input_field_count=4,
        target_parameter_keys=("tt.self.field_1", "tt.cross.1_to_2"),
        input_parameter_keys=("it.bridge.field_1", "it.care.field_3_to_2"),
        target_input_field_ids=(1, 2),
    )
    with torch.no_grad():
        decoder.target_relation_bias.edge_head_bias.copy_(
            torch.tensor([[0.75, -0.25], [1.25, 0.5]])
        )
        decoder.input_target_relation_bias.edge_head_bias.copy_(
            torch.tensor([[0.4, -0.1], [0.9, 0.3]])
        )
    return decoder


def _small_relations() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.zeros(2, 2, 2, dtype=torch.bool)
    target[0, 0, 0] = True  # field 1 adjacent-block self transition
    target[1, 1, 0] = True  # field 1 -> field 2 in the same block
    target_scopes = torch.tensor([1, 0])
    input_target = torch.zeros(2, 2, 4, dtype=torch.bool)
    input_target[0, 0, 0] = True  # field 1 bridge -> first future block
    input_target[1, 1, 2] = True  # input-only field 3 -> field 2, every block
    input_scopes = torch.tensor([0, 1])
    return target, target_scopes, input_target, input_scopes


def _event_geometry(block_index: torch.Tensor) -> tuple[torch.Tensor, ...]:
    relative_end = block_index.float() - 1.0
    relative_start = relative_end - 1.0
    return relative_start, relative_end, torch.ones_like(relative_end)


def test_output_field_bridge_uses_global_final_block_without_fallback() -> None:
    torch.manual_seed(5)
    encoder = InputFieldMemoryEncoder(
        hidden_size=6,
        input_field_count=4,
        target_field_ids=(1, 2),
        dropout=0.0,
        time_scale_hours=24.0,
    ).eval()
    event_embeddings = torch.randn(1, 5, 6)
    field_ids = torch.tensor([[1, 1, 2, 3, 4]])
    block_index = torch.tensor([[0, 1, 0, 0, 1]])
    latest_block = torch.tensor([1])
    event_mask = torch.ones(1, 5, dtype=torch.bool)
    event_start, event_end, event_span = _event_geometry(block_index)
    field_embeddings = torch.randn(4, 6)
    final_context = torch.randn(1, 6)

    tokens, observed = encoder(
        event_embeddings,
        field_ids,
        block_index,
        latest_block,
        event_mask,
        event_start,
        event_end,
        event_span,
        field_embeddings,
        final_context,
    )
    changed_old_history = event_embeddings.clone()
    changed_old_history[:, 0] += 100.0
    changed_old_history[:, 2] -= 100.0
    changed_tokens, changed_observed = encoder(
        changed_old_history,
        field_ids,
        block_index,
        latest_block,
        event_mask,
        event_start,
        event_end,
        event_span,
        field_embeddings,
        final_context,
    )

    # Field 1 has a final-block event, so its older event is contract-invisible.
    torch.testing.assert_close(tokens[:, 0], changed_tokens[:, 0])
    torch.testing.assert_close(tokens[:, 1], changed_tokens[:, 1])
    torch.testing.assert_close(observed, changed_observed)
    # Field 2 appears only in an older block: the bridge must be explicitly
    # unobserved rather than silently falling back to that old measurement.
    assert not observed[0, 1]
    assert torch.isfinite(tokens[0, 1]).all()

    empty_final_tokens, empty_final_observed = encoder(
        event_embeddings,
        field_ids,
        block_index,
        torch.tensor([2]),
        event_mask,
        event_start,
        event_end,
        event_span,
        field_embeddings,
        final_context,
    )
    assert not empty_final_observed[:, :2].any()
    assert torch.isfinite(empty_final_tokens[:, :2]).all()

    changed_context_tokens, _ = encoder(
        event_embeddings,
        field_ids,
        block_index,
        latest_block,
        event_mask,
        event_start,
        event_end,
        event_span,
        field_embeddings,
        final_context + torch.arange(6, dtype=final_context.dtype).unsqueeze(0),
    )
    assert not torch.allclose(tokens[:, 1], changed_context_tokens[:, 1])
    # Input-only field 3 pools all history and does not receive final-block geometry.
    torch.testing.assert_close(tokens[:, 2], changed_context_tokens[:, 2])


def test_missing_nonoutput_field_is_masked_but_all_output_bridges_remain_visible() -> None:
    encoder = InputFieldMemoryEncoder(
        hidden_size=4,
        input_field_count=4,
        target_field_ids=(1, 2),
        dropout=0.0,
        time_scale_hours=24.0,
    ).eval()
    block_index = torch.tensor([[0, 0]])
    event_start, event_end, event_span = _event_geometry(block_index)
    _, observed = encoder(
        torch.randn(1, 2, 4),
        torch.tensor([[1, 2]]),
        block_index,
        torch.tensor([1]),
        torch.ones(1, 2, dtype=torch.bool),
        event_start,
        event_end,
        event_span,
        torch.randn(4, 4),
        torch.randn(1, 4),
    )
    visible = observed | encoder.target_field_mask.unsqueeze(0)
    torch.testing.assert_close(
        visible,
        torch.tensor([[True, True, False, False]]),
    )

    decoder = _small_decoder().eval()
    _, _, input_target, input_scopes = _small_relations()
    memory_mask = torch.tensor([[True, True, True, True, False, False]])
    bias = decoder._memory_attention_bias(
        batch_size=1,
        memory_mask=memory_mask,
        positions=torch.arange(4),
        input_target_relation_adjacency=input_target,
        input_target_time_scope_ids=input_scopes,
        dtype=torch.float32,
    ).reshape(1, 2, 4, 6)
    # Missing input-only field 3 is not a valid key, so its registered care edge
    # remains -inf and cannot receive attention.
    assert torch.isneginf(bias[:, :, :, 4]).all()


def test_input_only_temporal_pooling_starts_at_exact_uniform_mean() -> None:
    torch.manual_seed(29)
    encoder = InputFieldMemoryEncoder(
        hidden_size=4,
        input_field_count=4,
        target_field_ids=(1, 2),
        dropout=0.0,
        time_scale_hours=24.0,
    ).eval()
    with torch.no_grad():
        encoder.observation_state.weight.zero_()
    embeddings = torch.tensor(
        [[[9.0, 1.0, -2.0, 0.5], [1.0, 7.0, 2.0, -0.5]]]
    )
    field_ids = torch.tensor([[3, 3]])
    block_index = torch.tensor([[0, 1]])
    starts = torch.tensor([[-48.0, -1.0]])
    ends = torch.tensor([[-24.0, 0.0]])
    spans = torch.tensor([[24.0, 1.0]])
    field_embeddings = torch.zeros(4, 4)
    latest_context = torch.zeros(1, 4)
    tokens, observed = encoder(
        embeddings,
        field_ids,
        block_index,
        torch.tensor([1]),
        torch.ones(1, 2, dtype=torch.bool),
        starts,
        ends,
        spans,
        field_embeddings,
        latest_context,
    )
    arithmetic_mean = embeddings.mean(dim=1)
    expected = encoder.output_norm(
        arithmetic_mean + encoder.fusion(arithmetic_mean)
    )
    torch.testing.assert_close(tokens[:, 2], expected)
    assert observed[0, 2]
    torch.testing.assert_close(
        encoder.input_only_temporal_weight,
        torch.zeros_like(encoder.input_only_temporal_weight),
    )


def test_input_only_temporal_pooling_distinguishes_near_and_far_without_new_rows() -> None:
    torch.manual_seed(31)
    encoder = InputFieldMemoryEncoder(
        hidden_size=4,
        input_field_count=4,
        target_field_ids=(1, 2),
        dropout=0.0,
        time_scale_hours=24.0,
    ).eval()
    embeddings = torch.tensor(
        [[[8.0, 0.0, 1.0, -1.0], [0.0, 8.0, -1.0, 1.0]]]
    )
    field_ids = torch.tensor([[3, 3]])
    block_index = torch.tensor([[0, 1]])
    starts = torch.tensor([[-48.0, -1.0]])
    ends = torch.tensor([[-24.0, 0.0]])
    spans = torch.tensor([[24.0, 1.0]])
    field_embeddings = torch.zeros(4, 4)
    latest_context = torch.zeros(1, 4)
    arguments = (
        embeddings,
        field_ids,
        block_index,
        torch.tensor([1]),
        torch.ones(1, 2, dtype=torch.bool),
        starts,
        ends,
        spans,
        field_embeddings,
        latest_context,
    )
    baseline, _ = encoder(*arguments)
    with torch.no_grad():
        encoder.input_only_temporal_weight[0, 0] = -12.0
    recency_weighted, _ = encoder(*arguments)
    assert not torch.allclose(baseline[:, 2], recency_weighted[:, 2])

    permutation = torch.tensor([1, 0])
    permuted, _ = encoder(
        embeddings[:, permutation],
        field_ids[:, permutation],
        block_index[:, permutation],
        torch.tensor([1]),
        torch.ones(1, 2, dtype=torch.bool),
        starts[:, permutation],
        ends[:, permutation],
        spans[:, permutation],
        field_embeddings,
        latest_context,
    )
    torch.testing.assert_close(recency_weighted, permuted)


def test_temporal_weights_do_not_change_latest_h1_output_bridge() -> None:
    torch.manual_seed(37)
    encoder = InputFieldMemoryEncoder(
        hidden_size=4,
        input_field_count=4,
        target_field_ids=(1, 2),
        dropout=0.0,
        time_scale_hours=24.0,
    ).eval()
    embeddings = torch.randn(1, 4, 4)
    field_ids = torch.tensor([[1, 1, 3, 3]])
    block_index = torch.tensor([[0, 1, 0, 1]])
    starts = torch.tensor([[-48.0, -1.0, -48.0, -1.0]])
    ends = torch.tensor([[-24.0, 0.0, -24.0, 0.0]])
    spans = torch.tensor([[24.0, 1.0, 24.0, 1.0]])
    arguments = (
        embeddings,
        field_ids,
        block_index,
        torch.tensor([1]),
        torch.ones(1, 4, dtype=torch.bool),
        starts,
        ends,
        spans,
        torch.randn(4, 4),
        torch.randn(1, 4),
    )
    original, _ = encoder(*arguments)
    with torch.no_grad():
        encoder.input_only_temporal_weight.normal_(mean=0.0, std=3.0)
    changed, _ = encoder(*arguments)
    torch.testing.assert_close(original[:, :2], changed[:, :2])
    assert not torch.allclose(original[:, 2], changed[:, 2])


def test_all_formal_input_only_temporal_parameters_receive_finite_gradient() -> None:
    torch.manual_seed(41)
    target_field_ids = tuple(
        field_id for field_id in range(1, 38) if field_id not in INPUT_ONLY_FIELD_IDS
    )
    encoder = InputFieldMemoryEncoder(
        hidden_size=8,
        input_field_count=37,
        target_field_ids=target_field_ids,
        dropout=0.0,
        time_scale_hours=24.0,
    )
    field_ids = torch.tensor(
        [[field_id for field_id in INPUT_ONLY_FIELD_IDS for _ in range(2)]]
    )
    event_count = field_ids.shape[1]
    block_index = torch.tensor([[value for _ in INPUT_ONLY_FIELD_IDS for value in (0, 1)]])
    starts = torch.tensor([[-72.0, -1.0] * len(INPUT_ONLY_FIELD_IDS)])
    ends = torch.tensor([[-48.0, 0.0] * len(INPUT_ONLY_FIELD_IDS)])
    spans = torch.tensor([[24.0, 1.0] * len(INPUT_ONLY_FIELD_IDS)])
    tokens, observed = encoder(
        torch.randn(1, event_count, 8),
        field_ids,
        block_index,
        torch.tensor([1]),
        torch.ones(1, event_count, dtype=torch.bool),
        starts,
        ends,
        spans,
        torch.randn(37, 8),
        torch.randn(1, 8),
    )
    input_only_indices = encoder.input_only_field_indices
    asymmetric = torch.randn_like(tokens.index_select(1, input_only_indices))
    (tokens.index_select(1, input_only_indices) * asymmetric).sum().backward()
    gradient = encoder.input_only_temporal_weight.grad
    assert observed.index_select(1, input_only_indices).all()
    assert gradient is not None and gradient.shape == (8, 6)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum(dim=1).gt(0).all()


def test_parameter_keys_are_independent_and_nonedges_remain_zero() -> None:
    bias_module = RegisteredRelationBias(
        ("edge.a", "edge.b"),
        num_attention_heads=2,
    )
    with torch.no_grad():
        bias_module.edge_head_bias.copy_(
            torch.tensor([[1.0, -1.0], [3.0, 2.0]])
        )
    adjacency = torch.zeros(2, 2, 3)
    adjacency[0, 0, 1] = 1
    adjacency[1, 1, 2] = 1
    result = bias_module(
        adjacency,
        query_field_indices=torch.tensor([0, 1]),
        key_field_indices=torch.tensor([0, 1, 2]),
    )
    torch.testing.assert_close(result[0, :, 0, 1], torch.tensor([1.0, -1.0]))
    torch.testing.assert_close(result[0, :, 1, 2], torch.tensor([3.0, 2.0]))
    torch.testing.assert_close(result[0, :, 0, 0], torch.zeros(2))


def test_visible_unregistered_input_pair_keeps_ordinary_attention() -> None:
    torch.manual_seed(13)
    decoder = _small_decoder().eval()
    target, target_scopes, input_target, input_scopes = _small_relations()
    memory = torch.randn(1, 6, 8)
    memory_mask = torch.ones(1, 6, dtype=torch.bool)
    memory_bias = decoder._memory_attention_bias(
        batch_size=1,
        memory_mask=memory_mask,
        positions=torch.tensor([0]),
        input_target_relation_adjacency=input_target,
        input_target_time_scope_ids=input_scopes,
        dtype=memory.dtype,
    ).reshape(1, 2, 1, 6)
    # Memory index 3 is visible input field 2.  No edge is registered from it
    # to target field 1, so the relation delta is finite zero, not -inf.
    torch.testing.assert_close(memory_bias[0, :, 0, 3], torch.zeros(2))

    queries = torch.randn(1, 2, 2, 8)
    original = decoder(
        queries,
        memory,
        memory_mask,
        query_positions=torch.tensor([0]),
        target_relation_adjacency=target,
        target_time_scope_ids=target_scopes,
        input_target_relation_adjacency=input_target,
        input_target_time_scope_ids=input_scopes,
    )
    changed_memory = memory.clone()
    changed_memory[:, 3] = torch.tensor(
        [[-2.0, 1.0, 0.5, 3.0, -1.5, 2.5, 4.0, -3.0]]
    )
    changed = decoder(
        queries,
        changed_memory,
        memory_mask,
        query_positions=torch.tensor([0]),
        target_relation_adjacency=target,
        target_time_scope_ids=target_scopes,
        input_target_relation_adjacency=input_target,
        input_target_time_scope_ids=input_scopes,
    )
    assert not torch.allclose(original, changed)


def test_every_registered_edge_receives_its_own_finite_nonzero_gradient() -> None:
    torch.manual_seed(23)
    contract = MultiresEventV2RelationContract.from_default_config()
    target_input_field_ids = tuple(
        contract.history_fields.index(field) + 1 for field in contract.target_fields
    )
    decoder = FieldStateTrajectoryDecoder(
        hidden_size=8,
        num_heads=2,
        layers=1,
        dropout=0.0,
        block_count=6,
        field_count=29,
        input_field_count=37,
        target_parameter_keys=contract.target_parameter_keys,
        input_parameter_keys=contract.input_target_parameter_keys,
        target_input_field_ids=target_input_field_ids,
    )
    queries = torch.randn(1, 6, 29, 8)
    context = torch.randn_like(queries)
    memory = torch.randn(1, 38, 8)
    output = decoder(
        queries,
        memory,
        torch.ones(1, 38, dtype=torch.bool),
        context_states=context,
        context_mask=torch.ones(1, 6, 29, dtype=torch.bool),
        target_relation_adjacency=torch.tensor(contract.target_relation_adjacency),
        target_time_scope_ids=torch.tensor(contract.target_time_scope_ids),
        input_target_relation_adjacency=torch.tensor(
            contract.input_target_relation_adjacency
        ),
        input_target_time_scope_ids=torch.tensor(contract.input_target_time_scope_ids),
    )
    asymmetric_weights = torch.randn_like(output)
    (output * asymmetric_weights).sum().backward()

    target_gradient = decoder.target_relation_bias.edge_head_bias.grad
    input_gradient = decoder.input_target_relation_bias.edge_head_bias.grad
    assert target_gradient is not None and target_gradient.shape == (52, 2)
    assert input_gradient is not None and input_gradient.shape == (39, 2)
    assert torch.isfinite(target_gradient).all()
    assert torch.isfinite(input_gradient).all()
    assert target_gradient.abs().sum(dim=1).gt(0).all()
    assert input_gradient.abs().sum(dim=1).gt(0).all()


def test_teacher_and_incremental_decoding_share_both_relation_paths() -> None:
    torch.manual_seed(19)
    decoder = _small_decoder().eval()
    target, target_scopes, input_target, input_scopes = _small_relations()
    queries = torch.randn(2, 2, 2, 8)
    feedback = torch.randn_like(queries)
    feedback_mask = torch.ones(2, 2, 2, dtype=torch.bool)
    memory = torch.randn(2, 6, 8)
    memory_mask = torch.tensor(
        [[True, True, True, True, True, False], [True, True, True, True, False, False]]
    )

    teacher = decoder(
        queries,
        memory,
        memory_mask,
        context_states=feedback,
        context_mask=feedback_mask,
        target_relation_adjacency=target,
        target_time_scope_ids=target_scopes,
        input_target_relation_adjacency=input_target,
        input_target_time_scope_ids=input_scopes,
    )
    cache = decoder.initialize_incremental_cache(
        queries,
        memory,
        memory_mask,
        target_relation_adjacency=target,
        target_time_scope_ids=target_scopes,
        input_target_relation_adjacency=input_target,
        input_target_time_scope_ids=input_scopes,
    )
    generated = []
    flat_feedback = feedback.reshape(2, 4, 8)
    for position in range(4):
        generated.append(decoder.incremental_step(cache))
        decoder.append_incremental_context(cache, flat_feedback[:, position])
    incremental = torch.stack(generated, dim=1)
    torch.testing.assert_close(incremental, teacher, rtol=1e-5, atol=1e-6)


def test_only_joint_causal_access_contract_is_available() -> None:
    access = build_joint_target_access_mask(2, 2)
    assert access[2, 0]
    assert access[3, 2]
    assert not access[0].any()
    assert not access.triu().any()


class MultiresEventV2RelationV2ModelingTest(unittest.TestCase):
    test_global_final_block = staticmethod(
        test_output_field_bridge_uses_global_final_block_without_fallback
    )
    test_field_visibility = staticmethod(
        test_missing_nonoutput_field_is_masked_but_all_output_bridges_remain_visible
    )
    test_temporal_uniform = staticmethod(
        test_input_only_temporal_pooling_starts_at_exact_uniform_mean
    )
    test_temporal_recency = staticmethod(
        test_input_only_temporal_pooling_distinguishes_near_and_far_without_new_rows
    )
    test_temporal_output_boundary = staticmethod(
        test_temporal_weights_do_not_change_latest_h1_output_bridge
    )
    test_temporal_gradients = staticmethod(
        test_all_formal_input_only_temporal_parameters_receive_finite_gradient
    )
    test_parameter_keys = staticmethod(
        test_parameter_keys_are_independent_and_nonedges_remain_zero
    )
    test_nonedge_attention = staticmethod(
        test_visible_unregistered_input_pair_keeps_ordinary_attention
    )
    test_all_edge_gradients = staticmethod(
        test_every_registered_edge_receives_its_own_finite_nonzero_gradient
    )
    test_teacher_incremental = staticmethod(
        test_teacher_and_incremental_decoding_share_both_relation_paths
    )
    test_joint_access = staticmethod(test_only_joint_causal_access_contract_is_available)
