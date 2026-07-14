from __future__ import annotations

import inspect
import unittest

import torch

from trauma_predict.modeling.multires_event_v2.field_state import (
    PrimitiveFeedbackEncoder,
    PrimitiveParameterHeads,
)
from trauma_predict.modeling.multires_event_v2.rollout import (
    AutoregressiveFieldStateRollout,
)
from trauma_predict.modeling.multires_event_v2.trajectory import (
    FieldStateTrajectoryDecoder,
)


def _decoder(blocks: int = 6, fields: int = 29) -> FieldStateTrajectoryDecoder:
    torch.manual_seed(13)
    return FieldStateTrajectoryDecoder(
        hidden_size=8,
        num_heads=2,
        layers=1,
        dropout=0.0,
        block_count=blocks,
        field_count=fields,
        relation_type_count=14,
    ).eval()


def test_teacher_forcing_cannot_read_current_or_future_truth() -> None:
    decoder = _decoder()
    queries = torch.randn(1, 6, 29, 8)
    context = torch.randn_like(queries)
    changed = context.clone()
    target_position = 31  # block 1, registered field order 2
    changed.reshape(1, 174, 8)[:, target_position:] += 100.0
    memory = torch.randn(1, 2, 8)
    memory_mask = torch.ones(1, 2, dtype=torch.bool)
    available = torch.ones(1, 6, 29, dtype=torch.bool)

    original = decoder(
        queries,
        memory,
        memory_mask,
        mode="trajectory",
        context_states=context,
        context_mask=available,
        query_positions=torch.tensor([target_position]),
    )
    future_changed = decoder(
        queries,
        memory,
        memory_mask,
        mode="trajectory",
        context_states=changed,
        context_mask=available,
        query_positions=torch.tensor([target_position]),
    )
    torch.testing.assert_close(original, future_changed)


def test_block_ignores_previous_blocks_but_both_modes_use_earlier_same_block_field() -> None:
    decoder = _decoder()
    queries = torch.randn(1, 6, 29, 8)
    context = torch.randn_like(queries)
    memory = torch.randn(1, 2, 8)
    memory_mask = torch.ones(1, 2, dtype=torch.bool)
    available = torch.ones(1, 6, 29, dtype=torch.bool)

    prior_block_changed = context.clone()
    prior_block_changed[:, 0, :, 0] += 50.0
    first_field_of_block_1 = torch.tensor([29])
    block_original = decoder(
        queries,
        memory,
        memory_mask,
        mode="block",
        context_states=context,
        context_mask=available,
        query_positions=first_field_of_block_1,
    )
    block_changed = decoder(
        queries,
        memory,
        memory_mask,
        mode="block",
        context_states=prior_block_changed,
        context_mask=available,
        query_positions=first_field_of_block_1,
    )
    trajectory_original = decoder(
        queries,
        memory,
        memory_mask,
        mode="trajectory",
        context_states=context,
        context_mask=available,
        query_positions=first_field_of_block_1,
    )
    trajectory_changed = decoder(
        queries,
        memory,
        memory_mask,
        mode="trajectory",
        context_states=prior_block_changed,
        context_mask=available,
        query_positions=first_field_of_block_1,
    )
    torch.testing.assert_close(block_original, block_changed)
    assert not torch.allclose(trajectory_original, trajectory_changed)

    earlier_same_block_changed = context.clone()
    earlier_same_block_changed[:, 1, 0, 0] += 50.0
    second_field_of_block_1 = torch.tensor([30])
    before = decoder(
        queries,
        memory,
        memory_mask,
        mode="block",
        context_states=context,
        context_mask=available,
        query_positions=second_field_of_block_1,
    )
    after = decoder(
        queries,
        memory,
        memory_mask,
        mode="block",
        context_states=earlier_same_block_changed,
        context_mask=available,
        query_positions=second_field_of_block_1,
    )
    assert not torch.allclose(before, after)


def test_rollout_samples_primitives_and_feedbacks_them_without_truth_argument() -> None:
    decoder = _decoder(blocks=2, fields=3)
    heads = PrimitiveParameterHeads(8, {"categorical_hours_0_4": 5}, dropout=0.0)
    feedback = PrimitiveFeedbackEncoder(8, {"categorical_hours_0_4": 1}, dropout=0.0)
    rollout = AutoregressiveFieldStateRollout(block_count=2, field_count=3).eval()
    queries = torch.randn(1, 2, 3, 8)
    memory = torch.randn(1, 2, 8)
    memory_mask = torch.ones(1, 2, dtype=torch.bool)

    assert "target_primitives" not in inspect.signature(rollout.forward).parameters

    def zeros_sampler(block: int, field: int, parameters: dict[str, torch.Tensor]):
        assert parameters["categorical_hours_0_4"].shape == (1, 5)
        return (
            {"categorical_hours_0_4": torch.zeros(1, 1)},
            {"categorical_hours_0_4": torch.ones(1, 1, dtype=torch.bool)},
        )

    def ones_sampler(block: int, field: int, parameters: dict[str, torch.Tensor]):
        return (
            {"categorical_hours_0_4": torch.ones(1, 1)},
            {"categorical_hours_0_4": torch.ones(1, 1, dtype=torch.bool)},
        )

    zeros, generated_zero, _ = rollout(
        queries,
        memory,
        memory_mask,
        decoder=decoder,
        primitive_heads=heads,
        feedback_encoder=feedback,
        mode="block",
        sampler=zeros_sampler,
    )
    ones, generated_one, _ = rollout(
        queries,
        memory,
        memory_mask,
        decoder=decoder,
        primitive_heads=heads,
        feedback_encoder=feedback,
        mode="block",
        sampler=ones_sampler,
    )
    torch.testing.assert_close(zeros[:, 0, 0], ones[:, 0, 0])
    assert not torch.allclose(zeros[:, 0, 1], ones[:, 0, 1])
    torch.testing.assert_close(zeros[:, 1, 0], ones[:, 1, 0])
    assert generated_zero["categorical_hours_0_4"].shape == (1, 2, 3, 1)
    assert generated_one["categorical_hours_0_4"].eq(1).all()


def test_cached_rollout_matches_reference_states_and_logits_in_all_modes() -> None:
    torch.manual_seed(41)
    decoder = FieldStateTrajectoryDecoder(
        hidden_size=8,
        num_heads=2,
        layers=2,
        dropout=0.0,
        block_count=2,
        field_count=3,
        relation_type_count=2,
    ).eval()
    with torch.no_grad():
        decoder.relation_bias.type_head_bias.copy_(
            torch.tensor([[0.1, -0.2], [0.3, 0.4]])
        )
    heads = PrimitiveParameterHeads(8, {"categorical_hours_0_4": 5}, dropout=0.0).eval()
    feedback = PrimitiveFeedbackEncoder(
        8,
        {"categorical_hours_0_4": 1},
        dropout=0.0,
    ).eval()
    rollout = AutoregressiveFieldStateRollout(block_count=2, field_count=3).eval()
    queries = torch.randn(2, 2, 3, 8)
    memory = torch.randn(2, 4, 8)
    memory_mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]]
    )
    relation_adjacency = torch.zeros(2, 3, 3)
    relation_adjacency[0].fill_diagonal_(1.0)
    relation_adjacency[1, :, 0] = 1.0
    relation_type_lags = torch.tensor([0, 1])

    def run(mode: str, *, use_cache: bool) -> tuple[torch.Tensor, torch.Tensor]:
        position_logits: list[torch.Tensor] = []

        def sampler(block: int, field: int, parameters: dict[str, torch.Tensor]):
            position_logits.append(parameters["categorical_hours_0_4"].detach().clone())
            sampled = torch.full((2, 1), float(block * 3 + field) / 5.0)
            return (
                {"categorical_hours_0_4": sampled},
                {
                    "categorical_hours_0_4": torch.ones(
                        2,
                        1,
                        dtype=torch.bool,
                    )
                },
            )

        relation_arguments = (
            {
                "relation_adjacency": relation_adjacency,
                "relation_type_lags": relation_type_lags,
            }
            if mode == "relational"
            else {}
        )
        field_states, _, _ = rollout(
            queries,
            memory,
            memory_mask,
            decoder=decoder,
            primitive_heads=heads,
            feedback_encoder=feedback,
            mode=mode,  # type: ignore[arg-type]
            sampler=sampler,
            use_cache=use_cache,
            **relation_arguments,
        )
        return field_states, torch.stack(position_logits, dim=1)

    assert "target_primitives" not in inspect.signature(
        decoder.initialize_incremental_cache
    ).parameters
    for mode in ("block", "trajectory", "relational"):
        reference_states, reference_logits = run(mode, use_cache=False)
        cached_states, cached_logits = run(mode, use_cache=True)
        torch.testing.assert_close(cached_states, reference_states, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(cached_logits, reference_logits, rtol=1e-5, atol=1e-6)


def test_incremental_cache_rejects_active_training_mode() -> None:
    decoder = FieldStateTrajectoryDecoder(
        hidden_size=8,
        num_heads=2,
        layers=1,
        dropout=0.1,
        block_count=2,
        field_count=3,
        relation_type_count=2,
    ).train()
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "inference-only"):
        decoder.initialize_incremental_cache(
            torch.randn(1, 2, 3, 8),
            torch.randn(1, 2, 8),
            torch.ones(1, 2, dtype=torch.bool),
            mode="trajectory",
        )


class MultiresEventV2TrajectoryTest(unittest.TestCase):
    test_teacher_causality = staticmethod(test_teacher_forcing_cannot_read_current_or_future_truth)
    test_block_trajectory_boundary = staticmethod(
        test_block_ignores_previous_blocks_but_both_modes_use_earlier_same_block_field
    )
    test_generated_feedback = staticmethod(
        test_rollout_samples_primitives_and_feedbacks_them_without_truth_argument
    )
    test_cached_reference_equivalence = staticmethod(
        test_cached_rollout_matches_reference_states_and_logits_in_all_modes
    )
    test_cache_is_inference_only = staticmethod(
        test_incremental_cache_rejects_active_training_mode
    )
