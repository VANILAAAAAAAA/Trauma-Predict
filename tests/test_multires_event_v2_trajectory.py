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


def _decoder(
    *,
    blocks: int = 2,
    fields: int = 3,
    layers: int = 1,
    dropout: float = 0.0,
) -> FieldStateTrajectoryDecoder:
    torch.manual_seed(13)
    return FieldStateTrajectoryDecoder(
        hidden_size=8,
        num_heads=2,
        layers=layers,
        dropout=dropout,
        block_count=blocks,
        field_count=fields,
        input_field_count=4,
        target_parameter_keys=("tt.cross.1_to_2", "tt.self.field_1"),
        input_parameter_keys=("it.bridge.field_1", "it.care.field_4_to_2"),
        target_input_field_ids=tuple(range(1, fields + 1)),
    )


def _relations(fields: int = 3) -> tuple[torch.Tensor, ...]:
    target = torch.zeros(2, fields, fields, dtype=torch.bool)
    target[0, 1, 0] = True
    target[1, 0, 0] = True
    target_scopes = torch.tensor([0, 1])
    input_target = torch.zeros(2, fields, 4, dtype=torch.bool)
    input_target[0, 0, 0] = True
    input_target[1, 1, 3] = True
    input_scopes = torch.tensor([0, 1])
    return target, target_scopes, input_target, input_scopes


def _relation_arguments(fields: int = 3) -> dict[str, torch.Tensor]:
    target, target_scopes, input_target, input_scopes = _relations(fields)
    return {
        "target_relation_adjacency": target,
        "target_time_scope_ids": target_scopes,
        "input_target_relation_adjacency": input_target,
        "input_target_time_scope_ids": input_scopes,
    }


class MultiresEventV2TrajectoryTest(unittest.TestCase):
    def test_teacher_forcing_cannot_read_current_or_future_truth(self) -> None:
        decoder = _decoder().eval()
        queries = torch.randn(1, 2, 3, 8)
        context = torch.randn_like(queries)
        changed = context.clone()
        target_position = 4
        changed.reshape(1, 6, 8)[:, target_position:] += 100.0
        memory = torch.randn(1, 6, 8)
        memory_mask = torch.ones(1, 6, dtype=torch.bool)
        available = torch.ones(1, 2, 3, dtype=torch.bool)

        original = decoder(
            queries,
            memory,
            memory_mask,
            context_states=context,
            context_mask=available,
            query_positions=torch.tensor([target_position]),
            **_relation_arguments(),
        )
        future_changed = decoder(
            queries,
            memory,
            memory_mask,
            context_states=changed,
            context_mask=available,
            query_positions=torch.tensor([target_position]),
            **_relation_arguments(),
        )
        torch.testing.assert_close(original, future_changed)

    def test_joint_contract_uses_prior_blocks_and_earlier_same_block_fields(self) -> None:
        decoder = _decoder().eval()
        queries = torch.randn(1, 2, 3, 8)
        context = torch.randn_like(queries)
        memory = torch.randn(1, 6, 8)
        memory_mask = torch.ones(1, 6, dtype=torch.bool)
        available = torch.ones(1, 2, 3, dtype=torch.bool)

        prior_block_changed = context.clone()
        prior_block_changed[:, 0, :, 0] += 50.0
        first_field_of_block_1 = torch.tensor([3])
        before = decoder(
            queries,
            memory,
            memory_mask,
            context_states=context,
            context_mask=available,
            query_positions=first_field_of_block_1,
            **_relation_arguments(),
        )
        after = decoder(
            queries,
            memory,
            memory_mask,
            context_states=prior_block_changed,
            context_mask=available,
            query_positions=first_field_of_block_1,
            **_relation_arguments(),
        )
        self.assertFalse(torch.allclose(before, after))

        earlier_same_block_changed = context.clone()
        earlier_same_block_changed[:, 1, 0, 0] += 50.0
        second_field_of_block_1 = torch.tensor([4])
        before = decoder(
            queries,
            memory,
            memory_mask,
            context_states=context,
            context_mask=available,
            query_positions=second_field_of_block_1,
            **_relation_arguments(),
        )
        after = decoder(
            queries,
            memory,
            memory_mask,
            context_states=earlier_same_block_changed,
            context_mask=available,
            query_positions=second_field_of_block_1,
            **_relation_arguments(),
        )
        self.assertFalse(torch.allclose(before, after))

    def test_rollout_samples_primitives_and_feedbacks_them_without_truth(self) -> None:
        decoder = _decoder().eval()
        heads = PrimitiveParameterHeads(
            8, {"categorical_hours_0_4": 5}, dropout=0.0
        ).eval()
        feedback = PrimitiveFeedbackEncoder(
            8, {"categorical_hours_0_4": 1}, dropout=0.0
        ).eval()
        rollout = AutoregressiveFieldStateRollout(block_count=2, field_count=3).eval()
        queries = torch.randn(1, 2, 3, 8)
        memory = torch.randn(1, 6, 8)
        memory_mask = torch.ones(1, 6, dtype=torch.bool)

        self.assertNotIn("target_primitives", inspect.signature(rollout.forward).parameters)

        def sampler(value: float):
            def sample(_block: int, _field: int, parameters: dict[str, torch.Tensor]):
                self.assertEqual(parameters["categorical_hours_0_4"].shape, (1, 5))
                return (
                    {"categorical_hours_0_4": torch.full((1, 1), value)},
                    {
                        "categorical_hours_0_4": torch.ones(
                            1, 1, dtype=torch.bool
                        )
                    },
                )

            return sample

        zeros, generated_zero, _ = rollout(
            queries,
            memory,
            memory_mask,
            decoder=decoder,
            primitive_heads=heads,
            feedback_encoder=feedback,
            sampler=sampler(0.0),
            **_relation_arguments(),
        )
        ones, generated_one, _ = rollout(
            queries,
            memory,
            memory_mask,
            decoder=decoder,
            primitive_heads=heads,
            feedback_encoder=feedback,
            sampler=sampler(1.0),
            **_relation_arguments(),
        )
        torch.testing.assert_close(zeros[:, 0, 0], ones[:, 0, 0])
        self.assertFalse(torch.allclose(zeros[:, 0, 1], ones[:, 0, 1]))
        self.assertFalse(torch.allclose(zeros[:, 1, 0], ones[:, 1, 0]))
        self.assertEqual(
            generated_zero["categorical_hours_0_4"].shape,
            (1, 2, 3, 1),
        )
        self.assertTrue(generated_one["categorical_hours_0_4"].eq(1).all())

    def test_cached_rollout_matches_reference_states_and_logits(self) -> None:
        torch.manual_seed(41)
        decoder = _decoder(layers=2).eval()
        with torch.no_grad():
            decoder.target_relation_bias.edge_head_bias.copy_(
                torch.tensor([[0.1, -0.2], [0.3, 0.4]])
            )
            decoder.input_target_relation_bias.edge_head_bias.copy_(
                torch.tensor([[0.2, 0.1], [-0.1, 0.3]])
            )
        heads = PrimitiveParameterHeads(
            8, {"categorical_hours_0_4": 5}, dropout=0.0
        ).eval()
        feedback = PrimitiveFeedbackEncoder(
            8, {"categorical_hours_0_4": 1}, dropout=0.0
        ).eval()
        rollout = AutoregressiveFieldStateRollout(block_count=2, field_count=3).eval()
        queries = torch.randn(2, 2, 3, 8)
        memory = torch.randn(2, 6, 8)
        memory_mask = torch.ones(2, 6, dtype=torch.bool)

        def run(*, use_cache: bool) -> tuple[torch.Tensor, torch.Tensor]:
            position_logits: list[torch.Tensor] = []

            def sample(block: int, field: int, parameters: dict[str, torch.Tensor]):
                position_logits.append(
                    parameters["categorical_hours_0_4"].detach().clone()
                )
                sampled = torch.full((2, 1), float(block * 3 + field) / 5.0)
                return (
                    {"categorical_hours_0_4": sampled},
                    {
                        "categorical_hours_0_4": torch.ones(
                            2, 1, dtype=torch.bool
                        )
                    },
                )

            field_states, _, _ = rollout(
                queries,
                memory,
                memory_mask,
                decoder=decoder,
                primitive_heads=heads,
                feedback_encoder=feedback,
                sampler=sample,
                use_cache=use_cache,
                **_relation_arguments(),
            )
            return field_states, torch.stack(position_logits, dim=1)

        self.assertNotIn(
            "target_primitives",
            inspect.signature(decoder.initialize_incremental_cache).parameters,
        )
        reference_states, reference_logits = run(use_cache=False)
        cached_states, cached_logits = run(use_cache=True)
        torch.testing.assert_close(cached_states, reference_states, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(cached_logits, reference_logits, rtol=1e-5, atol=1e-6)

    def test_incremental_cache_rejects_active_training_mode(self) -> None:
        decoder = _decoder(dropout=0.1).train()
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            decoder.initialize_incremental_cache(
                torch.randn(1, 2, 3, 8),
                torch.randn(1, 6, 8),
                torch.ones(1, 6, dtype=torch.bool),
                **_relation_arguments(),
            )


if __name__ == "__main__":
    unittest.main()
