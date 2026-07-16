from __future__ import annotations

import inspect
import unittest

import torch

from trauma_predict.data.multires_event_v2 import MultiresEventV2RelationContract
from trauma_predict.modeling.multires_event_v2.config import MultiResolutionEventV2Config
from trauma_predict.modeling.multires_event_v2.field_state import PrimitiveFeedbackEncoder
from trauma_predict.modeling.multires_event_v2.model import MultiResolutionEventV2Model


PARAMETER_DIMS = {
    "categorical_hours_0_4": 5,
    "respiratory_block_evidence": 1,
}
FEEDBACK_DIMS = {
    "categorical_hours_0_4": 1,
    "respiratory_block_evidence": 1,
}


def _config() -> MultiResolutionEventV2Config:
    return MultiResolutionEventV2Config(
        hidden_size=8,
        num_attention_heads=2,
        trajectory_encoder_layers=1,
        target_decoder_layers=1,
        block_compressor_layers=1,
        block_latent_count=2,
        dropout=0.0,
        primitive_head_dims=PARAMETER_DIMS,
        primitive_feedback_dims=FEEDBACK_DIMS,
    )


def _model() -> MultiResolutionEventV2Model:
    return MultiResolutionEventV2Model(
        _config(),
        MultiresEventV2RelationContract.from_default_config(),
    )


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    event_count = 5
    input_blocks = 3
    return {
        "event_field_ids": torch.tensor([[1, 2, 3, 4, 5]]).expand(batch_size, -1),
        "event_operator_ids": torch.ones(batch_size, event_count, dtype=torch.long),
        "event_condition_ids": torch.ones(batch_size, event_count, dtype=torch.long),
        "event_values": torch.randn(batch_size, event_count),
        "event_value_mask": torch.ones(batch_size, event_count, dtype=torch.bool),
        "event_study_slot_ids": torch.zeros(batch_size, event_count, dtype=torch.long),
        "block_index": torch.tensor([[0, 0, 1, 2, 2]]).expand(batch_size, -1),
        "latest_input_block_index": torch.full((batch_size,), 2, dtype=torch.long),
        "event_mask": torch.ones(batch_size, event_count, dtype=torch.bool),
        "block_role_ids": torch.ones(batch_size, input_blocks, dtype=torch.long),
        "resolution_ids": torch.ones(batch_size, input_blocks, dtype=torch.long),
        "relative_start": torch.tensor([[-12.0, -8.0, -4.0]]).expand(batch_size, -1),
        "relative_end": torch.tensor([[-8.0, -4.0, 0.0]]).expand(batch_size, -1),
        "span": torch.full((batch_size, input_blocks), 4.0),
        "block_mask": torch.ones(batch_size, input_blocks, dtype=torch.bool),
        "static_numeric": torch.randn(batch_size, 4),
        "static_numeric_mask": torch.ones(batch_size, 4, dtype=torch.bool),
        "static_categorical": torch.ones(batch_size, 5, dtype=torch.long),
    }


def _teacher_targets(batch_size: int = 2):
    values = {
        key: torch.randn(batch_size, 6, 29, width)
        for key, width in FEEDBACK_DIMS.items()
    }
    masks = {
        key: torch.ones(batch_size, 6, 29, width, dtype=torch.bool)
        for key, width in FEEDBACK_DIMS.items()
    }
    return values, masks


class MultiresEventV2ModelTest(unittest.TestCase):
    def test_forward_is_six_blocks_by_29_fields_with_relation_v2_metadata(self) -> None:
        torch.manual_seed(23)
        model = _model().eval()
        targets, target_masks = _teacher_targets()
        output = model(
            **_batch(),
            target_primitives=targets,
            target_primitive_masks=target_masks,
        )

        self.assertEqual(output["field_states"].shape, (2, 6, 29, 8))
        self.assertEqual(output["primitive_parameter_dims"], PARAMETER_DIMS)
        self.assertEqual(output["primitive_feedback_dims"], FEEDBACK_DIMS)
        self.assertEqual(set(output["primitive_parameters"]), set(PARAMETER_DIMS))
        self.assertEqual(
            output["primitive_parameters"]["categorical_hours_0_4"].shape,
            (2, 6, 29, 5),
        )
        self.assertEqual(
            output["primitive_parameters"]["respiratory_block_evidence"].shape,
            (2, 6, 29, 1),
        )
        self.assertEqual(model.target_relation_adjacency.shape, (52, 29, 29))
        self.assertEqual(model.input_target_relation_adjacency.shape, (39, 29, 37))
        self.assertEqual(model.target_decoder.target_relation_bias.relation_count, 52)
        self.assertEqual(model.target_decoder.input_target_relation_bias.relation_count, 39)
        self.assertEqual(
            tuple(model.target_field_ids.tolist()),
            (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 7, *range(14, 29), 35),
        )

    def test_single_route_state_dict_contains_both_edge_specific_parameter_banks(self) -> None:
        model = _model()
        state = model.state_dict()
        target_key = "target_decoder.target_relation_bias.edge_head_bias"
        input_key = "target_decoder.input_target_relation_bias.edge_head_bias"
        self.assertEqual(state[target_key].shape, (52, 2))
        self.assertEqual(state[input_key].shape, (39, 2))
        self.assertEqual(
            state["input_field_memory.input_only_temporal_weight"].shape,
            (8, 6),
        )
        self.assertEqual(
            len(model.target_decoder.target_relation_bias.parameter_keys),
            len(set(model.target_decoder.target_relation_bias.parameter_keys)),
        )
        self.assertEqual(
            len(model.target_decoder.input_target_relation_bias.parameter_keys),
            len(set(model.target_decoder.input_target_relation_bias.parameter_keys)),
        )

    def test_teacher_encoder_rejects_incomplete_primitive_contract(self) -> None:
        model = _model()
        targets, masks = _teacher_targets(batch_size=1)
        targets.pop("respiratory_block_evidence")
        with self.assertRaisesRegex(ValueError, "likelihood ids"):
            model.encode_teacher_targets(targets, masks, batch_size=1)

    def test_feedback_encoder_is_finite_for_wide_physical_unit_scales(self) -> None:
        encoder = PrimitiveFeedbackEncoder(8, {"lab_joint_value_state": 3}, dropout=0.0)
        values = torch.tensor([[[[-250.0, 0.0, 1_000_000.0]]]])
        masks = {"lab_joint_value_state": torch.ones_like(values, dtype=torch.bool)}
        state, valid = encoder(
            {"lab_joint_value_state": values},
            masks,
            leading_shape=(1, 1, 1),
        )
        self.assertTrue(valid.all())
        self.assertTrue(torch.isfinite(state).all())

    def test_runtime_relation_or_mode_overrides_fail_closed(self) -> None:
        model = _model().eval()
        targets, masks = _teacher_targets(batch_size=1)
        for key, value in (
            ("mode", "block"),
            ("relation_adjacency", torch.zeros(1)),
            ("disable_relation", True),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "does not accept undeclared runtime inputs"
            ):
                model(
                    **_batch(batch_size=1),
                    target_primitives=targets,
                    target_primitive_masks=masks,
                    **{key: value},
                )

    def test_cached_rollout_reuses_encoded_history_and_accepts_no_future_truth(self) -> None:
        model = _model().eval()
        batch = _batch(batch_size=1)
        memory, memory_mask, queries = model.encode_for_rollout(**batch)

        self.assertNotIn(
            "target_primitives", inspect.signature(model.encode_for_rollout).parameters
        )
        self.assertNotIn(
            "target_primitives", inspect.signature(model.rollout_from_encoded).parameters
        )
        self.assertNotIn("mode", inspect.signature(model.rollout_from_encoded).parameters)

        def sampler(_block: int, _field: int, parameters: dict[str, torch.Tensor]):
            generated_batch = next(iter(parameters.values())).shape[0]
            values = {
                name: torch.zeros(generated_batch, width)
                for name, width in FEEDBACK_DIMS.items()
            }
            masks = {
                name: torch.ones(generated_batch, width, dtype=torch.bool)
                for name, width in FEEDBACK_DIMS.items()
            }
            return values, masks

        def forbidden_reencode(**_kwargs):
            raise AssertionError("cached rollout must not re-run the input encoder")

        model._encode_input = forbidden_reencode  # type: ignore[method-assign]
        output = model.rollout_from_encoded(
            memory,
            memory_mask,
            queries,
            sampler=sampler,
        )
        self.assertEqual(output["field_states"].shape, (1, 6, 29, 8))
        self.assertEqual(
            output["generated_primitives"]["categorical_hours_0_4"].shape,
            (1, 6, 29, 1),
        )


if __name__ == "__main__":
    unittest.main()
