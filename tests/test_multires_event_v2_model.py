from __future__ import annotations

import inspect
import unittest

import torch

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


def _config(mode: str = "trajectory") -> MultiResolutionEventV2Config:
    return MultiResolutionEventV2Config(
        hidden_size=8,
        num_attention_heads=2,
        trajectory_encoder_layers=1,
        target_decoder_layers=1,
        block_compressor_layers=1,
        block_latent_count=2,
        dropout=0.0,
        mode=mode,
        primitive_head_dims=PARAMETER_DIMS,
        primitive_feedback_dims=FEEDBACK_DIMS,
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
    values = {key: torch.randn(batch_size, 6, 29, width) for key, width in FEEDBACK_DIMS.items()}
    masks = {
        key: torch.ones(batch_size, 6, 29, width, dtype=torch.bool)
        for key, width in FEEDBACK_DIMS.items()
    }
    return values, masks


def test_v2_forward_is_six_blocks_by_29_fields_with_likelihood_metadata() -> None:
    torch.manual_seed(23)
    model = MultiResolutionEventV2Model(_config()).eval()
    targets, target_masks = _teacher_targets()
    output = model(
        **_batch(),
        target_primitives=targets,
        target_primitive_masks=target_masks,
    )

    assert output["field_states"].shape == (2, 6, 29, 8)
    assert output["primitive_parameter_dims"] == PARAMETER_DIMS
    assert output["primitive_feedback_dims"] == FEEDBACK_DIMS
    assert set(output["primitive_parameters"]) == set(PARAMETER_DIMS)
    assert output["primitive_parameters"]["categorical_hours_0_4"].shape == (
        2,
        6,
        29,
        5,
    )
    assert output["primitive_parameters"]["respiratory_block_evidence"].shape == (
        2,
        6,
        29,
        1,
    )
    assert tuple(model.target_field_ids.tolist()) == (
        1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 7,
        *range(14, 29), 35,
    )


def test_three_modes_have_identical_parameter_structure_and_count() -> None:
    models = {
        mode: MultiResolutionEventV2Model(_config(mode))
        for mode in ("block", "trajectory", "relational")
    }
    state_keys = {mode: tuple(model.state_dict()) for mode, model in models.items()}
    counts = {
        mode: sum(parameter.numel() for parameter in model.parameters())
        for mode, model in models.items()
    }
    assert state_keys["block"] == state_keys["trajectory"] == state_keys["relational"]
    assert len(set(counts.values())) == 1


def test_teacher_encoder_rejects_opaque_or_incomplete_primitive_contract() -> None:
    model = MultiResolutionEventV2Model(_config())
    targets, masks = _teacher_targets(batch_size=1)
    targets.pop("respiratory_block_evidence")
    try:
        model.encode_teacher_targets(targets, masks, batch_size=1)
    except ValueError as error:
        assert "likelihood ids" in str(error)
    else:
        raise AssertionError("an incomplete target primitive dictionary must be rejected")


def test_feedback_encoder_is_finite_for_wide_physical_unit_scales() -> None:
    encoder = PrimitiveFeedbackEncoder(8, {"lab_joint_value_state": 3}, dropout=0.0)
    values = torch.tensor([[[[-250.0, 0.0, 1_000_000.0]]]])
    masks = {"lab_joint_value_state": torch.ones_like(values, dtype=torch.bool)}
    state, valid = encoder(
        {"lab_joint_value_state": values},
        masks,
        leading_shape=(1, 1, 1),
    )
    assert valid.all()
    assert torch.isfinite(state).all()


def test_cached_rollout_api_reuses_encoded_history_and_accepts_no_future_truth() -> None:
    model = MultiResolutionEventV2Model(_config(mode="block")).eval()
    batch = _batch(batch_size=1)
    memory, memory_mask, queries = model.encode_for_rollout(**batch)

    assert "target_primitives" not in inspect.signature(model.encode_for_rollout).parameters
    assert "target_primitives" not in inspect.signature(model.rollout_from_encoded).parameters

    def sampler(_block: int, _field: int, parameters: dict[str, torch.Tensor]):
        batch_size = next(iter(parameters.values())).shape[0]
        values = {
            name: torch.zeros(batch_size, width)
            for name, width in FEEDBACK_DIMS.items()
        }
        masks = {
            name: torch.ones(batch_size, width, dtype=torch.bool)
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
        mode="block",
    )
    assert output["field_states"].shape == (1, 6, 29, 8)
    assert output["generated_primitives"]["categorical_hours_0_4"].shape == (
        1,
        6,
        29,
        1,
    )


class MultiresEventV2ModelTest(unittest.TestCase):
    test_forward_shape_and_metadata = staticmethod(
        test_v2_forward_is_six_blocks_by_29_fields_with_likelihood_metadata
    )
    test_matched_parameter_structure = staticmethod(
        test_three_modes_have_identical_parameter_structure_and_count
    )
    test_teacher_contract_rejection = staticmethod(
        test_teacher_encoder_rejects_opaque_or_incomplete_primitive_contract
    )
    test_feedback_scale_transform = staticmethod(
        test_feedback_encoder_is_finite_for_wide_physical_unit_scales
    )
    test_cached_rollout = staticmethod(
        test_cached_rollout_api_reuses_encoded_history_and_accepts_no_future_truth
    )
