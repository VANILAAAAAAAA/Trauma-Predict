from __future__ import annotations

from collections.abc import Mapping
import inspect
import unittest
from pathlib import Path

import torch
from torch import nn
import yaml

from trauma_predict.data.multires_event_v2 import MultiresEventV2Contract
from trauma_predict.modeling.grud_h1_v2 import (
    GRUDH1JointM4Config,
    GRUDH1JointM4Model,
    build_grud_h1_joint_m4_model,
)
from trauma_predict.training.multires_event_v2_loss import (
    REGISTERED_CORE_FIELD_IDS,
    V2_PRIMITIVE_FEEDBACK_DIMS,
    V2_PRIMITIVE_HEAD_DIMS,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "configs/model/grud_h1_joint_m4_v2.yaml"


def _model() -> GRUDH1JointM4Model:
    torch.manual_seed(71)
    return GRUDH1JointM4Model(
        GRUDH1JointM4Config(hidden_size=8, dropout=0.0)
    )


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    time_steps = 4
    values = torch.randn(batch_size, time_steps, 118)
    observed = torch.zeros(batch_size, time_steps, 118, dtype=torch.bool)
    observed[:, 0, :] = True
    observed[:, 1, ::3] = True
    sequence = torch.ones(batch_size, time_steps, dtype=torch.bool)
    if batch_size > 1:
        sequence[1, 2:] = False
        observed[1, 2:] = False
    deltas = torch.arange(time_steps, dtype=torch.float32).view(1, -1, 1)
    deltas = deltas.expand(batch_size, -1, 118).clone()
    return {
        "h1_values": values,
        "h1_observed_mask": observed,
        "h1_delta_hours": deltas,
        "h1_sequence_mask": sequence,
        "static_numeric": torch.randn(batch_size, 4),
        "static_numeric_mask": torch.ones(batch_size, 4, dtype=torch.bool),
        "static_categorical": torch.ones(batch_size, 5, dtype=torch.long),
    }


def _teacher_targets(
    batch_size: int = 2,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    values = {
        key: torch.zeros(batch_size, 6, 29, width)
        for key, width in V2_PRIMITIVE_FEEDBACK_DIMS.items()
    }
    masks = {
        key: torch.ones(batch_size, 6, 29, width, dtype=torch.bool)
        for key, width in V2_PRIMITIVE_FEEDBACK_DIMS.items()
    }
    return values, masks


class GRUDH1V2ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls._torch_threads)

    def test_forward_uses_exact_joint_v2_output_contract(self) -> None:
        model = _model().eval()
        targets, masks = _teacher_targets()
        output = model(
            **_batch(),
            target_primitives=targets,
            target_primitive_masks=masks,
        )

        self.assertEqual(output["field_states"].shape, (2, 6, 29, 8))
        self.assertEqual(output["history_state"].shape, (2, 8))
        self.assertEqual(output["primitive_parameter_dims"], dict(V2_PRIMITIVE_HEAD_DIMS))
        self.assertEqual(output["primitive_feedback_dims"], dict(V2_PRIMITIVE_FEEDBACK_DIMS))
        self.assertEqual(set(output["primitive_parameters"]), set(V2_PRIMITIVE_HEAD_DIMS))
        for key, width in V2_PRIMITIVE_HEAD_DIMS.items():
            self.assertEqual(
                output["primitive_parameters"][key].shape,
                (2, 6, 29, width),
            )

        forbidden = (
            nn.MultiheadAttention,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
        )
        self.assertFalse(any(isinstance(module, forbidden) for module in model.modules()))
        self.assertFalse(any("relation" in name for name, _ in model.named_parameters()))
        self.assertNotIn("relation", inspect.signature(model.forward).parameters)

    def test_missing_values_and_padded_tail_do_not_change_history(self) -> None:
        model = _model().eval()
        batch = _batch()
        reference = model.encode_history(**batch)

        changed = {key: value.clone() for key, value in batch.items()}
        hidden_values = ~changed["h1_observed_mask"]
        changed["h1_values"][hidden_values] = 100000.0
        padded = ~changed["h1_sequence_mask"]
        changed["h1_values"][padded.unsqueeze(-1).expand_as(changed["h1_values"])] = -100000.0
        observed = model.encode_history(**changed)
        torch.testing.assert_close(observed, reference)

    def test_teacher_feedback_is_shifted_by_exactly_one_field(self) -> None:
        model = _model().eval()
        batch = _batch(batch_size=1)
        targets, masks = _teacher_targets(batch_size=1)
        baseline = model(
            **batch,
            target_primitives=targets,
            target_primitive_masks=masks,
        )["field_states"].reshape(1, 174, 8)

        changed = {key: value.clone() for key, value in targets.items()}
        likelihood = next(iter(V2_PRIMITIVE_FEEDBACK_DIMS))
        changed[likelihood].reshape(1, 174, -1)[:, 10] = 1000.0
        modified = model(
            **batch,
            target_primitives=changed,
            target_primitive_masks=masks,
        )["field_states"].reshape(1, 174, 8)

        torch.testing.assert_close(modified[:, :11], baseline[:, :11])
        self.assertFalse(torch.allclose(modified[:, 11], baseline[:, 11]))

    def test_rollout_is_registry_sampler_call_compatible(self) -> None:
        model = _model().eval()
        calls: list[tuple[int, int]] = []
        first_feedback_key = next(iter(V2_PRIMITIVE_FEEDBACK_DIMS))

        def sampler(
            block_index: int,
            field_index: int,
            parameters: Mapping[str, torch.Tensor],
        ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
            calls.append((block_index, field_index))
            self.assertEqual(set(parameters), set(V2_PRIMITIVE_HEAD_DIMS))
            batch_size = next(iter(parameters.values())).shape[0]
            values = {
                key: torch.zeros(batch_size, width)
                for key, width in V2_PRIMITIVE_FEEDBACK_DIMS.items()
            }
            masks = {
                key: torch.zeros(batch_size, width, dtype=torch.bool)
                for key, width in V2_PRIMITIVE_FEEDBACK_DIMS.items()
            }
            masks[first_feedback_key][:, 0] = True
            return values, masks

        output = model.rollout(**_batch(batch_size=1), sampler=sampler)
        self.assertEqual(calls, [(block, field) for block in range(6) for field in range(29)])
        self.assertEqual(output["field_states"].shape, (1, 6, 29, 8))
        for key, width in V2_PRIMITIVE_FEEDBACK_DIMS.items():
            self.assertEqual(output["generated_primitives"][key].shape, (1, 6, 29, width))
            self.assertEqual(
                output["generated_primitive_masks"][key].shape,
                (1, 6, 29, width),
            )

    def test_decay_parameters_receive_gradients_from_missing_intervals(self) -> None:
        model = _model().train()
        batch = _batch(batch_size=1)
        batch["h1_observed_mask"][:, 1:] = False
        batch["h1_values"][:, 0] = 2.0
        model.encode_history(**batch).square().mean().backward()
        self.assertIsNotNone(model.input_decay_weight.grad)
        self.assertIsNotNone(model.hidden_decay.weight.grad)
        self.assertTrue(torch.isfinite(model.input_decay_weight.grad).all())
        self.assertTrue(torch.isfinite(model.hidden_decay.weight.grad).all())
        self.assertGreater(float(model.input_decay_weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.hidden_decay.weight.grad.abs().sum()), 0.0)

    def test_builder_accepts_model_yaml_and_checks_task_contract(self) -> None:
        authored = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
        parsed = GRUDH1JointM4Config.from_mapping(authored)
        self.assertEqual(parsed.input_channels, 118)
        self.assertEqual(parsed.hidden_size, 192)
        self.assertEqual(parsed.future_block_count, 6)
        self.assertEqual(parsed.target_field_count, 29)
        model = build_grud_h1_joint_m4_model(
            {"architecture": {"hidden_size": 8, "dropout": 0.0}},
            contract={
                "target": {
                    "ordered_m4_blocks": 6,
                    "field_processes": 29,
                    "stochastic_factors": 414,
                }
            },
        )
        self.assertIsInstance(model, GRUDH1JointM4Model)
        with self.assertRaisesRegex(ValueError, "ordered_m4_blocks"):
            build_grud_h1_joint_m4_model(
                GRUDH1JointM4Config(hidden_size=8),
                contract={"target": {"ordered_m4_blocks": 5}},
            )

    def test_builder_accepts_real_v2_contract_object(self) -> None:
        layouts = {
            key: {"width": width, "layout": "test-layout"}
            for key, width in V2_PRIMITIVE_HEAD_DIMS.items()
        }
        contract = MultiresEventV2Contract(
            dataset_root=ROOT,
            manifest={},
            process_registry={
                "scope": {
                    "future_blocks": [f"M4_{index:02d}" for index in range(1, 7)],
                    "expanded_enabled_core_primitives": 414,
                }
            },
            emission_registry={
                "enabled_core_head_contract": {"layouts": layouts}
            },
            projection_registry={},
            contract_hashes={},
            contract_bundle_hash="",
            dense_fields=(),
            ordinal_fields=(),
            verbal_field="gcs_verbal",
            lab_fields=(),
            respiratory_field="respiratory_support",
            vasopressor_field="vasopressor",
            ned_field="ned",
            uop_field="urine_output",
            dense_abnormal_conditions={},
            respiratory_modalities=(),
            vasopressor_agents=(),
            ordinal_max={},
            registered_core_fields=tuple(f"field_{index}" for index in range(29)),
            registered_core_field_ids=tuple(REGISTERED_CORE_FIELD_IDS),
            relation_types=(),
            relation_type_lags=(),
            relation_adjacency=(),
            active_core_relation_edges=(),
            relation_total_edges=0,
            relation_active_core_edges=0,
            relation_deferred_edges=0,
        )
        model = build_grud_h1_joint_m4_model(
            GRUDH1JointM4Config(hidden_size=8),
            contract=contract,
        )
        self.assertIsInstance(model, GRUDH1JointM4Model)


if __name__ == "__main__":
    unittest.main()
