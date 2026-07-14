from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml

from trauma_predict.modeling.multires_event_v2.config import MultiResolutionEventV2Config
from trauma_predict.modeling.multires_event_v2.model import MultiResolutionEventV2Model
from trauma_predict.training.multires_event_v2_loss import (
    V2_PRIMITIVE_FEEDBACK_DIMS,
    V2_PRIMITIVE_HEAD_DIMS,
)


ROOT = Path(__file__).resolve().parents[1]
CONTROL_PATH = ROOT / "configs/model/multires_event_v2.yaml"
CANDIDATE_PATH = ROOT / "configs/model/multires_event_v2_capacity_48m.yaml"
DIAGNOSTIC_PATH = (
    ROOT / "configs/evaluation/multires_event_v2_capacity_diagnostic_v1.json"
)

EXPECTED_COUNTS = {
    "control_30m": 30_684_479,
    "candidate_48m": 47_801_855,
}

EXPECTED_COMPONENTS = {
    "control_30m": {
        "target_decoder": 14_359_408,
        "trajectory_encoder": 10_647_552,
        "primitive_heads": 3_001_807,
        "block_compressor": 1_779_072,
        "static_encoder": 314_496,
        "feedback_encoder": 212_352,
        "event_embedding": 155_904,
        "block_embedding": 154_368,
        "semantic_embeddings": 44_928,
        "field_queries": 14_592,
        "autoregressive_rollout": 0,
    },
    "candidate_48m": {
        "target_decoder": 22_418_992,
        "trajectory_encoder": 16_627_200,
        "primitive_heads": 4_627_663,
        "block_compressor": 2_776_800,
        "static_encoder": 485_280,
        "feedback_encoder": 311_520,
        "event_embedding": 240_960,
        "block_embedding": 239_040,
        "semantic_embeddings": 56_160,
        "field_queries": 18_240,
        "autoregressive_rollout": 0,
    },
}


def _read(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"model config is not a mapping: {path}")
    return value


def _build(value: dict, mode: str) -> MultiResolutionEventV2Model:
    architecture = dict(value["architecture"])
    architecture.update(
        mode=mode,
        primitive_head_dims=V2_PRIMITIVE_HEAD_DIMS,
        primitive_feedback_dims=V2_PRIMITIVE_FEEDBACK_DIMS,
    )
    return MultiResolutionEventV2Model(
        MultiResolutionEventV2Config.from_mapping(architecture)
    )


class MultiResolutionEventV2CapacityCandidateTests(unittest.TestCase):
    def test_capacity_diagnostic_is_predeclared_and_not_formal_authority(self) -> None:
        diagnostic = json.loads(DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
        self.assertEqual(diagnostic["status"], "frozen_before_execution")
        self.assertIs(diagnostic["formal_training_authorized"], False)
        self.assertEqual(diagnostic["mode"], "trajectory")
        self.assertEqual(
            {row["id"]: row["parameter_count"] for row in diagnostic["capacities"]},
            EXPECTED_COUNTS,
        )
        self.assertEqual(diagnostic["replication"]["optimizer_steps"], 500)
        self.assertEqual(
            diagnostic["replication"]["anchor_exposures_per_run"], 32_000
        )
        self.assertEqual(
            diagnostic["replication"]["model_and_sampler_seeds"],
            [20260714, 20260715, 20260716],
        )
        self.assertEqual(diagnostic["matched_factors"]["effective_batch"], 64)
        self.assertEqual(diagnostic["primary_estimand"]["bootstrap_unit"], "subject_id")

    def test_candidate_changes_width_only_and_is_not_formal_authority(self) -> None:
        control = _read(CONTROL_PATH)
        candidate = _read(CANDIDATE_PATH)
        control_architecture = dict(control["architecture"])
        candidate_architecture = dict(candidate["architecture"])
        differences = {
            key: (control_architecture.get(key), candidate_architecture.get(key))
            for key in set(control_architecture) | set(candidate_architecture)
            if control_architecture.get(key) != candidate_architecture.get(key)
        }
        self.assertEqual(differences, {"hidden_size": (384, 480)})
        self.assertIs(candidate["diagnostic_only"], True)
        self.assertIs(
            candidate["capacity_contract"]["formal_training_authorized"], False
        )
        self.assertEqual(
            candidate["capacity_contract"]["allowed_architecture_delta"],
            "hidden_size_only",
        )
        self.assertEqual(
            candidate["primitive_contract"], control["primitive_contract"]
        )

    def test_exact_total_and_component_counts_are_source_derived(self) -> None:
        for name, path in (
            ("control_30m", CONTROL_PATH),
            ("candidate_48m", CANDIDATE_PATH),
        ):
            model = _build(_read(path), "trajectory")
            components = {
                module_name: sum(parameter.numel() for parameter in module.parameters())
                for module_name, module in model.named_children()
            }
            self.assertEqual(sum(components.values()), EXPECTED_COUNTS[name])
            self.assertEqual(components, EXPECTED_COMPONENTS[name])

    def test_each_capacity_has_matched_mode_parameter_identity(self) -> None:
        for path in (CONTROL_PATH, CANDIDATE_PATH):
            models = {
                mode: _build(_read(path), mode)
                for mode in ("block", "trajectory", "relational")
            }
            counts = {
                mode: sum(parameter.numel() for parameter in model.parameters())
                for mode, model in models.items()
            }
            state_keys = {
                mode: tuple(model.state_dict()) for mode, model in models.items()
            }
            self.assertEqual(len(set(counts.values())), 1)
            self.assertEqual(
                state_keys["block"],
                state_keys["trajectory"],
                state_keys["relational"],
            )


if __name__ == "__main__":
    unittest.main()
