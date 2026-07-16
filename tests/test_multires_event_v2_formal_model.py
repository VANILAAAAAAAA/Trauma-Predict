from __future__ import annotations

from pathlib import Path
import unittest

from trauma_predict.data.multires_event_v2 import MultiresEventV2RelationContract
from trauma_predict.training.multires_event_v2 import (
    EXPECTED_FORMAL_MODEL_PARAMETER_COUNT,
    build_multires_event_v2_model,
    load_multires_event_v2_configs,
)


ROOT = Path(__file__).resolve().parents[1]


class RelationV2FormalModelTest(unittest.TestCase):
    def test_only_the_full_relation_v2_model_contract_is_active(self) -> None:
        train, _, model, _, _ = load_multires_event_v2_configs(
            ROOT / "configs/train/p100_multires_event_v2_relation_v2.yaml",
            repo_root=ROOT,
        )
        built = build_multires_event_v2_model(
            model,
            relation_contract=MultiresEventV2RelationContract.from_default_config(),
        )
        self.assertEqual(train["run_name"], "p100_multires_event_v2_relation_v2")
        self.assertEqual(
            sum(parameter.numel() for parameter in built.parameters()),
            EXPECTED_FORMAL_MODEL_PARAMETER_COUNT,
        )
        self.assertFalse(
            list((ROOT / "configs/model").glob("multires_event_v2_capacity*.yaml"))
        )


if __name__ == "__main__":
    unittest.main()
