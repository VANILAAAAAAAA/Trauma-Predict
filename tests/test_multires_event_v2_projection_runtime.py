from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

from trauma_predict.data.multires_event_v2 import MultiresEventV2Contract
from trauma_predict.eval.multires_event_v2_projections import (
    PhysicalProjectionSpec,
    PrimitiveVectorCoordinate,
    build_standardized_primitive_schema,
    empirical_coordinate_crps,
    empirical_crps,
    project_physical_primitives,
    score_standardized_primitive_ensemble,
    standardize_primitive_trajectory,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_ROOT = Path(
    "/mnt/d/Data/trauma_predict_work/"
    "multires_event_m4_target_v2_c4_20260713/full_r8"
)


def _spec(*, gate: str = "always") -> PhysicalProjectionSpec:
    return PhysicalProjectionSpec(
        projection_id="heart_rate.LAST.NONE",
        field="heart_rate",
        field_id=0,
        field_index=0,
        operator="LAST",
        condition="NONE",
        likelihood_id="dense_joint_value_state",
        component_index=0,
        gate=gate,
        value_kind="continuous",
        unit="bpm",
    )


class PhysicalProjectionRuntimeTest(unittest.TestCase):
    def test_active_nonfinite_value_fails_closed(self) -> None:
        bank = torch.zeros((1, 6, 29, 1), dtype=torch.float64)
        bank[0, 2, 0, 0] = float("nan")
        with self.assertRaisesRegex(
            FloatingPointError, "heart_rate.LAST.NONE"
        ):
            project_physical_primitives(
                {"dense_joint_value_state": bank},
                contract=None,  # type: ignore[arg-type]
                schema=(_spec(),),
            )

    def test_masked_nonfinite_value_is_not_part_of_physical_view(self) -> None:
        bank = torch.full((1, 6, 29, 1), float("nan"), dtype=torch.float64)
        observed = torch.zeros((1, 6, 29, 1), dtype=torch.float64)
        values, masks = project_physical_primitives(
            {
                "dense_joint_value_state": bank,
                "categorical_hours_0_4": observed,
            },
            contract=None,  # type: ignore[arg-type]
            schema=(_spec(gate="observed_hours_positive"),),
        )
        self.assertTrue(torch.isfinite(values).all())
        self.assertFalse(masks.any())

    def test_one_hot_phi_rejects_out_of_support_category(self) -> None:
        bank = torch.zeros((1, 6, 29, 1), dtype=torch.float64)
        bank[0, 0, 0, 0] = 3
        masks = torch.ones_like(bank, dtype=torch.bool)
        coordinate = PrimitiveVectorCoordinate(
            coordinate_id="gcs_verbal.latest_status.one_hot_1",
            within_block_id="gcs_verbal.latest_status.one_hot_1",
            primitive_id="gcs_verbal.latest_status",
            likelihood_id="gcs_verbal_latest_status",
            block_index=0,
            field="gcs_verbal",
            field_index=0,
            component_index=0,
            encoding="one_hot",
            output_index=1,
            scale_key=None,
            minimum=1,
            maximum=2,
        )
        with self.assertRaisesRegex(ValueError, "one-hot phi coordinate"):
            standardize_primitive_trajectory(
                {"gcs_verbal_latest_status": bank},
                {"gcs_verbal_latest_status": masks},
                (coordinate,),
                {"scales": {}, "lab_scales": {}},
            )


@unittest.skipUnless(
    (TARGET_ROOT / "dataset_manifest.json").is_file(),
    "formal r8 contract is unavailable",
)
class PromotionMetricRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = MultiresEventV2Contract.from_dataset_root(TARGET_ROOT)
        cls.schema = build_standardized_primitive_schema(cls.contract)
        cls.promotion = json.loads(
            (
                REPO_ROOT / "configs/evaluation/multires_event_v2_promotion_v2.json"
            ).read_text(encoding="utf-8")
        )

    def test_coordinate_crps_matches_scalar_empirical_definition(self) -> None:
        torch.manual_seed(17)
        samples = torch.randn(100, 6, 3, dtype=torch.float64)
        truth = torch.randn(6, 3, dtype=torch.float64)
        scored = empirical_coordinate_crps(samples, truth)
        for block_index, coordinate_index in ((0, 0), (3, 1), (5, 2)):
            self.assertAlmostEqual(
                float(scored[block_index, coordinate_index]),
                empirical_crps(
                    samples[:, block_index, coordinate_index],
                    float(truth[block_index, coordinate_index]),
                ),
                places=12,
            )

    def test_structural_scores_use_exact_field_and_edge_macro_contract(self) -> None:
        torch.manual_seed(23)
        samples = torch.randn(100, 6, 160, dtype=torch.float64)
        truth = torch.randn(6, 160, dtype=torch.float64)
        result = score_standardized_primitive_ensemble(
            samples,
            truth,
            self.schema,
            self.contract.active_core_relation_edges,
            self.promotion,
        )
        self.assertEqual(len(result["field_temporal_variogram"]), 29)
        self.assertEqual(len(result["relation_variogram_by_edge"]), 21)
        self.assertAlmostEqual(
            result["field_macro_lag1_variogram_score_p0_5"],
            sum(result["field_temporal_variogram"].values()) / 29,
            places=12,
        )
        self.assertAlmostEqual(
            result["relation_edge_macro_variogram_score_p0_5"],
            sum(result["relation_variogram_by_edge"].values()) / 21,
            places=12,
        )
        self.assertGreaterEqual(result["marginal_value_crps"], 0.0)
        self.assertGreaterEqual(result["marginal_state_crps"], 0.0)

    def test_relation_score_rejects_nonexact_edge_cover(self) -> None:
        samples = torch.zeros(100, 6, 160, dtype=torch.float64)
        truth = torch.zeros(6, 160, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "21 canonical"):
            score_standardized_primitive_ensemble(
                samples,
                truth,
                self.schema,
                self.contract.active_core_relation_edges[:-1],
                self.promotion,
            )

if __name__ == "__main__":
    unittest.main()
