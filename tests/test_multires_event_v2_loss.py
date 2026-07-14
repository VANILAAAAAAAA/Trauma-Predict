from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

import trauma_predict.modeling.multires_event_v2.emissions as emissions_module
import trauma_predict.training.multires_event_v2_loss as loss_module

from trauma_predict.training.multires_event_v2_loss import (
    EXPECTED_ENABLED_CORE_PRIMITIVES,
    REGISTERED_CORE_FIELD_IDS,
    PrimitiveLogProb,
    RegistryPrimitiveSampler,
    V2_PRIMITIVE_FEEDBACK_DIMS,
    V2_PRIMITIVE_HEAD_DIMS,
    V2_EMISSION_REGISTRY_VERSION,
    build_enabled_core_factors,
    compute_multires_event_v2_loss,
    expand_enabled_core_primitives,
    validate_emission_registry_head_contract,
)


LABS = (
    "lactate",
    "base_excess",
    "bicarbonate",
    "creatinine",
    "bun",
    "wbc",
    "hemoglobin",
    "platelet_count",
    "inr",
    "sodium",
    "potassium",
    "chloride",
    "glucose",
)
DENSE = (
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "mean_arterial_pressure",
    "respiratory_rate",
    "temperature",
    "spo2",
    "fio2",
    "peep",
)
FIELDS = (
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "mean_arterial_pressure",
    "respiratory_rate",
    "temperature",
    "gcs_eye",
    "gcs_verbal",
    "gcs_motor",
    "respiratory_support",
    "fio2",
    "peep",
    "spo2",
    *LABS,
    "vasopressor_support",
    "norepinephrine_equivalent_dose",
    "urine_output",
)
SCALE_HASH = "a" * 64


def _primitive(likelihood_id: str, suffix: str) -> dict[str, str]:
    return {
        "primitive_id_template": f"{{field}}.{{block}}.{suffix}",
        "log_prob_id_template": f"logp::{{field}}::{{block}}::{suffix}",
        "likelihood_id": likelihood_id,
    }


def _process(fields_from: str, *primitives: tuple[str, str]) -> dict[str, object]:
    return {
        "fields_from": fields_from,
        "scope": "per_block",
        "objective_status": "enabled_core",
        "primitives": [_primitive(*primitive) for primitive in primitives],
    }


def _registry() -> dict[str, object]:
    abnormal = (
        "heart_rate",
        "systolic_bp",
        "mean_arterial_pressure",
        "respiratory_rate",
        "temperature",
        "spo2",
    )
    return {
        "version": "2026-07-14-r9",
        "scope": {
            "future_blocks": tuple(f"M4_{index:02d}" for index in range(1, 7)),
            "expanded_enabled_core_primitives": EXPECTED_ENABLED_CORE_PRIMITIVES,
        },
        "registered_core_field_order": [
            {"position": position, "field_id": field_id, "field": FIELDS[position]}
            for position, field_id in enumerate(REGISTERED_CORE_FIELD_IDS)
        ],
        "field_sets": {
            "dense_continuous": DENSE,
            "dense_abnormal": abnormal,
            "gcs_ordinal_enabled": ("gcs_eye", "gcs_motor"),
            "gcs_verbal_reaggregated": ("gcs_verbal",),
            "intermittent_labs": LABS,
            "respiratory_support": ("respiratory_support",),
            "vasopressor_support": ("vasopressor_support",),
            "ned": ("norepinephrine_equivalent_dose",),
            "uop": ("urine_output",),
        },
        "condition_sets": {
            "dense_abnormal": {
                "heart_rate": ("HR_LT40", "HR_GT120"),
                "systolic_bp": ("SBP_LT90",),
                "mean_arterial_pressure": ("MAP_LT65", "MAP_GE65_LT70"),
                "respiratory_rate": ("RR_LT8", "RR_GE22"),
                "temperature": ("TEMP_GE38",),
                "spo2": ("SPO2_LE93",),
            }
        },
        "field_parameters": {
            "gcs_eye": {"ordinal_max": 4, "legal_triple_count": 20},
            "gcs_motor": {"ordinal_max": 6, "legal_triple_count": 56},
        },
        "process_templates": [
            _process("dense_continuous", ("categorical_hours_0_4", "dense.hours")),
            _process("dense_continuous", ("dense_joint_value_state", "dense.state")),
            _process(
                "dense_abnormal",
                ("dense_abnormal_duration_vector", "dense.abnormal"),
            ),
            _process("gcs_ordinal_enabled", ("categorical_hours_0_4", "gcs.hours")),
            _process("gcs_ordinal_enabled", ("gcs_ordinal_triple", "gcs.state")),
            _process(
                "gcs_verbal_reaggregated",
                ("categorical_hours_0_4", "verbal.hours"),
                (
                    "gcs_verbal_ungradable_hours_given_observed",
                    "verbal.ungradable",
                ),
                ("gcs_verbal_latest_status", "verbal.status"),
                (
                    "gcs_verbal_gradable_ordinal_triple",
                    "verbal.gradable_state",
                ),
            ),
            _process(
                "intermittent_labs",
                ("hurdle_negative_binomial_count", "lab.count"),
            ),
            _process("intermittent_labs", ("lab_joint_value_state", "lab.state")),
            _process(
                "respiratory_support",
                ("respiratory_block_evidence", "resp.block"),
                ("respiratory_edge_evidence_given_block", "resp.edge_evidence"),
            ),
            _process(
                "respiratory_support",
                ("respiratory_occupancy_vector", "resp.occupancy"),
            ),
            _process(
                "respiratory_support",
                ("respiratory_edge_state", "resp.edge_state"),
            ),
            _process(
                "respiratory_support",
                ("respiratory_onset_vector", "resp.onset"),
            ),
            _process(
                "vasopressor_support",
                ("vasopressor_duration_vector", "vaso.duration"),
            ),
            _process(
                "vasopressor_support",
                ("vasopressor_edge_state_vector", "vaso.edge"),
            ),
            _process(
                "vasopressor_support",
                ("vasopressor_onset_vector", "vaso.onset"),
            ),
            _process("ned", ("ned_joint_value_state", "ned.state")),
            _process("uop", ("hurdle_negative_binomial_count", "uop.count")),
            _process("uop", ("uop_sum_given_count", "uop.sum")),
        ],
    }


def _metadata() -> dict[str, object]:
    return {
        "field_ids": REGISTERED_CORE_FIELD_IDS,
        "valid_ranges": {field: (-20.0, 200.0) for field in DENSE},
        "lab_scale": {
            "schema": "multires_event_v2_lab_affine_scale_v1",
            "version": "2026-07-13-train-target-windows-v1",
            "coordinate_contract": "lab_shared_affine_canonical_v1",
            "content_sha256": SCALE_HASH,
            "fields": {
                field: {"unit": "registered_unit", "center": 10.0, "scale": 2.0} for field in LABS
            },
        },
    }


def _legacy_conditioned_component(
    raw_parameters: torch.Tensor,
    preceding_values: torch.Tensor,
    *,
    component: int,
    component_count: int,
    parameter_width: int,
    transform: str,
) -> torch.Tensor:
    """Reference the pre-optimization full-table sampling path."""

    return emissions_module.lower_triangular_conditioned_parameters(
        raw_parameters,
        preceding_values,
        component_count=component_count,
        parameter_width=parameter_width,
        transform=transform,
    )[..., component, :]


def _legacy_sample_zoi_logit_normal(
    parameters: emissions_module.ZOILogitNormalParameters,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
    component_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference the pre-optimization device-scalar pushforward."""

    if parameters.mixture_logits.shape[-1:] != (3,):
        raise ValueError("ZOI-logit-Normal sampling requires three mixture logits")
    shape = parameters.mixture_logits.shape[:-1]
    emissions_module._require_shape(
        parameters.interior_loc,
        shape,
        "sampled ZOI-logit-Normal interior_loc",
    )
    emissions_module._require_shape(
        parameters.interior_scale_raw,
        shape,
        "sampled ZOI-logit-Normal interior_scale_raw",
    )
    if not upper > lower:
        raise ValueError("ZOI-logit-Normal upper bound must exceed lower bound")
    emissions_module._require(
        torch.isfinite(parameters.mixture_logits),
        "sampled ZOI-logit-Normal mixture logits must be finite",
    )
    emissions_module._require(
        torch.isfinite(parameters.interior_loc)
        & torch.isfinite(parameters.interior_scale_raw),
        "sampled ZOI-logit-Normal interior parameters must be finite",
    )
    branch = emissions_module.sample_categorical(
        parameters.mixture_logits.float(), component_mask
    )
    q = torch.distributions.Normal(
        parameters.interior_loc.double(),
        F.softplus(parameters.interior_scale_raw.double())
        + emissions_module.POSITIVE_SCALE_FLOOR,
        validate_args=False,
    ).sample()
    interior = torch.sigmoid(q)
    emissions_module._require(
        (~branch.eq(1))
        | (torch.isfinite(interior) & interior.gt(0.0) & interior.lt(1.0)),
        "sampled logit-Normal interior saturated to an endpoint",
    )
    unit = torch.where(
        branch.eq(0),
        torch.zeros_like(interior),
        torch.where(branch.eq(2), torch.ones_like(interior), interior),
    )
    return unit.new_tensor(lower) + unit.new_tensor(upper - lower) * unit


def _sample_complete_registered_trajectory(
    device: torch.device,
    heads: dict[str, torch.Tensor],
) -> tuple[
    list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    tuple[torch.Tensor, ...],
]:
    sampler = RegistryPrimitiveSampler(
        _registry(),
        _metadata(),
        expected_lab_scale_artifact_hash=SCALE_HASH,
    )
    seed = 20260713
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    feedback = []
    for block_index in range(6):
        for field_index in range(29):
            values, masks = sampler(
                block_index,
                field_index,
                {
                    likelihood_id: bank[:, block_index, field_index]
                    for likelihood_id, bank in heads.items()
                },
            )
            feedback.append(
                (
                    {key: value.clone() for key, value in values.items()},
                    {key: value.clone() for key, value in masks.items()},
                )
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    states = [torch.get_rng_state().clone()]
    if device.type == "cuda":
        states.append(torch.cuda.get_rng_state(device).clone())
    return feedback, tuple(states)


class MultiresEventV2LossTest(unittest.TestCase):
    def test_r8_emission_layout_and_numerical_contract_fail_closed(self) -> None:
        emission = {
            "version": V2_EMISSION_REGISTRY_VERSION,
            "global_contract": {
                "numerical_constants": {
                    "positive_scale_floor": 0.0001,
                    "unit_interval_interior_family": "zero_one_inflated_logit_normal",
                    "unit_interval_interior_measure": "Lebesgue_dq",
                }
            },
            "enabled_core_head_contract": {
                "layouts": {
                    likelihood_id: {"width": width}
                    for likelihood_id, width in V2_PRIMITIVE_HEAD_DIMS.items()
                }
            },
        }
        validate_emission_registry_head_contract(emission)
        emission["enabled_core_head_contract"]["layouts"]["respiratory_occupancy_vector"][
            "width"
        ] = 36
        with self.assertRaisesRegex(ValueError, "differ from the model contract"):
            validate_emission_registry_head_contract(emission)
        emission["enabled_core_head_contract"]["layouts"]["respiratory_occupancy_vector"][
            "width"
        ] = V2_PRIMITIVE_HEAD_DIMS["respiratory_occupancy_vector"]
        numerical = emission["global_contract"]["numerical_constants"]
        numerical["unit_interval_interior_measure"] = "Lebesgue_du"
        with self.assertRaisesRegex(ValueError, "numerical constants differ"):
            validate_emission_registry_head_contract(emission)

    def test_full_414_factor_ancestral_roundtrip_and_gradients(self) -> None:
        registry = _registry()
        metadata = _metadata()
        specs = expand_enabled_core_primitives(registry)
        self.assertEqual(len(specs), 414)
        self.assertEqual(len({spec.primitive_id for spec in specs}), 414)

        batch_size = 2
        heads = {
            likelihood_id: torch.zeros(batch_size, 6, 29, width, requires_grad=True)
            for likelihood_id, width in V2_PRIMITIVE_HEAD_DIMS.items()
        }
        # Registry samples retain float64 physical support.  The feedback
        # encoder performs the separate neural-input float32 cast.
        with torch.no_grad():
            dense_head = heads["dense_joint_value_state"]
            dense_head[..., 0] = 100.0
            dense_head[..., 1] = -100.0
            dense_head[..., 2] = 100.0
            dense_head[..., 3:5] = -100.0
        sampler = RegistryPrimitiveSampler(
            registry,
            metadata,
            expected_lab_scale_artifact_hash=SCALE_HASH,
        )
        generated = {
            likelihood_id: torch.zeros(
                batch_size, 6, 29, width, dtype=torch.float64
            )
            for likelihood_id, width in V2_PRIMITIVE_FEEDBACK_DIMS.items()
        }
        component_masks = {
            likelihood_id: torch.zeros(batch_size, 6, 29, width, dtype=torch.bool)
            for likelihood_id, width in V2_PRIMITIVE_FEEDBACK_DIMS.items()
        }
        torch.manual_seed(7)
        for block_index in range(6):
            for field_index in range(29):
                current = {
                    likelihood_id: value[:, block_index, field_index].detach()
                    for likelihood_id, value in heads.items()
                }
                values, masks = sampler(block_index, field_index, current)
                for likelihood_id in generated:
                    self.assertEqual(values[likelihood_id].dtype, torch.float64)
                    generated[likelihood_id][:, block_index, field_index] = values[likelihood_id]
                    component_masks[likelihood_id][:, block_index, field_index] = masks[
                        likelihood_id
                    ]

        target_primitives = {
            likelihood_id: (
                value[..., 0] if V2_PRIMITIVE_FEEDBACK_DIMS[likelihood_id] == 1 else value
            )
            for likelihood_id, value in generated.items()
        }
        physical = {
            "dense_joint_value_state",
            "lab_joint_value_state",
            "respiratory_occupancy_vector",
            "vasopressor_duration_vector",
            "ned_joint_value_state",
            "uop_sum_given_count",
        }
        for likelihood_id in physical:
            target_primitives[likelihood_id] = target_primitives[likelihood_id].double()
        target_masks = {
            likelihood_id: value.any(dim=-1) for likelihood_id, value in component_masks.items()
        }
        factors = build_enabled_core_factors(
            {"primitive_parameters": heads},
            {
                "target_primitives": target_primitives,
                "target_primitive_masks": target_masks,
                "target_primitive_metadata": metadata,
            },
            registry,
            expected_lab_scale_artifact_hash=SCALE_HASH,
        )
        self.assertEqual(len(factors), 414)
        self.assertTrue(torch.stack([factor.log_prob for factor in factors]).isfinite().all())
        self.assertEqual({factor.log_prob.dtype for factor in factors}, {torch.float32})
        wrong_dtype = dict(target_primitives)
        wrong_dtype["dense_joint_value_state"] = wrong_dtype[
            "dense_joint_value_state"
        ].float()
        with self.assertRaisesRegex(ValueError, "physical target banks"):
            build_enabled_core_factors(
                {"primitive_parameters": heads},
                {
                    "target_primitives": wrong_dtype,
                    "target_primitive_masks": target_masks,
                    "target_primitive_metadata": metadata,
                },
                registry,
                expected_lab_scale_artifact_hash=SCALE_HASH,
            )

        inactive_targets = dict(target_primitives)
        inactive_masks = dict(target_masks)
        inactive_targets["categorical_hours_0_4"] = target_primitives[
            "categorical_hours_0_4"
        ].clone()
        inactive_targets["categorical_hours_0_4"][0, 0, 0] = 0
        inactive_targets["dense_abnormal_duration_vector"] = target_primitives[
            "dense_abnormal_duration_vector"
        ].clone()
        inactive_targets["dense_abnormal_duration_vector"][0, 0, 0] = 0
        for likelihood_id in (
            "dense_joint_value_state",
            "dense_abnormal_duration_vector",
        ):
            inactive_masks[likelihood_id] = target_masks[likelihood_id].clone()
            inactive_masks[likelihood_id][0, 0, 0] = False
        for invalid_value in (float("nan"), float("inf")):
            with self.subTest(nonfinite=invalid_value):
                invalid_heads = dict(heads)
                invalid_dense = heads["dense_joint_value_state"].detach().clone()
                invalid_dense[0, 0, 0, 5] = invalid_value
                invalid_heads["dense_joint_value_state"] = invalid_dense
                with self.assertRaisesRegex(FloatingPointError, "non-finite raw parameters"):
                    build_enabled_core_factors(
                        {"primitive_parameters": invalid_heads},
                        {
                            "target_primitives": inactive_targets,
                            "target_primitive_masks": inactive_masks,
                            "target_primitive_metadata": metadata,
                        },
                        registry,
                        expected_lab_scale_artifact_hash=SCALE_HASH,
                    )
        result = compute_multires_event_v2_loss(
            factors,
            expected_primitive_ids=[spec.primitive_id for spec in specs],
        )
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        for likelihood_id, value in heads.items():
            self.assertIsNotNone(value.grad, likelihood_id)
            self.assertTrue(torch.isfinite(value.grad).all(), likelihood_id)

    def test_sampling_optimizations_are_full_trajectory_and_rng_bitwise_exact(self) -> None:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            with self.subTest(device=str(device)):
                generator = torch.Generator(device=device).manual_seed(9173)
                heads = {
                    likelihood_id: torch.randn(
                        (2, 6, 29, width),
                        generator=generator,
                        device=device,
                    )
                    * 0.15
                    for likelihood_id, width in V2_PRIMITIVE_HEAD_DIMS.items()
                }
                # Recreate the former implementation in-process.  This covers
                # every registered position and all three autoregressive vector
                # families, not only isolated helper outputs.
                with (
                    patch.object(
                        emissions_module,
                        "_lower_triangular_conditioned_component",
                        new=_legacy_conditioned_component,
                    ),
                    patch.object(
                        emissions_module,
                        "sample_zoi_logit_normal",
                        new=_legacy_sample_zoi_logit_normal,
                    ),
                    patch.object(
                        loss_module,
                        "sample_zoi_logit_normal",
                        new=_legacy_sample_zoi_logit_normal,
                    ),
                ):
                    legacy_feedback, legacy_rng = _sample_complete_registered_trajectory(
                        device, heads
                    )
                optimized_feedback, optimized_rng = _sample_complete_registered_trajectory(
                    device, heads
                )

                self.assertEqual(len(legacy_feedback), 6 * 29)
                self.assertEqual(len(optimized_feedback), 6 * 29)
                for position, (legacy, optimized) in enumerate(
                    zip(legacy_feedback, optimized_feedback, strict=True)
                ):
                    legacy_values, legacy_masks = legacy
                    optimized_values, optimized_masks = optimized
                    self.assertEqual(legacy_values.keys(), optimized_values.keys())
                    self.assertEqual(legacy_masks.keys(), optimized_masks.keys())
                    for likelihood_id in V2_PRIMITIVE_FEEDBACK_DIMS:
                        self.assertTrue(
                            torch.equal(
                                legacy_values[likelihood_id],
                                optimized_values[likelihood_id],
                            ),
                            f"feedback drift at position={position}, "
                            f"likelihood={likelihood_id}",
                        )
                        self.assertTrue(
                            torch.equal(
                                legacy_masks[likelihood_id],
                                optimized_masks[likelihood_id],
                            ),
                            f"mask drift at position={position}, "
                            f"likelihood={likelihood_id}",
                        )
                self.assertEqual(len(legacy_rng), len(optimized_rng))
                for index, (legacy_state, optimized_state) in enumerate(
                    zip(legacy_rng, optimized_rng, strict=True)
                ):
                    self.assertTrue(
                        torch.equal(legacy_state, optimized_state),
                        f"RNG state drift at state index {index} on {device}",
                    )

    def test_scale_contract_and_r8_field_order_fail_closed(self) -> None:
        registry = _registry()
        metadata = _metadata()
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            RegistryPrimitiveSampler(
                registry,
                metadata,
                expected_lab_scale_artifact_hash="b" * 64,
            )
        invalid = copy.deepcopy(metadata)
        invalid["lab_scale"]["fields"][LABS[0]]["scale"] = 0.0
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            RegistryPrimitiveSampler(
                registry,
                invalid,
                expected_lab_scale_artifact_hash=SCALE_HASH,
            )
        wrong_order = copy.deepcopy(metadata)
        wrong_order["field_ids"] = tuple(range(1, 29)) + (35,)
        with self.assertRaisesRegex(ValueError, "registry order"):
            RegistryPrimitiveSampler(
                registry,
                wrong_order,
                expected_lab_scale_artifact_hash=SCALE_HASH,
            )

    def test_registry_sampler_nonfinite_parameters_fail_closed_on_cpu(self) -> None:
        sampler = RegistryPrimitiveSampler(
            _registry(),
            _metadata(),
            expected_lab_scale_artifact_hash=SCALE_HASH,
        )
        parameters = {
            likelihood_id: torch.zeros(2, width)
            for likelihood_id, width in V2_PRIMITIVE_HEAD_DIMS.items()
        }
        parameters["categorical_hours_0_4"][0, 0] = torch.nan
        with self.assertRaisesRegex(FloatingPointError, "non-finite raw sampling"):
            sampler(0, 0, parameters)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA async assertion requires CUDA")
    def test_registry_sampler_nonfinite_parameters_fail_closed_on_cuda(self) -> None:
        root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                (str(root / "src"), str(root), environment.get("PYTHONPATH", "")),
            )
        )
        code = """
import torch
from tests.test_multires_event_v2_loss import SCALE_HASH, _metadata, _registry
from trauma_predict.training.multires_event_v2_loss import (
    RegistryPrimitiveSampler,
    V2_PRIMITIVE_HEAD_DIMS,
)
sampler = RegistryPrimitiveSampler(
    _registry(), _metadata(), expected_lab_scale_artifact_hash=SCALE_HASH
)
parameters = {
    name: torch.zeros(2, width, device="cuda")
    for name, width in V2_PRIMITIVE_HEAD_DIMS.items()
}
parameters["categorical_hours_0_4"][0, 0] = torch.nan
sampler(0, 0, parameters)
torch.cuda.synchronize()
"""
        result = subprocess.run(
            (sys.executable, "-c", code),
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        diagnostic = result.stdout + result.stderr
        self.assertTrue(
            "non-finite raw sampling" in diagnostic
            or "device-side assert" in diagnostic,
            diagnostic,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA teacher check requires CUDA")
    def test_teacher_loss_is_finite_backward_and_nonfinite_fails_closed_on_cuda(self) -> None:
        raw = torch.tensor([0.5, -0.25], device="cuda", requires_grad=True)
        factors = tuple(
            PrimitiveLogProb(
                f"factor_{index}",
                "categorical_hours_0_4",
                -raw.square() / 414.0,
            )
            for index in range(414)
        )
        result = compute_multires_event_v2_loss(factors)
        result["loss"].backward()
        torch.cuda.synchronize()
        self.assertTrue(bool(torch.isfinite(result["loss"])))
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))

        root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                (str(root / "src"), str(root), environment.get("PYTHONPATH", "")),
            )
        )
        code = """
import torch
from trauma_predict.training.multires_event_v2_loss import (
    PrimitiveLogProb,
    compute_multires_event_v2_loss,
)
factor = PrimitiveLogProb(
    "invalid", "categorical_hours_0_4", torch.tensor([float("nan")], device="cuda")
)
compute_multires_event_v2_loss((factor,))
torch.cuda.synchronize()
"""
        invalid = subprocess.run(
            (sys.executable, "-c", code),
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertNotEqual(invalid.returncode, 0)
        diagnostic = invalid.stdout + invalid.stderr
        self.assertTrue(
            "non-finite enabled-core" in diagnostic
            or "device-side assert" in diagnostic,
            diagnostic,
        )

    def test_objective_is_direct_sum_without_target_count_normalization(self) -> None:
        factors = (
            PrimitiveLogProb("a", "categorical_hours_0_4", torch.tensor([-1.0, -2.0])),
            PrimitiveLogProb("b", "categorical_hours_0_4", torch.tensor([-3.0, -4.0])),
        )
        result = compute_multires_event_v2_loss(factors)
        torch.testing.assert_close(result["joint_log_prob"], torch.tensor([-4.0, -6.0]))
        torch.testing.assert_close(result["loss"], torch.tensor(5.0))

        with self.assertRaisesRegex(ValueError, "exactly once"):
            compute_multires_event_v2_loss((factors[0], factors[0]))
        projection = PrimitiveLogProb(
            "projection",
            "categorical_hours_0_4",
            torch.tensor([-1.0, -1.0]),
            source_kind="deterministic_projection",
        )
        with self.assertRaisesRegex(ValueError, "cannot contribute"):
            compute_multires_event_v2_loss((projection,))


if __name__ == "__main__":
    unittest.main()
