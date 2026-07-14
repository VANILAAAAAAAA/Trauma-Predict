from __future__ import annotations

import itertools
import math
import unittest

import torch

from trauma_predict.modeling.multires_event_v2.emissions import (
    DenseValueParameters,
    DenseValueTarget,
    EmissionSupportError,
    LabValueParameters,
    LabValueTarget,
    NEDParameters,
    NEDTarget,
    RespiratoryOccupancyParameters,
    StudentTParameters,
    UOPParameters,
    ZOILogitNormalParameters,
    autoregressive_binary_vector_log_prob,
    autoregressive_hurdle_count_vector_log_prob,
    categorical_log_prob,
    dense_abnormal_class_masks,
    dense_abnormal_duration_log_prob,
    dense_joint_value_log_prob,
    gcs_verbal_latest_status_log_prob,
    hurdle_negative_binomial_log_prob,
    lab_joint_value_log_prob,
    legal_gcs_triple_log_prob,
    legal_ordinal_triples,
    lower_triangular_conditioned_parameters,
    masked_categorical_log_prob,
    ned_joint_value_log_prob,
    ordinal_triple_class_mask,
    respiratory_edge_evidence_log_prob,
    respiratory_occupancy_log_prob,
    respiratory_onset_log_prob,
    sample_autoregressive_hurdle_count_vector,
    sample_categorical,
    sample_dense_abnormal_duration,
    sample_zoi_logit_normal,
    uop_sum_log_prob,
    vasopressor_duration_log_prob,
    vasopressor_onset_log_prob,
    zoi_logit_normal_log_prob,
)
from trauma_predict.training.multires_event_v2_loss import (
    _sample_dense_value_state,
    _sample_lab_value_state,
    _sample_ned_state,
)


def _leaf(*shape: int, value: float = 0.0) -> torch.Tensor:
    return torch.full(shape, value, dtype=torch.float64, requires_grad=True)


def _zoiln(batch: int) -> ZOILogitNormalParameters:
    return ZOILogitNormalParameters(_leaf(batch, 3), _leaf(batch), _leaf(batch))


def _student(batch: int) -> StudentTParameters:
    return StudentTParameters(_leaf(batch), _leaf(batch), _leaf(batch))


class MultiresEventV2EmissionTest(unittest.TestCase):
    def test_categorical_and_masked_categorical_are_normalized(self) -> None:
        logits = torch.tensor([[0.2, -0.1, 0.8]], dtype=torch.float64, requires_grad=True)
        targets = torch.arange(3).view(3, 1)
        tiled = logits.expand(3, -1)
        probability = categorical_log_prob(tiled, targets.squeeze(-1)).exp()
        self.assertAlmostEqual(float(probability.sum().detach()), 1.0, places=12)

        mask = torch.tensor([True, False, True])
        legal_targets = torch.tensor([0, 2])
        legal_logits = logits.expand(2, -1)
        masked_probability = masked_categorical_log_prob(
            legal_logits,
            legal_targets,
            mask,
        ).exp()
        self.assertAlmostEqual(float(masked_probability.sum().detach()), 1.0, places=12)
        with self.assertRaises(EmissionSupportError):
            masked_categorical_log_prob(logits, torch.tensor([1]), mask)

    def test_hurdle_negative_binomial_sums_to_one_and_has_finite_gradients(self) -> None:
        count = torch.arange(500, dtype=torch.float64)
        gate_base = _leaf(1, value=0.3)
        total_raw_base = _leaf(1, value=0.7)
        nb_logits_base = _leaf(1, value=-1.4)
        gate = gate_base.expand_as(count)
        total_raw = total_raw_base.expand_as(count)
        nb_logits = nb_logits_base.expand_as(count)
        log_prob = hurdle_negative_binomial_log_prob(count, gate, total_raw, nb_logits)
        self.assertAlmostEqual(float(log_prob.exp().sum().detach()), 1.0, places=10)
        loss = -log_prob[[0, 1, 4]].mean()
        loss.backward()
        for value in (gate_base, total_raw_base, nb_logits_base):
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_zoi_logit_normal_boundary_masses_and_q_density_normalize(self) -> None:
        points = 20000
        q_width = 16.0
        q = -8.0 + (torch.arange(points, dtype=torch.float64) + 0.5) * q_width / points
        midpoint = 2.0 + 4.0 * torch.sigmoid(q)
        values = torch.cat((torch.tensor([2.0, 6.0], dtype=torch.float64), midpoint))
        mixture = _leaf(values.numel(), 3)
        location = _leaf(values.numel())
        scale_raw = _leaf(values.numel())
        log_prob = zoi_logit_normal_log_prob(
            values,
            ZOILogitNormalParameters(mixture, location, scale_raw),
            lower=2.0,
            upper=6.0,
        )
        total = log_prob[:2].exp().sum() + log_prob[2:].exp().sum() * q_width / points
        self.assertLess(abs(float(total.detach()) - 1.0), 1e-4)
        self.assertTrue(torch.isfinite(log_prob[:2]).all())
        (-log_prob[[0, 1, 100]].mean()).backward()
        self.assertTrue(torch.isfinite(mixture.grad).all())
        self.assertTrue(torch.isfinite(location.grad).all())
        self.assertTrue(torch.isfinite(scale_raw.grad).all())

    def test_legal_gcs_state_spaces_and_within_hour_counterexample(self) -> None:
        self.assertEqual(legal_ordinal_triples(4).shape[0], 20)
        self.assertEqual(legal_ordinal_triples(5).shape[0], 35)
        self.assertEqual(legal_ordinal_triples(6).shape[0], 56)

        states = legal_ordinal_triples(5)
        logits = torch.zeros((states.shape[0], 35), dtype=torch.float64)
        counts = torch.full((states.shape[0],), 3)
        probability = legal_gcs_triple_log_prob(
            logits,
            states,
            maximum=5,
            observation_count=counts,
        ).exp()
        self.assertAlmostEqual(float(probability.sum()), 1.0, places=12)

        diagonal = states[(states[:, 0] == states[:, 1]) & (states[:, 1] == states[:, 2])]
        diagonal_probability = legal_gcs_triple_log_prob(
            torch.zeros((5, 35), dtype=torch.float64),
            diagonal,
            maximum=5,
            observation_count=torch.ones(5),
        ).exp()
        self.assertAlmostEqual(float(diagonal_probability.sum()), 1.0, places=12)
        endpoint = states[(states[:, 1] == states[:, 0]) | (states[:, 1] == states[:, 2])]
        endpoint_probability = legal_gcs_triple_log_prob(
            torch.zeros((endpoint.shape[0], 35), dtype=torch.float64),
            endpoint,
            maximum=5,
            observation_count=torch.full((endpoint.shape[0],), 2),
        ).exp()
        self.assertAlmostEqual(float(endpoint_probability.sum()), 1.0, places=12)
        # H=1 counts an observed hour, but the audit found multiple eye/motor
        # measurements within that hour, so non-diagonal triples remain legal.
        unrestricted = legal_gcs_triple_log_prob(
            torch.zeros((1, 20), dtype=torch.float64),
            torch.tensor([[1, 1, 2]]),
            maximum=4,
            observation_count=torch.ones(1),
            source_semantics="raw_point",
        )
        self.assertTrue(torch.isfinite(unrestricted).all())
        with self.assertRaises(EmissionSupportError):
            legal_gcs_triple_log_prob(
                torch.zeros((1, 35)),
                torch.tensor([[1, 2, 3]]),
                maximum=5,
                observation_count=torch.full((1,), 2),
            )

    def test_dense_abnormal_autoregressive_table_is_normalized(self) -> None:
        raw = torch.linspace(-1.0, 1.0, 30, dtype=torch.float64).view(1, 30)
        pairs = torch.cartesian_prod(torch.arange(1, 3), torch.arange(1, 3))
        probability = dense_abnormal_duration_log_prob(
            raw.expand(pairs.shape[0], -1),
            pairs,
            torch.full((pairs.shape[0],), 2),
            field="heart_rate",
            condition_keys=("HR_LT40", "HR_GT120"),
            minimum=torch.full((pairs.shape[0],), 30.0),
            maximum=torch.full((pairs.shape[0],), 130.0),
        ).exp()
        self.assertAlmostEqual(float(probability.sum()), 1.0, places=12)

        single = torch.arange(1, 3).view(-1, 1)
        single_probability = dense_abnormal_duration_log_prob(
            raw.expand(2, -1),
            single,
            torch.full((2,), 2),
            field="temperature",
            condition_keys=("TEMP_GE38",),
            minimum=torch.full((2,), 37.0),
            maximum=torch.full((2,), 39.0),
        ).exp()
        self.assertAlmostEqual(float(single_probability.sum()), 1.0, places=12)

        table = torch.zeros((2, 30), dtype=torch.float64)
        table[:, 5:30] = torch.arange(25, dtype=torch.float64).square().view(1, 25)
        conditional = dense_abnormal_duration_log_prob(
            table,
            torch.tensor([[1, 1], [2, 1]]),
            torch.full((2,), 4),
            field="heart_rate",
            condition_keys=("HR_LT40", "HR_GT120"),
            minimum=torch.full((2,), 30.0),
            maximum=torch.full((2,), 130.0),
        )
        self.assertNotEqual(float(conditional[0]), float(conditional[1]))

    def test_lower_triangular_conditioning_is_strictly_causal(self) -> None:
        cases = (
            (4, 3, "log1p_count", 30, 2.0),
            (6, 5, "duration_fraction", 105, 2.0),
            (6, 1, "binary", 21, 1.0),
        )
        for components, parameter_width, transform, width, changed_value in cases:
            raw = torch.zeros((1, width), dtype=torch.float64)
            base_width = components * parameter_width
            raw[..., base_width : base_width + parameter_width] = 1.0
            before = torch.zeros((1, components), dtype=torch.float64)
            after = before.clone()
            after[..., 0] = changed_value
            decoded_before = lower_triangular_conditioned_parameters(
                raw,
                before,
                component_count=components,
                parameter_width=parameter_width,
                transform=transform,
            )
            decoded_after = lower_triangular_conditioned_parameters(
                raw,
                after,
                component_count=components,
                parameter_width=parameter_width,
                transform=transform,
            )
            torch.testing.assert_close(decoded_before[..., 0, :], decoded_after[..., 0, :])
            self.assertFalse(torch.allclose(decoded_before[..., 1, :], decoded_after[..., 1, :]))

    def test_autoregressive_vector_families_have_finite_gradients(self) -> None:
        count_raw = _leaf(2, 30)
        count = torch.tensor([[0, 1, 2, 0], [1, 0, 3, 2]], dtype=torch.float64)
        count_lp = autoregressive_hurdle_count_vector_log_prob(count_raw, count)

        binary_raw = _leaf(2, 21)
        binary = torch.tensor([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]])
        binary_lp = autoregressive_binary_vector_log_prob(binary_raw, binary)

        duration_raw = _leaf(2, 105)
        duration = torch.tensor(
            [[0.0, 1.0, 4.0, 2.0, 0.5, 3.0], [4.0, 0.0, 1.0, 2.0, 3.0, 0.5]],
            dtype=torch.float64,
        )
        duration_lp = vasopressor_duration_log_prob(duration, duration_raw)
        loss = -(count_lp + binary_lp + duration_lp).mean()
        loss.backward()
        for raw in (count_raw, binary_raw, duration_raw):
            self.assertTrue(torch.isfinite(raw.grad).all())

    def test_dense_score_is_invariant_to_registered_raw_bounds(self) -> None:
        parameters = DenseValueParameters(
            _leaf(1, 2), _zoiln(1), _zoiln(1), _zoiln(1), _zoiln(1), _zoiln(1)
        )

        def target(lower: float, upper: float) -> DenseValueTarget:
            alpha, beta, u, v = 0.2, 0.5, 0.25, 0.75
            minimum = lower + (upper - lower) * alpha
            value_range = (upper - minimum) * beta
            last = minimum + value_range * u
            maximum = minimum + value_range
            mean = minimum + value_range * (u + v) / 2.0
            return DenseValueTarget(
                torch.tensor([2]),
                torch.tensor([minimum], dtype=torch.float64),
                torch.tensor([last], dtype=torch.float64),
                torch.tensor([maximum], dtype=torch.float64),
                torch.tensor([mean], dtype=torch.float64),
            )

        first = dense_joint_value_log_prob(parameters, target(0.0, 10.0), lower=0.0, upper=10.0)
        second = dense_joint_value_log_prob(
            parameters, target(100.0, 120.0), lower=100.0, upper=120.0
        )
        torch.testing.assert_close(first, second)

    def test_lab_mixed_coordinate_branches_are_normalized(self) -> None:
        points = 30000
        grid = -40.0 + (torch.arange(points, dtype=torch.float64) + 0.5) * 80.0 / points
        parameters = LabValueParameters(
            StudentTParameters(
                torch.zeros(points, dtype=torch.float64),
                torch.zeros(points, dtype=torch.float64),
                torch.zeros(points, dtype=torch.float64),
            ),
            torch.zeros((points, 2), dtype=torch.float64),
            StudentTParameters(
                torch.zeros(points, dtype=torch.float64),
                torch.zeros(points, dtype=torch.float64),
                torch.zeros(points, dtype=torch.float64),
            ),
            StudentTParameters(
                torch.zeros(points, dtype=torch.float64),
                torch.zeros(points, dtype=torch.float64),
                torch.zeros(points, dtype=torch.float64),
            ),
            torch.zeros(points, dtype=torch.float64),
            torch.zeros(points, dtype=torch.float64),
            ZOILogitNormalParameters(
                torch.zeros((points, 3), dtype=torch.float64),
                torch.zeros(points, dtype=torch.float64),
                torch.zeros(points, dtype=torch.float64),
            ),
        )
        single = lab_joint_value_log_prob(
            parameters,
            LabValueTarget(torch.ones(points), grid, grid, grid),
        )
        single_integral = single.exp().sum() * 80.0 / points
        self.assertLess(abs(float(single_integral) - 1.0), 2e-4)

        repeated_zero = lab_joint_value_log_prob(
            parameters,
            LabValueTarget(torch.full((points,), 2), grid, grid, grid),
        )
        repeated_integral = repeated_zero.exp().sum() * 80.0 / points
        self.assertLess(abs(float(repeated_integral) - 0.5), 1e-4)

        unit_points = 20000
        q_width = 16.0
        q = -8.0 + (torch.arange(unit_points, dtype=torch.float64) + 0.5) * q_width / unit_points
        u = torch.sigmoid(q)
        u_all = torch.cat((torch.tensor([0.0, 1.0], dtype=torch.float64), u))
        count = torch.full((u_all.numel(),), 3)
        p = LabValueParameters(
            StudentTParameters(
                torch.zeros_like(u_all), torch.zeros_like(u_all), torch.zeros_like(u_all)
            ),
            torch.zeros((u_all.numel(), 2), dtype=torch.float64),
            StudentTParameters(
                torch.zeros_like(u_all), torch.zeros_like(u_all), torch.zeros_like(u_all)
            ),
            StudentTParameters(
                torch.zeros_like(u_all), torch.zeros_like(u_all), torch.zeros_like(u_all)
            ),
            torch.zeros_like(u_all),
            torch.zeros_like(u_all),
            ZOILogitNormalParameters(
                torch.zeros((u_all.numel(), 3), dtype=torch.float64),
                torch.zeros_like(u_all),
                torch.zeros_like(u_all),
            ),
        )
        positive = lab_joint_value_log_prob(
            p,
            LabValueTarget(count, torch.zeros_like(u_all), u_all, torch.ones_like(u_all)),
        ).exp()
        integrated_u = positive[:2].sum() + positive[2:].sum() * q_width / unit_points
        student_at_zero = (
            torch.distributions.StudentT(
                2.0 + torch.nn.functional.softplus(torch.tensor(0.0)) + 1e-4,
                0.0,
                torch.nn.functional.softplus(torch.tensor(0.0)) + 1e-4,
            )
            .log_prob(torch.tensor(0.0))
            .exp()
        )
        normal_at_zero = (
            torch.distributions.Normal(0.0, torch.nn.functional.softplus(torch.tensor(0.0)) + 1e-4)
            .log_prob(torch.tensor(0.0))
            .exp()
        )
        expected = 0.5 * student_at_zero * normal_at_zero
        self.assertLess(abs(float(integrated_u) - float(expected)), 2e-5)

        endpoint_only = lab_joint_value_log_prob(
            p,
            LabValueTarget(
                torch.full((u_all.numel(),), 2),
                torch.zeros_like(u_all),
                torch.where(
                    torch.arange(u_all.numel()) % 2 == 0,
                    torch.zeros_like(u_all),
                    torch.ones_like(u_all),
                ),
                torch.ones_like(u_all),
            ),
        ).exp()
        # Count two renormalizes the LAST coordinate over exactly its two endpoints.
        torch.testing.assert_close(
            endpoint_only[0],
            endpoint_only[1],
        )

    def test_ned_and_uop_canonical_log_coordinates_are_normalized(self) -> None:
        points = 30000
        width = 24.0
        q = -12.0 + (torch.arange(points, dtype=torch.float64) + 0.5) * width / points
        maximum = torch.exp(q)
        ned_parameters = NEDParameters(
            torch.zeros((points, 2), dtype=torch.float64),
            torch.zeros(points, dtype=torch.float64),
            torch.full(
                (points,),
                math.log(math.expm1(1.0 - 1e-4)),
                dtype=torch.float64,
            ),
            ZOILogitNormalParameters(
                torch.zeros((points, 3), dtype=torch.float64),
                torch.zeros(points, dtype=torch.float64),
                torch.zeros(points, dtype=torch.float64),
            ),
            ZOILogitNormalParameters(
                torch.zeros((points, 3), dtype=torch.float64),
                torch.zeros(points, dtype=torch.float64),
                torch.zeros(points, dtype=torch.float64),
            ),
        )
        ned = ned_joint_value_log_prob(
            ned_parameters,
            NEDTarget(
                maximum,
                torch.zeros_like(maximum),
                maximum,
                torch.ones(points, dtype=torch.bool),
                torch.ones(points, dtype=torch.bool),
            ),
        )
        # Positive branch 1/2, LAST zero atom 1/3, MEAN upper atom 1/2
        # after the forbidden zero branch is renormalized away.
        self.assertLess(abs(float(ned.exp().sum() * width / points) - 1.0 / 12.0), 2e-5)

        uop_parameters = UOPParameters(
            torch.zeros((points, 2), dtype=torch.float64),
            torch.zeros(points, dtype=torch.float64),
            torch.full(
                (points,),
                math.log(math.expm1(1.0 - 1e-4)),
                dtype=torch.float64,
            ),
        )
        positive = (
            uop_sum_log_prob(
                uop_parameters,
                torch.exp(q),
                torch.ones(points),
            )
            .exp()
            .sum()
            * width
            / points
        )
        zero = (
            uop_sum_log_prob(
                UOPParameters(
                    torch.zeros((1, 2), dtype=torch.float64),
                    torch.zeros(1, dtype=torch.float64),
                    torch.zeros(1, dtype=torch.float64),
                ),
                torch.zeros(1, dtype=torch.float64),
                torch.ones(1),
            )
            .exp()
            .item()
        )
        self.assertLess(abs(float(positive + zero) - 1.0), 2e-5)

    def test_gcs_verbal_latest_status_obeys_count_support(self) -> None:
        logits = torch.zeros((3, 2), dtype=torch.float64, requires_grad=True)
        observed = torch.tensor([2, 2, 2])
        ungradable = torch.tensor([0, 2, 1])
        status = torch.tensor([0, 1, 1])
        log_prob = gcs_verbal_latest_status_log_prob(
            logits,
            status,
            observed,
            ungradable,
        )
        self.assertTrue(torch.isfinite(log_prob).all())
        self.assertEqual(float(log_prob[0].detach()), 0.0)
        self.assertEqual(float(log_prob[1].detach()), 0.0)
        with self.assertRaises(EmissionSupportError):
            gcs_verbal_latest_status_log_prob(
                logits[:1],
                torch.tensor([1]),
                observed[:1],
                ungradable[:1],
            )

    def test_dense_lab_ned_and_uop_joint_branches_are_finite_and_differentiable(self) -> None:
        dense_parameters = DenseValueParameters(
            _leaf(2, 2),
            _zoiln(2),
            _zoiln(2),
            _zoiln(2),
            _zoiln(2),
            _zoiln(2),
        )
        dense = dense_joint_value_log_prob(
            dense_parameters,
            DenseValueTarget(
                observed_hours=torch.tensor([2, 2]),
                minimum=torch.tensor([3.0, 2.0], dtype=torch.float64),
                last=torch.tensor([3.0, 5.0], dtype=torch.float64),
                maximum=torch.tensor([3.0, 8.0], dtype=torch.float64),
                mean=torch.tensor([3.0, 5.0], dtype=torch.float64),
            ),
            lower=0.0,
            upper=10.0,
        )

        lab_parameters = LabValueParameters(
            _student(3),
            _leaf(3, 2),
            _student(3),
            _student(3),
            _leaf(3),
            _leaf(3),
            _zoiln(3),
        )
        lab = lab_joint_value_log_prob(
            lab_parameters,
            LabValueTarget(
                observation_count=torch.tensor([1, 2, 3]),
                minimum=torch.tensor([2.0, 4.0, 1.0], dtype=torch.float64),
                last=torch.tensor([2.0, 4.0, 3.0], dtype=torch.float64),
                maximum=torch.tensor([2.0, 4.0, 5.0], dtype=torch.float64),
            ),
        )

        ned_parameters = NEDParameters(
            _leaf(2, 2),
            _leaf(2),
            _leaf(2),
            _zoiln(2),
            _zoiln(2),
        )
        ned = ned_joint_value_log_prob(
            ned_parameters,
            NEDTarget(
                maximum=torch.tensor([0.0, 0.2], dtype=torch.float64),
                last=torch.tensor([0.0, 0.1], dtype=torch.float64),
                mean=torch.tensor([0.0, 0.05], dtype=torch.float64),
                compatible_vasopressor_duration=torch.tensor([False, True]),
                compatible_vasopressor_edge=torch.tensor([False, True]),
            ),
        )

        uop_parameters = UOPParameters(_leaf(2, 2), _leaf(2), _leaf(2))
        uop = uop_sum_log_prob(
            uop_parameters,
            torch.tensor([0.0, 120.0], dtype=torch.float64),
            torch.tensor([1, 2]),
        )
        joint = dense.sum() + lab.sum() + ned.sum() + uop.sum()
        self.assertTrue(torch.isfinite(joint))
        (-joint).backward()
        leaves = [
            value
            for value in (
                dense_parameters.range_logits,
                dense_parameters.constant_value.mixture_logits,
                dense_parameters.minimum_coordinate.interior_loc,
                lab_parameters.range_logits,
                lab_parameters.single_value.location,
                lab_parameters.constant_value.scale_raw,
                lab_parameters.minimum.df_raw,
                lab_parameters.log_range_loc,
                ned_parameters.zero_positive_logits,
                ned_parameters.positive_max_loc,
                uop_parameters.zero_positive_logits,
                uop_parameters.positive_loc,
            )
        ]
        self.assertTrue(all(value.grad is not None for value in leaves))
        self.assertTrue(all(torch.isfinite(value.grad).all() for value in leaves))

    def test_joint_support_violations_raise_instead_of_clipping(self) -> None:
        parameters = DenseValueParameters(
            _leaf(1, 2),
            _zoiln(1),
            _zoiln(1),
            _zoiln(1),
            _zoiln(1),
            _zoiln(1),
        )
        with self.assertRaises(EmissionSupportError):
            dense_joint_value_log_prob(
                parameters,
                DenseValueTarget(
                    torch.tensor([2]),
                    torch.tensor([4.0]),
                    torch.tensor([3.0]),
                    torch.tensor([5.0]),
                    torch.tensor([4.0]),
                ),
                lower=0.0,
                upper=10.0,
            )
        ned_parameters = NEDParameters(
            _leaf(1, 2), _leaf(1), _leaf(1), _zoiln(1), _zoiln(1)
        )
        with self.assertRaises(EmissionSupportError):
            ned_joint_value_log_prob(
                ned_parameters,
                NEDTarget(
                    torch.tensor([0.2]),
                    torch.tensor([0.1]),
                    torch.tensor([0.1]),
                    torch.tensor([False]),
                    torch.tensor([False]),
                ),
            )

    def test_respiratory_active_set_simplex_and_evidence_support(self) -> None:
        parameters = RespiratoryOccupancyParameters(
            active_set_logits=_leaf(2, 31),
            alr_location=_leaf(2, 4),
            alr_scale_raw=_leaf(2, 4, value=math.log(math.expm1(1.0 - 1e-4))),
        )
        duration = torch.tensor(
            [[4.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 2.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        log_prob = respiratory_occupancy_log_prob(parameters, duration)
        self.assertTrue(torch.isfinite(log_prob).all())
        (-log_prob.sum()).backward()
        self.assertTrue(torch.isfinite(parameters.active_set_logits.grad).all())
        self.assertTrue(torch.isfinite(parameters.alr_location.grad).all())
        self.assertTrue(torch.isfinite(parameters.alr_scale_raw.grad).all())

        grid_count = 10000
        width = 24.0
        z = -12.0 + (torch.arange(grid_count, dtype=torch.float64) + 0.5) * width / grid_count
        proportion = torch.sigmoid(z)
        two_component = torch.stack(
            (
                4.0 * proportion,
                4.0 * (1.0 - proportion),
                torch.zeros_like(z),
                torch.zeros_like(z),
                torch.zeros_like(z),
            ),
            dim=-1,
        )
        uniform_parameters = RespiratoryOccupancyParameters(
            active_set_logits=torch.zeros((grid_count, 31), dtype=torch.float64),
            alr_location=torch.zeros((grid_count, 4), dtype=torch.float64),
            alr_scale_raw=torch.full(
                (grid_count, 4),
                math.log(math.expm1(1.0 - 1e-4)),
                dtype=torch.float64,
            ),
        )
        integrated = (
            respiratory_occupancy_log_prob(
                uniform_parameters,
                two_component,
            )
            .exp()
            .sum()
            * width
            / grid_count
        )
        self.assertAlmostEqual(float(integrated), 1.0 / 31.0, places=5)

        with self.assertRaisesRegex(EmissionSupportError, "must equal the block span"):
            respiratory_occupancy_log_prob(
                RespiratoryOccupancyParameters(
                    active_set_logits=torch.zeros((1, 31), dtype=torch.float64),
                    alr_location=torch.zeros((1, 4), dtype=torch.float64),
                    alr_scale_raw=torch.zeros((1, 4), dtype=torch.float64),
                ),
                torch.tensor(
                    [[4.0 + 2e-12, 0.0, 0.0, 0.0, 0.0]],
                    dtype=torch.float64,
                ),
            )

        with self.assertRaises(EmissionSupportError):
            respiratory_edge_evidence_log_prob(
                torch.zeros(1),
                torch.ones(1),
                torch.zeros(1),
            )

    def test_vasopressor_logit_normal_handles_zero_full_and_interior_durations(self) -> None:
        duration = torch.tensor(
            [[0.0, 4.0, 2.0, 0.0, 1.0, 4.0]],
            dtype=torch.float64,
        )
        parameters = _leaf(1, 105)
        log_prob = vasopressor_duration_log_prob(duration, parameters)
        self.assertTrue(torch.isfinite(log_prob).all())
        (-log_prob.mean()).backward()
        self.assertTrue(torch.isfinite(parameters.grad).all())

    def test_logit_normal_near_endpoints_remain_interior_and_fp16_sampling_promotes(self) -> None:
        near = torch.tensor([1e-8, 1.0 - 1e-8], dtype=torch.float64)
        q = torch.logit(near)
        parameters = ZOILogitNormalParameters(
            torch.tensor([[-20.0, 20.0, -20.0]] * 2, dtype=torch.float64),
            q.clone().requires_grad_(),
            torch.full((2,), -8.0, dtype=torch.float64, requires_grad=True),
        )
        log_prob = zoi_logit_normal_log_prob(near, parameters)
        self.assertTrue(torch.isfinite(log_prob).all())
        (-log_prob.sum()).backward()
        self.assertTrue(torch.isfinite(parameters.interior_loc.grad).all())
        self.assertTrue(torch.isfinite(parameters.interior_scale_raw.grad).all())

        torch.manual_seed(41)
        fp16 = ZOILogitNormalParameters(
            torch.tensor([[-100.0, 100.0, -100.0]] * 2, dtype=torch.float16),
            torch.tensor([-80.0, 15.0], dtype=torch.float16),
            torch.full((2,), -20.0, dtype=torch.float16),
        )
        sampled = sample_zoi_logit_normal(fp16)
        self.assertEqual(sampled.dtype, torch.float64)
        self.assertTrue(sampled.gt(0.0).all() and sampled.lt(1.0).all())

        saturated = ZOILogitNormalParameters(
            torch.tensor([[-100.0, 100.0, -100.0]], dtype=torch.float16),
            torch.tensor([100.0], dtype=torch.float16),
            torch.tensor([-20.0], dtype=torch.float16),
        )
        with self.assertRaisesRegex(EmissionSupportError, "saturated"):
            sample_zoi_logit_normal(saturated)

    def test_dense_v_repairs_only_out_of_range_double_arithmetic_residue(self) -> None:
        near_v = 6.6e-10
        mean_parameters = ZOILogitNormalParameters(
            torch.tensor([[-100.0, 100.0, -100.0]], dtype=torch.float64),
            torch.tensor([torch.logit(torch.tensor(near_v, dtype=torch.float64))]),
            torch.tensor([-8.0], dtype=torch.float64),
        )
        parameters = DenseValueParameters(
            range_logits=torch.zeros((1, 2), dtype=torch.float64),
            constant_value=_zoiln(1),
            minimum_coordinate=_zoiln(1),
            range_coordinate=_zoiln(1),
            last_coordinate=_zoiln(1),
            mean_coordinate=mean_parameters,
        )
        lower_mean = 0.25
        upper_mean = 1.75
        mean_span = upper_mean - lower_mean

        def score(mean: float) -> torch.Tensor:
            return dense_joint_value_log_prob(
                parameters,
                DenseValueTarget(
                    observed_hours=torch.tensor([4]),
                    minimum=torch.tensor([0.0], dtype=torch.float64),
                    last=torch.tensor([1.0], dtype=torch.float64),
                    maximum=torch.tensor([2.0], dtype=torch.float64),
                    mean=torch.tensor([mean], dtype=torch.float64),
                ),
                lower=-1.0,
                upper=3.0,
            )

        exact_endpoint = score(lower_mean)
        legal_near_interior = score(lower_mean + mean_span * near_v)
        self.assertGreater(
            float((legal_near_interior - exact_endpoint).detach()),
            100.0,
        )
        repaired = score(lower_mean - mean_span * 5e-13)
        self.assertTrue(torch.isfinite(repaired).all())
        with self.assertRaisesRegex(EmissionSupportError, "more than 1e-12"):
            score(lower_mean - mean_span * 2e-12)

    def test_dense_sampler_reconstructs_physical_state_in_float64(self) -> None:
        batch = 2048
        raw = torch.zeros((batch, 27), dtype=torch.float32)
        raw[:, 0] = -30.0
        raw[:, 1] = 30.0
        for offset in (2, 7, 12, 17, 22):
            raw[:, offset : offset + 3] = torch.tensor([-30.0, 30.0, -30.0])
            raw[:, offset + 3] = torch.linspace(-6.0, 6.0, batch)
            raw[:, offset + 4] = -2.0
        hours = torch.arange(batch).remainder(3).add(2)
        torch.manual_seed(47)
        state = _sample_dense_value_state(raw, hours, lower=-20.0, upper=200.0)
        self.assertEqual(state.dtype, torch.float64)
        last, minimum, maximum, mean = state.unbind(dim=-1)
        hour_float = hours.double()
        lower_mean = (last + (hour_float - 1.0) * minimum) / hour_float
        upper_mean = (last + (hour_float - 1.0) * maximum) / hour_float
        self.assertTrue((minimum < maximum).all())
        self.assertTrue((minimum <= last).all() and (last <= maximum).all())
        self.assertTrue((lower_mean <= mean).all() and (mean <= upper_mean).all())

    def test_dense_abnormal_support_is_exhaustive_normalized_and_sampled(self) -> None:
        cases = (
            ("heart_rate", ("HR_LT40", "HR_GT120"), 50.0, 100.0),
            ("heart_rate", ("HR_LT40", "HR_GT120"), 30.0, 100.0),
            ("heart_rate", ("HR_LT40", "HR_GT120"), 50.0, 130.0),
            ("heart_rate", ("HR_LT40", "HR_GT120"), 30.0, 130.0),
            ("systolic_bp", ("SBP_LT90",), 100.0, 120.0),
            ("systolic_bp", ("SBP_LT90",), 80.0, 120.0),
            ("mean_arterial_pressure", ("MAP_LT65", "MAP_GE65_LT70"), 60.0, 90.0),
            ("mean_arterial_pressure", ("MAP_LT65", "MAP_GE65_LT70"), 67.0, 90.0),
            ("mean_arterial_pressure", ("MAP_LT65", "MAP_GE65_LT70"), 75.0, 90.0),
            ("respiratory_rate", ("RR_LT8", "RR_GE22"), 6.0, 25.0),
            ("temperature", ("TEMP_GE38",), 36.0, 39.0),
            ("spo2", ("SPO2_LE93",), 90.0, 99.0),
        )
        base = torch.linspace(-1.0, 1.0, 30, dtype=torch.float64)
        for field, conditions, minimum, maximum in cases:
            for hours in range(1, 5):
                candidates = torch.tensor(
                    list(itertools.product(range(hours + 1), repeat=len(conditions))),
                    dtype=torch.long,
                )
                n = candidates.shape[0]
                observed = torch.full((n,), hours)
                minima = torch.full((n,), minimum)
                maxima = torch.full((n,), maximum)
                first_mask, second_mask = dense_abnormal_class_masks(
                    field=field,
                    condition_keys=conditions,
                    observed_hours=observed,
                    minimum=minima,
                    maximum=maxima,
                    first_duration=candidates[:, 0],
                )
                legal = first_mask.gather(-1, candidates[:, :1]).squeeze(-1)
                if second_mask is not None:
                    legal &= second_mask.gather(-1, candidates[:, 1:2]).squeeze(-1)
                legal_candidates = candidates[legal]
                probability = dense_abnormal_duration_log_prob(
                    base.expand(legal_candidates.shape[0], -1),
                    legal_candidates,
                    torch.full((legal_candidates.shape[0],), hours),
                    field=field,
                    condition_keys=conditions,
                    minimum=torch.full((legal_candidates.shape[0],), minimum),
                    maximum=torch.full((legal_candidates.shape[0],), maximum),
                ).exp()
                self.assertAlmostEqual(float(probability.sum()), 1.0, places=10)

                draws = 64
                sampled = sample_dense_abnormal_duration(
                    base.expand(draws, -1),
                    torch.full((draws,), hours),
                    field=field,
                    condition_keys=conditions,
                    minimum=torch.full((draws,), minimum),
                    maximum=torch.full((draws,), maximum),
                )
                sampled_first, sampled_second = dense_abnormal_class_masks(
                    field=field,
                    condition_keys=conditions,
                    observed_hours=torch.full((draws,), hours),
                    minimum=torch.full((draws,), minimum),
                    maximum=torch.full((draws,), maximum),
                    first_duration=sampled[:, 0],
                )
                self.assertTrue(
                    sampled_first.gather(-1, sampled[:, :1]).all(),
                    (field, hours, minimum, maximum),
                )
                if sampled_second is not None:
                    self.assertTrue(sampled_second.gather(-1, sampled[:, 1:2]).all())

        with self.assertRaises(EmissionSupportError):
            dense_abnormal_duration_log_prob(
                torch.zeros((1, 30)),
                torch.tensor([[4, 1]]),
                torch.tensor([4]),
                field="mean_arterial_pressure",
                condition_keys=("MAP_LT65", "MAP_GE65_LT70"),
                minimum=torch.tensor([52.0]),
                maximum=torch.tensor([70.0]),
            )

    def test_gcs_and_lab_count_conditioned_support_masks_are_exact(self) -> None:
        for maximum in (4, 6):
            states = legal_ordinal_triples(maximum)
            for hours in range(1, 5):
                probability = legal_gcs_triple_log_prob(
                    torch.zeros((states.shape[0], states.shape[0]), dtype=torch.float64),
                    states,
                    maximum=maximum,
                    observation_count=torch.full((states.shape[0],), hours),
                    source_semantics="raw_point",
                ).exp()
                self.assertAlmostEqual(float(probability.sum()), 1.0, places=12)

        verbal_states = legal_ordinal_triples(5)
        for hours in range(1, 5):
            count = torch.tensor([hours])
            mask = ordinal_triple_class_mask(
                verbal_states,
                count,
                source_semantics="hourly_sequence",
            )[0]
            legal_states = verbal_states[mask]
            probability = legal_gcs_triple_log_prob(
                torch.zeros((legal_states.shape[0], 35), dtype=torch.float64),
                legal_states,
                maximum=5,
                observation_count=torch.full((legal_states.shape[0],), hours),
            ).exp()
            self.assertAlmostEqual(float(probability.sum()), 1.0, places=12)
            sampled_index = sample_categorical(
                torch.where(mask, torch.zeros_like(mask, dtype=torch.float32), 100.0),
                mask,
            )
            self.assertTrue(mask[sampled_index])
            illegal_states = verbal_states[~mask]
            if illegal_states.numel():
                with self.assertRaises(EmissionSupportError):
                    legal_gcs_triple_log_prob(
                        torch.zeros((1, 35)),
                        illegal_states[:1],
                        maximum=5,
                        observation_count=torch.tensor([hours]),
                    )

        lab_parameters = LabValueParameters(
            _student(1),
            _leaf(1, 2),
            _student(1),
            _student(1),
            _leaf(1),
            _leaf(1),
            _zoiln(1),
        )
        with self.assertRaises(EmissionSupportError):
            lab_joint_value_log_prob(
                lab_parameters,
                LabValueTarget(
                    torch.tensor([2]),
                    torch.tensor([-7.0]),
                    torch.tensor([-6.5]),
                    torch.tensor([-6.0]),
                ),
            )
        # Full-r5 truth witness: N=2, MIN=-7, LAST=MAX=-6.
        self.assertTrue(
            torch.isfinite(
                lab_joint_value_log_prob(
                    lab_parameters,
                    LabValueTarget(
                        torch.tensor([2]),
                        torch.tensor([-7.0]),
                        torch.tensor([-6.0]),
                        torch.tensor([-6.0]),
                    ),
                )
            ).all()
        )
        raw = torch.zeros((256, 18))
        raw[:, 3:5] = torch.tensor([-100.0, 100.0])
        raw[:, 13:16] = torch.tensor([-100.0, 100.0, -100.0])
        sampled_lab = _sample_lab_value_state(
            raw,
            torch.full((256,), 2),
            center=0.0,
            scale=1.0,
        )
        self.assertTrue(
            (
                sampled_lab[:, 0].eq(sampled_lab[:, 1])
                | sampled_lab[:, 0].eq(sampled_lab[:, 2])
            ).all()
        )

    def test_boundary_path_support_is_renormalized_and_sampled(self) -> None:
        count_vectors = torch.tensor(
            [row for row in itertools.product(range(6), repeat=4) if any(row)],
            dtype=torch.float64,
        )
        raw = torch.zeros((count_vectors.shape[0], 30), dtype=torch.float64)
        raw[:, 2:12:3] = -5.0
        probability = autoregressive_hurdle_count_vector_log_prob(
            raw,
            count_vectors,
            require_any_positive=torch.ones(count_vectors.shape[0], dtype=torch.bool),
        ).exp()
        self.assertGreater(float(probability.sum()), 0.999999)

        sampled = sample_autoregressive_hurdle_count_vector(
            torch.zeros((1024, 30)),
            component_count=4,
            require_any_positive=torch.ones(1024, dtype=torch.bool),
        )
        self.assertTrue(sampled.gt(0).any(dim=-1).all())
        self.assertTrue((sampled[:, :-1].gt(0).any(dim=-1) & sampled[:, -1].eq(0)).any())

        respiratory_raw = _leaf(1, 30)
        zero_duration = torch.zeros((1, 4), dtype=torch.float64)
        with self.assertRaises(EmissionSupportError):
            respiratory_onset_log_prob(
                torch.zeros((1, 4)),
                respiratory_raw,
                torch.ones(1),
                zero_duration,
                torch.zeros(1),
                torch.zeros(1),
            )
        respiratory_lp = respiratory_onset_log_prob(
            torch.tensor([[0, 1, 0, 0]]),
            respiratory_raw,
            torch.ones(1),
            zero_duration,
            torch.ones(1),
            torch.ones(1),
        )

        vaso_raw = _leaf(1, 63)
        vaso_duration = torch.zeros((1, 6), dtype=torch.float64)
        vaso_edge = torch.tensor([[1, 0, 0, 0, 0, 0]])
        with self.assertRaises(EmissionSupportError):
            vasopressor_onset_log_prob(
                torch.zeros((1, 6)),
                vaso_raw,
                vaso_duration,
                vaso_edge,
            )
        vaso_lp = vasopressor_onset_log_prob(
            torch.tensor([[1, 0, 0, 0, 0, 0]]),
            vaso_raw,
            vaso_duration,
            vaso_edge,
        )
        loss = -(respiratory_lp + vaso_lp).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(respiratory_raw.grad).all())
        self.assertTrue(torch.isfinite(vaso_raw.grad).all())

    def test_ned_joint_duration_edge_and_mean_support_sampling(self) -> None:
        parameters = NEDParameters(
            _leaf(1, 2),
            _leaf(1),
            _leaf(1),
            _zoiln(1),
            _zoiln(1),
        )
        invalid = (
            NEDTarget(
                torch.tensor([0.2]),
                torch.tensor([0.0]),
                torch.tensor([0.1]),
                torch.tensor([False]),
                torch.tensor([False]),
            ),
            NEDTarget(
                torch.tensor([0.2]),
                torch.tensor([0.1]),
                torch.tensor([0.1]),
                torch.tensor([True]),
                torch.tensor([False]),
            ),
            NEDTarget(
                torch.tensor([0.2]),
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                torch.tensor([True]),
                torch.tensor([False]),
            ),
        )
        for target in invalid:
            with self.assertRaises(EmissionSupportError):
                ned_joint_value_log_prob(parameters, target)

        valid = NEDTarget(
            torch.tensor([0.2]),
            torch.tensor([0.0]),
            torch.tensor([0.1]),
            torch.tensor([True]),
            torch.tensor([False]),
        )
        log_prob = ned_joint_value_log_prob(parameters, valid)
        self.assertTrue(torch.isfinite(log_prob).all())
        (-log_prob.sum()).backward()
        self.assertTrue(torch.isfinite(parameters.zero_positive_logits.grad).all())

        raw = torch.zeros((256, 14))
        raw[:, :2] = torch.tensor([-100.0, 100.0])
        raw[:, 4:7] = torch.tensor([-100.0, 100.0, -100.0])
        raw[:, 9:12] = torch.tensor([100.0, -100.0, -100.0])
        sampled = _sample_ned_state(
            raw,
            torch.ones(256, dtype=torch.bool),
            torch.zeros(256, dtype=torch.bool),
        )
        self.assertTrue(sampled[:, 0].eq(0.0).all())
        self.assertTrue(sampled[:, 1].gt(0.0).all())
        self.assertTrue(sampled[:, 2].gt(0.0).all())


if __name__ == "__main__":
    unittest.main()
