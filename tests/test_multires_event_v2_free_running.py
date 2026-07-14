from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from tests.test_multires_event_v2_loss import (
    DENSE,
    FIELDS,
    LABS,
    SCALE_HASH,
    _metadata,
    _registry,
)
from trauma_predict.eval.multires_event_v2 import _teacher_nll_decomposition_rows
from trauma_predict.eval.multires_event_v2_free_running import (
    MODEL_INPUT_KEYS,
    _emit_rank_progress,
    common_random_seed,
    evaluate_free_running_v2,
    evaluate_multires_event_v2_promotion,
    validate_rank_local_artifact_preflight,
)
from trauma_predict.eval.multires_event_v2_projections import (
    PhysicalProjectionSpec,
    PrimitiveVectorCoordinate,
    build_standardized_primitive_schema,
    empirical_crps,
    empirical_energy_score,
    load_standardized_primitive_scale_artifact,
    required_standardized_scale_keys,
    standardize_primitive_trajectory,
)
from trauma_predict.eval.multires_event_v2_scale import (
    fit_standardized_primitive_scale_artifact,
)
from trauma_predict.modeling.multires_event_v2.field_state import (
    PrimitiveParameterHeads,
)
from trauma_predict.modeling.multires_event_v2.rollout import (
    AutoregressiveFieldStateRollout,
)
from trauma_predict.modeling.multires_event_v2.trajectory import (
    FieldStateTrajectoryDecoder,
)
from trauma_predict.training.multires_event_v2_loss import (
    REGISTERED_CORE_FIELD_IDS,
    RegistryPrimitiveSampler,
    V2_PRIMITIVE_FEEDBACK_DIMS,
    V2_PRIMITIVE_HEAD_DIMS,
    expand_enabled_core_primitives,
)
from trauma_predict.training.observability import append_jsonl, sha256_file


RESPIRATORY_MODALITIES = (
    "RESP_INVASIVE",
    "RESP_NONINVASIVE",
    "RESP_HIGH_FLOW",
    "RESP_OTHER_OXYGEN",
)
VASOPRESSOR_AGENTS = (
    "VASO_NOREPINEPHRINE",
    "VASO_EPINEPHRINE",
    "VASO_PHENYLEPHRINE",
    "VASO_VASOPRESSIN",
    "VASO_DOPAMINE",
    "VASO_OTHER",
)
PROMOTION_CONTRACT = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "configs/evaluation/multires_event_v2_promotion_v2.json"
    ).read_text(encoding="utf-8")
)


class MultiresEventV2RankArtifactTest(unittest.TestCase):
    def test_rank_one_progress_is_written_but_shared_metrics_remain_rank_zero_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"RANK": "1"},
        ):
            root = Path(directory)
            progress_path = root / "progress.rank00001.jsonl"
            _emit_rank_progress(
                path=progress_path,
                rank=1,
                mode="block",
                completed_anchors=0,
                started_at=time.monotonic(),
            )
            rows = [
                json.loads(line)
                for line in progress_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["rank"], 1)
            self.assertEqual(rows[0]["completed_anchors"], 0)
            self.assertRegex(sha256_file(progress_path), r"^[0-9a-f]{64}$")

            shared_path = root / "metrics.jsonl"
            append_jsonl(shared_path, {"event": "must_not_be_written"})
            self.assertFalse(shared_path.exists())

    def test_two_rank_gloo_preflight_writes_hashes_gathers_and_assembles(self) -> None:
        worker = (
            Path(__file__).resolve().parent
            / "helpers/multires_event_v2_rank_artifact_worker.py"
        )
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "rank-artifacts"
            started = time.monotonic()
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=2",
                    str(worker),
                    str(output_root),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            elapsed = time.monotonic() - started
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            self.assertLess(elapsed, 30.0)
            manifest_path = output_root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "PASSED")
            self.assertEqual(manifest["world_size"], 2)
            self.assertEqual(
                [row["rank"] for row in manifest["rank_artifacts"]],
                [0, 1],
            )
            for rank in (0, 1):
                path = output_root / f"progress.rank{rank:05d}.jsonl"
                self.assertTrue(path.is_file())
                self.assertEqual(
                    manifest["rank_artifacts"][rank]["sha256"],
                    sha256_file(path),
                )
            reopened = validate_rank_local_artifact_preflight(
                output_root,
                expected_mode="block",
                expected_world_size=2,
            )
            self.assertEqual(reopened["manifest_sha256"], sha256_file(manifest_path))
            tampered = output_root / "progress.rank00001.jsonl"
            tampered.write_text(
                tampered.read_text(encoding="utf-8")
                + json.dumps({"event": "tampered"})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "file/hash"):
                validate_rank_local_artifact_preflight(
                    output_root,
                    expected_mode="block",
                    expected_world_size=2,
                )

    def test_two_rank_gloo_preflight_reports_rank_one_failure_without_timeout(self) -> None:
        worker = (
            Path(__file__).resolve().parent
            / "helpers/multires_event_v2_rank_artifact_worker.py"
        )
        with tempfile.TemporaryDirectory() as directory:
            started = time.monotonic()
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=2",
                    str(worker),
                    str(Path(directory) / "rank-artifacts"),
                    "--inject-rank-one-writer-noop",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            elapsed = time.monotonic() - started
            combined = completed.stdout + completed.stderr
            self.assertNotEqual(completed.returncode, 0)
            self.assertLess(elapsed, 20.0)
            self.assertIn("rank 1 FileNotFoundError", combined)
            self.assertNotIn("Timeout(ms)=600000", combined)

    def test_two_rank_failure_matrix_never_waits_for_the_formal_timeout(self) -> None:
        worker = (
            Path(__file__).resolve().parent
            / "helpers/multires_event_v2_rank_artifact_worker.py"
        )
        cases = (
            ("write", 0, "FileNotFoundError"),
            ("write", 1, "FileNotFoundError"),
            ("hash", 0, "injected hash failure"),
            ("hash", 1, "injected hash failure"),
            ("gather", 0, "injected gather failure"),
            ("gather", 1, "injected gather failure"),
            ("scoring", 0, "injected scoring failure"),
            ("scoring", 1, "injected scoring failure"),
            ("report", 0, "injected report failure"),
            ("report", 1, "injected report failure"),
            ("optimizer", 0, "injected optimizer failure"),
            ("optimizer", 1, "injected optimizer failure"),
            ("checkpoint", 0, "injected checkpoint failure"),
            ("checkpoint", 1, "injected checkpoint failure"),
            ("finalization", 0, "injected finalization failure"),
            ("finalization", 1, "injected finalization failure"),
        )
        for stage, rank, marker in cases:
            with self.subTest(stage=stage, rank=rank), tempfile.TemporaryDirectory() as directory:
                started = time.monotonic()
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nproc_per_node=2",
                        str(worker),
                        str(Path(directory) / "rank-artifacts"),
                        "--inject-stage",
                        stage,
                        "--inject-rank",
                        str(rank),
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    text=True,
                    capture_output=True,
                    timeout=25,
                    check=False,
                )
                elapsed = time.monotonic() - started
                combined = completed.stdout + completed.stderr
                self.assertNotEqual(completed.returncode, 0, msg=combined)
                self.assertLess(elapsed, 20.0, msg=combined)
                self.assertIn(marker, combined)
                self.assertNotIn("Timeout(ms)=600000", combined)


class _Contract(SimpleNamespace):
    def validate_target_record(self, _record, *, verify_content_hash=True):
        return None


def _contract(root: Path | None = None) -> _Contract:
    registry = _registry()
    return _Contract(
        dataset_root=root or Path("/tmp/fake-v2"),
        manifest={
            "dataset_id": "synthetic-r6",
            "files": {"sample_manifest": {"sha256": "2" * 64}},
        },
        contract_bundle_hash="3" * 64,
        contract_hashes={
            "process": "4" * 64,
            "emission": "5" * 64,
            "projection": "6" * 64,
            "relation": "7" * 64,
            "sidecar_schema": "8" * 64,
        },
        process_registry=registry,
        core_fields=FIELDS,
        registered_core_field_ids=REGISTERED_CORE_FIELD_IDS,
        dense_fields=DENSE,
        dense_abnormal_conditions=registry["condition_sets"]["dense_abnormal"],
        ordinal_fields=("gcs_eye", "gcs_motor"),
        ordinal_max={"gcs_eye": 4, "gcs_motor": 6},
        verbal_field="gcs_verbal",
        lab_fields=LABS,
        respiratory_field="respiratory_support",
        respiratory_modalities=RESPIRATORY_MODALITIES,
        vasopressor_field="vasopressor_support",
        vasopressor_agents=VASOPRESSOR_AGENTS,
        ned_field="norepinephrine_equivalent_dose",
        uop_field="urine_output",
    )


def _scale_for(schema) -> dict[str, object]:
    return {
        "scales": {
            key: {"center": 0.0, "scale": 1.0}
            for key in required_standardized_scale_keys(schema)
        },
        "lab_scales": {
            field: {"center": 0.0, "scale": 1.0} for field in LABS
        },
    }


class MultiresEventV2PrimitiveVectorTest(unittest.TestCase):
    def test_selected_heads_preserve_cached_registry_rollout_and_rng_stream(self) -> None:
        from trauma_predict.eval.multires_event_v2_projections import (
            generated_coherence_report,
        )

        torch.manual_seed(71)
        hidden_size = 8
        batch_size = 1
        decoder = FieldStateTrajectoryDecoder(
            hidden_size=hidden_size,
            num_heads=2,
            layers=1,
            dropout=0.0,
            block_count=6,
            field_count=29,
            relation_type_count=14,
        ).eval()
        heads = PrimitiveParameterHeads(
            hidden_size,
            V2_PRIMITIVE_HEAD_DIMS,
            dropout=0.0,
        ).eval()
        class FastDeterministicFeedback(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.feedback_dims = dict(V2_PRIMITIVE_FEEDBACK_DIMS)

            def forward(self, values, masks, *, leading_shape):
                summary = torch.zeros(leading_shape, dtype=torch.float32)
                valid = torch.zeros(leading_shape, dtype=torch.bool)
                for likelihood_id in self.feedback_dims:
                    component_mask = masks[likelihood_id].bool()
                    safe = torch.nan_to_num(values[likelihood_id].float())
                    safe = torch.sign(safe) * torch.log1p(safe.abs())
                    summary = summary + (safe * component_mask).sum(dim=-1)
                    valid = valid | component_mask.any(dim=-1)
                basis = torch.arange(1, hidden_size + 1, dtype=torch.float32)
                return torch.tanh(summary.unsqueeze(-1) / basis), valid

        feedback = FastDeterministicFeedback().eval()
        rollout = AutoregressiveFieldStateRollout(6, 29).eval()
        queries = torch.randn(batch_size, 6, 29, hidden_size)
        memory = torch.randn(batch_size, 2, hidden_size)
        memory_mask = torch.ones(batch_size, 2, dtype=torch.bool)

        contract = _contract()
        contract.emission_registry = {
            "field_supports": {
                "dense_continuous": {
                    field: {"lower": -20.0, "upper": 200.0, "unit": "u"}
                    for field in DENSE
                }
            }
        }

        def generate(*, selected: bool):
            sampler = RegistryPrimitiveSampler(
                _registry(),
                _metadata(),
                expected_lab_scale_artifact_hash=SCALE_HASH,
            )
            calls = {name: 0 for name in V2_PRIMITIVE_HEAD_DIMS}
            handles = [
                module.register_forward_hook(
                    lambda _module, _inputs, _output, name=name: calls.__setitem__(
                        name,
                        calls[name] + 1,
                    )
                )
                for name, module in heads.heads.items()
            ]
            try:
                torch.manual_seed(9173)
                with torch.inference_mode():
                    outputs = rollout(
                        queries,
                        memory,
                        memory_mask,
                        decoder=decoder,
                        primitive_heads=heads,
                        feedback_encoder=feedback,
                        mode="trajectory",
                        sampler=sampler,
                        use_cache=True,
                        use_selected_heads=selected,
                    )
                    rng_tail = torch.rand(16)
            finally:
                for handle in handles:
                    handle.remove()
            return outputs, rng_tail, calls

        reference, reference_rng_tail, reference_calls = generate(selected=False)
        selected, selected_rng_tail, selected_calls = generate(selected=True)
        torch.testing.assert_close(selected[0], reference[0], rtol=0.0, atol=0.0)
        for selected_bank, reference_bank in zip(selected[1:], reference[1:], strict=True):
            self.assertEqual(tuple(selected_bank), tuple(reference_bank))
            for likelihood_id in selected_bank:
                self.assertTrue(
                    torch.equal(
                        selected_bank[likelihood_id],
                        reference_bank[likelihood_id],
                    ),
                    likelihood_id,
                )
        self.assertTrue(torch.equal(selected_rng_tail, reference_rng_tail))
        self.assertEqual(sum(reference_calls.values()), 174 * 19)
        self.assertEqual(sum(selected_calls.values()), 414)

        reports = generated_coherence_report(selected[1], selected[2], contract)
        self.assertEqual(len(reports), batch_size)
        self.assertTrue(all(bool(row["coherent"]) for row in reports))

    def test_phi_uses_natural_and_conditional_coordinates_and_expands_teacher_masks(self) -> None:
        contract = _contract()
        schema = build_standardized_primitive_schema(contract)
        self.assertEqual(len(required_standardized_scale_keys(schema)), 38)
        encodings = {row.encoding for row in schema}
        self.assertIn("bounded_unit", encodings)
        self.assertIn("natural_asinh_nonnegative_integer", encodings)
        self.assertIn("positive_log_robust_affine", encodings)
        self.assertIn("bounded_ratio", encodings)

        values = {
            key: torch.zeros((1, 6, 29, width), dtype=torch.float64)
            for key, width in V2_PRIMITIVE_FEEDBACK_DIMS.items()
        }
        teacher_masks = {
            key: torch.zeros((1, 6, 29), dtype=torch.bool)
            for key in V2_PRIMITIVE_FEEDBACK_DIMS
        }
        field_index = {field: index for index, field in enumerate(FIELDS)}
        heart = field_index["heart_rate"]
        lactate = field_index["lactate"]
        ned = field_index["norepinephrine_equivalent_dose"]
        uop = field_index["urine_output"]
        values["dense_joint_value_state"][:, :, heart] = torch.tensor(
            [80.0, 70.0, 90.0, 82.0]
        )
        teacher_masks["dense_joint_value_state"][:, :, heart] = True
        values["lab_joint_value_state"][:, :, lactate] = torch.tensor([2.0, 1.0, 3.0])
        teacher_masks["lab_joint_value_state"][:, :, lactate] = True
        values["ned_joint_value_state"][:, :, ned] = torch.tensor([1.0, 2.0, 0.5])
        teacher_masks["ned_joint_value_state"][:, :, ned] = True
        values["hurdle_negative_binomial_count"][:, :, uop, 0] = 3.0
        teacher_masks["hurdle_negative_binomial_count"][:, :, uop] = True
        values["uop_sum_given_count"][:, :, uop, 0] = 100.0
        teacher_masks["uop_sum_given_count"][:, :, uop] = True

        scale = _scale_for(schema)
        teacher_phi = standardize_primitive_trajectory(
            values, teacher_masks, schema, scale
        )
        generated_masks = {
            key: mask.unsqueeze(-1).expand_as(values[key])
            for key, mask in teacher_masks.items()
        }
        generated_phi = standardize_primitive_trajectory(
            values, generated_masks, schema, scale
        )
        self.assertTrue(torch.equal(teacher_phi, generated_phi))
        self.assertEqual(teacher_phi.dtype, torch.float64)

        positions = {
            row.within_block_id: index
            for index, row in enumerate(row for row in schema if row.block_index == 0)
        }
        prefix = "norepinephrine_equivalent_dose.ned_joint_value_state."
        self.assertEqual(teacher_phi[0, 0, positions[prefix + "positive_max_gate"]], 1.0)
        self.assertAlmostEqual(
            float(teacher_phi[0, 0, positions[prefix + "last_over_max"]]), 0.5
        )
        self.assertAlmostEqual(
            float(teacher_phi[0, 0, positions[prefix + "mean_over_max"]]), 0.25
        )
        uop_prefix = "urine_output.uop_sum_given_count."
        self.assertEqual(teacher_phi[0, 0, positions[uop_prefix + "positive_sum_gate"]], 1.0)

        zero = {key: value.clone() for key, value in values.items()}
        zero["ned_joint_value_state"][:, :, ned] = 0.0
        zero["uop_sum_given_count"][:, :, uop] = 0.0
        zero_phi = standardize_primitive_trajectory(zero, teacher_masks, schema, scale)
        self.assertFalse(torch.equal(teacher_phi, zero_phi))

    def test_score_kernels_have_known_two_member_values(self) -> None:
        samples = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
        truth = torch.tensor([1.0], dtype=torch.float64)
        self.assertAlmostEqual(empirical_energy_score(samples, truth), 0.5)
        self.assertAlmostEqual(empirical_crps(samples[:, 0], 1.0), 0.5)
        self.assertEqual(
            common_random_seed(7, "anchor", trajectory_start=0, trajectory_count=2),
            common_random_seed(7, "anchor", trajectory_start=0, trajectory_count=2),
        )

    def test_one_hundred_registry_samples_pass_exact_generated_coherence(self) -> None:
        from trauma_predict.eval.multires_event_v2_projections import (
            generated_coherence_report,
        )

        contract = _contract()
        contract.emission_registry = {
            "field_supports": {
                "dense_continuous": {
                    field: {"lower": -20.0, "upper": 200.0, "unit": "u"}
                    for field in DENSE
                }
            }
        }
        parameters = {
            key: torch.zeros(100, width)
            for key, width in V2_PRIMITIVE_HEAD_DIMS.items()
        }
        sampler = RegistryPrimitiveSampler(
            _registry(),
            _metadata(),
            expected_lab_scale_artifact_hash=SCALE_HASH,
        )
        generated = {
            key: torch.zeros(100, 6, 29, width, dtype=torch.float64)
            for key, width in V2_PRIMITIVE_FEEDBACK_DIMS.items()
        }
        masks = {
            key: torch.zeros(100, 6, 29, width, dtype=torch.bool)
            for key, width in V2_PRIMITIVE_FEEDBACK_DIMS.items()
        }
        torch.manual_seed(8)
        for block_index in range(6):
            for field_index in range(29):
                values, component_masks = sampler(
                    block_index, field_index, parameters
                )
                for key in generated:
                    generated[key][:, block_index, field_index] = values[key]
                    masks[key][:, block_index, field_index] = component_masks[key]
        active = masks["respiratory_occupancy_vector"][..., 0]
        residual = (
            generated["respiratory_occupancy_vector"].sum(dim=-1).sub(4.0).abs()
        )[active]
        self.assertLessEqual(float(residual.max()), 1e-12)
        reports = generated_coherence_report(generated, masks, contract)
        self.assertEqual(sum(bool(row["coherent"]) for row in reports), 100)

    def test_respiratory_branch_does_not_shift_downstream_crn_stream(self) -> None:
        from trauma_predict.training.multires_event_v2_loss import (
            _sample_respiratory_occupancy,
        )

        def downstream(*, active: bool, selected_code: int) -> torch.Tensor:
            raw = torch.full((7, 39), -100.0)
            raw[:, selected_code - 1] = 100.0
            torch.manual_seed(917)
            _sample_respiratory_occupancy(
                raw,
                torch.full((7,), active, dtype=torch.bool),
            )
            return torch.rand(16)

        reference = downstream(active=False, selected_code=1)
        self.assertTrue(torch.equal(reference, downstream(active=True, selected_code=1)))
        self.assertTrue(torch.equal(reference, downstream(active=True, selected_code=31)))

    def test_vectorized_respiratory_matches_all_legacy_active_sets_exactly(self) -> None:
        from trauma_predict.modeling.multires_event_v2.emissions import sample_categorical
        from trauma_predict.training.multires_event_v2_loss import (
            _sample_respiratory_occupancy,
        )

        def legacy(raw: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
            active_index = sample_categorical(raw[..., :31])
            full_alr = torch.distributions.Normal(
                raw[..., 31:35].double(),
                torch.nn.functional.softplus(raw[..., 35:39].double()) + 1e-4,
                validate_args=False,
            ).sample()
            rows: list[torch.Tensor] = []
            for batch_index in range(raw.shape[0]):
                if not bool(active[batch_index].item()):
                    rows.append(torch.zeros(5, dtype=torch.float64))
                    continue
                code = int(active_index[batch_index].item()) + 1
                selected = torch.tensor(
                    [bool((code >> bit) & 1) for bit in range(5)]
                )
                duration = torch.zeros(5, dtype=torch.float64)
                dimension = int(selected.sum().item()) - 1
                alr = full_alr[batch_index, :dimension]
                proportion = torch.softmax(
                    torch.cat((alr, alr.new_zeros(1))),
                    dim=-1,
                )
                duration[selected] = proportion * 4.0
                rows.append(torch.cat((duration[1:], duration[:1]), dim=-1))
            return torch.stack(rows)

        raw = torch.randn(31, 39)
        for active_set in range(31):
            raw[active_set, :31] = -100.0
            raw[active_set, active_set] = 100.0
        active = torch.arange(31).remainder(3).ne(0)
        torch.manual_seed(551)
        reference = legacy(raw, active)
        reference_rng_tail = torch.rand(16)
        torch.manual_seed(551)
        vectorized = _sample_respiratory_occupancy(raw, active)
        vectorized_rng_tail = torch.rand(16)
        self.assertTrue(torch.equal(vectorized, reference))
        self.assertTrue(torch.equal(vectorized_rng_tail, reference_rng_tail))
        closure = vectorized.sum(dim=-1).sub(4.0).abs()[active]
        self.assertLessEqual(float(closure.max()), 1e-12)


class MultiresEventV2ScaleFitterTest(unittest.TestCase):
    def test_uop_zero_count_null_sum_is_not_a_scale_observation(self) -> None:
        from trauma_predict.eval.multires_event_v2_scale import _collect_fit_values

        contract = _contract()
        values = {
            "urine_output|uop_sum_given_count|log_positive_sum": []
        }
        _collect_fit_values(
            values,
            field="urine_output",
            process={"observation_count": 0, "sum": None},
            contract=contract,
        )
        self.assertEqual(values["urine_output|uop_sum_given_count|log_positive_sum"], [])
        with self.assertRaisesRegex(ValueError, "zero-count"):
            _collect_fit_values(
                values,
                field="urine_output",
                process={"observation_count": 0, "sum": 0.0},
                contract=contract,
            )

    def test_fitter_deduplicates_windows_and_emits_38_positive_iqrs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset_manifest.json").write_text("{}\n", encoding="utf-8")
            contract = _contract(root)
            blocks = []
            fitted_fields = DENSE + (
                "norepinephrine_equivalent_dose",
                "urine_output",
            )
            for index in range(6):
                processes = {}
                for field_offset, field in enumerate(DENSE):
                    base = 10.0 + field_offset + index
                    processes[field] = {
                        "value_state": {
                            "last": base,
                            "min": base - 1.0,
                            "max": base + 2.0,
                            "mean": base + 0.5,
                        }
                    }
                maximum = math.exp(-3.0 + index * 0.25)
                processes["norepinephrine_equivalent_dose"] = {
                    "value_state": {
                        "last": maximum * 0.5,
                        "max": maximum,
                        "mean": maximum * 0.4,
                    }
                }
                processes["urine_output"] = {
                    "observation_count": 1,
                    "sum": 100.0 + 20.0 * index,
                }
                self.assertEqual(set(processes), set(fitted_fields))
                blocks.append(
                    {
                        "relative_start_hour": 4 * index,
                        "relative_end_hour": 4 * (index + 1),
                        "processes": processes,
                    }
                )
            rows = [
                {
                    "sample_id": f"duplicate-{index}",
                    "subject_id": "p1",
                    "stay_id": "s1",
                    "prediction_hour": 12,
                    "split": "train",
                    "blocks": blocks,
                }
                for index in range(2)
            ]
            shard = root / "train.jsonl.gz"
            with gzip.open(shard, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            shard_sha = hashlib.sha256(shard.read_bytes()).hexdigest()

            lab_payload = {
                "source": {
                    "sidecar_dataset_id": contract.manifest["dataset_id"],
                    "sidecar_dataset_manifest_sha256": hashlib.sha256(
                        (root / "dataset_manifest.json").read_bytes()
                    ).hexdigest(),
                    "sidecar_sample_manifest_sha256": "2" * 64,
                    "sidecar_contract_bundle_hash": "3" * 64,
                    "sidecar_process_contract_sha256": "4" * 64,
                    "sidecar_emission_contract_sha256": "5" * 64,
                },
                "fields": {
                    field: {"center": 0.0, "scale": 1.0} for field in LABS
                },
            }
            lab_hash = hashlib.sha256(
                json.dumps(
                    lab_payload, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            lab_payload["content_sha256"] = lab_hash
            lab_path = root / "lab.json"
            lab_path.write_text(json.dumps(lab_payload), encoding="utf-8")
            output = root / "phi.json"
            with patch(
                "trauma_predict.eval.multires_event_v2_scale."
                "MultiresEventV2Contract.from_dataset_root",
                return_value=contract,
            ), patch(
                "trauma_predict.eval.multires_event_v2_scale._train_target_shards",
                return_value=((shard, shard_sha),),
            ):
                payload = fit_standardized_primitive_scale_artifact(
                    target_root=root,
                    lab_scale_path=lab_path,
                    expected_lab_scale_sha256=lab_hash,
                    output_path=output,
                    expected_train_samples=2,
                )
            self.assertEqual(payload["fit_audit"]["fitted_key_count"], 38)
            self.assertEqual(payload["fit_audit"]["zero_iqr_keys"], [])
            self.assertEqual(
                payload["fit_population"]["collapsed_duplicate_field_windows"],
                6 * 11,
            )
            loaded = load_standardized_primitive_scale_artifact(
                output,
                expected_content_sha256=payload["content_sha256"],
                contract=contract,
                expected_lab_scale_artifact_hash=lab_hash,
            )
            self.assertEqual(len(loaded["scales"]), 38)


class MultiresEventV2FreeRunningTest(unittest.TestCase):
    def test_evaluator_encodes_a_multi_anchor_batch_once(self) -> None:
        class Model:
            def __init__(self):
                self.encode_calls = 0
                self.rollout_calls = 0

            def eval(self):
                return self

            def encode_for_rollout(self, **inputs):
                self.encode_calls += 1
                batch = next(iter(inputs.values())).shape[0]
                return (
                    torch.zeros(batch, 2, 3),
                    torch.ones(batch, 2, dtype=torch.bool),
                    torch.zeros(batch, 6, 29, 3),
                )

            def rollout_from_encoded(self, memory, _mask, _queries, **_kwargs):
                self.rollout_calls += 1
                batch = memory.shape[0]
                bank = torch.zeros(batch, 6, 29, 1)
                return {
                    "generated_primitives": {"dummy": bank},
                    "generated_primitive_masks": {
                        "dummy": torch.ones_like(bank, dtype=torch.bool)
                    },
                }

        model = Model()
        contract = SimpleNamespace(
            manifest={"dataset_id": "d"},
            contract_bundle_hash="1" * 64,
            contract_hashes={
                "process": "2" * 64,
                "emission": "3" * 64,
                "projection": "4" * 64,
                "relation": "5" * 64,
                "sidecar_schema": "6" * 64,
            },
            process_registry={},
            active_core_relation_edges=(),
        )
        batch = {
            "sample_id": ["a", "b"],
            "subject_id": ["p1", "p2"],
            "input_batch": {
                key: torch.zeros(2, 1) for key in MODEL_INPUT_KEYS
            },
            "target_primitives": {"dummy": torch.zeros(2, 6, 29, 1)},
            "target_primitive_masks": {
                "dummy": torch.ones(2, 6, 29, 1, dtype=torch.bool)
            },
            "target_primitive_metadata": {},
        }
        physical_schema = (
            PhysicalProjectionSpec(
                "dummy.LAST.NONE",
                "dummy",
                1,
                0,
                "LAST",
                "NONE",
                "dummy",
                0,
                "always",
                "continuous",
                "u",
            ),
        )
        primitive_schema = (
            PrimitiveVectorCoordinate(
                "dummy.M4_01",
                "dummy",
                "dummy.M4_01",
                "dummy",
                0,
                "dummy",
                0,
                0,
                "bounded_unit",
                None,
                None,
                0,
                1,
            ),
        )
        primitive_specs = (
            SimpleNamespace(
                primitive_id="dummy.M4_01",
                likelihood_id="dummy",
                block_index=0,
                field_index=0,
                field="dummy",
            ),
        )

        def project(primitives, *_args):
            size = next(iter(primitives.values())).shape[0]
            return torch.zeros(size, 6, 1), torch.ones(size, 6, 1, dtype=torch.bool)

        def standardize(primitives, *_args):
            size = next(iter(primitives.values())).shape[0]
            return torch.zeros(size, 6, 1, dtype=torch.float64)

        physical_scores = {
            "branch_calibration_rows": [
                {"family": "gate", "probability": 0.5, "outcome": 1}
            ],
            "crps_by_projection": {"dummy.LAST.NONE": 0.0},
            "brier_by_projection": {},
            "rps_by_projection": {},
            "median_mae_by_projection": {"dummy.LAST.NONE": 0.0},
            "coverage_by_projection": {
                "dummy.LAST.NONE": {
                    "truth_active_blocks": 6,
                    "scored_blocks": 6,
                    "generated_active_counts": [2] * 6,
                    "complete": True,
                }
            },
            "physical_metric_contract_status": "complete",
            "physical_metric_blockers": [],
        }
        with tempfile.TemporaryDirectory() as directory, patch.multiple(
            "trauma_predict.eval.multires_event_v2_free_running",
            build_physical_projection_schema=lambda _contract: physical_schema,
            build_standardized_primitive_schema=lambda _contract: primitive_schema,
            load_standardized_primitive_scale_artifact=lambda *_args, **_kwargs: {},
            expand_enabled_core_primitives=lambda _registry: primitive_specs,
            RegistryPrimitiveSampler=lambda *_args, **_kwargs: object(),
            project_physical_primitives=project,
            standardize_primitive_trajectory=standardize,
            generated_coherence_report=lambda primitives, *_args: [
                {"coherent": True, "violations": []}
                for _ in range(next(iter(primitives.values())).shape[0])
            ],
            score_physical_ensemble=lambda *_args: dict(physical_scores),
            score_standardized_primitive_ensemble=lambda *_args: {
                "energy_score": 0.0,
                "lag1_variogram_score_p0_5": 0.0,
                "field_macro_lag1_variogram_score_p0_5": 0.0,
                "relation_edge_macro_variogram_score_p0_5": 0.0,
                "marginal_value_crps": 0.0,
                "marginal_state_crps": 0.0,
                "relation_variogram_by_type": {},
            },
        ):
            result = evaluate_free_running_v2(
                model=model,
                loader=[batch],
                contract=contract,
                device=torch.device("cpu"),
                mode="trajectory",
                expected_samples=2,
                step=1,
                output_dir=directory,
                expected_lab_scale_artifact_hash="a" * 64,
                standardized_primitive_scale_path=Path(directory) / "unused.json",
                expected_standardized_primitive_scale_hash="b" * 64,
                input_normalization_sha256="c" * 64,
                promotion_metric_contract=PROMOTION_CONTRACT,
                trajectories_per_anchor=2,
            )
            self.assertEqual(result["anchors"], 2)
            self.assertEqual(model.encode_calls, 1)
            self.assertEqual(model.rollout_calls, 2)
            sample_path = (
                Path(directory) / "audit_trajectory_samples.rank00000.jsonl.gz"
            )
            with gzip.open(sample_path, "rt", encoding="utf-8") as handle:
                retained = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(retained), 2)
            self.assertTrue(
                all(row["trajectory_index"] == 0 for row in retained)
            )
            self.assertTrue(
                all("primitive_values_flat" in row for row in retained)
            )

    def test_full_promotion_is_conjunctive(self) -> None:
        identity = {
            "dataset_id": "d",
            "contract_bundle_hash": "1" * 64,
            "process_contract_sha256": "2" * 64,
            "emission_contract_sha256": "3" * 64,
            "projection_contract_sha256": "4" * 64,
            "relation_contract_sha256": "1" * 64,
            "sidecar_schema_sha256": "2" * 64,
            "lab_scale_artifact_sha256": "5" * 64,
            "standardized_primitive_scale_sha256": "6" * 64,
            "input_normalization_sha256": "8" * 64,
            "promotion_metric_contract_sha256": "f" * 64,
            "semantic_runtime_identity_sha256": "0" * 64,
            "crn_contract_sha256": "7" * 64,
            "source_tree_sha256": "9" * 64,
            "source_identity_sha256": "a" * 64,
            "git_commit": "b" * 40,
            "git_head_tree": "c" * 40,
            "matched_design_signature": "d" * 64,
            "selected_checkpoint_step": 100,
            "selected_checkpoint_model_sha256": "e" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher_paths = {}
            free_paths = {}
            teacher_values = {"block": 10.0, "trajectory": 9.8, "relational": 9.6}
            free_values = {"block": 2.0, "trajectory": 1.0, "relational": 0.5}
            for mode in ("block", "trajectory", "relational"):
                teacher_paths[mode] = root / f"teacher-{mode}.jsonl"
                free_paths[mode] = root / f"free-{mode}.jsonl"
                with teacher_paths[mode].open("w", encoding="utf-8") as handle:
                    for index in range(4):
                        handle.write(
                            json.dumps(
                                {
                                    "sample_id": f"a{index}",
                                    "subject_id": f"p{index // 2}",
                                    "joint_nll": teacher_values[mode],
                                    "identity": identity,
                                }
                            )
                            + "\n"
                        )
                with free_paths[mode].open("w", encoding="utf-8") as handle:
                    for index in range(4):
                        handle.write(
                            json.dumps(
                                {
                                    "sample_id": f"a{index}",
                                    "subject_id": f"p{index // 2}",
                                    "trajectories": 2,
                                    "coherent_trajectories": 2,
                                    "energy_score": free_values[mode],
                                    "lag1_variogram_score_p0_5": free_values[mode],
                                    "field_macro_lag1_variogram_score_p0_5": free_values[mode],
                                    "relation_edge_macro_variogram_score_p0_5": free_values[mode],
                                    "marginal_value_crps": free_values[mode],
                                    "marginal_state_crps": free_values[mode],
                                    "physical_scores": {
                                        "physical_metric_contract_status": "complete",
                                        "crps_by_projection": {
                                            "heart_rate.LAST.NONE": free_values[mode]
                                        },
                                    },
                                    "identity": identity,
                                }
                            )
                            + "\n"
                        )
            result = evaluate_multires_event_v2_promotion(
                block_teacher_path=teacher_paths["block"],
                trajectory_teacher_path=teacher_paths["trajectory"],
                relational_teacher_path=teacher_paths["relational"],
                block_free_running_path=free_paths["block"],
                trajectory_free_running_path=free_paths["trajectory"],
                relational_free_running_path=free_paths["relational"],
                promotion_metric_contract=PROMOTION_CONTRACT,
                expected_anchors=4,
            )
            self.assertTrue(result["promoted"])
            self.assertTrue(result["gates"]["trajectory_over_block"]["passed"])
            self.assertEqual(result["winner"], "relational")
            self.assertTrue(result["bootstrap"]["shared_subject_index_schedule"])
            self.assertEqual(result["care_and_procedure"]["status"], "not_applicable")
            relational_rows = [
                json.loads(line)
                for line in free_paths["relational"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            relational_rows[0]["subject_id"] = "different-subject"
            free_paths["relational"].write_text(
                "".join(json.dumps(row) + "\n" for row in relational_rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "subject identity differs"):
                evaluate_multires_event_v2_promotion(
                    block_teacher_path=teacher_paths["block"],
                    trajectory_teacher_path=teacher_paths["trajectory"],
                    relational_teacher_path=teacher_paths["relational"],
                    block_free_running_path=free_paths["block"],
                    trajectory_free_running_path=free_paths["trajectory"],
                    relational_free_running_path=free_paths["relational"],
                    promotion_metric_contract=PROMOTION_CONTRACT,
                    expected_anchors=4,
                )
            relational_rows[0]["subject_id"] = "p0"
            for key in ("relation_contract_sha256", "sidecar_schema_sha256"):
                with self.subTest(identity_key=key):
                    original = relational_rows[0]["identity"][key]
                    relational_rows[0]["identity"][key] = "0" * 64
                    free_paths["relational"].write_text(
                        "".join(json.dumps(row) + "\n" for row in relational_rows),
                        encoding="utf-8",
                    )
                    try:
                        with self.assertRaisesRegex(
                            ValueError, "data/contract/scale/CRN identity"
                        ):
                            evaluate_multires_event_v2_promotion(
                                block_teacher_path=teacher_paths["block"],
                                trajectory_teacher_path=teacher_paths["trajectory"],
                                relational_teacher_path=teacher_paths["relational"],
                                block_free_running_path=free_paths["block"],
                                trajectory_free_running_path=free_paths["trajectory"],
                                relational_free_running_path=free_paths["relational"],
                                promotion_metric_contract=PROMOTION_CONTRACT,
                                expected_anchors=4,
                            )
                    finally:
                        relational_rows[0]["identity"][key] = original
            for row in relational_rows:
                row["relation_edge_macro_variogram_score_p0_5"] = 2.0
                row["physical_scores"]["physical_metric_contract_status"] = (
                    "incomplete_conditional_sample_coverage"
                )
            free_paths["relational"].write_text(
                "".join(json.dumps(row) + "\n" for row in relational_rows),
                encoding="utf-8",
            )
            trajectory_winner = evaluate_multires_event_v2_promotion(
                block_teacher_path=teacher_paths["block"],
                trajectory_teacher_path=teacher_paths["trajectory"],
                relational_teacher_path=teacher_paths["relational"],
                block_free_running_path=free_paths["block"],
                trajectory_free_running_path=free_paths["trajectory"],
                relational_free_running_path=free_paths["relational"],
                promotion_metric_contract=PROMOTION_CONTRACT,
                expected_anchors=4,
            )
            self.assertEqual(trajectory_winner["winner"], "trajectory")
            self.assertTrue(trajectory_winner["trajectory_promoted"])
            self.assertFalse(trajectory_winner["relational_promoted"])


class MultiresEventV2TeacherDecompositionTest(unittest.TestCase):
    def test_414_factor_teacher_decomposition_sums_by_registered_axes(self) -> None:
        registry = _registry()
        count = len(expand_enabled_core_primitives(registry))
        primitive_log_prob = torch.full((2, count), -0.1)
        rows = _teacher_nll_decomposition_rows(
            {
                "primitive_log_prob": primitive_log_prob,
                "primitive_ids": tuple(
                    spec.primitive_id for spec in expand_enabled_core_primitives(registry)
                ),
                "per_sample_nll": -primitive_log_prob.sum(dim=-1),
            },
            registry,
            batch_size=2,
        )
        assert rows is not None
        self.assertEqual(len(rows[0]["by_block"]), 6)
        self.assertEqual(len(rows[0]["by_field"]), 29)
        self.assertAlmostEqual(sum(rows[0]["by_block"].values()), 41.4, places=4)


if __name__ == "__main__":
    unittest.main()
