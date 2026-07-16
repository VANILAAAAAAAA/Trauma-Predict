from __future__ import annotations

import gzip
import hashlib
import inspect
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from tests.test_multires_event_v2_loss import (
    DENSE,
    FIELDS,
    LABS,
    SCALE_HASH,
    _metadata,
    _registry,
)
from trauma_predict.data.multires_event_v2 import MultiresEventV2RelationContract
from trauma_predict.eval.multires_event_v2 import (
    MODEL_INPUT_KEYS,
    _teacher_nll_decomposition_rows,
)
from trauma_predict.eval.multires_event_v2_free_running import (
    _encode_batch_once,
    _emit_rank_progress,
    common_random_seed,
    evaluate_free_running_v2,
    probe_free_running_v2_capacity,
    validate_rank_local_artifact_preflight,
    verify_rank_local_artifact_preflight,
)
from trauma_predict.eval.multires_event_v2_metric_contract import (
    load_trajectory_metric_contract,
)
from trauma_predict.eval.multires_event_v2_projections import (
    build_standardized_primitive_schema,
    empirical_crps,
    empirical_energy_score,
    generated_coherence_report,
    load_standardized_primitive_scale_artifact,
    required_standardized_scale_keys,
    standardize_primitive_trajectory,
)
from trauma_predict.eval.multires_event_v2_scale import (
    fit_standardized_primitive_scale_artifact,
)
from trauma_predict.modeling.multires_event_v2.field_state import PrimitiveParameterHeads
from trauma_predict.modeling.multires_event_v2.rollout import AutoregressiveFieldStateRollout
from trauma_predict.modeling.multires_event_v2.trajectory import FieldStateTrajectoryDecoder
from trauma_predict.training.multires_event_v2_loss import (
    REGISTERED_CORE_FIELD_IDS,
    RegistryPrimitiveSampler,
    V2_PRIMITIVE_FEEDBACK_DIMS,
    V2_PRIMITIVE_HEAD_DIMS,
    expand_enabled_core_primitives,
)
from trauma_predict.training.observability import append_jsonl, sha256_file


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "configs/train/p100_multires_event_v2_relation_v2.yaml"
RESPIRATORY_MODALITIES = (
    "RESP_INVASIVE", "RESP_NONINVASIVE", "RESP_HIGH_FLOW", "RESP_OTHER_OXYGEN"
)
VASOPRESSOR_AGENTS = (
    "VASO_NOREPINEPHRINE", "VASO_EPINEPHRINE", "VASO_PHENYLEPHRINE",
    "VASO_VASOPRESSIN", "VASO_DOPAMINE", "VASO_OTHER",
)


class _Contract:
    def __init__(self, root: Path | None = None) -> None:
        registry = _registry()
        self.dataset_root = root or Path("/tmp/fake-v2")
        self.manifest = {
            "dataset_id": "synthetic-r9",
            "files": {"sample_manifest": {"sha256": "2" * 64}},
        }
        self.contract_bundle_hash = "3" * 64
        self.contract_hashes = {
            "process": "4" * 64,
            "emission": "5" * 64,
            "projection": "6" * 64,
            "relation": "7" * 64,
            "sidecar_schema": "8" * 64,
        }
        self.emission_registry = {
            "field_supports": {
                "dense_continuous": {
                    field: {"lower": -20.0, "upper": 200.0, "unit": "u"}
                    for field in DENSE
                }
            }
        }
        self.process_registry = registry
        self.core_fields = FIELDS
        self.registered_core_field_ids = REGISTERED_CORE_FIELD_IDS
        self.dense_fields = DENSE
        self.dense_abnormal_conditions = registry["condition_sets"]["dense_abnormal"]
        self.ordinal_fields = ("gcs_eye", "gcs_motor")
        self.ordinal_max = {"gcs_eye": 4, "gcs_motor": 6}
        self.verbal_field = "gcs_verbal"
        self.lab_fields = LABS
        self.respiratory_field = "respiratory_support"
        self.respiratory_modalities = RESPIRATORY_MODALITIES
        self.vasopressor_field = "vasopressor_support"
        self.vasopressor_agents = VASOPRESSOR_AGENTS
        self.ned_field = "norepinephrine_equivalent_dose"
        self.uop_field = "urine_output"

    def validate_target_record(self, _record, *, verify_content_hash=True):
        return None


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


class _FakeFreeRunningRelation:
    version = "relation-v2-test"
    bundle_hash = "9" * 64
    file_hashes = {"target": "a" * 64, "input_target": "b" * 64}
    target_edges = ()


class _FakeFreeRunningModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.encode_batch_sizes: list[int] = []
        self.rollout_batch_sizes: list[int] = []

    def encode_for_rollout(self, **inputs):
        batch = int(inputs["event_field_ids"].shape[0])
        self.encode_batch_sizes.append(batch)
        return {
            "memory": self.weight.mul(0.0).expand(batch, 1, 1),
            "memory_mask": torch.ones(batch, 1, dtype=torch.bool),
            "query_tokens": self.weight.mul(0.0).expand(batch, 6, 29, 1),
        }

    def rollout_from_encoded(self, memory, memory_mask, query_tokens, *, sampler):
        del memory_mask, query_tokens, sampler
        batch = int(memory.shape[0])
        self.rollout_batch_sizes.append(batch)
        values = {
            key: self.weight.mul(0.0).expand(batch, 6, 29, width)
            for key, width in V2_PRIMITIVE_FEEDBACK_DIMS.items()
        }
        return {
            "generated_primitives": values,
            "generated_primitive_masks": {
                key: torch.ones_like(value, dtype=torch.bool)
                for key, value in values.items()
            },
        }


def _fake_free_running_batch(sample_ids: list[str]) -> dict[str, object]:
    batch = len(sample_ids)
    input_batch = {
        key: (
            torch.zeros(batch, dtype=torch.long)
            if key == "latest_input_block_index"
            else torch.zeros(batch, 1)
        )
        for key in MODEL_INPUT_KEYS
    }
    values = torch.zeros(batch, 6, 29, 1)
    return {
        "input_batch": input_batch,
        "target_primitives": {"fake": values},
        "target_primitive_masks": {
            "fake": torch.ones_like(values, dtype=torch.bool)
        },
        "target_primitive_metadata": {},
        "sample_id": sample_ids,
        "subject_id": [f"subject-{sample_id}" for sample_id in sample_ids],
    }


def _fake_projection(primitives, *_args):
    batch = int(next(iter(primitives.values())).shape[0])
    return torch.zeros(batch, 6, 1), torch.ones(batch, 6, 1, dtype=torch.bool)


def _fake_standardization(primitives, *_args):
    batch = int(next(iter(primitives.values())).shape[0])
    return torch.zeros(batch, 6, 1)


def _fake_physical_scores(*_args):
    return {
        "branch_calibration_rows": [
            {"probability": 0.25, "outcome": 0, "family": "fake"}
        ],
        "coverage_by_projection": {
            "fake": {
                "truth_active_blocks": 1,
                "scored_blocks": 1,
                "generated_active_counts": [2],
            }
        },
        "crps_by_projection": {"fake": 1.0},
        "brier_by_projection": {"fake": 0.25},
        "median_mae_by_projection": {"fake": 0.5},
        "rps_by_projection": {"fake": 0.125},
        "physical_metric_contract_status": "complete",
    }


def _fake_trajectory_scores(*_args):
    return {
        "energy_score": 1.0,
        "lag1_variogram_score_p0_5": 2.0,
        "field_macro_lag1_variogram_score_p0_5": 3.0,
        "relation_edge_macro_variogram_score_p0_5": 4.0,
        "marginal_value_crps": 5.0,
        "marginal_state_crps": 6.0,
        "relation_variogram_by_type": {"fake": 7.0},
    }


class RelationV2FreeRunningSafetyTest(unittest.TestCase):
    def _require_local_tcpstore(self) -> None:
        if os.environ.get("CODEX_SANDBOX_NETWORK_DISABLED") == "1":
            self.skipTest("Codex sandbox forbids the PyTorch TCPStore")
        try:
            probe = socket.socket()
            probe.bind(("127.0.0.1", 0))
            probe.close()
        except OSError:
            self.skipTest("sandbox forbids the localhost TCPStore required by torchrun")

    def test_crn_is_deterministic_and_anchor_specific(self) -> None:
        reference = common_random_seed(
            17, "sample-a", trajectory_start=0, trajectory_count=100
        )
        self.assertEqual(
            reference,
            common_random_seed(17, "sample-a", trajectory_start=0, trajectory_count=100),
        )
        self.assertNotEqual(
            reference,
            common_random_seed(17, "sample-b", trajectory_start=0, trajectory_count=100),
        )
        self.assertNotEqual(
            reference,
            common_random_seed(17, "sample-a", trajectory_start=50, trajectory_count=50),
        )

    def test_free_running_apis_expose_no_relation_switch(self) -> None:
        forbidden = {"mode", "relation_adjacency", "relation_type_lags"}
        for function in (
            evaluate_free_running_v2,
            verify_rank_local_artifact_preflight,
            validate_rank_local_artifact_preflight,
        ):
            with self.subTest(function=function.__name__):
                self.assertTrue(
                    forbidden.isdisjoint(inspect.signature(function).parameters)
                )

    def test_input_encoder_runs_once_for_a_multi_anchor_batch(self) -> None:
        class Model:
            def __init__(self) -> None:
                self.calls = 0

            def encode_for_rollout(self, **inputs):
                self.calls += 1
                batch = inputs["event_field_ids"].shape[0]
                return (
                    torch.zeros(batch, 2, 3),
                    torch.ones(batch, 2, dtype=torch.bool),
                    torch.zeros(batch, 6, 29, 3),
                )

        model = Model()
        batch = {
            "input_batch": {
                key: (
                    torch.zeros(2, dtype=torch.long)
                    if key == "latest_input_block_index"
                    else torch.zeros(2, 1)
                )
                for key in MODEL_INPUT_KEYS
            }
        }
        encoded = _encode_batch_once(model, batch, expected_batch_size=2)
        self.assertEqual(model.calls, 1)
        self.assertEqual(set(encoded), {"memory", "memory_mask", "query_tokens"})
        self.assertEqual(encoded["query_tokens"].shape[:3], (2, 6, 29))

    def test_atomic_chunks_resume_before_encode_and_merge_like_uninterrupted(self) -> None:
        contract = _Contract()
        relation = _FakeFreeRunningRelation()
        loader = [
            _fake_free_running_batch(["sample-0", "sample-1"]),
            _fake_free_running_batch(["sample-2", "sample-3"]),
        ]
        evaluation_identity = {
            "source_tree_sha256": "c" * 64,
            "source_identity_sha256": "d" * 64,
            "git_commit": "e" * 40,
            "git_head_tree": "f" * 40,
            "run_contract_signature": "1" * 64,
            "selected_checkpoint_step": 4000,
            "selected_checkpoint_model_sha256": "2" * 64,
        }

        def run(root: Path, model, *, limit):
            return evaluate_free_running_v2(
                model=model,
                loader=loader,
                contract=contract,
                relation_contract=relation,
                device=torch.device("cpu"),
                expected_samples=4,
                step=4000,
                output_dir=root,
                expected_lab_scale_artifact_hash=SCALE_HASH,
                standardized_primitive_scale_path=root / "unused.json",
                expected_standardized_primitive_scale_hash="3" * 64,
                input_normalization_sha256="4" * 64,
                trajectory_metric_contract={},
                evaluation_identity=evaluation_identity,
                trajectories_per_anchor=2,
                trajectory_batch_size=2,
                precision="fp32",
                chunk_target_anchors=2,
                max_new_anchors=limit,
            )

        base = "trauma_predict.eval.multires_event_v2_free_running."
        with tempfile.TemporaryDirectory() as resumed_directory, tempfile.TemporaryDirectory(
        ) as full_directory, patch(
            base + "build_physical_projection_schema", return_value=[]
        ), patch(
            base + "build_standardized_primitive_schema", return_value=[]
        ), patch(
            base + "load_standardized_primitive_scale_artifact", return_value={}
        ), patch(
            base + "expand_enabled_core_primitives", return_value=[]
        ), patch(
            base + "RegistryPrimitiveSampler", return_value=object()
        ), patch(
            base + "project_physical_primitives", side_effect=_fake_projection
        ), patch(
            base + "standardize_primitive_trajectory", side_effect=_fake_standardization
        ), patch(
            base + "generated_coherence_report",
            side_effect=lambda primitives, *_args: [
                {"coherent": True, "violations": []}
                for _ in range(next(iter(primitives.values())).shape[0])
            ],
        ), patch(
            base + "score_physical_ensemble", side_effect=_fake_physical_scores
        ), patch(
            base + "score_standardized_primitive_ensemble",
            side_effect=_fake_trajectory_scores,
        ), patch(
            base + "_trajectory_export_row",
            side_effect=lambda **kwargs: {
                "schema_version": "fake-audit.v1",
                "sample_id": kwargs["sample_id"],
                "subject_id": kwargs["subject_id"],
                "trajectory_index": kwargs["trajectory_index"],
                "crn_seed": kwargs["crn_seed"],
            },
        ):
            resumed_root = Path(resumed_directory)
            first_model = _FakeFreeRunningModel()
            partial = run(resumed_root, first_model, limit=2)
            self.assertEqual(partial["status"], "INCOMPLETE")
            self.assertEqual(partial["anchors"], 2)
            self.assertEqual(first_model.encode_batch_sizes, [2])
            hosted = json.loads(
                (resumed_root / "hosted_progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(hosted["status"], "INCOMPLETE")
            self.assertEqual((hosted["completed"], hosted["expected"]), (2, 4))
            self.assertEqual(len(hosted["chunk_manifests"]), 1)
            self.assertEqual(hosted["set_sha256"], hosted["chunk_manifest_set_sha256"])
            chunk_manifest_path = (
                resumed_root / hosted["chunk_manifests"][0]["manifest_path"]
            )
            chunk_manifest = json.loads(
                chunk_manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(chunk_manifest["files"]),
                {
                    "scores",
                    "audit_trajectories",
                    "calibration_coverage_sufficient_stats",
                },
            )
            self.assertEqual(
                chunk_manifest["source_model_run_identity"][
                    "selected_checkpoint_model_sha256"
                ],
                evaluation_identity["selected_checkpoint_model_sha256"],
            )
            self.assertRegex(chunk_manifest["sample_schema_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(
                chunk_manifest["sample_schema_identity_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertEqual(
                chunk_manifest["crn_contract_sha256"],
                hosted["identity"]["crn_contract_sha256"],
            )

            stale = (
                resumed_root
                / "chunks/rank00000/chunk000001.tmp.interrupted"
            )
            stale.mkdir()
            (stale / "garbage").write_text("not adopted", encoding="utf-8")
            second_model = _FakeFreeRunningModel()
            resumed = run(resumed_root, second_model, limit=2)
            self.assertNotIn("status", resumed)
            self.assertEqual(resumed["anchors"], 4)
            # The first completed loader batch never reaches move_to_device or encode.
            self.assertEqual(second_model.encode_batch_sizes, [2])

            full_root = Path(full_directory)
            full_model = _FakeFreeRunningModel()
            uninterrupted = run(full_root, full_model, limit=None)
            self.assertEqual(full_model.encode_batch_sizes, [2, 2])
            self.assertEqual(uninterrupted["anchors"], resumed["anchors"])
            self.assertEqual(
                sha256_file(resumed_root / "per_anchor_scores.rank00000.jsonl"),
                sha256_file(full_root / "per_anchor_scores.rank00000.jsonl"),
            )
            self.assertEqual(
                sha256_file(
                    resumed_root / "audit_trajectory_samples.rank00000.jsonl.gz"
                ),
                sha256_file(full_root / "audit_trajectory_samples.rank00000.jsonl.gz"),
            )
            self.assertEqual(resumed["branch_calibration"], uninterrupted["branch_calibration"])
            self.assertEqual(
                resumed["conditional_sample_coverage_by_projection"],
                uninterrupted["conditional_sample_coverage_by_projection"],
            )
            hosted = json.loads(
                (resumed_root / "hosted_progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(hosted["status"], "COMPLETE")
            self.assertEqual(hosted["completed"], 4)
            self.assertEqual(len(hosted["chunk_manifests"]), 2)

            score_path = (
                resumed_root
                / "chunks/rank00000/chunk000000/scores.jsonl"
            )
            score_path.write_text(
                score_path.read_text(encoding="utf-8") + "{}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "file/hash"):
                run(resumed_root, _FakeFreeRunningModel(), limit=2)

    def test_formal_capacity_probe_restores_rng_and_parameter_state(self) -> None:
        model = _FakeFreeRunningModel()
        model.train()
        batch = _fake_free_running_batch(["probe-0", "probe-1"])
        cpu_rng_before = torch.get_rng_state().clone()
        with patch(
            "trauma_predict.eval.multires_event_v2_free_running.RegistryPrimitiveSampler",
            return_value=object(),
        ):
            result = probe_free_running_v2_capacity(
                model=model,
                validation_batch=batch,
                contract=_Contract(),
                device=torch.device("cpu"),
                expected_lab_scale_artifact_hash=SCALE_HASH,
                precision="fp32",
            )
        self.assertEqual(result["status"], "PASSED")
        self.assertEqual(result["encode_calls"], 1)
        self.assertEqual(model.encode_batch_sizes, [1])
        self.assertEqual(model.rollout_batch_sizes, [100])
        self.assertEqual(
            result["generated_primitive_shapes"]["dense_joint_value_state"],
            [100, 6, 29, 4],
        )
        self.assertTrue(result["parameter_state_unchanged"])
        self.assertTrue(result["rng"]["cpu_restored"])
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng_before))
        self.assertTrue(model.training)

    def test_rank_artifact_preflight_is_hash_bound_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = verify_rank_local_artifact_preflight(output_dir=directory)
            self.assertEqual(result["status"], "PASSED")
            reopened = validate_rank_local_artifact_preflight(
                directory,
                expected_world_size=1,
            )
            self.assertEqual(reopened["model_contract"], "relation_v2")
            progress = Path(directory) / "progress.rank00000.jsonl"
            progress.write_text(progress.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_rank_local_artifact_preflight(
                    directory,
                    expected_world_size=1,
                )

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
            self.assertEqual(rows[0]["model_contract"], "relation_v2")
            self.assertEqual(rows[0]["completed_anchors"], 0)
            self.assertRegex(sha256_file(progress_path), r"^[0-9a-f]{64}$")

            shared_path = root / "metrics.jsonl"
            append_jsonl(shared_path, {"event": "must_not_be_written"})
            self.assertFalse(shared_path.exists())

    def test_two_rank_preflight_and_injected_failure_close_without_timeout(self) -> None:
        """Exercise generic rank-artifact helpers independently of the formal world-size-one route."""
        self._require_local_tcpstore()
        worker = ROOT / "tests/helpers/multires_event_v2_rank_artifact_worker.py"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pass"
            completed = subprocess.run(
                [
                    sys.executable, "-m", "torch.distributed.run", "--standalone",
                    "--nproc_per_node=2", str(worker), str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stdout + completed.stderr)
            reopened = validate_rank_local_artifact_preflight(
                output,
                expected_world_size=2,
            )
            self.assertEqual([row["rank"] for row in reopened["rank_artifacts"]], [0, 1])

        with tempfile.TemporaryDirectory() as directory:
            started = time.monotonic()
            failed = subprocess.run(
                [
                    sys.executable, "-m", "torch.distributed.run", "--standalone",
                    "--nproc_per_node=2", str(worker), str(Path(directory) / "fail"),
                    "--inject-rank-one-writer-noop",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            combined = failed.stdout + failed.stderr
            self.assertNotEqual(failed.returncode, 0, msg=combined)
            self.assertLess(time.monotonic() - started, 25.0, msg=combined)
            self.assertIn("rank 1 FileNotFoundError", combined)

    def test_two_rank_eight_stage_failure_matrix_never_waits_for_formal_timeout(
        self,
    ) -> None:
        """Keep distributed failure propagation covered even though formal training uses one P100."""
        self._require_local_tcpstore()
        worker = ROOT / "tests/helpers/multires_event_v2_rank_artifact_worker.py"
        cases = tuple(
            (stage, rank, marker)
            for stage, marker in (
                ("write", "FileNotFoundError"),
                ("hash", "injected hash failure"),
                ("gather", "injected gather failure"),
                ("scoring", "injected scoring failure"),
                ("report", "injected report failure"),
                ("optimizer", "injected optimizer failure"),
                ("checkpoint", "injected checkpoint failure"),
                ("finalization", "injected finalization failure"),
            )
            for rank in (0, 1)
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
                    cwd=ROOT,
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

    def test_report_metrics_are_bound_to_23_relation_v2_cross_edges(self) -> None:
        train = yaml.safe_load(TRAIN.read_text(encoding="utf-8"))
        relations = MultiresEventV2RelationContract.from_default_config()
        payload = load_trajectory_metric_contract(
            ROOT / train["trajectory_metric_contract"],
            expected_sha256=train["trajectory_metric_contract_hash"],
            relation_contract=relations,
        )
        structural = [
            edge
            for edge in relations.target_edges
            if edge.time_scope == "same_future_block_registered_order"
            and edge.source_field != edge.target_field
        ]
        self.assertEqual(len(structural), 23)
        self.assertEqual(payload["relation_edge_cover"]["expected_edges"], 23)
        self.assertEqual(payload["decision_authority"], "report_only")
        self.assertTrue({"bootstrap", "gates", "winner_rule"}.isdisjoint(payload))

    def test_one_hundred_registry_samples_pass_generated_coherence(self) -> None:
        contract = _Contract()
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
                values, component_masks = sampler(block_index, field_index, parameters)
                for key in generated:
                    generated[key][:, block_index, field_index] = values[key]
                    masks[key][:, block_index, field_index] = component_masks[key]
        reports = generated_coherence_report(generated, masks, contract)
        self.assertEqual(sum(bool(row["coherent"]) for row in reports), 100)


class RelationV2PrimitiveVectorSafetyTest(unittest.TestCase):
    def test_selected_heads_preserve_cached_registry_rollout_and_rng_stream(self) -> None:
        torch.manual_seed(71)
        hidden_size = 8
        batch_size = 1
        relations = MultiresEventV2RelationContract.from_default_config()
        target_input_field_ids = tuple(
            relations.history_fields.index(field) + 1
            for field in relations.target_fields
        )
        decoder = FieldStateTrajectoryDecoder(
            hidden_size=hidden_size,
            num_heads=2,
            layers=1,
            dropout=0.0,
            block_count=6,
            field_count=29,
            input_field_count=37,
            target_parameter_keys=relations.target_parameter_keys,
            input_parameter_keys=relations.input_target_parameter_keys,
            target_input_field_ids=target_input_field_ids,
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
        memory = torch.randn(batch_size, 38, hidden_size)
        memory_mask = torch.ones(batch_size, 38, dtype=torch.bool)
        relation_arguments = {
            "target_relation_adjacency": torch.as_tensor(
                relations.target_relation_adjacency
            ),
            "target_time_scope_ids": torch.as_tensor(relations.target_time_scope_ids),
            "input_target_relation_adjacency": torch.as_tensor(
                relations.input_target_relation_adjacency
            ),
            "input_target_time_scope_ids": torch.as_tensor(
                relations.input_target_time_scope_ids
            ),
        }

        contract = _Contract()

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
                        sampler=sampler,
                        use_cache=True,
                        use_selected_heads=selected,
                        **relation_arguments,
                    )
                    rng_tail = torch.rand(16)
            finally:
                for handle in handles:
                    handle.remove()
            return outputs, rng_tail, calls

        reference, reference_rng_tail, reference_calls = generate(selected=False)
        selected, selected_rng_tail, selected_calls = generate(selected=True)
        torch.testing.assert_close(selected[0], reference[0], rtol=0.0, atol=0.0)
        for selected_bank, reference_bank in zip(
            selected[1:], reference[1:], strict=True
        ):
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

    def test_phi_uses_natural_and_conditional_coordinates_and_expands_teacher_masks(
        self,
    ) -> None:
        contract = _Contract()
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
            values,
            teacher_masks,
            schema,
            scale,
        )
        generated_masks = {
            key: mask.unsqueeze(-1).expand_as(values[key])
            for key, mask in teacher_masks.items()
        }
        generated_phi = standardize_primitive_trajectory(
            values,
            generated_masks,
            schema,
            scale,
        )
        self.assertTrue(torch.equal(teacher_phi, generated_phi))
        self.assertEqual(teacher_phi.dtype, torch.float64)

        positions = {
            row.within_block_id: index
            for index, row in enumerate(row for row in schema if row.block_index == 0)
        }
        prefix = "norepinephrine_equivalent_dose.ned_joint_value_state."
        self.assertEqual(
            teacher_phi[0, 0, positions[prefix + "positive_max_gate"]],
            1.0,
        )
        self.assertAlmostEqual(
            float(teacher_phi[0, 0, positions[prefix + "last_over_max"]]),
            0.5,
        )
        self.assertAlmostEqual(
            float(teacher_phi[0, 0, positions[prefix + "mean_over_max"]]),
            0.25,
        )
        uop_prefix = "urine_output.uop_sum_given_count."
        self.assertEqual(
            teacher_phi[0, 0, positions[uop_prefix + "positive_sum_gate"]],
            1.0,
        )

        zero = {key: value.clone() for key, value in values.items()}
        zero["ned_joint_value_state"][:, :, ned] = 0.0
        zero["uop_sum_given_count"][:, :, uop] = 0.0
        zero_phi = standardize_primitive_trajectory(
            zero,
            teacher_masks,
            schema,
            scale,
        )
        self.assertFalse(torch.equal(teacher_phi, zero_phi))

    def test_score_kernels_have_known_two_member_values(self) -> None:
        samples = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
        truth = torch.tensor([1.0], dtype=torch.float64)
        self.assertAlmostEqual(empirical_energy_score(samples, truth), 0.5)
        self.assertAlmostEqual(empirical_crps(samples[:, 0], 1.0), 0.5)

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

    def test_vectorized_respiratory_matches_all_31_active_sets_exactly(self) -> None:
        from trauma_predict.modeling.multires_event_v2.emissions import sample_categorical
        from trauma_predict.training.multires_event_v2_loss import (
            _sample_respiratory_occupancy,
        )

        def reference_sampler(raw: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
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
        active = torch.ones(31, dtype=torch.bool)
        torch.manual_seed(551)
        reference = reference_sampler(raw, active)
        reference_rng_tail = torch.rand(16)
        torch.manual_seed(551)
        vectorized = _sample_respiratory_occupancy(raw, active)
        vectorized_rng_tail = torch.rand(16)
        self.assertTrue(torch.equal(vectorized, reference))
        self.assertTrue(torch.equal(vectorized_rng_tail, reference_rng_tail))
        closure = vectorized.sum(dim=-1).sub(4.0).abs()
        self.assertLessEqual(float(closure.max()), 1e-12)


class RelationV2ScaleFitterSafetyTest(unittest.TestCase):
    def test_uop_zero_count_null_sum_is_not_a_scale_observation(self) -> None:
        from trauma_predict.eval.multires_event_v2_scale import _collect_fit_values

        values = {"urine_output|uop_sum_given_count|log_positive_sum": []}
        _collect_fit_values(
            values,
            field="urine_output",
            process={"observation_count": 0, "sum": None},
            contract=_Contract(),
        )
        self.assertEqual(
            values["urine_output|uop_sum_given_count|log_positive_sum"],
            [],
        )
        with self.assertRaisesRegex(ValueError, "zero-count"):
            _collect_fit_values(
                values,
                field="urine_output",
                process={"observation_count": 0, "sum": 0.0},
                contract=_Contract(),
            )

    def test_fitter_deduplicates_windows_and_emits_38_positive_iqrs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset_manifest.json").write_text("{}\n", encoding="utf-8")
            contract = _Contract(root)
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
                    lab_payload,
                    sort_keys=True,
                    separators=(",", ":"),
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
            self.assertTrue(
                all(float(row["scale"]) > 0.0 for row in loaded["scales"].values())
            )


class RelationV2TeacherDecompositionSafetyTest(unittest.TestCase):
    def test_414_factor_teacher_decomposition_sums_by_registered_axes(self) -> None:
        registry = _registry()
        specs = expand_enabled_core_primitives(registry)
        self.assertEqual(len(specs), 414)
        primitive_log_prob = torch.full((2, len(specs)), -0.1)
        rows = _teacher_nll_decomposition_rows(
            {
                "primitive_log_prob": primitive_log_prob,
                "primitive_ids": tuple(spec.primitive_id for spec in specs),
                "per_sample_nll": -primitive_log_prob.sum(dim=-1),
            },
            registry,
            batch_size=2,
        )
        assert rows is not None
        self.assertEqual(len(rows[0]["by_block"]), 6)
        self.assertEqual(len(rows[0]["by_field"]), 29)
        self.assertAlmostEqual(sum(rows[0]["by_block"].values()), 41.4, places=4)
        self.assertAlmostEqual(sum(rows[0]["by_field"].values()), 41.4, places=4)
        self.assertAlmostEqual(
            sum(rows[0]["by_objective_branch"].values()),
            41.4,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
