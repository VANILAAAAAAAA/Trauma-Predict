from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

from trauma_predict.training.multires_event_v2 import (
    BEST_CHECKPOINT_SCHEMA,
    RUN_ARTIFACT_PATHS,
    SELECTED_MODEL_SCHEMA,
    MultiresEventV2Runtime,
    _bind_runtime_to_run_artifacts,
    _completed_training_result,
    _evaluation_contract_identity,
    _export_run,
    _load_v2_best_model,
    _materialize_run_artifacts,
    _save_v2_best_model,
)
from trauma_predict.training.observability import sha256_file, sha256_payload, utc_now
from tests.test_multires_event_v2_kaggle_route import optimizer_health_row


class MultiresEventV2CheckpointIdentityTest(unittest.TestCase):
    def test_two_rank_best_checkpoint_uses_one_collective_order(self) -> None:
        worker = (
            Path(__file__).resolve().parent
            / "helpers/multires_event_v2_best_checkpoint_worker.py"
        )
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "best-checkpoint"
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
                timeout=25,
                check=False,
            )
            elapsed = time.monotonic() - started
            combined = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 0, msg=combined)
            self.assertLess(elapsed, 20.0, msg=combined)
            self.assertNotIn("collective mismatch", combined.lower())
            self.assertNotIn("timed out", combined.lower())
            pointer = json.loads(
                (output_root / "best_checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(pointer["step"], 250)
            self.assertEqual(pointer["joint_nll_subject_macro"], 1.25)

    def test_rank_zero_best_checkpoint_writer_is_collective_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "trauma_predict.training.multires_event_v2._barrier",
            side_effect=AssertionError("rank-zero writer entered a collective"),
        ) as barrier:
            _save_v2_best_model(
                output_dir=Path(directory),
                model=torch.nn.Identity(),
                identity_hashes={"runtime": "identity"},
                step=1,
                metric=0.0,
            )
            barrier.assert_not_called()

    def test_best_checkpoint_load_is_bound_to_schema_step_identity_and_model_bytes(self) -> None:
        identity = {"runtime": "r", "matched_design": "m"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = torch.nn.Linear(2, 1)
            with torch.no_grad():
                source.weight.fill_(3.0)
                source.bias.fill_(4.0)
            with patch("trauma_predict.training.multires_event_v2._barrier"):
                _save_v2_best_model(
                    output_dir=root,
                    model=source,
                    identity_hashes=identity,
                    step=7,
                    metric=2.5,
                )
            pointer = json.loads((root / "best_checkpoint.json").read_text())
            self.assertEqual(pointer["schema_version"], BEST_CHECKPOINT_SCHEMA)
            self.assertEqual(pointer["path"], "best_checkpoint")

            target = torch.nn.Linear(2, 1)
            with torch.no_grad():
                target.weight.zero_()
                target.bias.zero_()
            with patch("trauma_predict.training.multires_event_v2._barrier"):
                selected = _load_v2_best_model(
                    root,
                    target,
                    torch.device("cpu"),
                    expected_identity_hashes=identity,
                    expected_best_step=7,
                )
            self.assertTrue(torch.equal(target.weight, source.weight))
            self.assertEqual(selected["schema_version"], SELECTED_MODEL_SCHEMA)
            self.assertEqual(selected["selected_checkpoint_step"], 7)
            self.assertEqual(
                selected["selected_checkpoint_model_sha256"], pointer["model_sha256"]
            )

            bad_pointer = dict(pointer, unexpected=True)
            (root / "best_checkpoint.json").write_text(json.dumps(bad_pointer))
            with (
                patch("trauma_predict.training.multires_event_v2._barrier"),
                self.assertRaisesRegex(ValueError, "schema fields"),
            ):
                _load_v2_best_model(
                    root,
                    target,
                    torch.device("cpu"),
                    expected_identity_hashes=identity,
                    expected_best_step=7,
                )

            (root / "best_checkpoint.json").write_text(json.dumps(pointer))
            with (
                patch("trauma_predict.training.multires_event_v2._barrier"),
                self.assertRaisesRegex(ValueError, "trainer best_step"),
            ):
                _load_v2_best_model(
                    root,
                    target,
                    torch.device("cpu"),
                    expected_identity_hashes=identity,
                    expected_best_step=8,
                )

            (root / "best_checkpoint/identity_hashes.json").write_text(
                json.dumps({"runtime": "different"})
            )
            with (
                patch("trauma_predict.training.multires_event_v2._barrier"),
                self.assertRaisesRegex(RuntimeError, "identity files disagree"),
            ):
                _load_v2_best_model(
                    root,
                    target,
                    torch.device("cpu"),
                    expected_identity_hashes=identity,
                    expected_best_step=7,
                )

            (root / "best_checkpoint/identity_hashes.json").write_text(
                json.dumps(identity)
            )
            (root / "best_checkpoint/model.pt").write_bytes(b"corrupt")
            with (
                patch("trauma_predict.training.multires_event_v2._barrier"),
                self.assertRaisesRegex(ValueError, "SHA-256"),
            ):
                _load_v2_best_model(
                    root,
                    target,
                    torch.device("cpu"),
                    expected_identity_hashes=identity,
                    expected_best_step=7,
                )

    def test_final_identity_separates_training_completion_from_selected_model(self) -> None:
        runtime = MultiresEventV2Runtime(
            train_loader=None,
            eval_loader=None,
            train_sampler=None,
            eval_sampler=None,
            train_dataset=None,
            eval_dataset=None,
            contract=None,
            normalization=None,
            identity={
                "dataset_id": "dataset",
                "contract_bundle_hash": "1" * 64,
                "process_contract_sha256": "2" * 64,
                "emission_contract_sha256": "3" * 64,
                "projection_contract_sha256": "4" * 64,
                "relation_contract_sha256": "e" * 64,
                "sidecar_schema_sha256": "f" * 64,
                "lab_scale_artifact_sha256": "5" * 64,
                "standardized_primitive_scale_sha256": "6" * 64,
                "input_normalization_sha256": "7" * 64,
                "promotion_metric_contract_sha256": "d" * 64,
            },
        )
        source = {
            "source_tree_sha256": "8" * 64,
            "git_commit": "9" * 40,
            "git_head_tree": "a" * 40,
        }
        run_identity = {
            "source_tree": source["source_tree_sha256"],
            "source_identity": sha256_payload(source),
            "git_commit": source["git_commit"],
            "git_head_tree": source["git_head_tree"],
            "matched_design": "b" * 64,
        }
        selected = {
            "schema_version": SELECTED_MODEL_SCHEMA,
            "selected_checkpoint_step": 7,
            "selected_checkpoint_model_sha256": "c" * 64,
        }
        result = _completed_training_result(
            {"global_step": 10, "best_step": 7, "best_metric": 2.0}, selected
        )
        self.assertEqual(result["training_completed_step"], 10)
        self.assertEqual(result["selected_checkpoint_step"], 7)
        final_identity = _evaluation_contract_identity(
            runtime,
            source_identity=source,
            identity_hashes=run_identity,
            selected_model_identity=selected,
        )
        self.assertEqual(final_identity["selected_checkpoint_step"], 7)
        self.assertEqual(final_identity["source_identity_sha256"], sha256_payload(source))
        with self.assertRaisesRegex(ValueError, "not bound"):
            _evaluation_contract_identity(
                runtime,
                source_identity=source,
                identity_hashes={**run_identity, "source_tree": "0" * 64},
                selected_model_identity=selected,
            )

    def test_external_inputs_are_atomically_copied_and_runtime_paths_are_portable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            run = root / "run"
            (repo / "configs").mkdir(parents=True)
            normalization = root / "generated-normalization.json"
            normalization.write_text('{"normalization": true}\n')
            lab = repo / "configs/lab.json"
            phi = repo / "configs/phi.json"
            promotion = repo / "configs/promotion.json"
            train_path = repo / "configs/train.yaml"
            dataset_path = repo / "configs/dataset.yaml"
            model_path = repo / "configs/model.yaml"
            for path, text in (
                (lab, "lab\n"),
                (phi, "phi\n"),
                (promotion, "promotion\n"),
                (train_path, "train: true\n"),
                (dataset_path, "dataset: true\n"),
                (model_path, "model: true\n"),
            ):
                path.write_text(text)
            runtime = MultiresEventV2Runtime(
                train_loader=None,
                eval_loader=None,
                train_sampler=None,
                eval_sampler=None,
                train_dataset=None,
                eval_dataset=None,
                contract=None,
                normalization=SimpleNamespace(),
                identity={
                    "normalization_artifact": str(normalization),
                    "normalization_artifact_sha256": sha256_file(normalization),
                    "lab_scale_artifact_sha256": "1" * 64,
                    "standardized_primitive_scale_sha256": "2" * 64,
                    "promotion_metric_contract_sha256": sha256_file(promotion),
                },
            )
            train = {
                "lab_scale_artifact": "configs/lab.json",
                "standardized_primitive_scale_artifact": "configs/phi.json",
                "promotion_metric_contract": "configs/promotion.json",
                "training": {"required_world_size": 2},
            }
            semantic_runtime = {"fixture": "stable-runtime"}
            runtime_environment = {
                "schema_version": (
                    "trauma_predict.multires_event_v2_runtime_environment.v1"
                ),
                "captured_at": "2026-07-13T00:00:00Z",
                "semantic_runtime_identity": semantic_runtime,
                "semantic_runtime_identity_sha256": sha256_payload(semantic_runtime),
                "diagnostics": {},
            }
            with patch(
                "trauma_predict.training.multires_event_v2._runtime_environment_artifact",
                return_value=runtime_environment,
            ):
                _materialize_run_artifacts(
                    run,
                    runtime=runtime,
                    train=train,
                    repo_root=repo,
                    train_path=train_path,
                    dataset_path=dataset_path,
                    model_path=model_path,
                )
                portable = _bind_runtime_to_run_artifacts(run, runtime)
                self.assertEqual(
                    portable.identity["normalization_artifact"],
                    RUN_ARTIFACT_PATHS["input_normalization"],
                )
                self.assertEqual(
                    portable.identity["semantic_runtime_identity_sha256"],
                    sha256_payload(semantic_runtime),
                )
                for relative in RUN_ARTIFACT_PATHS.values():
                    self.assertTrue((run / relative).is_file())
                (run / RUN_ARTIFACT_PATHS["lab_affine_scale"]).write_text("changed\n")
                with self.assertRaisesRegex(RuntimeError, "conflicts"):
                    _materialize_run_artifacts(
                        run,
                        runtime=runtime,
                        train=train,
                        repo_root=repo,
                        train_path=train_path,
                        dataset_path=dataset_path,
                        model_path=model_path,
                    )

    def test_export_uses_run_relative_teacher_and_free_pointers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher_rows = root / "val_per_anchor_joint_nll.jsonl"
            teacher_rows.write_text("{}\n")
            free = root / "free_running"
            free.mkdir()
            for name in (
                "sample_schema.json",
                "audit_trajectory_samples.rank00000.jsonl.gz",
                "per_anchor_scores.rank00000.jsonl",
            ):
                (free / name).write_bytes(b"x")
            free_manifest = {
                "schema_version": "free.v1",
                "evaluation": {"sample_schema_path": "sample_schema.json"},
                "per_anchor_score_shards": [
                    {
                        "audit_trajectory_sample_path": (
                            "audit_trajectory_samples.rank00000.jsonl.gz"
                        ),
                        "per_anchor_score_path": "per_anchor_scores.rank00000.jsonl",
                    }
                ],
            }
            (free / "manifest.json").write_text(json.dumps(free_manifest))
            (free / "evaluation.json").write_text("{}\n")
            model = torch.nn.Linear(2, 1)
            best_dir = root / "best_checkpoint"
            best_dir.mkdir()
            torch.save(model.state_dict(), best_dir / "model.pt")
            selected_sha = sha256_file(best_dir / "model.pt")
            (root / "artifacts").mkdir()
            (root / "artifacts/manifest.json").write_text(
                json.dumps({"schema_version": "unit.artifacts.v1"})
            )
            selected = {
                "schema_version": SELECTED_MODEL_SCHEMA,
                "selected_checkpoint_step": 7,
                "selected_checkpoint_model_sha256": selected_sha,
                "selected_checkpoint_path": "best_checkpoint/model.pt",
                "best_checkpoint_manifest_path": "best_checkpoint.json",
                "best_checkpoint_manifest_sha256": "e" * 64,
            }
            row_identity = {
                "source_tree_sha256": "1" * 64,
                "source_identity_sha256": "2" * 64,
                "git_commit": "3" * 40,
                "git_head_tree": "4" * 40,
                "matched_design_signature": "5" * 64,
                "selected_checkpoint_step": 7,
                "selected_checkpoint_model_sha256": selected_sha,
            }
            evaluation = {
                "step": 7,
                "joint_nll_subject_macro": 2.0,
                "per_anchor_output_path": str(teacher_rows),
                "per_anchor_output_sha256": sha256_file(teacher_rows),
                "identity": row_identity,
            }
            free_evaluation = {
                "step": 7,
                "anchors": 6309,
                "trajectories_per_anchor": 100,
                "coherence": {"rate": 1.0},
                "manifest_path": "manifest.json",
                "manifest_sha256": sha256_file(free / "manifest.json"),
                "sample_schema_path": "sample_schema.json",
                "shards": free_manifest["per_anchor_score_shards"],
                "identity": row_identity,
            }
            training = {
                "global_step": 10,
                "max_steps": 10,
                "best_step": 7,
                "scaler_skipped_steps": 0,
                "training_completed_step": 10,
                "selected_checkpoint_step": 7,
                "selected_checkpoint_model_sha256": selected_sha,
            }
            training_config = yaml.safe_load(
                (
                    Path(__file__).resolve().parents[1]
                    / "configs/train/t4x2_multires_event_v2_trajectory.yaml"
                ).read_text(encoding="utf-8")
            )["training"]
            training_config = dict(training_config, max_steps=10, warmup_steps=2)
            with (root / "metrics.jsonl").open("w", encoding="utf-8") as handle:
                for step in range(1, 11):
                    handle.write(json.dumps(optimizer_health_row(step, training_config)))
                    handle.write("\n")
            _export_run(
                root,
                model,
                {
                    "mode": "trajectory",
                    "run_name": "unit",
                    "training": training_config,
                },
                {"runtime": "identity"},
                training,
                evaluation,
                free_evaluation,
                selected,
            )
            teacher_export = json.loads((root / "evaluation.json").read_text())
            self.assertEqual(
                teacher_export["per_anchor_output_path"],
                "val_per_anchor_joint_nll.jsonl",
            )
            run_manifest = json.loads((root / "run_manifest.json").read_text())
            self.assertEqual(run_manifest["training"]["training_completed_step"], 10)
            self.assertEqual(
                run_manifest["free_running_evaluation"]["manifest_path"],
                "free_running/manifest.json",
            )
            model_manifest = json.loads(
                (root / "final_model/model_manifest.json").read_text()
            )
            self.assertEqual(model_manifest["model_file"], "final_model/model.pt")
            self.assertEqual(model_manifest["selected_checkpoint_step"], 7)
            self.assertEqual(model_manifest["model_sha256"], selected_sha)
            self.assertEqual(
                run_manifest["final_model"]["sha256"], selected_sha
            )
            self.assertTrue((root / "SUCCESS").is_file())
            optimizer_summary = json.loads(
                (root / "optimizer_health_summary.json").read_text()
            )
            self.assertEqual(optimizer_summary["canonical_steps"], 10)
            success = json.loads((root / "SUCCESS").read_text())
            self.assertEqual(
                success["optimizer_health_summary_sha256"],
                sha256_file(root / "optimizer_health_summary.json"),
            )

    def test_utc_now_is_python310_compatible_utc_text(self) -> None:
        self.assertRegex(utc_now(), r"^\d{4}-\d{2}-\d{2}T.*Z$")


if __name__ == "__main__":
    unittest.main()
