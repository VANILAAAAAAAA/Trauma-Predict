from __future__ import annotations

import copy
import hashlib
import gzip
import importlib.util
import inspect
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from trauma_predict.training.multires_event_v2 import (
    OPTIMIZER_CONTRACT_VERSION,
    RAW_JOINT_NLL_REDUCTION,
    summarize_optimizer_health_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = REPO_ROOT / "notebooks/kaggle/run_multires_event_v2.py"
ENTRYPOINT_PATH = REPO_ROOT / "notebooks/kaggle/train_multires_event_v2.py"
NOTEBOOK_PATH = REPO_ROOT / "notebooks/kaggle/train_multires_event_v2.ipynb"
VERIFICATION_NOTEBOOK_PATH = (
    REPO_ROOT / "notebooks/kaggle/verify_multires_event_v2.ipynb"
)


def load_launcher():
    spec = importlib.util.spec_from_file_location("run_multires_event_v2", LAUNCHER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_entrypoint():
    spec = importlib.util.spec_from_file_location("train_multires_event_v2", ENTRYPOINT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_sha256(payload) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def optimizer_health_row(step: int, training: dict) -> dict:
    warmup = int(training["warmup_steps"])
    total = int(training["max_steps"])
    base_lr = float(training["learning_rate"])
    completed = step - 1
    if completed < warmup:
        factor = float(completed + 1) / float(warmup)
    else:
        factor = float(max(0, total - completed)) / float(max(1, total - warmup))
    learning_rate = base_lr * factor
    configuration = {
        "optimizer": "AdamW",
        "parameter_group_count": 1,
        "base_learning_rate": base_lr,
        "current_learning_rate": learning_rate,
        "weight_decay": float(training["weight_decay"]),
        "adamw_betas": list(training["adamw_betas"]),
        "adamw_eps": float(training["adamw_eps"]),
        "adamw_amsgrad": False,
        "adamw_maximize": False,
        "adamw_foreach": False,
        "adamw_fused": False,
    }
    return {
        "event": "v2_optimizer_health",
        "created_at": f"2026-07-13T00:00:{step % 60:02d}Z",
        "step": step,
        "local_anchors": 32,
        "world_size": 2,
        "global_anchors": 64,
        "optimizer_contract_version": OPTIMIZER_CONTRACT_VERSION,
        "loss_reduction": RAW_JOINT_NLL_REDUCTION,
        "expected_optimizer_step": step,
        "observed_optimizer_step_min": float(step),
        "observed_optimizer_step_max": float(step),
        "expected_learning_rate_used": learning_rate,
        "learning_rate_used": learning_rate,
        "optimizer_audit_wall_seconds": 0.1,
        "gradient_health": {
            "optimizer_contract_version": OPTIMIZER_CONTRACT_VERSION,
            "trainable_parameter_tensors": 2,
            "gradient_tensors": 2,
            "missing_gradient_tensors": 0,
            "all_gradients_finite": True,
            "global_l2_norm": 0.5,
            "global_l2_positive": True,
            "gradient_clipping": "disabled",
            "gradient_modified_after_unscale": False,
            "probe_parameter": "decoder.weight",
            "probe_flat_index": 0,
            "probe_gradient_abs": 0.25,
            "audit_wall_seconds": 0.04,
        },
        "optimizer_state_health": {
            "optimizer_contract_version": OPTIMIZER_CONTRACT_VERSION,
            "trainable_parameter_tensors": 2,
            "optimizer_state_entries": 2,
            "state_complete": True,
            "expected_optimizer_step": step,
            "observed_optimizer_step_min": float(step),
            "observed_optimizer_step_max": float(step),
            "state_steps_complete_equal_expected": True,
            "parameters_finite": True,
            "exp_avg_finite": True,
            "exp_avg_sq_finite": True,
            "exp_avg_sq_nonnegative": True,
            "exp_avg_sq_minimum": 0.0,
            "probe_parameter": "decoder.weight",
            "probe_flat_index": 0,
            "probe_value_before": 0.1,
            "probe_value_after": 0.1,
            "probe_parameter_changed": False,
            "optimizer_updated": True,
            "audit_wall_seconds": 0.06,
            "optimizer_configuration": configuration,
        },
        "scaler_scale_before": 32.0,
        "scaler_scale_after": 32.0,
        "scaler_skipped_steps": 0,
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resign_success(run_dir: Path) -> None:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    health = manifest["optimizer_health_summary"]
    write_json(
        run_dir / "SUCCESS",
        {
            "schema_version": "trauma_predict.multires_event_v2_success.v1",
            "run_manifest_sha256": sha256(manifest_path),
            "optimizer_health_summary_sha256": health["sha256"],
            "metrics_jsonl_sha256": health["metrics_sha256"],
        },
    )


class MultiresEventV2KaggleRouteTest(unittest.TestCase):
    def test_optimizer_health_summary_is_resume_aware_but_validates_every_raw_row(self) -> None:
        training = yaml.safe_load(
            (REPO_ROOT / "configs/train/t4x2_multires_event_v2_smoke.yaml").read_text()
        )["training"]
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.jsonl"
            rows = [
                optimizer_health_row(1, training),
                optimizer_health_row(1, training),
                optimizer_health_row(2, training),
            ]
            metrics.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            summary = summarize_optimizer_health_metrics(metrics, training=training)
            self.assertEqual(summary["raw_health_rows"], 3)
            self.assertEqual(summary["canonical_steps"], 2)
            self.assertEqual(summary["replayed_rows"], 1)

            rows[0]["scaler_scale_after"] = 64.0
            metrics.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "contract failed"):
                summarize_optimizer_health_metrics(metrics, training=training)

    def promotion_validation(self, root: Path, mode: str) -> dict[str, object]:
        root.mkdir(parents=True, exist_ok=True)
        promotion_path = root / "artifacts/promotion_metric_contract.json"
        promotion_path.parent.mkdir(parents=True, exist_ok=True)
        promotion_path.write_bytes(
            (REPO_ROOT / "configs/evaluation/multires_event_v2_promotion_v2.json").read_bytes()
        )
        source_identity = {
            "git_commit": "1" * 40,
            "git_head_tree": "2" * 40,
            "source_tree_sha256": "3" * 64,
        }
        contract_identity = {
            "dataset_id": "fixture-target",
            "contract_bundle_hash": "4" * 64,
            "relation_contract_sha256": "b" * 64,
            "sidecar_schema_sha256": "c" * 64,
            "objective_contract_sha256": "5" * 64,
        }
        return {
            "stage": mode,
            "mode": mode,
            "run_dir": str(root.resolve()),
            "run_manifest": str((root / "run_manifest.json").resolve()),
            "matched_design_signature": "6" * 64,
            "parameter_count": 123456,
            "input_normalization_sha256": "7" * 64,
            "source_identity": source_identity,
            "source_identity_sha256": "8" * 64,
            "contract_identity": contract_identity,
            "contract_identity_sha256": "9" * 64,
            "relation_contract_sha256": contract_identity[
                "relation_contract_sha256"
            ],
            "sidecar_schema_sha256": contract_identity[
                "sidecar_schema_sha256"
            ],
            "semantic_runtime_identity_sha256": "a" * 64,
            "run_manifest_sha256": "d" * 64,
            "optimizer_health_summary_sha256": "e" * 64,
            "metrics_sha256": "f" * 64,
        }

    def completed_run_fixture(self, root: Path, launcher, mode: str = "block") -> Path:
        run_dir = root / f"t4x2_multires_event_v2_{mode}"
        run_dir.mkdir(parents=True)
        train_config = yaml.safe_load(
            (REPO_ROOT / f"configs/train/t4x2_multires_event_v2_{mode}.yaml").read_text()
        )
        train_config["outputs"] = {
            "output_dir": str(run_dir),
            "metrics_jsonl": str(run_dir / "metrics.jsonl"),
        }
        dataset_config = yaml.safe_load(
            (REPO_ROOT / "configs/dataset/multires_event_v2_c4.yaml").read_text()
        )
        model_config = yaml.safe_load(
            (REPO_ROOT / "configs/model/multires_event_v2.yaml").read_text()
        )
        semantic_runtime = {
            "schema_version": "trauma_predict.multires_event_v2_semantic_runtime.v1",
            "python": {"implementation": "CPython", "version": "3.11.0"},
            "torch": "2.7.1+cu126",
            "cuda_runtime": "12.6",
            "cudnn": 90501,
            "devices": [
                {"name": "Tesla T4", "compute_capability": [7, 5]},
                {"name": "Tesla T4", "compute_capability": [7, 5]},
            ],
            "world_size": 2,
            "precision": "fp16",
            "requirements_sha256": "a" * 64,
            "lock_sha256": "b" * 64,
            "dependency_versions": {
                "numpy": "2.2.6",
                "PyYAML": "6.0.2",
                "safetensors": "0.5.3",
            },
        }
        semantic_runtime_sha = payload_sha256(semantic_runtime)
        runtime_environment = {
            "schema_version": "trauma_predict.multires_event_v2_runtime_environment.v1",
            "captured_at": "2026-07-13T00:00:00Z",
            "semantic_runtime_identity": semantic_runtime,
            "semantic_runtime_identity_sha256": semantic_runtime_sha,
            "diagnostics": {},
        }
        artifact_contents = {
            "input_normalization": b'{"normalization":"fixture"}\n',
            "lab_affine_scale": b'{"lab_scale":"fixture"}\n',
            "standardized_primitive_scale": b'{"phi_scale":"fixture"}\n',
            "promotion_metric_contract": (
                REPO_ROOT / launcher.EXPECTED_PROMOTION_METRIC_CONTRACT
            ).read_bytes(),
            "runtime_environment": (
                json.dumps(runtime_environment, sort_keys=True) + "\n"
            ).encode(),
            "train_config": yaml.safe_dump(train_config, sort_keys=False).encode(),
            "dataset_config": yaml.safe_dump(dataset_config, sort_keys=False).encode(),
            "model_config": yaml.safe_dump(model_config, sort_keys=False).encode(),
        }
        artifact_entries = {}
        for name, relative in launcher.PORTABLE_RUN_ARTIFACTS.items():
            path = run_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(artifact_contents[name])
            artifact_entries[name] = {"path": relative, "file_sha256": sha256(path)}
        artifact_entries["lab_affine_scale"]["semantic_sha256"] = (
            launcher.EXPECTED_LAB_SCALE_ARTIFACT_SHA256
        )
        artifact_entries["standardized_primitive_scale"]["semantic_sha256"] = (
            launcher.EXPECTED_STANDARDIZED_PRIMITIVE_SCALE_SHA256
        )
        artifact_entries["promotion_metric_contract"]["semantic_sha256"] = (
            launcher.EXPECTED_PROMOTION_METRIC_CONTRACT_SHA256
        )
        artifact_entries["runtime_environment"]["semantic_sha256"] = (
            semantic_runtime_sha
        )
        write_json(
            run_dir / "artifacts/manifest.json",
            {
                "schema_version": "trauma_predict.multires_event_v2_run_artifacts.v1",
                "artifacts": artifact_entries,
            },
        )
        resolved = {"train": train_config, "dataset": dataset_config, "model": model_config}
        write_json(run_dir / "resolved_config.json", resolved)
        matched_train = copy.deepcopy(train_config)
        matched_train.pop("mode")
        matched_train.pop("run_name")
        matched_train.pop("outputs")
        matched_signature = payload_sha256(
            {"train": matched_train, "dataset": dataset_config, "model": model_config}
        )

        normalization_sha = artifact_entries["input_normalization"]["file_sha256"]
        identity = {
            "base_dataset_id": launcher.BASE_AUTHORITY["dataset_id"],
            "base_fingerprint": launcher.BASE_AUTHORITY["fingerprint"],
            "base_dataset_manifest_sha256": launcher.BASE_AUTHORITY["manifest_sha256"],
            "target_dataset_id": launcher.TARGET_AUTHORITY["dataset_id"],
            "dataset_id": launcher.TARGET_AUTHORITY["dataset_id"],
            "target_dataset_manifest_sha256": launcher.TARGET_AUTHORITY["manifest_sha256"],
            "contract_bundle_hash": launcher.TARGET_AUTHORITY["contract_bundle_hash"],
            "process_contract_sha256": launcher.TARGET_AUTHORITY[
                "process_contract_sha256"
            ],
            "emission_contract_sha256": launcher.TARGET_AUTHORITY[
                "emission_contract_sha256"
            ],
            "projection_contract_sha256": launcher.TARGET_AUTHORITY[
                "projection_contract_sha256"
            ],
            "relation_contract_sha256": launcher.TARGET_AUTHORITY[
                "relation_contract_sha256"
            ],
            "sidecar_schema_sha256": launcher.TARGET_AUTHORITY[
                "sidecar_schema_sha256"
            ],
            "counts": dict(launcher.EXPECTED_COUNTS),
            "phase": "interval",
            "train_subjects": 1,
            "validation_subjects": 505,
            "enabled_factors": 414,
            "normalization_artifact": launcher.PORTABLE_RUN_ARTIFACTS[
                "input_normalization"
            ],
            "normalization_artifact_sha256": normalization_sha,
            "normalization_artifact_file_sha256": normalization_sha,
            "input_normalization_sha256": normalization_sha,
            "lab_scale_artifact": launcher.PORTABLE_RUN_ARTIFACTS["lab_affine_scale"],
            "lab_scale_artifact_hash": launcher.EXPECTED_LAB_SCALE_ARTIFACT_SHA256,
            "lab_scale_artifact_sha256": launcher.EXPECTED_LAB_SCALE_ARTIFACT_SHA256,
            "lab_scale_artifact_file_sha256": artifact_entries["lab_affine_scale"][
                "file_sha256"
            ],
            "standardized_primitive_scale_artifact": launcher.PORTABLE_RUN_ARTIFACTS[
                "standardized_primitive_scale"
            ],
            "standardized_primitive_scale_artifact_hash": (
                launcher.EXPECTED_STANDARDIZED_PRIMITIVE_SCALE_SHA256
            ),
            "standardized_primitive_scale_sha256": (
                launcher.EXPECTED_STANDARDIZED_PRIMITIVE_SCALE_SHA256
            ),
            "standardized_primitive_scale_artifact_file_sha256": artifact_entries[
                "standardized_primitive_scale"
            ]["file_sha256"],
            "promotion_metric_contract": launcher.PORTABLE_RUN_ARTIFACTS[
                "promotion_metric_contract"
            ],
            "promotion_metric_contract_sha256": (
                launcher.EXPECTED_PROMOTION_METRIC_CONTRACT_SHA256
            ),
            "promotion_metric_contract_file_sha256": artifact_entries[
                "promotion_metric_contract"
            ]["file_sha256"],
            "runtime_environment_artifact": launcher.PORTABLE_RUN_ARTIFACTS[
                "runtime_environment"
            ],
            "runtime_environment_artifact_file_sha256": artifact_entries[
                "runtime_environment"
            ]["file_sha256"],
            "semantic_runtime_identity_sha256": semantic_runtime_sha,
        }
        write_json(run_dir / "dataset_identity.json", identity)
        write_json(
            run_dir / "normalization_identity.json",
            {
                "artifact_path": launcher.PORTABLE_RUN_ARTIFACTS["input_normalization"],
                "artifact_sha256": normalization_sha,
                "artifact_file_sha256": normalization_sha,
            },
        )
        objective_contract = {
            "objective": train_config["objective"],
            "process_contract": {},
            "contract_bundle_hash": launcher.TARGET_AUTHORITY["contract_bundle_hash"],
            "contract_hashes": {
                "process": launcher.TARGET_AUTHORITY["process_contract_sha256"],
                "emission": launcher.TARGET_AUTHORITY["emission_contract_sha256"],
                "projection": launcher.TARGET_AUTHORITY[
                    "projection_contract_sha256"
                ],
                "relation": launcher.TARGET_AUTHORITY["relation_contract_sha256"],
                "sidecar_schema": launcher.TARGET_AUTHORITY[
                    "sidecar_schema_sha256"
                ],
            },
        }
        write_json(run_dir / "objective_contract.json", objective_contract)

        git_commit = "1" * 40
        git_tree = "2" * 40
        source_identity = {
            "schema_version": "trauma_predict.multires_event_v2_source_identity.v1",
            "git_commit": git_commit,
            "git_head_tree": git_tree,
            "git_clean": True,
            "git_status_sha256": "3" * 64,
            "source_tree_sha256": "4" * 64,
            "source_file_count": 1,
            "source_files": {"fixture.py": "5" * 64},
        }
        write_json(run_dir / "source_identity.json", source_identity)
        identity_hashes = {
            "train_config": payload_sha256(train_config),
            "dataset_config": payload_sha256(dataset_config),
            "model_config": payload_sha256(model_config),
            "runtime": payload_sha256(identity),
            "contract_bundle": launcher.TARGET_AUTHORITY["contract_bundle_hash"],
            "normalization": normalization_sha,
            "source_tree": source_identity["source_tree_sha256"],
            "source_identity": payload_sha256(source_identity),
            "git_commit": git_commit,
            "git_head_tree": git_tree,
            "matched_design": matched_signature,
            "semantic_runtime": semantic_runtime_sha,
        }
        write_json(run_dir / "identity_hashes.json", identity_hashes)
        write_json(
            run_dir / "model_identity.json",
            {
                "mode": mode,
                "parameter_count": 123456,
                "matched_design_signature": matched_signature,
            },
        )

        best_model = run_dir / "best_checkpoint/model.pt"
        best_model.parent.mkdir(parents=True)
        best_model.write_bytes(b"fixture-best-model")
        write_json(run_dir / "best_checkpoint/identity_hashes.json", identity_hashes)
        best_pointer = {
            "schema_version": "trauma_predict.multires_event_v2_best_checkpoint.v1",
            "updated_at": "2026-07-13T00:00:00Z",
            "step": 250,
            "joint_nll_subject_macro": 1.0,
            "path": "best_checkpoint",
            "model_sha256": sha256(best_model),
            "identity_hashes": identity_hashes,
        }
        write_json(run_dir / "best_checkpoint.json", best_pointer)
        selected_model_identity = {
            "schema_version": "trauma_predict.multires_event_v2_selected_model.v1",
            "selected_checkpoint_step": 250,
            "selected_checkpoint_model_sha256": best_pointer["model_sha256"],
            "selected_checkpoint_path": "best_checkpoint/model.pt",
            "best_checkpoint_manifest_path": "best_checkpoint.json",
            "best_checkpoint_manifest_sha256": sha256(run_dir / "best_checkpoint.json"),
        }
        evaluation_identity = {
            "dataset_id": launcher.TARGET_AUTHORITY["dataset_id"],
            "contract_bundle_hash": launcher.TARGET_AUTHORITY["contract_bundle_hash"],
            "process_contract_sha256": launcher.TARGET_AUTHORITY[
                "process_contract_sha256"
            ],
            "emission_contract_sha256": launcher.TARGET_AUTHORITY[
                "emission_contract_sha256"
            ],
            "projection_contract_sha256": launcher.TARGET_AUTHORITY[
                "projection_contract_sha256"
            ],
            "relation_contract_sha256": launcher.TARGET_AUTHORITY[
                "relation_contract_sha256"
            ],
            "sidecar_schema_sha256": launcher.TARGET_AUTHORITY[
                "sidecar_schema_sha256"
            ],
            "lab_scale_artifact_sha256": launcher.EXPECTED_LAB_SCALE_ARTIFACT_SHA256,
            "standardized_primitive_scale_sha256": (
                launcher.EXPECTED_STANDARDIZED_PRIMITIVE_SCALE_SHA256
            ),
            "input_normalization_sha256": normalization_sha,
            "promotion_metric_contract_sha256": (
                launcher.EXPECTED_PROMOTION_METRIC_CONTRACT_SHA256
            ),
            "semantic_runtime_identity_sha256": semantic_runtime_sha,
            "source_tree_sha256": source_identity["source_tree_sha256"],
            "source_identity_sha256": payload_sha256(source_identity),
            "git_commit": git_commit,
            "git_head_tree": git_tree,
            "matched_design_signature": matched_signature,
            "selected_checkpoint_step": 250,
            "selected_checkpoint_model_sha256": best_pointer["model_sha256"],
        }
        teacher_rows = run_dir / "val_per_anchor_joint_nll.jsonl"
        teacher_rows.write_text('{"sample_id":"fixture"}\n', encoding="utf-8")
        evaluation = {
            "phase": "final",
            "mode": mode,
            "step": 250,
            "samples": launcher.EXPECTED_COUNTS["val"],
            "subjects": 505,
            "primitive_factors_per_anchor": 414,
            "active_target_denominator": False,
            "deterministic_projection_loss": False,
            "joint_nll_subject_macro": 1.0,
            "per_anchor_output_sha256": sha256(teacher_rows),
            "identity": evaluation_identity,
        }
        write_json(run_dir / "evaluation.json", evaluation)
        free_root = run_dir / "free_running"
        free_root.mkdir()
        write_json(free_root / "sample_schema.json", {"schema_version": "fixture"})
        free_shards = []
        for rank, anchors in ((0, 3155), (1, 3154)):
            score_path = free_root / f"per_anchor_scores.rank{rank:05d}.jsonl"
            audit_path = free_root / f"primitive_samples.rank{rank:05d}.jsonl"
            progress_path = free_root / f"progress.rank{rank:05d}.jsonl"
            score_path.write_text('{"sample_id":"fixture"}\n', encoding="utf-8")
            audit_path.write_text('{"sample_id":"fixture"}\n', encoding="utf-8")
            progress_path.write_text('{"event":"progress"}\n', encoding="utf-8")
            free_shards.append(
                {
                    "rank": rank,
                    "anchors": anchors,
                    "retained_audit_trajectories": anchors,
                    "per_anchor_score_path": score_path.name,
                    "per_anchor_score_sha256": sha256(score_path),
                    "audit_trajectory_sample_path": audit_path.name,
                    "audit_trajectory_sample_sha256": sha256(audit_path),
                    "progress_metrics_path": progress_path.name,
                    "progress_metrics_sha256": sha256(progress_path),
                }
            )
        write_json(
            free_root / "manifest.json", {"per_anchor_score_shards": free_shards}
        )
        free_evaluation = {
            "mode": mode,
            "step": 250,
            "anchors": launcher.EXPECTED_COUNTS["val"],
            "subjects": 505,
            "trajectories_per_anchor": 100,
            "field_macro_lag1_variogram_score_p0_5": {
                "subject_macro": 1.0
            },
            "relation_edge_macro_variogram_score_p0_5": {
                "subject_macro": 1.0
            },
            "marginal_value_crps": {"subject_macro": 1.0},
            "marginal_state_crps": {"subject_macro": 1.0},
            "coherence": {
                "rate": 1.0,
                "coherent_trajectories": launcher.EXPECTED_COUNTS["val"] * 100,
            },
            "sample_schema_path": "sample_schema.json",
            "sample_schema_sha256": sha256(free_root / "sample_schema.json"),
            "identity": evaluation_identity,
        }
        write_json(free_root / "evaluation.json", free_evaluation)

        final_model = run_dir / "final_model/model.pt"
        final_model.parent.mkdir()
        final_model.write_bytes(best_model.read_bytes())
        completed_training = {
            "global_step": 4000,
            "max_steps": 4000,
            "best_step": 250,
            "scaler_skipped_steps": 0,
            "training_completed_step": 4000,
            "selected_checkpoint_step": 250,
            "selected_checkpoint_model_sha256": best_pointer["model_sha256"],
        }
        write_json(
            run_dir / "final_model/model_manifest.json",
            {
                "mode": mode,
                "model_file": "final_model/model.pt",
                "model_sha256": sha256(final_model),
                "selected_checkpoint_step": 250,
                "selected_checkpoint_model_sha256": best_pointer["model_sha256"],
                "training_completed_step": 4000,
                "identity_hashes": identity_hashes,
            },
        )
        free_pointer = {
            "path": "free_running/evaluation.json",
            "sha256": sha256(free_root / "evaluation.json"),
            "manifest_path": "free_running/manifest.json",
            "manifest_sha256": sha256(free_root / "manifest.json"),
            "trajectories_per_anchor": 100,
            "coherence_rate": 1.0,
        }
        metrics_path = run_dir / "metrics.jsonl"
        with metrics_path.open("w", encoding="utf-8") as handle:
            for step in range(1, 4001):
                handle.write(json.dumps(optimizer_health_row(step, train_config["training"])))
                handle.write("\n")
        optimizer_summary = summarize_optimizer_health_metrics(
            metrics_path, training=train_config["training"]
        )
        write_json(run_dir / "optimizer_health_summary.json", optimizer_summary)
        optimizer_health_pointer = {
            "path": "optimizer_health_summary.json",
            "sha256": sha256(run_dir / "optimizer_health_summary.json"),
            "metrics_path": "metrics.jsonl",
            "metrics_sha256": sha256(metrics_path),
        }
        run_manifest = {
            "schema_version": "trauma_predict.multires_event_v2_run_manifest.v1",
            "status": "SUCCEEDED",
            "route": "multires_event_v2_m4_trajectory",
            "run_name": train_config["run_name"],
            "mode": mode,
            "training": completed_training,
            "evaluation": evaluation,
            "free_running_evaluation": free_pointer,
            "selected_model_identity": selected_model_identity,
            "final_model": {
                "path": "final_model/model.pt",
                "sha256": sha256(final_model),
                "manifest_path": "final_model/model_manifest.json",
                "manifest_sha256": sha256(run_dir / "final_model/model_manifest.json"),
            },
            "artifact_manifest": {
                "path": "artifacts/manifest.json",
                "sha256": sha256(run_dir / "artifacts/manifest.json"),
            },
            "optimizer_health_summary": optimizer_health_pointer,
            "identity_hashes": identity_hashes,
        }
        write_json(run_dir / "run_manifest.json", run_manifest)
        write_json(
            run_dir / "SUCCESS",
            {
                "schema_version": "trauma_predict.multires_event_v2_success.v1",
                "run_manifest_sha256": sha256(run_dir / "run_manifest.json"),
                "optimizer_health_summary_sha256": optimizer_health_pointer["sha256"],
                "metrics_jsonl_sha256": optimizer_health_pointer["metrics_sha256"],
            },
        )
        return run_dir

    def capacity_report_fixture(self, root: Path, launcher, mode: str = "block") -> dict:
        root.mkdir(parents=True, exist_ok=True)
        rank_canary_root = root / "ddp_rank_artifact_canary"
        rank_canary_root.mkdir()
        rank_artifacts = []
        for rank in (0, 1):
            progress_path = rank_canary_root / f"progress.rank{rank:05d}.jsonl"
            progress_path.write_text(
                json.dumps(
                    {
                        "event": "v2_free_running_rank_progress",
                        "rank": rank,
                        "mode": mode,
                        "completed_anchors": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rank_artifacts.append(
                {
                    "rank": rank,
                    "path": progress_path.name,
                    "sha256": sha256(progress_path),
                    "rows": 1,
                }
            )
        rank_canary_manifest = rank_canary_root / "manifest.json"
        rank_canary_payload = {
            "schema_version": (
                "trauma_predict.multires_event_v2_rank_artifact_preflight.v1"
            ),
            "created_at": "2026-07-13T00:00:00Z",
            "status": "PASSED",
            "mode": mode,
            "world_size": 2,
            "rank_artifacts": rank_artifacts,
        }
        write_json(rank_canary_manifest, rank_canary_payload)
        semantic_canary_root = root / "ddp_semantic_canary"
        semantic_canary_root.mkdir()
        sample_schema = semantic_canary_root / "sample_schema.json"
        write_json(
            sample_schema,
            {"schema_version": "trauma_predict.multires_event_v2_sample_export.v2"},
        )
        semantic_shards = []
        for rank in (0, 1):
            audit_path = (
                semantic_canary_root
                / f"audit_trajectory_samples.rank{rank:05d}.jsonl.gz"
            )
            with gzip.open(audit_path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({"rank": rank, "trajectory_index": 0}) + "\n")
            score_path = semantic_canary_root / f"per_anchor_scores.rank{rank:05d}.jsonl"
            score_path.write_text(
                json.dumps({"sample_id": f"sample-{rank}", "mode": mode}) + "\n",
                encoding="utf-8",
            )
            progress_path = semantic_canary_root / f"progress.rank{rank:05d}.jsonl"
            progress_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "event": "v2_free_running_rank_progress",
                            "rank": rank,
                            "mode": mode,
                            "completed_anchors": completed,
                        }
                    )
                    + "\n"
                    for completed in (0, 1)
                ),
                encoding="utf-8",
            )
            semantic_shards.append(
                {
                    "rank": rank,
                    "anchors": 1,
                    "audit_trajectory_sample_path": audit_path.name,
                    "audit_trajectory_sample_sha256": sha256(audit_path),
                    "retained_audit_trajectories": 1,
                    "per_anchor_score_path": score_path.name,
                    "per_anchor_score_sha256": sha256(score_path),
                    "progress_metrics_path": progress_path.name,
                    "progress_metrics_sha256": sha256(progress_path),
                }
            )
        semantic_evaluation = {
            "mode": mode,
            "step": 0,
            "anchors": 2,
            "trajectories_per_anchor": (
                launcher.CAPACITY_SEMANTIC_CANARY_TRAJECTORIES_PER_ANCHOR
            ),
            "coherence": {
                "rate": 1.0,
                "coherent_trajectories": 200,
                "total_trajectories": 200,
            },
            "sample_schema_path": sample_schema.name,
            "sample_schema_sha256": sha256(sample_schema),
            "shards": semantic_shards,
        }
        semantic_canary_manifest = semantic_canary_root / "manifest.json"
        write_json(
            semantic_canary_manifest,
            {
                "schema_version": (
                    "trauma_predict.multires_event_v2_free_running_manifest.v1"
                ),
                "created_at": "2026-07-13T00:00:00Z",
                "evaluation": semantic_evaluation,
                "per_anchor_score_shards": semantic_shards,
            },
        )
        write_json(
            semantic_canary_root / "evaluation.json",
            {
                **semantic_evaluation,
                "manifest_path": semantic_canary_manifest.name,
                "manifest_sha256": sha256(semantic_canary_manifest),
            },
        )
        # Deliberately project a long hosted run: duration is recorded for
        # planning and is not a technical capacity failure.
        optimizer_seconds = 11.0
        teacher_seconds_per_anchor = 0.1
        free_seconds_per_anchor = 0.2
        components = {
            "optimizer": optimizer_seconds * 4000,
            "teacher_forced": teacher_seconds_per_anchor * 6309 * 17,
            "free_running": free_seconds_per_anchor * 6309,
        }
        projected = sum(components.values())
        projected_background = 10.0 + 30.0 + projected
        sample_set_sha = "7" * 64
        checkpoint_path = (
            root
            / "checkpoint_canary"
            / "checkpoints"
            / "checkpoint-00000002"
        )
        checkpoint_path.mkdir(parents=True)
        checkpoint_files = (
            "identity_hashes.json",
            "model.pt",
            "optimizer.pt",
            "rng-rank-0000.pt",
            "rng-rank-0001.pt",
            "sampler-rank-0000.pt",
            "sampler-rank-0001.pt",
            "scaler.pt",
            "scheduler.pt",
            "trainer_state.json",
        )
        for name in checkpoint_files:
            (checkpoint_path / name).write_bytes(f"fixture:{name}\n".encode("utf-8"))
        checkpoint_manifest = {
            "schema_version": launcher.V2_CHECKPOINT_SCHEMA,
            "created_at": "2026-07-13T00:00:00Z",
            "global_step": 2,
            "world_size": 2,
            "identity_hashes": {"fixture": "9" * 64},
            "files": list(checkpoint_files),
            "sha256": {
                name: sha256(checkpoint_path / name) for name in checkpoint_files
            },
        }
        checkpoint_manifest_path = checkpoint_path / "checkpoint_manifest.json"
        write_json(checkpoint_manifest_path, checkpoint_manifest)
        report = {
            "schema_version": launcher.CAPACITY_PROBE_SCHEMA,
            "created_at": "2026-07-13T00:00:00Z",
            "status": "PASSED",
            "mode": mode,
            "report_path": str((root / "capacity_probe.json").resolve()),
            "contract": {
                "optimizer_steps": 2,
                "per_device_train_batch_size": 32,
                "world_size": 2,
                "precision": "fp16",
                "validation_selection": "persisted_val_manifest_prefix",
                "validation_anchors": 100,
                "trajectories_per_anchor": 100,
                "formal_validation_anchors": 6309,
                "formal_trajectories_per_anchor": 100,
            },
            "identity": {
                "dataset_id": launcher.TARGET_AUTHORITY["dataset_id"],
                "contract_bundle_hash": launcher.TARGET_AUTHORITY[
                    "contract_bundle_hash"
                ],
                "relation_contract_sha256": launcher.TARGET_AUTHORITY[
                    "relation_contract_sha256"
                ],
                "sidecar_schema_sha256": launcher.TARGET_AUTHORITY[
                    "sidecar_schema_sha256"
                ],
                "input_normalization_sha256": "6" * 64,
                "promotion_metric_contract_sha256": (
                    launcher.EXPECTED_PROMOTION_METRIC_CONTRACT_SHA256
                ),
                "first_100_sample_ids_sha256": "8" * 64,
                "first_100_sample_id_set_sha256": sample_set_sha,
            },
            "hardware": [
                {
                    "rank": rank,
                    "local_rank": rank,
                    "device_name": "Tesla T4",
                    "compute_capability": [7, 5],
                    "total_memory_bytes": 16_000_000_000,
                    "peak_allocated_bytes": 8_000_000_000,
                    "peak_reserved_bytes": 9_000_000_000,
                }
                for rank in (0, 1)
            ],
            "distributed_canaries": {
                "rank_artifact": {
                    **rank_canary_payload,
                    "manifest_path": str(rank_canary_manifest.resolve()),
                    "manifest_sha256": sha256(rank_canary_manifest),
                },
                "semantic_rollout": {
                    "status": "PASSED",
                    "anchors": 2,
                    "trajectories_per_anchor": (
                        launcher.CAPACITY_SEMANTIC_CANARY_TRAJECTORIES_PER_ANCHOR
                    ),
                    "world_size": 2,
                    "wall_seconds": 1.0,
                    "coherence_rate": 1.0,
                    "manifest_path": str(semantic_canary_manifest.resolve()),
                    "manifest_sha256": sha256(semantic_canary_manifest),
                },
            },
            "optimizer": {
                "optimizer_contract_version": launcher.OPTIMIZER_CONTRACT_VERSION,
                "loss_reduction": launcher.RAW_JOINT_NLL_REDUCTION,
                "gradient_clipping": "disabled",
                "configured_contract": launcher.EXPECTED_OPTIMIZER_CONTRACT,
                "steps": [
                    {
                        "event": "v2_optimizer_health",
                        "step": step,
                        "local_anchors": 32,
                        "world_size": 2,
                        "global_anchors": 64,
                        "wall_seconds": optimizer_seconds,
                        "joint_nll_anchor_mean": 1.0,
                        "optimizer_contract_version": launcher.OPTIMIZER_CONTRACT_VERSION,
                        "loss_reduction": launcher.RAW_JOINT_NLL_REDUCTION,
                        "expected_optimizer_step": step,
                        "observed_optimizer_step_min": float(step),
                        "observed_optimizer_step_max": float(step),
                        "expected_learning_rate_used": 2.0e-4 * (step / 400.0),
                        "learning_rate_used": 2.0e-4 * (step / 400.0),
                        "optimizer_audit_wall_seconds": 0.1,
                        "gradient_health": {
                            "optimizer_contract_version": launcher.OPTIMIZER_CONTRACT_VERSION,
                            "trainable_parameter_tensors": 10,
                            "gradient_tensors": 10,
                            "missing_gradient_tensors": 0,
                            "all_gradients_finite": True,
                            "global_l2_norm": 0.5,
                            "global_l2_positive": True,
                            "gradient_clipping": "disabled",
                            "gradient_modified_after_unscale": False,
                            "probe_parameter": "decoder.weight",
                            "probe_flat_index": 3,
                            "probe_gradient_abs": 0.25,
                            "audit_wall_seconds": 0.04,
                        },
                        "optimizer_state_health": {
                            "optimizer_contract_version": launcher.OPTIMIZER_CONTRACT_VERSION,
                            "trainable_parameter_tensors": 10,
                            "optimizer_state_entries": 10,
                            "state_complete": True,
                            "expected_optimizer_step": step,
                            "observed_optimizer_step_min": float(step),
                            "observed_optimizer_step_max": float(step),
                            "state_steps_complete_equal_expected": True,
                            "parameters_finite": True,
                            "exp_avg_finite": True,
                            "exp_avg_sq_finite": True,
                            "exp_avg_sq_nonnegative": True,
                            "exp_avg_sq_minimum": 0.0,
                            "probe_parameter": "decoder.weight",
                            "probe_flat_index": 3,
                            "probe_value_before": 0.1,
                            "probe_value_after": 0.09,
                            "probe_parameter_changed": True,
                            "optimizer_updated": True,
                            "audit_wall_seconds": 0.06,
                            "optimizer_configuration": {
                                "optimizer": "AdamW",
                                "parameter_group_count": 1,
                                "base_learning_rate": 2.0e-4,
                                "current_learning_rate": 2.0e-4 * (step / 400.0),
                                "weight_decay": 0.01,
                                "adamw_betas": [0.9, 0.999],
                                "adamw_eps": 1.0e-8,
                                "adamw_amsgrad": False,
                                "adamw_maximize": False,
                                "adamw_foreach": False,
                                "adamw_fused": False,
                            },
                        },
                        "scaler_scale_before": 32.0,
                        "scaler_scale_after": 32.0,
                        "scaler_skipped_steps": 0,
                        "optimizer_updated": True,
                    }
                    for step in (1, 2)
                ],
                "scaler_skipped_steps": 0,
            },
            "checkpoint_resume_canary": {
                "schema_version": launcher.V2_CHECKPOINT_SCHEMA,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_manifest_sha256": sha256(checkpoint_manifest_path),
                "manifest_file_count": len(checkpoint_files),
                "restored_global_step": 2,
                "resume_alignment": {
                    "global_step": 2,
                    "expected_optimizer_step": 2,
                    "observed_optimizer_step_min": 2.0,
                    "observed_optimizer_step_max": 2.0,
                },
            },
            "teacher_probe": {
                "anchors": 100,
                "subjects": 20,
                "wall_seconds": 10.0,
                "joint_nll_subject_macro": 1.0,
            },
            "free_running_probe": {
                "anchors": 100,
                "trajectories_per_anchor": 100,
                "wall_seconds": 20.0,
                "structural_subject_macro": {
                    key: 1.0 for key in launcher.CAPACITY_STRUCTURAL_METRICS
                },
                "coherence_rate": 1.0,
                "coherent_trajectories": 10_000,
                "observed_sample_ids_sha256": sample_set_sha,
                "selection_verified": True,
            },
            "projection": {
                "formal_max_steps": 4000,
                "formal_eval_steps": 250,
                "interval_teacher_passes": 16,
                "final_teacher_passes": 1,
                "total_teacher_passes": 17,
                "optimizer_seconds_per_step": optimizer_seconds,
                "teacher_seconds_per_anchor": teacher_seconds_per_anchor,
                "free_running_seconds_per_anchor": free_seconds_per_anchor,
                "components_seconds": components,
                "projected_formal_runtime_seconds": projected,
            },
            "runtime_projection": {
                "policy": launcher.CAPACITY_RUNTIME_POLICY,
                "hard_limit_seconds": None,
                "gates_capacity_status": False,
                "elapsed_before_capacity_seconds": 10.0,
                "capacity_probe_elapsed_seconds": 30.0,
                "projected_formal_runtime_seconds": projected,
                "projected_background_runtime_seconds": projected_background,
            },
            "failures": [],
        }
        write_json(root / "capacity_probe.json", report)
        return report

    def target_fixture(self, root: Path, launcher) -> dict[str, str]:
        contract_root = root / "contracts"
        contract_root.mkdir(parents=True)
        key_by_file = {
            "target_process_registry_v2.json": "process",
            "target_emission_registry_v2.json": "emission",
            "target_projection_registry_v2.json": "projection",
            "field_category_matrix_v1.csv": "category",
            "field_relation_edges_v1.csv": "relation",
            "event_element_extension_v2.json": "element_extension",
            "target_sidecar_schema_v2.json": "sidecar_schema",
        }
        contract_hashes = {}
        for filename, key in key_by_file.items():
            path = contract_root / filename
            if key == "process":
                content = json.dumps({"version": "fixture-process-v5"})
            elif key == "emission":
                content = json.dumps({"version": "fixture-emission-v4"})
            elif key == "projection":
                content = json.dumps({"version": "fixture-projection-r6"})
            else:
                content = f"fixture:{filename}\n"
            path.write_text(content, encoding="utf-8")
            contract_hashes[key] = sha256(path)
        (root / "sample_manifest.csv").write_text("sample_id\n", encoding="utf-8")
        (root / "subject_split.csv").write_text("subject_id,split\n", encoding="utf-8")
        (root / "SUCCEEDED").write_text("ok\n", encoding="utf-8")
        target_shards = {}
        last_samples = {"train": 734, "val": 309, "test": 307}
        for split, count in launcher.EXPECTED_SHARD_COUNTS.items():
            split_root = root / split
            split_root.mkdir()
            for index in range(count):
                name = f"{split}-{index:05d}"
                content = b"{}\n"
                (split_root / f"{name}.jsonl").write_bytes(content)
                compressed = io.BytesIO()
                gzip_handle = gzip.GzipFile(
                    filename="", mode="wb", fileobj=compressed, mtime=0
                )
                text_handle = io.TextIOWrapper(
                    gzip_handle, encoding="utf-8", newline="\n"
                )
                text_handle.write(content.decode("utf-8"))
                text_handle.flush()
                text_handle.close()
                target_shards[name] = {
                    "path": f"target_shards/{split}/{name}.jsonl.gz",
                    "sha256": hashlib.sha256(compressed.getvalue()).hexdigest(),
                    "samples": last_samples[split] if index == count - 1 else 1000,
                }
        bundle = "b" * 64
        manifest = {
            "dataset_id": "multires_event_m4_target_v2_c4_full_20260713_test",
            "counts": {
                "samples": 50350,
                "by_split": {"train": 37734, "val": 6309, "test": 6307},
                "shards": 52,
            },
            "contract_hashes": contract_hashes,
            "contract_bundle_hash": bundle,
            "files": {"target_shards": target_shards},
        }
        manifest_path = root / "dataset_manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return {
            "dataset_id": manifest["dataset_id"],
            "manifest_sha256": sha256(manifest_path),
            "sample_manifest_sha256": sha256(root / "sample_manifest.csv"),
            "contract_bundle_hash": bundle,
            "process_contract_sha256": contract_hashes["process"],
            "emission_contract_sha256": contract_hashes["emission"],
            "projection_contract_sha256": contract_hashes["projection"],
            "relation_contract_sha256": contract_hashes["relation"],
            "sidecar_schema_sha256": contract_hashes["sidecar_schema"],
            "process_contract_version": "fixture-process-v5",
            "emission_contract_version": "fixture-emission-v4",
            "projection_contract_version": "fixture-projection-r6",
        }

    def test_target_locator_requires_one_exact_identity(self) -> None:
        launcher = load_launcher()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "target-a"
            first.mkdir()
            authority = self.target_fixture(first, launcher)
            with patch.object(launcher, "TARGET_AUTHORITY", authority):
                self.assertEqual(
                    launcher.find_exact_target_dataset(root), first.resolve()
                )
                second = root / "target-b"
                second.mkdir()
                self.target_fixture(second, launcher)
                with self.assertRaisesRegex(RuntimeError, "multiple exact"):
                    launcher.find_exact_target_dataset(root)

    def test_target_authority_rejects_relation_or_sidecar_schema_hash_drift(self) -> None:
        launcher = load_launcher()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "target"
            root.mkdir()
            authority = self.target_fixture(root, launcher)
            manifest = json.loads(
                (root / "dataset_manifest.json").read_text(encoding="utf-8")
            )
            with patch.object(launcher, "TARGET_AUTHORITY", authority):
                self.assertTrue(launcher._matches_target_authority(root, manifest))
                for key in ("relation", "sidecar_schema"):
                    with self.subTest(key=key):
                        mutated = copy.deepcopy(manifest)
                        mutated["contract_hashes"][key] = "0" * 64
                        self.assertFalse(
                            launcher._matches_target_authority(root, mutated)
                        )

    def test_target_prepare_restores_plain_hosted_shards_and_contracts(self) -> None:
        launcher = load_launcher()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            authority = self.target_fixture(source, launcher)
            destination = root / "prepared"
            logs = root / "logs"
            logs.mkdir()
            with patch.object(launcher, "TARGET_AUTHORITY", authority):
                prepared = launcher.prepare_target_root(source, destination, logs)
                self.assertEqual(prepared, destination.resolve())
                self.assertTrue(launcher.is_prepared_target(prepared))
                self.assertEqual(
                    len(list((prepared / "target_shards/train").glob("*.jsonl.gz"))), 38
                )
                report = json.loads((logs / "target_dataset_prepare.json").read_text())
                self.assertEqual(report["materialized_target_shards"], 52)
                self.assertEqual(
                    report["target_shard_layout"], "kaggle_hosted_extracted_target_tree"
                )

    def test_target_dataset_ref_is_frozen_and_override_must_match(self) -> None:
        launcher = load_launcher()
        expected = "vanilaaaa/trauma-predict-multires-event-v2-c4-r8-20260713"
        self.assertEqual(launcher.TARGET_DATASET_REF, expected)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_root = root / "logs"
            log_root.mkdir()
            downloaded = root / "downloaded"
            with patch.object(launcher, "KAGGLE_INPUT", root), patch.object(
                launcher, "download_exact_dataset", return_value=downloaded
            ) as download, patch.dict(os.environ, {}, clear=True):
                self.assertEqual(
                    launcher.explicit_or_download_target_root(log_root), downloaded
                )
                self.assertEqual(download.call_args.kwargs["dataset_ref"], expected)

            with patch.object(launcher, "KAGGLE_INPUT", root), patch.object(
                launcher, "download_exact_dataset", return_value=downloaded
            ) as download, patch.dict(
                os.environ, {"TRAUMA_PREDICT_V2_DATASET_REF": expected}, clear=True
            ):
                self.assertEqual(
                    launcher.explicit_or_download_target_root(log_root), downloaded
                )
                self.assertEqual(download.call_args.kwargs["dataset_ref"], expected)

            for invalid in ("", expected + "-other", f" {expected}"):
                with self.subTest(invalid=invalid), patch.object(
                    launcher, "KAGGLE_INPUT", root
                ), patch.object(launcher, "download_exact_dataset") as download, patch.dict(
                    os.environ, {"TRAUMA_PREDICT_V2_DATASET_REF": invalid}, clear=True
                ):
                    with self.assertRaisesRegex(ValueError, "must exactly equal"):
                        launcher.explicit_or_download_target_root(log_root)
                    download.assert_not_called()

    def test_zero_input_dataset_access_is_checked_before_download(self) -> None:
        launcher = load_launcher()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            launcher.v1_route, "configure_kaggle_credentials"
        ) as credentials, patch.object(launcher, "run_to_log") as run, patch.dict(
            os.environ,
            {
                "TRAUMA_PREDICT_DATA_ROOT": "",
                "TRAUMA_PREDICT_V2_TARGET_ROOT": "",
                "TRAUMA_PREDICT_V2_DATASET_REF": launcher.TARGET_DATASET_REF,
            },
            clear=False,
        ):
            launcher.preflight_dataset_download_access(Path(directory))
        credentials.assert_called_once_with()
        self.assertEqual(run.call_count, 2)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            [command[4] for command in commands],
            [launcher.BASE_DATASET_REF, launcher.TARGET_DATASET_REF],
        )
        self.assertTrue(all(command[-2:] == ["--page-size", "1"] for command in commands))

        with tempfile.TemporaryDirectory() as directory, patch.object(
            launcher, "run_to_log"
        ) as run, patch.dict(
            os.environ,
            {"TRAUMA_PREDICT_DATA_ROOT": "/manual/input"},
            clear=False,
        ), self.assertRaisesRegex(RuntimeError, "zero-Input"):
            launcher.preflight_dataset_download_access(Path(directory))
        run.assert_not_called()

    def test_action_selector_accepts_exactly_one_action_and_defaults_to_smoke(self) -> None:
        launcher = load_launcher()
        with patch.dict(
            os.environ,
            {"TRAUMA_PREDICT_V2_ACTION": "", "TRAUMA_PREDICT_V2_STAGES": ""},
            clear=False,
        ):
            self.assertEqual(launcher.selected_action(), "smoke")
        for action in (
            "smoke",
            "block",
            "trajectory",
            "relational",
            "verify_block",
            "promotion",
        ):
            self.assertEqual(launcher.selected_action(action), action)
        with self.assertRaisesRegex(ValueError, "exactly one action"):
            launcher.selected_action("block,trajectory")
        with self.assertRaisesRegex(ValueError, "unknown"):
            launcher.selected_action("modernbert")
        with patch.dict(
            os.environ,
            {"TRAUMA_PREDICT_V2_STAGES": "trajectory"},
            clear=False,
        ), self.assertRaisesRegex(ValueError, "forbidden"):
            launcher.selected_action("trajectory")

    def test_final_authority_has_no_placeholder(self) -> None:
        launcher = load_launcher()
        launcher.require_frozen_authority_constants()
        self.assertEqual(
            launcher.TARGET_DATASET_REF,
            "vanilaaaa/trauma-predict-multires-event-v2-c4-r8-20260713",
        )
        self.assertNotIn("PENDING", json.dumps(launcher.TARGET_AUTHORITY))
        self.assertRegex(launcher.EXPECTED_LAB_SCALE_ARTIFACT_SHA256, r"^[0-9a-f]{64}$")
        for key in ("relation_contract_sha256", "sidecar_schema_sha256"):
            with self.subTest(key=key):
                authority = dict(launcher.TARGET_AUTHORITY)
                authority[key] = "PENDING"
                with patch.object(launcher, "TARGET_AUTHORITY", authority), self.assertRaisesRegex(
                    RuntimeError, key.split("_")[0]
                ):
                    launcher.require_frozen_authority_constants()

    def test_hosted_training_has_a_source_level_authorization_gate(self) -> None:
        launcher = load_launcher()
        self.assertTrue(launcher.TRAINING_AUTHORIZED)
        launcher.require_training_authorization("block")
        for action in ("smoke", "trajectory", "relational"):
            with self.subTest(action=action), self.assertRaisesRegex(
                RuntimeError, "not authorized"
            ):
                launcher.require_training_authorization(action)
        self.assertTrue(launcher.VERIFICATION_AUTHORIZED)
        launcher.require_verification_authorization("block")
        for action in ("smoke", "trajectory", "relational"):
            with self.subTest(verification_action=action), self.assertRaisesRegex(
                RuntimeError, "not authorized"
            ):
                launcher.require_verification_authorization(action)
        with patch.dict(os.environ, {"TRAUMA_PREDICT_V2_TRAINING_AUTHORIZED": "1"}):
            with self.assertRaisesRegex(RuntimeError, "not authorized"):
                launcher.require_training_authorization("trajectory")

    def test_direct_cli_cannot_bypass_core_gate_but_dry_preflight_remains_available(self) -> None:
        entrypoint = load_entrypoint()
        args = SimpleNamespace(
            config=Path("configs/train/t4x2_multires_event_v2_block.yaml"),
            dry_run=False,
            capacity_probe_output=None,
            elapsed_before_capacity_seconds=None,
            rank_artifact_preflight_output=None,
            rank_artifact_preflight_mode=None,
            verification_only=False,
        )
        with patch.object(entrypoint, "parse_args", return_value=args), patch.object(
            entrypoint, "run_multires_event_v2_training"
        ) as training, self.assertRaisesRegex(RuntimeError, "capacity-gated single-torchrun"):
            entrypoint.main()
        training.assert_not_called()

        unauthorized_args = SimpleNamespace(
            **{
                **vars(args),
                "config": Path("configs/train/t4x2_multires_event_v2_trajectory.yaml"),
            }
        )
        with patch.object(
            entrypoint, "parse_args", return_value=unauthorized_args
        ), self.assertRaisesRegex(RuntimeError, "not authorized for run_name"):
            entrypoint.main()

        dry_args = SimpleNamespace(
            **{
                **vars(args),
                "config": Path("configs/train/t4x2_multires_event_v2_smoke.yaml"),
                "dry_run": True,
            }
        )
        with patch.object(entrypoint, "parse_args", return_value=dry_args), patch.object(
            entrypoint, "run_dry_preflight"
        ) as dry:
            entrypoint.main()
        dry.assert_called_once()

        with tempfile.TemporaryDirectory() as directory:
            early_args = SimpleNamespace(
                config=None,
                dry_run=False,
                capacity_probe_output=None,
                elapsed_before_capacity_seconds=None,
                rank_artifact_preflight_output=Path(directory) / "canary",
                rank_artifact_preflight_mode="trajectory",
                verification_only=False,
            )
            with patch.object(
                entrypoint, "parse_args", return_value=early_args
            ), patch.object(
                entrypoint, "run_multires_event_v2_rank_artifact_preflight_only"
            ) as early, patch.object(
                entrypoint, "require_multires_event_v2_training_authorization"
            ) as authorization:
                entrypoint.main()
            early.assert_called_once_with(
                output_dir=early_args.rank_artifact_preflight_output,
                mode="trajectory",
            )
            authorization.assert_not_called()

            verification_args = SimpleNamespace(
                config=Path("configs/train/t4x2_multires_event_v2_block.yaml"),
                dry_run=False,
                capacity_probe_output=Path(directory) / "verification",
                elapsed_before_capacity_seconds=1.25,
                rank_artifact_preflight_output=None,
                rank_artifact_preflight_mode=None,
                verification_only=True,
            )
            with patch.object(
                entrypoint, "parse_args", return_value=verification_args
            ), patch.object(
                entrypoint, "run_multires_event_v2_verification_probe"
            ) as verification, patch.object(
                entrypoint, "require_multires_event_v2_training_authorization"
            ) as authorization:
                entrypoint.main()
            verification.assert_called_once_with(
                (entrypoint.REPO_ROOT / verification_args.config).resolve(),
                repo_root=entrypoint.REPO_ROOT,
                output_dir=verification_args.capacity_probe_output,
                elapsed_before_capacity_seconds=1.25,
            )
            authorization.assert_not_called()

    def test_entrypoint_requires_repo_train_only_lab_scale_content_hash(self) -> None:
        entrypoint = load_entrypoint()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "schema": "multires_event_v2_lab_affine_scale_v1",
                "status": "frozen_train_only_fit",
                "fit_split": "train",
                "fields": {},
            }
            canonical = json.dumps(
                payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            content_hash = hashlib.sha256(canonical).hexdigest()
            payload["content_sha256"] = content_hash
            path = root / "lab_scale.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            train = {
                "lab_scale_artifact": "lab_scale.json",
                "lab_scale_artifact_hash": content_hash,
            }
            with patch.object(entrypoint, "REPO_ROOT", root):
                result = entrypoint.verify_repo_lab_scale_artifact(train)
                self.assertEqual(result["sha256"], content_hash)
                invalid = dict(train, lab_scale_artifact_hash="a" * 64)
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    entrypoint.verify_repo_lab_scale_artifact(invalid)

    def test_notebook_is_zero_input_two_cell_formal_block_bootstrap(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(notebook["cells"]), 2)
        code = "".join(notebook["cells"][1]["source"])
        markdown = "".join(notebook["cells"][0]["source"])
        self.assertIn("multires-event-v2-block-run-20260714-r8", code)
        self.assertIn("refs/tags/", code)
        self.assertIn("run_multires_event_v2.py", code)
        self.assertIn('REQUIRED_GIT_REF = "multires-event-v2-block-run-20260714-r8"', code)
        self.assertIn('V2_ACTION = "block"', code)
        self.assertIn("Do not attach any Kaggle Input", markdown)
        self.assertIn("MULTIRES_EVENT_V2_HOSTED_SMOKE_OK", markdown)
        self.assertNotIn("KAGGLE_INPUT", code)
        self.assertNotIn("SOURCE_BUNDLE", code)
        self.assertNotIn("git\", \"bundle", code)
        self.assertIn('FROZEN_ENV = {', code)
        self.assertIn("remove the conflicting environment override", code)
        self.assertIn("trauma-predict-multires-event-v2-c4-r8-20260713", code)
        self.assertIn('env["TRAUMA_PREDICT_V2_ACTION"] = V2_ACTION', code)
        self.assertIn('env["TRAUMA_PREDICT_DRY_RUN_ONLY"] = "0"', code)
        self.assertNotIn('env["TRAUMA_PREDICT_DRY_RUN_ONLY"] = "1"', code)
        self.assertIn('["git", "clone", REPO_URL, str(REPO_DIR)]', code)
        self.assertIn('["git", "fetch", "--force", "origin", "--tags"]', code)
        entrypoint_source = ENTRYPOINT_PATH.read_text(encoding="utf-8")
        self.assertIn('"relation_contract_sha256"', entrypoint_source)
        self.assertIn('"sidecar_schema_sha256"', entrypoint_source)

    def test_verification_notebook_is_zero_input_and_formal_step_zero(self) -> None:
        notebook = json.loads(VERIFICATION_NOTEBOOK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(notebook["cells"]), 2)
        markdown = "".join(notebook["cells"][0]["source"])
        code = "".join(notebook["cells"][1]["source"])
        self.assertIn("formal_optimizer_steps=0", markdown)
        self.assertIn(
            'REQUIRED_GIT_REF = "multires-event-v2-block-run-20260714-r8"',
            code,
        )
        self.assertIn('V2_ACTION = "verify_block"', code)
        self.assertNotIn("KAGGLE_INPUT", code)
        self.assertNotIn("SOURCE_BUNDLE", code)
        self.assertIn('["git", "clone", REPO_URL, str(REPO_DIR)]', code)
        self.assertIn('env["TRAUMA_PREDICT_DRY_RUN_ONLY"] = "0"', code)

    def test_route_is_single_action_single_torchrun_without_inline_promotion(self) -> None:
        launcher = load_launcher()
        source = LAUNCHER_PATH.read_text(encoding="utf-8")
        main_source = inspect.getsource(launcher.main)
        self.assertIn('"--nproc_per_node=2"', source)
        self.assertEqual(
            tuple(load_launcher().STAGE_CONFIGS),
            ("smoke", "block", "trajectory", "relational"),
        )
        self.assertIn("heartbeat(\n        label, log_path, seconds=HEARTBEAT_SECONDS", source)
        self.assertEqual(launcher.HEARTBEAT_SECONDS, 60)
        self.assertIn("STREAM_ERROR_MARKERS", source)
        self.assertNotIn("PENDING_FINAL", source)
        self.assertIn("require_frozen_authority_constants", source)
        self.assertNotIn("selected_stage_names", source)
        self.assertNotIn("for stage in", main_source)
        self.assertIn('if action == "promotion":', main_source)
        self.assertNotIn("ModernBERT", source)
        self.assertNotIn("transformers", source)
        self.assertNotIn('"pip", "check"', source)
        self.assertIn("MULTIRES_DEPENDENCY_IMPORT", source)
        self.assertTrue(ENTRYPOINT_PATH.is_file())

    def test_formal_torchrun_contains_one_capacity_gate_invocation(self) -> None:
        launcher = load_launcher()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            launcher, "run_to_log"
        ) as run:
            root = Path(directory)
            launcher.run_torchrun(
                launcher.STAGE_CONFIGS["trajectory"],
                root / "trajectory.log",
                env={"PYTHONPATH": "fixture"},
                label="TRAJECTORY",
                capacity_output_dir=root / "capacity",
                elapsed_before_capacity_seconds=12.5,
            )
            run.assert_called_once()
            command = run.call_args.args[0]
            self.assertEqual(command.count("-m"), 1)
            self.assertEqual(command.count("torch.distributed.run"), 1)
            self.assertEqual(command.count("--capacity-probe-output"), 1)
            self.assertEqual(command.count("--elapsed-before-capacity-seconds"), 1)
            torchrun_env = run.call_args.kwargs["env"]
            self.assertEqual(torchrun_env["TORCH_NCCL_ASYNC_ERROR_HANDLING"], "1")
            self.assertEqual(torchrun_env["TORCH_NCCL_ENABLE_MONITORING"], "1")
            self.assertEqual(
                torchrun_env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"],
                str(launcher.V2_NCCL_MONITOR_HEARTBEAT_TIMEOUT_SECONDS),
            )

    def test_predata_rank_canary_is_config_free_and_precedes_dataset_resolution(self) -> None:
        launcher = load_launcher()
        source = inspect.getsource(launcher.main)
        self.assertLess(
            source.index("run_rank_artifact_preflight_torchrun("),
            source.index("explicit_or_download_base_root("),
        )
        self.assertLess(
            source.index("preflight_dataset_download_access("),
            source.index("install_requirements("),
        )
        self.assertLess(
            source.index("runtime_guard("),
            source.index("explicit_or_download_base_root("),
        )
        self.assertLess(
            source.index("run_rank_artifact_preflight_torchrun("),
            source.index("prepare_target_root("),
        )
        self.assertLess(
            source.index("run_rank_artifact_preflight_torchrun("),
            source.index("install_requirements("),
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            launcher, "run_to_log"
        ) as run:
            root = Path(directory)
            launcher.run_rank_artifact_preflight_torchrun(
                mode="trajectory",
                output_dir=root / "canary",
                log_path=root / "canary.log",
                env={"PYTHONPATH": "fixture"},
            )
            run.assert_called_once()
            command = run.call_args.args[0]
            self.assertNotIn("--config", command)
            self.assertIn("--rank-artifact-preflight-output", command)
            self.assertIn("--rank-artifact-preflight-mode", command)
            self.assertEqual(command[-1], "trajectory")
            self.assertEqual(
                run.call_args.kwargs["env"]["TORCH_NCCL_ASYNC_ERROR_HANDLING"],
                "1",
            )

    def test_capacity_probe_output_is_attempt_specific_and_disjoint(self) -> None:
        launcher = load_launcher()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            formal_root = root / "t4x2_multires_event_v2_block"
            attempt_root = formal_root / "logs" / "attempt-0001"
            attempt_root.mkdir(parents=True)
            with patch.object(launcher, "OUTPUT_ROOT", root):
                probe_root = launcher.capacity_probe_output_for_attempt(
                    formal_root,
                    attempt_root,
                )
            self.assertEqual(
                probe_root,
                root
                / "_capacity-probes"
                / "t4x2_multires_event_v2_block"
                / "attempt-0001",
            )
            with self.assertRaises(ValueError):
                probe_root.relative_to(formal_root)
            with self.assertRaises(ValueError):
                formal_root.relative_to(probe_root)
            with patch.object(launcher, "OUTPUT_ROOT", root), self.assertRaisesRegex(
                ValueError,
                "formal logs root",
            ):
                launcher.capacity_probe_output_for_attempt(
                    formal_root,
                    root / "unbound-attempt-0001",
                )

    def test_capacity_report_fails_closed_under_contract_mutations(self) -> None:
        launcher = load_launcher()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_root = root / "valid"
            valid_report = self.capacity_report_fixture(valid_root, launcher)
            self.assertEqual(
                [row["learning_rate_used"] for row in valid_report["optimizer"]["steps"]],
                [5.000000000000001e-7, 1.0000000000000002e-6],
            )
            validation = launcher.validate_capacity_probe_report(
                valid_root, expected_mode="block"
            )
            self.assertEqual(validation["status"], "PASSED")
            completion_path = valid_root / "verification_complete.json"
            write_json(
                completion_path,
                {
                    "schema_version": (
                        "trauma_predict.multires_event_v2_verification_complete.v1"
                    ),
                    "status": "PASSED_STOPPED_BEFORE_FORMAL_TRAINING",
                    "formal_training_authorized": False,
                    "formal_optimizer_steps": 0,
                    "mode": "block",
                    "run_name": "t4x2_multires_event_v2_block",
                    "capacity_report_path": valid_report["report_path"],
                    "capacity_report_sha256": sha256(
                        Path(valid_report["report_path"])
                    ),
                },
            )
            verification = launcher.validate_verification_probe(
                valid_root,
                expected_mode="block",
            )
            self.assertEqual(verification["formal_optimizer_steps"], 0)
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            completion["formal_optimizer_steps"] = 1
            write_json(completion_path, completion)
            with self.assertRaisesRegex(ValueError, "completion contract"):
                launcher.validate_verification_probe(
                    valid_root,
                    expected_mode="block",
                )
            completion["formal_optimizer_steps"] = 0
            write_json(completion_path, completion)

            mutations = {
                "optimizer update": lambda row: row["optimizer"]["steps"][0].__setitem__(
                    "optimizer_updated", False
                ),
                "optimizer contract": lambda row: row["optimizer"].__setitem__(
                    "gradient_clipping", "global_norm"
                ),
                "missing gradient": lambda row: row["optimizer"]["steps"][0][
                    "gradient_health"
                ].__setitem__("missing_gradient_tensors", 1),
                "negative second moment": lambda row: row["optimizer"]["steps"][0][
                    "optimizer_state_health"
                ].__setitem__("exp_avg_sq_minimum", -1.0),
                "unchanged update probe": lambda row: row["optimizer"]["steps"][0][
                    "optimizer_state_health"
                ].__setitem__("probe_value_after", 0.1),
                "scaler drift": lambda row: row["optimizer"]["steps"][1].__setitem__(
                    "scaler_scale_after", 64.0
                ),
                "step learning rate": lambda row: row["optimizer"]["steps"][0].__setitem__(
                    "learning_rate_used", float("nan")
                ),
                "audit wall time": lambda row: row["optimizer"]["steps"][0].__setitem__(
                    "optimizer_audit_wall_seconds", -1.0
                ),
                "T4 hardware": lambda row: row["hardware"][0].__setitem__(
                    "device_name", "RTX 5070"
                ),
                "rank artifact canary": lambda row: row["distributed_canaries"][
                    "rank_artifact"
                ].__setitem__("status", "FAILED"),
                "semantic canary": lambda row: row["distributed_canaries"][
                    "semantic_rollout"
                ].__setitem__("coherence_rate", 0.5),
                "checkpoint identity": lambda row: row[
                    "checkpoint_resume_canary"
                ].__setitem__("checkpoint_manifest_sha256", "0" * 64),
                "checkpoint resume alignment": lambda row: row[
                    "checkpoint_resume_canary"
                ]["resume_alignment"].__setitem__("observed_optimizer_step_max", 1.0),
                "first 100 anchors": lambda row: row["free_running_probe"].__setitem__(
                    "selection_verified", False
                ),
                "relation contract identity": lambda row: row["identity"].__setitem__(
                    "relation_contract_sha256", "0" * 64
                ),
                "sidecar schema identity": lambda row: row["identity"].__setitem__(
                    "sidecar_schema_sha256", "0" * 64
                ),
                "structural metric": lambda row: row["free_running_probe"][
                    "structural_subject_macro"
                ].__setitem__(launcher.CAPACITY_STRUCTURAL_METRICS[0], float("nan")),
                "projection closure": lambda row: row["projection"][
                    "components_seconds"
                ].__setitem__("optimizer", 1.0),
                "runtime projection policy": lambda row: row[
                    "runtime_projection"
                ].__setitem__(
                    "policy",
                    "unsupported_hard_duration_gate",
                ),
                "runtime projection closure": lambda row: row[
                    "runtime_projection"
                ].__setitem__(
                    "projected_background_runtime_seconds",
                    row["runtime_projection"][
                        "projected_background_runtime_seconds"
                    ]
                    + 1.0,
                ),
            }
            for index, (label, mutate) in enumerate(mutations.items()):
                with self.subTest(label=label):
                    candidate_root = root / f"mutation-{index}"
                    candidate = self.capacity_report_fixture(candidate_root, launcher)
                    mutate(candidate)
                    write_json(candidate_root / "capacity_probe.json", candidate)
                    with self.assertRaises(ValueError):
                        launcher.validate_capacity_probe_report(
                            candidate_root, expected_mode="block"
                        )

            success_root = root / "success-marker"
            self.capacity_report_fixture(success_root, launcher)
            (success_root / "SUCCESS").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SUCCESS"):
                launcher.validate_capacity_probe_report(
                    success_root, expected_mode="block"
                )

    def test_promotion_action_never_initializes_cuda_data_or_training(self) -> None:
        launcher = load_launcher()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validated = {
                mode: self.promotion_validation(root / mode, mode)
                for mode in launcher.PROMOTION_MODES
            }

            def promote(_validated, attempt_dir):
                result = {"promoted": True, "winner": "relational"}
                write_json(attempt_dir / "promotion.json", result)
                return result

            with patch.object(launcher, "OUTPUT_ROOT", root), patch.object(
                launcher, "require_frozen_authority_constants"
            ), patch.object(
                launcher, "selected_action", return_value="promotion"
            ), patch.object(
                launcher,
                "verify_source_identity",
                return_value={"git_ref": "fixture", "commit": "1" * 40},
            ), patch.object(
                launcher,
                "verify_matched_suite_and_lab_scale",
                return_value={"matched_factor_signature": "2" * 64},
            ), patch.object(
                launcher, "is_kaggle_runtime", return_value=True
            ), patch.object(
                launcher, "validate_promotion_run_roots", return_value=validated
            ) as roots, patch.object(
                launcher,
                "run_promotion",
                side_effect=promote,
            ), patch.object(
                launcher, "require_t4x2_runtime"
            ) as cuda, patch.object(
                launcher, "explicit_or_download_base_root"
            ) as data, patch.object(
                launcher, "install_requirements"
            ) as install, patch.object(
                launcher, "run_torchrun"
            ) as train:
                launcher.main()
            roots.assert_called_once_with(require_all=True, require_attached=True)
            cuda.assert_not_called()
            data.assert_not_called()
            install.assert_not_called()
            train.assert_not_called()

    def test_completed_matched_run_is_reused_only_after_full_validation(self) -> None:
        launcher = load_launcher()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "block"
            run_dir.mkdir()
            with patch.object(launcher, "validate_completed_run") as validate:
                self.assertIsNone(
                    launcher.reusable_completed_run(
                        run_dir,
                        expected_mode="block",
                        require_free_running=True,
                    )
                )
                validate.assert_not_called()
                (run_dir / "SUCCESS").write_text("{}\n", encoding="utf-8")
                expected = self.promotion_validation(run_dir, "block")
                validate.return_value = expected
                self.assertEqual(
                    launcher.reusable_completed_run(
                        run_dir,
                        expected_mode="block",
                        require_free_running=True,
                    ),
                    expected,
                )
                validate.assert_called_once_with(
                    run_dir,
                    expected_mode="block",
                    require_free_running=True,
                )

    def test_completed_run_validation_binds_evaluation_and_selected_model_bytes(self) -> None:
        launcher = load_launcher()

        def git_text(*args):
            return "1" * 40 if args[-1] == "HEAD" else "2" * 40

        with tempfile.TemporaryDirectory() as directory, patch.object(
            launcher, "_git_text", side_effect=git_text
        ):
            root = Path(directory)
            valid = self.completed_run_fixture(root / "valid", launcher)
            validation = launcher.validate_completed_run(
                valid,
                expected_mode="block",
                require_free_running=True,
            )
            self.assertEqual(validation["parameter_count"], 123456)
            self.assertRegex(
                str(validation["semantic_runtime_identity_sha256"]),
                r"^[0-9a-f]{64}$",
            )

            cases = {}
            missing_final = self.completed_run_fixture(root / "missing-final", launcher)
            (missing_final / "final_model/model.pt").unlink()
            cases["missing final model"] = missing_final

            changed_best = self.completed_run_fixture(root / "changed-best", launcher)
            (changed_best / "best_checkpoint/model.pt").write_bytes(b"changed-best")
            cases["changed best model"] = changed_best

            changed_teacher = self.completed_run_fixture(root / "changed-teacher", launcher)
            teacher_path = changed_teacher / "val_per_anchor_joint_nll.jsonl"
            teacher_path.write_text('{"sample_id":"changed"}\n', encoding="utf-8")
            evaluation = json.loads(
                (changed_teacher / "evaluation.json").read_text(encoding="utf-8")
            )
            evaluation["per_anchor_output_sha256"] = sha256(teacher_path)
            write_json(changed_teacher / "evaluation.json", evaluation)
            cases["teacher evaluation detached from manifest"] = changed_teacher

            changed_selected = self.completed_run_fixture(root / "changed-selected", launcher)
            run_manifest_path = changed_selected / "run_manifest.json"
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            run_manifest["selected_model_identity"]["selected_checkpoint_step"] = 251
            write_json(run_manifest_path, run_manifest)
            resign_success(changed_selected)
            cases["changed selected checkpoint"] = changed_selected

            changed_final = self.completed_run_fixture(root / "changed-final", launcher)
            changed_final_path = changed_final / "final_model/model.pt"
            changed_final_path.write_bytes(b"changed-final")
            changed_model_manifest_path = changed_final / "final_model/model_manifest.json"
            changed_model_manifest = json.loads(
                changed_model_manifest_path.read_text(encoding="utf-8")
            )
            changed_model_manifest["model_sha256"] = sha256(changed_final_path)
            write_json(changed_model_manifest_path, changed_model_manifest)
            changed_run_manifest_path = changed_final / "run_manifest.json"
            changed_run_manifest = json.loads(
                changed_run_manifest_path.read_text(encoding="utf-8")
            )
            changed_run_manifest["final_model"]["sha256"] = sha256(changed_final_path)
            changed_run_manifest["final_model"]["manifest_sha256"] = sha256(
                changed_model_manifest_path
            )
            write_json(changed_run_manifest_path, changed_run_manifest)
            resign_success(changed_final)
            cases["changed final model"] = changed_final

            changed_config = self.completed_run_fixture(root / "changed-config", launcher)
            train_artifact_path = changed_config / "artifacts/config/train.yaml"
            train_artifact = yaml.safe_load(train_artifact_path.read_text(encoding="utf-8"))
            train_artifact["run_name"] = "changed-run-name"
            train_artifact_path.write_text(
                yaml.safe_dump(train_artifact, sort_keys=False), encoding="utf-8"
            )
            artifact_manifest_path = changed_config / "artifacts/manifest.json"
            artifact_manifest = json.loads(
                artifact_manifest_path.read_text(encoding="utf-8")
            )
            artifact_manifest["artifacts"]["train_config"]["file_sha256"] = sha256(
                train_artifact_path
            )
            write_json(artifact_manifest_path, artifact_manifest)
            changed_config_manifest_path = changed_config / "run_manifest.json"
            changed_config_manifest = json.loads(
                changed_config_manifest_path.read_text(encoding="utf-8")
            )
            changed_config_manifest["artifact_manifest"]["sha256"] = sha256(
                artifact_manifest_path
            )
            write_json(changed_config_manifest_path, changed_config_manifest)
            resign_success(changed_config)
            cases["changed portable config"] = changed_config

            changed_audit = self.completed_run_fixture(root / "changed-audit", launcher)
            (changed_audit / "free_running/primitive_samples.rank00000.jsonl").write_text(
                '{"sample_id":"changed-audit"}\n', encoding="utf-8"
            )
            cases["changed free-running audit shard"] = changed_audit

            changed_schema = self.completed_run_fixture(root / "changed-schema", launcher)
            write_json(
                changed_schema / "free_running/sample_schema.json",
                {"schema_version": "changed"},
            )
            cases["changed free-running sample schema"] = changed_schema

            changed_runtime = self.completed_run_fixture(root / "changed-runtime", launcher)
            runtime_path = changed_runtime / "artifacts/runtime_environment.json"
            runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime_payload["semantic_runtime_identity"]["torch"] = "mutated"
            write_json(runtime_path, runtime_payload)
            cases["changed runtime environment"] = changed_runtime

            changed_dataset_relation = self.completed_run_fixture(
                root / "changed-dataset-relation", launcher
            )
            dataset_identity_path = changed_dataset_relation / "dataset_identity.json"
            dataset_identity = json.loads(
                dataset_identity_path.read_text(encoding="utf-8")
            )
            dataset_identity["relation_contract_sha256"] = "0" * 64
            write_json(dataset_identity_path, dataset_identity)
            cases["changed dataset relation contract identity"] = (
                changed_dataset_relation
            )

            changed_objective_schema = self.completed_run_fixture(
                root / "changed-objective-schema", launcher
            )
            objective_path = changed_objective_schema / "objective_contract.json"
            objective = json.loads(objective_path.read_text(encoding="utf-8"))
            objective["contract_hashes"]["sidecar_schema"] = "0" * 64
            write_json(objective_path, objective)
            cases["changed objective sidecar schema identity"] = (
                changed_objective_schema
            )

            changed_teacher_relation = self.completed_run_fixture(
                root / "changed-teacher-relation", launcher
            )
            teacher_evaluation_path = changed_teacher_relation / "evaluation.json"
            teacher_evaluation = json.loads(
                teacher_evaluation_path.read_text(encoding="utf-8")
            )
            teacher_evaluation["identity"]["relation_contract_sha256"] = "0" * 64
            write_json(teacher_evaluation_path, teacher_evaluation)
            teacher_manifest_path = changed_teacher_relation / "run_manifest.json"
            teacher_manifest = json.loads(
                teacher_manifest_path.read_text(encoding="utf-8")
            )
            teacher_manifest["evaluation"] = teacher_evaluation
            write_json(teacher_manifest_path, teacher_manifest)
            resign_success(changed_teacher_relation)
            cases["changed teacher relation contract identity"] = (
                changed_teacher_relation
            )

            changed_free_schema = self.completed_run_fixture(
                root / "changed-free-schema", launcher
            )
            free_evaluation_path = changed_free_schema / "free_running/evaluation.json"
            free_evaluation = json.loads(
                free_evaluation_path.read_text(encoding="utf-8")
            )
            free_evaluation["identity"]["sidecar_schema_sha256"] = "0" * 64
            write_json(free_evaluation_path, free_evaluation)
            free_manifest_pointer_path = changed_free_schema / "run_manifest.json"
            free_run_manifest = json.loads(
                free_manifest_pointer_path.read_text(encoding="utf-8")
            )
            free_run_manifest["free_running_evaluation"]["sha256"] = sha256(
                free_evaluation_path
            )
            write_json(free_manifest_pointer_path, free_run_manifest)
            resign_success(changed_free_schema)
            cases["changed free-running sidecar schema identity"] = (
                changed_free_schema
            )

            empty_metrics = self.completed_run_fixture(root / "empty-metrics", launcher)
            metrics_path = empty_metrics / "metrics.jsonl"
            metrics_path.write_text("{}\n", encoding="utf-8")
            summary_path = empty_metrics / "optimizer_health_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["metrics_sha256"] = sha256(metrics_path)
            write_json(summary_path, summary)
            empty_manifest_path = empty_metrics / "run_manifest.json"
            empty_manifest = json.loads(empty_manifest_path.read_text(encoding="utf-8"))
            empty_manifest["optimizer_health_summary"]["sha256"] = sha256(summary_path)
            empty_manifest["optimizer_health_summary"]["metrics_sha256"] = sha256(
                metrics_path
            )
            write_json(empty_manifest_path, empty_manifest)
            resign_success(empty_metrics)
            cases["empty optimizer metrics with re-signed hash chain"] = empty_metrics

            partial = self.completed_run_fixture(root / "partial-training", launcher)
            partial_manifest_path = partial / "run_manifest.json"
            partial_manifest = json.loads(
                partial_manifest_path.read_text(encoding="utf-8")
            )
            partial_manifest["training"]["global_step"] = 300
            partial_manifest["training"]["max_steps"] = 300
            partial_manifest["training"]["training_completed_step"] = 300
            write_json(partial_manifest_path, partial_manifest)
            resign_success(partial)
            cases["partial 300-step run with re-signed manifest"] = partial

            missing_summary = self.completed_run_fixture(
                root / "missing-health-summary", launcher
            )
            (missing_summary / "optimizer_health_summary.json").unlink()
            cases["missing optimizer health summary"] = missing_summary

            for label, run_dir in cases.items():
                with self.subTest(label=label), self.assertRaises(
                    (FileNotFoundError, ValueError)
                ):
                    launcher.validate_completed_run(
                        run_dir,
                        expected_mode="block",
                        require_free_running=True,
                    )

    def test_portable_yaml_matches_resolved_config_with_frozen_env_expansion(self) -> None:
        launcher = load_launcher()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            templates = {
                "train_config": {
                    "output": "${TRAUMA_PREDICT_OUTPUT_ROOT}/block",
                    "metrics": "${TRAUMA_PREDICT_OUTPUT_ROOT}/block/metrics.jsonl",
                },
                "dataset_config": {"root": "${TRAUMA_PREDICT_DATA_ROOT}/full"},
                "model_config": {"hidden_size": 32},
            }
            for name, payload in templates.items():
                write_json(run_dir / launcher.PORTABLE_RUN_ARTIFACTS[name], payload)
            resolved = {
                "train": {
                    "output": "/original/output/block",
                    "metrics": "/original/output/block/metrics.jsonl",
                },
                "dataset": {"root": "/original/data/full"},
                "model": {"hidden_size": 32},
            }
            launcher._validate_portable_config_yaml(run_dir, resolved)
            inconsistent = copy.deepcopy(resolved)
            inconsistent["train"]["metrics"] = "/different/output/block/metrics.jsonl"
            with self.assertRaisesRegex(ValueError, "placeholder expansion"):
                launcher._validate_portable_config_yaml(run_dir, inconsistent)

    def test_promotion_resolves_three_independent_completed_run_roots(self) -> None:
        launcher = load_launcher()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dirs = {mode: root / mode for mode in launcher.PROMOTION_MODES}
            for run_dir in run_dirs.values():
                run_dir.mkdir()
                (run_dir / "SUCCESS").write_text("{}\n", encoding="utf-8")
            environment = {
                launcher.PROMOTION_RUN_ROOT_ENV[mode]: str(run_dir)
                for mode, run_dir in run_dirs.items()
            }

            def validate(run_dir, *, expected_mode, require_free_running):
                self.assertTrue(require_free_running)
                return self.promotion_validation(run_dir, expected_mode)

            with patch.dict(os.environ, environment, clear=False), patch.object(
                launcher, "validate_completed_run", side_effect=validate
            ) as validation:
                resolved = launcher.validate_promotion_run_roots(require_all=True)
            self.assertEqual(tuple(resolved), launcher.PROMOTION_MODES)
            self.assertEqual(validation.call_count, 3)
            for mode, run_dir in run_dirs.items():
                self.assertEqual(resolved[mode]["run_dir"], str(run_dir.resolve()))
                self.assertEqual(
                    resolved[mode]["run_root_source"],
                    launcher.PROMOTION_RUN_ROOT_ENV[mode],
                )

    def test_promotion_requires_explicit_roots_and_kaggle_inputs_when_hosted(self) -> None:
        launcher = load_launcher()
        with tempfile.TemporaryDirectory() as directory:
            input_root = Path(directory) / "input"
            input_root.mkdir()
            run_dirs = {
                mode: input_root / f"persisted-{mode}"
                for mode in launcher.PROMOTION_MODES
            }
            for run_dir in run_dirs.values():
                run_dir.mkdir()
                (run_dir / "SUCCESS").write_text("{}\n", encoding="utf-8")
            empty_environment = {
                env_name: "" for env_name in launcher.PROMOTION_RUN_ROOT_ENV.values()
            }

            def validate(run_dir, *, expected_mode, require_free_running):
                self.assertTrue(require_free_running)
                return self.promotion_validation(run_dir, expected_mode)

            with patch.dict(
                os.environ, empty_environment, clear=False
            ), self.assertRaisesRegex(FileNotFoundError, "unset"):
                launcher.validate_promotion_run_roots(require_all=True)

            explicit_environment = {
                launcher.PROMOTION_RUN_ROOT_ENV[mode]: str(run_dir)
                for mode, run_dir in run_dirs.items()
            }
            with patch.object(launcher, "KAGGLE_INPUT", input_root), patch.dict(
                os.environ, explicit_environment, clear=False
            ), patch.object(launcher, "validate_completed_run", side_effect=validate):
                resolved = launcher.validate_promotion_run_roots(
                    require_all=True,
                    require_attached=True,
                )
            for mode, run_dir in run_dirs.items():
                self.assertEqual(resolved[mode]["run_dir"], str(run_dir.resolve()))
                self.assertEqual(
                    resolved[mode]["run_root_source"],
                    launcher.PROMOTION_RUN_ROOT_ENV[mode],
                )

            outside = Path(directory) / "outside-block"
            outside.mkdir()
            (outside / "SUCCESS").write_text("{}\n", encoding="utf-8")
            outside_environment = dict(explicit_environment)
            outside_environment[launcher.PROMOTION_RUN_ROOT_ENV["block"]] = str(outside)
            with patch.object(launcher, "KAGGLE_INPUT", input_root), patch.dict(
                os.environ, outside_environment, clear=False
            ), self.assertRaisesRegex(ValueError, "attached Kaggle input"):
                launcher.validate_promotion_run_roots(
                    require_all=True,
                    require_attached=True,
                )

    def test_promotion_identity_comparison_fails_closed_by_category(self) -> None:
        launcher = load_launcher()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = {
                mode: self.promotion_validation(root / mode, mode)
                for mode in launcher.PROMOTION_MODES
            }
            mutations = {
                "matched design signature": lambda row: row.__setitem__(
                    "matched_design_signature", "a" * 64
                ),
                "parameter count": lambda row: row.__setitem__(
                    "parameter_count", int(row["parameter_count"]) + 1
                ),
                "full input normalization SHA256": lambda row: row.__setitem__(
                    "input_normalization_sha256", "b" * 64
                ),
                "source tree and Git identity": lambda row: row["source_identity"].__setitem__(
                    "git_commit", "c" * 40
                ),
                "full source identity SHA256": lambda row: row.__setitem__(
                    "source_identity_sha256", "d" * 64
                ),
                "contract identity": lambda row: row["contract_identity"].__setitem__(
                    "dataset_id", "different-target"
                ),
                "full contract identity SHA256": lambda row: row.__setitem__(
                    "contract_identity_sha256", "e" * 64
                ),
                "relation contract SHA256": lambda row: row.__setitem__(
                    "relation_contract_sha256", "d" * 64
                ),
                "sidecar schema SHA256": lambda row: row.__setitem__(
                    "sidecar_schema_sha256", "e" * 64
                ),
                "semantic runtime identity SHA256": lambda row: row.__setitem__(
                    "semantic_runtime_identity_sha256", "f" * 64
                ),
            }
            launcher.assert_matched_promotion_identity(baseline)
            for category, mutate in mutations.items():
                with self.subTest(category=category):
                    candidate = copy.deepcopy(baseline)
                    mutate(candidate["trajectory"])
                    if category in {
                        "relation contract SHA256",
                        "sidecar schema SHA256",
                    }:
                        with self.assertRaisesRegex(ValueError, "incomplete"):
                            launcher.assert_matched_promotion_identity(candidate)
                    else:
                        with self.assertRaisesRegex(RuntimeError, category):
                            launcher.assert_matched_promotion_identity(candidate)

    def test_run_promotion_uses_frozen_three_run_gate_and_persists_result(self) -> None:
        launcher = load_launcher()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = {
                mode: self.promotion_validation(root / mode, mode)
                for mode in launcher.PROMOTION_MODES
            }
            attempt_dir = root / "promotion-attempt"
            expected = {
                "promoted": True,
                "winner": "relational",
                "schema_version": "fixture",
            }
            with patch.object(
                launcher,
                "evaluate_multires_event_v2_promotion",
                return_value=expected,
            ) as evaluate:
                result = launcher.run_promotion(runs, attempt_dir)
            self.assertEqual(result, expected)
            self.assertEqual(
                json.loads((attempt_dir / "promotion.json").read_text(encoding="utf-8")),
                expected,
            )
            self.assertTrue((attempt_dir / "promotion_inputs.json").is_file())
            promotion_inputs = json.loads(
                (attempt_dir / "promotion_inputs.json").read_text(encoding="utf-8")
            )
            for mode in launcher.PROMOTION_MODES:
                persisted = promotion_inputs["runs"][mode]
                self.assertEqual(persisted["run_manifest_sha256"], "d" * 64)
                self.assertEqual(
                    persisted["optimizer_health_summary_sha256"], "e" * 64
                )
                self.assertEqual(persisted["metrics_sha256"], "f" * 64)
            kwargs = evaluate.call_args.kwargs
            self.assertEqual(kwargs["expected_anchors"], 6309)
            self.assertEqual(kwargs["bootstrap_repetitions"], 2000)
            self.assertEqual(kwargs["bootstrap_seed"], 20260713)
            self.assertEqual(
                kwargs["promotion_metric_contract"]["contract_version"],
                "2026-07-13-structural-promotion-v2",
            )
            for mode in launcher.PROMOTION_MODES:
                self.assertEqual(
                    kwargs[f"{mode}_teacher_path"],
                    (root / mode / "val_per_anchor_joint_nll.jsonl").resolve(),
                )
                self.assertEqual(
                    kwargs[f"{mode}_free_running_path"],
                    (root / mode / "free_running").resolve(),
                )


if __name__ == "__main__":
    unittest.main()
