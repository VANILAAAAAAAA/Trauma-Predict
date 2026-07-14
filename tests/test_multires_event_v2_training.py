from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

from trauma_predict.eval.multires_event_v2 import (
    evaluate_teacher_forced,
    exact_teacher_forced_loss,
    paired_subject_bootstrap_joint_nll,
    teacher_forced_model_inputs,
)
from trauma_predict.training.multires_event_v2 import (
    AUTHORIZED_TRAINING_RUN_NAMES,
    AUTHORIZED_VERIFICATION_RUN_NAMES,
    EXPECTED_FORMAL_MODEL_PARAMETER_COUNT,
    EXPECTED_OPTIMIZER_CONTRACT,
    MATCHED_MODES,
    OPTIMIZER_CONTRACT_VERSION,
    RAW_JOINT_NLL_REDUCTION,
    TRAINING_AUTHORIZED,
    VERIFICATION_AUTHORIZED,
    _audited_optimizer_step,
    _audit_optimizer_state_after_step,
    _audit_unscaled_gradients,
    _optimizer_step_health_payload,
    _save_v2_checkpoint,
    _hosted_verification_stop_step,
    _source_tree_identity,
    _verification_stop_after_formal_step2_requested,
    _verification_stop_after_resume_step3_requested,
    _validate_resume_optimizer_alignment,
    _validate_v2_checkpoint_integrity,
    _validated_optimizer_loss,
    build_multires_event_v2_model,
    build_multires_event_v2_optimizer,
    build_multires_event_v2_runtime,
    load_lab_scale_artifact,
    matched_design_signature,
    project_multires_event_v2_capacity_runtime,
    raw_414_factor_joint_nll_batch_mean,
    require_multires_event_v2_training_authorization,
    require_multires_event_v2_verification_authorization,
    _step_grad_scaler,
    validate_formal_model_parameter_count,
    validate_formal_target_field_order,
    validate_multires_event_v2_configs,
)
from trauma_predict.training.multires_event import _build_scheduler
from trauma_predict.training.multires_event_v2_loss import REGISTERED_CORE_FIELD_IDS
from trauma_predict.training.multires_event_v2_loss import V2_PRIMITIVE_FEEDBACK_DIMS


REPO_ROOT = Path(__file__).resolve().parents[1]


def _yaml(path: str) -> dict:
    return yaml.safe_load((REPO_ROOT / path).read_text(encoding="utf-8"))


class MultiresEventV2TrainingContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = _yaml("configs/dataset/multires_event_v2_c4.yaml")
        self.model = _yaml("configs/model/multires_event_v2_relational_primary.yaml")
        self.trains = {
            mode: _yaml(f"configs/train/t4x2_multires_event_v2_{mode}.yaml")
            for mode in MATCHED_MODES
        }

    def test_three_configs_are_mode_only_matched_and_exact_joint_nll(self) -> None:
        signatures = set()
        for mode, train in self.trains.items():
            validate_multires_event_v2_configs(train, self.dataset, self.model)
            self.assertEqual(train["mode"], mode)
            self.assertEqual(train["objective"]["stochastic_primitive_factors"], 414)
            self.assertEqual(train["objective"]["factor_composition"], "joint_log_probability_sum")
            self.assertFalse(train["objective"]["active_target_denominator"])
            self.assertIsNone(train["objective"]["family_weights"])
            self.assertEqual(train["evaluation"]["interval_anchor_policy"], "all_validation_anchors")
            self.assertEqual(train["evaluation"]["interval_expected_samples"], 6309)
            self.assertEqual(train["evaluation"]["final_expected_samples"], 6309)
            self.assertEqual(train["comparison"]["bootstrap_repetitions"], 2000)
            self.assertEqual(train["training"]["per_device_train_batch_size"], 32)
            self.assertEqual(train["training"]["per_device_eval_batch_size"], 32)
            self.assertEqual(train["training"]["gradient_accumulation_steps"], 1)
            self.assertEqual(train["training"]["train_samples_per_epoch"], 3072)
            self.assertEqual(train["training"]["grad_scaler_initial_scale"], 32.0)
            self.assertEqual(train["training"]["grad_scaler_growth_interval"], 1_000_000)
            self.assertEqual(train["training"]["max_consecutive_scaler_skips"], 0)
            self.assertEqual(
                train["training"]["optimizer_contract_version"],
                OPTIMIZER_CONTRACT_VERSION,
            )
            self.assertEqual(
                train["training"]["loss_reduction"], RAW_JOINT_NLL_REDUCTION
            )
            self.assertEqual(train["training"]["gradient_clipping"], "disabled")
            self.assertNotIn("max_grad_norm", train["training"])
            self.assertEqual(
                train["training"]["grad_scaler_overflow_policy"],
                "fail_run_preserve_matched_rows",
            )
            self.assertEqual(
                train["training"]["per_device_train_batch_size"]
                * train["training"]["required_world_size"]
                * train["training"]["gradient_accumulation_steps"],
                64,
            )
            self.assertEqual(
                train["evaluation"]["free_running_trajectory_batch_size"], 100
            )
            signatures.add(matched_design_signature(train, self.dataset, self.model))
        self.assertEqual(len(signatures), 1)

    def test_hosted_step2_stop_is_explicit_and_fail_closed(self) -> None:
        with patch.dict(
            os.environ,
            {"TRAUMA_PREDICT_V2_HOSTED_VERIFY_STOP_AFTER_FORMAL_STEP2": "1"},
        ):
            self.assertTrue(_verification_stop_after_formal_step2_requested())
        with patch.dict(
            os.environ,
            {"TRAUMA_PREDICT_V2_HOSTED_VERIFY_STOP_AFTER_FORMAL_STEP2": "0"},
        ):
            self.assertFalse(_verification_stop_after_formal_step2_requested())
        with patch.dict(
            os.environ,
            {"TRAUMA_PREDICT_V2_HOSTED_VERIFY_STOP_AFTER_FORMAL_STEP2": "yes"},
        ), self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            _verification_stop_after_formal_step2_requested()

        with patch.dict(
            os.environ,
            {
                "TRAUMA_PREDICT_V2_HOSTED_VERIFY_STOP_AFTER_FORMAL_STEP2": "0",
                "TRAUMA_PREDICT_V2_HOSTED_VERIFY_STOP_AFTER_RESUME_STEP3": "1",
            },
        ):
            self.assertTrue(_verification_stop_after_resume_step3_requested())
            self.assertEqual(
                _hosted_verification_stop_step(starting_global_step=2), 3
            )
            with self.assertRaisesRegex(ValueError, "must restore optimizer step 2"):
                _hosted_verification_stop_step(starting_global_step=0)

    def test_source_release_accepts_git_sha1_without_a_git_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "Trauma-Predict"
            shutil.copytree(REPO_ROOT / "src/trauma_predict", release / "src/trauma_predict")
            entry = release / "notebooks/kaggle/train_relational_primary.py"
            entry.parent.mkdir(parents=True)
            shutil.copy2(
                REPO_ROOT / "notebooks/kaggle/train_relational_primary.py", entry
            )
            for name in ("requirements-multires-kaggle.txt", "pyproject.toml"):
                shutil.copy2(REPO_ROOT / name, release / name)
            identity = _source_tree_identity(release)
            (release / "SOURCE_RELEASE.json").write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "trauma_predict.multires_event_v2_source_release.v1"
                        ),
                        "git_commit": "1" * 40,
                        "git_head_tree": "2" * 40,
                        "source_tree_sha256": identity["source_tree_sha256"],
                    }
                ),
                encoding="utf-8",
            )
            released = _source_tree_identity(release)
            self.assertEqual(released["git_commit"], "1" * 40)
            self.assertEqual(released["git_head_tree"], "2" * 40)
            self.assertTrue(released["git_clean"])

    def test_all_four_configs_share_the_explicit_optimizer_contract(self) -> None:
        payloads = []
        for suffix in ("smoke", *MATCHED_MODES):
            train = _yaml(f"configs/train/t4x2_multires_event_v2_{suffix}.yaml")
            validate_multires_event_v2_configs(train, self.dataset, self.model)
            payloads.append(
                {key: train["training"][key] for key in EXPECTED_OPTIMIZER_CONTRACT}
            )
            self.assertNotIn("max_grad_norm", train["training"])
        self.assertTrue(all(payload == EXPECTED_OPTIMIZER_CONTRACT for payload in payloads))

    def test_optimizer_contract_rejects_every_config_mutation_and_clipping_alias(self) -> None:
        mutations = {
            "optimizer_contract_version": "trauma_predict.multires_event_v2_optimizer.v2",
            "loss_reduction": "weighted_family_mean",
            "optimizer": "SGD",
            "learning_rate": 1.0e-4,
            "weight_decay": 0.02,
            "adamw_betas": [0.8, 0.999],
            "adamw_eps": 1.0e-7,
            "adamw_amsgrad": True,
            "adamw_maximize": True,
            "adamw_foreach": True,
            "adamw_fused": True,
            "gradient_clipping": "global_norm",
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                train = copy.deepcopy(self.trains["trajectory"])
                train["training"][key] = value
                with self.assertRaisesRegex(ValueError, f"training.{key}"):
                    validate_multires_event_v2_configs(train, self.dataset, self.model)
        train = copy.deepcopy(self.trains["trajectory"])
        train["training"]["max_grad_norm"] = 1.0
        with self.assertRaisesRegex(ValueError, "max_grad_norm is forbidden"):
            validate_multires_event_v2_configs(train, self.dataset, self.model)

    def test_smoke_and_formal_profiles_reject_cross_role_drift(self) -> None:
        smoke = _yaml("configs/train/t4x2_multires_event_v2_smoke.yaml")
        mutations = (
            ("smoke mode", smoke, ("mode",), "block"),
            ("smoke steps", smoke, ("training", "max_steps"), 4000),
            ("smoke resume", smoke, ("training", "resume"), True),
            (
                "smoke output",
                smoke,
                ("outputs", "output_dir"),
                "${TRAUMA_PREDICT_OUTPUT_ROOT}/t4x2_multires_event_v2_block",
            ),
            (
                "formal run name",
                self.trains["block"],
                ("run_name",),
                "t4x2_multires_event_v2_smoke",
            ),
            ("formal schedule", self.trains["block"], ("training", "eval_steps"), 1),
            (
                "formal subjects",
                self.trains["block"],
                ("training", "max_train_subjects"),
                16,
            ),
        )
        for label, source, path, value in mutations:
            with self.subTest(label=label):
                candidate = copy.deepcopy(source)
                target = candidate
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    validate_multires_event_v2_configs(candidate, self.dataset, self.model)

    def test_core_authorizes_only_relational_primary_and_no_ablation_gate(self) -> None:
        self.assertTrue(TRAINING_AUTHORIZED)
        self.assertEqual(
            AUTHORIZED_TRAINING_RUN_NAMES,
            ("t4x2_multires_event_v2_relational",),
        )
        require_multires_event_v2_training_authorization(self.trains["relational"])
        for mode in ("block", "trajectory"):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                RuntimeError, "not authorized for run_name"
            ):
                require_multires_event_v2_training_authorization(self.trains[mode])
        comparison = self.trains["relational"]["comparison"]
        self.assertEqual(comparison["primary_training_order"], ["relational"])
        self.assertFalse(comparison["ablations_are_prerequisites"])
        self.assertEqual(comparison["promotion_gate"], "none_for_primary_training")
        self.assertEqual(
            self.trains["relational"]["training"]["initial_checkpoint_step"],
            2,
        )

        self.assertTrue(VERIFICATION_AUTHORIZED)
        self.assertEqual(
            AUTHORIZED_VERIFICATION_RUN_NAMES,
            ("t4x2_multires_event_v2_relational",),
        )
        require_multires_event_v2_verification_authorization(
            self.trains["relational"]
        )

    def test_resume_alignment_binds_adam_step_scheduler_epoch_and_lr(self) -> None:
        training = self.trains["trajectory"]["training"]
        model = torch.nn.Linear(2, 1)
        optimizer = build_multires_event_v2_optimizer(model, training)
        scheduler = _build_scheduler(optimizer, training)
        fresh = _validate_resume_optimizer_alignment(
            optimizer, scheduler, training, global_step=0
        )
        self.assertEqual(fresh["optimizer_state_entries"], 0)
        model(torch.ones(2, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        resumed = _validate_resume_optimizer_alignment(
            optimizer, scheduler, training, global_step=1
        )
        self.assertEqual(resumed["expected_optimizer_step"], 1)
        scheduler.last_epoch = 2
        with self.assertRaisesRegex(RuntimeError, "scheduler.last_epoch"):
            _validate_resume_optimizer_alignment(
                optimizer, scheduler, training, global_step=1
            )
        scheduler.last_epoch = 1
        next(iter(optimizer.state.values()))["step"].fill_(2)
        with self.assertRaisesRegex(RuntimeError, "state steps"):
            _validate_resume_optimizer_alignment(
                optimizer, scheduler, training, global_step=1
            )

        for state in optimizer.state.values():
            state["step"].fill_(4000)
        scheduler.last_epoch = 4000
        optimizer.param_groups[0]["lr"] = 0.0
        completed = _validate_resume_optimizer_alignment(
            optimizer, scheduler, training, global_step=4000
        )
        self.assertEqual(completed["observed_optimizer_step_min"], 4000.0)
        self.assertEqual(completed["expected_learning_rate"], 0.0)
        self.assertEqual(completed["observed_learning_rate"], 0.0)

    def test_v2_checkpoint_is_hash_bound_before_resume(self) -> None:
        class State:
            def __init__(self, payload):
                self.payload = payload

            def state_dict(self):
                return self.payload

        class Sampler:
            def state_dict(self):
                return {"epoch": 3}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
            runtime = SimpleNamespace(train_sampler=Sampler())
            _save_v2_checkpoint(
                output_dir=root,
                model=model,
                optimizer=optimizer,
                scheduler=State({"last_epoch": 1}),
                scaler=State({"scale": 32.0}),
                trainer_state={"global_step": 1},
                identity_hashes={"source": "a" * 64},
                runtime=runtime,
                rank=0,
                keep_last=2,
            )
            checkpoint = root / "checkpoints/checkpoint-00000001"
            self.assertTrue(checkpoint.is_dir())
            _validate_v2_checkpoint_integrity(root, expected_world_size=1)
            model_path = checkpoint / "model.pt"
            model_path.write_bytes(model_path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "file/hash"):
                _validate_v2_checkpoint_integrity(root, expected_world_size=1)

    def test_raw_joint_nll_and_adamw_builder_are_exact_and_auditable(self) -> None:
        primitive_log_prob = torch.randn(3, 414, requires_grad=True)
        loss = raw_414_factor_joint_nll_batch_mean(
            {
                "primitive_count": 414,
                "primitive_log_prob": primitive_log_prob,
                "loss": torch.tensor(-999.0),
            }
        )
        torch.testing.assert_close(
            loss,
            -primitive_log_prob.sum(dim=-1).mean(),
            rtol=0.0,
            atol=0.0,
        )
        with self.assertRaisesRegex(ValueError, r"\[batch,414\]"):
            raw_414_factor_joint_nll_batch_mean(
                {
                    "primitive_count": 414,
                    "primitive_log_prob": torch.zeros(3, 413),
                }
            )

        model = torch.nn.Linear(3, 2)
        training = self.trains["trajectory"]["training"]
        optimizer = build_multires_event_v2_optimizer(model, training)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.999))
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.defaults["lr"], 2.0e-4)
        self.assertEqual(optimizer.defaults["weight_decay"], 0.01)
        self.assertEqual(optimizer.defaults["eps"], 1.0e-8)
        self.assertFalse(optimizer.defaults["amsgrad"])
        self.assertFalse(optimizer.defaults["maximize"])
        self.assertFalse(optimizer.defaults["foreach"])
        self.assertFalse(optimizer.defaults["fused"])
        output = model(torch.ones(4, 3)).square().mean()
        output.backward()
        gradient_health, probe = _audit_unscaled_gradients(model)
        self.assertGreater(gradient_health["global_l2_norm"], 0.0)
        self.assertEqual(gradient_health["gradient_clipping"], "disabled")
        optimizer.step()
        state_health = _audit_optimizer_state_after_step(
            model, optimizer, probe, expected_optimizer_step=1
        )
        self.assertTrue(state_health["probe_parameter_changed"])
        self.assertTrue(state_health["exp_avg_sq_nonnegative"])
        self.assertEqual(
            state_health["optimizer_configuration"]["parameter_group_count"], 1
        )

    def test_warmup_step_lr_is_persisted_before_scheduler_advance(self) -> None:
        model = torch.nn.Linear(3, 2)
        training = self.trains["trajectory"]["training"]
        optimizer = build_multires_event_v2_optimizer(model, training)
        scheduler = _build_scheduler(optimizer, training)
        learning_rates = []
        for step in range(1, 3):
            optimizer.zero_grad(set_to_none=True)
            model(torch.ones(4, 3)).square().mean().backward()
            gradient_health, probe = _audit_unscaled_gradients(model)
            gradient_health["audit_wall_seconds"] = 0.01
            optimizer.step()
            state_health = _audit_optimizer_state_after_step(
                model, optimizer, probe, expected_optimizer_step=step
            )
            state_health["audit_wall_seconds"] = 0.02
            payload = _optimizer_step_health_payload(
                optimizer, gradient_health, state_health, training=training
            )
            learning_rates.append(payload["learning_rate_used"])
            self.assertGreater(payload["optimizer_audit_wall_seconds"], 0.0)
            scheduler.step()
        self.assertAlmostEqual(learning_rates[0], 5.0e-7, places=15)
        self.assertAlmostEqual(learning_rates[1], 1.0e-6, places=15)

    def test_optimizer_loss_rejects_nonfinite_or_wrong_sized_per_sample_values(self) -> None:
        valid = {
            "primitive_count": 414,
            "primitive_log_prob": torch.zeros(2, 414),
            "per_sample_nll": torch.zeros(2),
            "loss": torch.zeros(()),
        }
        self.assertTrue(
            torch.isfinite(_validated_optimizer_loss(valid, expected_local_batch=2))
        )
        wrong_size = dict(valid, per_sample_nll=torch.zeros(1))
        with self.assertRaisesRegex(ValueError, "count must equal"):
            _validated_optimizer_loss(wrong_size, expected_local_batch=2)
        nonfinite_sample = dict(valid, per_sample_nll=torch.tensor([0.0, float("nan")]))
        with self.assertRaisesRegex(FloatingPointError, "finite and algebraically"):
            _validated_optimizer_loss(nonfinite_sample, expected_local_batch=2)
        nonfinite_loss = dict(valid, primitive_log_prob=torch.full((2, 414), float("inf")))
        with self.assertRaisesRegex(FloatingPointError, "optimizer loss"):
            _validated_optimizer_loss(nonfinite_loss, expected_local_batch=2)
        inconsistent = dict(valid, per_sample_nll=torch.ones(2), loss=torch.ones(()))
        with self.assertRaisesRegex(FloatingPointError, "algebraically identical"):
            _validated_optimizer_loss(inconsistent, expected_local_batch=2)
        broadcastable_wrong_batch = dict(
            valid,
            primitive_log_prob=torch.zeros(1, 414),
        )
        with self.assertRaisesRegex(ValueError, "exact local batch"):
            _validated_optimizer_loss(
                broadcastable_wrong_batch, expected_local_batch=2
            )

    def test_optimizer_health_fails_closed_on_gradient_and_state_corruption(self) -> None:
        training = self.trains["trajectory"]["training"]

        def ready_model():
            model = torch.nn.Linear(3, 2)
            optimizer = build_multires_event_v2_optimizer(model, training)
            model(torch.ones(4, 3)).square().mean().backward()
            return model, optimizer

        model, _ = ready_model()
        model.bias.grad = None
        with self.assertRaisesRegex(RuntimeError, "every trainable parameter gradient"):
            _audit_unscaled_gradients(model)

        model, _ = ready_model()
        for parameter in model.parameters():
            parameter.grad.zero_()
        with self.assertRaisesRegex(FloatingPointError, "must be positive"):
            _audit_unscaled_gradients(model)

        model, _ = ready_model()
        model.weight.grad.reshape(-1)[0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            _audit_unscaled_gradients(model)

        model, optimizer = ready_model()
        _, probe = _audit_unscaled_gradients(model)
        optimizer.step()
        optimizer.state[model.weight]["exp_avg_sq"].reshape(-1)[0] = -1.0
        with self.assertRaisesRegex(FloatingPointError, "health audit failed"):
            _audit_optimizer_state_after_step(
                model, optimizer, probe, expected_optimizer_step=1
            )

        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(1.0)
        optimizer = build_multires_event_v2_optimizer(model, training)
        _build_scheduler(optimizer, training)
        model.weight.grad = torch.full_like(model.weight, 1.0e-12)
        _, probe = _audit_unscaled_gradients(model)
        optimizer.step()
        state_health = _audit_optimizer_state_after_step(
            model, optimizer, probe, expected_optimizer_step=1
        )
        self.assertFalse(state_health["probe_parameter_changed"])
        self.assertTrue(state_health["state_steps_complete_equal_expected"])

    def test_dataset_config_rejects_relation_and_sidecar_schema_identity_drift(self) -> None:
        for key in ("relation_contract_sha256", "sidecar_schema_sha256"):
            with self.subTest(key=key):
                dataset = copy.deepcopy(self.dataset)
                dataset["target"][key] = "0" * 64
                with self.assertRaisesRegex(ValueError, f"dataset.target.{key}"):
                    validate_multires_event_v2_configs(
                        self.trains["trajectory"], dataset, self.model
                    )

    def test_grad_scaler_step_distinguishes_skipped_attempt_from_real_update(self) -> None:
        class Scaler:
            def __init__(self, before: float, after: float) -> None:
                self.scale = before
                self.after = after
                self.steps = 0

            def get_scale(self) -> float:
                return self.scale

            def step(self, _optimizer: object) -> None:
                self.steps += 1

            def update(self) -> None:
                self.scale = self.after

        skipped = Scaler(128.0, 64.0)
        updated = Scaler(128.0, 128.0)
        grown = Scaler(128.0, 256.0)
        self.assertEqual(_step_grad_scaler(skipped, object()), (False, 128.0, 64.0))
        self.assertEqual(_step_grad_scaler(updated, object()), (True, 128.0, 128.0))
        self.assertEqual(_step_grad_scaler(grown, object()), (True, 128.0, 256.0))

    def test_audited_optimizer_step_rejects_any_scaler_drift_from_32(self) -> None:
        class Scaler:
            def __init__(self, scale: float) -> None:
                self.scale = scale

            def get_scale(self) -> float:
                return self.scale

            def step(self, optimizer) -> None:
                optimizer.step()

            def update(self) -> None:
                return None

        model = torch.nn.Linear(2, 1)
        optimizer = build_multires_event_v2_optimizer(
            model, self.trains["trajectory"]["training"]
        )
        model(torch.ones(2, 2)).square().mean().backward()
        with self.assertRaisesRegex(FloatingPointError, "exactly 32.0"):
            _audited_optimizer_step(
                model, optimizer, Scaler(64.0), expected_optimizer_step=1
            )

    def test_capacity_projection_preserves_all_formal_validation_passes(self) -> None:
        projection = project_multires_event_v2_capacity_runtime(
            self.trains["trajectory"]["training"],
            optimizer_step_seconds=(1.0, 1.5),
            teacher_probe_seconds=100.0,
            free_running_probe_seconds=200.0,
        )
        self.assertEqual(projection["formal_max_steps"], 4000)
        self.assertEqual(projection["formal_eval_steps"], 250)
        self.assertEqual(projection["interval_teacher_passes"], 16)
        self.assertEqual(projection["final_teacher_passes"], 1)
        self.assertEqual(projection["total_teacher_passes"], 17)
        self.assertEqual(projection["optimizer_seconds_per_step"], 1.5)
        self.assertAlmostEqual(
            projection["components_seconds"]["teacher_forced"],
            6309.0 * 17.0,
        )
        self.assertAlmostEqual(
            projection["components_seconds"]["free_running"],
            6309.0 * 2.0,
        )
        self.assertAlmostEqual(
            projection["projected_formal_runtime_seconds"],
            sum(projection["components_seconds"].values()),
        )
        with self.assertRaisesRegex(ValueError, "exactly two"):
            project_multires_event_v2_capacity_runtime(
                self.trains["trajectory"]["training"],
                optimizer_step_seconds=(1.0,),
                teacher_probe_seconds=100.0,
                free_running_probe_seconds=200.0,
            )

    def test_full_r9_and_immutable_v1_identities_are_frozen(self) -> None:
        self.assertEqual(
            self.dataset["target"]["dataset_id"],
            "multires_event_m4_target_v2_c4_full_20260714_r9",
        )
        self.assertEqual(
            self.dataset["target"]["dataset_manifest_sha256"],
            "6c4e1e300686195fb2c58bfcbd74df6c7cb905d7031985cb7a7624d5c7061f1e",
        )
        self.assertEqual(
            self.dataset["base"]["dataset_manifest_sha256"],
            "4e7742900907e0e2f774099ba1dd485468210ff3da9ddaef3ec3bf67957000c3",
        )
        self.assertEqual(self.dataset["split_authority"], "base/sample_manifest.csv")
        self.assertEqual(self.dataset["join_guards"], ["base_content_hash", "target_content_hash"])

    def test_v2_checkpoint_runtime_requests_v1_final_dataset_not_v1_interval_sampler(self) -> None:
        dataset = copy.deepcopy(self.dataset)
        dataset["base"]["root"] = "/tmp/v2-unit-base"
        dataset["target"]["root"] = "/tmp/v2-unit-target"
        dataset["normalization"]["path"] = "/tmp/v2-unit-normalization.json"
        captured = {}

        class StopAfterV1Config(RuntimeError):
            pass

        def capture(config, *_args, **_kwargs):
            captured.update(config)
            raise StopAfterV1Config

        with (
            patch("trauma_predict.training.multires_event_v2._verify_artifact_files"),
            patch(
                "trauma_predict.training.multires_event_v2.build_v1_runtime",
                side_effect=capture,
            ),
            self.assertRaises(StopAfterV1Config),
        ):
            build_multires_event_v2_runtime(
                self.trains["trajectory"],
                dataset,
                repo_root=REPO_ROOT,
                rank=0,
                world_size=1,
                phase="interval",
            )
        self.assertEqual(
            captured["evaluation"],
            {"phase": "final", "final_expected_samples": 6309},
        )

    def test_model_parameter_structure_is_identical_across_modes(self) -> None:
        compact = copy.deepcopy(self.model)
        compact["architecture"].update(
            hidden_size=8,
            num_attention_heads=2,
            trajectory_encoder_layers=1,
            target_decoder_layers=1,
            block_latent_count=2,
            dropout=0.0,
        )
        signatures = []
        for mode in MATCHED_MODES:
            model = build_multires_event_v2_model(compact, mode=mode)
            signatures.append(
                tuple((name, tuple(value.shape)) for name, value in model.state_dict().items())
            )
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[1], signatures[2])

    def test_all_four_formal_configs_build_the_exact_frozen_parameter_count(self) -> None:
        config_paths = (
            "configs/train/t4x2_multires_event_v2_smoke.yaml",
            "configs/train/t4x2_multires_event_v2_block.yaml",
            "configs/train/t4x2_multires_event_v2_trajectory.yaml",
            "configs/train/t4x2_multires_event_v2_relational.yaml",
        )
        for path in config_paths:
            train = _yaml(path)
            validate_multires_event_v2_configs(train, self.dataset, self.model)
            model = build_multires_event_v2_model(self.model, mode=str(train["mode"]))
            self.assertEqual(
                validate_formal_model_parameter_count(model),
                EXPECTED_FORMAL_MODEL_PARAMETER_COUNT,
                path,
            )
            del model

    def test_formal_config_rejects_target_field_order_drift(self) -> None:
        drifted = copy.deepcopy(self.model)
        ids = list(drifted["architecture"]["target_field_ids"])
        ids[0], ids[1] = ids[1], ids[0]
        drifted["architecture"]["target_field_ids"] = ids
        with self.assertRaisesRegex(ValueError, "exactly match the ordered full_r9"):
            validate_multires_event_v2_configs(
                self.trains["trajectory"], self.dataset, drifted
            )

    def test_runtime_guard_compares_config_to_mounted_contract_in_exact_order(self) -> None:
        contract = SimpleNamespace(
            registered_core_field_ids=tuple(reversed(REGISTERED_CORE_FIELD_IDS))
        )
        with self.assertRaisesRegex(ValueError, "exactly match full_r9"):
            validate_formal_target_field_order(self.model, contract)

    def test_parameter_guard_rejects_even_one_parameter_of_drift(self) -> None:
        model = build_multires_event_v2_model(self.model, mode="trajectory")
        model.register_parameter(
            "readiness_guard_drift",
            torch.nn.Parameter(torch.zeros(1)),
        )
        with self.assertRaisesRegex(ValueError, "parameter count differs"):
            validate_formal_model_parameter_count(model)

    def test_formal_builder_applies_parameter_guard_automatically(self) -> None:
        with (
            patch(
                "trauma_predict.training.multires_event_v2."
                "EXPECTED_FORMAL_MODEL_PARAMETER_COUNT",
                EXPECTED_FORMAL_MODEL_PARAMETER_COUNT + 1,
            ),
            self.assertRaisesRegex(ValueError, "parameter count differs"),
        ):
            build_multires_event_v2_model(self.model, mode="block")

    def test_teacher_adapter_adds_scalar_component_axis_without_mutating_loss_truth(self) -> None:
        leading = (2, 6, 29)
        primitives = {
            key: torch.zeros((*leading, width)) if width > 1 else torch.zeros(leading)
            for key, width in V2_PRIMITIVE_FEEDBACK_DIMS.items()
        }
        masks = {key: torch.ones(leading, dtype=torch.bool) for key in primitives}
        input_batch = {
            key: torch.zeros((2, 1))
            for key in (
                "event_field_ids", "event_operator_ids", "event_condition_ids",
                "event_values", "event_value_mask", "event_study_slot_ids", "block_index",
                "event_mask", "block_role_ids", "resolution_ids", "relative_start",
                "relative_end", "span", "block_mask", "static_numeric",
                "static_numeric_mask", "static_categorical",
            )
        }
        batch = {
            "input_batch": input_batch,
            "target_primitives": primitives,
            "target_primitive_masks": masks,
            "relation_adjacency": torch.zeros((14, 29, 29), dtype=torch.bool),
            "relation_type_lags": torch.zeros(14, dtype=torch.long),
        }
        inputs = teacher_forced_model_inputs(batch, mode="trajectory")
        scalar_key = "categorical_hours_0_4"
        self.assertEqual(primitives[scalar_key].shape, leading)
        self.assertEqual(inputs["target_primitives"][scalar_key].shape, (*leading, 1))
        self.assertIs(batch["target_primitives"], primitives)

    def test_emission_parameters_are_promoted_to_float32_before_joint_nll(self) -> None:
        class Model:
            def __call__(self, **kwargs):
                return {"primitive_parameters": {"bank": torch.ones(1, dtype=torch.float16)}}

        leading = (1, 6, 29)
        primitives = {
            key: torch.zeros((*leading, width)) if width > 1 else torch.zeros(leading)
            for key, width in V2_PRIMITIVE_FEEDBACK_DIMS.items()
        }
        batch = {
            "input_batch": {
                key: torch.zeros((1, 1))
                for key in (
                    "event_field_ids", "event_operator_ids", "event_condition_ids",
                    "event_values", "event_value_mask", "event_study_slot_ids", "block_index",
                    "event_mask", "block_role_ids", "resolution_ids", "relative_start",
                    "relative_end", "span", "block_mask", "static_numeric",
                    "static_numeric_mask", "static_categorical",
                )
            },
            "target_primitives": primitives,
            "target_primitive_masks": {
                key: torch.ones(leading, dtype=torch.bool) for key in primitives
            },
            "relation_adjacency": torch.zeros((14, 29, 29), dtype=torch.bool),
            "relation_type_lags": torch.zeros(14, dtype=torch.long),
        }

        def audited_loss(outputs, *_args, **_kwargs):
            self.assertEqual(outputs["primitive_parameters"]["bank"].dtype, torch.float32)
            return {
                "loss": torch.tensor(1.0),
                "per_sample_nll": torch.tensor([1.0]),
                "primitive_count": 414,
            }

        with patch(
            "trauma_predict.eval.multires_event_v2.compute_registry_multires_event_v2_loss",
            side_effect=audited_loss,
        ):
            _, result = exact_teacher_forced_loss(
                Model(),
                batch,
                {},
                mode="trajectory",
                expected_lab_scale_artifact_hash="a" * 64,
            )
        self.assertEqual(result["primitive_count"], 414)

    def test_subject_macro_is_anchor_mean_within_subject_then_macro(self) -> None:
        class Model:
            def eval(self):
                return self

        loader = [
            {
                "sample_id": ["a", "b", "c"],
                "subject_id": ["p1", "p1", "p2"],
                "nll": torch.tensor([2.0, 4.0, 10.0]),
            }
        ]

        def fake_loss(_model, batch, _registry, **_kwargs):
            return {}, {"per_sample_nll": batch["nll"], "primitive_count": 414}

        with patch(
            "trauma_predict.eval.multires_event_v2.exact_teacher_forced_loss",
            side_effect=fake_loss,
        ):
            result = evaluate_teacher_forced(
                model=Model(),
                loader=loader,
                registry={},
                device=torch.device("cpu"),
                mode="trajectory",
                expected_samples=3,
                phase="final",
                step=7,
                precision="fp16",
            )
        self.assertAlmostEqual(result["joint_nll_anchor_mean"], 16.0 / 3.0)
        self.assertAlmostEqual(result["joint_nll_subject_macro"], 6.5)
        self.assertFalse(result["active_target_denominator"])

    def test_lab_scale_loader_binds_train_only_content_hash_to_sidecar(self) -> None:
        labs = (
            "lactate", "base_excess", "bicarbonate", "creatinine", "bun", "wbc",
            "hemoglobin", "platelet_count", "inr", "sodium", "potassium", "chloride",
            "glucose",
        )
        units = {field: f"unit-{field}" for field in labs}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset_manifest.json").write_text("{}\n", encoding="utf-8")
            manifest_sha = hashlib.sha256((root / "dataset_manifest.json").read_bytes()).hexdigest()
            contract = SimpleNamespace(
                dataset_root=root,
                manifest={
                    "dataset_id": "sidecar-r4",
                    "files": {"sample_manifest": {"sha256": "2" * 64}},
                },
                contract_bundle_hash="3" * 64,
                contract_hashes={"process": "4" * 64, "emission": "5" * 64},
                lab_fields=labs,
                core_fields=labs,
                registered_core_field_ids=tuple(range(14, 27)),
                emission_registry={
                    "field_supports": {
                        "intermittent_labs": {
                            field: {"unit": unit, "finite_support": None}
                            for field, unit in units.items()
                        }
                    }
                },
            )
            payload = {
                "schema": "multires_event_v2_lab_affine_scale_v1",
                "version": "2026-07-13-train-target-windows-v1",
                "status": "frozen_train_only_fit",
                "fit_split": "train",
                "coordinate_contract": "lab_shared_affine_canonical_v1",
                "transform": {
                    "forward": "z=(x-center)/scale",
                    "inverse": "x=center+scale*z",
                    "clipping": "forbidden",
                    "center": "linear_interpolation_median_of_fit_multiset",
                    "scale": "q75_minus_q25_of_fit_multiset",
                    "scale_fallback": "none_fail_if_nonpositive",
                    "shared_coordinates": ["last", "min", "max"],
                },
                "fit_population": {
                    "authority": "persisted_full_sidecar_train_target_shards",
                    "physical_window_key": [
                        "subject_id", "stay_id", "absolute_start_hour", "absolute_end_hour", "field"
                    ],
                    "duplicate_truth_policy": "require_exact_canonical_json_then_count_once",
                    "coordinate_multiset_per_active_unique_window": ["last", "min", "max"],
                    "train_samples": 2,
                    "train_subjects": 1,
                    "train_subject_ids_sha256": "6" * 64,
                    "unique_physical_field_windows": 13,
                    "collapsed_duplicate_field_windows": 0,
                    "window_truth_ledger_sha256": "7" * 64,
                },
                "source": {
                    "sidecar_dataset_id": "sidecar-r4",
                    "sidecar_dataset_manifest_sha256": manifest_sha,
                    "sidecar_sample_manifest_sha256": "2" * 64,
                    "sidecar_contract_bundle_hash": "3" * 64,
                    "sidecar_process_contract_sha256": "4" * 64,
                    "sidecar_emission_contract_sha256": "5" * 64,
                    "process_registry_sha256": "4" * 64,
                    "v1_element_registry_sha256": "8" * 64,
                },
                "field_order": list(labs),
                "fields": {
                    field: {
                        "field": field,
                        "field_id": index,
                        "unit": units[field],
                        "center": 10.0,
                        "scale": 2.0,
                        "q25": 9.0,
                        "q75": 11.0,
                        "coordinate_count": 3,
                        "unique_window_count": 1,
                    }
                    for index, field in enumerate(labs, start=14)
                },
            }
            canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            payload["content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            path = root / "lab-scale.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            compact = load_lab_scale_artifact(
                path,
                expected_content_sha256=payload["content_sha256"],
                contract=contract,
            )
            self.assertEqual(compact["content_sha256"], payload["content_sha256"])
            self.assertEqual(compact["fields"]["lactate"]["scale"], 2.0)
            with self.assertRaisesRegex(ValueError, "training-config run identity"):
                load_lab_scale_artifact(path, expected_content_sha256="0" * 64, contract=contract)

    def test_paired_bootstrap_requires_identical_rows_and_resamples_subjects(self) -> None:
        control_rows = [
            {"sample_id": "a", "subject_id": "p1", "joint_nll": 5.0},
            {"sample_id": "b", "subject_id": "p1", "joint_nll": 5.0},
            {"sample_id": "c", "subject_id": "p2", "joint_nll": 10.0},
        ]
        candidate_rows = [
            {"sample_id": "a", "subject_id": "p1", "joint_nll": 4.0},
            {"sample_id": "b", "subject_id": "p1", "joint_nll": 4.0},
            {"sample_id": "c", "subject_id": "p2", "joint_nll": 8.0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "control.jsonl"
            candidate = root / "candidate.jsonl"
            control.write_text(
                "".join(json.dumps(row) + "\n" for row in control_rows), encoding="utf-8"
            )
            candidate.write_text(
                "".join(json.dumps(row) + "\n" for row in candidate_rows), encoding="utf-8"
            )
            result = paired_subject_bootstrap_joint_nll(
                control, candidate, expected_anchors=3
            )
            self.assertAlmostEqual(result["observed_delta"], -1.5)
            self.assertTrue(result["ci95_upper_lt_zero"])
            self.assertFalse(result["promotion_contract_valid"])
            self.assertIsNone(result["promotion_decision"])
            candidate.write_text(
                "".join(json.dumps(row) + "\n" for row in candidate_rows[:-1]), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "identical persisted validation anchors"):
                paired_subject_bootstrap_joint_nll(control, candidate, expected_anchors=3)


if __name__ == "__main__":
    unittest.main()
