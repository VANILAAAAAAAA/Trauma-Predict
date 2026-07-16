from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from trauma_predict.data.multires_event_v2 import MultiresEventV2RelationContract
from trauma_predict.eval.multires_event_v2 import (
    evaluate_teacher_forced,
    exact_teacher_forced_loss,
    teacher_forced_model_inputs,
)
from trauma_predict.training.multires_event import _build_scheduler
from trauma_predict.training.multires_event_v2 import (
    AUTHORIZED_TRAINING_RUN_NAMES,
    EXPECTED_FORMAL_MODEL_PARAMETER_COUNT,
    EXPECTED_OPTIMIZER_CONTRACT,
    RAW_JOINT_NLL_REDUCTION,
    _audited_optimizer_step,
    _audit_optimizer_state_after_step,
    _audit_unscaled_gradients,
    _optimizer_step_health_payload,
    _save_v2_checkpoint,
    _step_grad_scaler,
    _validate_resume_optimizer_alignment,
    _validate_v2_checkpoint_integrity,
    _validated_optimizer_loss,
    build_multires_event_v2_model,
    build_multires_event_v2_optimizer,
    build_multires_event_v2_runtime,
    load_lab_scale_artifact,
    load_multires_event_v2_configs,
    raw_414_factor_joint_nll_batch_mean,
    require_multires_event_v2_training_authorization,
    run_multires_event_v2_training,
    validate_formal_model_parameter_count,
    validate_formal_target_field_order,
    validate_multires_event_v2_configs,
)
from trauma_predict.training.multires_event_v2_loss import (
    REGISTERED_CORE_FIELD_IDS,
    V2_PRIMITIVE_FEEDBACK_DIMS,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "configs/train/p100_multires_event_v2_relation_v2.yaml"


def _load() -> tuple[dict, dict, dict]:
    train, dataset, model, _, _ = load_multires_event_v2_configs(
        TRAIN,
        repo_root=ROOT,
    )
    return train, dataset, model


def _model_input(batch_size: int) -> dict[str, torch.Tensor]:
    keys = (
        "event_field_ids", "event_operator_ids", "event_condition_ids",
        "event_values", "event_value_mask", "event_study_slot_ids", "block_index",
        "event_mask", "block_role_ids", "resolution_ids", "relative_start",
        "relative_end", "span", "block_mask", "static_numeric",
        "static_numeric_mask", "static_categorical",
    )
    result = {key: torch.zeros((batch_size, 1)) for key in keys}
    result["latest_input_block_index"] = torch.zeros(batch_size, dtype=torch.long)
    return result


class RelationV2TrainingSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.train, self.dataset, self.model = _load()
        self.relations = MultiresEventV2RelationContract.from_default_config()

    def test_single_route_and_raw_414_contract(self) -> None:
        self.assertEqual(
            AUTHORIZED_TRAINING_RUN_NAMES,
            ("p100_multires_event_v2_relation_v2",),
        )
        require_multires_event_v2_training_authorization(self.train)
        self.assertEqual(self.train["training"]["loss_reduction"], RAW_JOINT_NLL_REDUCTION)
        self.assertEqual(self.train["objective"]["stochastic_primitive_factors"], 414)
        self.assertIsNone(self.train["objective"]["family_weights"])
        self.assertEqual(self.train["training"]["gradient_clipping"], "disabled")
        self.assertIs(self.train["training"]["resume"], True)

    def test_formal_route_has_no_capacity_gate_and_canary_precedes_data_runtime(self) -> None:
        import trauma_predict.training.multires_event_v2 as training_module

        for name in (
            "run_multires_event_v2_capacity_probe",
            "run_multires_event_v2_capacity_gated_training",
            "run_multires_event_v2_verification_probe",
            "project_multires_event_v2_capacity_runtime",
        ):
            self.assertFalse(hasattr(training_module, name), name)
            self.assertNotIn(name, training_module.__all__)
        source = inspect.getsource(run_multires_event_v2_training)
        canary = source.index("_run_v2_best_checkpoint_collective_canary(")
        runtime = source.index("build_multires_event_v2_runtime(")
        model = source.index("build_multires_event_v2_model(")
        self.assertLess(canary, runtime)
        self.assertLess(canary, model)

    def test_raw_joint_nll_and_adamw_are_exact(self) -> None:
        primitive_log_prob = torch.randn(3, 414, requires_grad=True)
        loss = raw_414_factor_joint_nll_batch_mean(
            {"primitive_count": 414, "primitive_log_prob": primitive_log_prob}
        )
        torch.testing.assert_close(
            loss,
            -primitive_log_prob.sum(dim=-1).mean(),
            rtol=0.0,
            atol=0.0,
        )
        with self.assertRaises(ValueError):
            raw_414_factor_joint_nll_batch_mean(
                {"primitive_count": 414, "primitive_log_prob": torch.zeros(3, 413)}
            )
        model = torch.nn.Linear(3, 2)
        optimizer = build_multires_event_v2_optimizer(model, self.train["training"])
        self.assertEqual(len(optimizer.param_groups), 1)
        for key, expected in EXPECTED_OPTIMIZER_CONTRACT.items():
            self.assertEqual(self.train["training"][key], expected)
        self.assertNotIn("max_grad_norm", self.train["training"])

    def test_optimizer_config_scaler_and_resume_alignment_fail_closed(self) -> None:
        for key, value in (
            ("optimizer", "SGD"),
            ("learning_rate", 1.0e-4),
            ("gradient_clipping", "global_norm"),
        ):
            changed = copy.deepcopy(self.train)
            changed["training"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_multires_event_v2_configs(changed, self.dataset, self.model)

        class Scaler:
            def __init__(self, before: float, after: float) -> None:
                self.scale = before
                self.after = after

            def get_scale(self) -> float:
                return self.scale

            def step(self, _optimizer: object) -> None:
                return None

            def update(self) -> None:
                self.scale = self.after

        self.assertEqual(_step_grad_scaler(Scaler(128.0, 64.0), object()), (False, 128.0, 64.0))
        self.assertEqual(_step_grad_scaler(Scaler(128.0, 128.0), object()), (True, 128.0, 128.0))

        linear = torch.nn.Linear(2, 1)
        optimizer = build_multires_event_v2_optimizer(linear, self.train["training"])
        scheduler = _build_scheduler(optimizer, self.train["training"])
        fresh = _validate_resume_optimizer_alignment(
            optimizer, scheduler, self.train["training"], global_step=0
        )
        self.assertEqual(fresh["optimizer_state_entries"], 0)
        linear(torch.ones(2, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        _validate_resume_optimizer_alignment(
            optimizer, scheduler, self.train["training"], global_step=1
        )
        scheduler.last_epoch = 2
        with self.assertRaises(RuntimeError):
            _validate_resume_optimizer_alignment(
                optimizer, scheduler, self.train["training"], global_step=1
            )

    def test_gradient_and_adam_state_health_remain_audited(self) -> None:
        model = torch.nn.Linear(3, 2)
        optimizer = build_multires_event_v2_optimizer(model, self.train["training"])
        model(torch.ones(4, 3)).square().mean().backward()
        gradient_health, probe = _audit_unscaled_gradients(model)
        self.assertGreater(gradient_health["global_l2_norm"], 0.0)
        self.assertEqual(gradient_health["gradient_clipping"], "disabled")
        optimizer.step()
        state = _audit_optimizer_state_after_step(
            model, optimizer, probe, expected_optimizer_step=1
        )
        self.assertTrue(state["exp_avg_sq_nonnegative"])
        model.weight.grad = None
        with self.assertRaises(RuntimeError):
            _audit_unscaled_gradients(model)

    def test_warmup_learning_rate_is_audited_before_scheduler_advance(self) -> None:
        model = torch.nn.Linear(3, 2)
        training = self.train["training"]
        optimizer = build_multires_event_v2_optimizer(model, training)
        scheduler = _build_scheduler(optimizer, training)
        observed = []
        for step in (1, 2):
            optimizer.zero_grad(set_to_none=True)
            model(torch.ones(4, 3)).square().mean().backward()
            gradient, probe = _audit_unscaled_gradients(model)
            gradient["audit_wall_seconds"] = 0.01
            optimizer.step()
            state = _audit_optimizer_state_after_step(
                model, optimizer, probe, expected_optimizer_step=step
            )
            state["audit_wall_seconds"] = 0.02
            health = _optimizer_step_health_payload(
                optimizer, gradient, state, training=training
            )
            observed.append(health["learning_rate_used"])
            self.assertEqual(
                health["learning_rate_used"],
                health["expected_learning_rate_used"],
            )
            scheduler.step()
        self.assertAlmostEqual(observed[0], 5.0e-7, places=15)
        self.assertAlmostEqual(observed[1], 1.0e-6, places=15)

    def test_audited_optimizer_rejects_any_grad_scaler_drift_from_32(self) -> None:
        class Scaler:
            def __init__(self, scale: float) -> None:
                self.scale = scale

            def get_scale(self) -> float:
                return self.scale

            def step(self, optimizer: torch.optim.Optimizer) -> None:
                optimizer.step()

            def update(self) -> None:
                return None

        model = torch.nn.Linear(2, 1)
        optimizer = build_multires_event_v2_optimizer(
            model, self.train["training"]
        )
        model(torch.ones(2, 2)).square().mean().backward()
        with self.assertRaisesRegex(FloatingPointError, "exactly 32.0"):
            _audited_optimizer_step(
                model,
                optimizer,
                Scaler(64.0),
                expected_optimizer_step=1,
            )

    def test_optimizer_health_corruption_matrix_fails_closed(self) -> None:
        def ready_gradient():
            model = torch.nn.Linear(3, 2)
            optimizer = build_multires_event_v2_optimizer(
                model, self.train["training"]
            )
            model(torch.ones(4, 3)).square().mean().backward()
            return model, optimizer

        model, _ = ready_gradient()
        model.bias.grad = None
        with self.assertRaises(RuntimeError):
            _audit_unscaled_gradients(model)

        model, _ = ready_gradient()
        for parameter in model.parameters():
            parameter.grad.zero_()
        with self.assertRaises(FloatingPointError):
            _audit_unscaled_gradients(model)

        model, _ = ready_gradient()
        model.weight.grad.reshape(-1)[0] = float("nan")
        with self.assertRaises(FloatingPointError):
            _audit_unscaled_gradients(model)

        def ready_state():
            model, optimizer = ready_gradient()
            _, probe = _audit_unscaled_gradients(model)
            optimizer.step()
            return model, optimizer, probe

        mutations = {
            "negative_second_moment": lambda model, optimizer: optimizer.state[
                model.weight
            ]["exp_avg_sq"].reshape(-1).__setitem__(0, -1.0),
            "nonfinite_first_moment": lambda model, optimizer: optimizer.state[
                model.weight
            ]["exp_avg"].reshape(-1).__setitem__(0, float("nan")),
            "wrong_step": lambda model, optimizer: optimizer.state[model.weight][
                "step"
            ].fill_(2),
            "nonfinite_parameter": lambda model, _optimizer: model.weight.data.reshape(
                -1
            ).__setitem__(0, float("nan")),
        }
        for label, mutate in mutations.items():
            model, optimizer, probe = ready_state()
            mutate(model, optimizer)
            with self.subTest(label=label), self.assertRaises(FloatingPointError):
                _audit_optimizer_state_after_step(
                    model, optimizer, probe, expected_optimizer_step=1
                )

        model, optimizer, probe = ready_state()
        del optimizer.state[model.weight]["exp_avg"]
        with self.assertRaises(RuntimeError):
            _audit_optimizer_state_after_step(
                model, optimizer, probe, expected_optimizer_step=1
            )

    def test_optimizer_loss_rejects_wrong_or_nonfinite_rows(self) -> None:
        valid = {
            "primitive_count": 414,
            "primitive_log_prob": torch.zeros(2, 414),
            "per_sample_nll": torch.zeros(2),
            "loss": torch.zeros(()),
        }
        self.assertTrue(torch.isfinite(_validated_optimizer_loss(valid, expected_local_batch=2)))
        with self.assertRaises(ValueError):
            _validated_optimizer_loss(
                dict(valid, per_sample_nll=torch.zeros(1)), expected_local_batch=2
            )
        with self.assertRaises(FloatingPointError):
            _validated_optimizer_loss(
                dict(valid, per_sample_nll=torch.tensor([0.0, float("nan")])),
                expected_local_batch=2,
            )

    def test_checkpoint_bytes_are_hash_bound(self) -> None:
        class State:
            def __init__(self, payload: dict) -> None:
                self.payload = payload

            def state_dict(self) -> dict:
                return self.payload

        class Sampler:
            def state_dict(self) -> dict:
                return {"epoch": 3}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = torch.nn.Linear(2, 1)
            _save_v2_checkpoint(
                output_dir=root,
                model=model,
                optimizer=torch.optim.AdamW(model.parameters(), lr=1.0e-3),
                scheduler=State({"last_epoch": 1}),
                scaler=State({"scale": 32.0}),
                trainer_state={"global_step": 1},
                identity_hashes={"relation_contract": self.relations.bundle_hash},
                runtime=SimpleNamespace(train_sampler=Sampler()),
                rank=0,
                keep_last=2,
            )
            _validate_v2_checkpoint_integrity(root, expected_world_size=1)
            model_path = root / "checkpoints/checkpoint-00000001/model.pt"
            model_path.write_bytes(model_path.read_bytes() + b"tamper")
            with self.assertRaises(ValueError):
                _validate_v2_checkpoint_integrity(root, expected_world_size=1)

    def test_runtime_recovers_all_6309_validation_anchors_from_v1_final_route(self) -> None:
        dataset = copy.deepcopy(self.dataset)
        dataset["base"]["root"] = "/tmp/v2-unit-base"
        dataset["target"]["root"] = "/tmp/v2-unit-target"
        captured: dict = {}

        class StopAfterV1Config(RuntimeError):
            pass

        def capture(config, *_args, **_kwargs):
            captured.update(config)
            raise StopAfterV1Config

        with tempfile.TemporaryDirectory() as directory:
            normalization = Path(directory) / "normalization.json"
            normalization.write_text('{"frozen": true}\n', encoding="utf-8")
            dataset["normalization"]["path"] = str(normalization)
            dataset["normalization"]["artifact_sha256"] = hashlib.sha256(
                normalization.read_bytes()
            ).hexdigest()
            with (
                patch("trauma_predict.training.multires_event_v2._verify_artifact_files"),
                patch(
                    "trauma_predict.training.multires_event_v2.build_v1_runtime",
                    side_effect=capture,
                ),
                self.assertRaises(StopAfterV1Config),
            ):
                build_multires_event_v2_runtime(
                    self.train,
                    dataset,
                    repo_root=ROOT,
                    rank=0,
                    world_size=1,
                    phase="interval",
                )
        self.assertEqual(
            captured["evaluation"],
            {"phase": "final", "final_expected_samples": 6309},
        )
        source = inspect.getsource(run_multires_event_v2_training)
        final_rebuild = source.index("final_runtime = build_multires_event_v2_runtime(")
        self.assertIn('phase="final"', source[final_rebuild:])
        self.assertEqual(self.train["evaluation"]["final_expected_samples"], 6309)

    def test_train_only_lab_scale_is_hash_bound_to_full_r9(self) -> None:
        artifact_path = ROOT / self.train["lab_scale_artifact"]
        original = json.loads(artifact_path.read_text(encoding="utf-8"))
        source = original["source"]
        target = self.dataset["target"]
        self.assertEqual(original["fit_split"], "train")
        self.assertEqual(original["fit_population"]["train_samples"], 37734)
        self.assertEqual(source["sidecar_dataset_id"], target["dataset_id"])
        self.assertEqual(
            source["sidecar_sample_manifest_sha256"],
            target["sample_manifest_sha256"],
        )
        self.assertEqual(
            source["sidecar_contract_bundle_hash"],
            target["contract_bundle_hash"],
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "dataset_manifest.json"
            manifest_path.write_text("{}\n", encoding="utf-8")
            payload = copy.deepcopy(original)
            payload["source"]["sidecar_dataset_manifest_sha256"] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()

            def write_payload() -> Path:
                canonical = json.dumps(
                    {
                        key: value
                        for key, value in payload.items()
                        if key != "content_sha256"
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                payload["content_sha256"] = hashlib.sha256(
                    canonical.encode("utf-8")
                ).hexdigest()
                path = root / "lab-scale.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                return path

            field_order = tuple(payload["field_order"])
            contract = SimpleNamespace(
                dataset_root=root,
                manifest={
                    "dataset_id": payload["source"]["sidecar_dataset_id"],
                    "files": {
                        "sample_manifest": {
                            "sha256": payload["source"][
                                "sidecar_sample_manifest_sha256"
                            ]
                        }
                    },
                },
                contract_bundle_hash=payload["source"][
                    "sidecar_contract_bundle_hash"
                ],
                contract_hashes={
                    "process": payload["source"][
                        "sidecar_process_contract_sha256"
                    ],
                    "emission": payload["source"][
                        "sidecar_emission_contract_sha256"
                    ],
                },
                lab_fields=field_order,
                core_fields=field_order,
                registered_core_field_ids=tuple(
                    int(payload["fields"][field]["field_id"])
                    for field in field_order
                ),
                emission_registry={
                    "field_supports": {
                        "intermittent_labs": {
                            field: {"unit": payload["fields"][field]["unit"]}
                            for field in field_order
                        }
                    }
                },
            )
            path = write_payload()
            compact = load_lab_scale_artifact(
                path,
                expected_content_sha256=payload["content_sha256"],
                contract=contract,
            )
            self.assertEqual(compact["content_sha256"], payload["content_sha256"])

            payload["fit_split"] = "val"
            path = write_payload()
            with self.assertRaisesRegex(ValueError, "fit_split"):
                load_lab_scale_artifact(
                    path,
                    expected_content_sha256=payload["content_sha256"],
                    contract=contract,
                )

    def test_target_field_order_and_parameter_count_are_frozen(self) -> None:
        validate_formal_target_field_order(
            self.model,
            SimpleNamespace(registered_core_field_ids=REGISTERED_CORE_FIELD_IDS),
        )
        with self.assertRaises(ValueError):
            validate_formal_target_field_order(
                self.model,
                SimpleNamespace(
                    registered_core_field_ids=tuple(reversed(REGISTERED_CORE_FIELD_IDS))
                ),
            )
        built = build_multires_event_v2_model(
            self.model, relation_contract=self.relations
        )
        self.assertEqual(
            validate_formal_model_parameter_count(built),
            EXPECTED_FORMAL_MODEL_PARAMETER_COUNT,
        )
        built.register_parameter("drift", torch.nn.Parameter(torch.zeros(1)))
        with self.assertRaises(ValueError):
            validate_formal_model_parameter_count(built)

    def test_teacher_adapter_and_float32_loss_boundary(self) -> None:
        leading = (1, 6, 29)
        primitives = {
            key: torch.zeros((*leading, width)) if width > 1 else torch.zeros(leading)
            for key, width in V2_PRIMITIVE_FEEDBACK_DIMS.items()
        }
        batch = {
            "input_batch": _model_input(1),
            "target_primitives": primitives,
            "target_primitive_masks": {
                key: torch.ones(leading, dtype=torch.bool) for key in primitives
            },
        }
        inputs = teacher_forced_model_inputs(batch)
        scalar = "categorical_hours_0_4"
        self.assertEqual(primitives[scalar].shape, leading)
        self.assertEqual(inputs["target_primitives"][scalar].shape, (*leading, 1))
        self.assertNotIn("mode", inputs)
        self.assertNotIn("relation_adjacency", inputs)

        class Model:
            def __call__(self, **_kwargs):
                return {"primitive_parameters": {"bank": torch.ones(1, dtype=torch.float16)}}

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
                Model(), batch, {}, expected_lab_scale_artifact_hash="a" * 64
            )
        self.assertEqual(result["primitive_count"], 414)

    def test_subject_macro_is_within_subject_then_macro(self) -> None:
        class Model:
            def eval(self):
                return self

        loader = [{
            "sample_id": ["a", "b", "c"],
            "subject_id": ["p1", "p1", "p2"],
            "nll": torch.tensor([2.0, 4.0, 10.0]),
        }]

        def fake_loss(_model, batch, _registry, **_kwargs):
            return {}, {"per_sample_nll": batch["nll"], "primitive_count": 414}

        with patch(
            "trauma_predict.eval.multires_event_v2.exact_teacher_forced_loss",
            side_effect=fake_loss,
        ):
            result = evaluate_teacher_forced(
                model=Model(), loader=loader, registry={}, device=torch.device("cpu"),
                expected_samples=3, phase="final", step=7, precision="fp16",
            )
        self.assertAlmostEqual(result["joint_nll_anchor_mean"], 16.0 / 3.0)
        self.assertAlmostEqual(result["joint_nll_subject_macro"], 6.5)

    def test_legacy_checkpoint_parameterization_cannot_load(self) -> None:
        built = build_multires_event_v2_model(self.model, relation_contract=self.relations)
        state = built.state_dict()
        legacy_like = {
            key: value
            for key, value in state.items()
            if "input_field_memory" not in key and "input_target_relation" not in key
        }
        self.assertFalse(
            any("input_field_memory" in key for key in legacy_like)
        )
        self.assertFalse(
            any("input_target_relation" in key for key in legacy_like)
        )
        with self.assertRaises(RuntimeError):
            built.load_state_dict(legacy_like, strict=True)
        self.assertEqual(self.model["formal_contract"]["legacy_checkpoint_loading"], "forbidden")
        self.assertNotEqual(EXPECTED_FORMAL_MODEL_PARAMETER_COUNT, 47_801_855)

    def test_pre_temporal_fusion_relation_v2_checkpoint_cannot_load(self) -> None:
        built = build_multires_event_v2_model(self.model, relation_contract=self.relations)
        temporal_shapes = {
            "input_field_memory.input_only_temporal_weight": (8, 6),
        }
        state = built.state_dict()
        for key, shape in temporal_shapes.items():
            self.assertIn(key, state)
            self.assertEqual(tuple(state[key].shape), shape)

        pre_temporal = {
            key: value for key, value in state.items() if key not in temporal_shapes
        }
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            built.load_state_dict(pre_temporal, strict=True)

        pre_temporal_parameter_count = sum(
            parameter.numel()
            for name, parameter in built.named_parameters()
            if name not in temporal_shapes
        )
        self.assertEqual(pre_temporal_parameter_count, 48_728_391)
        self.assertEqual(EXPECTED_FORMAL_MODEL_PARAMETER_COUNT, 48_728_439)

    def test_builder_rejects_relation_contract_identity_mutation(self) -> None:
        with self.assertRaises(ValueError):
            build_multires_event_v2_model(
                self.model,
                relation_contract=replace(self.relations, bundle_hash="0" * 64),
            )


if __name__ == "__main__":
    unittest.main()
