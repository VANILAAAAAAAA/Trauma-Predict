from __future__ import annotations

import copy
import inspect
import tempfile
import unittest
from pathlib import Path

import torch

from trauma_predict.training.grud_h1_v2 import (
    EXPECTED_EVAL_STEPS,
    EXPECTED_INTERVAL_SUBJECTS,
    EXPECTED_LOGGING_STEPS,
    EXPECTED_OPTIMIZER_STEPS,
    EXPECTED_SAVE_STEPS,
    MODEL_INPUT_KEYS,
    _save_checkpoint,
    build_grud_h1_v2_optimizer,
    fixed_one_anchor_per_subject_indices,
    load_grud_h1_v2_configs,
    raw_414_factor_joint_nll_batch_mean,
    run_grud_h1_v2_training,
    summarize_subject_macro_nll,
    teacher_forced_model_inputs,
    validate_grud_h1_v2_configs,
)
from trauma_predict.training.multires_event import _build_grad_scaler, _build_scheduler
from trauma_predict.training.multires_event_v2_loss import V2_PRIMITIVE_FEEDBACK_DIMS


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "configs/train/p100_grud_h1_joint_m4_v2.yaml"


def _load() -> tuple[dict, dict, dict]:
    train, dataset, model, _, _ = load_grud_h1_v2_configs(TRAIN, repo_root=ROOT)
    return train, dataset, model


class _Dataset:
    def __init__(self) -> None:
        self.sample_ids = ("s0", "s1", "s2", "s3", "s4")
        self.subject_ids = ("a", "a", "b", "c", "c")
        self.shard_keys = ("x", "x", "y", "z", "z")

    def __len__(self) -> int:
        return len(self.sample_ids)


class GRUDH1V2TrainingContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.train, self.dataset, self.model = _load()

    def test_frozen_training_and_evaluation_schedule(self) -> None:
        validate_grud_h1_v2_configs(self.train, self.dataset, self.model)
        training = self.train["training"]
        evaluation = self.train["evaluation"]
        self.assertEqual(training["max_steps"], EXPECTED_OPTIMIZER_STEPS)
        self.assertEqual(training["logging_steps"], EXPECTED_LOGGING_STEPS)
        self.assertEqual(training["eval_steps"], EXPECTED_EVAL_STEPS)
        self.assertEqual(training["save_steps"], EXPECTED_SAVE_STEPS)
        self.assertEqual(training["train_samples_per_epoch"], 3072)
        self.assertIs(training["fresh_start"], True)
        self.assertIs(training["resume"], False)
        self.assertIs(training["forced_stop"], False)
        self.assertEqual(
            evaluation["interval_anchor_policy"],
            "one_fixed_anchor_per_validation_subject",
        )
        self.assertEqual(evaluation["interval_expected_subjects"], EXPECTED_INTERVAL_SUBJECTS)
        self.assertIs(evaluation["final_evaluation_in_training_notebook"], False)
        self.assertIs(evaluation["free_running_in_training_notebook"], False)

    def test_config_rejects_resume_schedule_and_task_drift(self) -> None:
        changes = (
            ("training", "resume", True),
            ("training", "forced_stop", True),
            ("training", "max_steps", 250),
            ("training", "loss_reduction", "mean_active_targets"),
            ("training", "logging_steps", 50),
            ("training", "eval_steps", 500),
            ("training", "save_steps", 250),
            ("evaluation", "final_evaluation_in_training_notebook", True),
            ("evaluation", "free_running_in_training_notebook", True),
            ("objective", "stochastic_primitive_factors", 413),
        )
        for section, key, value in changes:
            changed = copy.deepcopy(self.train)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_grud_h1_v2_configs(changed, self.dataset, self.model)

    def test_raw_joint_nll_has_no_hidden_factor_denominator(self) -> None:
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
        self.assertNotEqual(float(loss.detach()), float((loss / 414.0).detach()))
        with self.assertRaises(ValueError):
            raw_414_factor_joint_nll_batch_mean(
                {"primitive_count": 414, "primitive_log_prob": torch.zeros(2, 413)}
            )

    def test_teacher_forcing_passes_only_h1_inputs_and_registered_truth(self) -> None:
        input_batch = {
            key: torch.zeros((2, 1), dtype=torch.float32) for key in MODEL_INPUT_KEYS
        }
        input_batch["unregistered_input"] = torch.ones(2)
        primitives = {}
        masks = {}
        for likelihood_id, width in V2_PRIMITIVE_FEEDBACK_DIMS.items():
            shape = (2, 6, 29) if width == 1 else (2, 6, 29, width)
            primitives[likelihood_id] = torch.zeros(shape)
            masks[likelihood_id] = torch.ones(shape, dtype=torch.bool)
        result = teacher_forced_model_inputs(
            {
                "input_batch": input_batch,
                "target_primitives": primitives,
                "target_primitive_masks": masks,
            }
        )
        self.assertEqual(
            set(result),
            set(MODEL_INPUT_KEYS) | {"target_primitives", "target_primitive_masks"},
        )
        for likelihood_id, width in V2_PRIMITIVE_FEEDBACK_DIMS.items():
            self.assertEqual(result["target_primitives"][likelihood_id].shape[-1], width)

    def test_interval_subset_is_fixed_and_has_one_anchor_per_subject(self) -> None:
        dataset = _Dataset()
        first = fixed_one_anchor_per_subject_indices(dataset, seed=17)
        second = fixed_one_anchor_per_subject_indices(dataset, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len({dataset.subject_ids[index] for index in first}), 3)

    def test_subject_macro_is_not_anchor_weighted(self) -> None:
        result = summarize_subject_macro_nll(
            (
                {"sample_id": "a0", "subject_id": "a", "joint_nll": 1.0},
                {"sample_id": "a1", "subject_id": "a", "joint_nll": 3.0},
                {"sample_id": "b0", "subject_id": "b", "joint_nll": 8.0},
            )
        )
        self.assertEqual(result["samples"], 3)
        self.assertEqual(result["subjects"], 2)
        self.assertAlmostEqual(float(result["joint_nll_anchor_mean"]), 4.0)
        self.assertAlmostEqual(float(result["joint_nll_subject_macro"]), 5.0)

    def test_adamw_checkpoint_contains_complete_fresh_state(self) -> None:
        model = torch.nn.Linear(2, 1)
        training = self.train["training"]
        optimizer = build_grud_h1_v2_optimizer(model, training)
        scheduler = _build_scheduler(optimizer, training)
        scaler = _build_grad_scaler(torch, torch.device("cpu"), training)
        model(torch.ones(2, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            checkpoint = _save_checkpoint(
                output_dir=output_dir,
                step=500,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                train_config=self.train,
                dataset_config=self.dataset,
                model_config=self.model,
                runtime_identity={"h1": "frozen", "target": "r9"},
                validation={"joint_nll_subject_macro": 10.0},
            )
            self.assertEqual(checkpoint, output_dir / "checkpoint-500" / "checkpoint.pt")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["step"], 500)
            self.assertIn("optimizer_state_dict", payload)
            self.assertIn("scheduler_state_dict", payload)
            self.assertIn("grad_scaler_state_dict", payload)

    def test_training_route_contains_no_final_or_free_running_call(self) -> None:
        source = inspect.getsource(run_grud_h1_v2_training)
        self.assertNotIn("evaluate_final", source)
        self.assertNotIn("free_running", source)
        self.assertIn("checkpoint-", source)
        self.assertIn("training_manifest.json", source)


if __name__ == "__main__":
    unittest.main()
