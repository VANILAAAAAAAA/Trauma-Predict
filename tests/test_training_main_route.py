from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from trauma_predict.data.main_route_contract import (
    HOUR_SPECIAL_TOKENS,
    MAIN_ROUTE,
    expected_hour_placeholders,
    validate_main_route_record,
)
from trauma_predict.data.main_route import (
    HourValueNormalizer,
    MainRouteBatchCollator,
    encode_next24_labels,
)
from trauma_predict.training.main_route import (
    _hour_input_context,
    _next_hour_target_for_active_losses,
    select_prediction_records,
    validate_main_route_config,
    validate_resume_checkpoint_stage,
)
from trauma_predict.training.config import load_yaml_config
from trauma_predict.training.runtime import quarantine_rng_state_files
from trauma_predict.training.stages import resolve_training_stage_contract


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    def __init__(self) -> None:
        tokens = [
            "<pad>",
            "<s>",
            "</s>",
            "<unk>",
            "<SAMPLE>",
            "</SAMPLE>",
            "schema=icu_state_major_textual_v1",
            "STATIC:",
            "static{age=70}",
            "DAY:",
            "D0",
            "i=0",
            "len=24",
            "dq{vital=dense;lab=drawn;uop=measured}",
            "HOUR",
            "len=2:",
            "<STATE>",
        ]
        tokens.extend(HOUR_SPECIAL_TOKENS)
        self.vocab = {token: index for index, token in enumerate(tokens)}
        self.unk_token_id = self.vocab["<unk>"]
        self.padding_side = "right"

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.vocab["<unk>"])

    def __call__(self, text: str, add_special_tokens: bool = True, truncation: bool = False):
        pieces = text.split()
        input_ids = [self.convert_tokens_to_ids(piece) for piece in pieces]
        if add_special_tokens:
            input_ids = [self.vocab["<s>"], *input_ids, self.vocab["</s>"]]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

    def pad(self, encoded_items, padding=True, pad_to_multiple_of=None, return_tensors=None):
        import torch

        width = max(len(item["input_ids"]) for item in encoded_items)
        if pad_to_multiple_of:
            remainder = width % pad_to_multiple_of
            if remainder:
                width += pad_to_multiple_of - remainder
        input_ids = []
        attention_mask = []
        for item in encoded_items:
            pad_len = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.vocab["<pad>"]] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class TrainingMainRouteTest(unittest.TestCase):
    def test_main_route_config_rejects_text_generation_task(self) -> None:
        config = {
            "schema_version": "trauma_predict.train_config.v1",
            "training_stage": "joint_baseline",
            "model": {
                "base_model": "google/flan-t5-base",
                "task": "next24_text_generation",
                "max_input_tokens": 1024,
                "hour_adapter_hidden": 256,
            },
            "training": _training_block(active_next24=True),
        }

        with self.assertRaisesRegex(ValueError, "main_hour_adapter_structured_heads"):
            validate_main_route_config(config)

    def test_main_route_config_accepts_structured_route(self) -> None:
        validate_main_route_config({
            "schema_version": "trauma_predict.train_config.v1",
            "training_stage": "joint_baseline",
            "model": {
                "base_model": "allenai/longformer-base-4096",
                "task": "main_hour_adapter_structured_heads",
                "max_input_tokens": 4096,
                "hour_adapter_hidden": 256,
            },
            "training": _training_block(active_next24=True),
        })

    def test_main_route_config_requires_explicit_training_stage(self) -> None:
        config = {
            "schema_version": "trauma_predict.train_config.v1",
            "model": {
                "base_model": "allenai/longformer-base-4096",
                "task": "main_hour_adapter_structured_heads",
                "max_input_tokens": 4096,
                "hour_adapter_hidden": 256,
            },
            "training": _training_block(active_next24=True),
        }

        with self.assertRaisesRegex(ValueError, "training_stage"):
            validate_main_route_config(config)

    def test_stage_a_contract_allows_next_hour_values_only(self) -> None:
        contract = resolve_training_stage_contract({
            "schema_version": "trauma_predict.train_config.v1",
            "run_name": "t4x2_stage_a_hour",
            "training_stage": "stage_a_next_hour",
            "training": _training_block(active_next24=False),
        })

        self.assertEqual(contract.active_loss_names, ["next_hour_values"])
        self.assertEqual(contract.loss_weights["next_hour_vent"], 0.0)
        self.assertEqual(contract.loss_weights["next24_domain"], 0.0)
        self.assertEqual(contract.loss_weights["next24_binary"], 0.0)
        self.assertEqual(contract.loss_weights["next24_multiclass"], 0.0)

    def test_stage_a_contract_rejects_next24_loss_activation(self) -> None:
        config = {
            "schema_version": "trauma_predict.train_config.v1",
            "run_name": "t4x2_stage_a_hour",
            "training_stage": "stage_a_next_hour",
            "training": _training_block(active_next24=True),
        }

        with self.assertRaisesRegex(ValueError, "active_losses does not match stage_a_next_hour"):
            resolve_training_stage_contract(config)

    def test_stage_a1_contract_requires_residual_values_only_warm_start(self) -> None:
        contract = resolve_training_stage_contract({
            "schema_version": "trauma_predict.train_config.v1",
            "run_name": "t4x2_stage_a1_residual",
            "training_stage": "stage_a1_residual",
            "model": {
                "base_model": "answerdotai/ModernBERT-base",
                "task": "main_hour_adapter_structured_heads",
                "max_input_tokens": 4096,
                "hour_adapter_hidden": 256,
                "hour_field_hidden": 64,
                "next_hour_value_mode": "h0_residual",
            },
            "training": _stage_a1_training_block(),
        })

        self.assertEqual(contract.active_loss_names, ["next_hour_values"])
        self.assertEqual(contract.next_hour_value_mode, "h0_residual")
        self.assertEqual(contract.next_hour_delta_loss_weight, 1.0)
        self.assertEqual(contract.warm_start_checkpoint, "/tmp/stage_a/checkpoint-4000")

    def test_stage_b_contract_is_reserved_but_not_runnable(self) -> None:
        config = {
            "schema_version": "trauma_predict.train_config.v1",
            "run_name": "stage_b_next24",
            "training_stage": "stage_b_next24",
            "model": {
                "base_model": "allenai/longformer-base-4096",
                "task": "main_hour_adapter_structured_heads",
                "max_input_tokens": 4096,
                "hour_adapter_hidden": 256,
            },
            "training": _stage_b_training_block(),
        }

        contract = resolve_training_stage_contract(config)
        self.assertFalse(contract.implemented)
        self.assertEqual(contract.stage_a_checkpoint, "/tmp/stage_a/checkpoint-500")
        self.assertEqual(
            contract.active_loss_names,
            ["next24_domain", "next24_binary", "next24_multiclass"],
        )
        with self.assertRaisesRegex(NotImplementedError, "training runner is not implemented"):
            validate_main_route_config(config)

    def test_stage_c_contract_is_reserved_but_not_runnable(self) -> None:
        config = {
            "schema_version": "trauma_predict.train_config.v1",
            "run_name": "stage_c_alternating",
            "training_stage": "stage_c_alternating",
            "model": {
                "base_model": "allenai/longformer-base-4096",
                "task": "main_hour_adapter_structured_heads",
                "max_input_tokens": 4096,
                "hour_adapter_hidden": 256,
            },
            "training": {
                **_training_block(active_next24=True, active_vent=False),
                "alternating_summary_steps": 4,
            },
        }

        contract = resolve_training_stage_contract(config)
        self.assertFalse(contract.implemented)
        self.assertEqual(contract.alternating_summary_steps, 4)
        with self.assertRaisesRegex(NotImplementedError, "training runner is not implemented"):
            validate_main_route_config(config)

    def test_tracked_training_configs_validate_stage_contracts(self) -> None:
        old_checkpoint = os.environ.get("STAGE_A_CHECKPOINT_DIR")
        os.environ["STAGE_A_CHECKPOINT_DIR"] = "/tmp/stage_a/checkpoint-4000"
        try:
            for path in sorted((REPO_ROOT / "configs" / "train").glob("*.yaml")):
                with self.subTest(path=path.name):
                    validate_main_route_config(load_yaml_config(path))
        finally:
            if old_checkpoint is None:
                os.environ.pop("STAGE_A_CHECKPOINT_DIR", None)
            else:
                os.environ["STAGE_A_CHECKPOINT_DIR"] = old_checkpoint

    def test_stage_a_configs_use_preferred_encoder_and_values_only_loss(self) -> None:
        expected_active_losses = {
            "next_hour_values": True,
            "next_hour_vent": False,
            "next24_domain": False,
            "next24_binary": False,
            "next24_multiclass": False,
        }
        for name in [
            "t4x2_stage_a_hour.yaml",
            "p100_stage_a_hour.yaml",
            "t4x2_stage_a_hour_smoke.yaml",
        ]:
            with self.subTest(config=name):
                config = load_yaml_config(REPO_ROOT / "configs" / "train" / name)
                self.assertEqual(config["training_stage"], "stage_a_next_hour")
                self.assertEqual(config["model"]["base_model"], "answerdotai/ModernBERT-base")
                self.assertEqual(config["training"]["active_losses"], expected_active_losses)
                self.assertEqual(config["training"]["loss_weights"]["next_hour_vent"], 0.0)
                self.assertIs(config["training"]["disable_tqdm"], True)
                if "smoke" not in name:
                    self.assertEqual(config["training"]["logging_steps"], 250)
                    self.assertEqual(config["training"]["max_steps"], 4000)

    def test_kaggle_dry_run_rejects_invalid_stage_a_before_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "out"
            config_path = root / "bad_stage_a.yaml"
            config_path.write_text(
                "\n".join([
                    "schema_version: trauma_predict.train_config.v1",
                    "run_name: bad_stage_a",
                    "training_stage: stage_a_next_hour",
                    "model:",
                    "  base_model: allenai/longformer-base-4096",
                    "  task: main_hour_adapter_structured_heads",
                    "  max_input_tokens: 4096",
                    "  hour_adapter_hidden: 256",
                    "data:",
                    "  config_path: configs/dataset/first_train.yaml",
                    "training:",
                    "  precision: fp16",
                    "  learning_rate: 2.0e-5",
                    "  max_steps: 1",
                    "  active_losses:",
                    "    next_hour_values: true",
                    "    next_hour_vent: true",
                    "    next24_domain: true",
                    "    next24_binary: true",
                    "    next24_multiclass: true",
                    "  loss_weights:",
                    "    next_hour_values: 1.0",
                    "    next_hour_vent: 0.25",
                    "    next24_domain: 0.25",
                    "    next24_binary: 0.5",
                    "    next24_multiclass: 0.5",
                    "outputs:",
                    f"  output_dir: {output_root / 'bad_stage_a'}",
                ]) + "\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT / "src")

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "notebooks" / "kaggle" / "train_kaggle.py"),
                    "--config",
                    str(config_path),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("active_losses does not match stage_a_next_hour", result.stderr)
            self.assertFalse((output_root / "bad_stage_a" / "run_config_snapshot.json").exists())

    def test_kaggle_dry_run_rejects_reserved_stage_b_before_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "out"
            config_path = root / "stage_b.yaml"
            config_path.write_text(
                "\n".join([
                    "schema_version: trauma_predict.train_config.v1",
                    "run_name: stage_b_next24",
                    "training_stage: stage_b_next24",
                    "model:",
                    "  base_model: allenai/longformer-base-4096",
                    "  task: main_hour_adapter_structured_heads",
                    "  max_input_tokens: 4096",
                    "  hour_adapter_hidden: 256",
                    "data:",
                    "  config_path: configs/dataset/first_train.yaml",
                    "training:",
                    "  precision: fp16",
                    "  learning_rate: 2.0e-5",
                    "  max_steps: 1",
                    "  stage_a_checkpoint: /tmp/stage_a/checkpoint-500",
                    "  active_losses:",
                    "    next_hour_values: false",
                    "    next_hour_vent: false",
                    "    next24_domain: true",
                    "    next24_binary: true",
                    "    next24_multiclass: true",
                    "  loss_weights:",
                    "    next_hour_values: 0.0",
                    "    next_hour_vent: 0.0",
                    "    next24_domain: 0.25",
                    "    next24_binary: 0.5",
                    "    next24_multiclass: 0.5",
                    "outputs:",
                    f"  output_dir: {output_root / 'stage_b_next24'}",
                ]) + "\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT / "src")

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "notebooks" / "kaggle" / "train_kaggle.py"),
                    "--config",
                    str(config_path),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("training runner is not implemented", result.stderr)
            self.assertFalse((output_root / "stage_b_next24" / "run_config_snapshot.json").exists())

    def test_next24_label_encoding_preserves_structured_slots(self) -> None:
        labels = encode_next24_labels({
            "label": "NEXT_24H",
            "len_hours": 24,
            "sections": {
                "shock": {"map_low_hours": "prolonged"},
                "resp": {"spo2_min": "critical_low"},
                "tx": {"surg": "present", "crystalloid": "high"},
            },
        })

        self.assertEqual(labels["domains"], [1.0, 1.0, 0.0, 0.0, 1.0])
        self.assertIn(1.0, labels["binary_fields"])
        self.assertIn(3, labels["multiclass_fields"])

    def test_record_contract_rejects_extra_hour_token(self) -> None:
        record = _main_route_record()
        record["input_text"] = str(record["input_text"]).replace("<H-01> <H0>", "<H-02> <H-01> <H0>")

        with self.assertRaisesRegex(ValueError, "HOUR text tokens do not match"):
            validate_main_route_record(record, _required_fields())

    def test_record_contract_rejects_hour_len_mismatch(self) -> None:
        record = _main_route_record()
        record["input_text"] = str(record["input_text"]).replace("HOUR len=2:", "HOUR len=24:")

        with self.assertRaisesRegex(ValueError, "HOUR len does not match"):
            validate_main_route_record(record, _required_fields())

    def test_record_contract_accepts_dynamic_short_hour_window(self) -> None:
        record = _main_route_record()

        validate_main_route_record(record, _required_fields())

    def test_record_contract_accepts_dynamic_min_and_full_hour_windows(self) -> None:
        for length in (1, 24):
            with self.subTest(length=length):
                record = _record_with_hour_length(length)

                validate_main_route_record(record, _required_fields())

    def test_record_contract_rejects_next_hour_without_observed_vitals(self) -> None:
        record = copy.deepcopy(_main_route_record())
        next_hour = record["targets"]["next_hour"]
        next_hour["mask"] = {field: 0 for field in next_hour["mask"]}
        next_hour["hour_mask"] = [0] * 7
        next_hour["values"] = {field: None for field in next_hour["values"]}
        next_hour["hour_values"] = [None] * 7

        with self.assertRaisesRegex(ValueError, "target.next_hour has no observed vital values"):
            validate_main_route_record(record, _required_fields())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_collator_aligns_hour_placeholders_and_side_tensors(self) -> None:
        collator = MainRouteBatchCollator(
            tokenizer=FakeTokenizer(),
            max_input_tokens=128,
            normalizer=HourValueNormalizer.from_config(None),
        )
        batch = collator([_main_route_record()])

        self.assertEqual(batch["hour_values"].shape, (1, 2, 7))
        self.assertEqual(batch["hour_mask"].shape, (1, 2, 7))
        self.assertEqual(batch["hour_vent"].tolist(), [[[0.0], [1.0]]])
        self.assertEqual(batch["hour_position_mask"].tolist(), [[True, True]])
        self.assertGreaterEqual(batch["state_position"].item(), 0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_stage_a_collator_emits_only_next_hour_value_labels(self) -> None:
        collator = MainRouteBatchCollator(
            tokenizer=FakeTokenizer(),
            max_input_tokens=128,
            normalizer=HourValueNormalizer.from_config(None),
            active_losses={
                "next_hour_values": True,
                "next_hour_vent": False,
                "next24_domain": False,
                "next24_binary": False,
                "next24_multiclass": False,
            },
        )

        batch = collator([_main_route_record()])

        self.assertIn("next_hour_values", batch)
        self.assertIn("next_hour_mask", batch)
        self.assertNotIn("next_hour_vent", batch)
        self.assertNotIn("next24_domain_labels", batch)
        self.assertNotIn("next24_binary_labels", batch)
        self.assertNotIn("next24_multiclass_labels", batch)

    def test_stage_a_prediction_helpers_keep_h0_and_omit_vent_target(self) -> None:
        record = _main_route_record()

        hour_context = _hour_input_context(record)
        self.assertEqual(hour_context["h0"]["placeholder"], "<H0>")
        self.assertEqual(hour_context["h0"]["hour_values"], record["hour_values"][-1])
        self.assertEqual(hour_context["h0"]["hour_mask"], record["hour_mask"][-1])
        self.assertEqual(hour_context["h0"]["hour_vent"], record["hour_vent"][-1])

        target = _next_hour_target_for_active_losses(
            record["targets"]["next_hour"],
            {
                "next_hour_values": True,
                "next_hour_vent": False,
                "next24_domain": False,
                "next24_binary": False,
                "next24_multiclass": False,
            },
        )
        self.assertIn("hour_values", target)
        self.assertIn("hour_mask", target)
        self.assertNotIn("vent_on", target)
        self.assertNotIn("hour_vent", target)

    def test_prediction_record_selection_supports_full_eval(self) -> None:
        records = [{"sample_id": f"s{index}"} for index in range(4)]

        self.assertEqual(select_prediction_records(records, None), records)
        self.assertEqual(select_prediction_records(records, "all"), records)
        self.assertEqual(select_prediction_records(records, "full"), records)
        self.assertEqual(select_prediction_records(records, 2), records[:2])
        with self.assertRaisesRegex(ValueError, "max_prediction_samples"):
            select_prediction_records(records, 0)

    def test_collator_rejects_missing_hour_special_token(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.vocab.pop("<H-23>")

        with self.assertRaisesRegex(ValueError, "does not know HOUR tokens"):
            MainRouteBatchCollator(
                tokenizer=tokenizer,
                max_input_tokens=128,
                normalizer=HourValueNormalizer.from_config(None),
            )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_hour_state_adapter_outputs_encoder_hidden_size(self) -> None:
        import torch

        from trauma_predict.modeling.main_route import HourStateAdapter

        adapter = HourStateAdapter(hidden_size=32, adapter_hidden_size=16, dropout=0.0, field_hidden_size=8)
        values = torch.zeros((2, 24, 7))
        mask = torch.ones((2, 24, 7))
        vent = torch.zeros((2, 24, 1))

        output = adapter(values, mask, vent)

        self.assertEqual(output.shape, (2, 24, 32))
        self.assertEqual(len(adapter.vital_value_projections), 7)
        self.assertEqual(len(adapter.vital_mask_embeddings), 7)
        self.assertEqual(adapter.field_embedding.num_embeddings, 8)
        self.assertEqual(adapter.field_hidden_size, 8)

    def test_quarantine_rng_state_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint-500"
            checkpoint.mkdir()
            rng_state = checkpoint / "rng_state_0.pth"
            model_file = checkpoint / "model.safetensors"
            rng_state.write_bytes(b"rng")
            model_file.write_bytes(b"model")

            quarantined = quarantine_rng_state_files(str(checkpoint))

            self.assertFalse(rng_state.exists())
            self.assertTrue((checkpoint / "rng_state_0.pth.ignored_for_torch_weights_only").exists())
            self.assertTrue(model_file.exists())
            self.assertEqual(len(quarantined), 1)

    def test_resume_checkpoint_stage_metadata_must_match(self) -> None:
        contract = resolve_training_stage_contract({
            "schema_version": "trauma_predict.train_config.v1",
            "run_name": "t4x2_stage_a_hour",
            "training_stage": "stage_a_next_hour",
            "training": _training_block(active_next24=False),
        })
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint-500"
            checkpoint.mkdir()
            (checkpoint / "training_stage_metadata.json").write_text(
                json.dumps({
                    "route": MAIN_ROUTE,
                    **contract.to_metadata(),
                    "training_stage": "joint_baseline",
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "training_stage mismatch"):
                validate_resume_checkpoint_stage(str(checkpoint), contract)

    def test_resume_checkpoint_stage_metadata_accepts_matching_contract(self) -> None:
        contract = resolve_training_stage_contract({
            "schema_version": "trauma_predict.train_config.v1",
            "run_name": "t4x2_stage_a_hour",
            "training_stage": "stage_a_next_hour",
            "training": _training_block(active_next24=False),
        })
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint-500"
            checkpoint.mkdir()
            (checkpoint / "training_stage_metadata.json").write_text(
                json.dumps({
                    "route": MAIN_ROUTE,
                    **contract.to_metadata(),
                }),
                encoding="utf-8",
            )

            validate_resume_checkpoint_stage(str(checkpoint), contract)

    def test_verify_dataset_notebook_uses_v2_and_does_not_print_token_clone(self) -> None:
        notebook_text = (REPO_ROOT / "notebooks" / "kaggle" / "verify_private_dataset.ipynb").read_text(
            encoding="utf-8"
        )

        self.assertIn("trauma-predict-main-route-first-train-8h-v2", notebook_text)
        self.assertNotIn("run([\\\"git\\\", \\\"clone\\\", clone_url", notebook_text)

    def test_stage_a_notebook_uses_modernbert_launcher_and_tag(self) -> None:
        notebook_text = (REPO_ROOT / "notebooks" / "kaggle" / "train_stage_a_hour.ipynb").read_text(
            encoding="utf-8"
        )
        launcher_text = (REPO_ROOT / "notebooks" / "kaggle" / "run_stage_a_hour.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("stage-a-hour-modernbert-4000-20260709", notebook_text)
        self.assertIn("run_stage_a_hour.py", notebook_text)
        self.assertIn("answerdotai/ModernBERT-base", launcher_text)
        self.assertIn('"transformers": "4.48.3"', launcher_text)
        self.assertIn("run_to_log", launcher_text)
        self.assertIn("log_dir", launcher_text)
        self.assertIn("torchrun_train.log", launcher_text)
        self.assertNotIn("for line in lines[-20:]", launcher_text)
        self.assertNotIn("stage-a-hour-training-20260708-r2", notebook_text)

    def test_stage_a1_notebook_uses_residual_launcher(self) -> None:
        notebook_text = (REPO_ROOT / "notebooks" / "kaggle" / "train_stage_a1_residual.ipynb").read_text(
            encoding="utf-8"
        )
        launcher_text = (REPO_ROOT / "notebooks" / "kaggle" / "run_stage_a1_residual.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("run_stage_a1_residual.py", notebook_text)
        self.assertIn("stage_a1_residual", launcher_text)
        self.assertIn("STAGE_A1_SMOKE_RUN_OK", launcher_text)
        self.assertIn("STAGE_A_CHECKPOINT_DIR", launcher_text)


def _main_route_record() -> dict[str, object]:
    return {
        "schema": "standard_textual_v1_main_record_v2",
        "route": "main_hour_adapter_structured_heads",
        "dataset_id": "synthetic",
        "sample_id": "s1",
        "subject_id": "101",
        "hadm_id": "201",
        "stay_id": "301",
        "prediction_hour": 48,
        "split": "train",
        "input_text": (
            "<SAMPLE> schema=icu_state_major_textual_v1 STATIC: static{age=70} DAY: "
            "D0 i=0 len=24 dq{vital=dense;lab=drawn;uop=measured} "
            "HOUR len=2: <H-01> <H0> <STATE> </SAMPLE>"
        ),
        "hour_value_order": ["hr", "sbp", "dbp", "map", "rr", "temp", "spo2"],
        "hour_placeholders": ["<H-01>", "<H0>"],
        "hour_values": [
            [90.0, 120.0, None, 78.0, 22.0, 37.2, 95.0],
            [92.0, 118.0, 65.0, 80.0, 24.0, 37.4, 94.0],
        ],
        "hour_mask": [
            [1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        "hour_vent": [[0], [1]],
        "targets": {
            "next_hour": {
                "label": "NEXT_HOUR",
                "relative_hour": "H+1",
                "value_order": ["hr", "sbp", "dbp", "map", "rr", "temp", "spo2"],
                "values": {
                    "hr": 93.0,
                    "sbp": 116.0,
                    "dbp": 64.0,
                    "map": 79.0,
                    "rr": 25.0,
                    "temp": 37.5,
                    "spo2": 93.0,
                },
                "mask": {"hr": 1, "sbp": 1, "dbp": 1, "map": 1, "rr": 1, "temp": 1, "spo2": 1},
                "hour_values": [93.0, 116.0, 64.0, 79.0, 25.0, 37.5, 93.0],
                "hour_mask": [1, 1, 1, 1, 1, 1, 1],
                "vent_on": 1,
                "hour_vent": [1],
            },
            "next24h": {
                "label": "NEXT_24H",
                "len_hours": 24,
                "sections": {
                    "shock": {"map_low_hours": "brief"},
                    "resp": {"vent_hours": "partial_window", "spo2_min": "low"},
                    "tx": {"antibiotics": "present"},
                },
            },
        },
        "target_text": "NEXT_HOUR\nNEXT_24H",
    }


def _record_with_hour_length(length: int) -> dict[str, object]:
    record = copy.deepcopy(_main_route_record())
    placeholders = expected_hour_placeholders(length)
    record["input_text"] = (
        "<SAMPLE> schema=icu_state_major_textual_v1 STATIC: static{age=70} DAY: "
        "D0 i=0 len=24 dq{vital=dense;lab=drawn;uop=measured} "
        f"HOUR len={length}: {' '.join(placeholders)} <STATE> </SAMPLE>"
    )
    record["hour_placeholders"] = placeholders
    value_row = [92.0, 118.0, 65.0, 80.0, 24.0, 37.4, 94.0]
    mask_row = [1, 1, 1, 1, 1, 1, 1]
    vent_row = [1]
    record["hour_values"] = [value_row[:] for _ in range(length)]
    record["hour_mask"] = [mask_row[:] for _ in range(length)]
    record["hour_vent"] = [vent_row[:] for _ in range(length)]
    return record


def _required_fields() -> list[str]:
    return [
        "schema",
        "route",
        "sample_id",
        "subject_id",
        "hadm_id",
        "stay_id",
        "prediction_hour",
        "split",
        "input_text",
        "hour_value_order",
        "hour_placeholders",
        "hour_values",
        "hour_mask",
        "hour_vent",
        "targets",
        "target_text",
    ]


def _training_block(active_next24: bool, active_vent: bool | None = None) -> dict[str, object]:
    resolved_active_vent = active_next24 if active_vent is None else active_vent
    return {
        "precision": "fp16",
        "learning_rate": 2e-5,
        "max_steps": 1,
        "active_losses": {
            "next_hour_values": True,
            "next_hour_vent": resolved_active_vent,
            "next24_domain": active_next24,
            "next24_binary": active_next24,
            "next24_multiclass": active_next24,
        },
        "loss_weights": {
            "next_hour_values": 1.0,
            "next_hour_vent": 0.25 if resolved_active_vent else 0.0,
            "next24_domain": 0.25 if active_next24 else 0.0,
            "next24_binary": 0.5 if active_next24 else 0.0,
            "next24_multiclass": 0.5 if active_next24 else 0.0,
        },
    }


def _stage_a1_training_block() -> dict[str, object]:
    return {
        "precision": "fp16",
        "learning_rate": 1e-5,
        "max_steps": 2000,
        "warm_start_checkpoint": "/tmp/stage_a/checkpoint-4000",
        "next_hour_delta_loss_weight": 1.0,
        "active_losses": {
            "next_hour_values": True,
            "next_hour_vent": False,
            "next24_domain": False,
            "next24_binary": False,
            "next24_multiclass": False,
        },
        "loss_weights": {
            "next_hour_values": 1.0,
            "next_hour_vent": 0.0,
            "next24_domain": 0.0,
            "next24_binary": 0.0,
            "next24_multiclass": 0.0,
        },
    }


def _stage_b_training_block() -> dict[str, object]:
    return {
        "precision": "fp16",
        "learning_rate": 2e-5,
        "max_steps": 1,
        "stage_a_checkpoint": "/tmp/stage_a/checkpoint-500",
        "active_losses": {
            "next_hour_values": False,
            "next_hour_vent": False,
            "next24_domain": True,
            "next24_binary": True,
            "next24_multiclass": True,
        },
        "loss_weights": {
            "next_hour_values": 0.0,
            "next_hour_vent": 0.0,
            "next24_domain": 0.25,
            "next24_binary": 0.5,
            "next24_multiclass": 0.5,
        },
    }


if __name__ == "__main__":
    unittest.main()
