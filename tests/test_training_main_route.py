from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from trauma_predict.data.main_route import (
    HourValueNormalizer,
    MainRouteBatchCollator,
    encode_next24_labels,
)
from trauma_predict.training.main_route import validate_main_route_config
from trauma_predict.training.runtime import quarantine_rng_state_files


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
            "<H-01>",
            "<H0>",
            "<STATE>",
        ]
        self.vocab = {token: index for index, token in enumerate(tokens)}

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
            "model": {
                "base_model": "google/flan-t5-base",
                "task": "next24_text_generation",
                "max_input_tokens": 1024,
                "hour_adapter_hidden": 256,
            },
            "training": {"learning_rate": 2e-5, "max_steps": 1},
        }

        with self.assertRaisesRegex(ValueError, "main_hour_adapter_structured_heads"):
            validate_main_route_config(config)

    def test_main_route_config_accepts_structured_route(self) -> None:
        validate_main_route_config({
            "schema_version": "trauma_predict.train_config.v1",
            "model": {
                "base_model": "allenai/longformer-base-4096",
                "task": "main_hour_adapter_structured_heads",
                "max_input_tokens": 4096,
                "hour_adapter_hidden": 256,
            },
            "training": {"precision": "fp16", "learning_rate": 2e-5, "max_steps": 1},
        })

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
    def test_hour_state_adapter_outputs_encoder_hidden_size(self) -> None:
        import torch

        from trauma_predict.modeling.main_route import HourStateAdapter

        adapter = HourStateAdapter(hidden_size=32, adapter_hidden_size=16, dropout=0.0)
        values = torch.zeros((2, 24, 7))
        mask = torch.ones((2, 24, 7))
        vent = torch.zeros((2, 24, 1))

        output = adapter(values, mask, vent)

        self.assertEqual(output.shape, (2, 24, 32))

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


if __name__ == "__main__":
    unittest.main()
