from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
import unittest

import torch

from trauma_predict.data.multires_event import (
    MultiresEventDataset,
    RobustNormalizer,
    SupervisionContract,
)
from trauma_predict.data.multires_event_v2 import (
    MultiresEventV2Collator,
    MultiresEventV2Contract,
    MultiresEventV2Dataset,
    MultiresEventV2RelationContract,
)
from trauma_predict.eval.multires_event_v2 import (
    exact_teacher_forced_loss,
    move_to_device,
)
from trauma_predict.modeling.multires_event_v2.input_field_memory import (
    INPUT_ONLY_FIELD_IDS,
)
from trauma_predict.training.multires_event_v2 import (
    EXPECTED_FORMAL_MODEL_PARAMETER_COUNT,
    _validated_optimizer_loss,
    build_multires_event_v2_model,
    load_lab_scale_artifact,
    load_multires_event_v2_configs,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = ROOT / "configs/train/p100_multires_event_v2_relation_v2.yaml"
BASE_ROOT = Path(
    os.environ.get(
        "TRAUMA_PREDICT_DATA_ROOT",
        "/mnt/d/Data/trauma_predict_work/"
        "multires_event_v1_c4_full_20260712/full",
    )
)
TARGET_ROOT = Path(
    os.environ.get(
        "TRAUMA_PREDICT_V2_TARGET_ROOT",
        "/mnt/d/Data/trauma_predict_work/"
        "multires_event_m4_target_v2_c4_20260714/full_r9",
    )
)
NORMALIZATION_PATH = Path(
    os.environ.get(
        "TRAUMA_PREDICT_V2_NORMALIZATION_PATH",
        "/mnt/d/Data/trauma_predict_work/"
        "kaggle_relational_primary_r9_bundle_6712a7c_final_train_v2/"
        "multires_event_v1_input_normalization.json",
    )
)
OLD_V8_CHECKPOINT = Path(
    os.environ.get(
        "TRAUMA_PREDICT_V2_OLD_CHECKPOINT",
        "/mnt/d/Data/trauma_predict_work/"
        "kaggle_multires_event_v2_relational_primary_partial_20260715_v1/"
        "retained_run/checkpoints/checkpoint-00004000/model.pt",
    )
)
RUN_FORMAL_AUDIT = (
    os.environ.get("TRAUMA_PREDICT_RUN_FORMAL_GRADIENT_AUDIT") == "1"
)
REAL_AUDIT_SAMPLE_IDS = (
    "hadm_28942795_stay_34406519_h42",
    "hadm_29646384_stay_38740124_h66",
)
EXPECTED_NORMALIZATION_SHA256 = (
    "4f54dbeaab4b2becd349d1d8fcaac7b6bdea2567a20874ee7d29338c1f930add"
)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    except AttributeError:  # pragma: no cover - older supported torch
        return torch.cuda.amp.autocast(dtype=torch.float16)


def _real_batch() -> tuple[dict[str, object], object, MultiresEventV2RelationContract]:
    train, dataset_config, _, _, _ = load_multires_event_v2_configs(
        TRAIN_CONFIG,
        repo_root=ROOT,
    )
    supervision = SupervisionContract.from_json(
        ROOT / str(dataset_config["supervision_path"])
    )
    base = MultiresEventDataset(BASE_ROOT, "train", supervision)
    target_contract = MultiresEventV2Contract.from_dataset_root(TARGET_ROOT)
    joined = MultiresEventV2Dataset(
        base,
        TARGET_ROOT,
        contract=target_contract,
        strict=True,
    )
    index_by_sample = {
        sample_id: index for index, sample_id in enumerate(joined.sample_ids)
    }
    missing = set(REAL_AUDIT_SAMPLE_IDS).difference(index_by_sample)
    if missing:
        raise AssertionError(
            f"persisted formal-gradient audit samples are absent: {sorted(missing)}"
        )
    records = [joined[index_by_sample[sample_id]] for sample_id in REAL_AUDIT_SAMPLE_IDS]

    # The two persisted records jointly activate every input-only field, and each
    # field occurs in at least two distinct existing history blocks in one anchor.
    for field_id in INPUT_ONLY_FIELD_IDS:
        per_anchor_blocks = []
        for record in records:
            input_record = record["input_record"]
            blocks = {
                int(event[4])
                for event in input_record["input_events"]
                if int(event[0]) == field_id
            }
            per_anchor_blocks.append(blocks)
        if max(map(len, per_anchor_blocks)) < 2:
            raise AssertionError(
                f"input-only field_id={field_id} lacks persisted multi-block coverage"
            )

    normalizer = RobustNormalizer.from_json(
        NORMALIZATION_PATH,
        expected_dataset_fingerprint=base.dataset_fingerprint,
        expected_supervision_sha256=supervision.source_sha256,
    )
    lab_scale = load_lab_scale_artifact(
        ROOT / str(train["lab_scale_artifact"]),
        expected_content_sha256=str(train["lab_scale_artifact_hash"]),
        contract=target_contract,
    )
    collator = MultiresEventV2Collator(
        contract=target_contract,
        supervision=supervision,
        templates=base.templates,
        normalization=normalizer,
    )
    batch = collator(records)
    metadata = dict(batch["target_primitive_metadata"])
    metadata["lab_scale"] = lab_scale
    batch["target_primitive_metadata"] = metadata
    relation_contract = MultiresEventV2RelationContract.from_default_config()
    return batch, target_contract.process_registry, relation_contract


@unittest.skipUnless(
    RUN_FORMAL_AUDIT,
    "set TRAUMA_PREDICT_RUN_FORMAL_GRADIENT_AUDIT=1 for the 48.7M-model audit",
)
@unittest.skipUnless(
    BASE_ROOT.is_dir()
    and TARGET_ROOT.is_dir()
    and NORMALIZATION_PATH.is_file(),
    "formal V1/r9/normalization artifacts are not mounted",
)
class RelationV2FormalGradientAuditTest(unittest.TestCase):
    def test_exact_real_r9_nll_reaches_all_relation_and_temporal_parameters(self) -> None:
        self.assertEqual(_sha256(NORMALIZATION_PATH), EXPECTED_NORMALIZATION_SHA256)
        batch, registry, relation_contract = _real_batch()
        _, _, model_config, _, _ = load_multires_event_v2_configs(
            TRAIN_CONFIG,
            repo_root=ROOT,
        )
        torch.manual_seed(20260716)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_multires_event_v2_model(
            model_config,
            relation_contract=relation_contract,
        ).to(device)
        model.train()
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            EXPECTED_FORMAL_MODEL_PARAMETER_COUNT,
        )
        self.assertEqual(EXPECTED_FORMAL_MODEL_PARAMETER_COUNT, 48_728_439)

        moved = move_to_device(batch, device)
        outputs, loss_result = exact_teacher_forced_loss(
            model,
            moved,
            registry,
            expected_lab_scale_artifact_hash=(
                "cae827b1f8b1c6a156da4bad340af1b9b0411ca2f5fbe0b9aa8d36ed06cb87bb"
            ),
            autocast=lambda: _autocast(device),
        )
        self.assertEqual(loss_result["primitive_count"], 414)
        self.assertEqual(tuple(loss_result["primitive_log_prob"].shape), (2, 414))
        self.assertEqual(tuple(outputs["field_states"].shape), (2, 6, 29, 480))
        loss = _validated_optimizer_loss(loss_result, expected_local_batch=2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        named = dict(model.named_parameters())
        audited = {
            "target": (
                "target_decoder.target_relation_bias.edge_head_bias",
                (52, 8),
                relation_contract.target_parameter_keys,
            ),
            "input_target": (
                "target_decoder.input_target_relation_bias.edge_head_bias",
                (39, 8),
                relation_contract.input_target_parameter_keys,
            ),
            "temporal": (
                "input_field_memory.input_only_temporal_weight",
                (8, 6),
                tuple(f"field_id={field_id}" for field_id in INPUT_ONLY_FIELD_IDS),
            ),
        }
        for label, (name, shape, row_keys) in audited.items():
            with self.subTest(parameter_bank=label):
                parameter = named[name]
                gradient = parameter.grad
                self.assertEqual(tuple(parameter.shape), shape)
                self.assertIsNotNone(gradient)
                assert gradient is not None
                self.assertTrue(torch.isfinite(gradient).all())
                row_magnitude = gradient.detach().float().abs().sum(dim=1)
                zero_rows = [
                    row_keys[index]
                    for index in torch.nonzero(row_magnitude.eq(0), as_tuple=False)
                    .flatten()
                    .cpu()
                    .tolist()
                ]
                self.assertEqual(zero_rows, [])

        missing_gradients = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        nonfinite_gradients = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all().item())
        ]
        self.assertEqual(missing_gradients, [])
        self.assertEqual(nonfinite_gradients, [])

    @unittest.skipUnless(
        OLD_V8_CHECKPOINT.is_file(),
        "the retained 47,801,855-parameter v8 checkpoint is not mounted",
    )
    def test_actual_v8_checkpoint_is_strictly_rejected(self) -> None:
        _, _, model_config, _, _ = load_multires_event_v2_configs(
            TRAIN_CONFIG,
            repo_root=ROOT,
        )
        model = build_multires_event_v2_model(
            model_config,
            relation_contract=MultiresEventV2RelationContract.from_default_config(),
        )
        try:
            old_state = torch.load(
                OLD_V8_CHECKPOINT,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:  # pragma: no cover - older supported torch
            old_state = torch.load(OLD_V8_CHECKPOINT, map_location="cpu")
        self.assertIn("target_decoder.relation_bias.type_head_bias", old_state)
        self.assertNotIn(
            "target_decoder.input_target_relation_bias.edge_head_bias",
            old_state,
        )
        self.assertNotIn(
            "input_field_memory.input_only_temporal_weight",
            old_state,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "(Missing key|Unexpected key|size mismatch)",
        ):
            model.load_state_dict(old_state, strict=True)


if __name__ == "__main__":
    unittest.main()
