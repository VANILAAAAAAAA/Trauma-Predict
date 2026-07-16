from __future__ import annotations

import copy
import inspect
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import trauma_predict.training.multires_event_v2 as training_module
from trauma_predict.data.multires_event_v2 import MultiresEventV2RelationContract
from trauma_predict.eval.multires_event_v2 import (
    evaluate_teacher_forced,
    exact_teacher_forced_loss,
    teacher_forced_model_inputs,
)
from trauma_predict.eval.multires_event_v2_free_running import (
    evaluate_free_running_v2,
    validate_rank_local_artifact_preflight,
    verify_rank_local_artifact_preflight,
)
from trauma_predict.eval.multires_event_v2_metric_contract import (
    load_trajectory_metric_contract,
)
from trauma_predict.modeling.multires_event_v2.config import MultiResolutionEventV2Config
from trauma_predict.training.config import load_yaml_config_unexpanded
from trauma_predict.training.multires_event_v2 import (
    EXPECTED_FORMAL_MODEL_PARAMETER_COUNT,
    ROUTE,
    build_multires_event_v2_model,
    load_multires_event_v2_configs,
    validate_multires_event_v2_configs,
    validate_relation_runtime_axes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPO_ROOT / "configs/train/p100_multires_event_v2_relation_v2.yaml"


def _load() -> tuple[dict, dict, dict]:
    train, dataset, model, _, _ = load_multires_event_v2_configs(
        TRAIN_PATH,
        repo_root=REPO_ROOT,
    )
    return train, dataset, model


def _drift(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if value is None:
        return 1
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.125
    if isinstance(value, str):
        return value + "_drift"
    if isinstance(value, list):
        return [*value, "drift"]
    if isinstance(value, dict):
        return {**value, "drift": True}
    raise TypeError(type(value))


class RelationV2RouteTest(unittest.TestCase):
    def test_hosted_locations_expand_after_authored_contract_validation(self) -> None:
        hosted = {
            "TRAUMA_PREDICT_DATA_ROOT": "/kaggle/temp/relation_v2_p100/mounted_data/base",
            "TRAUMA_PREDICT_V2_TARGET_ROOT": "/kaggle/temp/relation_v2_p100/mounted_data/target",
            "TRAUMA_PREDICT_OUTPUT_ROOT": "/kaggle/working",
        }
        with patch.dict(os.environ, hosted, clear=False):
            train, dataset, _, _, _ = load_multires_event_v2_configs(
                TRAIN_PATH,
                repo_root=REPO_ROOT,
            )
        self.assertEqual(dataset["base"]["root"], hosted["TRAUMA_PREDICT_DATA_ROOT"])
        self.assertEqual(
            dataset["target"]["root"],
            hosted["TRAUMA_PREDICT_V2_TARGET_ROOT"],
        )
        self.assertEqual(
            dataset["normalization"]["path"],
            "/kaggle/working/contracts/multires_event_v1_input_normalization.json",
        )
        self.assertEqual(
            train["outputs"]["output_dir"],
            "/kaggle/working/p100_multires_event_v2_relation_v2",
        )
        self.assertEqual(
            train["outputs"]["metrics_jsonl"],
            "/kaggle/working/p100_multires_event_v2_relation_v2/metrics.jsonl",
        )

    def test_hosted_environment_cannot_authorize_hardcoded_runtime_paths(self) -> None:
        hosted = {
            "TRAUMA_PREDICT_DATA_ROOT": "/kaggle/temp/relation_v2_p100/mounted_data/base",
            "TRAUMA_PREDICT_V2_TARGET_ROOT": "/kaggle/temp/relation_v2_p100/mounted_data/target",
            "TRAUMA_PREDICT_OUTPUT_ROOT": "/kaggle/working",
        }
        authored_train = load_yaml_config_unexpanded(TRAIN_PATH)
        dataset_path = REPO_ROOT / authored_train["dataset"]["config_path"]
        model_path = REPO_ROOT / authored_train["model"]["config_path"]
        authored_model = load_yaml_config_unexpanded(model_path)
        cases = (
            (
                ("base", "root"),
                hosted["TRAUMA_PREDICT_DATA_ROOT"],
                "dataset.base.root",
            ),
            (
                ("target", "root"),
                hosted["TRAUMA_PREDICT_V2_TARGET_ROOT"],
                "dataset.target.root",
            ),
            (
                ("normalization", "path"),
                "/kaggle/working/contracts/multires_event_v1_input_normalization.json",
                "dataset.normalization",
            ),
        )
        for keys, value, error in cases:
            authored_dataset = load_yaml_config_unexpanded(dataset_path)
            authored_dataset[keys[0]][keys[1]] = value
            payloads = {
                TRAIN_PATH.resolve(): authored_train,
                dataset_path.resolve(): authored_dataset,
                model_path.resolve(): authored_model,
            }

            def load_authored(path: Path) -> dict:
                return copy.deepcopy(payloads[Path(path).resolve()])

            with (
                self.subTest(path=".".join(keys)),
                patch.dict(os.environ, hosted, clear=False),
                patch.object(
                    training_module,
                    "load_yaml_config_unexpanded",
                    side_effect=load_authored,
                ),
                self.assertRaisesRegex(ValueError, error),
            ):
                load_multires_event_v2_configs(TRAIN_PATH, repo_root=REPO_ROOT)

    def test_missing_hosted_location_remains_unexpanded_and_fails_closed(self) -> None:
        hosted_without_target = {
            "TRAUMA_PREDICT_DATA_ROOT": "/kaggle/temp/relation_v2_p100/mounted_data/base",
            "TRAUMA_PREDICT_OUTPUT_ROOT": "/kaggle/working",
        }
        with patch.dict(os.environ, hosted_without_target, clear=True):
            _, dataset, _, _, _ = load_multires_event_v2_configs(
                TRAIN_PATH,
                repo_root=REPO_ROOT,
            )
        self.assertEqual(
            dataset["target"]["root"],
            "${TRAUMA_PREDICT_V2_TARGET_ROOT}",
        )
        with self.assertRaisesRegex(ValueError, "unexpanded environment variable"):
            training_module.resolve_repo_path(dataset["target"]["root"], REPO_ROOT)

    def test_single_route_is_hash_bound_and_builds_exact_model(self) -> None:
        train, dataset, model = _load()
        self.assertEqual(train["route"], dataset["route"])
        self.assertEqual(dataset["route"], model["route"])
        self.assertEqual(model["route"], ROUTE)
        self.assertEqual(train["run_name"], "p100_multires_event_v2_relation_v2")
        self.assertNotIn("mode", train)
        self.assertNotIn("comparison", train)
        training = train["training"]
        self.assertIs(training["resume"], True)
        self.assertEqual(training["required_cuda_devices"], 1)
        self.assertEqual(training["required_world_size"], 1)
        self.assertEqual(training["required_device_name_substring"], "P100")
        self.assertEqual(training["per_device_train_batch_size"], 64)
        self.assertEqual(training["per_device_eval_batch_size"], 32)
        self.assertEqual(training["gradient_accumulation_steps"], 1)
        self.assertEqual(
            training["per_device_train_batch_size"]
            * training["required_world_size"]
            * training["gradient_accumulation_steps"],
            64,
        )
        self.assertEqual(
            model["architecture"]["input_only_temporal_fusion"],
            "block_geometry_softmax_v1",
        )
        self.assertEqual(EXPECTED_FORMAL_MODEL_PARAMETER_COUNT, 48_728_439)
        self.assertEqual(
            model["formal_contract"]["exact_parameter_count"],
            EXPECTED_FORMAL_MODEL_PARAMETER_COUNT,
        )
        relations = MultiresEventV2RelationContract.from_default_config()
        self.assertEqual(relations.bundle_hash, model["relation_contract"]["bundle_sha256"])
        self.assertEqual(dict(relations.file_hashes), model["relation_contract"]["files"])
        self.assertEqual((len(relations.target_edges), len(relations.input_target_edges)), (52, 39))
        built = build_multires_event_v2_model(model, relation_contract=relations)
        self.assertEqual(
            sum(parameter.numel() for parameter in built.parameters()),
            EXPECTED_FORMAL_MODEL_PARAMETER_COUNT,
        )

    def test_architecture_rejects_switches_and_unknown_keys(self) -> None:
        _, _, model = _load()
        cases = (
            ("mode", "block"),
            ("disable_relation", True),
            ("relation_gate", 0.0),
            ("edge_policy", "none"),
            ("target_relation_override", []),
            ("input_target_relation_override", []),
            ("unknown_architecture_key", 1),
        )
        for key, value in cases:
            with self.subTest(key=key):
                architecture = copy.deepcopy(model["architecture"])
                architecture[key] = value
                with self.assertRaises(ValueError):
                    MultiResolutionEventV2Config.from_mapping(architecture)

    def test_all_active_config_sections_fail_closed_on_unknown_keys(self) -> None:
        train, dataset, model = _load()
        cases = (
            (train, "ablation", True),
            (dataset, "variant", "block"),
            (model, "disable_relation", True),
        )
        for source, key, value in cases:
            with self.subTest(key=key):
                changed_train, changed_dataset, changed_model = (
                    copy.deepcopy(train), copy.deepcopy(dataset), copy.deepcopy(model)
                )
                if source is train:
                    changed_train[key] = value
                elif source is dataset:
                    changed_dataset[key] = value
                else:
                    changed_model[key] = value
                with self.assertRaises(ValueError):
                    validate_multires_event_v2_configs(
                        changed_train, changed_dataset, changed_model
                    )
        for section in ("objective", "evaluation", "training"):
            changed = copy.deepcopy(train)
            changed[section]["unknown"] = True
            with self.subTest(section=section), self.assertRaises(ValueError):
                validate_multires_event_v2_configs(changed, dataset, model)
        for section in ("formal_contract", "relation_contract"):
            changed = copy.deepcopy(model)
            changed[section]["unknown"] = True
            with self.subTest(section=section), self.assertRaises(ValueError):
                validate_multires_event_v2_configs(train, dataset, changed)

    def test_every_training_and_architecture_value_is_frozen(self) -> None:
        train, dataset, model = _load()
        for section_name, source in (
            ("training", train["training"]),
            ("architecture", model["architecture"]),
            ("formal_contract", model["formal_contract"]),
            ("relation_contract", model["relation_contract"]),
        ):
            for key, value in source.items():
                with self.subTest(section=section_name, key=key):
                    changed_train = copy.deepcopy(train)
                    changed_model = copy.deepcopy(model)
                    target = (
                        changed_train[section_name]
                        if section_name == "training"
                        else changed_model[section_name]
                    )
                    target[key] = _drift(value)
                    with self.assertRaises(ValueError):
                        validate_multires_event_v2_configs(
                            changed_train,
                            dataset,
                            changed_model,
                        )

    def test_every_dataset_nested_section_is_exact_and_fail_closed(self) -> None:
        train, dataset, model = _load()
        for section in (
            "base", "target", "expected_counts", "normalization", "loader", "preflight"
        ):
            changed = copy.deepcopy(dataset)
            changed[section]["unexpected"] = True
            with self.subTest(section=section, case="unknown"), self.assertRaises(ValueError):
                validate_multires_event_v2_configs(train, changed, model)
            for key, value in dataset[section].items():
                changed = copy.deepcopy(dataset)
                changed[section][key] = _drift(value)
                with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                    validate_multires_event_v2_configs(train, changed, model)

    def test_active_apis_have_no_runtime_relation_switch(self) -> None:
        forbidden = {"mode", "relation_adjacency", "relation_type_lags"}
        for function in (
            teacher_forced_model_inputs,
            exact_teacher_forced_loss,
            evaluate_teacher_forced,
            evaluate_free_running_v2,
            verify_rank_local_artifact_preflight,
            validate_rank_local_artifact_preflight,
            build_multires_event_v2_model,
        ):
            with self.subTest(function=function.__name__):
                self.assertTrue(
                    forbidden.isdisjoint(inspect.signature(function).parameters)
                )

    def test_metric_contract_uses_23_relation_v2_cross_edges(self) -> None:
        train, _, _ = _load()
        relations = MultiresEventV2RelationContract.from_default_config()
        payload = load_trajectory_metric_contract(
            REPO_ROOT / train["trajectory_metric_contract"],
            expected_sha256=train["trajectory_metric_contract_hash"],
            relation_contract=relations,
        )
        self.assertEqual(payload["decision_authority"], "report_only")
        self.assertEqual(payload["relation_edge_cover"]["expected_edges"], 23)
        self.assertTrue({"bootstrap", "gates", "winner_rule", "comparison"}.isdisjoint(payload))

    def test_runtime_axis_binding_rejects_mutated_target_and_history_axes(self) -> None:
        relations = MultiresEventV2RelationContract.from_default_config()
        templates = SimpleNamespace(
            by_key={
                (field.field_id, 1, 1): SimpleNamespace(
                    field_id=field.field_id,
                    field=field.field,
                )
                for field in relations.fields
            }
        )
        target = SimpleNamespace(core_fields=relations.target_fields)
        validate_relation_runtime_axes(relations, target, templates)
        with self.assertRaises(ValueError):
            validate_relation_runtime_axes(
                relations,
                SimpleNamespace(core_fields=tuple(reversed(relations.target_fields))),
                templates,
            )
        templates.by_key[(1, 1, 1)].field = "wrong_name"
        with self.assertRaises(ValueError):
            validate_relation_runtime_axes(relations, target, templates)

    def test_legacy_configs_are_absent(self) -> None:
        forbidden = (
            "configs/model/multires_event_v2.yaml",
            "configs/model/multires_event_v2_capacity_48m.yaml",
            "configs/model/multires_event_v2_relational_primary.yaml",
            "configs/dataset/multires_event_v2_c4.yaml",
            "configs/train/t4x2_multires_event_v2_block.yaml",
            "configs/train/t4x2_multires_event_v2_trajectory.yaml",
            "configs/train/t4x2_multires_event_v2_relational.yaml",
            "configs/train/t4x2_multires_event_v2_smoke.yaml",
        )
        self.assertFalse([name for name in forbidden if (REPO_ROOT / name).exists()])


if __name__ == "__main__":
    unittest.main()
