from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import inspect
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from trauma_predict.training.multires_event_v2 import (
    OPTIMIZER_CONTRACT_VERSION,
    RAW_JOINT_NLL_REDUCTION,
    summarize_optimizer_health_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = REPO_ROOT / "notebooks/kaggle/run_multires_event_v2.py"
DATASET_EVIDENCE_PATH = (
    REPO_ROOT / "notebooks/kaggle/historical_v8_dataset_evidence.py"
)
ENTRYPOINT_PATH = REPO_ROOT / "notebooks/kaggle/train_multires_event_v2.py"
NOTEBOOK_PATH = REPO_ROOT / "notebooks/kaggle/train_multires_event_v2.ipynb"
VERIFICATION_NOTEBOOK_PATH = REPO_ROOT / "notebooks/kaggle/verify_multires_event_v2.ipynb"
PRIMARY_ENTRYPOINT_PATH = REPO_ROOT / "notebooks/kaggle/train_relational_primary.py"
PRIMARY_NOTEBOOK_PATH = (
    REPO_ROOT / "notebooks/kaggle/train_multires_event_v2_relational_primary.ipynb"
)
BUNDLE_LAUNCHER_PATH = REPO_ROOT / "notebooks/kaggle/run_relational_primary_bundle.py"
BUNDLE_BUILDER_PATH = REPO_ROOT / "tools/build_relational_primary_bundle.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_launcher():
    return _load(LAUNCHER_PATH, "run_multires_event_v2")


def load_dataset_evidence():
    return _load(DATASET_EVIDENCE_PATH, "historical_v8_dataset_evidence")


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
        "local_anchors": 64,
        "world_size": 1,
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MultiresEventV2KaggleRouteTest(unittest.TestCase):
    def test_optimizer_health_summary_is_resume_aware_but_validates_every_raw_row(self) -> None:
        training = dict(yaml.safe_load(
            (
                REPO_ROOT
                / "configs/train/p100_multires_event_v2_relation_v2.yaml"
            ).read_text(encoding="utf-8")
        )["training"])
        training["max_steps"] = 2
        training["warmup_steps"] = 1
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
        launcher = load_dataset_evidence()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "target-a"
            first.mkdir()
            authority = self.target_fixture(first, launcher)
            with patch.object(launcher, "TARGET_AUTHORITY", authority):
                self.assertEqual(launcher.find_exact_target_dataset(root), first.resolve())
                second = root / "target-b"
                second.mkdir()
                self.target_fixture(second, launcher)
                with self.assertRaisesRegex(RuntimeError, "multiple exact"):
                    launcher.find_exact_target_dataset(root)

    def test_target_authority_rejects_relation_or_sidecar_schema_hash_drift(self) -> None:
        launcher = load_dataset_evidence()
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
                        self.assertFalse(launcher._matches_target_authority(root, mutated))

    def test_target_prepare_restores_plain_hosted_shards_and_writes_log(self) -> None:
        launcher = load_dataset_evidence()
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
                    len(list((prepared / "target_shards/train").glob("*.jsonl.gz"))),
                    38,
                )
                report = json.loads((logs / "target_dataset_prepare.json").read_text())
                self.assertEqual(report["materialized_target_shards"], 52)
                self.assertEqual(
                    report["target_shard_layout"],
                    "kaggle_hosted_extracted_target_tree",
                )

    def test_target_dataset_ref_is_historical_and_override_must_match(self) -> None:
        launcher = load_dataset_evidence()
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
        launcher = load_dataset_evidence()
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

    def test_historical_launchers_fail_closed_before_config_or_training(self) -> None:
        modules = (
            (load_launcher(), "HISTORICAL_MULTIRES_EVENT_V2_KAGGLE_DISABLED"),
            (_load(ENTRYPOINT_PATH, "historical_v2_entrypoint"), "HISTORICAL_MULTIRES_EVENT_V2_ENTRYPOINT_DISABLED"),
            (_load(PRIMARY_ENTRYPOINT_PATH, "historical_primary_entrypoint"), "HISTORICAL_RELATIONAL_PRIMARY_DISABLED"),
            (_load(BUNDLE_LAUNCHER_PATH, "historical_bundle_launcher"), "HISTORICAL_RELATIONAL_PRIMARY_BUNDLE_DISABLED"),
            (_load(BUNDLE_BUILDER_PATH, "historical_bundle_builder"), "HISTORICAL_RELATIONAL_PRIMARY_BUNDLE_BUILD_DISABLED"),
        )
        for module, marker in modules:
            with self.subTest(module=module.__name__):
                self.assertEqual(module.HOSTED_ROUTE_STATUS, "pending")
                with self.assertRaisesRegex(RuntimeError, marker):
                    module.main()

    def test_existing_v2_notebooks_are_two_cell_fail_closed_history(self) -> None:
        for path in (NOTEBOOK_PATH, VERIFICATION_NOTEBOOK_PATH, PRIMARY_NOTEBOOK_PATH):
            with self.subTest(path=path.name):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(len(notebook["cells"]), 2)
                markdown = "".join(notebook["cells"][0]["source"])
                code = "".join(notebook["cells"][1]["source"])
                self.assertIn("Historical", markdown)
                self.assertIn("raise RuntimeError", code)
                self.assertNotIn("subprocess", code)
                self.assertNotIn("torch.distributed.run", code)

    def test_historical_surfaces_stay_closed_while_p100_relation_v2_is_active(self) -> None:
        self.assertFalse(
            (REPO_ROOT / "notebooks/kaggle/train_multires_event_v2_relation_v2.ipynb").exists()
        )
        stale_config = "configs/train/t4x2_multires_event_v2_relational.yaml"
        relation_v2_config = "configs/train/p100_multires_event_v2_relation_v2.yaml"
        for path in (
            ENTRYPOINT_PATH,
            PRIMARY_ENTRYPOINT_PATH,
            LAUNCHER_PATH,
            DATASET_EVIDENCE_PATH,
            BUNDLE_LAUNCHER_PATH,
            BUNDLE_BUILDER_PATH,
            NOTEBOOK_PATH,
            VERIFICATION_NOTEBOOK_PATH,
            PRIMARY_NOTEBOOK_PATH,
        ):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn(stale_config, source)
                self.assertNotIn(relation_v2_config, source)
        readme = (REPO_ROOT / "notebooks/kaggle/README.md").read_text(encoding="utf-8")
        self.assertIn("active V2 delivery Notebook", readme)
        self.assertIn("trauma_predict_relation_v2_p100_r9.ipynb", readme)
        self.assertIn("48,728,439-parameter 52+39 Relation V2 route", readme)
        self.assertTrue(
            (
                REPO_ROOT
                / "notebooks/kaggle/trauma_predict_relation_v2_p100_r9.ipynb"
            ).is_file()
        )

    def test_historical_launcher_no_longer_imports_promotion_or_training_route(self) -> None:
        source = LAUNCHER_PATH.read_text(encoding="utf-8")
        self.assertNotIn(
            "from trauma_predict.eval.multires_event_v2_promotion_contract import",
            source,
        )
        self.assertNotIn("from trauma_predict.training.multires_event_v2 import", source)
        self.assertNotIn("def selected_action", source)
        self.assertNotIn("def run_promotion", source)
        self.assertNotIn("STAGE_CONFIGS", source)
        self.assertLess(len(source.splitlines()), 40)
        main_source = inspect.getsource(load_launcher().main)
        self.assertIn("raise RuntimeError(HISTORICAL_DISABLED_MESSAGE)", main_source)


if __name__ == "__main__":
    unittest.main()
