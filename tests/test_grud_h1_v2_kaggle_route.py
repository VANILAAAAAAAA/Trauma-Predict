from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "tools/build_grud_h1_v2_bundle.py"
LAUNCHER_PATH = ROOT / "notebooks/kaggle/run_grud_h1_joint_m4_bundle.py"
TRAIN_ENTRY_PATH = ROOT / "notebooks/kaggle/train_grud_h1_joint_m4_v2.py"
NOTEBOOK_PATH = ROOT / "notebooks/kaggle/trauma_predict_grud_h1_joint_m4_v2.ipynb"
METADATA_PATH = ROOT / "notebooks/kaggle/kernel-metadata-grud-h1-v2.template.json"
TRAIN_CONFIG_PATH = ROOT / "configs/train/p100_grud_h1_joint_m4_v2.yaml"
TRAINING_PATH = ROOT / "src/trauma_predict/training/grud_h1_v2.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class GRUDH1V2KaggleRouteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = _load(BUILDER_PATH, "grud_h1_v2_bundle_builder")
        cls.launcher = _load(LAUNCHER_PATH, "grud_h1_v2_bundle_launcher")
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        cls.train_config = yaml.safe_load(TRAIN_CONFIG_PATH.read_text(encoding="utf-8"))

    def test_fresh_4000_contract_and_identity_match_every_surface(self) -> None:
        config = self.train_config
        training = config["training"]
        evaluation = config["evaluation"]
        self.assertEqual(config["route"], self.builder.ROUTE)
        self.assertEqual(config["run_name"], self.builder.RUN_NAME)
        self.assertTrue(training["fresh_start"])
        self.assertFalse(training["resume"])
        self.assertFalse(training["forced_stop"])
        self.assertEqual(training["max_steps"], 4000)
        self.assertFalse(evaluation["final_evaluation_in_training_notebook"])
        self.assertFalse(evaluation["free_running_in_training_notebook"])

        builder_source = BUILDER_PATH.read_text(encoding="utf-8")
        launcher_source = LAUNCHER_PATH.read_text(encoding="utf-8")
        training_source = TRAINING_PATH.read_text(encoding="utf-8")
        self.assertEqual(self.builder.BUNDLE_SCHEMA, self.launcher.SCIENCE_SCHEMA)
        self.assertEqual(self.builder.DATASET_REF, self.launcher.SCIENCE_DATASET_REF)
        self.assertEqual(self.builder.RUNTIME_DATASET_REF, self.launcher.RUNTIME_DATASET_REF)
        self.assertEqual(Path(self.builder.SCIENCE_DATA_ARCHIVE_NAME).suffix, ".blob")
        self.assertEqual(Path(self.builder.SOURCE_ARCHIVE_NAME).suffix, ".blob")
        for literal in (
            '"fresh_start": True',
            '"target_step": 4000',
            '"forced_stop": False',
        ):
            self.assertIn(literal, builder_source)
        self.assertIn("restored_step=0", training_source)
        self.assertIn("target_step={EXPECTED_OPTIMIZER_STEPS}", training_source)
        self.assertNotIn("--resume", launcher_source)
        self.assertNotIn("prior-output", launcher_source)
        self.assertNotIn("continuous control", launcher_source.lower())

    def test_notebook_is_thin_and_uses_only_two_prebound_datasets(self) -> None:
        self.assertEqual(self.metadata["machine_shape"], "NvidiaTeslaP100")
        self.assertTrue(self.metadata["enable_gpu"])
        self.assertFalse(self.metadata["enable_internet"])
        self.assertEqual(
            self.metadata["dataset_sources"],
            [self.builder.DATASET_REF, self.builder.RUNTIME_DATASET_REF],
        )
        self.assertEqual(self.metadata["kernel_sources"], [])
        self.assertEqual(self.metadata["competition_sources"], [])
        self.assertEqual(self.metadata["model_sources"], [])

        self.assertEqual(len(self.notebook["cells"]), 4)
        code_cells = [
            "".join(cell["source"])
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        self.assertEqual(len(code_cells), 3)
        code = "\n".join(code_cells)
        self.assertIn("Path('/kaggle/input')", code)
        self.assertIn("bootstrap.bind_science_input()", code_cells[0])
        self.assertEqual(code_cells[1].strip(), "RUNTIME_ENVIRONMENT = bootstrap.install_p100_runtime()")
        self.assertEqual(
            code_cells[2].strip(),
            "bootstrap.run_training(SCIENCE_STATE, RUNTIME_ENVIRONMENT)",
        )
        imported = {
            alias.name
            for node in ast.walk(ast.parse(code_cells[0]))
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("torch", imported)
        for forbidden in (
            "kagglehub",
            "dataset_download",
            "kaggle datasets download",
            "git clone",
            "git fetch",
            "requests.",
            "urllib.",
        ):
            self.assertNotIn(forbidden, code.lower())

    def test_hosted_logs_have_one_marker_namespace_and_phase_markers(self) -> None:
        launcher_source = LAUNCHER_PATH.read_text(encoding="utf-8")
        training_source = TRAINING_PATH.read_text(encoding="utf-8")
        train_entry_source = TRAIN_ENTRY_PATH.read_text(encoding="utf-8")
        for marker in (
            "GRUD_V2_INPUT_OK",
            "GRUD_V2_DATA_OK",
            "GRUD_V2_P100_RUNTIME_ARCHIVE_OK",
            "GRUD_V2_P100_RUNTIME_OK",
            "GRUD_V2_HEARTBEAT",
            "GRUD_V2_EXPORT_OK",
            "GRUD_V2_NOTEBOOK_FINISHED status=SUCCEEDED step=4000",
        ):
            self.assertIn(marker, launcher_source)
        for marker in (
            "GRUD_V2_MODEL_OK",
            "GRUD_V2_TRAINING_START",
            "GRUD_V2_TRAIN_NLL",
            "GRUD_V2_VAL_NLL",
            "GRUD_V2_CHECKPOINT_OK",
            "GRUD_V2_TRAINING_FINISHED",
        ):
            self.assertIn(marker, training_source)
        self.assertIn("mode=fresh target_step=4000", train_entry_source)
        self.assertIn('if stripped.startswith("GRUD_V2_")', launcher_source)
        self.assertIn("timeout=5.0", launcher_source)
        self.assertIn("now - last_heartbeat >= 300", launcher_source)

    def test_manifest_locator_requires_exactly_one_prebound_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            first.mkdir()
            payload = {
                "schema": self.launcher.SCIENCE_SCHEMA,
                "dataset_ref": self.launcher.SCIENCE_DATASET_REF,
            }
            (first / "grud_v2_bundle_manifest.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            bundle, observed = self.launcher._find_manifest(
                root,
                "grud_v2_bundle_manifest.json",
                schema=self.launcher.SCIENCE_SCHEMA,
                dataset_ref=self.launcher.SCIENCE_DATASET_REF,
            )
            self.assertEqual(bundle, first.resolve())
            self.assertEqual(observed, payload)
            second = root / "second"
            second.mkdir()
            (second / "grud_v2_bundle_manifest.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "exactly one pre-bound"):
                self.launcher._find_manifest(
                    root,
                    "grud_v2_bundle_manifest.json",
                    schema=self.launcher.SCIENCE_SCHEMA,
                    dataset_ref=self.launcher.SCIENCE_DATASET_REF,
                )

    def test_science_archive_is_deterministic_regular_only_and_hashable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h1 = root / "h1"
            target = root / "target"
            normalization = root / "normalization.json"
            for path, content in (
                (h1 / "dataset_manifest.json", b'{"h1":true}\n'),
                (h1 / "sample_manifest.csv", b"sample_id,split\ns1,train\n"),
                (h1 / "h1_event_templates.json", b"{}\n"),
                (h1 / "SUCCEEDED", b"ok\n"),
                (h1 / "contracts/contract.json", b"{}\n"),
                (h1 / "h1_shards/train/train-00000.jsonl.gz", b"h1-shard"),
                (target / "dataset_manifest.json", b'{"target":true}\n'),
                (target / "sample_manifest.csv", b"sample_id,split\ns1,train\n"),
                (target / "SUCCEEDED", b"ok\n"),
                (target / "contracts/contract.json", b"{}\n"),
                (target / "target_shards/train/train-00000.jsonl.gz", b"target-shard"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            normalization.write_bytes(b'{"train_only":true}\n')
            h1_locks = {
                name: self.builder.sha256_file(h1 / name)
                for name in (
                    "dataset_manifest.json",
                    "sample_manifest.csv",
                    "h1_event_templates.json",
                )
            }
            target_locks = {
                name: self.builder.sha256_file(target / name)
                for name in ("dataset_manifest.json", "sample_manifest.csv")
            }
            output_a = root / "output-a"
            output_b = root / "output-b"
            output_a.mkdir()
            output_b.mkdir()
            with (
                mock.patch.object(self.builder, "H1_LOCKS", h1_locks),
                mock.patch.object(self.builder, "TARGET_LOCKS", target_locks),
                mock.patch.object(
                    self.builder,
                    "NORMALIZATION_SHA256",
                    self.builder.sha256_file(normalization),
                ),
                mock.patch.object(self.builder, "_validate_counts"),
            ):
                row_a = self.builder.build_data_archive(
                    output_a, h1, target, normalization
                )
                row_b = self.builder.build_data_archive(
                    output_b, h1, target, normalization
                )
            archive_a = output_a / row_a["path"]
            archive_b = output_b / row_b["path"]
            self.assertEqual(row_a["path"], self.builder.SCIENCE_DATA_ARCHIVE_NAME)
            self.assertEqual(archive_a.suffix, ".blob")
            self.assertEqual(archive_a.read_bytes(), archive_b.read_bytes())
            self.assertEqual(row_a["sha256"], row_b["sha256"])
            with tarfile.open(archive_a, "r:") as handle:
                members = handle.getmembers()
            self.assertTrue(members)
            self.assertTrue(all(member.isfile() for member in members))
            self.assertTrue(all(member.mtime == 0 for member in members))
            self.assertTrue(all(member.mode == 0o444 for member in members))
            self.assertTrue(
                all(
                    not Path(member.name).is_absolute()
                    and ".." not in Path(member.name).parts
                    for member in members
                )
            )

    def test_safe_extractor_rejects_traversal_and_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            traversal = root / "traversal.tar"
            with tarfile.open(traversal, "w", format=tarfile.USTAR_FORMAT) as handle:
                member = tarfile.TarInfo("../escape")
                member.size = 1
                handle.addfile(member, io.BytesIO(b"x"))
            with self.assertRaisesRegex(ValueError, "escapes destination"):
                self.launcher._extract_regular_ustar(traversal, root / "out-a")
            self.assertFalse((root / "escape").exists())

            links = root / "links.tar"
            with tarfile.open(links, "w", format=tarfile.USTAR_FORMAT) as handle:
                member = tarfile.TarInfo("link")
                member.type = tarfile.SYMTYPE
                member.linkname = "target"
                handle.addfile(member)
            with self.assertRaisesRegex(ValueError, "regular files only"):
                self.launcher._extract_regular_ustar(links, root / "out-b")

    def test_training_output_accepts_nested_checkpoint_rows_and_verifies_both(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            selected = output / "checkpoint-500/checkpoint.pt"
            final = output / "checkpoint-4000/checkpoint.pt"
            selected.parent.mkdir()
            final.parent.mkdir()
            selected.write_bytes(b"selected")
            final.write_bytes(b"final")
            manifest = {
                "schema_version": "trauma_predict.grud_h1_v2_training_manifest.v1",
                "route": "grud_h1_to_joint_m4_v2",
                "status": "SUCCEEDED",
                "completed_step": 4000,
                "selected_checkpoint": {
                    "path": "checkpoint-500/checkpoint.pt",
                    "sha256": _sha256(b"selected"),
                    "step": 500,
                },
                "step_4000_checkpoint": {
                    "path": "checkpoint-4000/checkpoint.pt",
                    "sha256": _sha256(b"final"),
                },
            }
            manifest_path = output / "training_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            observed = self.launcher._validate_training_output(output)
            self.assertEqual(observed, (selected, manifest_path, _sha256(b"selected")))

            manifest["step_4000_checkpoint"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "step-4000 checkpoint"):
                self.launcher._validate_training_output(output)

    def test_bundle_output_cannot_be_parent_or_child_of_an_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            h1 = root / "h1"
            target = root / "target"
            normalization = root / "normalization.json"
            for path in (repo, h1, target):
                path.mkdir()
            normalization.write_text("{}\n", encoding="utf-8")
            common = {
                "repo_root": repo,
                "h1_root": h1,
                "target_root": target,
                "normalization_path": normalization,
                "dataset_ref": self.builder.DATASET_REF,
                "notebook_ref": self.builder.NOTEBOOK_REF,
            }
            with self.assertRaisesRegex(ValueError, "overlaps"):
                self.builder.build_bundle(output=repo / "bundle", **common)
            with self.assertRaisesRegex(ValueError, "overlaps"):
                self.builder.build_bundle(output=root, **common)


if __name__ == "__main__":
    unittest.main()
