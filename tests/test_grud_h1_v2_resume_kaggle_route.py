from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "tools/build_grud_h1_v2_resume_bundle.py"
LAUNCHER_PATH = ROOT / "notebooks/kaggle/run_grud_h1_joint_m4_resume_bundle.py"
NOTEBOOK_PATH = ROOT / "notebooks/kaggle/trauma_predict_grud_h1_joint_m4_v2_resume_2500.ipynb"
METADATA_PATH = ROOT / "notebooks/kaggle/kernel-metadata-grud-h1-v2-resume-2500.template.json"
CONFIG_PATH = ROOT / "configs/train/p100_grud_h1_joint_m4_v2_resume_2500.yaml"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class GRUDH1V2ResumeKaggleRouteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = _load(BUILDER_PATH, "grud_resume_builder")
        cls.launcher = _load(LAUNCHER_PATH, "grud_resume_launcher")
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        cls.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_every_surface_agrees_on_2500_to_4000(self) -> None:
        self.assertEqual(self.builder.BUNDLE_SCHEMA, self.launcher.RESUME_SCHEMA)
        self.assertEqual(self.builder.DATASET_REF, self.launcher.RESUME_DATASET_REF)
        self.assertEqual(self.builder.NOTEBOOK_REF, self.launcher.NOTEBOOK_REF)
        self.assertEqual(self.config["training"]["max_steps"], 4000)
        self.assertEqual(self.config["resume_state"]["checkpoint_step"], 2500)
        self.assertEqual(self.builder.RESUME_STEP, 2500)
        self.assertEqual(self.builder.TARGET_STEP, 4000)
        self.assertEqual(self.builder.CHECKPOINT_SHA256, self.launcher.EXPECTED_CHECKPOINT_SHA256)
        self.assertEqual(Path(self.builder.SOURCE_ARCHIVE_NAME).suffix, ".blob")
        self.assertEqual(Path(self.builder.STATE_ARCHIVE_NAME).suffix, ".blob")

    def test_notebook_is_thin_and_binds_three_private_datasets(self) -> None:
        self.assertTrue(self.metadata["enable_gpu"])
        self.assertFalse(self.metadata["enable_internet"])
        self.assertEqual(self.metadata["machine_shape"], "NvidiaTeslaP100")
        self.assertEqual(
            self.metadata["dataset_sources"],
            [
                self.builder.SCIENCE_DATASET_REF,
                self.builder.RUNTIME_DATASET_REF,
                self.builder.DATASET_REF,
            ],
        )
        self.assertEqual(len(self.notebook["cells"]), 4)
        cell_ids = [cell.get("id") for cell in self.notebook["cells"]]
        self.assertTrue(all(cell_ids))
        self.assertEqual(len(cell_ids), len(set(cell_ids)))
        code_cells = [
            "".join(cell["source"])
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        self.assertEqual(len(code_cells), 3)
        self.assertIn("bootstrap.bind_resume_inputs()", code_cells[0])
        self.assertEqual(
            code_cells[1].strip(), "RUNTIME_ENVIRONMENT = bootstrap.install_p100_runtime()"
        )
        self.assertEqual(
            code_cells[2].strip(),
            "bootstrap.run_training(RESUME_STATE, RUNTIME_ENVIRONMENT)",
        )
        self.assertNotIn("kagglehub", "\n".join(code_cells).lower())

    def test_state_archive_is_deterministic_and_hash_locked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recovery = root / "recovery"
            paths = self.builder._recovery_paths(recovery)
            contents = {
                "checkpoint": b"checkpoint",
                "checkpoint_manifest": b"checkpoint-manifest",
                "source_metrics": b"metrics",
                "source_training_manifest": b"training-manifest",
                "training_log": b"training-log",
                "runtime_log": b"runtime-log",
                "kernel_log": b"kernel-log",
            }
            for key, content in contents.items():
                paths[key].parent.mkdir(parents=True, exist_ok=True)
                paths[key].write_bytes(content)
            locks = {
                "CHECKPOINT_SHA256": _sha256(contents["checkpoint"]),
                "CHECKPOINT_MANIFEST_SHA256": _sha256(contents["checkpoint_manifest"]),
                "SOURCE_METRICS_SHA256": _sha256(contents["source_metrics"]),
                "SOURCE_TRAINING_MANIFEST_SHA256": _sha256(
                    contents["source_training_manifest"]
                ),
            }
            output_a = root / "a"
            output_b = root / "b"
            output_a.mkdir()
            output_b.mkdir()
            patches = [mock.patch.object(self.builder, key, value) for key, value in locks.items()]
            for patcher in patches:
                patcher.start()
            try:
                row_a = self.builder.build_state_archive(output_a, recovery)
                row_b = self.builder.build_state_archive(output_b, recovery)
            finally:
                for patcher in reversed(patches):
                    patcher.stop()
            archive_a = output_a / row_a["path"]
            archive_b = output_b / row_b["path"]
            self.assertEqual(archive_a.read_bytes(), archive_b.read_bytes())
            self.assertEqual(row_a["sha256"], row_b["sha256"])

    def test_completed_output_must_preserve_resume_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            selected = output / "checkpoint-3500/checkpoint.pt"
            final = output / "checkpoint-4000/checkpoint.pt"

            def write_checkpoint(path: Path, step: int) -> str:
                path.parent.mkdir()
                torch.save(
                    {
                        "schema_version": "trauma_predict.grud_h1_v2_checkpoint.v2",
                        "model_contract": "grud_h1_joint_m4_v2",
                        "step": step,
                        "model_state_dict": {},
                        "optimizer_state_dict": {},
                        "scheduler_state_dict": {},
                        "grad_scaler_state_dict": {},
                        "validation": {"step": step},
                        "trainer_state": {
                            "global_step": step,
                            "epoch": step // 48,
                            "microbatches_consumed_in_epoch": 1,
                            "gradient_accumulation_steps": 2,
                            "optimizer_steps_per_epoch": 48,
                            "exact_rng_continuity_from_this_checkpoint": True,
                            "source_non_bitwise_resume_step": 2500,
                        },
                        "sampler_state": {"epoch": step // 48},
                        "rng_state": {
                            "python": (),
                            "torch_cpu": torch.get_rng_state(),
                            "torch_cuda": [],
                        },
                    },
                    path,
                )
                digest = self.launcher.sha256_file(path)
                path.with_name("manifest.json").write_text(
                    json.dumps(
                        {
                            "schema_version": (
                                "trauma_predict.grud_h1_v2_checkpoint_manifest.v2"
                            ),
                            "step": step,
                            "checkpoint": "checkpoint.pt",
                            "checkpoint_sha256": digest,
                            "exact_resume_state": True,
                        }
                    ),
                    encoding="utf-8",
                )
                return digest

            selected_sha256 = write_checkpoint(selected, 3500)
            final_sha256 = write_checkpoint(final, 4000)
            manifest = {
                "schema_version": "trauma_predict.grud_h1_v2_training_manifest.v1",
                "route": "grud_h1_to_joint_m4_v2",
                "execution_mode": "resume_2500_to_4000",
                "status": "SUCCEEDED",
                "completed_step": 4000,
                "resume_lineage": {
                    "checkpoint_step": 2500,
                    "checkpoint_sha256": self.launcher.EXPECTED_CHECKPOINT_SHA256,
                    "rng_continuity": "deterministic_reset_not_bitwise_equivalent",
                },
                "selected_checkpoint": {
                    "path": "checkpoint-3500/checkpoint.pt",
                    "sha256": selected_sha256,
                    "step": 3500,
                },
                "step_4000_checkpoint": {
                    "path": "checkpoint-4000/checkpoint.pt",
                    "sha256": final_sha256,
                },
            }
            manifest_path = output / "training_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            observed = self.launcher._validate_training_output(output)
            self.assertEqual(observed, (selected, manifest_path, selected_sha256))
            with (
                mock.patch.object(
                    self.launcher,
                    "EXPECTED_RUNTIME_TORCH_VERSION",
                    str(torch.__version__),
                ),
                mock.patch.dict(
                    self.launcher.os.environ,
                    {
                        "TRAUMA_PREDICT_RUNTIME_LOCK_SHA256": (
                            self.launcher.EXPECTED_RUNTIME_CONTRACT_SHA256
                        )
                    },
                ),
            ):
                self.launcher._audit_training_checkpoint_payloads(output)
            manifest["resume_lineage"]["checkpoint_step"] = 2000
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lineage"):
                self.launcher._validate_training_output(output)


if __name__ == "__main__":
    unittest.main()
