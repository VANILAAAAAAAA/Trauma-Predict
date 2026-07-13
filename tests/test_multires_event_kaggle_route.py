from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = REPO_ROOT / "notebooks/kaggle/run_multires_event_v1.py"
NOTEBOOK_PATH = REPO_ROOT / "notebooks/kaggle/train_multires_event_v1.ipynb"


def load_launcher():
    spec = importlib.util.spec_from_file_location("run_multires_event_v1", LAUNCHER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MultiresEventKaggleRouteTest(unittest.TestCase):
    def exact_manifest(self) -> dict:
        return {
            "dataset_id": "multires_event_v1_c4_full_20260712",
            "fingerprint": "d58d003b6a9b2dd7c1f8d269a1867b534ea475a91118d7d4d44804bee69f9e47",
            "counts": {
                "samples": 50350,
                "selected_by_split": {"train": 37734, "val": 6309, "test": 6307},
                "completed_shards": 52,
            },
        }

    def test_locator_requires_one_exact_attached_dataset(self) -> None:
        launcher = load_launcher()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "dataset-a"
            first.mkdir()
            (first / "dataset_manifest.json").write_text(json.dumps(self.exact_manifest()))
            self.assertEqual(launcher.find_exact_attached_dataset(root), first.resolve())
            second = root / "dataset-b"
            second.mkdir()
            (second / "dataset_manifest.json").write_text(json.dumps(self.exact_manifest()))
            with self.assertRaisesRegex(RuntimeError, "multiple exact"):
                launcher.find_exact_attached_dataset(root)

    def test_notebook_pins_immutable_tag_and_delegates(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(notebook["cells"]), 2)
        code = "".join(notebook["cells"][1]["source"])
        self.assertIn('REQUIRED_GIT_REF = "multires-event-v1-baseline-run-20260712"', code)
        self.assertNotIn("TRAUMA_PREDICT_GIT_REF", code)
        self.assertIn("run_multires_event_v1.py", code)

    def test_shards_zip_extraction_keeps_only_split_tree(self) -> None:
        launcher = load_launcher()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "shards.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("shards/train/train-00000.jsonl.gz", b"gzip-bytes")
                handle.writestr("val/val-00000.jsonl.gz", b"gzip-bytes")
            destination = root / "prepared"
            destination.mkdir()
            self.assertEqual(launcher.safe_extract_shards(archive, destination), 2)
            self.assertTrue((destination / "shards/train/train-00000.jsonl.gz").is_file())
            self.assertTrue((destination / "shards/val/val-00000.jsonl.gz").is_file())

    def test_launcher_is_torchrun_smoke_then_full_without_hf(self) -> None:
        source = LAUNCHER_PATH.read_text(encoding="utf-8")
        self.assertIn('"--nproc_per_node=2"', source)
        self.assertLess(source.index("run_torchrun(SMOKE_CONFIG"), source.index("run_torchrun(FULL_CONFIG"))
        self.assertIn("heartbeat(label, log_path, seconds=300)", source)
        self.assertIn("MULTIRES_EVENT_RUN_CONTRACT", source)
        for key in ("max_steps", "logging_steps", "eval_steps", "save_steps"):
            self.assertIn(f'"{key}"', source)
        self.assertIn('log_path.open("a"', source)
        self.assertIn('"datasets",\n            "download"', source)
        self.assertIn('"--unzip"', source)
        self.assertIn("safe_extract_shards", source)
        self.assertNotIn("ModernBERT", source)
        self.assertNotIn("transformers", source)
        self.assertNotIn("accelerate", source)


if __name__ == "__main__":
    unittest.main()
