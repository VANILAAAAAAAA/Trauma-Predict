from __future__ import annotations

import importlib.util
import io
import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "notebooks/kaggle/run_relational_primary_bundle.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("relational_primary_bundle", LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load relational primary launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RelationalPrimaryBundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = _load_launcher()

    def test_launcher_has_no_network_clone_install_or_capacity_probe_path(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        forbidden = (
            "git clone",
            "git fetch",
            "kaggle datasets download",
            "pip install",
            "capacity-probe",
            "run_multires_event_v2_capacity_gated_training",
        )
        for text in forbidden:
            self.assertNotIn(text, source)
        self.assertIn("RELATIONAL_PRIMARY_MOUNTED_PREFLIGHT_OK", source)
        self.assertIn("RELATIONAL_PRIMARY_HOSTED_FORMAL_STEP2_VERIFIED", source)
        self.assertIn("RELATIONAL_PRIMARY_HOSTED_FORMAL_RESUME_STEP3_VERIFIED", source)
        self.assertIn("train_relational_primary.py", source)

    def test_bundle_discovery_requires_one_exact_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_bundle_manifest.json").write_text(
                json.dumps({"schema": self.launcher.MANIFEST_SCHEMA}),
                encoding="utf-8",
            )
            observed_root, payload = self.launcher._find_bundle(root)
            self.assertEqual(observed_root, root.resolve())
            self.assertEqual(payload["schema"], self.launcher.MANIFEST_SCHEMA)

    def test_dependency_preflight_is_early_and_fail_closed(self) -> None:
        versions = {"numpy": "2.2.6", "PyYAML": "6.0.2", "safetensors": "0.5.3"}
        with patch.object(
            self.launcher.importlib.metadata,
            "version",
            side_effect=lambda package: versions[package],
        ):
            self.assertEqual(self.launcher._validate_runtime_dependencies(), versions)
        with patch.object(
            self.launcher.importlib.metadata,
            "version",
            return_value="3.0.0",
        ), self.assertRaisesRegex(RuntimeError, "unsupported numpy"):
            self.launcher._validate_runtime_dependencies()

    def test_safe_extract_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.tar.gz"
            content = b"forbidden"
            with tarfile.open(archive, "w:gz") as handle:
                row = tarfile.TarInfo("../escape.txt")
                row.size = len(content)
                handle.addfile(row, io.BytesIO(content))
            with self.assertRaisesRegex(ValueError, "escapes destination"):
                self.launcher._safe_extract(archive, root / "output")
            self.assertFalse((root / "escape.txt").exists())

    def test_safe_extract_rejects_unregistered_small_pack_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "small.tar"
            with tarfile.open(archive, "w") as handle:
                for name in ("declared.blob", "extra.blob"):
                    content = name.encode("utf-8")
                    row = tarfile.TarInfo(name)
                    row.size = len(content)
                    handle.addfile(row, io.BytesIO(content))
            with self.assertRaisesRegex(ValueError, "members do not match"):
                self.launcher._safe_extract(
                    archive,
                    root / "output",
                    expected_file_members={"declared.blob"},
                )

    def test_materialized_dataset_view_uses_direct_mounted_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            payload = bundle / "payload_000"
            payload.write_bytes(b"mounted-bytes")
            inventory_payload = {
                "schema": "trauma_predict.mounted_file_inventory.v2",
                "file_count": 1,
                "direct_mounted_file_count": 1,
                "packed_file_count": 0,
                "files": [
                    {
                        "storage": "mounted",
                        "mounted_path": payload.name,
                        "destination": "nested/file.bin",
                        "size_bytes": payload.stat().st_size,
                        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                    }
                ],
            }
            inventory = bundle / "base_inventory.json"
            inventory.write_text(json.dumps(inventory_payload), encoding="utf-8")
            declared = {
                "inventory": {
                    "path": inventory.name,
                    "sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
                }
            }
            view = self.launcher._materialize_dataset_view(
                bundle,
                declared,
                root / "view",
                root / "packed",
                label="base",
            )
            linked = view / "nested/file.bin"
            self.assertTrue(linked.is_symlink())
            self.assertEqual(linked.read_bytes(), payload.read_bytes())

    def test_materialized_dataset_view_extracts_only_hash_bound_small_pack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            content = b"small-contract-payload"
            archive = bundle / "payload_base_small_pack.blob"
            with tarfile.open(archive, "w") as handle:
                row = tarfile.TarInfo("packed_000.blob")
                row.size = len(content)
                handle.addfile(row, io.BytesIO(content))
            inventory_payload = {
                "schema": "trauma_predict.mounted_file_inventory.v2",
                "file_count": 1,
                "direct_mounted_file_count": 0,
                "packed_file_count": 1,
                "packed_payload": {
                    "path": archive.name,
                    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    "file_count": 1,
                    "uncompressed_bytes": len(content),
                    "archive_bytes": archive.stat().st_size,
                },
                "files": [
                    {
                        "storage": "packed",
                        "archive_member": "packed_000.blob",
                        "destination": "contracts/file.json",
                        "size_bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ],
            }
            inventory = bundle / "base_inventory.json"
            inventory.write_text(json.dumps(inventory_payload), encoding="utf-8")
            declared = {
                "inventory": {
                    "path": inventory.name,
                    "sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
                }
            }
            view = self.launcher._materialize_dataset_view(
                bundle,
                declared,
                root / "view",
                root / "packed",
                label="base",
            )
            linked = view / "contracts/file.json"
            self.assertTrue(linked.is_symlink())
            self.assertEqual(linked.read_bytes(), content)

    def test_materialized_dataset_view_rejects_missing_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = root / "inventory.json"
            inventory.write_text(
                json.dumps(
                    {
                        "schema": "trauma_predict.mounted_file_inventory.v2",
                        "file_count": 1,
                        "direct_mounted_file_count": 1,
                        "packed_file_count": 0,
                        "files": [
                            {
                                "storage": "mounted",
                                "mounted_path": "absent.bin",
                                "destination": "file.bin",
                                "size_bytes": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            declared = {
                "inventory": {
                    "path": inventory.name,
                    "sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
                }
            }
            with self.assertRaisesRegex(FileNotFoundError, "missing mounted base payload"):
                self.launcher._materialize_dataset_view(
                    root,
                    declared,
                    root / "view",
                    root / "packed",
                    label="base",
                )


if __name__ == "__main__":
    unittest.main()
