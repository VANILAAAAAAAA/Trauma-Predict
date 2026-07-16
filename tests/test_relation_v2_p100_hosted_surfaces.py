from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "notebooks/kaggle/run_relation_v2_p100_bundle.py"
BUILDER_PATH = ROOT / "tools/build_relation_v2_p100_bundle.py"
NOTEBOOK_PATH = (
    ROOT / "notebooks/kaggle/trauma_predict_relation_v2_p100_r9.ipynb"
)
KERNEL_TEMPLATE_PATH = (
    ROOT / "notebooks/kaggle/kernel-metadata-relation-v2-p100.template.json"
)
RUNTIME_CONTRACT_PATH = (
    ROOT / "configs/runtime/p100_torch_2_10_cu126_cp312.json"
)
RUNTIME_ARCHIVE_NAME = "p100_torch_2_10_cu126_cp312_wheelhouse.blob"


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "".join(notebook["cells"][1]["source"])


def _notebook_function(name: str):
    tree = ast.parse(_notebook_source())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Path": Path,
        "hashlib": hashlib,
        "os": __import__("os"),
        "tarfile": tarfile,
    }
    exec(compile(module, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace[name]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RelationV2P100HostedSurfacesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = _load(LAUNCHER_PATH, "relation_v2_p100_launcher")
        cls.builder = _load(BUILDER_PATH, "relation_v2_p100_builder")

    def test_frozen_route_identity_matches_across_surfaces(self) -> None:
        metadata = json.loads(KERNEL_TEMPLATE_PATH.read_text(encoding="utf-8"))
        notebook_source = _notebook_source()
        model_config = yaml.safe_load(
            (ROOT / "configs/model/multires_event_v2_relation_v2.yaml").read_text(
                encoding="utf-8"
            )
        )
        frozen_count = int(model_config["formal_contract"]["exact_parameter_count"])
        self.assertEqual(frozen_count, 48_728_439)
        self.assertEqual(self.launcher.EXPECTED_PARAMETERS, frozen_count)
        self.assertEqual(self.builder.MODEL_PARAMETER_COUNT, frozen_count)
        self.assertEqual(
            self.launcher.EXPECTED_TRAINING_STOP_STEPS,
            (250, 1500, 2750, 4000),
        )
        self.assertEqual(self.builder.DEFAULT_FREE_RUNNING_MAX_NEW_ANCHORS, 2048)
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaP100")
        self.assertFalse(metadata["enable_internet"])
        self.assertEqual(metadata["dataset_sources"], [self.builder.DATASET_REF])
        self.assertEqual(metadata["kernel_sources"], [self.builder.NOTEBOOK_REF])
        self.assertNotIn("kagglehub", notebook_source)
        self.assertIn("Path('/kaggle/input')", notebook_source)
        self.assertIn("--no-index", notebook_source)
        self.assertIn("--no-deps", notebook_source)
        self.assertIn("2.10.0+cu126", notebook_source)
        self.assertIn("sm_60", notebook_source)
        self.assertIn("TRAUMA_PREDICT_RUNTIME_SITE_PACKAGES", notebook_source)
        notebook_tree = ast.parse(notebook_source)
        imported = {
            alias.name
            for node in ast.walk(notebook_tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("torch", imported)
        self.assertIn("run_relation_v2_p100_bundle.py", notebook_source)
        self.assertIn("hosted_stage_manifest.json", notebook_source)
        self.assertIn("--skip-prior-output-download", notebook_source)
        self.assertIn("--prior-output-root", notebook_source)
        self.assertIn("RELATION_V2_P100_BOOTSTRAP_FRESH_START", notebook_source)
        self.assertIn("RELATION_V2_P100_BOOTSTRAP_BOUND_PRIOR_OUTPUT", notebook_source)
        self.assertNotIn("git clone", notebook_source)
        self.assertNotIn("KAGGLE_KEY", notebook_source)
        self.assertNotIn("torch.distributed.run", notebook_source)

    def test_bundle_v3_uses_one_safe_runtime_archive_under_kaggle_temp(self) -> None:
        notebook_source = _notebook_source()
        builder_source = BUILDER_PATH.read_text(encoding="utf-8")
        launcher_source = LAUNCHER_PATH.read_text(encoding="utf-8")
        schema = "trauma_predict.multires_event_v2_relation_v2_p100_bundle.v3"
        self.assertEqual(self.builder.BUNDLE_SCHEMA, schema)
        self.assertEqual(self.launcher.MANIFEST_SCHEMA, schema)
        self.assertIn(f"SCHEMA = '{schema}'", notebook_source)
        self.assertEqual(self.builder.RUNTIME_ARCHIVE_NAME, RUNTIME_ARCHIVE_NAME)
        self.assertEqual(
            self.launcher.EXPECTED_RUNTIME_ARCHIVE_NAME,
            RUNTIME_ARCHIVE_NAME,
        )
        self.assertIn(
            f"RUNTIME_ARCHIVE_NAME = '{RUNTIME_ARCHIVE_NAME}'",
            notebook_source,
        )
        self.assertIn("Path('/kaggle/temp/relation_v2_p100_runtime", notebook_source)
        self.assertIn("Path('/kaggle/temp/relation_v2_p100_wheelhouse", notebook_source)
        self.assertNotIn("Path('/kaggle/working/relation_v2_p100_runtime", notebook_source)
        self.assertIn("extract_runtime_wheelhouse(archive_path", notebook_source)
        self.assertNotIn("wheel_paths = [resolve_file", notebook_source)
        self.assertIn('"archive": {', builder_source)
        self.assertIn("_validate_runtime_archive(bundle, runtime)", launcher_source)

    def test_runtime_builder_emits_only_contract_and_deterministic_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            wheelhouse = root / "wheelhouse"
            output_a = root / "bundle-a"
            output_b = root / "bundle-b"
            contract_path = repo / self.builder.RUNTIME_CONTRACT_RELATIVE
            contract_path.parent.mkdir(parents=True)
            wheelhouse.mkdir()
            output_a.mkdir()
            output_b.mkdir()
            contract_path.write_text('{"fixture":true}\n', encoding="utf-8")
            contract_digest = hashlib.sha256(contract_path.read_bytes()).hexdigest()
            contents = {
                "alpha-1.0-cp312-cp312-manylinux_x86_64.whl": b"alpha-wheel",
                "torch-2.10.0+cu126-cp312-cp312-manylinux_x86_64.whl": b"torch-wheel",
            }
            rows = []
            for name, content in sorted(contents.items()):
                (wheelhouse / name).write_bytes(content)
                rows.append(
                    {
                        "path": name,
                        "size_bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
            contract = {"files": rows}
            with (
                mock.patch.object(
                    self.builder,
                    "load_runtime_contract",
                    return_value=contract,
                ),
                mock.patch.object(
                    self.builder,
                    "RUNTIME_CONTRACT_SHA256",
                    contract_digest,
                ),
            ):
                runtime_a = self.builder.build_runtime_wheelhouse(
                    repo, wheelhouse, output_a
                )
                runtime_b = self.builder.build_runtime_wheelhouse(
                    repo, wheelhouse, output_b
                )
            expected_top_level = {
                self.builder.RUNTIME_CONTRACT_RELATIVE.name,
                RUNTIME_ARCHIVE_NAME,
            }
            self.assertEqual(
                {path.name for path in output_a.iterdir()}, expected_top_level
            )
            self.assertEqual(
                {path.name for path in output_b.iterdir()}, expected_top_level
            )
            self.assertEqual(runtime_a["files"], rows)
            self.assertEqual(runtime_a["archive"]["path"], RUNTIME_ARCHIVE_NAME)
            self.assertEqual(
                runtime_a["archive"]["sha256"],
                runtime_b["archive"]["sha256"],
            )
            self.assertEqual(
                (output_a / RUNTIME_ARCHIVE_NAME).read_bytes(),
                (output_b / RUNTIME_ARCHIVE_NAME).read_bytes(),
            )
            with tarfile.open(output_a / RUNTIME_ARCHIVE_NAME, "r:") as handle:
                members = handle.getmembers()
            self.assertEqual(
                [member.name for member in members],
                [row["path"] for row in rows],
            )
            self.assertTrue(all(member.isfile() for member in members))
            self.assertTrue(all(member.mtime == 0 for member in members))

    def test_launcher_validates_archive_and_every_logical_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            archive = bundle / RUNTIME_ARCHIVE_NAME
            contents = {
                f"wheel-{index:02d}.whl": f"wheel-{index:02d}".encode("ascii")
                for index in range(28)
            }
            rows = [
                {
                    "path": name,
                    "size_bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
                for name, content in sorted(contents.items())
            ]
            with tarfile.open(
                archive, "w", format=tarfile.USTAR_FORMAT
            ) as handle:
                for row in rows:
                    member = tarfile.TarInfo(row["path"])
                    member.size = row["size_bytes"]
                    member.mode = 0o444
                    member.uid = 0
                    member.gid = 0
                    member.uname = ""
                    member.gname = ""
                    member.mtime = 0
                    handle.addfile(member, io.BytesIO(contents[row["path"]]))
            archive_row = {
                "path": RUNTIME_ARCHIVE_NAME,
                "size_bytes": archive.stat().st_size,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
            runtime = {"archive": archive_row, "files": rows}
            self.assertEqual(
                self.launcher._validate_runtime_archive(bundle, runtime),
                archive,
            )
            tampered = json.loads(json.dumps(runtime))
            tampered["files"][13]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "member hash differs"):
                self.launcher._validate_runtime_archive(bundle, tampered)

    def test_notebook_runtime_extractor_verifies_member_hashes(self) -> None:
        extract = _notebook_function("extract_runtime_wheelhouse")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / RUNTIME_ARCHIVE_NAME
            expected_content = b"registered-wheel"
            observed_content = b"tampered-wheel"
            name = "torch-2.10.0+cu126-cp312-cp312-manylinux_x86_64.whl"
            with tarfile.open(archive, "w") as handle:
                member = tarfile.TarInfo(name)
                member.size = len(observed_content)
                handle.addfile(member, io.BytesIO(observed_content))
            rows = [
                {
                    "path": name,
                    "size_bytes": len(observed_content),
                    "sha256": hashlib.sha256(expected_content).hexdigest(),
                }
            ]
            with self.assertRaisesRegex(ValueError, "member hash differs"):
                extract(archive, root / "wheelhouse", rows)

    def test_notebook_runtime_extractor_rejects_traversal_and_links(self) -> None:
        extract = _notebook_function("extract_runtime_wheelhouse")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            traversal_archive = root / "traversal.blob"
            content = b"forbidden"
            with tarfile.open(traversal_archive, "w") as handle:
                member = tarfile.TarInfo("../escape.whl")
                member.size = len(content)
                handle.addfile(member, io.BytesIO(content))
            rows = [
                {
                    "path": "../escape.whl",
                    "size_bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            ]
            with self.assertRaisesRegex(ValueError, "escapes destination"):
                extract(traversal_archive, root / "traversal-output", rows)
            self.assertFalse((root / "escape.whl").exists())

            link_archive = root / "link.blob"
            with tarfile.open(link_archive, "w") as handle:
                member = tarfile.TarInfo("linked.whl")
                member.type = tarfile.SYMTYPE
                member.linkname = "outside.whl"
                handle.addfile(member)
            link_rows = [
                {
                    "path": "linked.whl",
                    "size_bytes": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                }
            ]
            with self.assertRaisesRegex(ValueError, "Invalid runtime archive member"):
                extract(link_archive, root / "link-output", link_rows)

    def test_p100_runtime_lock_is_complete_and_frozen(self) -> None:
        runtime = self.builder.load_runtime_contract(ROOT)
        self.assertEqual(runtime["schema"], self.builder.RUNTIME_WHEELHOUSE_SCHEMA)
        self.assertEqual(runtime["python_abi"], "cp312")
        self.assertEqual(runtime["torch_version"], "2.10.0+cu126")
        self.assertEqual(runtime["cuda_version"], "12.6")
        self.assertEqual(runtime["required_cuda_arch"], "sm_60")
        self.assertEqual(runtime["inventory_sha256"], self.builder.RUNTIME_INVENTORY_SHA256)
        self.assertEqual(runtime["file_count"], 28)
        self.assertEqual(runtime["total_bytes"], 3_587_233_664)
        self.assertEqual(len(runtime["files"]), 28)
        names = {row["path"] for row in runtime["files"]}
        self.assertIn(
            "torch-2.10.0+cu126-cp312-cp312-manylinux_2_28_x86_64.whl",
            names,
        )
        self.assertEqual(
            hashlib.sha256(RUNTIME_CONTRACT_PATH.read_bytes()).hexdigest(),
            self.builder.RUNTIME_CONTRACT_SHA256,
        )

    def test_launcher_preserves_isolated_runtime_for_training_child(self) -> None:
        launcher_source = LAUNCHER_PATH.read_text(encoding="utf-8")
        self.assertIn("EXPECTED_RUNTIME_CONTRACT_SHA256", launcher_source)
        self.assertIn("_validate_isolated_torch_runtime", launcher_source)
        self.assertIn("torch.cuda.get_arch_list", launcher_source)
        self.assertIn("TRAUMA_PREDICT_RUNTIME_SITE_PACKAGES=runtime_root", launcher_source)
        self.assertIn(
            'PYTHONPATH=os.pathsep.join((runtime_root, str(repo_root / "src")))',
            launcher_source,
        )
        self.assertNotIn('PYTHONPATH=str(repo_root / "src")', launcher_source)

    def test_source_and_prior_output_are_hash_bound(self) -> None:
        launcher_source = LAUNCHER_PATH.read_text(encoding="utf-8")
        builder_source = BUILDER_PATH.read_text(encoding="utf-8")
        for text in (
            "trauma_predict.source_release_inventory.v1",
            "SOURCE_RELEASE.json",
            "source_tree_sha256",
            "git_head_tree",
        ):
            self.assertIn(text, launcher_source)
            self.assertIn(text, builder_source)
        self.assertIn(
            '"notebooks/kaggle/run_relation_v2_p100_bundle.py"',
            builder_source,
        )
        self.assertIn("hosted_stage_manifest.json", launcher_source)
        self.assertIn("notebook_output_download", launcher_source)
        self.assertIn("run_files", launcher_source)
        self.assertNotIn('"kaggle", "kernels", "status"', launcher_source)

    def test_only_explicit_prior_output_404_allows_a_fresh_start(self) -> None:
        class NotFound(RuntimeError):
            status_code = 404

        self.assertTrue(self.launcher._prior_output_not_found(NotFound("missing")))
        self.assertTrue(
            self.launcher._prior_output_not_found(
                RuntimeError("404 not found: notebook has no output")
            )
        )
        self.assertFalse(
            self.launcher._prior_output_not_found(
                RuntimeError("temporary authentication failure")
            )
        )
        self.assertFalse(
            self.launcher._prior_output_not_found(
                RuntimeError(
                    "New Notebooks cannot be attached in non-interactive sessions"
                )
            )
        )

    def test_safe_extract_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.tar"
            content = b"forbidden"
            with tarfile.open(archive, "w") as handle:
                row = tarfile.TarInfo("../escape.txt")
                row.size = len(content)
                handle.addfile(row, io.BytesIO(content))
            with self.assertRaisesRegex(ValueError, "escapes destination"):
                self.launcher._safe_extract_regular_files(
                    archive,
                    root / "output",
                    expected_members={
                        "../escape.txt": {
                            "size_bytes": len(content),
                            "sha256": hashlib.sha256(content).hexdigest(),
                        }
                    },
                    label="fixture",
                )
            self.assertFalse((root / "escape.txt").exists())

    def test_clean_git_source_release_round_trips_every_tracked_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            for relative in sorted(self.builder.REQUIRED_SOURCE_PATHS):
                path = repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"fixture:{relative}\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Hosted Test",
                    "-c",
                    "user.email=hosted-test@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                cwd=repo,
                check=True,
            )
            bundle = root / "bundle"
            bundle.mkdir()
            source, inventory = self.builder.build_source_release(repo, bundle)
            extracted = self.launcher._extract_source(
                bundle,
                {"source": source},
                root / "scratch",
            )
            self.assertEqual(inventory["file_count"], len(self.builder.REQUIRED_SOURCE_PATHS) + 1)
            self.assertEqual(
                json.loads((extracted / "SOURCE_RELEASE.json").read_text(encoding="utf-8"))[
                    "source_tree_sha256"
                ],
                source["source_tree_sha256"],
            )
            from trauma_predict.training.multires_event_v2 import _source_tree_identity

            runtime_identity = _source_tree_identity(extracted)
            self.assertEqual(
                runtime_identity["source_tree_sha256"],
                source["source_tree_sha256"],
            )
            self.assertEqual(runtime_identity["git_commit"], source["git_commit"])
            self.assertTrue(runtime_identity["git_clean"])
            for relative in self.builder.REQUIRED_SOURCE_PATHS:
                self.assertTrue((extracted / relative).is_file())

    def test_checkpoint_validation_hashes_every_state_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-00000250"
            checkpoint.mkdir()
            names = {
                "model.pt",
                "optimizer.pt",
                "scheduler.pt",
                "scaler.pt",
                "trainer_state.json",
                "identity_hashes.json",
                "rng-rank-0000.pt",
                "sampler-rank-0000.pt",
            }
            identity = {"run_contract": "a" * 64}
            hashes = {}
            for name in names:
                content = f"fixture:{name}".encode("utf-8")
                (checkpoint / name).write_bytes(content)
                hashes[name] = hashlib.sha256(content).hexdigest()
            manifest = {
                "schema_version": self.launcher.CHECKPOINT_SCHEMA,
                "global_step": 250,
                "world_size": 1,
                "identity_hashes": identity,
                "files": sorted(names),
                "sha256": hashes,
            }
            (checkpoint / "checkpoint_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            observed = self.launcher._validate_checkpoint(
                checkpoint,
                expected_step=250,
                expected_identity_hashes=identity,
            )
            self.assertEqual(observed["global_step"], 250)
            (checkpoint / "model.pt").write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "file/hash"):
                self.launcher._validate_checkpoint(
                    checkpoint,
                    expected_step=250,
                    expected_identity_hashes=identity,
                )

    def test_verified_symlink_dataset_view_passes_identity_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            dataset_id = "fixture-dataset"
            contents = {
                "dataset_manifest.json": json.dumps(
                    {"status": "SUCCEEDED", "dataset_id": dataset_id}
                ).encode("utf-8"),
                "sample_manifest.csv": b"sample_id\nfixture\n",
                "subject_split.csv": b"subject_id,split\n1,train\n",
                "SUCCEEDED": b"ok\n",
            }
            files = []
            declared = {"dataset_id": dataset_id}
            authority = {"dataset_id": dataset_id}
            key_by_name = {
                "dataset_manifest.json": "dataset_manifest_sha256",
                "sample_manifest.csv": "sample_manifest_sha256",
                "subject_split.csv": "subject_split_sha256",
                "SUCCEEDED": "succeeded_sha256",
            }
            for index, (name, content) in enumerate(contents.items()):
                payload_name = f"payload-{index}.blob"
                (bundle / payload_name).write_bytes(content)
                digest = hashlib.sha256(content).hexdigest()
                files.append(
                    {
                        "destination": name,
                        "mounted_path": payload_name,
                        "storage": "mounted",
                        "size_bytes": len(content),
                        "sha256": digest,
                    }
                )
                declared[key_by_name[name]] = digest
                authority[key_by_name[name]] = digest
            inventory = {
                "schema": self.launcher.DATA_INVENTORY_SCHEMA,
                "file_count": len(files),
                "packed_file_count": 0,
                "direct_mounted_file_count": len(files),
                "files": files,
            }
            inventory_path = bundle / "inventory.json"
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
            declared["inventory"] = {
                "path": inventory_path.name,
                "size_bytes": inventory_path.stat().st_size,
                "sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            }
            view = self.launcher._materialize_dataset_view(
                bundle,
                declared,
                root / "view",
                root / "packed",
                label="fixture",
            )
            self.assertTrue((view / "dataset_manifest.json").is_symlink())
            self.launcher._validate_dataset_identity(
                view,
                declared,
                authority,
                label="fixture",
            )

    def test_stop_selection_never_repeats_a_completed_training_stage(self) -> None:
        expected = {
            0: 250,
            2: 250,
            250: 1500,
            1499: 1500,
            1500: 2750,
            2750: 4000,
            4000: 0,
        }
        for current, target in expected.items():
            with self.subTest(current=current):
                self.assertEqual(self.launcher._select_stop_step(current), target)

    def test_free_running_progress_binds_chunks_identity_and_anchor_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / self.launcher.RUN_NAME
            chunk = run_dir / "free_running/chunks/rank00000/chunk000000/manifest.json"
            chunk.parent.mkdir(parents=True)
            chunk.write_text('{"fixture": true}\n', encoding="utf-8")
            identity = {"selected_checkpoint_model_sha256": "b" * 64}
            chunks = [
                {
                    "rank": 0,
                    "chunk_index": 0,
                    "anchors": 100,
                    "manifest_path": "chunks/rank00000/chunk000000/manifest.json",
                    "manifest_sha256": hashlib.sha256(chunk.read_bytes()).hexdigest(),
                }
            ]
            progress = {
                "schema_version": (
                    "trauma_predict.multires_event_v2_free_running_hosted_progress.v1"
                ),
                "status": "INCOMPLETE",
                "completed": 100,
                "expected": 6309,
                "completed_anchors": 100,
                "expected_anchors": 6309,
                "new_anchors": 100,
                "identity": identity,
                "identity_sha256": self.launcher._sha256_payload(identity),
                "chunk_manifests": chunks,
                "chunk_manifest_set_sha256": self.launcher._sha256_payload(chunks),
            }
            progress_path = run_dir / "free_running/hosted_progress.json"
            progress_path.write_text(json.dumps(progress), encoding="utf-8")
            observed = self.launcher._validate_free_running_progress(run_dir)
            self.assertEqual(observed["completed_anchors"], 100)
            progress["completed_anchors"] = 99
            progress_path.write_text(json.dumps(progress), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "progress contract"):
                self.launcher._validate_free_running_progress(run_dir)


if __name__ == "__main__":
    unittest.main()
