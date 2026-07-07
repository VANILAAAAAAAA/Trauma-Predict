from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def tracked_or_pending_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return sorted(line for line in result.stdout.splitlines() if line)


class RepoHygieneTest(unittest.TestCase):
    def test_no_agent_artifact_paths(self) -> None:
        forbidden = [
            path
            for path in tracked_or_pending_files()
            if path == "agent-artifact" or path.startswith("agent-artifact/")
        ]
        self.assertEqual(forbidden, [])

    def test_no_data_roots_or_training_artifacts(self) -> None:
        forbidden_prefixes = ("data/", "datasets/", "artifacts/", "outputs/", "runs/", "checkpoints/")
        forbidden_suffixes = (
            ".jsonl",
            ".jsonl.gz",
            ".jsonl.zst",
            ".parquet",
            ".npz",
            ".pt",
            ".pth",
            ".ckpt",
            ".safetensors",
        )
        offenders = [
            path
            for path in tracked_or_pending_files()
            if path.startswith(forbidden_prefixes) or path.endswith(forbidden_suffixes)
        ]
        self.assertEqual(offenders, [])

    def test_file_index_is_current(self) -> None:
        subprocess.run([sys.executable, "tools/update_file_index.py", "--check"], cwd=ROOT, check=True)


if __name__ == "__main__":
    unittest.main()
