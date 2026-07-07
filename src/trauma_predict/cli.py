from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="trauma-predict")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check-index", help="Validate docs/FILE_INDEX.md against tracked files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "check-index":
        root = Path(__file__).resolve().parents[2]
        subprocess.run(["python", "tools/update_file_index.py", "--check"], cwd=root, check=True)
