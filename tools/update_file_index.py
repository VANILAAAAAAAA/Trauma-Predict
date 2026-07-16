#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "docs" / "FILE_INDEX.md"
TABLE_ROW = re.compile(r"^\|\s*`?([^`|]+?)`?\s*\|")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate docs/FILE_INDEX.md coverage.")
    parser.add_argument("--check", action="store_true", help="Fail if tracked files are missing or stale in the index.")
    return parser.parse_args()


def repo_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and (ROOT / line.strip()).exists()
    }


def indexed_files() -> set[str]:
    files: set[str] = set()
    for line in INDEX_PATH.read_text(encoding="utf-8").splitlines():
        match = TABLE_ROW.match(line)
        if not match:
            continue
        value = match.group(1).strip()
        if value in {"Path", "---"}:
            continue
        files.add(value)
    return files


def check_index() -> None:
    expected = repo_files()
    observed = indexed_files()
    missing = sorted(expected - observed)
    stale = sorted(observed - expected)
    if missing or stale:
        if missing:
            print("Missing from docs/FILE_INDEX.md:")
            for path in missing:
                print(f"  {path}")
        if stale:
            print("Listed in docs/FILE_INDEX.md but not present:")
            for path in stale:
                print(f"  {path}")
        raise SystemExit(1)
    print(f"file_index=PASS files={len(expected)}")


def main() -> None:
    args = parse_args()
    if args.check:
        check_index()
    else:
        for path in sorted(repo_files()):
            print(path)


if __name__ == "__main__":
    main()
