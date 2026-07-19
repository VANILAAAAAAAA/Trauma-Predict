from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from trauma_predict.data.grud_h1_v2 import H1ChannelRegistry
from trauma_predict.data.multires_event import RobustNormalizer, SupervisionContract


EXPECTED_H1_MANIFEST_SHA256 = (
    "6762897d5f516dc3442a7a206bc3bf19c3e43e32a2444f2807a475d3db61412b"
)
EXPECTED_H1_FINGERPRINT = (
    "96b77af36c2929860ce10f3cb11ca05867d988524141abf6b0cc9d7fdb9f99aa"
)
SEED = 20260713


@dataclass(frozen=True)
class _InputOnlyLayout:
    active_direct_indices: tuple[int, ...] = ()
    derived_primary_f24_indices: tuple[int, ...] = ()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_rows(manifest_path: Path) -> list[dict[str, str]]:
    by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") == "train":
                by_subject[str(row["subject_id"])].append(dict(row))
    if not by_subject:
        raise ValueError("H1 manifest contains no train subjects")
    selected: list[dict[str, str]] = []
    for subject_id, rows in sorted(by_subject.items()):
        ordered = sorted(rows, key=lambda row: (int(row["prediction_hour"]), row["sample_id"]))
        digest = hashlib.sha256(f"{SEED}:{subject_id}".encode("utf-8")).digest()
        selected.append(ordered[int.from_bytes(digest[:8], "big") % len(ordered)])
    return selected


def _iter_fit_records(h1_root: Path, selected: Iterable[Mapping[str, str]]) -> Iterable[dict[str, Any]]:
    by_shard: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in selected:
        by_shard[str(row["h1_shard_key"])].append(row)
    for shard_key, rows in sorted(by_shard.items()):
        split = str(rows[0]["split"])
        path = h1_root / "h1_shards" / split / f"{shard_key}.jsonl.gz"
        requested = {int(row["h1_line_index"]): row for row in rows}
        found: set[int] = set()
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                authority = requested.get(line_index)
                if authority is None:
                    continue
                record = json.loads(line)
                if str(record.get("sample_id")) != str(authority["sample_id"]):
                    raise ValueError("normalization H1 shard identity mismatch")
                geometry = record["input_geometry"]
                block_count = int(geometry["block_count"])
                yield {
                    "split": "train",
                    "subject_id": str(record["subject_id"]),
                    "block_table": [
                        {"block_id": index, "resolution": "H1"}
                        for index in range(block_count)
                    ],
                    "input_events": record["input_events"],
                    "target_events": [],
                    "target_mask": [],
                    "static": record.get("static") or {},
                }
                found.add(line_index)
        if found != set(requested):
            raise ValueError(f"normalization selection is incomplete in shard {shard_key}")


def fit_normalization(
    *,
    h1_root: Path,
    supervision_path: Path,
    output: Path,
) -> None:
    h1_root = h1_root.resolve()
    manifest_path = h1_root / "sample_manifest.csv"
    if sha256_file(manifest_path) != EXPECTED_H1_MANIFEST_SHA256:
        raise ValueError("H1 sample manifest differs from the frozen authority")
    dataset_manifest = json.loads((h1_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    if dataset_manifest.get("fingerprint") != EXPECTED_H1_FINGERPRINT:
        raise ValueError("H1 dataset fingerprint differs from the frozen authority")
    registry = H1ChannelRegistry.from_json(h1_root / "h1_event_templates.json")
    supervision = SupervisionContract.from_json(supervision_path)
    selected = _selected_rows(manifest_path)
    normalizer = RobustNormalizer.fit(
        _iter_fit_records(h1_root, selected),
        templates=registry.templates,
        target_layout=_InputOnlyLayout(),
        supervision=supervision,
        dataset_fingerprint=EXPECTED_H1_FINGERPRINT,
        clip_value=10.0,
        epsilon=1e-6,
        max_values_per_key=200_000,
        seed=SEED,
    )
    if normalizer.subject_count != len(selected):
        raise AssertionError("normalization subject count changed during fitting")
    normalizer.save_json(output)
    print(
        "GRUD_V2_NORMALIZATION_OK "
        f"subjects={normalizer.subject_count} event_stats={len(normalizer.event_stats)} "
        f"fallbacks={len(normalizer.fallback_event_keys)} sha256={sha256_file(output)}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1-root", type=Path, required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fit_normalization(
        h1_root=args.h1_root,
        supervision_path=args.supervision.resolve(),
        output=args.output.resolve(),
    )


if __name__ == "__main__":
    main()
