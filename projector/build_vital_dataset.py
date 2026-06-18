#!/usr/bin/env python3
"""Build a small fixed-slot vital dataset for numeric projector smoke tests.

Reads selected stays from sample builder/day sample/day_samples.jsonl and scans
MIMIC-IV chartevents for 7 vital signs. Produces sliding windows:

  vital_values [N, 24, 7]
  vital_mask   [N, 24, 7]
  target_values [N, 7]
  target_mask   [N, 7]

No LOCF. Missing values are represented only by mask=0.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

VITAL_ORDER = ["hr", "sbp", "dbp", "map", "rr", "temp", "fio2"]
VITAL_INDEX = {v: i for i, v in enumerate(VITAL_ORDER)}

# Conservative physiologic range filter for numeric projection smoke tests.
# Values outside these ranges are treated as source-table outliers and set missing.
VALID_RANGE = {
    "hr": (20.0, 250.0),
    "sbp": (40.0, 300.0),
    "dbp": (20.0, 200.0),
    "map": (30.0, 200.0),
    "rr": (4.0, 80.0),
    "temp": (25.0, 45.0),
    "fio2": (0.21, 1.0),
}

CHARTEVENT_MAP = {
    220045: "hr",
    220050: "sbp", 220179: "sbp",
    220051: "dbp", 220180: "dbp",
    220052: "map", 220181: "map",
    220210: "rr",
    223761: "temp_f", 223762: "temp_c", 226329: "temp_c",
    223835: "fio2",
}


def dt(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")


def fnum(text: str | None) -> float | None:
    if text is None or text == "":
        return None
    try:
        v = float(text)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def load_selected(workdir: Path, max_stays: int) -> list[dict[str, Any]]:
    p = workdir / "sample builder/day sample/day_samples.jsonl"
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows[:max_stays]


def load_cohort(workdir: Path, selected: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    wanted = {str(s["stay_id"]) for s in selected}
    cohort = workdir / "data dicision/trauma cohort/cohort/mimiciv_trauma_cohort_los48.csv"
    out: dict[str, dict[str, Any]] = {}
    with cohort.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["stay_id"] in wanted:
                intime = dt(r["intime"])
                out[r["stay_id"]] = {
                    "subject_id": r["subject_id"],
                    "hadm_id": r["hadm_id"],
                    "stay_id": r["stay_id"],
                    "intime": intime,
                    "outtime": dt(r["outtime"]),
                    "total_hours": int((dt(r["outtime"]) - intime).total_seconds() // 3600),
                }
    missing = wanted - set(out)
    if missing:
        raise RuntimeError(f"selected stays missing from cohort: {sorted(missing)}")
    return out


def scan_chartevents(mimic_root: Path, stays: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[int, dict[str, tuple[datetime, float]]]], dict[str, int]]:
    """Return stay_id -> hour_idx -> vital -> (latest_charttime, value)."""
    by_stay = set(stays)
    hourly: dict[str, dict[int, dict[str, tuple[datetime, float]]]] = defaultdict(lambda: defaultdict(dict))
    rejected_by_range = {v: 0 for v in VITAL_ORDER}
    path = mimic_root / "icu/chartevents.csv.gz"
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {c: i for i, c in enumerate(header)}
        for row in reader:
            stay_id = row[idx["stay_id"]]
            if stay_id not in by_stay:
                continue
            try:
                itemid = int(row[idx["itemid"]])
            except Exception:
                continue
            field = CHARTEVENT_MAP.get(itemid)
            if not field:
                continue
            value = fnum(row[idx["valuenum"]])
            if value is None:
                continue
            if field == "temp_f":
                field = "temp"
                value = (value - 32.0) * 5.0 / 9.0
            elif field == "temp_c":
                field = "temp"
            elif field == "fio2" and value > 1:
                value = value / 100.0
            if field not in VITAL_INDEX:
                continue
            lo, hi = VALID_RANGE[field]
            if value < lo or value > hi:
                rejected_by_range[field] += 1
                continue
            charttime = dt(row[idx["charttime"]])
            st = stays[stay_id]
            if charttime < st["intime"] or charttime >= st["outtime"]:
                continue
            hour_idx = int((charttime - st["intime"]).total_seconds() // 3600)
            prev = hourly[stay_id][hour_idx].get(field)
            if prev is None or charttime >= prev[0]:
                hourly[stay_id][hour_idx][field] = (charttime, value)
    return hourly, rejected_by_range


def make_examples(stays: dict[str, dict[str, Any]], hourly: dict[str, dict[int, dict[str, tuple[datetime, float]]]], history_len: int) -> dict[str, Any]:
    values = []
    masks = []
    targets = []
    target_masks = []
    stay_ids = []
    anchor_hours = []
    subject_ids = []

    for stay_id, st in stays.items():
        total_h = st["total_hours"]
        if total_h <= history_len + 1:
            continue
        arr = np.zeros((total_h, len(VITAL_ORDER)), dtype=np.float32)
        m = np.zeros_like(arr)
        for h, fields in hourly.get(stay_id, {}).items():
            if h < 0 or h >= total_h:
                continue
            for f, (_t, v) in fields.items():
                j = VITAL_INDEX[f]
                arr[h, j] = float(v)
                m[h, j] = 1.0
        for anchor in range(history_len - 1, total_h - 1):
            xmask = m[anchor - history_len + 1: anchor + 1]
            ymask = m[anchor + 1]
            # Require some history and at least one measured target.
            if xmask.sum() < 12 or ymask.sum() < 1:
                continue
            values.append(arr[anchor - history_len + 1: anchor + 1])
            masks.append(xmask)
            targets.append(arr[anchor + 1])
            target_masks.append(ymask)
            stay_ids.append(stay_id)
            subject_ids.append(st["subject_id"])
            anchor_hours.append(anchor)

    if not values:
        raise RuntimeError("no examples built")
    return {
        "values": np.stack(values),
        "masks": np.stack(masks),
        "targets": np.stack(targets),
        "target_masks": np.stack(target_masks),
        "stay_ids": np.array(stay_ids),
        "subject_ids": np.array(subject_ids),
        "anchor_hours": np.array(anchor_hours, dtype=np.int32),
    }


def split_by_stay(stay_ids: np.ndarray) -> np.ndarray:
    uniq = sorted(set(stay_ids.tolist()))
    labels = {}
    for i, sid in enumerate(uniq):
        if i < max(1, int(0.67 * len(uniq))):
            labels[sid] = 0  # train
        elif i < max(2, int(0.84 * len(uniq))):
            labels[sid] = 1  # val
        else:
            labels[sid] = 2  # test
    return np.array([labels[sid] for sid in stay_ids], dtype=np.int64)


def compute_stats(values: np.ndarray, masks: np.ndarray, targets: np.ndarray, target_masks: np.ndarray, splits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = splits == 0
    means = np.zeros((len(VITAL_ORDER),), dtype=np.float32)
    stds = np.ones_like(means)
    for j in range(len(VITAL_ORDER)):
        vals = []
        xv = values[train, :, j][masks[train, :, j] > 0]
        yv = targets[train, j][target_masks[train, j] > 0]
        if xv.size:
            vals.append(xv)
        if yv.size:
            vals.append(yv)
        if vals:
            allv = np.concatenate(vals).astype(np.float32)
            means[j] = float(allv.mean())
            s = float(allv.std())
            stds[j] = s if s > 1e-6 else 1.0
    return means, stds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/home/vanila/code/EHR-Predict")
    ap.add_argument("--mimic-root", default="/mnt/d/Data/mimic-iv-2.2")
    ap.add_argument("--history-len", type=int, default=24)
    ap.add_argument("--max-stays", type=int, default=6)
    ap.add_argument("--out-dir", default="projector/artifacts")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    out_dir = workdir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = load_selected(workdir, args.max_stays)
    stays = load_cohort(workdir, selected)
    hourly, rejected_by_range = scan_chartevents(Path(args.mimic_root), stays)
    ex = make_examples(stays, hourly, args.history_len)
    splits = split_by_stay(ex["stay_ids"])
    means, stds = compute_stats(ex["values"], ex["masks"], ex["targets"], ex["target_masks"], splits)

    npz = out_dir / "vital_dataset_sample.npz"
    np.savez_compressed(
        npz,
        vital_values=ex["values"],
        vital_mask=ex["masks"],
        target_values=ex["targets"],
        target_mask=ex["target_masks"],
        split=splits,
        stay_ids=ex["stay_ids"],
        subject_ids=ex["subject_ids"],
        anchor_hours=ex["anchor_hours"],
        vital_order=np.array(VITAL_ORDER),
        mean=means,
        std=stds,
    )

    manifest = {
        "dataset": str(npz),
        "n_examples": int(ex["values"].shape[0]),
        "shape": {
            "vital_values": list(ex["values"].shape),
            "vital_mask": list(ex["masks"].shape),
            "target_values": list(ex["targets"].shape),
            "target_mask": list(ex["target_masks"].shape),
        },
        "vital_order": VITAL_ORDER,
        "split_counts": {"train": int((splits == 0).sum()), "val": int((splits == 1).sum()), "test": int((splits == 2).sum())},
        "input_observed_rate_by_vital": {VITAL_ORDER[j]: float(ex["masks"][:, :, j].mean()) for j in range(len(VITAL_ORDER))},
        "target_observed_rate_by_vital": {VITAL_ORDER[j]: float(ex["target_masks"][:, j].mean()) for j in range(len(VITAL_ORDER))},
        "mean": {VITAL_ORDER[j]: float(means[j]) for j in range(len(VITAL_ORDER))},
        "std": {VITAL_ORDER[j]: float(stds[j]) for j in range(len(VITAL_ORDER))},
        "source": "official MIMIC-IV chartevents, selected C4 trauma stays from day_samples.jsonl",
        "no_locf": True,
        "valid_range_filter": VALID_RANGE,
        "rejected_by_range": rejected_by_range,
    }
    (out_dir / "vital_dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
