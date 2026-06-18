#!/usr/bin/env python3
"""Extract one HADM's raw MIMIC-IV rows for UW G1-G4 field discussion.

This script intentionally does NOT build hourly/daily summaries.
It only filters official MIMIC-IV raw tables to UW-like relevant fields and writes raw tables.

Default behavior:
  - if --hadm-id is omitted, use the first HADM in the C4 trauma cohort CSV;
  - write one folder: sample_MtoU_raw/hadm_<hadm_id>/

Later batch extension:
  - loop over hadm_ids and call extract_one_hadm(...);
  - keep the same per-HADM folder contract or concatenate tables downstream.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def gz_csv(path: str):
    return gzip.open(path, "rt", newline="", encoding="utf-8", errors="replace")


def read_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def flatten_itemid_map(section: dict, group_name: str) -> Dict[str, Tuple[str, str]]:
    """Return itemid string -> (group_name, canonical_variable)."""
    out = {}
    for canonical, ids in section.items():
        for itemid in ids:
            out[str(itemid)] = (group_name, canonical)
    return out


def build_item_maps(cfg: dict) -> dict:
    itemids = cfg["itemids"]
    maps = {}
    for table_section, section in itemids.items():
        # table_section example: chartevents_g1_vitals
        table = table_section.split("_")[0]
        maps.setdefault(table, {})
        maps[table].update(flatten_itemid_map(section, table_section))
    return maps


def load_cohort_rows(cohort_csv: str, hadm_id: str | None) -> Tuple[str, List[dict]]:
    rows = []
    first_hadm = None
    with open(cohort_csv, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            if first_hadm is None:
                first_hadm = row["hadm_id"]
            if hadm_id is None:
                hadm_id = first_hadm
            if row["hadm_id"] == str(hadm_id):
                rows.append(row)
    if not rows:
        raise SystemExit(f"HADM not found in cohort: {hadm_id}")
    return str(hadm_id), rows


def write_rows(path: str, rows: List[dict], extra_cols: List[str] | None = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    extra_cols = extra_cols or []
    cols = []
    for c in extra_cols:
        if c not in cols:
            cols.append(c)
    for row in rows:
        for c in row.keys():
            if c not in cols:
                cols.append(c)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if cols:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow(row)
        else:
            f.write("")


def filter_table_by_keys(
    source_path: str,
    out_path: str,
    key_filter: dict,
    item_map: Dict[str, Tuple[str, str]] | None = None,
    item_col: str = "itemid",
    regex_filter: re.Pattern | None = None,
    regex_cols: List[str] | None = None,
) -> dict:
    """Fast raw-row filter using csv.reader instead of DictReader.

    The output keeps original source columns and only prepends
    `uw_group,canonical_variable` when itemid/regex filtering is used.
    No aggregation or summary is performed.
    """
    rows_out = []
    scanned = key_matched = field_matched = 0
    with gz_csv(source_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {c: i for i, c in enumerate(header)}
        key_idx = []
        for col, allowed in key_filter.items():
            if allowed is None:
                continue
            if col not in idx:
                raise KeyError(f"Column {col!r} not found in {source_path}")
            key_idx.append((idx[col], allowed))
        item_idx = idx.get(item_col)
        regex_idx = [idx[c] for c in (regex_cols or []) if c in idx]

        for raw in reader:
            scanned += 1
            ok = True
            for i, allowed in key_idx:
                if i >= len(raw) or raw[i] not in allowed:
                    ok = False
                    break
            if not ok:
                continue
            key_matched += 1

            prefix = ["", ""]
            if item_map is not None:
                itemid = raw[item_idx] if item_idx is not None and item_idx < len(raw) else ""
                if itemid not in item_map:
                    continue
                group, canonical = item_map[itemid]
                prefix = [group, canonical]
                field_matched += 1
            elif regex_filter is not None:
                text = " ".join(raw[i] if i < len(raw) else "" for i in regex_idx)
                if not regex_filter.search(text):
                    continue
                prefix = ["g2_or_g3_medication_raw", "antibiotic_or_antimicrobial"]
                field_matched += 1
            else:
                field_matched += 1
            rows_out.append(prefix + raw)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["uw_group", "canonical_variable"] + header)
        w.writerows(rows_out)
    return {"source": source_path, "output": out_path, "rows_scanned": scanned, "key_matched_rows": key_matched, "field_matched_rows": field_matched, "written_rows": len(rows_out)}


def make_field_map_rows(cfg: dict) -> List[dict]:
    rows = []
    for section, canonical_map in cfg["itemids"].items():
        source_table = section.split("_")[0]
        for canonical, ids in canonical_map.items():
            rows.append({
                "uw_group": section,
                "canonical_variable": canonical,
                "source_table": source_table,
                "source_itemids": ";".join(str(x) for x in ids),
                "extraction_rule": "raw rows only; no aggregation; no hourly/daily summary",
            })
    for name, pattern in cfg.get("prescription_regex", {}).items():
        rows.append({
            "uw_group": "g2_or_g3_medication_raw",
            "canonical_variable": name,
            "source_table": "prescriptions",
            "source_itemids": "drug regex",
            "extraction_rule": pattern,
        })
    return rows


def extract_one_hadm(cfg: dict, hadm_id: str | None):
    cohort_csv = cfg["cohort_csv"]
    mimic_root = cfg["mimic_root"]
    output_root = cfg["output_root"]
    hadm_id, cohort_rows = load_cohort_rows(cohort_csv, hadm_id)
    subject_ids = {r["subject_id"] for r in cohort_rows}
    stay_ids = {r["stay_id"] for r in cohort_rows}
    out_dir = os.path.join(output_root, f"hadm_{hadm_id}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    write_rows(os.path.join(out_dir, "00_cohort_rows.csv"), cohort_rows)
    write_rows(os.path.join(out_dir, "00_field_map.csv"), make_field_map_rows(cfg))

    item_maps = build_item_maps(cfg)
    report = {
        "hadm_id": hadm_id,
        "subject_ids": sorted(subject_ids),
        "stay_ids": sorted(stay_ids),
        "output_dir": out_dir,
        "cohort_rows": len(cohort_rows),
        "tables": {},
        "note": "Raw filtered rows only. No hourly/daily summary or aggregation performed.",
    }

    def src(table_key: str) -> str:
        return os.path.join(mimic_root, cfg["tables"][table_key])

    # G2/source context raw rows.
    report["tables"]["admissions"] = filter_table_by_keys(src("admissions"), os.path.join(out_dir, "01_admissions_raw.csv"), {"hadm_id": {hadm_id}})
    report["tables"]["patients"] = filter_table_by_keys(src("patients"), os.path.join(out_dir, "02_patients_raw.csv"), {"subject_id": subject_ids})
    report["tables"]["icustays"] = filter_table_by_keys(src("icustays"), os.path.join(out_dir, "03_icustays_raw.csv"), {"stay_id": stay_ids})
    report["tables"]["diagnoses_icd"] = filter_table_by_keys(src("diagnoses_icd"), os.path.join(out_dir, "04_diagnoses_icd_raw.csv"), {"hadm_id": {hadm_id}})

    # UW G1/G3/G4 raw event rows.
    report["tables"]["chartevents"] = filter_table_by_keys(src("chartevents"), os.path.join(out_dir, "10_chartevents_G1_G3_raw.csv"), {"stay_id": stay_ids}, item_map=item_maps.get("chartevents", {}))
    report["tables"]["labevents"] = filter_table_by_keys(src("labevents"), os.path.join(out_dir, "20_labevents_G4_raw.csv"), {"hadm_id": {hadm_id}}, item_map=item_maps.get("labevents", {}))
    report["tables"]["inputevents"] = filter_table_by_keys(src("inputevents"), os.path.join(out_dir, "30_inputevents_G3_raw.csv"), {"stay_id": stay_ids}, item_map=item_maps.get("inputevents", {}))
    report["tables"]["outputevents"] = filter_table_by_keys(src("outputevents"), os.path.join(out_dir, "40_outputevents_G4_uop_raw.csv"), {"stay_id": stay_ids}, item_map=item_maps.get("outputevents", {}))
    report["tables"]["procedureevents"] = filter_table_by_keys(src("procedureevents"), os.path.join(out_dir, "50_procedureevents_G3_raw.csv"), {"stay_id": stay_ids}, item_map=item_maps.get("procedureevents", {}))

    # Medication orders useful for abx48-like raw evidence; not summarized.
    abx_pattern = cfg.get("prescription_regex", {}).get("antibiotic_or_antimicrobial")
    if abx_pattern:
        report["tables"]["prescriptions_abx"] = filter_table_by_keys(
            src("prescriptions"),
            os.path.join(out_dir, "60_prescriptions_antibiotics_raw.csv"),
            {"hadm_id": {hadm_id}},
            regex_filter=re.compile(abx_pattern, re.I),
            regex_cols=["drug", "formulary_drug_cd", "prod_strength", "route"],
        )

    report_path = os.path.join(out_dir, "manifest.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"hadm_id": hadm_id, "output_dir": out_dir, "manifest": report_path, "tables": {k: v["written_rows"] for k, v in report["tables"].items()}}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/mnt/d/Data/Trauma cohort MIMICIV/pipeline/02_sample_mtou_raw/sample_MtoU_raw_config.json")
    ap.add_argument("--hadm-id", default=None, help="Default: first HADM in the C4 trauma cohort CSV")
    args = ap.parse_args()
    cfg = read_config(args.config)
    extract_one_hadm(cfg, args.hadm_id)


if __name__ == "__main__":
    main()
