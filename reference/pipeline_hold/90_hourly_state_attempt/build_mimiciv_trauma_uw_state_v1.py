#!/usr/bin/env python3
"""Build UW-like hourly state tables for the MIMIC-IV trauma C4 cohort.

Outputs:
  static_profile.csv: one row per hadm_id
  hourly_state.csv: one row per hadm_id + relative_hour + active stay hour
  field_registry.csv: canonical variables, source tables, aggregation and leakage role
  build_report.json: aggregate validation counts

Stdlib-only by design for local WSL environment.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DT_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]


def parse_dt(x: str) -> Optional[datetime]:
    if not x:
        return None
    x = str(x).strip()
    for fmt in DT_FORMATS:
        try:
            return datetime.strptime(x, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(x)
    except Exception:
        return None


def parse_float(x) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def gz_csv(path: str):
    return gzip.open(path, "rt", newline="", encoding="utf-8", errors="replace")


def norm_itemids(cfg: dict, table: str) -> Dict[int, str]:
    out = {}
    for canon, ids in cfg["itemids"].get(table, {}).items():
        for itemid in ids:
            out[int(itemid)] = canon
    return out


def value_for_canonical(canon: str, raw: Optional[float]) -> Optional[float]:
    if raw is None:
        return None
    if canon == "temp_f":
        return (raw - 32.0) * 5.0 / 9.0
    if canon == "fio2":
        if raw > 1.0 and raw <= 100.0:
            return raw / 100.0
        return raw
    if canon == "base_excess_chartevents":
        return raw
    return raw


def canonical_for_chartevent(canon: str) -> str:
    if canon == "temp_f":
        return "temp"
    if canon == "temp_c":
        return "temp"
    if canon == "base_excess_chartevents":
        return "base_excess"
    return canon


def hour_index(t: datetime, first_intime: datetime, max_hours: int) -> Optional[int]:
    h = int(math.floor((t - first_intime).total_seconds() / 3600.0)) + 1
    if 1 <= h <= max_hours:
        return h
    return None


def interval_hour_range(start: datetime, end: datetime, first_intime: datetime, max_hours: int) -> range:
    if end <= start:
        end = start + timedelta(minutes=1)
    h0 = int(math.floor((start - first_intime).total_seconds() / 3600.0)) + 1
    h1 = int(math.floor(((end - timedelta(seconds=1)) - first_intime).total_seconds() / 3600.0)) + 1
    h0 = max(1, h0)
    h1 = min(max_hours, h1)
    if h1 < h0:
        return range(0)
    return range(h0, h1 + 1)


def overlap_hours(hour_start: datetime, hour_end: datetime, start: datetime, end: datetime) -> float:
    s = max(hour_start, start)
    e = min(hour_end, end)
    if e <= s:
        return 0.0
    return (e - s).total_seconds() / 3600.0


def load_cohort(path: str, mode: str, sample_max_hadm: int) -> Tuple[dict, dict, list]:
    by_hadm = defaultdict(list)
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            by_hadm[row["hadm_id"]].append(row)
    hadms = sorted(by_hadm)
    if mode == "sample":
        hadms = hadms[:sample_max_hadm]
    selected = {h: by_hadm[h] for h in hadms}

    hadm_meta = {}
    stay_meta = {}
    for hadm, rows in selected.items():
        first = min(parse_dt(r["intime"]) for r in rows if parse_dt(r["intime"]))
        last = max(parse_dt(r["outtime"]) for r in rows if parse_dt(r["outtime"]))
        r0 = rows[0]
        hadm_meta[hadm] = {
            "subject_id": r0.get("subject_id", ""),
            "hadm_id": hadm,
            "first_icu_intime": first,
            "last_icu_outtime": last,
            "age_at_admit": r0.get("age_at_admit", ""),
            "gender": r0.get("gender", ""),
            "hospital_los_hours": r0.get("hospital_los_hours", ""),
            "hospital_expire_flag": r0.get("hospital_expire_flag", ""),
            "trauma_icd_codes": r0.get("trauma_icd_codes", ""),
            "trauma_icd_versions": r0.get("trauma_icd_versions", ""),
            "trauma_mechanisms": r0.get("trauma_mechanisms", ""),
            "is_vent3_subset": r0.get("is_vent3_subset", ""),
            "n_icu_stays": len(rows),
        }
        for r in rows:
            sid = r["stay_id"]
            stay_meta[sid] = {
                "hadm_id": hadm,
                "subject_id": r.get("subject_id", ""),
                "stay_id": sid,
                "intime": parse_dt(r.get("intime", "")),
                "outtime": parse_dt(r.get("outtime", "")),
                "first_icu_intime": first,
            }
    return hadm_meta, stay_meta, hadms


def init_hour_rows(hadm_meta: dict, stay_meta: dict, max_hours: int) -> Dict[Tuple[str, int], dict]:
    rows = {}
    for sid, sm in stay_meta.items():
        if not sm["intime"] or not sm["outtime"] or not sm["first_icu_intime"]:
            continue
        for h in interval_hour_range(sm["intime"], sm["outtime"], sm["first_icu_intime"], max_hours):
            key = (sm["hadm_id"], h)
            if key not in rows:
                hour_start = sm["first_icu_intime"] + timedelta(hours=h - 1)
                rows[key] = {
                    "subject_id": sm["subject_id"],
                    "hadm_id": sm["hadm_id"],
                    "relative_hour": h,
                    "relative_day": (h - 1) // 24 + 1,
                    "hour_within_day": (h - 1) % 24,
                    "active_stay_id": sid,
                    "hour_start": hour_start.strftime("%Y-%m-%d %H:%M:%S"),
                    "is_icu_hour": 1,
                }
    return rows


def aggregate_numeric_events(events: Dict[Tuple[str, int, str], List[Tuple[datetime, float]]], rows: dict):
    for (hadm, h, var), vals in events.items():
        key = (hadm, h)
        if key not in rows or not vals:
            continue
        vals = sorted(vals, key=lambda x: x[0])
        nums = [v for _, v in vals if v is not None]
        if not nums:
            continue
        row = rows[key]
        row[f"{var}_mean"] = round(sum(nums) / len(nums), 4)
        row[f"{var}_min"] = round(min(nums), 4)
        row[f"{var}_max"] = round(max(nums), 4)
        row[f"{var}_last"] = round(nums[-1], 4)
        row[f"{var}_count"] = len(nums)
        row[f"{var}_observed"] = 1


def scan_chartevents(cfg, mimic_root, stay_meta, rows, max_hours, report):
    path = os.path.join(mimic_root, cfg["tables"]["chartevents"])
    item_to_var = norm_itemids(cfg, "chartevents")
    stay_ids = set(stay_meta)
    events = defaultdict(list)
    total = matched = used = 0
    with gz_csv(path) as f:
        r = csv.DictReader(f)
        for row in r:
            total += 1
            sid = row.get("stay_id")
            if sid not in stay_ids:
                continue
            itemid = int(row.get("itemid") or 0)
            if itemid not in item_to_var:
                continue
            matched += 1
            sm = stay_meta[sid]
            t = parse_dt(row.get("charttime", ""))
            if not t or not sm["intime"] or not (sm["intime"] <= t < sm["outtime"]):
                continue
            h = hour_index(t, sm["first_icu_intime"], max_hours)
            if h is None:
                continue
            canon0 = item_to_var[itemid]
            canon = canonical_for_chartevent(canon0)
            if canon == "vent_mode_observed":
                rows[(sm["hadm_id"], h)]["vent_mode_observed"] = 1
                used += 1
                continue
            raw = parse_float(row.get("valuenum") or row.get("value"))
            val = value_for_canonical(canon0, raw)
            if val is None:
                continue
            events[(sm["hadm_id"], h, canon)].append((t, val))
            used += 1
    aggregate_numeric_events(events, rows)
    report["chartevents"] = {"rows_scanned": total, "matched_item_rows": matched, "used_rows": used}


def scan_labevents(cfg, mimic_root, hadm_meta, rows, max_hours, report):
    path = os.path.join(mimic_root, cfg["tables"]["labevents"])
    item_to_var = norm_itemids(cfg, "labevents")
    hadms = set(hadm_meta)
    per_var_events = defaultdict(list)  # (hadm,var) -> [(h,t,val)]
    total = matched = used = 0
    with gz_csv(path) as f:
        r = csv.DictReader(f)
        for row in r:
            total += 1
            hadm = row.get("hadm_id")
            if hadm not in hadms:
                continue
            itemid = int(row.get("itemid") or 0)
            if itemid not in item_to_var:
                continue
            matched += 1
            hm = hadm_meta[hadm]
            t = parse_dt(row.get("storetime") or row.get("charttime") or "")
            if not t or not (hm["first_icu_intime"] <= t < hm["last_icu_outtime"]):
                continue
            h = hour_index(t, hm["first_icu_intime"], max_hours)
            if h is None:
                continue
            val = parse_float(row.get("valuenum") or row.get("value"))
            if val is None:
                continue
            var = item_to_var[itemid]
            per_var_events[(hadm, var)].append((h, t, val))
            used += 1
    # latest memory features per hour
    for (hadm, var), evs in per_var_events.items():
        evs.sort(key=lambda x: (x[0], x[1]))
        by_hour = defaultdict(list)
        for h, t, v in evs:
            by_hour[h].append((t, v))
        last_val = None
        last_h = None
        max_h_for_hadm = max([h for (hh, h) in rows if hh == hadm], default=0)
        for h in range(1, max_h_for_hadm + 1):
            key = (hadm, h)
            if key not in rows:
                continue
            obs = by_hour.get(h, [])
            if obs:
                obs.sort(key=lambda x: x[0])
                nums = [v for _, v in obs]
                last_val = nums[-1]
                last_h = h
                rows[key][f"{var}_observed"] = 1
                rows[key][f"{var}_last"] = round(last_val, 4)
                rows[key][f"{var}_mean_this_hour"] = round(sum(nums) / len(nums), 4)
                rows[key][f"{var}_count_this_hour"] = len(nums)
            else:
                rows[key][f"{var}_observed"] = 0
                if last_val is not None:
                    rows[key][f"{var}_last"] = round(last_val, 4)
            if last_h is None:
                rows[key][f"{var}_hours_since_last"] = "never"
            else:
                rows[key][f"{var}_hours_since_last"] = h - last_h
    report["labevents"] = {"rows_scanned": total, "matched_item_rows": matched, "used_rows": used}


def add_amount(rows, hadm, h, var, amount):
    key = (hadm, h)
    if key not in rows:
        return
    rows[key][var] = round(float(rows[key].get(var, 0) or 0) + amount, 4)
    rows[key][var.replace("_delta_", "_observed_").replace("_ml", "")] = 1


def scan_inputevents(cfg, mimic_root, stay_meta, rows, max_hours, report):
    path = os.path.join(mimic_root, cfg["tables"]["inputevents"])
    item_to_var = norm_itemids(cfg, "inputevents")
    stay_ids = set(stay_meta)
    total = matched = used = 0
    map_var = {"rbc_ml": "rbc_delta_ml", "crystalloid_ml": "crystalloid_delta_ml", "antibiotic_dose": "antibiotic_delta_count"}
    with gz_csv(path) as f:
        r = csv.DictReader(f)
        for row in r:
            total += 1
            sid = row.get("stay_id")
            if sid not in stay_ids:
                continue
            itemid = int(row.get("itemid") or 0)
            if itemid not in item_to_var:
                continue
            matched += 1
            sm = stay_meta[sid]
            start = parse_dt(row.get("starttime", ""))
            end = parse_dt(row.get("endtime", "")) or start
            if not start or not sm["intime"]:
                continue
            start = max(start, sm["intime"])
            end = min(end or start, sm["outtime"])
            if end <= start:
                end = start + timedelta(minutes=1)
            var0 = item_to_var[itemid]
            var = map_var[var0]
            amount = 1.0 if var0 == "antibiotic_dose" else parse_float(row.get("amount"))
            if amount is None:
                continue
            total_hours = max((end - start).total_seconds() / 3600.0, 1/60)
            for h in interval_hour_range(start, end, sm["first_icu_intime"], max_hours):
                hs = sm["first_icu_intime"] + timedelta(hours=h - 1)
                he = hs + timedelta(hours=1)
                frac = overlap_hours(hs, he, start, end) / total_hours
                if frac > 0:
                    add_amount(rows, sm["hadm_id"], h, var, amount * frac)
                    used += 1
    # cumulative channels
    for hadm in sorted({v["hadm_id"] for v in stay_meta.values()}):
        cum = defaultdict(float)
        for h in sorted([h for (hh, h) in rows if hh == hadm]):
            row = rows[(hadm, h)]
            for delta, cvar in [("rbc_delta_ml", "rbc_cum_ml"), ("crystalloid_delta_ml", "crystalloid_cum_ml"), ("antibiotic_delta_count", "antibiotic_cum_count")]:
                cum[delta] += float(row.get(delta, 0) or 0)
                row[cvar] = round(cum[delta], 4)
    report["inputevents"] = {"rows_scanned": total, "matched_item_rows": matched, "used_hour_allocations": used}


def scan_outputevents(cfg, mimic_root, stay_meta, rows, max_hours, report):
    path = os.path.join(mimic_root, cfg["tables"]["outputevents"])
    item_to_var = norm_itemids(cfg, "outputevents")
    stay_ids = set(stay_meta)
    total = matched = used = 0
    with gz_csv(path) as f:
        r = csv.DictReader(f)
        for row in r:
            total += 1
            sid = row.get("stay_id")
            if sid not in stay_ids:
                continue
            itemid = int(row.get("itemid") or 0)
            if itemid not in item_to_var:
                continue
            matched += 1
            sm = stay_meta[sid]
            t = parse_dt(row.get("charttime", ""))
            if not t or not (sm["intime"] <= t < sm["outtime"]):
                continue
            h = hour_index(t, sm["first_icu_intime"], max_hours)
            val = parse_float(row.get("value"))
            if h is None or val is None:
                continue
            add_amount(rows, sm["hadm_id"], h, "uop_delta_ml", val)
            used += 1
    for hadm in sorted({v["hadm_id"] for v in stay_meta.values()}):
        cum = 0.0
        for h in sorted([h for (hh, h) in rows if hh == hadm]):
            row = rows[(hadm, h)]
            cum += float(row.get("uop_delta_ml", 0) or 0)
            row["uop_cum_ml"] = round(cum, 4)
    report["outputevents"] = {"rows_scanned": total, "matched_item_rows": matched, "used_rows": used}


def scan_procedureevents(cfg, mimic_root, stay_meta, rows, max_hours, report):
    path = os.path.join(mimic_root, cfg["tables"]["procedureevents"])
    item_to_var = norm_itemids(cfg, "procedureevents")
    stay_ids = set(stay_meta)
    total = matched = used = 0
    or_events = defaultdict(list)
    with gz_csv(path) as f:
        r = csv.DictReader(f)
        for row in r:
            total += 1
            sid = row.get("stay_id")
            if sid not in stay_ids:
                continue
            itemid = int(row.get("itemid") or 0)
            if itemid not in item_to_var:
                continue
            matched += 1
            sm = stay_meta[sid]
            var = item_to_var[itemid]
            start = parse_dt(row.get("starttime") or row.get("charttime") or "")
            end = parse_dt(row.get("endtime") or "") or start
            if not start:
                continue
            if var in ("or_sent", "or_received"):
                or_events[sid].append((start, var))
                continue
            if var in ("vent_invasive", "vent_noninvasive"):
                start = max(start, sm["intime"])
                end = min(end or start, sm["outtime"])
                if end <= start:
                    end = start + timedelta(hours=1)
                status_col = "vent_invasive_status" if var == "vent_invasive" else "vent_noninvasive_status"
                for h in interval_hour_range(start, end, sm["first_icu_intime"], max_hours):
                    key = (sm["hadm_id"], h)
                    if key in rows:
                        rows[key][status_col] = 1
                        used += 1
    # pair OR Sent -> OR Received within each stay
    for sid, evs in or_events.items():
        sm = stay_meta[sid]
        evs.sort()
        start = None
        for t, var in evs:
            if var == "or_sent":
                start = t
            elif var == "or_received" and start is not None and t > start:
                s = max(start, sm["intime"])
                e = min(t, sm["outtime"])
                for h in interval_hour_range(s, e, sm["first_icu_intime"], max_hours):
                    key = (sm["hadm_id"], h)
                    if key in rows:
                        rows[key]["in_surgery"] = 1
                        used += 1
                start = None
    for hadm in sorted({v["hadm_id"] for v in stay_meta.values()}):
        vent_cum = surg_cum = 0.0
        surg_days = set()
        for h in sorted([h for (hh, h) in rows if hh == hadm]):
            row = rows[(hadm, h)]
            if row.get("vent_invasive_status") == 1:
                vent_cum += 1.0
            if row.get("in_surgery") == 1:
                surg_cum += 1.0
                surg_days.add(row["relative_day"])
            row["vent_hours_cum"] = round(vent_cum, 2)
            row["ventDaySum"] = round(vent_cum / 24.0, 4)
            row["surgHours"] = round(surg_cum, 2)
            row["surgSum"] = len(surg_days)
    report["procedureevents"] = {"rows_scanned": total, "matched_item_rows": matched, "used_hour_allocations": used}


def scan_diagnoses_for_static(cfg, mimic_root, hadm_meta, report):
    path = os.path.join(mimic_root, cfg["tables"]["diagnoses_icd"])
    hadms = set(hadm_meta)
    total = matched = 0
    # broad skull fracture/intracranial injury ICD-9 + head/neck injury ICD-10 proxy.
    icd9_prefix = ("800", "801", "802", "803", "804", "850", "851", "852", "853", "854")
    icd10_prefix = ("S00", "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09")
    with gz_csv(path) as f:
        r = csv.DictReader(f)
        for row in r:
            total += 1
            hadm = row.get("hadm_id")
            if hadm not in hadms:
                continue
            code = re.sub(r"[^A-Za-z0-9]", "", row.get("icd_code", "")).upper()
            ver = str(row.get("icd_version", ""))
            if (ver == "9" and code.startswith(icd9_prefix)) or (ver == "10" and code.startswith(icd10_prefix)):
                hadm_meta[hadm]["head_injury_icd_proxy"] = 1
                matched += 1
    for hm in hadm_meta.values():
        hm.setdefault("head_injury_icd_proxy", 0)
    report["diagnoses_static"] = {"rows_scanned": total, "head_injury_proxy_matches": matched}


def add_derived_hourly(rows: dict):
    for row in rows.values():
        # Default missing/zero treatment channels.
        for col in ["rbc_delta_ml", "rbc_cum_ml", "crystalloid_delta_ml", "crystalloid_cum_ml", "antibiotic_delta_count", "antibiotic_cum_count", "uop_delta_ml", "uop_cum_ml", "vent_invasive_status", "vent_noninvasive_status", "vent_mode_observed", "in_surgery", "vent_hours_cum", "ventDaySum", "surgHours", "surgSum"]:
            row.setdefault(col, 0)
        # UW-like derived acid/base fields.
        be = parse_float(row.get("base_excess_last"))
        if be is not None:
            row["baseDef_observed_until_hour"] = round(max(0.0, -be), 4)
        na = parse_float(row.get("sodium_last")); k = parse_float(row.get("potassium_last")); cl = parse_float(row.get("chloride_last")); bic = parse_float(row.get("bicarb_last"))
        if None not in (na, k, cl, bic):
            row["StrongIon_proxy"] = round((na + k) - (cl + bic), 4)


def write_csv(path: str, rows: Iterable[dict], preferred_cols: List[str]):
    rows = list(rows)
    cols = []
    for c in preferred_cols:
        if c not in cols:
            cols.append(c)
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def field_registry_rows() -> List[dict]:
    rows = []
    def add(group, name, role, source, unit, agg, missing, leakage, notes=""):
        rows.append({"group": group, "canonical_variable": name, "model_role": role, "source_table": source, "unit": unit, "aggregation_rule": agg, "missingness_rule": missing, "leakage_role": leakage, "notes": notes})
    for v,u in [("hr","bpm"),("sbp","mmHg"),("map","mmHg"),("dbp","mmHg"),("rr","insp/min"),("temp","C"),("fio2","fraction")]:
        add("G1_vitals", v, "hour_input_and_next_state_candidate", "icu.chartevents", u, "hourly mean/min/max/last/count", "observed flag + missing if absent", "input_safe_if_charttime<=t")
    for v,u in [("age_at_admit","years"),("gender","category"),("transfer_indicator","category"),("trauma_mechanisms","category"),("head_injury_icd_proxy","binary")]:
        add("G2_static", v, "static_input_or_stratification", "cohort/admissions/diagnoses", u, "one per hadm", "missing category", "static_or_retrospective_proxy")
    for v,u in [("crystalloid_delta_ml","mL"),("crystalloid_cum_ml","mL"),("rbc_delta_ml","mL"),("rbc_cum_ml","mL"),("vent_invasive_status","binary"),("ventDaySum","days"),("in_surgery","binary"),("surgHours","hours"),("surgSum","days"),("antibiotic_delta_count","count")]:
        add("G3_cumulative_exposure", v, "treatment_delta_or_cumulative", "icu.inputevents/procedureevents", u, "hourly delta + cumulative until hour", "zero if no event observed", "input_safe_if_event_time<=t", "surgSum means days in surgery; OR proxy only")
    for v,u in [("bicarb","mEq/L"),("base_excess","mEq/L"),("baseDef_observed_until_hour","mEq/L"),("lactate","mmol/L"),("bun","mg/dL"),("creatinine","mg/dL"),("wbc","K/uL"),("lymphocytes","% or count"),("neutrophils","% or count"),("uop_delta_ml","mL"),("uop_cum_ml","mL"),("StrongIon_proxy","mEq/L")]:
        add("G4_lab_output_memory", v, "lab_memory_or_output", "hosp.labevents/icu.outputevents", u, "latest observed + observed_this_hour + hours_since_last; output uses delta/cum", "never/recency flag", "input_safe_using_storetime_when_available")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["sample", "all"], default=None)
    ap.add_argument("--sample-max-hadm", type=int, default=None)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    if args.mode:
        cfg["mode"] = args.mode
    if args.sample_max_hadm is not None:
        cfg["sample_max_hadm"] = args.sample_max_hadm
    mimic_root = cfg["mimic_root"]
    out_dir = cfg["output_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    report = {"config": {"mode": cfg["mode"], "sample_max_hadm": cfg.get("sample_max_hadm"), "max_hours": cfg["max_hours"]}}
    hadm_meta, stay_meta, hadms = load_cohort(cfg["cohort_csv"], cfg["mode"], int(cfg.get("sample_max_hadm", 25)))
    rows = init_hour_rows(hadm_meta, stay_meta, int(cfg["max_hours"]))
    report["cohort"] = {"hadm_count": len(hadm_meta), "stay_count": len(stay_meta), "hour_rows_initialized": len(rows)}

    scan_diagnoses_for_static(cfg, mimic_root, hadm_meta, report)
    scan_chartevents(cfg, mimic_root, stay_meta, rows, int(cfg["max_hours"]), report)
    scan_labevents(cfg, mimic_root, hadm_meta, rows, int(cfg["max_hours"]), report)
    scan_inputevents(cfg, mimic_root, stay_meta, rows, int(cfg["max_hours"]), report)
    scan_outputevents(cfg, mimic_root, stay_meta, rows, int(cfg["max_hours"]), report)
    scan_procedureevents(cfg, mimic_root, stay_meta, rows, int(cfg["max_hours"]), report)
    add_derived_hourly(rows)

    static_rows = []
    for hm in hadm_meta.values():
        static_rows.append({k: (v.strftime("%Y-%m-%d %H:%M:%S") if isinstance(v, datetime) else v) for k, v in hm.items()})
    hourly_rows = [rows[k] for k in sorted(rows, key=lambda x: (int(x[0]), x[1]))]

    write_csv(os.path.join(out_dir, "static_profile.csv"), static_rows, ["subject_id","hadm_id","age_at_admit","gender","n_icu_stays","first_icu_intime","last_icu_outtime","hospital_los_hours","hospital_expire_flag","is_vent3_subset","head_injury_icd_proxy","trauma_icd_codes","trauma_icd_versions","trauma_mechanisms"])
    preferred_hourly = ["subject_id","hadm_id","relative_hour","relative_day","hour_within_day","active_stay_id","hour_start","is_icu_hour"]
    write_csv(os.path.join(out_dir, "hourly_state.csv"), hourly_rows, preferred_hourly)
    write_csv(os.path.join(out_dir, "field_registry.csv"), field_registry_rows(), ["group","canonical_variable","model_role","source_table","unit","aggregation_rule","missingness_rule","leakage_role","notes"])
    report["outputs"] = {"output_dir": out_dir, "static_rows": len(static_rows), "hourly_rows": len(hourly_rows), "field_registry_rows": len(field_registry_rows())}
    with open(os.path.join(out_dir, "build_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report["outputs"], ensure_ascii=False))


if __name__ == "__main__":
    main()
