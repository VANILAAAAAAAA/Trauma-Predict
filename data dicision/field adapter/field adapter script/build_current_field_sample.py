#!/usr/bin/env python3
"""Build current include-only field sample from one raw MIMIC-IV HADM sample.

Input: data dicision/trauma MIMICIV sample/samples/hadm_<hadm_id>/
Output: Input design/patient sample_processed/hadm_<hadm_id>/

This is a field-adapter demonstration sample, not a final training sample builder.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean

DT_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]

G1_FIELDS = ["hr", "sbp", "dbp", "map", "rr", "temp", "fio2"]
STATIC_FIELDS = ["age", "male", "mechanism_cat", "transfer", "initial_ed_sbp"]
G2STAR_FIELDS = ["base_def_48", "lactate_48", "rbc_48", "crys_48"]
G3_FIELDS = ["bolus_sum_until_h", "rbc_sum_until_h", "vent_h", "vent_day_sum_until_h"]
G4_FIELDS = ["bicarb", "strong_ion", "bun", "creatinine", "wbc", "lymphocytes", "neutrophils", "uop"]
INTERMEDIATE_LABS = ["base_excess", "lactate", "Na", "K", "Cl"]


def parse_dt(x: str | None):
    if not x:
        return None
    x = x.strip()
    if not x:
        return None
    for fmt in DT_FORMATS:
        try:
            return datetime.strptime(x, fmt)
        except ValueError:
            pass
    # fromisoformat handles some remaining variants
    try:
        return datetime.fromisoformat(x)
    except ValueError:
        return None


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for k in row.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def to_float(x):
    try:
        if x is None or x == "":
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def hour_index(ts: datetime, intime: datetime) -> int | None:
    delta = ts - intime
    h = math.floor(delta.total_seconds() / 3600)
    return int(h) if h >= 0 else None


def hours_overlap(start: datetime, end: datetime, hour_start: datetime) -> float:
    hour_end = hour_start + timedelta(hours=1)
    a = max(start, hour_start)
    b = min(end, hour_end)
    return max(0.0, (b - a).total_seconds() / 3600)


def mechanism_to_cat(row: dict):
    mech = (row.get("trauma_mechanisms") or "").upper()
    ttype = (row.get("trauma_types") or "").upper()
    # E-code workbook convention: MV/FALL/etc = blunt; CUT/GUN = penetrating; other/burn = 3
    if any(x in mech for x in ["GUN", "CUT"]):
        return 2, "penetrating_by_trauma_mechanisms"
    if any(x in mech for x in ["MV", "MC", "FALL", "PEDESTRIAN", "BIKE", "TRANSPORT", "STRUCK", "MACHINE", "RAILWAY", "AIR", "SNOW"]):
        return 1, "blunt_by_trauma_mechanisms"
    if "PEN" in ttype:
        return 2, "penetrating_by_trauma_types"
    if ttype:
        return 1 if "MV" in ttype else 3, "fallback_by_trauma_types"
    return None, "missing_mechanism"


def build_static(raw_dir: Path) -> tuple[dict, list[dict]]:
    cohort = read_csv(raw_dir / "00_cohort_rows.csv")[0]
    admissions = read_csv(raw_dir / "01_admissions_raw.csv")
    adm = admissions[0] if admissions else {}
    age = to_float(cohort.get("age_at_admit"))
    male = 1 if (cohort.get("gender") or "").upper() == "M" else 0 if cohort.get("gender") else None
    mech, mech_rule = mechanism_to_cat(cohort)
    loc = " ".join([adm.get("admission_location", ""), adm.get("admission_type", "")]).upper()
    transfer = 1 if "TRANSFER" in loc else 0 if loc else None
    # No ED linkage in this sample; keep missing rather than inventing ED SBP from ICU values.
    initial_ed_sbp = None
    row = {
        "hadm_id": cohort.get("hadm_id"),
        "subject_id": cohort.get("subject_id"),
        "stay_id": cohort.get("stay_id"),
        "age": age,
        "male": male,
        "mechanism_cat": mech,
        "transfer": transfer,
        "initial_ed_sbp": initial_ed_sbp,
    }
    provenance = [
        {"field": "age", "source": "00_cohort_rows.age_at_admit", "status": "observed", "rule": "use cohort age_at_admit"},
        {"field": "male", "source": "00_cohort_rows.gender", "status": "observed", "rule": "M=1, otherwise 0"},
        {"field": "mechanism_cat", "source": "00_cohort_rows.trauma_mechanisms/trauma_types", "status": "derived", "rule": mech_rule},
        {"field": "transfer", "source": "01_admissions_raw.admission_location/admission_type", "status": "derived", "rule": "contains TRANSFER -> 1 else 0"},
        {"field": "initial_ed_sbp", "source": "ED triage/vitalsign", "status": "missing", "rule": "no ED linkage rows for this HADM"},
    ]
    return row, provenance


def build_hourly(raw_dir: Path, max_hours: int):
    icu = read_csv(raw_dir / "03_icustays_raw.csv")[0]
    intime = parse_dt(icu["intime"])
    if intime is None:
        raise ValueError("ICU intime missing")

    # G1 chart events aggregated by hour + variable.
    g1_by_hour = defaultdict(lambda: defaultdict(list))
    chartevents = read_csv(raw_dir / "10_chartevents_G1_G3_raw.csv")
    for r in chartevents:
        var = r.get("canonical_variable")
        if var not in G1_FIELDS:
            continue
        ts = parse_dt(r.get("charttime"))
        v = to_float(r.get("valuenum"))
        if ts is None or v is None:
            continue
        if var == "temp":
            uom = (r.get("valueuom") or "").strip().lower()
            if "f" in uom or "°f" in uom or "fahrenheit" in uom:
                v = (v - 32) * 5 / 9
        if var == "fio2" and v > 1 and v <= 100:
            v = v / 100
        h = hour_index(ts, intime)
        if h is not None and h < max_hours:
            g1_by_hour[h][var].append(v)

    # Input events: assign amount to start hour (sample-level demo; interval splitting later in production builder).
    iv_delta = [0.0] * max_hours
    rbc_delta = [0.0] * max_hours
    for r in read_csv(raw_dir / "30_inputevents_G3_raw.csv"):
        cv = r.get("canonical_variable")
        ts = parse_dt(r.get("starttime"))
        amount = to_float(r.get("amount"))
        if ts is None or amount is None:
            continue
        h = hour_index(ts, intime)
        if h is None or h >= max_hours:
            continue
        if cv == "crystalloid" and (r.get("amountuom") or "").lower() == "ml":
            iv_delta[h] += amount
        elif cv == "rbc" and (r.get("amountuom") or "").lower() == "ml":
            rbc_delta[h] += amount

    # Vent interval.
    vent = [0] * max_hours
    for r in read_csv(raw_dir / "50_procedureevents_G3_raw.csv"):
        if not (r.get("canonical_variable") or "").startswith("vent"):
            continue
        st = parse_dt(r.get("starttime"))
        en = parse_dt(r.get("endtime"))
        if st is None or en is None:
            continue
        for h in range(max_hours):
            hs = intime + timedelta(hours=h)
            if hours_overlap(st, en, hs) > 0:
                vent[h] = 1

    # UOP hourly delta.
    uop = [0.0] * max_hours
    uop_obs = [0] * max_hours
    for r in read_csv(raw_dir / "40_outputevents_G4_uop_raw.csv"):
        ts = parse_dt(r.get("charttime"))
        v = to_float(r.get("value"))
        if ts is None or v is None:
            continue
        h = hour_index(ts, intime)
        if h is not None and h < max_hours:
            uop[h] += v
            uop_obs[h] = 1

    # Labs by canonical variable. Convert lymph/neut percent to absolute K/uL using same-time WBC if needed.
    lab_events = defaultdict(list)
    raw_labs = read_csv(raw_dir / "20_labevents_G4_raw.csv")
    wbc_by_time = {}
    for r in raw_labs:
        if r.get("canonical_variable") == "wbc":
            ts = parse_dt(r.get("charttime"))
            v = to_float(r.get("valuenum"))
            if ts and v is not None:
                wbc_by_time[ts] = v
    for r in raw_labs:
        var = r.get("canonical_variable")
        ts = parse_dt(r.get("charttime"))
        v = to_float(r.get("valuenum"))
        if ts is None or v is None:
            continue
        if var in {"lymphocytes", "neutrophils"} and (r.get("valueuom") or "") == "%":
            wbc = wbc_by_time.get(ts)
            if wbc is not None:
                v = wbc * v / 100.0
            else:
                # Cannot align to absolute count; skip to avoid wrong unit.
                continue
        if var in {"Na", "K", "Cl"}:
            key = var.lower()
        else:
            key = var
        lab_events[key].append((ts, v))

    for k in lab_events:
        lab_events[k].sort(key=lambda x: x[0])

    rows = []
    meta_rows = []
    last_values = {v: None for v in G1_FIELDS + ["bicarb", "bun", "creatinine", "wbc", "lymphocytes", "neutrophils", "na", "k", "cl", "base_excess", "lactate"]}
    last_hours = {k: None for k in last_values}
    lab_idx = {k: 0 for k in lab_events}
    bolus = 0.0
    rbc_cum = 0.0
    vent_days_seen = set()

    for h in range(max_hours):
        row = {"hour_index": h}
        meta = {"hour_index": h}
        # G1
        for var in G1_FIELDS:
            obs_vals = g1_by_hour[h].get(var, [])
            if obs_vals:
                last_values[var] = mean(obs_vals)
                last_hours[var] = h
                obs = 1
            else:
                obs = 0
            row[var] = last_values[var]
            meta[f"{var}_observed"] = obs
            meta[f"{var}_recency_h"] = None if last_hours[var] is None else h - last_hours[var]

        # G3
        bolus += iv_delta[h]
        rbc_cum += rbc_delta[h]
        if vent[h]:
            vent_days_seen.add(h // 24 + 1)
        row["bolus_sum_until_h"] = round(bolus, 3)
        row["rbc_sum_until_h"] = round(rbc_cum, 3)
        row["vent_h"] = vent[h]
        row["vent_day_sum_until_h"] = len(vent_days_seen)

        # Labs: advance events up to current hour end.
        hour_end = intime + timedelta(hours=h + 1)
        observed_this_hour = set()
        for var, events in lab_events.items():
            idx = lab_idx[var]
            while idx < len(events) and events[idx][0] <= hour_end:
                ev_ts, ev_v = events[idx]
                ev_h = hour_index(ev_ts, intime)
                if ev_h is not None and ev_h <= h:
                    last_values[var] = ev_v
                    last_hours[var] = ev_h
                    if ev_h == h:
                        observed_this_hour.add(var)
                idx += 1
            lab_idx[var] = idx

        # Strong ion from current memory values.
        si = None
        if all(last_values.get(k) is not None for k in ["na", "k", "cl", "bicarb"]):
            si = (last_values["na"] + last_values["k"]) - (last_values["cl"] + last_values["bicarb"])

        for var in ["bicarb", "bun", "creatinine", "wbc", "lymphocytes", "neutrophils"]:
            row[var] = last_values[var]
            meta[f"{var}_observed"] = 1 if var in observed_this_hour else 0
            meta[f"{var}_recency_h"] = None if last_hours[var] is None else h - last_hours[var]
        row["strong_ion"] = None if si is None else round(si, 3)
        meta["strong_ion_observed"] = 1 if all(k in observed_this_hour for k in ["na", "k", "cl", "bicarb"]) else 0
        meta["strong_ion_recency_h"] = None if any(last_hours.get(k) is None for k in ["na", "k", "cl", "bicarb"]) else max(h - last_hours[k] for k in ["na", "k", "cl", "bicarb"])
        row["uop"] = round(uop[h], 3)
        meta["uop_observed"] = uop_obs[h]
        rows.append(row)
        meta_rows.append(meta)

    # First 48h summary.
    def vals_in_first48(var):
        out = []
        for ts, v in lab_events.get(var, []):
            h = hour_index(ts, intime)
            if h is not None and 0 <= h < 48:
                out.append(v)
        return out

    be_vals = vals_in_first48("base_excess")
    lac_vals = vals_in_first48("lactate")
    g2star = {
        "base_def_48": max([max(0.0, -v) for v in be_vals], default=None),
        "lactate_48": max(lac_vals) if lac_vals else None,
        "rbc_48": round(sum(rbc_delta[:48]), 3),
        "crys_48": round(sum(iv_delta[:48]), 3),
    }
    return rows, meta_rows, g2star


def build_daily(hourly_rows: list[dict], max_days: int = 13):
    rows = []
    for d in range(1, max_days + 1):
        start = (d - 1) * 24
        end = d * 24
        block = hourly_rows[start:end]
        if len(block) < 24:
            continue
        out = {"day_index": d, "start_hour": start, "end_hour": end - 1}
        for var in G1_FIELDS:
            vals = [to_float(r.get(var)) for r in block if to_float(r.get(var)) is not None]
            out[f"{var}_last"] = vals[-1] if vals else None
            out[f"{var}_min"] = min(vals) if vals else None
            out[f"{var}_max"] = max(vals) if vals else None
            out[f"{var}_mean"] = round(mean(vals), 3) if vals else None
            out[f"{var}_n_obs_memory"] = len(vals)
        out["iv_fluid_total_day"] = round(hourly_rows[end - 1]["bolus_sum_until_h"] - (hourly_rows[start - 1]["bolus_sum_until_h"] if start > 0 else 0), 3)
        out["rbc_total_day"] = round(hourly_rows[end - 1]["rbc_sum_until_h"] - (hourly_rows[start - 1]["rbc_sum_until_h"] if start > 0 else 0), 3)
        out["vent_hours_day"] = sum(int(r["vent_h"]) for r in block)
        out["vent_day_sum_until_block_end"] = hourly_rows[end - 1]["vent_day_sum_until_h"]
        for var in ["bicarb", "strong_ion", "bun", "creatinine", "wbc", "lymphocytes", "neutrophils", "uop"]:
            vals = [to_float(r.get(var)) for r in block if to_float(r.get(var)) is not None]
            out[f"{var}_last"] = vals[-1] if vals else None
            out[f"{var}_min"] = min(vals) if vals else None
            out[f"{var}_max"] = max(vals) if vals else None
        rows.append(out)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-hours", type=int, default=312)
    args = ap.parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    static, provenance = build_static(raw_dir)
    hourly, hourly_meta, g2star = build_hourly(raw_dir, args.max_hours)
    daily = build_daily(hourly, args.max_hours // 24)

    write_csv(out_dir / "00_static_fields.csv", [static], ["hadm_id", "subject_id", "stay_id"] + STATIC_FIELDS)
    write_csv(out_dir / "01_hourly_current_fields_first312h.csv", hourly, ["hour_index"] + G1_FIELDS + G3_FIELDS + G4_FIELDS)
    write_csv(out_dir / "01_hourly_observed_recency_metadata.csv", hourly_meta)
    write_csv(out_dir / "02_first48h_fields.csv", [g2star], G2STAR_FIELDS)
    write_csv(out_dir / "03_daily_summary_preview.csv", daily)
    write_csv(out_dir / "04_field_provenance.csv", provenance)

    manifest = {
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "max_hours": args.max_hours,
        "static_fields": STATIC_FIELDS,
        "hourly_fields": G1_FIELDS + G3_FIELDS + G4_FIELDS,
        "first48h_fields": G2STAR_FIELDS,
        "daily_rows": len(daily),
        "hourly_rows": len(hourly),
        "notes": [
            "Processed include-only field adapter sample for schema discussion.",
            "Not a final training sample builder.",
            "initial_ed_sbp is missing because this HADM has no ED linkage rows in the raw sample.",
            "lymphocytes/neutrophils are converted to K/uL from same-time WBC and differential percentages when direct absolute counts are unavailable.",
            "Inputevents amounts are assigned to start hour for this sample preview; production builder should split intervals by overlap.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "files": sorted(p.name for p in out_dir.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
