#!/usr/bin/env python3
"""Generate representative MIMIC DAY-token samples.

This script is intentionally local/prototype-stage. It reads official MIMIC-IV raw
CSV.gz tables, applies the current tokenizer/day_token.md rules, then selects a
small set of stays that cover the DAY token design points.

Outputs:
  - day_samples.jsonl
  - day_samples.md

No pandas required.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path("/home/vanila/code/EHR-Predict")
COHORT = ROOT / "data dicision/trauma cohort/cohort/mimiciv_trauma_cohort_los48.csv"
MIMIC = Path("/mnt/d/Data/mimic-iv-2.2")
OUT_DIR = ROOT / "sample builder/day sample"
OUT_JSONL = OUT_DIR / "day_samples.jsonl"
OUT_MD = OUT_DIR / "day_samples.md"

RANDOM_SEED = 20260614
POOL_N = 300
MAX_SELECTED = 6
MAX_DAYS_PER_SAMPLE = 13

CORE_VITALS = ["hr", "sbp", "dbp", "map", "rr", "temp"]

CHARTEVENT_MAP = {
    220045: "hr",
    220050: "sbp",
    220179: "sbp",
    220051: "dbp",
    220180: "dbp",
    220052: "map",
    220181: "map",
    220210: "rr",
    223761: "temp_f",
    223762: "temp_c",
    226329: "temp_c",
    223835: "fio2",
}

LABEVENT_MAP = {
    50882: "bicarb",
    50803: "bicarb",
    50804: "bicarb",
    51739: "bicarb",
    50912: "creatinine",
    52546: "creatinine",
    52024: "creatinine",
    51006: "bun",
    52647: "bun",
    51300: "wbc",
    51301: "wbc",
    51755: "wbc",
    51756: "wbc",
    50813: "lactate",
    52442: "lactate",
    53154: "lactate",
    50802: "base_excess",
}

UOP_ITEMIDS = {226559, 226560, 226566, 226627, 226631, 226713, 227489}
RBC_ITEMIDS = {225168, 226368, 227070}
VENT_PROCEDURE_ITEMIDS = {225792}  # Invasive Ventilation

TARGET_TOKENS = {
    "[map_low_hours_bin_prolonged]",
    "[map_low_hours_bin_persistent]",
    "[systolic_bp_min_bin_hypotension]",
    "[heart_rate_max_bin_extreme_tachycardia]",
    "[lactate_48h_bin_severe]",
    "[base_deficit_48h_bin_severe]",
    "[vent_hours_bin_full_window]",
    "[vent_course_bin_prolonged]",
    "[fio2_max_bin_very_high_support]",
    "[respiratory_rate_high_hours_bin_brief]",
    "[creatinine_change_bin_kdigo_delta]",
    "[creatinine_ratio_bin_kdigo_ratio]",
    "[bicarbonate_min_bin_low]",
    "[bun_creatinine_ratio_bin_prerenal_pattern]",
    "[wbc_bin_high]",
    "[wbc_bin_low]",
    "[rbc_transfusion_event_present]",
    "[rbc_48h_event_present]",
    "[core_vital_slots_sparse]",
    "[core_vital_slots_none]",
    "[labs_not_drawn]",
    "[uop_sparse]",
    "[uop_not_measured]",
}


def parse_dt(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")


def to_float(text: str | None) -> float | None:
    if text is None or text == "":
        return None
    try:
        v = float(text)
    except Exception:
        return None
    return v if math.isfinite(v) else None


@dataclass
class Stay:
    sample_id: str
    subject_id: str
    hadm_id: str
    stay_id: str
    intime: datetime
    outtime: datetime
    ndays: int
    age: float
    icu_los_days: float


@dataclass
class DayRaw:
    core_slots: dict[str, set[int]] = field(default_factory=lambda: {k: set() for k in CORE_VITALS})
    values: dict[str, dict[int, list[float]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    labs: dict[str, dict[int, list[float]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    uop_slots: set[int] = field(default_factory=set)
    rbc_ml: float = 0.0
    vent_hours: set[int] = field(default_factory=set)


def load_sample_pool() -> list[Stay]:
    eligible: list[dict[str, Any]] = []
    with open(COHORT, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            los = float(r.get("icu_los_days") or 0)
            if los < 2 or r.get("has_chartevents_data") != "1":
                continue
            intime = parse_dt(r["intime"])
            outtime = parse_dt(r["outtime"])
            ndays = int((outtime - intime).total_seconds() // 86400)
            if ndays < 2:
                continue
            eligible.append(r)

    random.seed(RANDOM_SEED)
    chosen = random.sample(eligible, min(POOL_N, len(eligible)))
    pool: list[Stay] = []
    for i, r in enumerate(chosen, start=1):
        pool.append(
            Stay(
                sample_id=f"pool_{i:03d}",
                subject_id=r["subject_id"],
                hadm_id=r["hadm_id"],
                stay_id=r["stay_id"],
                intime=parse_dt(r["intime"]),
                outtime=parse_dt(r["outtime"]),
                ndays=int((parse_dt(r["outtime"]) - parse_dt(r["intime"])).total_seconds() // 86400),
                age=float(r.get("age_at_admit") or r.get("anchor_age") or 0),
                icu_los_days=float(r.get("icu_los_days") or 0),
            )
        )
    return pool


def day_hour(stay: Stay, t: datetime) -> tuple[int, int] | None:
    if t < stay.intime or t >= stay.outtime:
        return None
    delta = (t - stay.intime).total_seconds()
    d = int(delta // 86400)
    h = int((delta % 86400) // 3600)
    if d < 0 or d >= stay.ndays:
        return None
    return d, h


def initialize_records(pool: list[Stay]) -> dict[tuple[str, int], DayRaw]:
    return {(s.stay_id, d): DayRaw() for s in pool for d in range(s.ndays)}


def scan_chartevents(pool: list[Stay], records: dict[tuple[str, int], DayRaw]) -> int:
    by_stay = {s.stay_id: s for s in pool}
    n = 0
    path = MIMIC / "icu/chartevents.csv.gz"
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {c: i for i, c in enumerate(header)}
        for row in reader:
            itemid = int(row[idx["itemid"]])
            field = CHARTEVENT_MAP.get(itemid)
            if not field:
                continue
            stay = by_stay.get(row[idx["stay_id"]])
            if not stay:
                continue
            value = to_float(row[idx["valuenum"]])
            if value is None:
                continue
            dh = day_hour(stay, parse_dt(row[idx["charttime"]]))
            if dh is None:
                continue
            d, h = dh
            rec = records[(stay.stay_id, d)]
            if field == "temp_f":
                field = "temp"
                value = (value - 32.0) * 5.0 / 9.0
            elif field == "temp_c":
                field = "temp"
            elif field == "fio2" and value > 1:
                value = value / 100.0
            if field in rec.core_slots:
                rec.core_slots[field].add(h)
            rec.values[field][h].append(value)
            n += 1
    return n


def scan_labevents(pool: list[Stay], records: dict[tuple[str, int], DayRaw]) -> int:
    by_hadm: dict[str, list[Stay]] = defaultdict(list)
    for s in pool:
        by_hadm[s.hadm_id].append(s)
    n = 0
    path = MIMIC / "hosp/labevents.csv.gz"
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {c: i for i, c in enumerate(header)}
        for row in reader:
            try:
                itemid = int(row[idx["itemid"]])
            except Exception:
                continue
            field = LABEVENT_MAP.get(itemid)
            if not field:
                continue
            stays = by_hadm.get(row[idx["hadm_id"]])
            if not stays:
                continue
            value = to_float(row[idx["valuenum"]])
            if value is None:
                continue
            t = parse_dt(row[idx["charttime"]])
            for stay in stays:
                dh = day_hour(stay, t)
                if dh is None:
                    continue
                d, h = dh
                records[(stay.stay_id, d)].labs[field][h].append(value)
                n += 1
    return n


def scan_outputevents(pool: list[Stay], records: dict[tuple[str, int], DayRaw]) -> int:
    by_stay = {s.stay_id: s for s in pool}
    n = 0
    path = MIMIC / "icu/outputevents.csv.gz"
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {c: i for i, c in enumerate(header)}
        for row in reader:
            try:
                itemid = int(row[idx["itemid"]])
            except Exception:
                continue
            if itemid not in UOP_ITEMIDS:
                continue
            stay = by_stay.get(row[idx["stay_id"]])
            if not stay:
                continue
            value = to_float(row[idx["value"]])
            if value is None:
                continue
            dh = day_hour(stay, parse_dt(row[idx["charttime"]]))
            if dh is None:
                continue
            d, h = dh
            records[(stay.stay_id, d)].uop_slots.add(h)
            n += 1
    return n


def scan_inputevents_rbc(pool: list[Stay], records: dict[tuple[str, int], DayRaw]) -> int:
    by_stay = {s.stay_id: s for s in pool}
    n = 0
    path = MIMIC / "icu/inputevents.csv.gz"
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {c: i for i, c in enumerate(header)}
        for row in reader:
            try:
                itemid = int(row[idx["itemid"]])
            except Exception:
                continue
            if itemid not in RBC_ITEMIDS:
                continue
            stay = by_stay.get(row[idx["stay_id"]])
            if not stay:
                continue
            amount = to_float(row[idx["amount"]])
            if amount is None or amount <= 0:
                continue
            dh = day_hour(stay, parse_dt(row[idx["starttime"]]))
            if dh is None:
                continue
            d, _h = dh
            records[(stay.stay_id, d)].rbc_ml += amount
            n += 1
    return n


def scan_procedureevents_vent(pool: list[Stay], records: dict[tuple[str, int], DayRaw]) -> int:
    by_stay = {s.stay_id: s for s in pool}
    n = 0
    path = MIMIC / "icu/procedureevents.csv.gz"
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {c: i for i, c in enumerate(header)}
        for row in reader:
            try:
                itemid = int(row[idx["itemid"]])
            except Exception:
                continue
            if itemid not in VENT_PROCEDURE_ITEMIDS:
                continue
            stay = by_stay.get(row[idx["stay_id"]])
            if not stay:
                continue
            start = parse_dt(row[idx["starttime"]])
            end = parse_dt(row[idx["endtime"]])
            interval_start = max(start, stay.intime)
            interval_end = min(end, stay.outtime)
            if interval_end <= interval_start:
                continue
            for d in range(stay.ndays):
                for h in range(24):
                    hs = stay.intime + timedelta(days=d, hours=h)
                    he = hs + timedelta(hours=1)
                    if he > interval_start and hs < interval_end:
                        records[(stay.stay_id, d)].vent_hours.add(h)
                        n += 1
    return n


def flat_values(day: DayRaw, group: str, field: str) -> list[float]:
    src = day.values if group == "values" else day.labs
    return [v for vals in src[field].values() for v in vals]


def first48_summary(stay: Stay, records: dict[tuple[str, int], DayRaw]) -> dict[str, float | None]:
    lactate: list[float] = []
    base_deficit: list[float] = []
    rbc_ml = 0.0
    for d in (0, 1):
        rec = records.get((stay.stay_id, d))
        if rec is None:
            continue
        lactate.extend(flat_values(rec, "labs", "lactate"))
        base_excess = flat_values(rec, "labs", "base_excess")
        base_deficit.extend([max(0.0, -v) for v in base_excess])
        rbc_ml += rec.rbc_ml
    return {
        "lactate48": max(lactate) if lactate else None,
        "base_deficit48": max(base_deficit) if base_deficit else None,
        "rbc48_ml": rbc_ml,
    }


def emit(tokens: list[str], token: str) -> None:
    tokens.append(token)


def build_day_tokens(
    stay: Stay,
    d: int,
    rec: DayRaw,
    prev_creatinine_max: float | None,
    first48: dict[str, float | None],
    vent_day_counter_before: int,
) -> tuple[list[str], dict[str, Any], float | None, int]:
    tokens: list[str] = ["[day_window_len_24h]"]
    domains: list[str] = []

    def domain_marker(domain: str) -> None:
        marker = f"[{domain}]"
        if marker not in tokens:
            tokens.append(marker)
            domains.append(domain)

    # Perfusion / Shock
    perf: list[str] = []
    map_low = sum(1 for _h, vals in rec.values["map"].items() if any(v < 65 for v in vals))
    if 1 <= map_low <= 3:
        perf.append("[map_low_hours_bin_brief]")
    elif 4 <= map_low <= 8:
        perf.append("[map_low_hours_bin_intermittent]")
    elif 9 <= map_low <= 16:
        perf.append("[map_low_hours_bin_prolonged]")
    elif map_low > 16:
        perf.append("[map_low_hours_bin_persistent]")

    sbp = flat_values(rec, "values", "sbp")
    if sbp:
        mn = min(sbp)
        if mn < 90:
            perf.append("[systolic_bp_min_bin_hypotension]")
        elif mn <= 100:
            perf.append("[systolic_bp_min_bin_low]")
        elif stay.age >= 65 and mn <= 109:
            perf.append("[systolic_bp_min_bin_geriatric_low]")
    hr = flat_values(rec, "values", "hr")
    if any(v >= 131 for v in hr):
        perf.append("[heart_rate_max_bin_extreme_tachycardia]")

    if d == 1:
        lactate48 = first48["lactate48"]
        if lactate48 is not None:
            if 2.0 < lactate48 <= 5.0:
                perf.append("[lactate_48h_bin_elevated]")
            elif lactate48 > 5.0:
                perf.append("[lactate_48h_bin_severe]")
        base48 = first48["base_deficit48"]
        if base48 is not None:
            if 3.0 <= base48 < 6.0:
                perf.append("[base_deficit_48h_bin_mild]")
            elif 6.0 <= base48 < 10.0:
                perf.append("[base_deficit_48h_bin_moderate]")
            elif base48 >= 10.0:
                perf.append("[base_deficit_48h_bin_severe]")
    if perf:
        domain_marker("perfusion_shock")
        tokens.extend(perf)

    # Oxygenation / Ventilation
    oxy: list[str] = []
    vent_hours = len(rec.vent_hours)
    is_vent_day = vent_hours > 0
    eligible_hours = 24
    if 0 < vent_hours < 0.5 * eligible_hours:
        oxy.append("[vent_hours_bin_partial_window]")
    elif 0.5 * eligible_hours <= vent_hours < eligible_hours:
        oxy.append("[vent_hours_bin_most_window]")
    elif vent_hours == eligible_hours:
        oxy.append("[vent_hours_bin_full_window]")

    vent_course_index = vent_day_counter_before + 1 if is_vent_day else None
    if vent_course_index is not None:
        if vent_course_index == 1:
            oxy.append("[vent_course_bin_first_day]")
        elif 2 <= vent_course_index <= 3:
            oxy.append("[vent_course_bin_early]")
        elif vent_course_index >= 4:
            oxy.append("[vent_course_bin_prolonged]")

    fio2 = flat_values(rec, "values", "fio2")
    if fio2:
        mx = max(fio2)
        if 0.40 < mx <= 0.60:
            oxy.append("[fio2_max_bin_high_support]")
        elif mx > 0.60:
            oxy.append("[fio2_max_bin_very_high_support]")

    rr_high_hours = sum(1 for _h, vals in rec.values["rr"].items() if any(v >= 25 for v in vals))
    if 1 <= rr_high_hours <= 3:
        oxy.append("[respiratory_rate_high_hours_bin_brief]")
    elif 4 <= rr_high_hours <= 8:
        oxy.append("[respiratory_rate_high_hours_bin_intermediate]")
    elif rr_high_hours >= 9:
        oxy.append("[respiratory_rate_high_hours_bin_prolonged]")
    if oxy:
        domain_marker("oxygenation_ventilation")
        tokens.extend(oxy)

    # Renal / Metabolic
    renal: list[str] = []
    creatinine = flat_values(rec, "labs", "creatinine")
    current_creatinine_max = max(creatinine) if creatinine else prev_creatinine_max
    if creatinine and prev_creatinine_max is not None:
        crmax = max(creatinine)
        if crmax - prev_creatinine_max >= 0.3:
            renal.append("[creatinine_change_bin_kdigo_delta]")
        if prev_creatinine_max > 0 and crmax / prev_creatinine_max >= 1.5:
            renal.append("[creatinine_ratio_bin_kdigo_ratio]")

    bicarb = flat_values(rec, "labs", "bicarb")
    if any(v < 22 for v in bicarb):
        renal.append("[bicarbonate_min_bin_low]")

    bun = flat_values(rec, "labs", "bun")
    if bun and creatinine:
        max_bun = max(bun)
        min_cr = min(v for v in creatinine if v > 0) if any(v > 0 for v in creatinine) else None
        if min_cr is not None and max_bun > 20 and max_bun / min_cr > 20:
            renal.append("[bun_creatinine_ratio_bin_prerenal_pattern]")

    # urine_output_status_kdigo_low is emitted only when a reliable time-aligned weight registry is available; this prototype has no weight channel.
    if renal:
        domain_marker("renal_metabolic")
        tokens.extend(renal)

    # Immune / Hematologic
    immune: list[str] = []
    wbc = flat_values(rec, "labs", "wbc")
    if any(v > 12 for v in wbc):
        immune.append("[wbc_bin_high]")
    if any(v < 4 for v in wbc):
        immune.append("[wbc_bin_low]")
    if rec.rbc_ml > 0:
        immune.append("[rbc_transfusion_event_present]")
    if immune:
        domain_marker("immune_hematologic")
        tokens.extend(immune)

    # Resuscitation Burden
    resus: list[str] = []
    if d == 1 and (first48["rbc48_ml"] or 0) > 0:
        resus.append("[rbc_48h_event_present]")
    # crystalloid_48h not emitted in this MIMIC sample prototype: crystalloid itemid registry is not frozen.
    if resus:
        domain_marker("resuscitation_burden")
        tokens.extend(resus)

    # Data Quality
    dq: list[str] = []
    core_slots = sum(len(rec.core_slots[f]) for f in CORE_VITALS)
    if core_slots >= 120:
        dq.append("[core_vital_slots_dense]")
    elif core_slots >= 72:
        dq.append("[core_vital_slots_partial]")
    elif core_slots >= 1:
        dq.append("[core_vital_slots_sparse]")
    else:
        dq.append("[core_vital_slots_none]")

    lab_draws = sum(len(vals) for f in ["bicarb", "bun", "creatinine", "wbc"] for vals in rec.labs[f].values())
    if lab_draws == 0:
        dq.append("[labs_not_drawn]")

    uop_slots = len(rec.uop_slots)
    if uop_slots >= 6:
        dq.append("[uop_measured]")
    elif uop_slots >= 1:
        dq.append("[uop_sparse]")
    else:
        dq.append("[uop_not_measured]")

    if dq:
        domain_marker("data_quality")
        tokens.extend(dq)

    audit = {
        "map_low_hours": map_low,
        "vent_hours": vent_hours,
        "rr_high_hours": rr_high_hours,
        "core_vital_slots": core_slots,
        "lab_draws": lab_draws,
        "uop_slots": uop_slots,
        "rbc_ml_day": round(rec.rbc_ml, 3),
        "has_weight_for_kdigo_uop": False,
        "domains": domains,
    }
    if lactate48 := first48.get("lactate48"):
        audit["lactate48"] = lactate48
    if base48 := first48.get("base_deficit48"):
        audit["base_deficit48"] = base48
    rbc48_ml = first48.get("rbc48_ml")
    if rbc48_ml is not None and rbc48_ml > 0:
        audit["rbc48_ml"] = round(rbc48_ml, 3)

    if is_vent_day:
        vent_day_counter_before += 1
    return tokens, audit, current_creatinine_max, vent_day_counter_before


def build_all_samples(pool: list[Stay], records: dict[tuple[str, int], DayRaw]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for stay in pool:
        first48 = first48_summary(stay, records)
        prev_cr: float | None = None
        vent_counter = 0
        day_entries: list[dict[str, Any]] = []
        for d in range(stay.ndays):
            rec = records[(stay.stay_id, d)]
            tokens, audit, prev_cr, vent_counter = build_day_tokens(stay, d, rec, prev_cr, first48, vent_counter)
            day_entries.append(
                {
                    "source_day_index": d,
                    "tokens": tokens,
                    "sequence": " ".join(tokens),
                    "audit": audit,
                }
            )
        union_tokens = set(t for day in day_entries for t in day["tokens"] if t.startswith("[") and t not in {"[perfusion_shock]", "[oxygenation_ventilation]", "[renal_metabolic]", "[immune_hematologic]", "[resuscitation_burden]", "[data_quality]"})
        samples.append(
            {
                "sample_id": stay.sample_id,
                "subject_id": stay.subject_id,
                "hadm_id": stay.hadm_id,
                "stay_id": stay.stay_id,
                "icu_los_days": stay.icu_los_days,
                "completed_day_count": stay.ndays,
                "first48_summary": first48,
                "day_entries": day_entries,
                "union_tokens": sorted(union_tokens),
            }
        )
    return samples


def select_representative(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select 1 low-abnormality baseline stay plus 5 high-coverage stays."""
    severe_tokens = {
        "[map_low_hours_bin_prolonged]",
        "[map_low_hours_bin_persistent]",
        "[systolic_bp_min_bin_hypotension]",
        "[heart_rate_max_bin_extreme_tachycardia]",
        "[lactate_48h_bin_severe]",
        "[base_deficit_48h_bin_severe]",
        "[vent_hours_bin_full_window]",
        "[vent_course_bin_prolonged]",
        "[fio2_max_bin_very_high_support]",
        "[creatinine_change_bin_kdigo_delta]",
        "[creatinine_ratio_bin_kdigo_ratio]",
        "[rbc_transfusion_event_present]",
        "[rbc_48h_event_present]",
        "[core_vital_slots_sparse]",
        "[core_vital_slots_none]",
        "[labs_not_drawn]",
        "[uop_not_measured]",
    }

    def baseline_score(s: dict[str, Any]) -> tuple[int, int, int, int]:
        toks = set(s["union_tokens"])
        severe_count = len(toks & severe_tokens)
        target_count = len(toks & TARGET_TOKENS)
        dense_days = sum(1 for d in s["day_entries"] if "[core_vital_slots_dense]" in d["tokens"])
        measured_uop_days = sum(1 for d in s["day_entries"] if "[uop_measured]" in d["tokens"])
        # Lower severe/target burden is better; more dense/uop-measured days is better.
        return (-severe_count, -target_count, dense_days, measured_uop_days)

    baseline = max(samples, key=baseline_score)
    selected: list[dict[str, Any]] = [baseline]
    covered: set[str] = set()
    remaining = [s for s in samples if s is not baseline]
    for _ in range(MAX_SELECTED - 1):
        best = None
        best_score = None
        for s in remaining:
            toks = set(s["union_tokens"])
            gain = len((toks & TARGET_TOKENS) - covered)
            rare_gain = sum(1 for t in toks if t in TARGET_TOKENS and t not in covered)
            token_count = len(toks & TARGET_TOKENS)
            score = (gain, rare_gain, token_count, s["completed_day_count"])
            if best_score is None or score > best_score:
                best = s
                best_score = score
        if best is None:
            break
        selected.append(best)
        covered |= set(best["union_tokens"]) & TARGET_TOKENS
        remaining = [s for s in remaining if s is not best]
    # Rename selected samples for readability.
    for i, s in enumerate(selected, start=1):
        s["sample_label"] = f"day_sample_{i:02d}"
        s["covered_target_tokens"] = sorted(set(s["union_tokens"]) & TARGET_TOKENS)
    return selected


def render_selected(selected: list[dict[str, Any]], raw_counts: dict[str, int]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for s in selected:
            # Keep at most the last 13 completed DAY entries; assign DAY_REL tokens for display.
            entries = s["day_entries"][-MAX_DAYS_PER_SAMPLE:]
            k = len(entries)
            display_entries = []
            for i, day in enumerate(entries):
                rel = -(k - i)
                seq = f"[DAY_REL_{rel}] " + day["sequence"] + " [SEP]"
                display_entries.append({**day, "day_rel": rel, "display_sequence": seq})
            out = {**s, "day_entries": display_entries}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    lines: list[str] = []
    lines.append("# MIMIC DAY Token Samples")
    lines.append("")
    lines.append("Source: C4 trauma cohort, official MIMIC-IV raw tables. Generated by `generate_day_samples.py`.")
    lines.append("")
    lines.append("Scope note: this prototype does not emit `crystalloid_48h` because the MIMIC crystalloid/maintenance-fluid itemid registry is not frozen. RBC daily/48h events are emitted from explicit RBC inputevents itemids only.")
    lines.append("")
    lines.append("## Raw scan counts")
    lines.append("")
    lines.append("| Table/stream | Matched rows/events |")
    lines.append("|---|---:|")
    for k, v in raw_counts.items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("## Selected stays")
    lines.append("")
    lines.append("| Sample | HADM | Stay | Completed days | Main covered tokens |")
    lines.append("|---|---:|---:|---:|---|")
    for s in selected:
        toks = ", ".join(s["covered_target_tokens"][:12])
        if len(s["covered_target_tokens"]) > 12:
            toks += ", ..."
        lines.append(f"| `{s['sample_label']}` | {s['hadm_id']} | {s['stay_id']} | {s['completed_day_count']} | {toks} |")
    lines.append("")
    for s in selected:
        lines.append(f"## {s['sample_label']} — hadm {s['hadm_id']} / stay {s['stay_id']}")
        lines.append("")
        lines.append(f"Completed ICU days: {s['completed_day_count']}; ICU LOS days: {s['icu_los_days']:.2f}")
        lines.append("")
        lines.append("Covered target tokens:")
        lines.append("")
        lines.append("```text")
        lines.append(" ".join(s["covered_target_tokens"]))
        lines.append("```")
        lines.append("")
        entries = s["day_entries"][-MAX_DAYS_PER_SAMPLE:]
        k = len(entries)
        for i, day in enumerate(entries):
            rel = -(k - i)
            lines.append(f"### source_day_index={day['source_day_index']} / DAY_REL_{rel}")
            lines.append("")
            lines.append("```text")
            lines.append(f"[DAY_REL_{rel}] {day['sequence']} [SEP]")
            lines.append("```")
            audit = day["audit"]
            lines.append("")
            lines.append(
                f"audit: core_slots={audit['core_vital_slots']}/144; lab_draws={audit['lab_draws']}; "
                f"uop_slots={audit['uop_slots']}/24; map_low_hours={audit['map_low_hours']}; "
                f"vent_hours={audit['vent_hours']}; rbc_ml_day={audit['rbc_ml_day']}"
            )
            lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    pool = load_sample_pool()
    records = initialize_records(pool)
    raw_counts = {
        "pool_stays": len(pool),
        "day_chunks": len(records),
    }
    raw_counts["chartevents"] = scan_chartevents(pool, records)
    raw_counts["labevents"] = scan_labevents(pool, records)
    raw_counts["outputevents_uop"] = scan_outputevents(pool, records)
    raw_counts["inputevents_rbc"] = scan_inputevents_rbc(pool, records)
    raw_counts["procedureevents_vent_hours"] = scan_procedureevents_vent(pool, records)
    all_samples = build_all_samples(pool, records)
    selected = select_representative(all_samples)
    render_selected(selected, raw_counts)
    print(json.dumps({"outputs": [str(OUT_JSONL), str(OUT_MD)], "selected": [s["sample_label"] for s in selected], "raw_counts": raw_counts}, indent=2))


if __name__ == "__main__":
    main()
