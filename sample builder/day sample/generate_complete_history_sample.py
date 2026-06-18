#!/usr/bin/env python3
"""Generate one complete history sample: STATIC + DAY(window) + HOUR.

Single-patient, reusable, deterministic. Applies the DAY V1 window contract
including [day_window_len_XXh], FIRST48-once at source_day_index=1, RR burden
tokens, and the fixed HOUR vital-slots contract (text rendering only).

Output:
  sample builder/day sample/complete_history_sample.md
  sample builder/day sample/complete_history_sample.json

All rules mirror tokenizer/day_token.md and tokenizer/hour_token.md V1 contracts.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/home/vanila/code/EHR-Predict")
COHORT = ROOT / "data dicision/trauma cohort/cohort/mimiciv_trauma_cohort_los48.csv"
MIMIC = Path("/mnt/d/Data/mimic-iv-2.2")
ED = Path("/mnt/d/Data/mimic-iv-ed/2.2/ed")
OUT_DIR = ROOT / "sample builder/day sample"
OUT_MD = OUT_DIR / "complete_history_sample.md"
OUT_JSON = OUT_DIR / "complete_history_sample.json"

# Patient: day_sample_06 from the existing pool; ED-linked, medium-long course.
HADM_ID = "26488509"
STAY_ID = "31292653"

# Anchor: day_index 3, hour 12 → current partial DAY_REL_0 covers 0–12h.
CURRENT_DAY_INDEX = 3
PRED_HOUR_IN_DAY = 12
RECENT_H = 24
NDAYS = 19  # total completed ICU-day count for this stay

# ---------------------------------------------------------------------------
# Registry maps (same as generate_day_samples.py)
# ---------------------------------------------------------------------------
CORE_VITALS = ["hr", "sbp", "dbp", "map", "rr", "temp"]

CHARTEVENT_MAP = {
    220045: "hr", 220050: "sbp", 220179: "sbp",
    220051: "dbp", 220180: "dbp",
    220052: "map", 220181: "map",
    220210: "rr",
    223761: "temp_f", 223762: "temp_c", 226329: "temp_c",
    223835: "fio2",
}
LABEVENT_MAP = {
    50882: "bicarb", 50803: "bicarb", 50804: "bicarb", 51739: "bicarb",
    50912: "creatinine", 52546: "creatinine", 52024: "creatinine",
    51006: "bun", 52647: "bun",
    51300: "wbc", 51301: "wbc", 51755: "wbc", 51756: "wbc",
    50813: "lactate", 52442: "lactate", 53154: "lactate",
    50802: "base_excess",
}
UOP_ITEMIDS = {226559, 226560, 226566, 226627, 226631, 226713, 227489}
RBC_ITEMIDS = {225168, 226368, 227070}
VENT_PROCEDURE = {225792}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

def fnum(x: str | None) -> float | None:
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v if v == v else None  # NaN guard

def age_bin(age: float) -> str:
    a = int(age)
    if a < 40:       return "[age_bin_18_39]"
    elif a < 55:     return "[age_bin_40_54]"
    elif a < 65:     return "[age_bin_55_64]"
    elif a < 75:     return "[age_bin_65_74]"
    elif a < 85:     return "[age_bin_75_84]"
    else:            return "[age_bin_85_89]"

def sbp_bin(v: float) -> str:
    if v <= 89:      return "[initial_ed_sbp_bin_hypotension]"
    elif v <= 110:   return "[initial_ed_sbp_bin_borderline_low]"
    else:            return "[initial_ed_sbp_bin_not_low]"

def rsi_bin(rsi: float) -> str:
    if rsi <= 1.0:   return "[reverse_shock_index_bin_high_risk]"
    elif rsi <= 1.7: return "[reverse_shock_index_bin_intermediate]"
    else:            return "[reverse_shock_index_bin_low_risk]"

# ---------------------------------------------------------------------------
# data containers
# ---------------------------------------------------------------------------
@dataclass
class DayRaw:
    values: dict[str, dict[int, list[float]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    labs: dict[str, dict[int, list[float]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    core_slots: dict[str, set[int]] = field(default_factory=lambda: {k: set() for k in CORE_VITALS})
    uop_slots: set[int] = field(default_factory=set)
    rbc_ml: float = 0.0
    rbc_by_hour: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    vent_hours: set[int] = field(default_factory=set)

@dataclass
class HourRaw:
    values: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    vent: bool = False
    rbc_ml: float = 0.0

# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
def load_cohort_static() -> tuple[dict[str, str], datetime, float]:
    """Return cohort row, intime, and age for the selected stay."""
    with open(COHORT, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["hadm_id"] == HADM_ID and r["stay_id"] == STAY_ID:
                return r, dt(r["intime"]), float(r.get("age_at_admit") or r.get("anchor_age") or 0)
    raise SystemExit(f"Stay {STAY_ID} not found in cohort")

def load_admission() -> dict[str, str]:
    with gzip.open(MIMIC / "hosp/admissions.csv.gz", "rt", newline="") as f:
        for r in csv.DictReader(f):
            if r["hadm_id"] == HADM_ID:
                return r
    return {}

def load_ed_triage() -> tuple[str | None, float | None, float | None]:
    """return ed_stay_id, ed_sbp, ed_hr for this hadm"""
    ed_stay = None
    with gzip.open(ED / "edstays.csv.gz", "rt", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("hadm_id") == HADM_ID:
                ed_stay = r["stay_id"]
                break
    if not ed_stay:
        return None, None, None
    with gzip.open(ED / "triage.csv.gz", "rt", newline="") as f:
        for r in csv.DictReader(f):
            if r["stay_id"] == ed_stay:
                return ed_stay, fnum(r.get("sbp")), fnum(r.get("heartrate"))
    return ed_stay, None, None

def has_head_injury() -> bool:
    HEAD_CODES = ("S02", "S04", "S06", "S07", "S09", "800", "801", "803", "804", "850", "851", "852", "853", "854")
    with gzip.open(MIMIC / "hosp/diagnoses_icd.csv.gz", "rt", newline="") as f:
        for r in csv.DictReader(f):
            if r["hadm_id"] == HADM_ID:
                code = r["icd_code"].replace(".", "").upper()
                if any(code.startswith(hc) for hc in HEAD_CODES):
                    return True
    return False

def day_hour(intime: datetime, t: datetime) -> tuple[int, int] | None:
    delta = (t - intime).total_seconds()
    d = int(delta // 86400)
    h = int((delta % 86400) // 3600)
    return (d, h) if 0 <= d < NDAYS else None

def scan_day_raws(intime: datetime) -> list[DayRaw]:
    days = [DayRaw() for _ in range(NDAYS)]
    stay_lookup = {STAY_ID}

    # --- chartevents ---
    with gzip.open(MIMIC / "icu/chartevents.csv.gz", "rt", newline="") as f:
        reader = csv.reader(f); h = next(reader); idx = {c: i for i, c in enumerate(h)}
        for row in reader:
            if row[idx["stay_id"]] not in stay_lookup:
                continue
            itemid = int(row[idx["itemid"]])
            field = CHARTEVENT_MAP.get(itemid)
            if not field:
                continue
            v = fnum(row[idx["valuenum"]])
            if v is None:
                continue
            dh = day_hour(intime, dt(row[idx["charttime"]]))
            if dh is None:
                continue
            d, hh = dh
            rec = days[d]
            if field == "temp_f":
                field = "temp"; v = (v - 32.0) * 5.0 / 9.0
            elif field == "temp_c":
                field = "temp"
            elif field == "fio2" and v > 1:
                v /= 100.0
            if field in rec.core_slots:
                rec.core_slots[field].add(hh)
            rec.values[field][hh].append(v)

    # --- labevents ---
    with gzip.open(MIMIC / "hosp/labevents.csv.gz", "rt", newline="") as f:
        reader = csv.reader(f); h = next(reader); idx = {c: i for i, c in enumerate(h)}
        for row in reader:
            if row[idx["hadm_id"]] != HADM_ID:
                continue
            itemid = int(row[idx["itemid"]])
            field = LABEVENT_MAP.get(itemid)
            if not field:
                continue
            v = fnum(row[idx["valuenum"]])
            if v is None:
                continue
            dh = day_hour(intime, dt(row[idx["charttime"]]))
            if dh is None:
                continue
            d, hh = dh
            days[d].labs[field][hh].append(v)

    # --- outputevents (uop) ---
    with gzip.open(MIMIC / "icu/outputevents.csv.gz", "rt", newline="") as f:
        reader = csv.reader(f); h = next(reader); idx = {c: i for i, c in enumerate(h)}
        for row in reader:
            if row[idx["stay_id"]] not in stay_lookup:
                continue
            itemid = int(row[idx["itemid"]])
            if itemid not in UOP_ITEMIDS:
                continue
            if fnum(row[idx["value"]]) is None:
                continue
            dh = day_hour(intime, dt(row[idx["charttime"]]))
            if dh is None:
                continue
            days[dh[0]].uop_slots.add(dh[1])

    # --- inputevents (rbc) ---
    with gzip.open(MIMIC / "icu/inputevents.csv.gz", "rt", newline="") as f:
        reader = csv.reader(f); h = next(reader); idx = {c: i for i, c in enumerate(h)}
        for row in reader:
            if row[idx["stay_id"]] not in stay_lookup:
                continue
            itemid = int(row[idx["itemid"]])
            if itemid not in RBC_ITEMIDS:
                continue
            amt = fnum(row[idx["amount"]])
            if not amt or amt <= 0:
                continue
            dh = day_hour(intime, dt(row[idx["starttime"]]))
            if dh is None:
                continue
            days[dh[0]].rbc_ml += amt
            days[dh[0]].rbc_by_hour[dh[1]] += amt

    # --- procedureevents (vent) ---
    with gzip.open(MIMIC / "icu/procedureevents.csv.gz", "rt", newline="") as f:
        reader = csv.reader(f); h = next(reader); idx = {c: i for i, c in enumerate(h)}
        for row in reader:
            if row[idx["stay_id"]] not in stay_lookup:
                continue
            itemid = int(row[idx["itemid"]])
            if itemid not in VENT_PROCEDURE:
                continue
            start = dt(row[idx["starttime"]])
            end = dt(row[idx["endtime"]])
            interval_start = max(start, intime)
            for d in range(NDAYS):
                for hh in range(24):
                    hs = intime + timedelta(days=d, hours=hh)
                    he = hs + timedelta(hours=1)
                    if he > interval_start and hs < end:
                        days[d].vent_hours.add(hh)

    return days

def scan_hour_raws(intime: datetime, observed_until: datetime) -> list[HourRaw]:
    recent_start = observed_until - timedelta(hours=RECENT_H)
    hours = [HourRaw() for _ in range(RECENT_H)]
    stay_lookup = {STAY_ID}

    # vitals
    with gzip.open(MIMIC / "icu/chartevents.csv.gz", "rt", newline="") as f:
        reader = csv.reader(f); h = next(reader); idx = {c: i for i, c in enumerate(h)}
        for row in reader:
            if row[idx["stay_id"]] not in stay_lookup:
                continue
            itemid = int(row[idx["itemid"]])
            field = CHARTEVENT_MAP.get(itemid)
            if not field:
                continue
            v = fnum(row[idx["valuenum"]])
            if v is None:
                continue
            t = dt(row[idx["charttime"]])
            if not (recent_start <= t < observed_until):
                continue
            hh = int((t - recent_start).total_seconds() // 3600)
            if not (0 <= hh < RECENT_H):
                continue
            if field == "temp_f":
                field = "temp"; v = (v - 32.0) * 5.0 / 9.0
            elif field == "temp_c":
                field = "temp"
            elif field == "fio2" and v > 1:
                v /= 100.0
            hours[hh].values[field].append(v)

    # vent
    with gzip.open(MIMIC / "icu/procedureevents.csv.gz", "rt", newline="") as f:
        reader = csv.reader(f); h = next(reader); idx = {c: i for i, c in enumerate(h)}
        for row in reader:
            if row[idx["stay_id"]] not in stay_lookup:
                continue
            itemid = int(row[idx["itemid"]])
            if itemid not in VENT_PROCEDURE:
                continue
            start = dt(row[idx["starttime"]])
            end = dt(row[idx["endtime"]])
            for hh in range(RECENT_H):
                hs = recent_start + timedelta(hours=hh)
                he = hs + timedelta(hours=1)
                if he > start and hs < end:
                    hours[hh].vent = True

    # rbc
    with gzip.open(MIMIC / "icu/inputevents.csv.gz", "rt", newline="") as f:
        reader = csv.reader(f); h = next(reader); idx = {c: i for i, c in enumerate(h)}
        for row in reader:
            if row[idx["stay_id"]] not in stay_lookup:
                continue
            itemid = int(row[idx["itemid"]])
            if itemid not in RBC_ITEMIDS:
                continue
            amt = fnum(row[idx["amount"]])
            if not amt or amt <= 0:
                continue
            st = dt(row[idx["starttime"]])
            en = dt(row[idx["endtime"]])
            dur = (en - st).total_seconds()
            for hh in range(RECENT_H):
                hs = recent_start + timedelta(hours=hh)
                he = hs + timedelta(hours=1)
                if dur > 0:
                    overlap = max(0.0, (min(en, he) - max(st, hs)).total_seconds())
                    if overlap > 0:
                        hours[hh].rbc_ml += amt * overlap / dur
                elif hs <= st < he:
                    hours[hh].rbc_ml += amt

    return hours

# ---------------------------------------------------------------------------
# flat values helper
# ---------------------------------------------------------------------------
def flat(dr: DayRaw, grp: str, fld: str) -> list[float]:
    src = dr.values if grp == "values" else dr.labs
    return [v for vals in src[fld].values() for v in vals]

# ---------------------------------------------------------------------------
# first48
# ---------------------------------------------------------------------------
def first48_summary(days: list[DayRaw]) -> dict[str, float | None]:
    lac = []; bd = []; rbc = 0.0
    for d in (0, 1):
        lac += flat(days[d], "labs", "lactate")
        bd += [max(0.0, -v) for v in flat(days[d], "labs", "base_excess")]
        rbc += days[d].rbc_ml
    return {"lactate48": max(lac) if lac else None,
            "base_deficit48": max(bd) if bd else None,
            "rbc48_ml": rbc}

# ---------------------------------------------------------------------------
# DAY token builder (applies V1 window contract)
# ---------------------------------------------------------------------------
def build_day_tokens(current_day_index: int, days: list[DayRaw], first48: dict) -> list[list[str]]:
    prev_cr: float | None = None
    vent_count = 0
    result: list[list[str]] = []
    for d in range(NDAYS):
        dr = days[d]
        is_current_partial = (d == current_day_index)
        eligible_h = PRED_HOUR_IN_DAY + 1 if is_current_partial else 24
        allowed_hours = set(range(eligible_h))
        window_tok = f"[day_window_len_{eligible_h:02d}h]"
        tokens: list[str] = [window_tok]
        domains: list[str] = []

        def wvals(group: str, field: str) -> list[float]:
            src = dr.values if group == "values" else dr.labs
            return [v for h, vals in src[field].items() if h in allowed_hours for v in vals]

        rbc_ml_window = sum(v for h, v in dr.rbc_by_hour.items() if h in allowed_hours)

        def dm(name: str) -> None:
            m = f"[{name}]"
            if m not in tokens:
                tokens.append(m)
                domains.append(name)

        # --- perfusion/shock ---
        perf: list[str] = []
        ml = sum(1 for h, vs in dr.values["map"].items() if h in allowed_hours and any(v < 65 for v in vs))
        if 1 <= ml <= 3:     perf.append("[map_low_hours_bin_brief]")
        elif 4 <= ml <= 8:   perf.append("[map_low_hours_bin_intermittent]")
        elif 9 <= ml <= 16:  perf.append("[map_low_hours_bin_prolonged]")
        elif ml > 16:        perf.append("[map_low_hours_bin_persistent]")

        sbp = wvals("values", "sbp")
        if sbp:
            mn = min(sbp)
            if mn < 90:                    perf.append("[systolic_bp_min_bin_hypotension]")
            elif mn <= 100:                perf.append("[systolic_bp_min_bin_low]")
            elif age >= 65 and mn <= 109:  perf.append("[systolic_bp_min_bin_geriatric_low]")

        hr = wvals("values", "hr")
        if any(v >= 131 for v in hr):
            perf.append("[heart_rate_max_bin_extreme_tachycardia]")

        # FIRST48 once at source_day_index=1
        first48_visible = (d == 1) and ((not is_current_partial) or eligible_h == 24)
        if first48_visible:
            l48 = first48["lactate48"]
            if l48 is not None:
                if 2.0 < l48 <= 5.0:  perf.append("[lactate_48h_bin_elevated]")
                elif l48 > 5.0:       perf.append("[lactate_48h_bin_severe]")
            b48 = first48["base_deficit48"]
            if b48 is not None:
                if 3.0 <= b48 < 6.0:   perf.append("[base_deficit_48h_bin_mild]")
                elif 6.0 <= b48 < 10.0: perf.append("[base_deficit_48h_bin_moderate]")
                elif b48 >= 10.0:       perf.append("[base_deficit_48h_bin_severe]")

        if perf:
            dm("perfusion_shock"); tokens += perf

        # --- oxygenation/ventilation ---
        oxy: list[str] = []
        raw_vh = dr.vent_hours
        if is_current_partial:
            raw_vh = {h for h in raw_vh if h <= PRED_HOUR_IN_DAY}
        vh = len(raw_vh)
        if 0 < vh < 0.5 * eligible_h:              oxy.append("[vent_hours_bin_partial_window]")
        elif 0.5 * eligible_h <= vh < eligible_h:   oxy.append("[vent_hours_bin_most_window]")
        elif vh == eligible_h and vh > 0:           oxy.append("[vent_hours_bin_full_window]")

        vent_course_idx = vent_count + 1 if vh > 0 else 0
        if vent_course_idx == 1:    oxy.append("[vent_course_bin_first_day]")
        elif    2 <= vent_course_idx <= 3:  oxy.append("[vent_course_bin_early]")
        elif    vent_course_idx >= 4:        oxy.append("[vent_course_bin_prolonged]")

        fio = wvals("values", "fio2")
        if fio:
            mx = max(fio)
            if 0.40 < mx <= 0.60:  oxy.append("[fio2_max_bin_high_support]")
            elif mx > 0.60:        oxy.append("[fio2_max_bin_very_high_support]")

        rr_hh = sum(1 for h, vs in dr.values["rr"].items() if h in allowed_hours and any(v >= 25 for v in vs))
        if 1 <= rr_hh <= 3:       oxy.append("[respiratory_rate_high_hours_bin_brief]")
        elif 4 <= rr_hh <= 8:     oxy.append("[respiratory_rate_high_hours_bin_intermediate]")
        elif rr_hh >= 9:          oxy.append("[respiratory_rate_high_hours_bin_prolonged]")

        if oxy:
            dm("oxygenation_ventilation"); tokens += oxy

        # --- renal/metabolic ---
        renal: list[str] = []
        cr = wvals("labs", "creatinine")
        cur_cr = max(cr) if cr else prev_cr
        if cr and prev_cr is not None:
            mxcr = max(cr)
            if mxcr - prev_cr >= 0.3: renal.append("[creatinine_change_bin_kdigo_delta]")
            if prev_cr > 0 and mxcr / prev_cr >= 1.5: renal.append("[creatinine_ratio_bin_kdigo_ratio]")
        if any(v < 22 for v in wvals("labs", "bicarb")):
            renal.append("[bicarbonate_min_bin_low]")

        bun = wvals("labs", "bun")
        if bun and cr:
            mb = max(bun)
            min_cr = min(v for v in cr if v > 0) if any(v > 0 for v in cr) else None
            if min_cr is not None and mb > 20 and mb / min_cr > 20:
                renal.append("[bun_creatinine_ratio_bin_prerenal_pattern]")

        # no kdigo_uop — no weight registry
        if renal:
            dm("renal_metabolic"); tokens += renal

        # --- immune/hematologic ---
        imm: list[str] = []
        wbc = wvals("labs", "wbc")
        if any(v > 12 for v in wbc):  imm.append("[wbc_bin_high]")
        if any(v < 4 for v in wbc):   imm.append("[wbc_bin_low]")
        if rbc_ml_window > 0:         imm.append("[rbc_transfusion_event_present]")
        if imm:
            dm("immune_hematologic"); tokens += imm

        # --- resuscitation burden ---
        res: list[str] = []
        if d == 1 and (first48["rbc48_ml"] or 0) > 0:
            res.append("[rbc_48h_event_present]")
        if res:
            dm("resuscitation_burden"); tokens += res

        # --- data quality ---
        dq: list[str] = []
        cs = sum(len(dr.core_slots[f] & allowed_hours) for f in CORE_VITALS)
        denom = eligible_h * 6
        if cs >= 0.83 * denom:        dq.append("[core_vital_slots_dense]")
        elif cs >= 0.5 * denom:       dq.append("[core_vital_slots_partial]")
        elif cs > 0:                  dq.append("[core_vital_slots_sparse]")
        else:                         dq.append("[core_vital_slots_none]")

        lab_draws = sum(len(vals) for f_ in ["bicarb","bun","creatinine","wbc"] for h, vals in dr.labs[f_].items() if h in allowed_hours)
        if lab_draws == 0:
            dq.append("[labs_not_drawn]")

        us = len(dr.uop_slots & allowed_hours)
        if us >= 6:     dq.append("[uop_measured]")
        elif us >= 1:   dq.append("[uop_sparse]")
        else:           dq.append("[uop_not_measured]")

        if dq:
            dm("data_quality"); tokens += dq

        result.append(tokens)
        prev_cr = cur_cr
        if vh > 0:
            vent_count += 1

    return result

# ---------------------------------------------------------------------------
# STATIC builder
# ---------------------------------------------------------------------------
def build_static(cohort_row: dict, admission: dict, ed_sbp: float | None, ed_hr: float | None, head: bool) -> list[str]:
    tokens = ["[STATIC]", age_bin(age)]
    tokens.append("[sex_M]" if cohort_row.get("gender") == "M" else "[sex_F]")
    mech = cohort_row.get("trauma_types", "")
    tokens.append("[injury_mechanism_blunt]" if "Blunt" in mech else "[injury_mechanism_other]")

    loc = (admission or {}).get("admission_location", "")
    tokens.append("[transfer_transfer]" if "TRANSFER" in loc.upper() else "[transfer_direct]")

    if ed_sbp is not None and ed_hr is not None and ed_hr > 0:
        rsi = ed_sbp / ed_hr
        tokens += ["[ed_linkage_yes]",
                   "[initial_ed_sbp]", f"<{ed_sbp:.0f}>", sbp_bin(ed_sbp),
                   "[reverse_shock_index]", f"<{rsi:.2f}>", rsi_bin(rsi)]
    else:
        tokens.append("[ed_linkage_no]")

    tokens.append("[head_injury_yes]" if head else "[head_injury_no]")
    tokens.append("[SEP]")
    return tokens

# ---------------------------------------------------------------------------
# HOUR builder (text rendering only — training uses fixed [T,7] plus masks)
# ---------------------------------------------------------------------------
def build_hour_lines(hours: list[HourRaw], intime: datetime) -> list[str]:
    ORDER = ["hr", "sbp", "dbp", "map", "rr", "temp", "fio2"]
    TOK = {"hr": "[heart_rate]", "sbp": "[systolic_bp]", "dbp": "[diastolic_bp]",
           "map": "[mean_arterial_pressure]", "rr": "[respiratory_rate]",
           "temp": "[temperature]", "fio2": "[fio2]"}
    lines: list[str] = []
    for hh in range(RECENT_H):
        hr = hours[hh]
        rel = hh - (RECENT_H - 1)
        parts = [f"[HOUR_REL_{rel}]"]
        for f in ORDER:
            vals = hr.values.get(f, [])
            if vals:
                v = vals[-1]  # last in hour
                if f == "temp":    txt = f"{v:.1f}"
                elif f == "fio2":  txt = f"{v:.2f}"
                else:              txt = f"{v:.0f}"
                parts += [TOK[f], f"<{txt}>"]
        if hr.vent:   parts.append("[vent_on]")
        if hr.rbc_ml > 0: parts += ["[rbc_transfusion_1h]", f"<{hr.rbc_ml:.0f}>"]
        if rel == 0: parts.append("[CUR]")
        parts.append("[SEP]")
        lines.append(" ".join(parts))
    return lines

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    global age  # used by build_day_tokens
    cohort_row, intime, age = load_cohort_static()
    admission = load_admission()
    _, ed_sbp, ed_hr = load_ed_triage()
    head = has_head_injury()

    observed_until = intime + timedelta(days=CURRENT_DAY_INDEX, hours=PRED_HOUR_IN_DAY + 1)

    print(f"Scanning MIMIC raw tables for hadm {HADM_ID} stay {STAY_ID} …")
    days = scan_day_raws(intime)
    hours = scan_hour_raws(intime, observed_until)

    first48 = first48_summary(days)
    day_tokens_all = build_day_tokens(CURRENT_DAY_INDEX, days, first48)

    # select visible DAY blocks: prior completed (0..CURRENT_DAY_INDEX-1) + current partial (CURRENT_DAY_INDEX)
    visible_days = day_tokens_all[:CURRENT_DAY_INDEX+1]
    static_tokens = build_static(cohort_row, admission, ed_sbp, ed_hr, head)
    hour_lines = build_hour_lines(hours, intime)

    # build text rendering
    text_lines = ["```text"]
    text_lines.append(" ".join(static_tokens))
    text_lines.append("")
    for di, toks in enumerate(visible_days):
        rel = di - CURRENT_DAY_INDEX
        text_lines.append(f"[DAY_REL_{rel}] " + " ".join(toks) + " [SEP]")
        text_lines.append("")
    text_lines += hour_lines
    text_lines.append("```")
    rendered = "\n".join(text_lines)

    # json
    json_out = {
        "hadm_id": HADM_ID, "stay_id": STAY_ID,
        "current_day_index": CURRENT_DAY_INDEX, "pred_hour_in_day": PRED_HOUR_IN_DAY,
        "recent_h": RECENT_H, "observed_until": str(observed_until),
        "static_tokens": " ".join(static_tokens),
        "day_blocks": [
            {"day_rel": di - CURRENT_DAY_INDEX, "source_day_index": di, "tokens": " ".join(toks)}
            for di, toks in enumerate(visible_days)
        ],
        "hour_blocks": hour_lines,
    }

    OUT_MD.write_text(f"# Complete History Sample — hadm {HADM_ID} / stay {STAY_ID}\n\n{rendered}\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(json_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Done → {OUT_MD}")
    print(f"      {OUT_JSON}")
    print()
    print(rendered)

if __name__ == "__main__":
    main()
