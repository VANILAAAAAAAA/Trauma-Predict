#!/usr/bin/env python3
"""Generate 5-10 complete EHR-Predict history samples with one raw-table scan.

Outputs:
  complete_history_samples.jsonl
  complete_history_samples.md

The builder applies current V1 contracts:
- STATIC + prior DAY windows + current DAY_REL_0 + recent HOUR
- each DAY block has [day_window_len_XXh]
- current DAY_REL_0 truncates all rules at observed_until_t
- FIRST48 tokens emit once on source_day_index=1 when full first 48h is visible
- RR is duration burden, not max trigger
- HOUR text is review rendering; model-side contract is vital_values[T,7]+vital_mask[T,7]
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path('/home/vanila/code/EHR-Predict')
MIMIC = Path('/mnt/d/Data/mimic-iv-2.2')
ED = Path('/mnt/d/Data/mimic-iv-ed/2.2/ed')
COHORT = ROOT / 'data dicision/trauma cohort/cohort/mimiciv_trauma_cohort_los48.csv'
DAY_SAMPLE_DIR = ROOT / 'sample builder/day sample'
IN_SELECTED = DAY_SAMPLE_DIR / 'day_samples.jsonl'
OUT_JSONL = DAY_SAMPLE_DIR / 'complete_history_samples.jsonl'
OUT_MD = DAY_SAMPLE_DIR / 'complete_history_samples.md'
MAX_SAMPLES = 6
RECENT_H = 24

CORE_VITALS = ['hr', 'sbp', 'dbp', 'map', 'rr', 'temp']
HOUR_ORDER = ['hr', 'sbp', 'dbp', 'map', 'rr', 'temp', 'fio2']
HOUR_TOK = {
    'hr': '[heart_rate]', 'sbp': '[systolic_bp]', 'dbp': '[diastolic_bp]',
    'map': '[mean_arterial_pressure]', 'rr': '[respiratory_rate]',
    'temp': '[temperature]', 'fio2': '[fio2]',
}
CHARTEVENT_MAP = {
    220045: 'hr', 220050: 'sbp', 220179: 'sbp', 220051: 'dbp', 220180: 'dbp',
    220052: 'map', 220181: 'map', 220210: 'rr', 223761: 'temp_f', 223762: 'temp_c',
    226329: 'temp_c', 223835: 'fio2',
}
LABEVENT_MAP = {
    50882: 'bicarb', 50803: 'bicarb', 50804: 'bicarb', 51739: 'bicarb',
    50912: 'creatinine', 52546: 'creatinine', 52024: 'creatinine',
    51006: 'bun', 52647: 'bun', 51300: 'wbc', 51301: 'wbc', 51755: 'wbc', 51756: 'wbc',
    50813: 'lactate', 52442: 'lactate', 53154: 'lactate', 50802: 'base_excess',
}
UOP_ITEMIDS = {226559, 226560, 226566, 226627, 226631, 226713, 227489}
RBC_ITEMIDS = {225168, 226368, 227070}
VENT_PROCEDURE = {225792}
HEAD_CODES = ('S02', 'S04', 'S06', 'S07', 'S09', '800', '801', '803', '804', '850', '851', '852', '853', '854')
SOURCE_TABLES = [
    'mimic-iv-2.2/icu/chartevents.csv.gz', 'mimic-iv-2.2/hosp/labevents.csv.gz',
    'mimic-iv-2.2/icu/outputevents.csv.gz', 'mimic-iv-2.2/icu/inputevents.csv.gz',
    'mimic-iv-2.2/icu/procedureevents.csv.gz', 'mimic-iv-2.2/hosp/admissions.csv.gz',
    'mimic-iv-2.2/hosp/diagnoses_icd.csv.gz', 'mimic-iv-ed/2.2/ed/edstays.csv.gz',
    'mimic-iv-ed/2.2/ed/triage.csv.gz'
]


def dt(s: str) -> datetime:
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')


def fnum(x: str | None) -> float | None:
    if x is None or x == '':
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


@dataclass
class Stay:
    sample_label: str
    subject_id: str
    hadm_id: str
    stay_id: str
    intime: datetime
    outtime: datetime
    ndays: int
    age: float
    gender: str
    trauma_types: str
    current_day: int
    pred_hour: int

    @property
    def observed_until(self) -> datetime:
        return self.intime + timedelta(days=self.current_day, hours=self.pred_hour + 1)

    @property
    def recent_start(self) -> datetime:
        return self.observed_until - timedelta(hours=RECENT_H)


@dataclass
class DayRaw:
    values: dict[str, dict[int, list[float]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    labs: dict[str, dict[int, list[float]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    core_slots: dict[str, set[int]] = field(default_factory=lambda: {k: set() for k in CORE_VITALS})
    uop_slots: set[int] = field(default_factory=set)
    rbc_by_hour: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    vent_hours: set[int] = field(default_factory=set)


@dataclass
class HourRaw:
    values: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    vent: bool = False
    rbc_ml: float = 0.0


def anchor_for(ndays: int) -> tuple[int, int]:
    day = min(3, max(0, ndays - 1))
    pred_h = 23 if day == 1 else 12
    return day, pred_h


def load_selected() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in IN_SELECTED.read_text(encoding='utf-8').splitlines() if line.strip()]
    return rows[:MAX_SAMPLES]


def load_stays() -> dict[str, Stay]:
    selected = load_selected()
    wanted = {str(r['stay_id']): r for r in selected}
    stays: dict[str, Stay] = {}
    with COHORT.open(newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['stay_id'] not in wanted:
                continue
            base = wanted[r['stay_id']]
            intime, outtime = dt(r['intime']), dt(r['outtime'])
            ndays = int((outtime - intime).total_seconds() // 86400)
            cur_day, pred_h = anchor_for(ndays)
            stays[r['stay_id']] = Stay(
                sample_label=base.get('sample_label', f"sample_{len(stays)+1:02d}"),
                subject_id=r['subject_id'], hadm_id=r['hadm_id'], stay_id=r['stay_id'],
                intime=intime, outtime=outtime, ndays=ndays,
                age=float(r.get('age_at_admit') or r.get('anchor_age') or 0),
                gender=r.get('gender', ''), trauma_types=r.get('trauma_types', ''),
                current_day=cur_day, pred_hour=pred_h,
            )
    missing = set(wanted) - set(stays)
    if missing:
        raise RuntimeError(f'missing cohort rows: {sorted(missing)}')
    return stays


def day_hour(stay: Stay, t: datetime) -> tuple[int, int] | None:
    if t < stay.intime or t >= stay.outtime:
        return None
    sec = (t - stay.intime).total_seconds()
    d = int(sec // 86400)
    h = int((sec % 86400) // 3600)
    return (d, h) if 0 <= d < stay.ndays else None


def recent_hour(stay: Stay, t: datetime) -> int | None:
    if not (stay.recent_start <= t < stay.observed_until):
        return None
    h = int((t - stay.recent_start).total_seconds() // 3600)
    return h if 0 <= h < RECENT_H else None


def normalize_chart_field(field: str, value: float) -> tuple[str, float]:
    if field == 'temp_f':
        return 'temp', (value - 32.0) * 5.0 / 9.0
    if field == 'temp_c':
        return 'temp', value
    if field == 'fio2' and value > 1:
        return 'fio2', value / 100.0
    return field, value


def scan_raw(stays: dict[str, Stay]) -> tuple[dict[tuple[str, int], DayRaw], dict[str, list[HourRaw]], dict[str, int]]:
    day_records = {(sid, d): DayRaw() for sid, st in stays.items() for d in range(st.ndays)}
    hour_records = {sid: [HourRaw() for _ in range(RECENT_H)] for sid in stays}
    by_hadm = defaultdict(list)
    for st in stays.values():
        by_hadm[st.hadm_id].append(st)
    counts = {'chartevents': 0, 'labevents': 0, 'outputevents_uop': 0, 'inputevents_rbc': 0, 'procedureevents_vent_hours': 0}

    with gzip.open(MIMIC / 'icu/chartevents.csv.gz', 'rt', newline='') as f:
        reader = csv.reader(f); header = next(reader); idx = {c:i for i,c in enumerate(header)}
        for row in reader:
            st = stays.get(row[idx['stay_id']])
            if not st: continue
            field = CHARTEVENT_MAP.get(int(row[idx['itemid']]))
            if not field: continue
            val = fnum(row[idx['valuenum']])
            if val is None: continue
            t = dt(row[idx['charttime']])
            dh = day_hour(st, t)
            if dh is not None:
                d, h = dh
                field2, val2 = normalize_chart_field(field, val)
                rec = day_records[(st.stay_id, d)]
                if field2 in rec.core_slots:
                    rec.core_slots[field2].add(h)
                rec.values[field2][h].append(val2)
                counts['chartevents'] += 1
            rh = recent_hour(st, t)
            if rh is not None:
                field2, val2 = normalize_chart_field(field, val)
                if field2 in HOUR_ORDER:
                    hour_records[st.stay_id][rh].values[field2].append(val2)

    with gzip.open(MIMIC / 'hosp/labevents.csv.gz', 'rt', newline='') as f:
        reader = csv.reader(f); header = next(reader); idx = {c:i for i,c in enumerate(header)}
        for row in reader:
            itemid = int(row[idx['itemid']])
            field = LABEVENT_MAP.get(itemid)
            if not field: continue
            val = fnum(row[idx['valuenum']])
            if val is None: continue
            for st in by_hadm.get(row[idx['hadm_id']], []):
                dh = day_hour(st, dt(row[idx['charttime']]))
                if dh is None: continue
                d, h = dh
                day_records[(st.stay_id, d)].labs[field][h].append(val)
                counts['labevents'] += 1

    with gzip.open(MIMIC / 'icu/outputevents.csv.gz', 'rt', newline='') as f:
        reader = csv.reader(f); header = next(reader); idx = {c:i for i,c in enumerate(header)}
        for row in reader:
            st = stays.get(row[idx['stay_id']])
            if not st: continue
            if int(row[idx['itemid']]) not in UOP_ITEMIDS: continue
            if fnum(row[idx['value']]) is None: continue
            dh = day_hour(st, dt(row[idx['charttime']]))
            if dh is None: continue
            day_records[(st.stay_id, dh[0])].uop_slots.add(dh[1])
            counts['outputevents_uop'] += 1

    with gzip.open(MIMIC / 'icu/inputevents.csv.gz', 'rt', newline='') as f:
        reader = csv.reader(f); header = next(reader); idx = {c:i for i,c in enumerate(header)}
        for row in reader:
            st = stays.get(row[idx['stay_id']])
            if not st: continue
            if int(row[idx['itemid']]) not in RBC_ITEMIDS: continue
            amt = fnum(row[idx['amount']])
            if not amt or amt <= 0: continue
            start = dt(row[idx['starttime']]); end = dt(row[idx['endtime']])
            dh = day_hour(st, start)
            if dh is not None:
                day_records[(st.stay_id, dh[0])].rbc_by_hour[dh[1]] += amt
                counts['inputevents_rbc'] += 1
            dur = (end - start).total_seconds()
            for h in range(RECENT_H):
                hs = st.recent_start + timedelta(hours=h); he = hs + timedelta(hours=1)
                overlap = max(0.0, (min(end, he) - max(start, hs)).total_seconds()) if dur > 0 else (3600.0 if hs <= start < he else 0.0)
                if overlap > 0:
                    hour_records[st.stay_id][h].rbc_ml += amt * (overlap / dur if dur > 0 else 1.0)

    with gzip.open(MIMIC / 'icu/procedureevents.csv.gz', 'rt', newline='') as f:
        reader = csv.reader(f); header = next(reader); idx = {c:i for i,c in enumerate(header)}
        for row in reader:
            st = stays.get(row[idx['stay_id']])
            if not st: continue
            if int(row[idx['itemid']]) not in VENT_PROCEDURE: continue
            start = dt(row[idx['starttime']]); end = dt(row[idx['endtime']])
            for d in range(st.ndays):
                for h in range(24):
                    hs = st.intime + timedelta(days=d, hours=h); he = hs + timedelta(hours=1)
                    if he > start and hs < end:
                        day_records[(st.stay_id, d)].vent_hours.add(h)
                        counts['procedureevents_vent_hours'] += 1
            for h in range(RECENT_H):
                hs = st.recent_start + timedelta(hours=h); he = hs + timedelta(hours=1)
                if he > start and hs < end:
                    hour_records[st.stay_id][h].vent = True
    return day_records, hour_records, counts


def load_static_maps(stays: dict[str, Stay]) -> tuple[dict[str, dict[str,str]], dict[str, tuple[float|None,float|None]], set[str]]:
    admissions = {}
    wanted_hadm = {st.hadm_id for st in stays.values()}
    with gzip.open(MIMIC / 'hosp/admissions.csv.gz', 'rt', newline='') as f:
        for r in csv.DictReader(f):
            if r['hadm_id'] in wanted_hadm:
                admissions[r['hadm_id']] = r
    ed_stay_by_hadm = {}
    with gzip.open(ED / 'edstays.csv.gz', 'rt', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('hadm_id') in wanted_hadm:
                ed_stay_by_hadm[r['hadm_id']] = r['stay_id']
    triage = {}
    wanted_ed = set(ed_stay_by_hadm.values())
    with gzip.open(ED / 'triage.csv.gz', 'rt', newline='') as f:
        for r in csv.DictReader(f):
            if r['stay_id'] in wanted_ed:
                triage[r['stay_id']] = (fnum(r.get('sbp')), fnum(r.get('heartrate')))
    ed_map = {hadm: triage.get(edid, (None, None)) for hadm, edid in ed_stay_by_hadm.items()}
    head_hadm = set()
    with gzip.open(MIMIC / 'hosp/diagnoses_icd.csv.gz', 'rt', newline='') as f:
        for r in csv.DictReader(f):
            if r['hadm_id'] not in wanted_hadm: continue
            code = r['icd_code'].replace('.', '').upper()
            if any(code.startswith(hc) for hc in HEAD_CODES):
                head_hadm.add(r['hadm_id'])
    return admissions, ed_map, head_hadm


def age_bin(age: float) -> str:
    if age < 40: return '[age_bin_18_39]'
    if age < 55: return '[age_bin_40_54]'
    if age < 65: return '[age_bin_55_64]'
    if age < 75: return '[age_bin_65_74]'
    if age < 85: return '[age_bin_75_84]'
    return '[age_bin_85_89]'


def sbp_bin(v: float) -> str:
    if v <= 89: return '[initial_ed_sbp_bin_hypotension]'
    if v <= 110: return '[initial_ed_sbp_bin_borderline_low]'
    return '[initial_ed_sbp_bin_not_low]'


def rsi_bin(v: float) -> str:
    if v <= 1.0: return '[reverse_shock_index_bin_high_risk]'
    if v <= 1.7: return '[reverse_shock_index_bin_intermediate]'
    return '[reverse_shock_index_bin_low_risk]'


def build_static(st: Stay, admissions: dict[str,dict[str,str]], ed_map: dict[str,tuple[float|None,float|None]], head_hadm: set[str]) -> list[str]:
    toks = ['[STATIC]', age_bin(st.age), '[sex_M]' if st.gender == 'M' else '[sex_F]']
    toks.append('[injury_mechanism_blunt]' if 'Blunt' in st.trauma_types else '[injury_mechanism_other]')
    loc = admissions.get(st.hadm_id, {}).get('admission_location', '')
    toks.append('[transfer_transfer]' if 'TRANSFER' in loc.upper() else '[transfer_direct]')
    sbp, hr = ed_map.get(st.hadm_id, (None, None))
    if sbp is not None and hr is not None and hr > 0:
        rsi = sbp / hr
        toks += ['[ed_linkage_yes]', '[initial_ed_sbp]', f'<{sbp:.0f}>', sbp_bin(sbp), '[reverse_shock_index]', f'<{rsi:.2f}>', rsi_bin(rsi)]
    else:
        toks.append('[ed_linkage_no]')
    toks.append('[head_injury_yes]' if st.hadm_id in head_hadm else '[head_injury_no]')
    toks.append('[SEP]')
    return toks


def flat(rec: DayRaw, group: str, field: str, allowed: set[int]) -> list[float]:
    src = rec.values if group == 'values' else rec.labs
    return [v for h, vals in src[field].items() if h in allowed for v in vals]


def first48_summary(st: Stay, recs: dict[tuple[str,int], DayRaw]) -> dict[str, float|None]:
    lac, bd, rbc = [], [], 0.0
    for d in (0, 1):
        rec = recs.get((st.stay_id, d))
        if not rec: continue
        lac += flat(rec, 'labs', 'lactate', set(range(24)))
        bd += [max(0.0, -v) for v in flat(rec, 'labs', 'base_excess', set(range(24)))]
        rbc += sum(rec.rbc_by_hour.values())
    return {'lactate48': max(lac) if lac else None, 'base_deficit48': max(bd) if bd else None, 'rbc48_ml': rbc}


def build_day_tokens(st: Stay, recs: dict[tuple[str,int], DayRaw]) -> list[list[str]]:
    f48 = first48_summary(st, recs)
    out = []
    prev_cr = None
    vent_count = 0
    for d in range(st.current_day + 1):
        rec = recs[(st.stay_id, d)]
        eligible_h = st.pred_hour + 1 if d == st.current_day else 24
        allowed = set(range(eligible_h))
        toks = [f'[day_window_len_{eligible_h:02d}h]']
        def dm(name: str):
            tok = f'[{name}]'
            if tok not in toks: toks.append(tok)

        perf = []
        ml = sum(1 for h, vals in rec.values['map'].items() if h in allowed and any(v < 65 for v in vals))
        if 1 <= ml <= 3: perf.append('[map_low_hours_bin_brief]')
        elif 4 <= ml <= 8: perf.append('[map_low_hours_bin_intermittent]')
        elif 9 <= ml <= 16: perf.append('[map_low_hours_bin_prolonged]')
        elif ml > 16: perf.append('[map_low_hours_bin_persistent]')
        sbp = flat(rec, 'values', 'sbp', allowed)
        if sbp:
            mn = min(sbp)
            if mn < 90: perf.append('[systolic_bp_min_bin_hypotension]')
            elif mn <= 100: perf.append('[systolic_bp_min_bin_low]')
            elif st.age >= 65 and mn <= 109: perf.append('[systolic_bp_min_bin_geriatric_low]')
        if any(v >= 131 for v in flat(rec, 'values', 'hr', allowed)):
            perf.append('[heart_rate_max_bin_extreme_tachycardia]')
        first48_visible = d == 1 and (d < st.current_day or eligible_h == 24)
        if first48_visible:
            if f48['lactate48'] is not None:
                perf.append('[lactate_48h_bin_severe]' if f48['lactate48'] > 5 else '[lactate_48h_bin_elevated]' if f48['lactate48'] > 2 else '')
            if f48['base_deficit48'] is not None:
                bd = f48['base_deficit48']
                perf.append('[base_deficit_48h_bin_severe]' if bd >= 10 else '[base_deficit_48h_bin_moderate]' if bd >= 6 else '[base_deficit_48h_bin_mild]' if bd >= 3 else '')
        perf = [p for p in perf if p]
        if perf: dm('perfusion_shock'); toks += perf

        oxy = []
        vh_set = rec.vent_hours & allowed
        vh = len(vh_set)
        if 0 < vh < 0.5 * eligible_h: oxy.append('[vent_hours_bin_partial_window]')
        elif 0.5 * eligible_h <= vh < eligible_h: oxy.append('[vent_hours_bin_most_window]')
        elif vh == eligible_h and vh > 0: oxy.append('[vent_hours_bin_full_window]')
        vc = vent_count + 1 if vh > 0 else 0
        if vc == 1: oxy.append('[vent_course_bin_first_day]')
        elif 2 <= vc <= 3: oxy.append('[vent_course_bin_early]')
        elif vc >= 4: oxy.append('[vent_course_bin_prolonged]')
        fio = flat(rec, 'values', 'fio2', allowed)
        if fio:
            mx = max(fio)
            if mx > 0.60: oxy.append('[fio2_max_bin_very_high_support]')
            elif mx > 0.40: oxy.append('[fio2_max_bin_high_support]')
        rr_h = sum(1 for h, vals in rec.values['rr'].items() if h in allowed and any(v >= 25 for v in vals))
        if 1 <= rr_h <= 3: oxy.append('[respiratory_rate_high_hours_bin_brief]')
        elif 4 <= rr_h <= 8: oxy.append('[respiratory_rate_high_hours_bin_intermediate]')
        elif rr_h >= 9: oxy.append('[respiratory_rate_high_hours_bin_prolonged]')
        if oxy: dm('oxygenation_ventilation'); toks += oxy

        renal = []
        cr = flat(rec, 'labs', 'creatinine', allowed)
        cur_cr = max(cr) if cr else prev_cr
        if cr and prev_cr is not None:
            mxcr = max(cr)
            if mxcr - prev_cr >= 0.3: renal.append('[creatinine_change_bin_kdigo_delta]')
            if prev_cr > 0 and mxcr / prev_cr >= 1.5: renal.append('[creatinine_ratio_bin_kdigo_ratio]')
        if any(v < 22 for v in flat(rec, 'labs', 'bicarb', allowed)): renal.append('[bicarbonate_min_bin_low]')
        bun = flat(rec, 'labs', 'bun', allowed)
        if bun and cr:
            pos_cr = [v for v in cr if v > 0]
            if pos_cr and max(bun) > 20 and max(bun) / min(pos_cr) > 20:
                renal.append('[bun_creatinine_ratio_bin_prerenal_pattern]')
        if renal: dm('renal_metabolic'); toks += renal

        imm = []
        wbc = flat(rec, 'labs', 'wbc', allowed)
        if any(v > 12 for v in wbc): imm.append('[wbc_bin_high]')
        if any(v < 4 for v in wbc): imm.append('[wbc_bin_low]')
        rbc = sum(v for h, v in rec.rbc_by_hour.items() if h in allowed)
        if rbc > 0: imm.append('[rbc_transfusion_event_present]')
        if imm: dm('immune_hematologic'); toks += imm

        res = []
        if first48_visible and (f48['rbc48_ml'] or 0) > 0: res.append('[rbc_48h_event_present]')
        if res: dm('resuscitation_burden'); toks += res

        dq = []
        cs = sum(len(rec.core_slots[f] & allowed) for f in CORE_VITALS)
        denom = eligible_h * 6
        if cs >= 0.83 * denom: dq.append('[core_vital_slots_dense]')
        elif cs >= 0.5 * denom: dq.append('[core_vital_slots_partial]')
        elif cs > 0: dq.append('[core_vital_slots_sparse]')
        else: dq.append('[core_vital_slots_none]')
        lab_draws = sum(len(vals) for f in ['bicarb','bun','creatinine','wbc'] for h, vals in rec.labs[f].items() if h in allowed)
        if lab_draws == 0: dq.append('[labs_not_drawn]')
        us = len(rec.uop_slots & allowed)
        dq.append('[uop_measured]' if us >= 6 else '[uop_sparse]' if us >= 1 else '[uop_not_measured]')
        dm('data_quality'); toks += dq

        out.append(toks)
        prev_cr = cur_cr
        if vh > 0: vent_count += 1
    return out


def render_hour(hours: list[HourRaw]) -> list[str]:
    out = []
    for i, hr in enumerate(hours):
        rel = i - (RECENT_H - 1)
        parts = [f'[HOUR_REL_{rel}]']
        for f in HOUR_ORDER:
            vals = hr.values.get(f, [])
            if vals:
                v = vals[-1]
                txt = f'{v:.1f}' if f == 'temp' else f'{v:.2f}' if f == 'fio2' else f'{v:.0f}'
                parts += [HOUR_TOK[f], f'<{txt}>']
        if hr.vent: parts.append('[vent_on]')
        if hr.rbc_ml > 0: parts += ['[rbc_transfusion_1h]', f'<{hr.rbc_ml:.0f}>']
        if rel == 0: parts.append('[CUR]')
        parts.append('[SEP]')
        out.append(' '.join(parts))
    return out


def metrics(seq: str) -> dict[str, Any]:
    hours = [l for l in seq.splitlines() if l.startswith('[HOUR_REL')]
    days = [l for l in seq.splitlines() if l.startswith('[DAY_REL')]
    fields = ['heart_rate','systolic_bp','diastolic_bp','mean_arterial_pressure','respiratory_rate','temperature','fio2']
    return {
        'day_blocks': len(days), 'hour_blocks': len(hours),
        'window_tokens': len(re.findall(r'\[day_window_len_\d\dh\]', seq)),
        'first48_counts': {'lactate_48h': seq.count('lactate_48h'), 'base_deficit_48h': seq.count('base_deficit_48h'), 'rbc_48h_event_present': seq.count('rbc_48h_event_present')},
        'ed_linkage_yes': '[ed_linkage_yes]' in seq,
        'rr_burden_token': 'respiratory_rate_high_hours_bin' in seq,
        'old_token_present': any(x in seq for x in ['vent_hours_bin_full_day','respiratory_rate_max_bin_high','bun_creatinine_ratio_bin_prerenal]']),
        'hour_field_counts': {f: sum(1 for h in hours if f'[{f}]' in h) for f in fields},
        'vent_on_hours': seq.count('[vent_on]'),
    }


def main() -> None:
    stays = load_stays()
    print(f'loaded selected stays: {len(stays)}')
    day_records, hour_records, scan_counts = scan_raw(stays)
    admissions, ed_map, head_hadm = load_static_maps(stays)
    samples = []
    token_counter = Counter()
    for st in stays.values():
        static = build_static(st, admissions, ed_map, head_hadm)
        day_tokens = build_day_tokens(st, day_records)
        hour_lines = render_hour(hour_records[st.stay_id])
        seq_lines = ['```text', ' '.join(static), '']
        for i, toks in enumerate(day_tokens):
            rel = i - st.current_day
            seq_lines.append(f'[DAY_REL_{rel}] ' + ' '.join(toks) + ' [SEP]')
            seq_lines.append('')
        seq_lines += hour_lines
        seq_lines.append('```')
        seq = '\n'.join(seq_lines)
        token_counter.update(re.findall(r'\[[^\]]+\]', seq))
        samples.append({
            'sample_label': st.sample_label, 'subject_id': st.subject_id, 'hadm_id': st.hadm_id, 'stay_id': st.stay_id,
            'completed_day_count': st.ndays, 'current_day_index': st.current_day, 'pred_hour_in_day': st.pred_hour,
            'observed_until': str(st.observed_until), 'source_tables': SOURCE_TABLES, 'sequence': seq, 'metrics': metrics(seq),
        })

    with OUT_JSONL.open('w', encoding='utf-8') as f:
        for s in samples: f.write(json.dumps(s, ensure_ascii=False) + '\n')

    lines = ['# Complete History Samples — STATIC + DAY windows + HOUR', '',
             'Generated from selected C4 trauma cohort stays and official MIMIC-IV raw tables.', '',
             '## Raw scan counts', '', '| stream | matched rows/events |', '|---|---:|']
    for k, v in scan_counts.items(): lines.append(f'| `{k}` | {v} |')
    lines += ['', '## Validation summary', '', '| Sample | HADM | Stay | Anchor | DAY | HOUR | ED | FIRST48≤1 | RR burden | Old tokens |', '|---|---:|---:|---|---:|---:|---|---|---|---|']
    for s in samples:
        m = s['metrics']; f48 = all(v <= 1 for v in m['first48_counts'].values())
        lines.append(f"| `{s['sample_label']}` | {s['hadm_id']} | {s['stay_id']} | day {s['current_day_index']} / h{s['pred_hour_in_day']} | {m['day_blocks']} | {m['hour_blocks']} | {'yes' if m['ed_linkage_yes'] else 'no'} | {'yes' if f48 else 'NO'} | {'yes' if m['rr_burden_token'] else 'no'} | {'FAIL' if m['old_token_present'] else 'PASS'} |")
    lines += ['', '## Interpretation check', '', '- All samples are computed from source tables listed in JSONL; no free-text fabrication.', '- DAY blocks provide longitudinal burden and current partial-day summary through `[day_window_len_XXh]`.', '- HOUR is review rendering; model-side input remains fixed `vital_values[T,7]` and `vital_mask[T,7]`.', '- FIRST48 tokens appear at most once per sample; RR is duration-burden when present.', '', '## Token coverage highlights', '']
    for tok, cnt in token_counter.most_common(35): lines.append(f'- `{tok}`: {cnt}')
    lines.append('')
    for s in samples:
        lines += [f"## {s['sample_label']} — hadm {s['hadm_id']} / stay {s['stay_id']}", '', f"Anchor: ICU day {s['current_day_index']} hour {s['pred_hour_in_day']}; observed_until `{s['observed_until']}`", '', s['sequence'], '']
    OUT_MD.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'outputs': [str(OUT_JSONL), str(OUT_MD)], 'n_samples': len(samples), 'scan_counts': scan_counts}, indent=2))


if __name__ == '__main__':
    main()
