#!/usr/bin/env python3
"""Map DAY summary rules to bucket/status tokens only.

This is a rule mapper, not a numeric-value projector. It emits sparse DAY tokens
for completed 24h blocks before the recent HOUR window.

Global contract:
  - HOUR numeric projection is limited to 7 vitals: hr, sbp, dbp, map, rr, temp, fio2.
  - DAY/REPORT are token-prediction targets.
  - Non-vital numeric values are converted to evidence-backed or candidate bucket/status tokens.
  - Gate tokens never carry numeric values and never receive numeric projection.

References:
  - tokenizer/reference/summary_gate_rules.md
  - tokenizer/reference/bucket_evidence_review.md
  - tokenizer/reference/uw_cat_thresholds.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


VITAL_FIELDS = ("hr", "sbp", "dbp", "map", "rr", "temp", "fio2")
DOMAIN_ORDER = (
    "hemodynamic",
    "respiratory",
    "renal_output",
    "metabolic",
    "inflammatory_hematologic",
    "treatment_resuscitation",
    "data_quality",
)

# Rule thresholds. Duration/count buckets without direct clinical evidence are kept
# as coarse burden tokens and should be audited in sparsity reports before freezing.
THRESHOLDS = {
    "map_low": 65.0,
    "sbp_low": 100.0,
    "sbp_trauma_hypotension": 90.0,
    "sbp_geriatric": 110.0,
    "geriatric_age": 65.0,
    "hr_high": 131.0,
    "rr_high": 25.0,
    "fio2_high": 0.40,
    "fio2_very_high": 0.60,
    "creatinine_delta": 0.3,
    "creatinine_ratio": 1.5,
    "uop_kdigo_ml_per_kg_h": 0.5,
    "uop_kdigo_consecutive_h": 6,
    "lactate_high": 2.0,
    "bicarbonate_low": 22.0,
    "wbc_low_k_per_uL": 4.0,
    "wbc_high_k_per_uL": 12.0,
}

LAB_FIELDS = (
    "bicarb",
    "bicarbonate",
    "bun",
    "creatinine",
    "wbc",
    "lactate",
    "lactate_48",
    "lactate48",
    "base_def_48",
    "baseDef48",
    "base_deficit",
    "hemoglobin",
    "hgb",
    "platelet",
    "platelets",
)


@dataclass
class EmittedToken:
    token: str
    rule: str
    reliability: str
    status: str = "enabled"
    audit: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "rule": self.rule,
            "reliability": self.reliability,
            "status": self.status,
            "audit": self.audit,
        }


@dataclass
class DayMappedSummary:
    day_rel: int
    source_day_index: int
    hour_start: int
    hour_end: int
    domains: Dict[str, List[EmittedToken]]
    warnings: List[str]

    def token_sequence(self) -> List[str]:
        seq = [f"[DAY_REL_{self.day_rel}]"]
        for domain in DOMAIN_ORDER:
            records = self.domains.get(domain, [])
            if not records:
                continue
            seq.append(f"[{domain}]")
            seq.extend(r.token for r in records)
        seq.append("[SEP]")
        return seq

    def as_dict(self) -> Dict[str, Any]:
        return {
            "day_rel": self.day_rel,
            "source_day_index": self.source_day_index,
            "hour_start": self.hour_start,
            "hour_end": self.hour_end,
            "domains": {
                k: [r.as_dict() for r in v]
                for k, v in self.domains.items()
                if v
            },
            "warnings": self.warnings,
            "sequence": self.token_sequence(),
        }


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        x = float(value)
    else:
        s = str(value).strip()
        if s == "" or s.lower() in {"na", "nan", "none", "null"}:
            return None
        try:
            x = float(s)
        except ValueError:
            return None
    if not math.isfinite(x):
        return None
    return x


def normalize_fio2(value: Any) -> Optional[float]:
    x = to_float(value)
    if x is None:
        return None
    if 1.0 < x <= 100.0:
        x = x / 100.0
    if x < 0:
        return None
    return x


def first_present(row: Dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row:
            return row.get(name)
    return None


def truthy_positive(value: Any) -> bool:
    x = to_float(value)
    return x is not None and x > 0


def values(rows: Sequence[Dict[str, Any]], names: Sequence[str], normalizer=to_float) -> List[float]:
    out: List[float] = []
    for row in rows:
        raw = first_present(row, names)
        x = normalizer(raw)
        if x is not None:
            out.append(x)
    return out


def count_present(rows: Sequence[Dict[str, Any]], names: Sequence[str], normalizer=to_float) -> int:
    return len(values(rows, names, normalizer=normalizer))


def longest_true_run(flags: Sequence[bool]) -> int:
    best = cur = 0
    for flag in flags:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def burden_bucket(prefix: str, count: int) -> Optional[str]:
    if count <= 0:
        return None
    if count <= 3:
        return f"[{prefix}_bin_1_3h]"
    if count <= 8:
        return f"[{prefix}_bin_4_8h]"
    if count <= 16:
        return f"[{prefix}_bin_9_16h]"
    return f"[{prefix}_bin_gt_16h]"


def vent_hours_bucket(count: int) -> Optional[str]:
    if count <= 0:
        return None
    if count < 12:
        return "[vent_hours_bin_1_11h]"
    if count < 24:
        return "[vent_hours_bin_12_23h]"
    return "[vent_hours_bin_24h]"


def vent_day_index_bucket(idx: int) -> str:
    if idx <= 1:
        return "[vent_day_index_bin_day_1]"
    if idx <= 3:
        return "[vent_day_index_bin_day_2_3]"
    return "[vent_day_index_bin_ge_4]"


def daily_amount_from_event_or_cumulative(
    rows: Sequence[Dict[str, Any]],
    event_names: Sequence[str],
    cumulative_names: Sequence[str],
) -> Tuple[Optional[float], str]:
    event_vals = values(rows, event_names)
    if event_vals:
        return sum(x for x in event_vals if x > 0), "event_sum"
    cum_vals = values(rows, cumulative_names)
    if len(cum_vals) >= 2:
        diffs = [max(0.0, cum_vals[i] - cum_vals[i - 1]) for i in range(1, len(cum_vals))]
        return sum(diffs), "positive_cumulative_delta_within_day"
    if len(cum_vals) == 1:
        return 0.0, "single_cumulative_value_no_delta"
    return None, "missing"


def read_hourly_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    if "hour_index" not in rows[0] and "hourTally" not in rows[0]:
        raise ValueError("hourly CSV needs hour_index or hourTally")
    for row in rows:
        if "hour_index" not in row and "hourTally" in row:
            # UW hourTally usually starts at 1; keep relative order and use zero-based internally.
            h = to_float(row["hourTally"])
            row["hour_index"] = "" if h is None else str(int(h) - 1)
    rows.sort(key=lambda r: int(float(r["hour_index"])))
    return rows


def group_completed_days(
    rows: Sequence[Dict[str, Any]],
    cutoff_hour: Optional[int],
    recent_hours: int,
    max_day_blocks: int,
) -> List[Tuple[int, List[Dict[str, Any]]]]:
    if not rows:
        return []
    hour_values = [int(float(r["hour_index"])) for r in rows]
    if cutoff_hour is None:
        cutoff_hour = max(hour_values)
    recent_start = cutoff_hour - recent_hours + 1
    completed = [r for r in rows if int(float(r["hour_index"])) < recent_start]

    by_day: Dict[int, List[Dict[str, Any]]] = {}
    for r in completed:
        h = int(float(r["hour_index"]))
        by_day.setdefault(h // 24, []).append(r)

    full_days: List[Tuple[int, List[Dict[str, Any]]]] = []
    for day_idx, day_rows in sorted(by_day.items()):
        day_rows = sorted(day_rows, key=lambda r: int(float(r["hour_index"])))
        hours = {int(float(r["hour_index"])) for r in day_rows}
        expected = set(range(day_idx * 24, day_idx * 24 + 24))
        if expected.issubset(hours):
            full_days.append((day_idx, [r for r in day_rows if int(float(r["hour_index"])) in expected]))
    return full_days[-max_day_blocks:]


def emit_hemodynamic(rows: Sequence[Dict[str, Any]], age: Optional[float]) -> List[EmittedToken]:
    out: List[EmittedToken] = []
    map_vals = values(rows, ("map", "MAP"))
    sbp_vals = values(rows, ("sbp", "SBP", "systolic_bp"))
    hr_vals = values(rows, ("hr", "heart_rate"))

    map_low = sum(1 for x in map_vals if x < THRESHOLDS["map_low"])
    tok = burden_bucket("map_low_hours", map_low)
    if tok:
        out.append(EmittedToken(
            token=tok,
            rule="MAP <65 mmHg for at least one observed hour",
            reliability="strong gate / candidate burden bucket",
            audit={"map_low_hours": map_low},
        ))

    if sbp_vals:
        sbp_min = min(sbp_vals)
        if sbp_min < THRESHOLDS["sbp_trauma_hypotension"]:
            token = "[systolic_bp_min_bin_lt_90]"
        elif sbp_min <= THRESHOLDS["sbp_low"]:
            token = "[systolic_bp_min_bin_90_100]"
        elif age is not None and age >= THRESHOLDS["geriatric_age"] and sbp_min < THRESHOLDS["sbp_geriatric"]:
            token = "[systolic_bp_min_bin_101_109_geriatric]"
        else:
            token = ""
        if token:
            out.append(EmittedToken(
                token=token,
                rule="daily minimum SBP crosses low/geriatric trauma gate",
                reliability="moderate-plus",
                audit={"systolic_bp_min": sbp_min, "age": age},
            ))

    if hr_vals:
        hr_max = max(hr_vals)
        if hr_max >= THRESHOLDS["hr_high"]:
            out.append(EmittedToken(
                token="[heart_rate_max_bin_ge_131]",
                rule="daily maximum HR >=131 bpm",
                reliability="strong",
                audit={"heart_rate_max": hr_max},
            ))
    return out


def emit_respiratory(rows: Sequence[Dict[str, Any]], vent_day_index: Optional[int]) -> List[EmittedToken]:
    out: List[EmittedToken] = []
    rr_vals = values(rows, ("rr", "respiratory_rate"))
    fio2_vals = values(rows, ("fio2", "FiO2"), normalizer=normalize_fio2)
    vent_vals = values(rows, ("vent_h", "vent_on", "vent", "ventilation_status"))

    vent_hours = sum(1 for x in vent_vals if x > 0)
    token = vent_hours_bucket(vent_hours)
    if token:
        out.append(EmittedToken(
            token=token,
            rule="invasive ventilation active for at least one hour in the day",
            reliability="strong gate / candidate burden bucket",
            audit={"vent_hours": vent_hours},
        ))
        if vent_day_index is not None:
            out.append(EmittedToken(
                token=vent_day_index_bucket(vent_day_index),
                rule="visible-history ventilation day index bucket",
                reliability="candidate longitudinal burden token",
                audit={"vent_day_index": vent_day_index},
            ))

    if fio2_vals:
        fio2_max = max(fio2_vals)
        if fio2_max > THRESHOLDS["fio2_very_high"]:
            token = "[fio2_max_bin_gt_0_60]"
        elif fio2_max > THRESHOLDS["fio2_high"]:
            token = "[fio2_max_bin_0_41_0_60]"
        else:
            token = ""
        if token:
            out.append(EmittedToken(
                token=token,
                rule="daily maximum FiO2 >0.40 fraction",
                reliability="candidate",
                audit={"fio2_max_fraction": fio2_max},
            ))

    if rr_vals:
        rr_max = max(rr_vals)
        if rr_max >= THRESHOLDS["rr_high"]:
            out.append(EmittedToken(
                token="[respiratory_rate_max_bin_ge_25]",
                rule="daily maximum RR >=25 /min",
                reliability="strong",
                audit={"respiratory_rate_max": rr_max},
            ))
    return out


def emit_renal(
    rows: Sequence[Dict[str, Any]],
    previous_day_rows: Optional[Sequence[Dict[str, Any]]],
    weight_kg: Optional[float],
    warnings: List[str],
) -> List[EmittedToken]:
    out: List[EmittedToken] = []
    cr_vals = values(rows, ("creatinine", "creat"))
    prev_cr_vals = values(previous_day_rows or [], ("creatinine", "creat"))
    if cr_vals and prev_cr_vals:
        cr_max = max(cr_vals)
        prev_cr_max = max(prev_cr_vals)
        delta = cr_max - prev_cr_max
        ratio = cr_max / prev_cr_max if prev_cr_max > 0 else None
        if delta >= THRESHOLDS["creatinine_delta"]:
            out.append(EmittedToken(
                token="[creatinine_change_bin_ge_0_3]",
                rule="creatinine increase >=0.3 mg/dL vs previous completed day max",
                reliability="strong",
                audit={"creatinine_max": cr_max, "previous_creatinine_max": prev_cr_max, "delta": delta},
            ))
        if ratio is not None and ratio >= THRESHOLDS["creatinine_ratio"]:
            out.append(EmittedToken(
                token="[creatinine_ratio_bin_ge_1_5]",
                rule="creatinine >=1.5x previous completed day max",
                reliability="strong",
                audit={"creatinine_max": cr_max, "previous_creatinine_max": prev_cr_max, "ratio": ratio},
            ))

    uop_vals = values(rows, ("uop", "urine_output", "urine_output_ml"))
    if uop_vals:
        if weight_kg is None:
            warnings.append("uop observed but [urine_output_low_hours_kdigo] disabled because weight_kg is missing")
        else:
            low_threshold = THRESHOLDS["uop_kdigo_ml_per_kg_h"] * weight_kg
            flags = [x < low_threshold for x in uop_vals]
            longest = longest_true_run(flags)
            if longest >= int(THRESHOLDS["uop_kdigo_consecutive_h"]):
                out.append(EmittedToken(
                    token="[urine_output_low_hours_kdigo]",
                    rule="urine output <0.5 mL/kg/h for >=6 consecutive observed hours",
                    reliability="strong when weight exists / source-dependent otherwise",
                    audit={"weight_kg": weight_kg, "low_uop_ml_per_h_threshold": low_threshold, "longest_low_run_h": longest},
                ))
    return out


def bucket_base_deficit(x: float) -> str:
    if x < 3:
        return "[base_deficit_48h_bin_normal]"
    if x < 6:
        return "[base_deficit_48h_bin_mild]"
    if x < 10:
        return "[base_deficit_48h_bin_moderate]"
    return "[base_deficit_48h_bin_severe]"


def bucket_lactate(x: float) -> str:
    if x <= 2.9:
        return "[lactate_48h_bin_normal_or_low]"
    if x <= 5.0:
        return "[lactate_48h_bin_elevated]"
    return "[lactate_48h_bin_severe]"


def emit_metabolic(rows: Sequence[Dict[str, Any]], source_day_index: int, warnings: List[str]) -> List[EmittedToken]:
    out: List[EmittedToken] = []
    bicarb_vals = values(rows, ("bicarb", "bicarbonate"))
    if bicarb_vals:
        bicarb_min = min(bicarb_vals)
        if bicarb_min < THRESHOLDS["bicarbonate_low"]:
            out.append(EmittedToken(
                token="[bicarbonate_min_bin_lt_22]",
                rule="daily minimum bicarbonate <22 mEq/L",
                reliability="candidate",
                audit={"bicarbonate_min": bicarb_min},
            ))

    first48_emit_window = source_day_index == 1
    lactate_vals = values(rows, ("lactate_48", "lactate48"))
    base_def_vals = values(rows, ("base_def_48", "baseDef48", "base_deficit_48"))
    if (lactate_vals or base_def_vals) and not first48_emit_window:
        warnings.append("FIRST48 metabolic fields observed but suppressed because source_day_index != 1")
    if first48_emit_window:
        if lactate_vals:
            lactate = max(lactate_vals)
            if lactate > THRESHOLDS["lactate_high"]:
                out.append(EmittedToken(
                    token=bucket_lactate(lactate),
                    rule="lactate_48h high gate; FIRST48 visible only after 48h complete",
                    reliability="strong gate / UW bucket",
                    audit={"lactate_48h": lactate},
                ))
        if base_def_vals:
            base_def = max(base_def_vals)
            if base_def >= 3:
                out.append(EmittedToken(
                    token=bucket_base_deficit(base_def),
                    rule="base_deficit_48h abnormality gate; FIRST48 visible only after 48h complete",
                    reliability="moderate-plus / UW bucket",
                    audit={"base_deficit_48h": base_def},
                ))
    return out


def normalize_wbc_k_per_uL(x: float) -> float:
    # MIMIC/UW often use K/uL; if a source is in cells/mm3, convert large values.
    return x / 1000.0 if x > 1000 else x


def emit_inflammatory(rows: Sequence[Dict[str, Any]]) -> List[EmittedToken]:
    out: List[EmittedToken] = []
    wbc_vals_raw = values(rows, ("wbc", "WBC"))
    wbc_vals = [normalize_wbc_k_per_uL(x) for x in wbc_vals_raw]
    if wbc_vals:
        wbc_max = max(wbc_vals)
        wbc_min = min(wbc_vals)
        if wbc_max > THRESHOLDS["wbc_high_k_per_uL"]:
            out.append(EmittedToken(
                token="[wbc_max_bin_gt_12]",
                rule="WBC >12 K/uL",
                reliability="moderate",
                audit={"wbc_max_k_per_uL": wbc_max},
            ))
        if wbc_min < THRESHOLDS["wbc_low_k_per_uL"]:
            out.append(EmittedToken(
                token="[wbc_min_bin_lt_4]",
                rule="WBC <4 K/uL",
                reliability="moderate",
                audit={"wbc_min_k_per_uL": wbc_min},
            ))
    return out


def emit_treatment(rows: Sequence[Dict[str, Any]], warnings: List[str]) -> List[EmittedToken]:
    out: List[EmittedToken] = []
    rbc_total, rbc_mode = daily_amount_from_event_or_cumulative(
        rows,
        event_names=("rbc_transfusion_1h_ml", "rbc_input_1h_ml"),
        cumulative_names=("rbc_sum_until_h", "RBCsum", "rbcSum", "RBC_sum"),
    )
    if rbc_total is not None and rbc_total > 0:
        out.append(EmittedToken(
            token="[rbc_daily_total_present]",
            rule="RBC transfusion total >0 in the day",
            reliability="moderate-plus",
            audit={"rbc_daily_total_source_value": rbc_total, "source_mode": rbc_mode},
        ))

    bolus_total, bolus_mode = daily_amount_from_event_or_cumulative(
        rows,
        event_names=("bolus_input_1h_ml", "crystalloid_input_1h_ml"),
        cumulative_names=("bolus_sum_until_h", "bolusSum", "crys_sum_until_h"),
    )
    if bolus_total is not None and bolus_total > 0:
        warnings.append(
            "bolus/crystalloid >0 observed but no DAY token emitted; >0 mL is held pending treatment-resuscitation rule redesign"
        )
    return out


def coverage_token(prefix: str, observed_hours: int) -> str:
    if observed_hours >= 22:
        return f"[{prefix}_coverage_bin_complete]"
    if observed_hours >= 12:
        return f"[{prefix}_coverage_bin_partial]"
    if observed_hours >= 1:
        return f"[{prefix}_coverage_bin_sparse]"
    return f"[{prefix}_coverage_bin_none]"


def emit_data_quality(rows: Sequence[Dict[str, Any]], warnings: List[str]) -> List[EmittedToken]:
    out: List[EmittedToken] = []

    vital_hours = 0
    for row in rows:
        if any(to_float(first_present(row, (f,))) is not None for f in ("hr", "sbp", "dbp", "map", "rr", "temp", "fio2")):
            vital_hours += 1
    out.append(EmittedToken(
        token=coverage_token("vital", vital_hours),
        rule="daily vital-sign observed-hour coverage bucket",
        reliability="moderate",
        audit={"vital_observed_hours": vital_hours},
    ))

    output_hours = count_present(rows, ("uop", "urine_output", "urine_output_ml"))
    out.append(EmittedToken(
        token=coverage_token("output", output_hours),
        rule="daily urine/output observed-hour coverage bucket",
        reliability="moderate",
        audit={"output_observed_hours": output_hours},
    ))

    # Lab measurement coverage should use raw measurement indicators/storetime,
    # not carried-forward lab memory. Current processed hourly samples do not
    # expose raw measurement flags, so emit a conservative token only when clear.
    raw_flag_fields = [f for f in rows[0].keys() if f.endswith("_observed") or f.endswith("_measured")]
    lab_raw_fields = [f for f in raw_flag_fields if any(name in f.lower() for name in ("bicarb", "bun", "creatinine", "wbc", "lactate", "base_def", "hgb", "platelet"))]
    if lab_raw_fields:
        domains = set()
        for f in lab_raw_fields:
            if any(truthy_positive(r.get(f)) for r in rows):
                lower = f.lower()
                if "creatinine" in lower or "bun" in lower:
                    domains.add("renal")
                elif "bicarb" in lower or "lactate" in lower or "base_def" in lower:
                    domains.add("metabolic")
                elif "wbc" in lower or "hgb" in lower or "platelet" in lower:
                    domains.add("hematologic")
                else:
                    domains.add(f)
        if len(domains) >= 3:
            token = "[lab_measurement_coverage_bin_multi_domain]"
        elif len(domains) >= 1:
            token = "[lab_measurement_coverage_bin_limited]"
        else:
            token = "[lab_measurement_coverage_bin_none]"
        out.append(EmittedToken(
            token=token,
            rule="raw lab measurement domain coverage bucket",
            reliability="moderate",
            audit={"lab_domains_observed": sorted(domains)},
        ))
    else:
        warnings.append("lab measurement coverage not emitted: current rows lack raw lab observed/measured flags; carried-forward lab values are not used as lab measurement coverage")
    return out


def map_days(
    rows: Sequence[Dict[str, Any]],
    age: Optional[float],
    weight_kg: Optional[float],
    cutoff_hour: Optional[int],
    recent_hours: int,
    max_day_blocks: int,
) -> List[DayMappedSummary]:
    day_groups = group_completed_days(rows, cutoff_hour, recent_hours, max_day_blocks)
    results: List[DayMappedSummary] = []
    vent_days_seen = 0
    prev_rows: Optional[Sequence[Dict[str, Any]]] = None
    n = len(day_groups)

    for i, (day_idx, day_rows) in enumerate(day_groups):
        warnings: List[str] = []
        domains: Dict[str, List[EmittedToken]] = {d: [] for d in DOMAIN_ORDER}
        day_rel = -(n - i)

        vent_vals = values(day_rows, ("vent_h", "vent_on", "vent", "ventilation_status"))
        vent_hours = sum(1 for x in vent_vals if x > 0)
        if vent_hours > 0:
            vent_days_seen += 1
            vent_day_idx = vent_days_seen
        else:
            vent_day_idx = None

        domains["hemodynamic"] = emit_hemodynamic(day_rows, age)
        domains["respiratory"] = emit_respiratory(day_rows, vent_day_idx)
        domains["renal_output"] = emit_renal(day_rows, prev_rows, weight_kg, warnings)
        domains["metabolic"] = emit_metabolic(day_rows, day_idx, warnings)
        domains["inflammatory_hematologic"] = emit_inflammatory(day_rows)
        domains["treatment_resuscitation"] = emit_treatment(day_rows, warnings)
        domains["data_quality"] = emit_data_quality(day_rows, warnings)

        start_h = day_idx * 24
        results.append(DayMappedSummary(
            day_rel=day_rel,
            source_day_index=day_idx,
            hour_start=start_h,
            hour_end=start_h + 23,
            domains=domains,
            warnings=warnings,
        ))
        prev_rows = day_rows
    return results


def build_self_test_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    # day 0: mostly normal, no labs/output -> data_quality only
    # day 1: unstable hemodynamic/respiratory + low bicarbonate/WBC high + RBC delta
    # day 2: FIRST48 visible; lactate/base deficit can emit
    for h in range(72):
        day = h // 24
        row: Dict[str, Any] = {
            "hour_index": str(h),
            "hr": "90",
            "sbp": "120",
            "map": "80",
            "rr": "18",
            "fio2": "0.30",
            "vent_h": "0",
            "creatinine": "1.0",
            "bicarb": "24",
            "wbc": "8",
            "uop": "60",
            "rbc_sum_until_h": "0",
            "bolus_sum_until_h": "0",
            "lactate_48": "",
            "base_def_48": "",
        }
        if day == 1:
            if h % 24 < 10:
                row["map"] = "60"
            row["sbp"] = "95"
            row["hr"] = "140"
            row["rr"] = "28"
            row["fio2"] = "0.50"
            row["vent_h"] = "1"
            row["creatinine"] = "1.5"
            row["bicarb"] = "18"
            row["wbc"] = "15"
            row["rbc_sum_until_h"] = "1" if h >= 30 else "0"
            row["uop"] = "20" if 24 <= h < 31 else "60"
        if day == 1:
            row["lactate_48"] = "5.2"
            row["base_def_48"] = "7"
        rows.append(row)
    return rows


def run_self_test() -> Dict[str, Any]:
    rows = build_self_test_rows()
    summaries = map_days(rows, age=70, weight_kg=70, cutoff_hour=95, recent_hours=24, max_day_blocks=13)
    seqs = [s.token_sequence() for s in summaries]
    flat = [tok for seq in seqs for tok in seq]
    required = {
        "[map_low_hours_bin_9_16h]",
        "[systolic_bp_min_bin_90_100]",
        "[heart_rate_max_bin_ge_131]",
        "[vent_hours_bin_24h]",
        "[fio2_max_bin_0_41_0_60]",
        "[respiratory_rate_max_bin_ge_25]",
        "[creatinine_change_bin_ge_0_3]",
        "[urine_output_low_hours_kdigo]",
        "[bicarbonate_min_bin_lt_22]",
        "[wbc_max_bin_gt_12]",
        "[rbc_daily_total_present]",
        "[vital_coverage_bin_complete]",
        "[output_coverage_bin_complete]",
        "[lactate_48h_bin_severe]",
        "[base_deficit_48h_bin_moderate]",
    }
    missing = sorted(required.difference(flat))
    if missing:
        raise AssertionError(f"missing expected tokens: {missing}")
    return {"self_test": "passed", "summaries": [s.as_dict() for s in summaries]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map DAY summary rules to token-only sparse DAY summaries.")
    parser.add_argument("--hourly-csv", type=Path, help="Canonical hourly CSV with hour_index/hourTally.")
    parser.add_argument("--age", type=float, default=None, help="Age for geriatric SBP gate.")
    parser.add_argument("--weight-kg", type=float, default=None, help="Weight for KDIGO urine-output gate. If absent, uop KDIGO token is disabled.")
    parser.add_argument("--cutoff-hour", type=int, default=None, help="Landmark hour. Default: max hour in CSV.")
    parser.add_argument("--recent-hours", type=int, default=24, help="Recent HOUR window excluded from DAY summaries.")
    parser.add_argument("--max-day-blocks", type=int, default=13, help="Max completed DAY summaries.")
    parser.add_argument("--tokens-only", action="store_true", help="Print one token sequence per DAY.")
    parser.add_argument("--self-test", action="store_true", help="Run built-in rule tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        print(json.dumps(run_self_test(), ensure_ascii=False, indent=2))
        return
    if not args.hourly_csv:
        raise SystemExit("Provide --hourly-csv or --self-test")
    rows = read_hourly_csv(args.hourly_csv)
    summaries = map_days(
        rows,
        age=args.age,
        weight_kg=args.weight_kg,
        cutoff_hour=args.cutoff_hour,
        recent_hours=args.recent_hours,
        max_day_blocks=args.max_day_blocks,
    )
    if args.tokens_only:
        for s in summaries:
            print(" ".join(s.token_sequence()))
    else:
        print(json.dumps([s.as_dict() for s in summaries], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
