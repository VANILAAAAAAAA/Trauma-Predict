#!/usr/bin/env python3
"""LEGACY numeric sketch for sparse DAY summaries.

Do not use this as the current DAY/REPORT mapper. It emits token + numeric
placeholder pairs and predates the 2026-06-14 global contract that DAY/REPORT
are token-prediction tasks.

Use instead:
  Input design/summary design/day_rule_mapper.py

That mapper emits gate/bucket/status tokens only, with no numeric projection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Evidence-backed / candidate gates.
# IDs refer to tokenizer/reference/day_summary_evidence_chain.md.
DEFAULT_THRESHOLDS = {
    # Hemodynamic
    "map_low_mmHg": 65.0,          # HEM-01, HEM-02
    "sbp_low_mmHg": 100.0,         # HEM-06 qSOFA screening context
    "sbp_geriatric_mmHg": 110.0,   # HEM-05 geriatric trauma context
    "geriatric_age_years": 65.0,
    "hr_high_bpm": 131.0,          # HEM-03 NEWS2 high-risk threshold
    "hr_low_bpm": 40.0,            # HEM-03; current lean token set has no HR-min token
    # Respiratory
    "rr_high_bpm": 25.0,           # RESP-01 NEWS2 high-risk threshold
    "rr_low_bpm": 8.0,             # RESP-01; current lean token set has no RR-min token
    "fio2_high_fraction": 0.40,    # RESP-04 candidate; indirect evidence only
}


@dataclass
class TokenRecord:
    token: str
    value: float
    unit: str
    gate: str
    evidence: List[str]
    evidence_level: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "value": self.value,
            "unit": self.unit,
            "gate": self.gate,
            "evidence": self.evidence,
            "evidence_level": self.evidence_level,
        }


@dataclass
class DaySummary:
    day_rel: int
    source_day_index: int
    hour_start: int
    hour_end: int
    domains: Dict[str, List[TokenRecord]]
    warnings: List[str]

    def token_sequence(self) -> List[str]:
        seq = [f"[DAY_REL_{self.day_rel}]"]
        for domain in ("hemodynamic", "respiratory"):
            records = self.domains.get(domain, [])
            if not records:
                continue
            seq.append(f"[{domain}]")
            for rec in records:
                seq.extend([rec.token, f"<{format_number(rec.value)}>" ])
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


def format_number(x: float) -> str:
    if math.isfinite(x) and abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.4g}"


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
    """Return FiO2 as fraction in [0, 1] when possible."""
    x = to_float(value)
    if x is None:
        return None
    # MIMIC/UW may carry either fraction (0.21) or percent (21-100).
    if 1.0 < x <= 100.0:
        x = x / 100.0
    if x < 0:
        return None
    return x


def truthy_positive(value: Any) -> bool:
    x = to_float(value)
    return x is not None and x > 0


def read_hourly_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return []
    if "hour_index" not in rows[0]:
        raise ValueError("hourly CSV must contain hour_index")
    rows.sort(key=lambda r: int(float(r["hour_index"])))
    return rows


def group_completed_days(
    rows: Sequence[Dict[str, Any]],
    cutoff_hour: Optional[int] = None,
    recent_hours: int = 24,
    max_day_blocks: int = 13,
) -> List[Tuple[int, List[Dict[str, Any]]]]:
    """Return full 24h days before the recent HOUR window.

    If cutoff_hour is the current landmark hour, hours in
    [cutoff_hour - recent_hours + 1, cutoff_hour] are reserved for HOUR blocks.
    DAY blocks are completed 24h chunks before that recent window.
    """
    if not rows:
        return []
    hour_values = [int(float(r["hour_index"])) for r in rows]
    if cutoff_hour is None:
        cutoff_hour = max(hour_values)
    recent_start = cutoff_hour - recent_hours + 1
    completed_rows = [r for r in rows if int(float(r["hour_index"])) < recent_start]

    by_day: Dict[int, List[Dict[str, Any]]] = {}
    for r in completed_rows:
        h = int(float(r["hour_index"]))
        day_idx = h // 24
        by_day.setdefault(day_idx, []).append(r)

    full_days = []
    for day_idx, day_rows in sorted(by_day.items()):
        hours = {int(float(r["hour_index"])) for r in day_rows}
        expected = set(range(day_idx * 24, day_idx * 24 + 24))
        if expected.issubset(hours):
            # keep rows in hour order and only the exact 24h day window
            exact_rows = [r for r in sorted(day_rows, key=lambda x: int(float(x["hour_index"]))) if int(float(r["hour_index"])) in expected]
            full_days.append((day_idx, exact_rows))
    return full_days[-max_day_blocks:]


def values(rows: Sequence[Dict[str, Any]], field: str, normalizer=to_float) -> List[float]:
    out: List[float] = []
    for r in rows:
        if field not in r:
            continue
        x = normalizer(r.get(field))
        if x is not None:
            out.append(x)
    return out


def count_where(xs: Iterable[float], predicate) -> int:
    return sum(1 for x in xs if predicate(x))


def build_hemodynamic(
    day_rows: Sequence[Dict[str, Any]],
    age: Optional[float],
    thresholds: Dict[str, float],
) -> Tuple[List[TokenRecord], List[str]]:
    records: List[TokenRecord] = []
    warnings: List[str] = []

    map_values = values(day_rows, "map")
    sbp_values = values(day_rows, "sbp")
    hr_values = values(day_rows, "hr")

    map_low_hours = count_where(map_values, lambda x: x < thresholds["map_low_mmHg"])
    if map_low_hours > 0:
        records.append(TokenRecord(
            token="[map_low_hours]",
            value=float(map_low_hours),
            unit="hours",
            gate=f"count(map < {thresholds['map_low_mmHg']} mmHg) > 0",
            evidence=["HEM-01", "HEM-02"],
            evidence_level="G",
        ))

    if sbp_values:
        sbp_min = min(sbp_values)
        sbp_gate = None
        evidence = []
        if sbp_min <= thresholds["sbp_low_mmHg"]:
            sbp_gate = f"systolic_bp_min <= {thresholds['sbp_low_mmHg']} mmHg"
            evidence = ["HEM-06"]
        if age is not None and age >= thresholds["geriatric_age_years"] and sbp_min < thresholds["sbp_geriatric_mmHg"]:
            if sbp_gate:
                sbp_gate += f" OR age >= {thresholds['geriatric_age_years']} and systolic_bp_min < {thresholds['sbp_geriatric_mmHg']} mmHg"
                evidence.append("HEM-05")
            else:
                sbp_gate = f"age >= {thresholds['geriatric_age_years']} and systolic_bp_min < {thresholds['sbp_geriatric_mmHg']} mmHg"
                evidence = ["HEM-05"]
        if sbp_gate:
            records.append(TokenRecord(
                token="[systolic_bp_min]",
                value=float(sbp_min),
                unit="mmHg",
                gate=sbp_gate,
                evidence=sorted(set(evidence)),
                evidence_level="G/T",
            ))

    if hr_values:
        hr_max = max(hr_values)
        hr_min = min(hr_values)
        if hr_max >= thresholds["hr_high_bpm"]:
            records.append(TokenRecord(
                token="[heart_rate_max]",
                value=float(hr_max),
                unit="bpm",
                gate=f"heart_rate_max >= {thresholds['hr_high_bpm']} bpm",
                evidence=["HEM-03"],
                evidence_level="G",
            ))
        if hr_min <= thresholds["hr_low_bpm"]:
            warnings.append(
                f"HR low gate fired (min={format_number(hr_min)} <= {format_number(thresholds['hr_low_bpm'])}) but lean V1 has no [heart_rate_min] token."
            )

    return records, warnings


def last_non_missing(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    return xs[-1]


def build_respiratory(
    day_rows: Sequence[Dict[str, Any]],
    thresholds: Dict[str, float],
    vent_day_index: int,
) -> Tuple[List[TokenRecord], List[str]]:
    records: List[TokenRecord] = []
    warnings: List[str] = []

    vent_hours = sum(1 for r in day_rows if truthy_positive(r.get("vent_h")))
    fio2_values = values(day_rows, "fio2", normalize_fio2)
    rr_values = values(day_rows, "rr")

    if vent_hours > 0:
        records.append(TokenRecord(
            token="[vent_hours]",
            value=float(vent_hours),
            unit="hours",
            gate="vent_h active hours > 0",
            evidence=["RESP-03"],
            evidence_level="G/T",
        ))
        # Keep as optional burden marker only when ventilation exists.
        records.append(TokenRecord(
            token="[vent_day_index]",
            value=float(vent_day_index),
            unit="ventilated_days_so_far",
            gate="vent_h active hours > 0",
            evidence=["RESP-03"],
            evidence_level="G/T",
        ))

    if fio2_values:
        fio2_max = max(fio2_values)
        if fio2_max > thresholds["fio2_high_fraction"]:
            records.append(TokenRecord(
                token="[fio2_max]",
                value=float(fio2_max),
                unit="fraction",
                gate=f"fio2_max > {thresholds['fio2_high_fraction']} (candidate oxygen-demand gate)",
                evidence=["RESP-04"],
                evidence_level="C",
            ))

    if rr_values:
        rr_max = max(rr_values)
        rr_min = min(rr_values)
        if rr_max >= thresholds["rr_high_bpm"]:
            records.append(TokenRecord(
                token="[respiratory_rate_max]",
                value=float(rr_max),
                unit="breaths/min",
                gate=f"respiratory_rate_max >= {thresholds['rr_high_bpm']} breaths/min",
                evidence=["RESP-01"],
                evidence_level="G",
            ))
        if rr_min <= thresholds["rr_low_bpm"]:
            warnings.append(
                f"RR low gate fired (min={format_number(rr_min)} <= {format_number(thresholds['rr_low_bpm'])}) but lean V1 has no [respiratory_rate_min] token."
            )

    return records, warnings


def build_day_summaries(
    rows: Sequence[Dict[str, Any]],
    cutoff_hour: Optional[int] = None,
    recent_hours: int = 24,
    max_day_blocks: int = 13,
    age: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> List[DaySummary]:
    thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    grouped = group_completed_days(rows, cutoff_hour, recent_hours, max_day_blocks)
    n = len(grouped)
    summaries: List[DaySummary] = []
    vent_days_so_far = 0

    for idx, (source_day_index, day_rows) in enumerate(grouped):
        # Relative numbering: last completed day before recent HOUR window is -1.
        day_rel = -(n - idx)
        domains: Dict[str, List[TokenRecord]] = {}
        warnings: List[str] = []

        hemo_records, hemo_warnings = build_hemodynamic(day_rows, age, thresholds)
        if hemo_records:
            domains["hemodynamic"] = hemo_records
        warnings.extend(hemo_warnings)

        day_has_vent = any(truthy_positive(r.get("vent_h")) for r in day_rows)
        if day_has_vent:
            vent_days_so_far += 1
        resp_records, resp_warnings = build_respiratory(day_rows, thresholds, vent_days_so_far if day_has_vent else vent_days_so_far)
        if resp_records:
            domains["respiratory"] = resp_records
        warnings.extend(resp_warnings)

        hours = [int(float(r["hour_index"])) for r in day_rows]
        summaries.append(DaySummary(
            day_rel=day_rel,
            source_day_index=source_day_index,
            hour_start=min(hours),
            hour_end=max(hours),
            domains=domains,
            warnings=warnings,
        ))
    return summaries


def summaries_to_payload(summaries: Sequence[DaySummary]) -> Dict[str, Any]:
    return {
        "schema": "ehrpredict_day_summary_v1_hemodynamic_respiratory_only",
        "domains_implemented": ["hemodynamic", "respiratory"],
        "reference": "tokenizer/reference/day_summary_evidence_chain.md",
        "summaries": [s.as_dict() for s in summaries],
    }


def run_self_test() -> None:
    # 3 completed days + 24h recent window. Only first three days summarized.
    rows: List[Dict[str, Any]] = []
    for h in range(96):
        day = h // 24
        r = {
            "hour_index": h,
            "hr": 88,
            "sbp": 120,
            "map": 80,
            "rr": 18,
            "fio2": 0.21,
            "vent_h": 0,
        }
        if day == 1:
            if h % 24 in {0, 1, 2}:
                r["map"] = 60
            r["sbp"] = 95
            r["hr"] = 135 if h % 24 == 5 else 110
        if day == 2:
            r["vent_h"] = 1
            r["fio2"] = 0.50
            r["rr"] = 26 if h % 24 == 10 else 20
        rows.append(r)

    summaries = build_day_summaries(rows, cutoff_hour=95, recent_hours=24, age=70)
    assert len(summaries) == 3, len(summaries)
    assert not summaries[0].domains, summaries[0].as_dict()
    hemo = summaries[1].domains.get("hemodynamic", [])
    assert [r.token for r in hemo] == ["[map_low_hours]", "[systolic_bp_min]", "[heart_rate_max]"], [r.as_dict() for r in hemo]
    resp = summaries[2].domains.get("respiratory", [])
    assert [r.token for r in resp] == ["[vent_hours]", "[vent_day_index]", "[fio2_max]", "[respiratory_rate_max]"], [r.as_dict() for r in resp]
    print(json.dumps({"self_test": "passed", "summaries": [s.as_dict() for s in summaries]}, ensure_ascii=False, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build sparse DAY summary tokens for hemodynamic + respiratory domains.")
    parser.add_argument("--hourly-csv", type=Path, help="Canonical hourly CSV with hour_index/hr/sbp/map/rr/fio2/vent_h columns.")
    parser.add_argument("--cutoff-hour", type=int, default=None, help="Current landmark hour. Default: max hour_index in CSV.")
    parser.add_argument("--recent-hours", type=int, default=24, help="Hours reserved for recent HOUR blocks.")
    parser.add_argument("--max-day-blocks", type=int, default=13, help="Maximum completed DAY blocks to emit.")
    parser.add_argument("--age", type=float, default=None, help="Optional age for geriatric SBP gate.")
    parser.add_argument("--tokens-only", action="store_true", help="Print token sequences only.")
    parser.add_argument("--self-test", action="store_true", help="Run built-in gate tests.")
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_test()
        return 0

    if args.hourly_csv is None:
        parser.error("--hourly-csv is required unless --self-test is used")

    rows = read_hourly_csv(args.hourly_csv)
    summaries = build_day_summaries(
        rows,
        cutoff_hour=args.cutoff_hour,
        recent_hours=args.recent_hours,
        max_day_blocks=args.max_day_blocks,
        age=args.age,
    )
    if args.tokens_only:
        for s in summaries:
            print(" ".join(s.token_sequence()))
    else:
        print(json.dumps(summaries_to_payload(summaries), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
