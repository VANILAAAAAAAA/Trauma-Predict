#!/usr/bin/env python3
"""Extract a teacher-flow MIMIC-IV trauma cohort.

Layer contract, adapted from the attached MIMIC-III flow to MIMIC-IV:
  C0 MIMIC-IV hospital admissions
  C1 valid admissions: >=1 ICU stay and corresponding CHARTEVENTS data
  C2 trauma admissions: ICD-9 E-code OR ICD-10 external-cause evidence, both by Excel exact allowlist matching (GitHub-style)
  C3 adult trauma: age 18-89
  C4 hospital duration >=48h
  C5 final: mechanical ventilator days >= configured threshold

No pandas/openpyxl dependency. Reads CSV.GZ with stdlib csv+gzip and reads .xlsx
through zipped Office Open XML files.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Set, Tuple

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

DEFAULT_INVASIVE_VENT_MODES = {
    "(S) CMV",
    "APRV",
    "APRV/BIPHASIC+APNPRESS",
    "APRV/BIPHASIC+APNVOL",
    "APV (CMV)",
    "AMBIENT",
    "APNEA VENTILATION",
    "CMV",
    "CMV/ASSIST",
    "CMV/ASSIST/AUTOFLOW",
    "CMV/AUTOFLOW",
    "CPAP/PPS",
    "CPAP/PSV",
    "CPAP/PSV+APN TCPL",
    "CPAP/PSV+APNPRES",
    "CPAP/PSV+APNVOL",
    "MMV",
    "MMV/AUTOFLOW",
    "MMV/PSV",
    "MMV/PSV/AUTOFLOW",
    "P-CMV",
    "PCV+",
    "PCV+/PSV",
    "PCV+ASSIST",
    "PRES/AC",
    "PRVC/AC",
    "PRVC/SIMV",
    "PSV/SBT",
    "SIMV",
    "SIMV/AUTOFLOW",
    "SIMV/PRES",
    "SIMV/PSV",
    "SIMV/PSV/AUTOFLOW",
    "SIMV/VOL",
    "SYNCHRON MASTER",
    "SYNCHRON SLAVE",
    "VOL/AC",
    # Hamilton modes from mimic-code MIMIC-IV ventilation.sql
    "APV (SIMV)",
    "P-SIMV",
    "VS",
    "ASV",
}
DEFAULT_INVASIVE_O2_DEVICES = {"ENDOTRACHEAL TUBE"}
DEFAULT_VENT_MODE_ITEMIDS = {"223849", "229314"}  # Ventilator Mode, Ventilator Mode Hamilton
DEFAULT_O2_DEVICE_ITEMIDS = {"226732"}  # O2 Delivery Device(s)


def require_file(path: str, label: str) -> None:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")


def open_csv(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return open(path, "r", newline="", encoding="utf-8")


def read_csv_dicts(path: str) -> Iterable[dict]:
    with open_csv(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def parse_dt(value: str) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unsupported datetime: {value}")


def normalize_icd(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper().strip())


def normalize_text(value: str) -> str:
    return " ".join((value or "").strip().upper().split())


def xlsx_shared_strings(z: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    out = []
    for si in root.findall("a:si", NS):
        out.append("".join(t.text or "" for t in si.findall(".//a:t", NS)))
    return out


def col_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def xlsx_rows(path: str) -> Iterable[Tuple[str, List[str]]]:
    """Yield (sheet_xml_name, row_values) from all worksheets."""
    with zipfile.ZipFile(path) as z:
        shared = xlsx_shared_strings(z)
        sheets = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
        for sheet in sheets:
            root = ET.fromstring(z.read(sheet))
            for row in root.findall(".//a:sheetData/a:row", NS):
                cells: Dict[int, str] = {}
                max_idx = -1
                for c in row.findall("a:c", NS):
                    ref = c.attrib.get("r", "A1")
                    idx = col_index(ref)
                    max_idx = max(max_idx, idx)
                    ct = c.attrib.get("t", "")
                    if ct == "s":
                        v = c.find("a:v", NS)
                        text = "" if v is None else (v.text or "")
                        if text.isdigit():
                            pos = int(text)
                            text = shared[pos] if pos < len(shared) else text
                    elif ct == "inlineStr":
                        is_el = c.find("a:is", NS)
                        if is_el is not None:
                            t_el = is_el.find("a:t", NS)
                            text = (t_el.text or "") if t_el is not None else ""
                        else:
                            text = ""
                    else:
                        v = c.find("a:v", NS)
                        text = "" if v is None else (v.text or "")
                    cells[idx] = text.strip()
                if max_idx >= 0:
                    yield sheet, [cells.get(i, "") for i in range(max_idx + 1)]


def load_ecode_dictionary(path: str, cohort_config: dict | None = None) -> Tuple[Dict[Tuple[str, str], dict], dict]:
    """Load the clean E-code dictionary.

    The workbook is expected to be pre-normalized:
    - ICD-9 codes are E-prefixed 5-char no-decimal keys (e.g. E8000).
    - ICD-10 codes are no-dot uppercase keys (e.g. T7411XA).
    - No exclude column; excluded rows have already been removed.

    All codes are read as-is and stored as (version, code) keys.
    """
    cohort_config = cohort_config or {}
    allow: Dict[Tuple[str, str], dict] = {}
    code_sets: Dict[str, Set[str]] = {"9": set(), "10": set()}
    stats: Dict[str, Any] = {
        "excel_rows": 0,
        "allowed_excel_rows": 0,
        "excluded_excel_rows": 0,
        "icd9_rows": 0,
        "icd10_rows": 0,
        "icd9_codes": 0,
        "icd10_codes": 0,
        "icd9_excluded_by_excel_column": 0,
        "icd10_excluded_by_excel_column": 0,
        "exclude_column": "",
        "exclude_applies_to": [],
        "sheets": defaultdict(int),
    }
    current_header: Dict[str, int] | None = None
    current_version: str | None = None

    for sheet, values in xlsx_rows(path):
        norm_headers = [v.strip().lower() for v in values]
        if "ecode" in norm_headers:
            current_header = {h: i for i, h in enumerate(norm_headers) if h}
            if "country" in current_header or "term original" in current_header or "intent" in current_header:
                current_version = "10"
            else:
                current_version = "9"
            continue
        if not current_header or "ecode" not in current_header or not current_version:
            continue
        ecode_raw = values[current_header["ecode"]] if current_header["ecode"] < len(values) else ""
        code = normalize_icd(ecode_raw)
        if not code:
            continue

        version = current_version
        stats["excel_rows"] += 1

        desc_idx = current_header.get("term original", current_header.get("description", -1))
        mech_idx = current_header.get("mechanism", -1)
        intent_idx = current_header.get("intent", -1)
        trauma_idx = current_header.get("trauma type", current_header.get("section", -1))
        rec = {
            "excel_sheet": sheet,
            "source_ecode": ecode_raw,
            "description": values[desc_idx].strip() if 0 <= desc_idx < len(values) else "",
            "mechanism": values[mech_idx].strip() if 0 <= mech_idx < len(values) else "",
            "intent": values[intent_idx].strip() if 0 <= intent_idx < len(values) else "",
            "trauma_type": values[trauma_idx].strip() if 0 <= trauma_idx < len(values) else "",
            "excel_exclude": "",
        }
        allow[(version, code)] = rec
        code_sets[version].add(code)
        stats["allowed_excel_rows"] += 1
        stats["sheets"][sheet] += 1
        stats["icd9_rows" if version == "9" else "icd10_rows"] += 1
    stats["sheets"] = dict(stats["sheets"])
    stats["icd9_codes"] = len(code_sets["9"])
    stats["icd10_codes"] = len(code_sets["10"])
    return allow, stats


def read_patients(path: str) -> Dict[str, dict]:
    patients = {}
    for row in read_csv_dicts(path):
        patients[row["subject_id"]] = row
    return patients


def read_admissions(path: str) -> Dict[str, dict]:
    admissions = {}
    for row in read_csv_dicts(path):
        admittime = parse_dt(row["admittime"])
        dischtime = parse_dt(row["dischtime"])
        los_hours = None
        if admittime and dischtime:
            los_hours = (dischtime - admittime).total_seconds() / 3600.0
        row["_admittime_dt"] = admittime
        row["_dischtime_dt"] = dischtime
        row["hospital_los_hours"] = los_hours
        admissions[row["hadm_id"]] = row
    return admissions


def read_icu_stays(path: str) -> Dict[str, List[dict]]:
    stays_by_hadm: Dict[str, List[dict]] = defaultdict(list)
    for row in read_csv_dicts(path):
        row["_intime_dt"] = parse_dt(row["intime"])
        row["_outtime_dt"] = parse_dt(row["outtime"])
        stays_by_hadm[row["hadm_id"]].append(row)
    return stays_by_hadm


def parse_ecode_ranges(range_strings: List[str]) -> List[Tuple[int, int]]:
    ranges = []
    for item in range_strings:
        item = item.strip().upper().replace(" ", "")
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
        else:
            left = right = item
        left_n = int(normalize_icd(left).lstrip("E"))
        right_n = int(normalize_icd(right).lstrip("E"))
        ranges.append((left_n, right_n))
    return ranges


def in_numeric_ranges(value: int, ranges: List[Tuple[int, int]]) -> bool:
    return any(lo <= value <= hi for lo, hi in ranges)


def icd9_ecode_number(code: str) -> int | None:
    code = normalize_icd(code)
    if not code.startswith("E"):
        return None
    digits = code[1:]
    if not digits.isdigit():
        return None
    return int(digits)


def collect_trauma_matches(path: str, allow: Dict[Tuple[str, str], dict], cohort_config: dict) -> Tuple[Dict[str, List[dict]], dict]:
    exclude_ranges = parse_ecode_ranges(cohort_config.get("exclude_icd9_ecode_ranges", []))
    use_icd10 = bool(cohort_config.get("use_icd10", True))

    matches: Dict[str, List[dict]] = defaultdict(list)
    seen = set()
    stats = {
        "diagnosis_rows_scanned": 0,
        "matched_rows": 0,
        "matched_icd9_rows": 0,
        "matched_icd10_rows": 0,
    }
    for row in read_csv_dicts(path):
        stats["diagnosis_rows_scanned"] += 1
        version = str(row.get("icd_version", "")).strip()
        raw_code = row.get("icd_code", "")
        code = normalize_icd(raw_code)
        matched = False
        meta = {}
        if version == "9":
            # Repo-style: require E prefix, pad short codes to 5 chars, exact match
            if not code.startswith("E"):
                continue
            if len(code) < 5:
                code = code + "0"
            num = icd9_ecode_number(code)
            exact_key = ("9", code)
            matched = exact_key in allow and not (num is not None and in_numeric_ranges(num, exclude_ranges))
            meta = allow.get(exact_key, {})
        elif version == "10" and use_icd10:
            matched = ("10", code) in allow
            meta = allow.get(("10", code), {})
        if not matched:
            continue
        hadm_id = row["hadm_id"]
        uniq = (hadm_id, row.get("seq_num", ""), version, code)
        if uniq in seen:
            continue
        seen.add(uniq)
        stats["matched_rows"] += 1
        stats["matched_icd9_rows" if version == "9" else "matched_icd10_rows"] += 1
        matches[hadm_id].append({
            "seq_num": row.get("seq_num", ""),
            "icd_version": version,
            "icd_code": code,
            "source_ecode": meta.get("source_ecode", ""),
            "mechanism": meta.get("mechanism", ""),
            "intent": meta.get("intent", ""),
            "trauma_type": meta.get("trauma_type", ""),
            "description": meta.get("description", ""),
        })
    return matches, stats


def is_invasive_vent_event(row: dict, cohort_config: dict) -> bool:
    itemid = row.get("itemid", "")
    value = normalize_text(row.get("value", ""))
    vent_mode_itemids = set(str(x) for x in cohort_config.get("vent_mode_itemids", DEFAULT_VENT_MODE_ITEMIDS))
    o2_device_itemids = set(str(x) for x in cohort_config.get("o2_device_itemids", DEFAULT_O2_DEVICE_ITEMIDS))
    invasive_modes = {normalize_text(x) for x in cohort_config.get("invasive_vent_modes", sorted(DEFAULT_INVASIVE_VENT_MODES))}
    invasive_o2_devices = {normalize_text(x) for x in cohort_config.get("invasive_o2_devices", sorted(DEFAULT_INVASIVE_O2_DEVICES))}
    if itemid in vent_mode_itemids and value in invasive_modes:
        return True
    if itemid in o2_device_itemids and value in invasive_o2_devices:
        return True
    return False


def scan_chartevents(path: str, cohort_config: dict) -> Tuple[Set[str], Dict[str, Set[str]], dict]:
    """Return hadm IDs with any CHARTEVENTS and invasive ventilation calendar days.

    Ventilator days follow the ML4UWHealth/MIMIC-III pattern: count distinct
    chart dates with a qualifying mechanical ventilation event, regardless of
    hours ventilated on that date. MIMIC-IV event classification follows the
    MIT-LCP mimic-code ventilation concept for InvasiveVent-relevant settings.
    """
    valid_hadms: Set[str] = set()
    vent_days_by_hadm: Dict[str, Set[str]] = defaultdict(set)
    target_itemids = set(str(x) for x in cohort_config.get("vent_mode_itemids", DEFAULT_VENT_MODE_ITEMIDS))
    target_itemids |= set(str(x) for x in cohort_config.get("o2_device_itemids", DEFAULT_O2_DEVICE_ITEMIDS))
    stats = {
        "chartevents_rows_scanned": 0,
        "chartevents_hadm_with_any_row": 0,
        "vent_relevant_rows_scanned": 0,
        "invasive_vent_event_rows": 0,
    }
    for row in read_csv_dicts(path):
        stats["chartevents_rows_scanned"] += 1
        hadm_id = row.get("hadm_id", "")
        if hadm_id:
            valid_hadms.add(hadm_id)
        if row.get("itemid", "") not in target_itemids:
            continue
        stats["vent_relevant_rows_scanned"] += 1
        if hadm_id and is_invasive_vent_event(row, cohort_config):
            charttime = row.get("charttime", "")
            if charttime:
                vent_days_by_hadm[hadm_id].add(charttime[:10])
                stats["invasive_vent_event_rows"] += 1
    stats["chartevents_hadm_with_any_row"] = len(valid_hadms)
    return valid_hadms, vent_days_by_hadm, stats


def age_at_admit(patient: dict, admission: dict) -> int | None:
    try:
        anchor_age = int(float(patient.get("anchor_age", "")))
        anchor_year = int(float(patient.get("anchor_year", "")))
        admit_year = admission["_admittime_dt"].year
        return anchor_age + (admit_year - anchor_year)
    except Exception:
        return None


def stay_count(hadm_set: Set[str], stays_by_hadm: Dict[str, List[dict]]) -> int:
    return sum(len(stays_by_hadm.get(h, [])) for h in hadm_set)


def layer_record(name: str, rule: str, hadm_set: Set[str], prev_set: Set[str] | None, stays_by_hadm: Dict[str, List[dict]]) -> dict:
    return {
        "layer": name,
        "rule": rule,
        "hadm_count": len(hadm_set),
        "stay_count": stay_count(hadm_set, stays_by_hadm),
        "excluded_from_previous_hadm": "" if prev_set is None else len(prev_set - hadm_set),
    }


def write_layers_md(path: str, config: dict, ecode_stats: dict, match_stats: dict, chartevent_stats: dict, layers: List[dict], extra: dict) -> None:
    """Write a paper-style flow report matching the reference inclusion diagram.

    Internal extraction still computes a validity layer before trauma matching, but
    the report presents the first step as the paper does:
        all hospital admissions -> valid trauma patients
    with side exclusions for invalid hospital admissions and non-trauma admissions.
    """
    min_age = int(config["cohort"].get("min_age", 18))
    max_age = int(config["cohort"].get("max_age", 89))
    min_los = float(config["cohort"].get("hospital_los_hours_min", 48))
    vent_threshold = int(config["cohort"].get("mechanical_ventilation_day_min", 3))
    invalid_hospital = int(extra.get("invalid_hospital_admission", layers[0]["hadm_count"] - layers[1]["hadm_count"]))
    non_trauma = int(extra.get("non_trauma_admission", layers[1]["hadm_count"] - layers[2]["hadm_count"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# MIMIC-IV Trauma Cohort — Paper-style Flow  \n\n")
        f.write("Flow style follows the reference MIMIC-III diagram, adapted to MIMIC-IV.  \n")
        f.write(f"E-codes: clean pre-normalized ICD-9/ICD-10 exact allowlist from workbook.  \n")
        f.write("ICD-10 excluded rows were removed before workbook ingestion; extractor applies no workbook exclude column.  \n")
        f.write(f"Age: {min_age}–{max_age} | LOS: &ge;{min_los:g}h | Vent: &ge;{vent_threshold}d  \n\n")

        f.write("## 1. Cohort Flow\n\n")
        f.write("```mermaid\n")
        f.write("flowchart TD\n")
        f.write(f"    C0[\"<b>MIMIC-IV Hospital Admission</b><br/>P = {layers[0]['hadm_count']:,}\"]\n")
        f.write(f"    X0[\"Invalid hospital admission: p = {invalid_hospital:,}<br/>Non-trauma admission: p = {non_trauma:,}\"]\n")
        f.write(f"    C1[\"<b>Valid Trauma Patients</b><br/>(according to ICD-9/10 E Code)<br/>P = {layers[2]['hadm_count']:,}\"]\n")
        f.write(f"    C2[\"<b>Trauma Adult Patients</b><br/>(age in [{min_age}, {max_age}])<br/>P = {layers[3]['hadm_count']:,}\"]\n")
        f.write(f"    X2[\"Hospital Days &lt; {min_los:g}h:<br/>Died (p = {extra['los_lt48_died']:,})<br/>Discharged Alive (p = {extra['los_lt48_alive']:,})\"]\n")
        f.write(f"    C3[\"<b>Hospital Duration &gt;= {min_los:g}h</b><br/>P = {layers[4]['hadm_count']:,}\"]\n")
        f.write(f"    X3[\"Not intubated (p = {extra['not_intubated']:,})<br/>Intubated &lt; {vent_threshold} days (p = {extra['intubated_less_threshold']:,})\"]\n")
        f.write(f"    C4[\"<b>Ventilator Days ≥{vent_threshold}</b><br/><br/>Final Cohort<br/>P = {layers[5]['hadm_count']:,}\"]\n")
        f.write("    C0 --> C1\n")
        f.write("    C0 --> X0\n")
        f.write("    C1 --> C2\n")
        f.write("    C2 --> C3\n")
        f.write("    C2 --> X2\n")
        f.write("    C3 --> C4\n")
        f.write("    C3 --> X3\n")
        f.write("```\n\n")

        f.write("## 2. Paper-style Layer Table\n\n")
        f.write("| Step | Rule | HADM/P | ICU stays in step | Exclusion shown at this step |\n")
        f.write("|---|---|---:|---:|---|\n")
        f.write(f"| Hospital Admission | all MIMIC-IV hospital admissions | {layers[0]['hadm_count']} | {layers[0]['stay_count']} | invalid hospital admission: {invalid_hospital}; non-trauma admission: {non_trauma} |\n")
        f.write(f"| Valid Trauma Patients | valid hospital admission + ICD-9/10 E-code evidence from workbook | {layers[2]['hadm_count']} | {layers[2]['stay_count']} | age excluded: {extra['c2_age_under'] + extra['c2_age_over'] + extra['c2_age_unknown']} |\n")
        f.write(f"| Trauma Adult Patients | age_at_admit between {min_age} and {max_age} | {layers[3]['hadm_count']} | {layers[3]['stay_count']} | hospital days &lt; {min_los:g}h: {extra['los_lt48_total']} |\n")
        f.write(f"| Hospital Duration &gt;= {min_los:g}h | adult trauma + hospital_los_hours >= {min_los:g} | {layers[4]['hadm_count']} | {layers[4]['stay_count']} | not intubated: {extra['not_intubated']}; intubated &lt; {vent_threshold}d: {extra['intubated_less_threshold']} |\n")
        f.write(f"| Ventilator Days ≥{vent_threshold} | hospital duration + invasive ventilator days >= {vent_threshold} | {layers[5]['hadm_count']} | {layers[5]['stay_count']} | final cohort |\n")

        f.write("\n## 3. Internal Mapping and Exclusion Details\n\n")
        f.write("- **E-code definition**: ICD-9 and ICD-10 codes are taken from `qualified_traumatic_Ecodes_clean.xlsx`; ICD-9 is already E-prefixed 5-char no-decimal form, ICD-10 is already no-dot form, and excluded ICD-10 rows were removed before ingestion.\n")
        f.write(f"- **Invalid hospital admission**: {invalid_hospital:,}. In this MIMIC-IV adaptation, this is the paper-style side exclusion before the trauma layer: admissions that do not enter the valid ICU/CHARTEVENTS data layer (no ICU stay: {extra['c0_no_icu']:,}; ICU admission without CHARTEVENTS: {extra['c0_icu_no_chartevents']:,}).\n")
        f.write(f"- **Non-trauma admission**: {non_trauma:,} = valid ICU/CHARTEVENTS admissions without non-excluded ICD-9/10 E-code evidence.\n")
        f.write(f"- **Age exclusion**: total {extra['c2_age_under'] + extra['c2_age_over'] + extra['c2_age_unknown']:,}; &lt;{min_age}: {extra['c2_age_under']}, &gt;{max_age}: {extra['c2_age_over']}, unknown: {extra['c2_age_unknown']}.\n")
        f.write(f"- **Hospital LOS exclusion**: {extra['los_lt48_total']:,}; died: {extra['los_lt48_died']:,}, discharged alive: {extra['los_lt48_alive']:,}.\n")
        f.write(f"- **Ventilator-day exclusion**: never intubated {extra['not_intubated']:,}; intubated &lt; {vent_threshold} days {extra['intubated_less_threshold']:,}.\n")


def write_cohort_csv(
    path: str,
    final_hadms: Set[str],
    admissions: Dict[str, dict],
    patients: Dict[str, dict],
    stays_by_hadm: Dict[str, List[dict]],
    matches: Dict[str, List[dict]],
    vent_days_by_hadm: Dict[str, Set[str]],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "subject_id", "hadm_id", "stay_id", "admittime", "dischtime", "hospital_los_hours", "hospital_expire_flag",
        "age_at_admit", "gender", "anchor_age", "anchor_year", "intime", "outtime", "icu_los_days",
        "vent_day_count", "has_chartevents_data",
        "trauma_icd_codes", "trauma_icd_versions", "trauma_seq_nums", "trauma_mechanisms", "trauma_intents", "trauma_types",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for hadm_id in sorted(final_hadms, key=lambda x: int(x)):
            adm = admissions[hadm_id]
            pat = patients.get(adm["subject_id"], {})
            age = age_at_admit(pat, adm)
            mm = matches.get(hadm_id, [])
            codes = sorted({m["icd_code"] for m in mm})
            versions = sorted({m["icd_version"] for m in mm})
            seqs = sorted({m["seq_num"] for m in mm}, key=lambda s: int(s) if str(s).isdigit() else 9999)
            mechs = sorted({m["mechanism"] for m in mm if m.get("mechanism")})
            intents = sorted({m["intent"] for m in mm if m.get("intent")})
            types = sorted({m["trauma_type"] for m in mm if m.get("trauma_type")})
            for stay in sorted(stays_by_hadm.get(hadm_id, []), key=lambda r: r.get("intime", "")):
                writer.writerow({
                    "subject_id": adm["subject_id"],
                    "hadm_id": hadm_id,
                    "stay_id": stay.get("stay_id", ""),
                    "admittime": adm.get("admittime", ""),
                    "dischtime": adm.get("dischtime", ""),
                    "hospital_los_hours": f"{adm['hospital_los_hours']:.2f}" if adm.get("hospital_los_hours") is not None else "",
                    "hospital_expire_flag": adm.get("hospital_expire_flag", ""),
                    "age_at_admit": age if age is not None else "",
                    "gender": pat.get("gender", ""),
                    "anchor_age": pat.get("anchor_age", ""),
                    "anchor_year": pat.get("anchor_year", ""),
                    "intime": stay.get("intime", ""),
                    "outtime": stay.get("outtime", ""),
                    "icu_los_days": stay.get("los", ""),
                    "vent_day_count": len(vent_days_by_hadm.get(hadm_id, set())),
                    "has_chartevents_data": "1",
                    "trauma_icd_codes": ";".join(codes),
                    "trauma_icd_versions": ";".join(versions),
                    "trauma_seq_nums": ";".join(seqs),
                    "trauma_mechanisms": ";".join(mechs),
                    "trauma_intents": ";".join(intents),
                    "trauma_types": ";".join(types),
                })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    for label, path in config["sources"].items():
        if label == "injury_severity" and not path:
            continue
        require_file(path, label)

    min_age = int(config["cohort"].get("min_age", 18))
    max_age = int(config["cohort"].get("max_age", 89))
    min_los = float(config["cohort"].get("hospital_los_hours_min", 48))
    vent_threshold = int(config["cohort"].get("mechanical_ventilation_day_min", 3))

    print("[1/7] load E-code Excel")
    allow, ecode_stats = load_ecode_dictionary(config["sources"]["ecode_excel"], config["cohort"])
    print(
        f"  loaded {len(allow)} allowed codes: "
        f"ICD9={ecode_stats['icd9_codes']} ICD10={ecode_stats['icd10_codes']} "
        f"excluded_by_excel={ecode_stats.get('excluded_excel_rows', 0)}"
    )

    print("[2/7] load admissions/patients/ICU stays")
    admissions = read_admissions(config["sources"]["admissions"])
    patients = read_patients(config["sources"]["patients"])
    stays_by_hadm = read_icu_stays(config["sources"]["icustays"])
    c0 = set(admissions.keys())
    icu_hadms = set(stays_by_hadm.keys())
    c0_with_patient = {
        h for h in c0
        if admissions[h].get("subject_id", "") in patients
    }
    print(
        f"  hospital admissions={len(c0)} with_patient={len(c0_with_patient)} "
        f"with_icu_stay={len(c0 & icu_hadms)} ICU stays={stay_count(c0, stays_by_hadm)}"
    )

    print("[3/7] scan CHARTEVENTS for valid-data layer and invasive ventilation days")
    chartevent_hadms, vent_days_by_hadm, chartevent_stats = scan_chartevents(config["sources"]["chartevents"], config["cohort"])
    c1 = c0_with_patient & icu_hadms & chartevent_hadms
    print(f"  valid hospital admissions with patient+ICU stay+CHARTEVENTS={len(c1)} invasive vent hadm={len(vent_days_by_hadm)}")

    print("[4/7] scan diagnoses for ICD-9 E-code + ICD-10 external-cause trauma matches")
    matches, match_stats = collect_trauma_matches(config["sources"]["diagnoses_icd"], allow, config["cohort"])
    c2 = c1 & set(matches.keys())
    print(f"  trauma matched hadm={len(c2)}")

    print("[5/7] apply age, hospital LOS, and ventilator-day layers")
    c3 = set(); age_under = 0; age_over = 0; age_unknown = 0
    for h in c2:
        adm = admissions[h]
        pat = patients.get(adm["subject_id"], {})
        age = age_at_admit(pat, adm)
        if age is None:
            age_unknown += 1
        elif age < min_age:
            age_under += 1
        elif age > max_age:
            age_over += 1
        else:
            c3.add(h)
    c4 = {h for h in c3 if admissions[h].get("hospital_los_hours") is not None and admissions[h]["hospital_los_hours"] >= min_los}
    c5 = {h for h in c4 if len(vent_days_by_hadm.get(h, set())) >= vent_threshold}

    # Sub-breakdowns. These are mutually exclusive for the paper-style
    # "invalid hospital admission" side box.
    c0_missing_patient = c0 - c0_with_patient
    c0_no_icu = c0_with_patient - icu_hadms
    c0_icu_no_chart = c0_with_patient & icu_hadms - chartevent_hadms
    invalid_hospital_admission = c0 - c1
    non_trauma_admission = c1 - c2
    los_lt48 = c3 - c4
    los_lt48_died = sum(1 for h in los_lt48 if str(admissions[h].get("hospital_expire_flag", "")).strip() == "1")
    los_lt48_alive = sum(1 for h in los_lt48 if str(admissions[h].get("hospital_expire_flag", "")).strip() != "1")
    not_intubated = sum(1 for h in c4 if len(vent_days_by_hadm.get(h, set())) == 0)
    intubated_less = sum(1 for h in c4 if 0 < len(vent_days_by_hadm.get(h, set())) < vent_threshold)

    layers = [
        layer_record("C0", "MIMIC-IV Hospital Admission: all HADM IDs in hosp/admissions", c0, None, stays_by_hadm),
        layer_record("C1_VALID_DATA", "valid hospital admission: patient row + ICU stay + >=1 corresponding CHARTEVENTS row", c1, c0, stays_by_hadm),
        layer_record("C2_TRAUMA", "valid trauma patients: C1 + clean workbook trauma E-code evidence (ICD-9 exact E-code + ICD-10 exact external-cause code)", c2, c1, stays_by_hadm),
        layer_record("C3_ADULT", f"C2 + age_at_admit between {min_age} and {max_age}", c3, c2, stays_by_hadm),
        layer_record("C4_LOS", f"C3 + hospital_los_hours >= {min_los:g}", c4, c3, stays_by_hadm),
        layer_record("C5_FINAL", f"C4 + invasive mechanical ventilator days >= {vent_threshold}", c5, c4, stays_by_hadm),
    ]

    validations = [
        ("C1 subset C0", c1.issubset(c0)),
        ("C2 subset C1", c2.issubset(c1)),
        ("C3 subset C2", c3.issubset(c2)),
        ("C4 subset C3", c4.issubset(c3)),
        ("C5 subset C4", c5.issubset(c4)),
        ("ICD10 exact allowlist enabled", bool(config["cohort"].get("use_icd10", True))),
        ("Clean workbook has no Excel exclude column", ecode_stats.get("excluded_excel_rows", 0) == 0),
        ("Clean workbook path configured", os.path.basename(config["sources"].get("ecode_excel", "")) == "qualified_traumatic_Ecodes_clean.xlsx"),
    ]
    if not all(ok for _, ok in validations):
        for name, ok in validations:
            print(f"  validation {name}: {'PASS' if ok else 'FAIL'}")
        raise RuntimeError("Layer validation failed")

    extra = {
        "invalid_hospital_admission": len(invalid_hospital_admission),
        "non_trauma_admission": len(non_trauma_admission),
        "c0_missing_admission": 0,
        "c0_missing_patient": len(c0_missing_patient),
        "c0_no_icu": len(c0_no_icu),
        "c0_icu_no_chartevents": len(c0_icu_no_chart),
        "c0_no_chartevents": len(c0_no_icu) + len(c0_icu_no_chart),
        "c2_age_under": age_under,
        "c2_age_over": age_over,
        "c2_age_unknown": age_unknown,
        "los_lt48_total": len(los_lt48),
        "los_lt48_died": los_lt48_died,
        "los_lt48_alive": los_lt48_alive,
        "not_intubated": not_intubated,
        "intubated_less_threshold": intubated_less,
        "validations": validations,
    }

    print("[6/7] write final CSV/MD")
    write_cohort_csv(config["outputs"]["cohort_csv"], c5, admissions, patients, stays_by_hadm, matches, vent_days_by_hadm)
    write_layers_md(config["outputs"]["layers_md"], config, ecode_stats, match_stats, chartevent_stats, layers, extra)

    print("[7/7] done")
    for row in layers:
        print(f"  {row['layer']}: hadm={row['hadm_count']} stays={row['stay_count']} excluded={row['excluded_from_previous_hadm']}")
    print(f"  C0→ValidTrauma side box: invalid_hospital={extra['invalid_hospital_admission']} non_trauma={extra['non_trauma_admission']} no_icu={extra['c0_no_icu']} icu_no_chartevents={extra['c0_icu_no_chartevents']}")
    print(f"  C2→C3: age_under={extra['c2_age_under']} age_over={extra['c2_age_over']} age_unknown={extra['c2_age_unknown']}")
    print(f"  Vent filter from C4: not_intubated={extra['not_intubated']} intubated_less_than_{vent_threshold}={extra['intubated_less_threshold']}")
    print(f"  CSV: {config['outputs']['cohort_csv']}")
    print(f"  MD: {config['outputs']['layers_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
