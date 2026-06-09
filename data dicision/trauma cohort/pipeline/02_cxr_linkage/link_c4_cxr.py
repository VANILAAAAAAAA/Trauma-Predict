#!/usr/bin/env python3
"""C4 trauma cohort → MIMIC-CXR linkage.

CXR metadata only. No images are downloaded by this script.

Outputs under data dicision/trauma cohort/cxr_linkage/:
  - c4_cxr_linkage.json        aggregate HADM/study/image counts + validation
  - c4_cxr_hadm_layers.csv     one row per C4 HADM, layer flags
  - c4_cxr_dicom_layers.csv    one row per matched CXR image row, exact layer flags

Layer definitions are C4-scoped:
  L2 = C4 HADM with same-admission CXR or ICU-window CXR
  L3 = C4 HADM with ICU-window CXR
  L4 = C4 HADM with ICU-window CXR within first 312h of ICU
  L5 = C4 HADM with pre-ICU CXR and no ICU-window CXR
"""
import csv
import gzip
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta

PROJECT_ROOT = "/home/vanila/code/EHR-Predict"
COHORT_CSV = os.path.join(PROJECT_ROOT, "data dicision", "trauma cohort", "cohort", "mimiciv_trauma_cohort_los48.csv")
CXR_CSV = "/mnt/d/Data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz"
OUT_DIR = os.path.join(PROJECT_ROOT, "data dicision", "trauma cohort", "cxr_linkage")
EARLY_ICU_HOURS = 312


def parse_dt(date_str, time_str):
    try:
        d = str(int(float(date_str)))
        y, m, day = int(d[:4]), int(d[4:6]), int(d[6:8])
        if time_str:
            t = float(time_str)
            h = int(t // 10000)
            mi = int((t % 10000) // 100)
            s = int(t % 100)
            return datetime(y, m, day, h, mi, s)
        return datetime(y, m, day)
    except (ValueError, TypeError):
        return None


def layer_counts(hadm_rows, image_rows):
    layers = ["L2_cohort_cxr", "L3_icu_window_cxr", "L4_icu_first312h_cxr", "L5_pre_icu_only_cxr"]
    out = {}
    for layer in layers:
        hadms = {r["hadm_id"] for r in hadm_rows.values() if r[layer]}
        studies = {r["study_id"] for r in image_rows if r[layer]}
        images = {r["dicom_id"] for r in image_rows if r[layer]}
        out[layer] = {
            "hadm_id": len(hadms),
            "study_id": len(studies),
            "dicom_id": len(images),
        }
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading C4 cohort...")
    cohort = {}
    subj_to_hadm = defaultdict(list)
    with open(COHORT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hid = row["hadm_id"]
            if hid not in cohort:
                cohort[hid] = {
                    "subject_id": row["subject_id"],
                    "admittime": datetime.fromisoformat(row["admittime"]),
                    "dischtime": datetime.fromisoformat(row["dischtime"]) if row["dischtime"] else None,
                    "stays": [],
                }
                subj_to_hadm[row["subject_id"]].append(hid)
            cohort[hid]["stays"].append({
                "stay_id": row["stay_id"],
                "intime": datetime.fromisoformat(row["intime"]),
                "outtime": datetime.fromisoformat(row["outtime"]) if row["outtime"] else None,
            })
    print(f"  C4 HADM: {len(cohort)}")

    # Initialize HADM rows.
    hadm_rows = {}
    for hid, info in cohort.items():
        hadm_rows[hid] = {
            "hadm_id": hid,
            "subject_id": info["subject_id"],
            "L2_cohort_cxr": False,
            "L3_icu_window_cxr": False,
            "L4_icu_first312h_cxr": False,
            "L5_pre_icu_only_cxr": False,
            "same_admission_image_rows": 0,
            "pre_icu_image_rows": 0,
            "icu_window_image_rows": 0,
            "icu_first312h_image_rows": 0,
        }

    print("Scanning CXR metadata and classifying rows...")
    provisional_image_rows = []
    with gzip.open(CXR_CSV, "rt", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["subject_id"]
            if sid not in subj_to_hadm:
                continue
            dt = parse_dt(row.get("StudyDate", ""), row.get("StudyTime", ""))
            if dt is None:
                continue
            for hid in subj_to_hadm[sid]:
                info = cohort[hid]
                adm_dt = info["admittime"]
                dis_dt = info["dischtime"] or datetime.max
                same_admission = adm_dt <= dt <= dis_dt
                if not same_admission:
                    continue

                icu_window = False
                icu_first312h = False
                pre_icu = False
                matched_stay_ids = []
                earliest_icu = min(s["intime"] for s in info["stays"])
                if adm_dt <= dt < earliest_icu:
                    pre_icu = True

                for stay in info["stays"]:
                    out_dt = stay["outtime"] or datetime.max
                    if stay["intime"] <= dt <= out_dt:
                        icu_window = True
                        matched_stay_ids.append(stay["stay_id"])
                        if dt <= stay["intime"] + timedelta(hours=EARLY_ICU_HOURS):
                            icu_first312h = True

                hr = hadm_rows[hid]
                hr["L2_cohort_cxr"] = True
                hr["same_admission_image_rows"] += 1
                if pre_icu:
                    hr["pre_icu_image_rows"] += 1
                if icu_window:
                    hr["L3_icu_window_cxr"] = True
                    hr["icu_window_image_rows"] += 1
                if icu_first312h:
                    hr["L4_icu_first312h_cxr"] = True
                    hr["icu_first312h_image_rows"] += 1

                provisional_image_rows.append({
                    "hadm_id": hid,
                    "subject_id": sid,
                    "study_id": row["study_id"],
                    "dicom_id": row.get("dicom_id", ""),
                    "study_dt": dt.isoformat(sep=" "),
                    "view": row.get("ViewPosition", ""),
                    "matched_stay_ids": ";".join(matched_stay_ids),
                    "same_admission": same_admission,
                    "pre_icu": pre_icu,
                    "L2_cohort_cxr": True,
                    "L3_icu_window_cxr": icu_window,
                    "L4_icu_first312h_cxr": icu_first312h,
                    "L5_pre_icu_only_cxr": False,  # set after HADM-level no-ICU check
                })

    # L5: HADM has pre-ICU CXR and no ICU-window CXR.
    for hid, hr in hadm_rows.items():
        if hr["pre_icu_image_rows"] > 0 and not hr["L3_icu_window_cxr"]:
            hr["L5_pre_icu_only_cxr"] = True

    image_rows = []
    for r in provisional_image_rows:
        if r["pre_icu"] and hadm_rows[r["hadm_id"]]["L5_pre_icu_only_cxr"]:
            r["L5_pre_icu_only_cxr"] = True
        image_rows.append(r)

    counts = layer_counts(hadm_rows, image_rows)
    validation = {
        "L3_subset_L2": all((not r["L3_icu_window_cxr"]) or r["L2_cohort_cxr"] for r in hadm_rows.values()),
        "L4_subset_L3": all((not r["L4_icu_first312h_cxr"]) or r["L3_icu_window_cxr"] for r in hadm_rows.values()),
        "L5_subset_L2": all((not r["L5_pre_icu_only_cxr"]) or r["L2_cohort_cxr"] for r in hadm_rows.values()),
        "L5_disjoint_L3": all(not (r["L5_pre_icu_only_cxr"] and r["L3_icu_window_cxr"]) for r in hadm_rows.values()),
    }

    report = {
        "cohort": "C4",
        "cohort_hadm_count": len(cohort),
        "early_icu_hours": EARLY_ICU_HOURS,
        "cxr_layer_counts": counts,
        "validation": validation,
        "note": "C4-scoped CXR linkage. L2/L3/L4/L5 are C4 HADM layers. True all-disease L1 is not computed here.",
    }

    with open(os.path.join(OUT_DIR, "c4_cxr_linkage.json"), "w") as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(OUT_DIR, "c4_cxr_hadm_layers.csv"), "w", newline="") as f:
        fields = [
            "hadm_id", "subject_id", "L2_cohort_cxr", "L3_icu_window_cxr", "L4_icu_first312h_cxr", "L5_pre_icu_only_cxr",
            "same_admission_image_rows", "pre_icu_image_rows", "icu_window_image_rows", "icu_first312h_image_rows",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for hid in sorted(hadm_rows):
            row = dict(hadm_rows[hid])
            for k in ["L2_cohort_cxr", "L3_icu_window_cxr", "L4_icu_first312h_cxr", "L5_pre_icu_only_cxr"]:
                row[k] = int(row[k])
            writer.writerow(row)

    with open(os.path.join(OUT_DIR, "c4_cxr_dicom_layers.csv"), "w", newline="") as f:
        fields = [
            "hadm_id", "subject_id", "study_id", "dicom_id", "study_dt", "view", "matched_stay_ids",
            "same_admission", "pre_icu", "L2_cohort_cxr", "L3_icu_window_cxr", "L4_icu_first312h_cxr", "L5_pre_icu_only_cxr",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in image_rows:
            out = dict(row)
            for k in ["same_admission", "pre_icu", "L2_cohort_cxr", "L3_icu_window_cxr", "L4_icu_first312h_cxr", "L5_pre_icu_only_cxr"]:
                out[k] = int(out[k])
            writer.writerow(out)

    print("\nCXR linkage results:")
    for layer, c in counts.items():
        print(f"  {layer}: HADM={c['hadm_id']} studies={c['study_id']} images={c['dicom_id']}")
    print("Validation:", json.dumps(validation, indent=2))
    print("Output:", OUT_DIR)


if __name__ == "__main__":
    main()
