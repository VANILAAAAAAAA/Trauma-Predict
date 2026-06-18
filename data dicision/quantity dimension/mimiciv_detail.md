# MIMIC-IV Detail — quantity/forms for current fields

| Group | Field | MIMIC-IV source / derivation | Raw data form | Canonical unit/scale | 5-sample form | Note |
| --- | --- | --- | --- | --- | --- | --- |
| G2 | age | hosp.patients.anchor_age | static scalar | years | {67, 67, 67, 67, 67} |  |
| G2 | male | hosp.patients.gender | static category M/F -> binary | binary | {1, 1, 1, 1, 1} |  |
| G2 | mechanism_cat | hosp.diagnoses_icd + E-code mapping | diagnosis/external-cause code category | category | {1, 1, 1, 1, 1} |  |
| G2 | transfer | hosp.admissions.admission_location / transfer derivation | admission location/category | category | {0, 0, 0, 0, 0} |  |
| G2 | initial_ed_sbp | MIMIC-IV-ED triage/vitalsign sbp | ED scalar measurement | mmHg | {128, 128, 128, 128, 128} |  |
| G2 | rsi | ED SBP / ED HR | derived scalar ratio | ratio SBP/HR | {1.778, 1.778, 1.778, 1.778, 1.778} |  |
| G2 | head_injury | hosp.diagnoses_icd ICD anatomy rule | diagnosis-derived binary | binary | {1, 1, 1, 1, 1} |  |
| G1 | hr | icu.chartevents itemid 220045 | event rows -> hourly aggregate | bpm | {71.6, 63, 66, 69, 69} |  |
| G1 | sbp | icu.chartevents BP itemids, sample 220179 | event rows -> hourly aggregate | mmHg | {93.83, 90, 124, 114, 114} |  |
| G1 | dbp | icu.chartevents BP itemids, sample 220180 | event rows -> hourly aggregate | mmHg | {49.33, 48, 72, 62, 62} |  |
| G1 | map | icu.chartevents MAP itemids, sample 220181 | event rows -> hourly aggregate | mmHg | {62.17, 59, 94, 81, 81} |  |
| G1 | rr | icu.chartevents itemid 220210 | event rows -> hourly aggregate | breaths/min | {14.4, 16, 18, 19, 19} |  |
| G1 | temp | icu.chartevents temp itemids; sample °F item 223761 | event rows; °F/°C normalize -> hourly aggregate | degC | {35.22, 35.22, 35.22, 35.72, 35.72} | Raw sample is °F; canonical must convert to °C. |
| G1 | fio2 | icu.chartevents FiO2 itemids; percent/fraction normalize | event rows; percent/fraction normalize -> hourly aggregate | fraction [0,1] | {0.75, 0.75, 0.75, 0.75, 0.75} | MIMIC may store percent or fraction; canonical fraction required. |
| G3 | vent_h | icu.chartevents/procedureevents ventilation evidence -> hourly active flag | status/evidence events -> hourly binary | binary | {1, 1, 1, 1, 1} |  |
| G3 | vent_day_sum_until_h | derived from vent_h by completed ventilated days | derived cumulative days | days | {1, 1, 1, 1, 1} |  |
| G3 | bolus_sum_until_h | icu.inputevents crystalloid/bolus interval amount; no cumulative default | interval input amount -> hourly mL event, not cumulative | MIMIC mL event amount; UW source-scale cumulative only for alignment | {50, 32.5, 250, 14.75, 250} | MIMIC raw amount is mL interval event; sample includes crystalloids under both fluids and antibiotics-IV categories, so treatment/resuscitation inclusion rule must be audited. |
| G3 | rbc_sum_until_h | icu.inputevents PRBC interval amount; no cumulative default | interval input amount -> hourly mL event, not cumulative | MIMIC mL event amount; UW source-scale cumulative only for alignment | {276, 350, 277, 280, 52.02} | MIMIC PRBC amount is mL interval event; final HOUR/DAY should use mL unless an explicit unit-conversion rule is chosen. |
| G4 | bicarb | hosp.labevents itemid 50882 | intermittent lab rows | mEq/L | {26, 26, 27, NA, NA} |  |
| G4 | strong_ion | derived (Na+K)-(Cl+bicarb) from labevents | derived intermittent lab value | mEq/L | {NA, NA, NA, NA, NA} |  |
| G4 | bun | hosp.labevents itemid 51006 | intermittent lab rows | mg/dL | {12, 9, 10, NA, NA} |  |
| G4 | creatinine | hosp.labevents itemid 50912 | intermittent lab rows | mg/dL | {0.7, 0.6, 0.5, NA, NA} |  |
| G4 | wbc | hosp.labevents itemid 51301 | intermittent lab rows | K/uL | {12, 10, 8.8, NA, NA} |  |
| G4 | lymphocytes | hosp.labevents itemid 51244 in sample (%) | intermittent lab rows | UW/MIMIC unit audit; MIMIC raw sample percent | {19.9, NA, NA, NA, NA} | Raw sample unit is %. If absolute count is needed, derive from WBC or use absolute-count itemids; do not label as K/uL unless confirmed. |
| G4 | neutrophils | hosp.labevents itemid 51256 in sample (%) | intermittent lab rows | UW/MIMIC unit audit; MIMIC raw sample percent | {71.3, NA, NA, NA, NA} | Raw sample unit is %. If absolute count is needed, derive from WBC or use absolute-count itemids; do not label as K/uL unless confirmed. |
| G4 | uop | icu.outputevents urine itemid; sample 226560 ml | output event rows -> hourly/interval mL | mL per recorded hour/interval | {300, 100, 250, 300, NA} | outputevents value is mL per event/interval; no weight-normalized KDIGO gate unless weight is added. |
| G2* | base_def_48 | derived from blood gas base excess/deficit within first 48h | first-48h derived summary | mEq/L deficit magnitude | derived first-48h scalar; sample not built | Only include after completed 48h window to avoid leakage. |
| G2* | lactate_48 | hosp.labevents/blood gas lactate within first 48h | first-48h derived summary | mmol/L | derived first-48h scalar; sample not built | Only include after completed 48h window to avoid leakage. |
| G2* | rbc_48 | derived sum of PRBC inputevents first 48h | first-48h derived summary from inputevents | UW source-scale units/count; MIMIC mL-derived if rebuilt | derived first-48h scalar; sample not built | Only include after completed 48h window to avoid leakage. |
| G2* | crys_48 | derived sum of crystalloid inputevents first 48h | first-48h derived summary from inputevents | mL | derived first-48h scalar; sample not built | Only include after completed 48h window to avoid leakage. |
