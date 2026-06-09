# STALE / DO NOT USE — HADM Trajectory Input Sample from EHR2Path-derived JSON

> **Invalidated 2026-06-04:** this file used `/mnt/d/Data/mimic-iv-2.2/all_data_24_hours_los_noisy/...`, which is an EHR2Path-derived processed JSON directory, not an official MIMIC raw table. It should not be used as the requested raw-MIMIC/UW-aligned input sample. Keep only as an audit trail of the rejected sample.
>
> Purpose originally attempted: show one current trauma-cohort admission trajectory as an input-design sample before the `summary design/` work. This is a local research artifact; do not treat raw IDs as publishable identifiers.

## 0. Source and scope

| Item | Value |
|---|---|
| cohort source | `/mnt/d/Data/Data decision output/mimiciv_trauma_cohort.csv` |
| trajectory source | `/mnt/d/Data/mimic-iv-2.2/all_data_24_hours_los_noisy/stay_17640354_20483724.json` |
| selection rule | fixed random seed `20260604` among cohort admissions with existing EHR2Path trajectory JSON and >=8 landmarks |
| ISS/AIS | not used |
| sample type | Markdown preview of structured input blocks; not final training CSV/parquet |
| leakage policy | `hospital_expire_flag`, `dischtime`, LOS totals, `Sepsis`, `infectionDay`, `infectionHour`, and future labels are not rendered as encoder input |

## 1. Selected admission / cohort context

| Field | Value | Model role | Note |
|---|---:|---|---|
| `subject_id` | 17640354 | key_only | local key; not a model token |
| `hadm_id` | 20483724 | key_only | requested admission trajectory key |
| `stay_id` | 37162165 | key_only | first/current ICU stay in cohort row |
| `age_at_admit` | 54 | static_input | admission-known age |
| `gender` | M | static_input | mapped to sex_male |
| `trauma_icd_codes` | X58XXXA | cohort_evidence | cohort/stratification, not default realtime input |
| `trauma_mechanisms` | Natural/environmental, Other | optional_static_proxy_or_stratification | ICD-derived, retrospective provenance |
| `trauma_intents` | Unintentional | optional_static_proxy_or_stratification | ICD-derived |
| `trauma_types` | Other/unspecified | optional_static_proxy_or_stratification | ICD-derived |

Admission/stay spans from local metadata. **Audit context only**: `end_time` and `hours` are not encoder input because they encode future LOS/discharge information.

| stay_type | start_time | end_time | hours | model_role |
|---|---|---|---:|---|
| ed | 2156-02-24 12:00:00 | 2156-02-24 17:00:00 | 5.0 | audit_metadata_only |
| admission | 2156-02-24 15:00:00 | 2156-04-23 19:00:00 | 1420.0 | audit_metadata_only |
| icu | 2156-02-24 17:00:00 | 2156-04-09 13:00:00 | 1076.0 | audit_metadata_only |

## 2. STATIC block draft

```yaml
block: STATIC
key_context:
  hadm_id: 20483724
  subject_id: 17640354   # key_only; remove/hash for publication
static_input:
  age: 54
  sex_male: 1
  transfer_indicator: 1   # source: ED chief complaint `ich, transfer`
  head_injury_indicator: 1   # source: ED diagnosis/chief complaint; diagnosis=`traum subrac hem w/o loss of consciousness, init, fall on same level, unspecified, initial encounter`
  initial_ed_sbp: 130.8
  initial_ed_hr: 81.1
  rsi_candidate_sbp_over_hr: 1.61   # formula placeholder; registry must fix rSI definition
audit_or_stratification_only:
  trauma_icd_codes: "X58XXXA"
  trauma_mechanism_label: "Natural/environmental, Other"
  trauma_intent_label: "Unintentional"
  trauma_type_label: "Other/unspecified"
excluded_from_input:
  - hospital_los_hours
  - icu_los_days
  - hospital_expire_flag
  - dischtime
```

## 3. UW-aligned trajectory table

Each row is one existing EHR2Path landmark. Values are compact latest/window summaries parsed from the trajectory JSON; labs remain intermittent memory-style fields, and `uop_foley_ml_window` is a Foley-output window sum from the rendered local window, not a final canonical hourly output variable.

|landmark_h|relative_day|hr|sbp|dbp|map|bp_source|rr|spo2|fio2|temp_c|bicarb|creatinine|bun|wbc|lactate|uop_foley_ml_window|vent|vent_mode|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|11|1|67.3|119.5|69|89.3|arterial|21.7|100|40|36.3|22|0.80|16|15|NA|105|1|CMV/ASSIST/AutoFlow|
|72|4|112|118|65|84|arterial|29|96|30|37.3|20|0.70|8|13|0.70|2230|1|CPAP/PSV|
|230|10|78|168|88|116|arterial|22|100|30|38.1|18|0.40|15|18.3|0.90|3730|1|CPAP/PSV|
|342|15|100|171|91|119|arterial|32|99|40|38|20|0.50|11|15.7|NA|5205|1|CPAP/PSV|
|463|20|94|145|99|109|noninvasive|24|100|35|37.2|23|0.40|17|16.7|NA|NA|1|NA|
|575|24|106|128|89|97|noninvasive|27|97|40|37.4|25|0.80|11|11.8|NA|2515|1|CPAP/PSV|
|696|30|108|137|109|118|noninvasive|33|98|40|36.8|25|0.80|20|11.8|NA|1450|0|NA|
|701|30|117|168|122|133|noninvasive|33|97|40|36.8|25|0.80|20|11.8|NA|1450|0|NA|
|912|39|94|151|101|114|noninvasive|31|97|35|36.9|25|0.50|13|9|NA|1195|1|NA|
|938|40|90|140|106|115|noninvasive|18|98|35|36.7|NA|NA|NA|NA|NA|2050|0|NA|
|1085|46|84|136|107|117|noninvasive|27|100|35|37.1|25|0.50|9|12.5|NA|1445|0|NA|
|1104|47|90|140|101|113|noninvasive|28|100|35|36.6|25|0.50|10|8.6|NA|215|0|NA|
|1133|48|NA|NA|NA|NA|missing|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|0|NA|
|1313|55|NA|NA|NA|NA|missing|NA|NA|NA|NA|26|0.50|16|9.1|NA|NA|0|NA|
|1418|60|NA|NA|NA|NA|missing|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|0|NA|

## 4. Example HOUR block rendering

Example cutoff: `landmark_h=342`. This shows the kind of block that will later feed the summary design.

```text
[HOUR h=342 day=15 source=ehr2path_json]
V: HR=100 bpm; SBP=171 mmHg (arterial); DBP=91 mmHg; MAP=119 mmHg; RR=32/min; SpO2=99%; FiO2=40; Temp=38 C
L: bicarb=20 mEq/L; Cr=0.50 mg/dL; BUN=11 mg/dL; WBC=15.7 K/uL; lactate=NA mmol/L
Tx/Output: vent=1 mode=CPAP/PSV; Foley_output_window=5205 ml
[/HOUR]
```

## 5. Example model-facing token records for the same cutoff

These are illustrative token records; final numeric normalization/severity transforms belong in `summary design/` registry.

```yaml
landmark_hour: 342
tokens:
  - segment: recent_hour
    source_type: vital
    canonical_variable: heart_rate
    unit: bpm
    value: 100
    observed_flag: true
    window: ehr2path_local_landmark_window
    leakage_role: input
    availability_status: observed_or_rendered_from_observed_until_t
  - segment: recent_hour
    source_type: vital
    canonical_variable: systolic_blood_pressure
    unit: mmHg
    value: 171
    observed_flag: true
    window: ehr2path_local_landmark_window
    leakage_role: input
    availability_status: observed_or_rendered_from_observed_until_t
  - segment: recent_hour
    source_type: vital
    canonical_variable: mean_arterial_pressure
    unit: mmHg
    value: 119
    observed_flag: true
    window: ehr2path_local_landmark_window
    leakage_role: input
    availability_status: observed_or_rendered_from_observed_until_t
  - segment: recent_hour
    source_type: vital
    canonical_variable: respiratory_rate
    unit: /min
    value: 32
    observed_flag: true
    window: ehr2path_local_landmark_window
    leakage_role: input
    availability_status: observed_or_rendered_from_observed_until_t
  - segment: recent_hour
    source_type: respiratory
    canonical_variable: fio2
    unit: percent_or_fraction_raw
    value: 40
    observed_flag: true
    window: ehr2path_local_landmark_window
    leakage_role: input
    availability_status: observed_or_rendered_from_observed_until_t
  - segment: recent_hour
    source_type: lab_memory
    canonical_variable: bicarbonate
    unit: mEq/L
    value: 20
    observed_flag: true
    window: ehr2path_local_landmark_window
    leakage_role: input
    availability_status: observed_or_rendered_from_observed_until_t
  - segment: recent_hour
    source_type: lab_memory
    canonical_variable: creatinine
    unit: mg/dL
    value: 0.50
    observed_flag: true
    window: ehr2path_local_landmark_window
    leakage_role: input
    availability_status: observed_or_rendered_from_observed_until_t
  - segment: recent_hour
    source_type: lab_memory
    canonical_variable: white_blood_cell_count
    unit: K/uL
    value: 15.70
    observed_flag: true
    window: ehr2path_local_landmark_window
    leakage_role: input
    availability_status: observed_or_rendered_from_observed_until_t
  - segment: recent_hour
    source_type: intervention
    canonical_variable: ventilation_status
    unit: binary
    value: 1
    observed_flag: true
    window: ehr2path_local_landmark_window
    leakage_role: input
    availability_status: observed_or_rendered_from_observed_until_t
  - segment: recent_hour
    source_type: output
    canonical_variable: urine_output_foley_window
    unit: ml/window
    value: 5205
    observed_flag: true
    window: ehr2path_local_landmark_window
    leakage_role: input
    availability_status: observed_or_rendered_from_observed_until_t
```

## 6. Notes / limitations before summary design

- This sample uses the existing EHR2Path processed trajectory JSON because full UW-style dense `hourly_state.csv` is not currently present on disk.
- The trajectory is still useful for input design because it preserves time blocks and observed clinical facts, but final training input should be rebuilt from raw MIMIC tables through the UW registry.
- `hospital_los_hours`, `icu_los_days`, discharge/death fields, and sepsis/infection labels are intentionally excluded from encoder input.
- `rSI` formula and UW category thresholds are not finalized; they are shown as placeholders/held fields, not semantic bins.
- `summary design/` should next define deterministic DAY signatures from these HOUR/landmark facts: dense vitals, sparse labs, treatments, output, masks, recency, and source provenance.

