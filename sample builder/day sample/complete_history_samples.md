# Complete History Samples — STATIC + DAY windows + HOUR

Generated from selected C4 trauma cohort stays and official MIMIC-IV raw tables.

## Raw scan counts

| stream | matched rows/events |
|---|---:|
| `chartevents` | 17824 |
| `labevents` | 2043 |
| `outputevents_uop` | 1549 |
| `inputevents_rbc` | 30 |
| `procedureevents_vent_hours` | 2318 |

## Validation summary

| Sample | HADM | Stay | Anchor | DAY | HOUR | ED | FIRST48≤1 | RR burden | Old tokens |
|---|---:|---:|---|---:|---:|---|---|---|---|
| `day_sample_01` | 22916536 | 37653135 | day 1 / h23 | 2 | 24 | no | yes | yes | PASS |
| `day_sample_05` | 26488509 | 31292653 | day 3 / h12 | 4 | 24 | yes | yes | no | PASS |
| `day_sample_03` | 27434217 | 33485623 | day 3 / h12 | 4 | 24 | no | yes | no | PASS |
| `day_sample_04` | 27851010 | 37582802 | day 3 / h12 | 4 | 24 | no | yes | yes | PASS |
| `day_sample_06` | 29079804 | 35580228 | day 3 / h12 | 4 | 24 | no | yes | yes | PASS |
| `day_sample_02` | 29423991 | 30481610 | day 3 / h12 | 4 | 24 | yes | yes | yes | PASS |

## Interpretation check

- All samples are computed from source tables listed in JSONL; no free-text fabrication.
- DAY blocks provide longitudinal burden and current partial-day summary through `[day_window_len_XXh]`.
- HOUR is review rendering; model-side input remains fixed `vital_values[T,7]` and `vital_mask[T,7]`.
- FIRST48 tokens appear at most once per sample; RR is duration-burden when present.

## Token coverage highlights

- `[SEP]`: 172
- `[heart_rate]`: 118
- `[systolic_bp]`: 118
- `[diastolic_bp]`: 118
- `[mean_arterial_pressure]`: 118
- `[respiratory_rate]`: 98
- `[vent_on]`: 96
- `[temperature]`: 66
- `[fio2]`: 29
- `[data_quality]`: 22
- `[renal_metabolic]`: 19
- `[bicarbonate_min_bin_low]`: 19
- `[day_window_len_24h]`: 17
- `[oxygenation_ventilation]`: 17
- `[uop_measured]`: 17
- `[perfusion_shock]`: 15
- `[immune_hematologic]`: 14
- `[core_vital_slots_dense]`: 12
- `[vent_hours_bin_full_window]`: 12
- `[wbc_bin_high]`: 9
- `[vent_course_bin_early]`: 8
- `[fio2_max_bin_high_support]`: 8
- `[bun_creatinine_ratio_bin_prerenal_pattern]`: 8
- `[respiratory_rate_high_hours_bin_prolonged]`: 8
- `[creatinine_change_bin_kdigo_delta]`: 7
- `[STATIC]`: 6
- `[transfer_direct]`: 6
- `[DAY_REL_-1]`: 6
- `[DAY_REL_0]`: 6
- `[HOUR_REL_-23]`: 6
- `[HOUR_REL_-22]`: 6
- `[HOUR_REL_-21]`: 6
- `[HOUR_REL_-20]`: 6
- `[HOUR_REL_-19]`: 6
- `[HOUR_REL_-18]`: 6

## day_sample_01 — hadm 22916536 / stay 37653135

Anchor: ICU day 1 hour 23; observed_until `2167-05-31 20:37:00`

```text
[STATIC] [age_bin_85_89] [sex_F] [injury_mechanism_other] [transfer_direct] [ed_linkage_no] [head_injury_no] [SEP]

[DAY_REL_-1] [day_window_len_24h] [oxygenation_ventilation] [respiratory_rate_high_hours_bin_brief] [data_quality] [core_vital_slots_dense] [uop_measured] [SEP]

[DAY_REL_0] [day_window_len_24h] [perfusion_shock] [systolic_bp_min_bin_geriatric_low] [data_quality] [core_vital_slots_dense] [uop_measured] [SEP]

[HOUR_REL_-23] [heart_rate] <85> [systolic_bp] <144> [diastolic_bp] <61> [mean_arterial_pressure] <94> [respiratory_rate] <15> [SEP]
[HOUR_REL_-22] [heart_rate] <75> [systolic_bp] <137> [diastolic_bp] <59> [mean_arterial_pressure] <89> [respiratory_rate] <18> [SEP]
[HOUR_REL_-21] [heart_rate] <73> [systolic_bp] <135> [diastolic_bp] <59> [mean_arterial_pressure] <87> [respiratory_rate] <18> [SEP]
[HOUR_REL_-20] [heart_rate] <74> [systolic_bp] <140> [diastolic_bp] <62> [mean_arterial_pressure] <92> [respiratory_rate] <18> [temperature] <36.7> [SEP]
[HOUR_REL_-19] [heart_rate] <71> [systolic_bp] <142> [diastolic_bp] <66> [mean_arterial_pressure] <94> [respiratory_rate] <17> [SEP]
[HOUR_REL_-18] [heart_rate] <70> [systolic_bp] <131> [diastolic_bp] <60> [mean_arterial_pressure] <86> [respiratory_rate] <17> [SEP]
[HOUR_REL_-17] [heart_rate] <72> [systolic_bp] <142> [diastolic_bp] <70> [mean_arterial_pressure] <97> [respiratory_rate] <19> [SEP]
[HOUR_REL_-16] [heart_rate] <76> [systolic_bp] <147> [diastolic_bp] <66> [mean_arterial_pressure] <97> [respiratory_rate] <16> [temperature] <35.8> [SEP]
[HOUR_REL_-15] [heart_rate] <66> [systolic_bp] <138> [diastolic_bp] <59> [mean_arterial_pressure] <88> [respiratory_rate] <18> [SEP]
[HOUR_REL_-14] [heart_rate] <67> [systolic_bp] <126> [diastolic_bp] <53> [mean_arterial_pressure] <80> [respiratory_rate] <16> [SEP]
[HOUR_REL_-13] [heart_rate] <68> [systolic_bp] <124> [diastolic_bp] <55> [mean_arterial_pressure] <80> [respiratory_rate] <17> [SEP]
[HOUR_REL_-12] [heart_rate] <65> [systolic_bp] <123> [diastolic_bp] <51> [mean_arterial_pressure] <78> [respiratory_rate] <17> [temperature] <35.6> [SEP]
[HOUR_REL_-11] [heart_rate] <70> [systolic_bp] <129> [diastolic_bp] <56> [mean_arterial_pressure] <83> [respiratory_rate] <17> [SEP]
[HOUR_REL_-10] [heart_rate] <72> [systolic_bp] <133> [diastolic_bp] <58> [mean_arterial_pressure] <85> [respiratory_rate] <18> [SEP]
[HOUR_REL_-9] [heart_rate] <69> [systolic_bp] <134> [diastolic_bp] <58> [mean_arterial_pressure] <86> [respiratory_rate] <17> [SEP]
[HOUR_REL_-8] [heart_rate] <77> [systolic_bp] <116> [diastolic_bp] <54> [mean_arterial_pressure] <76> [respiratory_rate] <16> [temperature] <35.6> [SEP]
[HOUR_REL_-7] [heart_rate] <107> [systolic_bp] <111> [diastolic_bp] <61> [mean_arterial_pressure] <80> [respiratory_rate] <18> [SEP]
[HOUR_REL_-6] [heart_rate] <124> [systolic_bp] <123> [diastolic_bp] <70> [mean_arterial_pressure] <90> [respiratory_rate] <18> [SEP]
[HOUR_REL_-5] [heart_rate] <119> [systolic_bp] <128> [diastolic_bp] <72> [mean_arterial_pressure] <93> [respiratory_rate] <17> [SEP]
[HOUR_REL_-4] [heart_rate] <111> [systolic_bp] <135> [diastolic_bp] <77> [mean_arterial_pressure] <99> [respiratory_rate] <18> [temperature] <36.9> [SEP]
[HOUR_REL_-3] [heart_rate] <65> [systolic_bp] <115> [diastolic_bp] <50> [mean_arterial_pressure] <73> [respiratory_rate] <16> [SEP]
[HOUR_REL_-2] [heart_rate] <64> [systolic_bp] <111> [diastolic_bp] <47> [mean_arterial_pressure] <69> [respiratory_rate] <17> [SEP]
[HOUR_REL_-1] [heart_rate] <72> [systolic_bp] <129> [diastolic_bp] <55> [mean_arterial_pressure] <81> [respiratory_rate] <15> [SEP]
[HOUR_REL_0] [heart_rate] <69> [systolic_bp] <111> [diastolic_bp] <45> [mean_arterial_pressure] <68> [respiratory_rate] <12> [temperature] <36.8> [CUR] [SEP]
```

## day_sample_05 — hadm 26488509 / stay 31292653

Anchor: ICU day 3 hour 12; observed_until `2192-03-21 10:05:53`

```text
[STATIC] [age_bin_75_84] [sex_M] [injury_mechanism_other] [transfer_direct] [ed_linkage_yes] [initial_ed_sbp] <143> [initial_ed_sbp_bin_not_low] [reverse_shock_index] <2.01> [reverse_shock_index_bin_low_risk] [head_injury_no] [SEP]

[DAY_REL_-3] [day_window_len_24h] [perfusion_shock] [map_low_hours_bin_prolonged] [systolic_bp_min_bin_hypotension] [oxygenation_ventilation] [vent_hours_bin_partial_window] [vent_course_bin_first_day] [fio2_max_bin_very_high_support] [renal_metabolic] [bicarbonate_min_bin_low] [immune_hematologic] [rbc_transfusion_event_present] [data_quality] [core_vital_slots_partial] [uop_measured] [SEP]

[DAY_REL_-2] [day_window_len_24h] [perfusion_shock] [map_low_hours_bin_brief] [systolic_bp_min_bin_low] [lactate_48h_bin_severe] [base_deficit_48h_bin_severe] [oxygenation_ventilation] [vent_hours_bin_full_window] [vent_course_bin_early] [fio2_max_bin_high_support] [renal_metabolic] [creatinine_change_bin_kdigo_delta] [creatinine_ratio_bin_kdigo_ratio] [bicarbonate_min_bin_low] [immune_hematologic] [rbc_transfusion_event_present] [resuscitation_burden] [rbc_48h_event_present] [data_quality] [core_vital_slots_dense] [uop_measured] [SEP]

[DAY_REL_-1] [day_window_len_24h] [perfusion_shock] [map_low_hours_bin_intermittent] [systolic_bp_min_bin_low] [oxygenation_ventilation] [vent_hours_bin_full_window] [vent_course_bin_early] [fio2_max_bin_high_support] [renal_metabolic] [creatinine_change_bin_kdigo_delta] [bicarbonate_min_bin_low] [immune_hematologic] [rbc_transfusion_event_present] [data_quality] [core_vital_slots_dense] [uop_measured] [SEP]

[DAY_REL_0] [day_window_len_13h] [perfusion_shock] [map_low_hours_bin_brief] [systolic_bp_min_bin_low] [oxygenation_ventilation] [vent_hours_bin_full_window] [vent_course_bin_prolonged] [renal_metabolic] [bicarbonate_min_bin_low] [data_quality] [core_vital_slots_dense] [uop_measured] [SEP]

[HOUR_REL_-23] [heart_rate] <64> [systolic_bp] <119> [diastolic_bp] <52> [mean_arterial_pressure] <69> [respiratory_rate] <22> [temperature] <37.6> [fio2] <0.50> [vent_on] [SEP]
[HOUR_REL_-22] [heart_rate] <63> [systolic_bp] <138> [diastolic_bp] <57> [mean_arterial_pressure] <78> [respiratory_rate] <22> [temperature] <37.6> [vent_on] [SEP]
[HOUR_REL_-21] [heart_rate] <74> [systolic_bp] <107> [diastolic_bp] <45> [mean_arterial_pressure] <63> [respiratory_rate] <22> [temperature] <37.4> [vent_on] [SEP]
[HOUR_REL_-20] [heart_rate] <72> [systolic_bp] <125> [diastolic_bp] <94> [mean_arterial_pressure] <103> [respiratory_rate] <22> [temperature] <37.3> [vent_on] [SEP]
[HOUR_REL_-19] [heart_rate] <72> [systolic_bp] <102> [diastolic_bp] <89> [mean_arterial_pressure] <92> [respiratory_rate] <22> [temperature] <37.4> [vent_on] [SEP]
[HOUR_REL_-18] [heart_rate] <76> [systolic_bp] <141> [diastolic_bp] <58> [mean_arterial_pressure] <82> [respiratory_rate] <22> [temperature] <37.4> [fio2] <0.50> [vent_on] [SEP]
[HOUR_REL_-17] [heart_rate] <65> [systolic_bp] <139> [diastolic_bp] <57> [mean_arterial_pressure] <80> [respiratory_rate] <22> [temperature] <37.5> [vent_on] [SEP]
[HOUR_REL_-16] [heart_rate] <73> [systolic_bp] <128> [diastolic_bp] <54> [mean_arterial_pressure] <75> [respiratory_rate] <22> [temperature] <37.5> [fio2] <0.40> [vent_on] [SEP]
[HOUR_REL_-15] [heart_rate] <79> [systolic_bp] <111> [diastolic_bp] <49> [mean_arterial_pressure] <66> [respiratory_rate] <22> [temperature] <37.6> [fio2] <0.40> [vent_on] [SEP]
[HOUR_REL_-14] [heart_rate] <75> [systolic_bp] <123> [diastolic_bp] <56> [mean_arterial_pressure] <74> [respiratory_rate] <22> [temperature] <37.7> [vent_on] [SEP]
[HOUR_REL_-13] [heart_rate] <67> [systolic_bp] <120> [diastolic_bp] <53> [mean_arterial_pressure] <71> [respiratory_rate] <22> [temperature] <37.7> [vent_on] [SEP]
[HOUR_REL_-12] [heart_rate] <67> [systolic_bp] <122> [diastolic_bp] <57> [mean_arterial_pressure] <75> [respiratory_rate] <22> [temperature] <37.6> [vent_on] [SEP]
[HOUR_REL_-11] [heart_rate] <63> [systolic_bp] <105> [diastolic_bp] <50> [mean_arterial_pressure] <66> [respiratory_rate] <22> [temperature] <37.6> [fio2] <0.40> [vent_on] [SEP]
[HOUR_REL_-10] [heart_rate] <70> [systolic_bp] <116> [diastolic_bp] <55> [mean_arterial_pressure] <73> [respiratory_rate] <22> [temperature] <37.5> [vent_on] [SEP]
[HOUR_REL_-9] [heart_rate] <67> [systolic_bp] <120> [diastolic_bp] <55> [mean_arterial_pressure] <73> [respiratory_rate] <22> [temperature] <37.4> [vent_on] [SEP]
[HOUR_REL_-8] [heart_rate] <66> [systolic_bp] <111> [diastolic_bp] <52> [mean_arterial_pressure] <70> [respiratory_rate] <22> [temperature] <37.4> [vent_on] [SEP]
[HOUR_REL_-7] [heart_rate] <65> [systolic_bp] <140> [diastolic_bp] <55> [mean_arterial_pressure] <81> [respiratory_rate] <22> [temperature] <37.3> [fio2] <0.40> [vent_on] [SEP]
[HOUR_REL_-6] [heart_rate] <65> [systolic_bp] <132> [diastolic_bp] <57> [mean_arterial_pressure] <80> [respiratory_rate] <22> [temperature] <37.1> [vent_on] [SEP]
[HOUR_REL_-5] [heart_rate] <69> [systolic_bp] <136> [diastolic_bp] <50> [mean_arterial_pressure] <73> [respiratory_rate] <22> [temperature] <37.0> [vent_on] [SEP]
[HOUR_REL_-4] [heart_rate] <67> [systolic_bp] <133> [diastolic_bp] <50> [mean_arterial_pressure] <71> [respiratory_rate] <22> [temperature] <37.0> [vent_on] [SEP]
[HOUR_REL_-3] [heart_rate] <68> [systolic_bp] <126> [diastolic_bp] <48> [mean_arterial_pressure] <68> [respiratory_rate] <22> [temperature] <37.0> [fio2] <0.40> [vent_on] [SEP]
[HOUR_REL_-2] [heart_rate] <68> [systolic_bp] <125> [diastolic_bp] <49> [mean_arterial_pressure] <67> [respiratory_rate] <22> [temperature] <37.1> [vent_on] [SEP]
[HOUR_REL_-1] [heart_rate] <71> [systolic_bp] <100> [diastolic_bp] <47> [mean_arterial_pressure] <61> [respiratory_rate] <22> [temperature] <37.2> [vent_on] [SEP]
[HOUR_REL_0] [heart_rate] <68> [systolic_bp] <139> [diastolic_bp] <56> [mean_arterial_pressure] <77> [respiratory_rate] <22> [temperature] <37.2> [vent_on] [CUR] [SEP]
```

## day_sample_03 — hadm 27434217 / stay 33485623

Anchor: ICU day 3 hour 12; observed_until `2130-04-15 06:02:00`

```text
[STATIC] [age_bin_75_84] [sex_F] [injury_mechanism_other] [transfer_direct] [ed_linkage_no] [head_injury_no] [SEP]

[DAY_REL_-3] [day_window_len_24h] [renal_metabolic] [bicarbonate_min_bin_low] [bun_creatinine_ratio_bin_prerenal_pattern] [immune_hematologic] [wbc_bin_high] [data_quality] [core_vital_slots_none] [uop_not_measured] [SEP]

[DAY_REL_-2] [day_window_len_24h] [perfusion_shock] [lactate_48h_bin_elevated] [base_deficit_48h_bin_severe] [renal_metabolic] [bicarbonate_min_bin_low] [bun_creatinine_ratio_bin_prerenal_pattern] [immune_hematologic] [wbc_bin_high] [data_quality] [core_vital_slots_none] [uop_not_measured] [SEP]

[DAY_REL_-1] [day_window_len_24h] [renal_metabolic] [bicarbonate_min_bin_low] [bun_creatinine_ratio_bin_prerenal_pattern] [data_quality] [core_vital_slots_none] [uop_not_measured] [SEP]

[DAY_REL_0] [day_window_len_13h] [renal_metabolic] [bicarbonate_min_bin_low] [bun_creatinine_ratio_bin_prerenal_pattern] [immune_hematologic] [wbc_bin_high] [data_quality] [core_vital_slots_none] [uop_not_measured] [SEP]

[HOUR_REL_-23] [SEP]
[HOUR_REL_-22] [SEP]
[HOUR_REL_-21] [SEP]
[HOUR_REL_-20] [SEP]
[HOUR_REL_-19] [SEP]
[HOUR_REL_-18] [SEP]
[HOUR_REL_-17] [SEP]
[HOUR_REL_-16] [SEP]
[HOUR_REL_-15] [SEP]
[HOUR_REL_-14] [SEP]
[HOUR_REL_-13] [SEP]
[HOUR_REL_-12] [SEP]
[HOUR_REL_-11] [SEP]
[HOUR_REL_-10] [SEP]
[HOUR_REL_-9] [SEP]
[HOUR_REL_-8] [SEP]
[HOUR_REL_-7] [SEP]
[HOUR_REL_-6] [SEP]
[HOUR_REL_-5] [SEP]
[HOUR_REL_-4] [SEP]
[HOUR_REL_-3] [SEP]
[HOUR_REL_-2] [SEP]
[HOUR_REL_-1] [SEP]
[HOUR_REL_0] [CUR] [SEP]
```

## day_sample_04 — hadm 27851010 / stay 37582802

Anchor: ICU day 3 hour 12; observed_until `2131-06-15 02:33:00`

```text
[STATIC] [age_bin_75_84] [sex_F] [injury_mechanism_blunt] [transfer_direct] [ed_linkage_no] [head_injury_no] [SEP]

[DAY_REL_-3] [day_window_len_24h] [perfusion_shock] [map_low_hours_bin_prolonged] [systolic_bp_min_bin_low] [oxygenation_ventilation] [respiratory_rate_high_hours_bin_brief] [renal_metabolic] [bicarbonate_min_bin_low] [immune_hematologic] [wbc_bin_high] [rbc_transfusion_event_present] [data_quality] [core_vital_slots_dense] [uop_measured] [SEP]

[DAY_REL_-2] [day_window_len_24h] [perfusion_shock] [map_low_hours_bin_prolonged] [systolic_bp_min_bin_hypotension] [heart_rate_max_bin_extreme_tachycardia] [lactate_48h_bin_elevated] [base_deficit_48h_bin_moderate] [oxygenation_ventilation] [vent_hours_bin_most_window] [vent_course_bin_first_day] [fio2_max_bin_very_high_support] [respiratory_rate_high_hours_bin_prolonged] [renal_metabolic] [bicarbonate_min_bin_low] [bun_creatinine_ratio_bin_prerenal_pattern] [immune_hematologic] [wbc_bin_low] [rbc_transfusion_event_present] [resuscitation_burden] [rbc_48h_event_present] [data_quality] [core_vital_slots_dense] [uop_measured] [SEP]

[DAY_REL_-1] [day_window_len_24h] [perfusion_shock] [map_low_hours_bin_persistent] [systolic_bp_min_bin_low] [oxygenation_ventilation] [vent_hours_bin_full_window] [vent_course_bin_early] [fio2_max_bin_high_support] [respiratory_rate_high_hours_bin_intermediate] [renal_metabolic] [creatinine_change_bin_kdigo_delta] [bicarbonate_min_bin_low] [bun_creatinine_ratio_bin_prerenal_pattern] [data_quality] [core_vital_slots_dense] [uop_measured] [SEP]

[DAY_REL_0] [day_window_len_13h] [perfusion_shock] [map_low_hours_bin_intermittent] [oxygenation_ventilation] [vent_hours_bin_full_window] [vent_course_bin_early] [fio2_max_bin_high_support] [respiratory_rate_high_hours_bin_intermediate] [renal_metabolic] [bicarbonate_min_bin_low] [bun_creatinine_ratio_bin_prerenal_pattern] [data_quality] [core_vital_slots_dense] [uop_measured] [SEP]

[HOUR_REL_-23] [heart_rate] <53> [systolic_bp] <117> [diastolic_bp] <35> [mean_arterial_pressure] <56> [respiratory_rate] <7> [vent_on] [SEP]
[HOUR_REL_-22] [heart_rate] <53> [systolic_bp] <112> [diastolic_bp] <36> [mean_arterial_pressure] <56> [respiratory_rate] <20> [temperature] <36.9> [vent_on] [SEP]
[HOUR_REL_-21] [heart_rate] <53> [systolic_bp] <131> [diastolic_bp] <37> [mean_arterial_pressure] <64> [respiratory_rate] <17> [fio2] <0.60> [vent_on] [SEP]
[HOUR_REL_-20] [heart_rate] <54> [systolic_bp] <139> [diastolic_bp] <37> [mean_arterial_pressure] <65> [respiratory_rate] <19> [vent_on] [SEP]
[HOUR_REL_-19] [heart_rate] <55> [systolic_bp] <119> [diastolic_bp] <32> [mean_arterial_pressure] <56> [respiratory_rate] <10> [vent_on] [SEP]
[HOUR_REL_-18] [heart_rate] <59> [systolic_bp] <119> [diastolic_bp] <44> [mean_arterial_pressure] <64> [respiratory_rate] <10> [temperature] <37.1> [fio2] <0.50> [vent_on] [SEP]
[HOUR_REL_-17] [heart_rate] <79> [systolic_bp] <124> [diastolic_bp] <33> [mean_arterial_pressure] <58> [respiratory_rate] <9> [vent_on] [SEP]
[HOUR_REL_-16] [heart_rate] <55> [systolic_bp] <139> [diastolic_bp] <39> [mean_arterial_pressure] <71> [respiratory_rate] <10> [fio2] <0.50> [vent_on] [SEP]
[HOUR_REL_-15] [heart_rate] <39> [systolic_bp] <132> [diastolic_bp] <34> [mean_arterial_pressure] <63> [respiratory_rate] <20> [fio2] <0.60> [vent_on] [SEP]
[HOUR_REL_-14] [heart_rate] <50> [systolic_bp] <151> [diastolic_bp] <41> [mean_arterial_pressure] <71> [respiratory_rate] <27> [temperature] <37.5> [fio2] <0.60> [vent_on] [SEP]
[HOUR_REL_-13] [heart_rate] <51> [systolic_bp] <145> [diastolic_bp] <33> [mean_arterial_pressure] <63> [respiratory_rate] <26> [vent_on] [SEP]
[HOUR_REL_-12] [heart_rate] <52> [systolic_bp] <143> [diastolic_bp] <37> [mean_arterial_pressure] <66> [respiratory_rate] <26> [vent_on] [SEP]
[HOUR_REL_-11] [heart_rate] <51> [systolic_bp] <134> [diastolic_bp] <35> [mean_arterial_pressure] <62> [respiratory_rate] <21> [vent_on] [SEP]
[HOUR_REL_-10] [heart_rate] <51> [systolic_bp] <127> [diastolic_bp] <33> [mean_arterial_pressure] <58> [respiratory_rate] <26> [temperature] <37.6> [fio2] <0.60> [vent_on] [SEP]
[HOUR_REL_-9] [heart_rate] <60> [systolic_bp] <130> [diastolic_bp] <35> [mean_arterial_pressure] <62> [respiratory_rate] <26> [vent_on] [SEP]
[HOUR_REL_-8] [heart_rate] <60> [systolic_bp] <126> [diastolic_bp] <38> [mean_arterial_pressure] <63> [respiratory_rate] <26> [vent_on] [SEP]
[HOUR_REL_-7] [heart_rate] <57> [systolic_bp] <126> [diastolic_bp] <38> [mean_arterial_pressure] <63> [respiratory_rate] <21> [vent_on] [SEP]
[HOUR_REL_-6] [heart_rate] <51> [systolic_bp] <124> [diastolic_bp] <35> [mean_arterial_pressure] <59> [respiratory_rate] <26> [temperature] <37.5> [fio2] <0.50> [vent_on] [SEP]
[HOUR_REL_-5] [heart_rate] <62> [systolic_bp] <147> [diastolic_bp] <39> [mean_arterial_pressure] <71> [respiratory_rate] <23> [vent_on] [SEP]
[HOUR_REL_-4] [heart_rate] <59> [systolic_bp] <127> [diastolic_bp] <35> [mean_arterial_pressure] <61> [respiratory_rate] <23> [vent_on] [SEP]
[HOUR_REL_-3] [heart_rate] <57> [systolic_bp] <131> [diastolic_bp] <36> [mean_arterial_pressure] <64> [respiratory_rate] <25> [fio2] <0.50> [vent_on] [SEP]
[HOUR_REL_-2] [heart_rate] <57> [systolic_bp] <131> [diastolic_bp] <38> [mean_arterial_pressure] <65> [respiratory_rate] <10> [temperature] <37.6> [vent_on] [SEP]
[HOUR_REL_-1] [heart_rate] <58> [systolic_bp] <142> [diastolic_bp] <40> [mean_arterial_pressure] <70> [respiratory_rate] <0> [vent_on] [SEP]
[HOUR_REL_0] [heart_rate] <61> [systolic_bp] <140> [diastolic_bp] <40> [mean_arterial_pressure] <70> [respiratory_rate] <9> [vent_on] [CUR] [SEP]
```

## day_sample_06 — hadm 29079804 / stay 35580228

Anchor: ICU day 3 hour 12; observed_until `2188-12-17 11:25:00`

```text
[STATIC] [age_bin_55_64] [sex_M] [injury_mechanism_other] [transfer_direct] [ed_linkage_no] [head_injury_yes] [SEP]

[DAY_REL_-3] [day_window_len_24h] [perfusion_shock] [map_low_hours_bin_brief] [systolic_bp_min_bin_hypotension] [heart_rate_max_bin_extreme_tachycardia] [oxygenation_ventilation] [vent_hours_bin_full_window] [vent_course_bin_first_day] [fio2_max_bin_very_high_support] [respiratory_rate_high_hours_bin_prolonged] [renal_metabolic] [bicarbonate_min_bin_low] [bun_creatinine_ratio_bin_prerenal_pattern] [immune_hematologic] [wbc_bin_high] [data_quality] [core_vital_slots_partial] [uop_measured] [SEP]

[DAY_REL_-2] [day_window_len_24h] [perfusion_shock] [lactate_48h_bin_elevated] [base_deficit_48h_bin_severe] [oxygenation_ventilation] [vent_hours_bin_full_window] [vent_course_bin_early] [fio2_max_bin_high_support] [respiratory_rate_high_hours_bin_prolonged] [renal_metabolic] [creatinine_change_bin_kdigo_delta] [creatinine_ratio_bin_kdigo_ratio] [bicarbonate_min_bin_low] [immune_hematologic] [wbc_bin_low] [data_quality] [core_vital_slots_sparse] [uop_measured] [SEP]

[DAY_REL_-1] [day_window_len_24h] [oxygenation_ventilation] [vent_hours_bin_full_window] [vent_course_bin_early] [fio2_max_bin_high_support] [respiratory_rate_high_hours_bin_prolonged] [renal_metabolic] [creatinine_change_bin_kdigo_delta] [bicarbonate_min_bin_low] [data_quality] [core_vital_slots_partial] [uop_measured] [SEP]

[DAY_REL_0] [day_window_len_13h] [oxygenation_ventilation] [vent_hours_bin_full_window] [vent_course_bin_prolonged] [respiratory_rate_high_hours_bin_intermediate] [renal_metabolic] [creatinine_change_bin_kdigo_delta] [bicarbonate_min_bin_low] [data_quality] [core_vital_slots_dense] [uop_measured] [SEP]

[HOUR_REL_-23] [heart_rate] <71> [systolic_bp] <148> [diastolic_bp] <66> [mean_arterial_pressure] <89> [temperature] <36.7> [fio2] <0.50> [vent_on] [SEP]
[HOUR_REL_-22] [heart_rate] <74> [systolic_bp] <128> [diastolic_bp] <60> [mean_arterial_pressure] <80> [temperature] <36.8> [vent_on] [SEP]
[HOUR_REL_-21] [heart_rate] <78> [systolic_bp] <129> [diastolic_bp] <63> [mean_arterial_pressure] <82> [temperature] <36.9> [vent_on] [SEP]
[HOUR_REL_-20] [heart_rate] <72> [systolic_bp] <121> [diastolic_bp] <58> [mean_arterial_pressure] <77> [temperature] <36.8> [vent_on] [SEP]
[HOUR_REL_-19] [heart_rate] <72> [systolic_bp] <131> [diastolic_bp] <62> [mean_arterial_pressure] <82> [temperature] <37.0> [fio2] <0.40> [vent_on] [SEP]
[HOUR_REL_-18] [heart_rate] <72> [systolic_bp] <150> [diastolic_bp] <62> [mean_arterial_pressure] <87> [temperature] <37.0> [vent_on] [SEP]
[HOUR_REL_-17] [heart_rate] <74> [systolic_bp] <127> [diastolic_bp] <56> [mean_arterial_pressure] <76> [temperature] <37.1> [vent_on] [SEP]
[HOUR_REL_-16] [heart_rate] <73> [systolic_bp] <168> [diastolic_bp] <72> [mean_arterial_pressure] <100> [temperature] <37.2> [vent_on] [SEP]
[HOUR_REL_-15] [heart_rate] <74> [systolic_bp] <139> [diastolic_bp] <64> [mean_arterial_pressure] <86> [temperature] <37.2> [fio2] <0.40> [vent_on] [SEP]
[HOUR_REL_-14] [heart_rate] <73> [systolic_bp] <164> [diastolic_bp] <71> [mean_arterial_pressure] <97> [temperature] <37.3> [vent_on] [SEP]
[HOUR_REL_-13] [heart_rate] <71> [systolic_bp] <151> [diastolic_bp] <65> [mean_arterial_pressure] <90> [temperature] <37.3> [vent_on] [SEP]
[HOUR_REL_-12] [heart_rate] <73> [systolic_bp] <139> [diastolic_bp] <61> [mean_arterial_pressure] <85> [temperature] <37.3> [fio2] <0.40> [vent_on] [SEP]
[HOUR_REL_-11] [heart_rate] <75> [systolic_bp] <146> [diastolic_bp] <64> [mean_arterial_pressure] <89> [temperature] <37.3> [vent_on] [SEP]
[HOUR_REL_-10] [heart_rate] <78> [systolic_bp] <129> [diastolic_bp] <55> [mean_arterial_pressure] <77> [temperature] <37.2> [vent_on] [SEP]
[HOUR_REL_-9] [heart_rate] <74> [systolic_bp] <160> [diastolic_bp] <48> [mean_arterial_pressure] <103> [temperature] <37.2> [vent_on] [SEP]
[HOUR_REL_-8] [heart_rate] <74> [systolic_bp] <131> [diastolic_bp] <57> [mean_arterial_pressure] <79> [temperature] <37.1> [vent_on] [SEP]
[HOUR_REL_-7] [heart_rate] <74> [systolic_bp] <156> [diastolic_bp] <65> [mean_arterial_pressure] <91> [temperature] <36.8> [fio2] <0.40> [vent_on] [SEP]
[HOUR_REL_-6] [heart_rate] <77> [systolic_bp] <109> [diastolic_bp] <49> [mean_arterial_pressure] <66> [temperature] <37.0> [vent_on] [SEP]
[HOUR_REL_-5] [heart_rate] <90> [systolic_bp] <155> [diastolic_bp] <67> [mean_arterial_pressure] <94> [temperature] <37.2> [vent_on] [SEP]
[HOUR_REL_-4] [heart_rate] <85> [systolic_bp] <158> [diastolic_bp] <70> [mean_arterial_pressure] <100> [temperature] <37.1> [fio2] <0.40> [vent_on] [SEP]
[HOUR_REL_-3] [heart_rate] <82> [systolic_bp] <145> [diastolic_bp] <65> [mean_arterial_pressure] <91> [respiratory_rate] <36> [temperature] <37.0> [vent_on] [SEP]
[HOUR_REL_-2] [heart_rate] <88> [systolic_bp] <171> [diastolic_bp] <71> [mean_arterial_pressure] <103> [respiratory_rate] <36> [temperature] <37.0> [vent_on] [SEP]
[HOUR_REL_-1] [heart_rate] <72> [systolic_bp] <133> [diastolic_bp] <73> [mean_arterial_pressure] <87> [respiratory_rate] <36> [temperature] <36.9> [vent_on] [SEP]
[HOUR_REL_0] [heart_rate] <87> [systolic_bp] <204> [diastolic_bp] <113> [mean_arterial_pressure] <136> [respiratory_rate] <26> [temperature] <37.0> [vent_on] [CUR] [SEP]
```

## day_sample_02 — hadm 29423991 / stay 30481610

Anchor: ICU day 3 hour 12; observed_until `2127-09-07 20:08:22`

```text
[STATIC] [age_bin_18_39] [sex_M] [injury_mechanism_blunt] [transfer_direct] [ed_linkage_yes] [initial_ed_sbp] <187> [initial_ed_sbp_bin_not_low] [reverse_shock_index] <1.85> [reverse_shock_index_bin_low_risk] [head_injury_no] [SEP]

[DAY_REL_-3] [day_window_len_24h] [perfusion_shock] [map_low_hours_bin_brief] [systolic_bp_min_bin_hypotension] [heart_rate_max_bin_extreme_tachycardia] [oxygenation_ventilation] [vent_hours_bin_partial_window] [vent_course_bin_first_day] [fio2_max_bin_very_high_support] [respiratory_rate_high_hours_bin_prolonged] [renal_metabolic] [bicarbonate_min_bin_low] [immune_hematologic] [wbc_bin_high] [data_quality] [core_vital_slots_partial] [uop_measured] [SEP]

[DAY_REL_-2] [day_window_len_24h] [perfusion_shock] [map_low_hours_bin_intermittent] [systolic_bp_min_bin_hypotension] [lactate_48h_bin_severe] [base_deficit_48h_bin_mild] [oxygenation_ventilation] [vent_hours_bin_full_window] [vent_course_bin_early] [fio2_max_bin_high_support] [respiratory_rate_high_hours_bin_prolonged] [renal_metabolic] [creatinine_change_bin_kdigo_delta] [creatinine_ratio_bin_kdigo_ratio] [bicarbonate_min_bin_low] [immune_hematologic] [wbc_bin_high] [data_quality] [core_vital_slots_dense] [uop_measured] [SEP]

[DAY_REL_-1] [day_window_len_24h] [perfusion_shock] [systolic_bp_min_bin_low] [oxygenation_ventilation] [vent_hours_bin_full_window] [vent_course_bin_early] [fio2_max_bin_high_support] [respiratory_rate_high_hours_bin_prolonged] [renal_metabolic] [bicarbonate_min_bin_low] [immune_hematologic] [wbc_bin_high] [data_quality] [core_vital_slots_partial] [uop_measured] [SEP]

[DAY_REL_0] [day_window_len_13h] [oxygenation_ventilation] [vent_hours_bin_full_window] [vent_course_bin_prolonged] [respiratory_rate_high_hours_bin_prolonged] [immune_hematologic] [wbc_bin_high] [data_quality] [core_vital_slots_dense] [uop_sparse] [SEP]

[HOUR_REL_-23] [heart_rate] <97> [systolic_bp] <129> [diastolic_bp] <70> [mean_arterial_pressure] <88> [respiratory_rate] <28> [vent_on] [SEP]
[HOUR_REL_-22] [heart_rate] <96> [systolic_bp] <135> [diastolic_bp] <75> [mean_arterial_pressure] <93> [respiratory_rate] <28> [vent_on] [SEP]
[HOUR_REL_-21] [heart_rate] <101> [systolic_bp] <132> [diastolic_bp] <71> [mean_arterial_pressure] <90> [respiratory_rate] <28> [vent_on] [SEP]
[HOUR_REL_-20] [heart_rate] <103> [systolic_bp] <129> [diastolic_bp] <67> [mean_arterial_pressure] <87> [respiratory_rate] <28> [temperature] <37.2> [fio2] <0.40> [vent_on] [SEP]
[HOUR_REL_-19] [heart_rate] <101> [systolic_bp] <127> [diastolic_bp] <75> [mean_arterial_pressure] <92> [respiratory_rate] <28> [vent_on] [SEP]
[HOUR_REL_-18] [vent_on] [SEP]
[HOUR_REL_-17] [vent_on] [SEP]
[HOUR_REL_-16] [heart_rate] <97> [systolic_bp] <115> [diastolic_bp] <61> [mean_arterial_pressure] <79> [respiratory_rate] <28> [temperature] <37.2> [fio2] <0.50> [vent_on] [SEP]
[HOUR_REL_-15] [heart_rate] <102> [systolic_bp] <124> [diastolic_bp] <66> [mean_arterial_pressure] <85> [respiratory_rate] <28> [vent_on] [SEP]
[HOUR_REL_-14] [heart_rate] <96> [systolic_bp] <117> [diastolic_bp] <65> [mean_arterial_pressure] <82> [respiratory_rate] <28> [vent_on] [SEP]
[HOUR_REL_-13] [heart_rate] <96> [systolic_bp] <119> [diastolic_bp] <65> [mean_arterial_pressure] <83> [respiratory_rate] <28> [vent_on] [SEP]
[HOUR_REL_-12] [heart_rate] <104> [systolic_bp] <140> [diastolic_bp] <83> [mean_arterial_pressure] <103> [respiratory_rate] <25> [temperature] <37.2> [fio2] <0.40> [vent_on] [SEP]
[HOUR_REL_-11] [heart_rate] <101> [systolic_bp] <114> [diastolic_bp] <61> [mean_arterial_pressure] <78> [respiratory_rate] <28> [vent_on] [SEP]
[HOUR_REL_-10] [heart_rate] <98> [systolic_bp] <117> [diastolic_bp] <62> [mean_arterial_pressure] <80> [respiratory_rate] <28> [vent_on] [SEP]
[HOUR_REL_-9] [heart_rate] <99> [systolic_bp] <120> [diastolic_bp] <65> [mean_arterial_pressure] <83> [respiratory_rate] <28> [fio2] <0.40> [vent_on] [SEP]
[HOUR_REL_-8] [heart_rate] <100> [systolic_bp] <135> [diastolic_bp] <75> [mean_arterial_pressure] <94> [respiratory_rate] <28> [temperature] <37.2> [fio2] <0.40> [vent_on] [SEP]
[HOUR_REL_-7] [heart_rate] <99> [systolic_bp] <125> [diastolic_bp] <68> [mean_arterial_pressure] <85> [respiratory_rate] <12> [vent_on] [SEP]
[HOUR_REL_-6] [heart_rate] <102> [systolic_bp] <127> [diastolic_bp] <70> [mean_arterial_pressure] <89> [respiratory_rate] <28> [vent_on] [SEP]
[HOUR_REL_-5] [heart_rate] <97> [systolic_bp] <117> [diastolic_bp] <64> [mean_arterial_pressure] <81> [respiratory_rate] <28> [fio2] <0.40> [vent_on] [SEP]
[HOUR_REL_-4] [heart_rate] <99> [systolic_bp] <127> [diastolic_bp] <68> [mean_arterial_pressure] <87> [respiratory_rate] <28> [temperature] <37.4> [fio2] <0.40> [vent_on] [SEP]
[HOUR_REL_-3] [heart_rate] <98> [systolic_bp] <119> [diastolic_bp] <70> [mean_arterial_pressure] <85> [respiratory_rate] <28> [vent_on] [SEP]
[HOUR_REL_-2] [heart_rate] <98> [systolic_bp] <119> [diastolic_bp] <72> [mean_arterial_pressure] <88> [respiratory_rate] <28> [vent_on] [SEP]
[HOUR_REL_-1] [heart_rate] <96> [systolic_bp] <127> [diastolic_bp] <69> [mean_arterial_pressure] <86> [respiratory_rate] <28> [temperature] <37.7> [vent_on] [SEP]
[HOUR_REL_0] [heart_rate] <107> [systolic_bp] <138> [diastolic_bp] <84> [mean_arterial_pressure] <102> [respiratory_rate] <25> [fio2] <0.40> [vent_on] [CUR] [SEP]
```
