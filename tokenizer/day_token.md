# DAY Token Dictionary

## Structural

Every DAY block is an observed time window. Prior DAY blocks are 24h windows; `[DAY_REL_0]` is the current ICU-day window truncated at `observed_until_t`.

| Token | Rule |
|---|---|
| `[DAY_REL_0]` | current ICU-day summary from day start to prediction point; truncate every rule at `observed_until_t` |
| `[DAY_REL_-d]` | prior completed ICU-day summary; `d=1..13` before current ICU day |
| `[day_window_len_01h]` ... `[day_window_len_24h]` | emit exactly one immediately after `[DAY_REL_*]`; prior completed DAY uses `24h`; current partial DAY uses elapsed observed hours; do not emit `[DAY_REL_0]` when elapsed hours is `0` |
| `[SEP]` | end DAY block |

## Domain Markers

| Token | Rule |
|---|---|
| `[perfusion_shock]` | emit if any perfusion/shock token appears |
| `[oxygenation_ventilation]` | emit if any oxygenation/ventilation token appears |
| `[renal_metabolic]` | emit if any renal/metabolic token appears |
| `[immune_hematologic]` | emit if any immune/hematologic token appears |
| `[resuscitation_burden]` | emit if any resuscitation-burden token appears |
| `[data_quality]` | emit if any data-quality token appears |

## Perfusion / Shock

Is the patient perfusing? MAP, SBP, HR, lactate, base deficit.

| Field | Token | Rule | Evidence |
|---|---|---|---|
| `map_low_hours` | `[map_low_hours_bin_brief]` | `1–3` hours with MAP `<65` mmHg within the eligible DAY window | [HEM-01](https://pubmed.ncbi.nlm.nih.gov/26903338/) [HEM-02](https://pubmed.ncbi.nlm.nih.gov/34605781/) |
| | `[map_low_hours_bin_intermittent]` | `4–8` hours | same |
| | `[map_low_hours_bin_prolonged]` | `9–16` hours | same |
| | `[map_low_hours_bin_persistent]` | `>16` hours | same |
| `systolic_bp_min` | `[systolic_bp_min_bin_hypotension]` | window minimum SBP `<90` mmHg | [HEM-06](https://pubmed.ncbi.nlm.nih.gov/26903338/) |
| | `[systolic_bp_min_bin_low]` | window minimum SBP `90–100` mmHg | [HEM-06](https://pubmed.ncbi.nlm.nih.gov/26903338/) |
| | `[systolic_bp_min_bin_geriatric_low]` | age `>=65` and window minimum SBP `101–109` mmHg | [HEM-05](https://pubmed.ncbi.nlm.nih.gov/25757122/) [PMC4620031](https://pmc.ncbi.nlm.nih.gov/articles/PMC4620031/) |
| `heart_rate_max` | `[heart_rate_max_bin_extreme_tachycardia]` | window maximum HR `>=131` bpm | [HEM-03](https://www.rcp.ac.uk/media/a4ibkkbf/news2-final-report_0_0.pdf) |
| `lactate_48h` | `[lactate_48h_bin_elevated]` | emit once on the second ICU-day 24h window (`source_day_index=1`, first 48h fully visible); lactate `>2.0` and `<=5.0 mmol/L` | [MET-01](https://pubmed.ncbi.nlm.nih.gov/26903338/) [MET-02](https://pubmed.ncbi.nlm.nih.gov/34605781/) |
| | `[lactate_48h_bin_severe]` | emit once on the second ICU-day 24h window; lactate `>5.0 mmol/L` | same; UW threshold `>=5.1` in [uw_cat_thresholds](reference/uw_cat_thresholds.md) |
| `base_deficit_48h` | `[base_deficit_48h_bin_mild]` | emit once on the second ICU-day 24h window; base deficit `3–5.9` | [MET-04](https://pubmed.ncbi.nlm.nih.gov/23497602/) [MET-05](https://pubmed.ncbi.nlm.nih.gov/23510230/) |
| | `[base_deficit_48h_bin_moderate]` | emit once on the second ICU-day 24h window; base deficit `6–9.9` | same |
| | `[base_deficit_48h_bin_severe]` | emit once on the second ICU-day 24h window; base deficit `>=10` | same |

## Oxygenation / Ventilation

Is gas exchange adequate? Ventilation burden, FiO2, respiratory-rate burden.

| Field | Token | Rule | Evidence |
|---|---|---|---|
| `vent_hours` | `[vent_hours_bin_partial_window]` | ventilation active `>0` and `<50%` of eligible DAY-window hours | [RESP-03](https://pubmed.ncbi.nlm.nih.gov/22797452/) [RESP-03 SSC](https://pubmed.ncbi.nlm.nih.gov/34605781/) |
| | `[vent_hours_bin_most_window]` | ventilation active `>=50%` and `<100%` of eligible DAY-window hours | same |
| | `[vent_hours_bin_full_window]` | ventilation active for `100%` of eligible DAY-window hours | same |
| `vent_course` | `[vent_course_bin_first_day]` | first visible-history ventilation window with active ventilation | [RESP-03](https://pubmed.ncbi.nlm.nih.gov/22797452/) |
| | `[vent_course_bin_early]` | ventilation-window index `2–3` | same |
| | `[vent_course_bin_prolonged]` | ventilation-window index `>=4` | same |
| `fio2_max` | `[fio2_max_bin_high_support]` | window maximum FiO2 `>0.40` and `<=0.60` fraction | [RESP-04](https://pubmed.ncbi.nlm.nih.gov/22797452/) |
| | `[fio2_max_bin_very_high_support]` | window maximum FiO2 `>0.60` fraction | same |
| `respiratory_rate_high_hours` | `[respiratory_rate_high_hours_bin_brief]` | `1–3` hours with RR `>=25` /min within the eligible DAY window | [RESP-01](https://www.rcp.ac.uk/media/a4ibkkbf/news2-final-report_0_0.pdf); changed from max trigger to duration trigger |
| | `[respiratory_rate_high_hours_bin_intermediate]` | `4–8` hours with RR `>=25` /min | same |
| | `[respiratory_rate_high_hours_bin_prolonged]` | `>=9` hours with RR `>=25` /min | same |

## Renal / Metabolic

Are kidneys and metabolism working? Creatinine, BUN, urine output, bicarbonate, strong ion.

| Field | Token | Rule | Evidence |
|---|---|---|---|
| `creatinine_change` | `[creatinine_change_bin_kdigo_delta]` | window max minus prior completed-window max `>=0.3 mg/dL` | [REN-01](https://kdigo.org/guidelines/acute-kidney-injury/) [KDIGO PDF](https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf) |
| `creatinine_ratio` | `[creatinine_ratio_bin_kdigo_ratio]` | window max / prior completed-window max `>=1.5` | [REN-02](https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf) |
| `urine_output` | `[urine_output_status_kdigo_low]` | weight registry available and reliable; UOP `<0.5 mL/kg/h` for `>=6` consecutive observed hours within the eligible DAY window | [REN-03](https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf); follows weight-registry direction, not UW hard coupling |
| `bicarbonate_min` | `[bicarbonate_min_bin_low]` | window minimum bicarbonate `<22 mEq/L` | [MET-06](https://ccforum.biomedcentral.com/articles/10.1186/cc2908) |
| `bun_creatinine_ratio` | `[bun_creatinine_ratio_bin_prerenal_pattern]` | window max BUN `>20 mg/dL` and same-window BUN/Cr `>20` | [REN-04](https://www.ncbi.nlm.nih.gov/books/NBK303/); UW nofill 27.3%, MIMIC sample 41.7% |
| `strong_ion` | hold | no frozen gate; keep as audit/candidate acid-base field until a threshold is justified | [MET-06](https://ccforum.biomedcentral.com/articles/10.1186/cc2908) |

## Immune / Hematologic

Infection, inflammation, bleeding? WBC, lymphocytes, neutrophils, RBC transfusion.

| Field | Token | Rule | Evidence |
|---|---|---|---|
| `wbc` | `[wbc_bin_high]` | `>12 K/uL` or `>12000 /mm3` within the eligible DAY window | [INF-01](https://pubmed.ncbi.nlm.nih.gov/1597042/) [INF-02](https://pubmed.ncbi.nlm.nih.gov/26903338/) |
| | `[wbc_bin_low]` | `<4 K/uL` or `<4000 /mm3` within the eligible DAY window | same |
| `neutrophil_lymphocyte_ratio` | hold | candidate only; UW differential coverage is low (`any NLR` 4.9% days) and MIMIC percent-vs-absolute conversion must be audited before freezing | [INF-03](https://pmc.ncbi.nlm.nih.gov/articles/PMC6657279/) |
| `rbc_transfusion` | `[rbc_transfusion_event_present]` | RBC transfusion `>0` in the eligible DAY window | [TX-03](https://pubmed.ncbi.nlm.nih.gov/26680135/) [TX-04](https://pubmed.ncbi.nlm.nih.gov/25647203/) [TX-05](https://ccforum.biomedcentral.com/articles/10.1186/s13054-023-04327-7) |

## Resuscitation Burden

How much support? RBC, surgery, antibiotics, and early resuscitation burden. Crystalloid remains held until fluid registry is clean.

| Field | Token | Rule | Evidence |
|---|---|---|---|
| `crystalloid_48h` | hold | V1 disabled: MIMIC crystalloid, maintenance fluid, carrier/diluent fluid, and OR/PACU intake registry is not frozen; do not emit volume bins until resuscitation fluid can be separated from routine fluids | [TX-02](https://pubmed.ncbi.nlm.nih.gov/30400852/) [PMC6219036](https://pmc.ncbi.nlm.nih.gov/articles/PMC6219036/); UW thresholds remain reference-only in [uw_cat_thresholds](reference/uw_cat_thresholds.md) |
| `bolus_daily` | hold | `>0 mL` does not distinguish resuscitation from maintenance/diluent/carrier | [TX-01](https://pubmed.ncbi.nlm.nih.gov/34605781/) [TX-02](https://pubmed.ncbi.nlm.nih.gov/30400852/) |
| `rbc_48h` | `[rbc_48h_event_present]` | emit once on the second ICU-day 24h window (`source_day_index=1`) when full first 48h is visible and RBC48 `>0` | RBC event presence is a treatment signal; UW RBC48 `>0` in 47.2% patients; amount bucket remains hold |
| `rbc_48h_amount` | hold | no frozen amount bucket; no UW Cat threshold identified | [bucket evidence](reference/bucket_evidence_review.md) |
| `surg48` | `[surg48_surgeries_bin_1]` | emit once on the second ICU-day 24h window (`source_day_index=1`); 1 surgery in first 48h | UW surg48 = 0–4 integer count |
| | `[surg48_surgeries_bin_2]` | emit once; 2 surgeries | same |
| | `[surg48_surgeries_bin_3plus]` | emit once; ≥3 surgeries | same |
| `surgery_in_window` | `[surgery_in_window]` | `surgSum` changed within the eligible DAY window; cumulative surgical days updated on surgery completion, so delta>0 indicates a surgery completed in this window | UW `surgSum` step-increments on surgical day completion |
| `antibiotics_in_window` | `[antibiotics_in_window]` | antibiotics given within the eligible DAY window | UW `abx48` 0/1 but computed per-window; MIMIC `prescriptions` filtered by antibiotic formulary + window starttime |

## Data Quality

Emit exactly one `core_vital_slots` token and one `uop_measurement_status` token per DAY block. Emit `[labs_not_drawn]` only when no lab was drawn; `[labs_drawn]` is silent/default. FiO2 is excluded from core vital coverage because UW shows FiO2 has any observation on only 60.9% of days and often represents oxygen-support applicability rather than missingness.

Rules use the eligible DAY window:

```text
prior DAY window = 24h
DAY_REL_0 window = current ICU day start -> observed_until_t
core vital denominator = day_window_len_hours * 6 fields
```

| Group | Token | Rule | Data basis |
|---|---|---|---|
| `core_vital_slots` | `[core_vital_slots_dense]` | observed slots among `hr,sbp,dbp,map,rr,temp` `>=83%` of eligible slots (`>=120/144` for 24h window) | UW 39.7%; MIMIC sample 70.3%; UW p75=126/144 |
| | `[core_vital_slots_partial]` | observed slots `50–82%` of eligible slots (`72–119/144` for 24h window) | UW 30.7%; MIMIC sample 22.2% |
| | `[core_vital_slots_sparse]` | observed slots `>0` and `<50%` of eligible slots (`1–71/144` for 24h window) | UW 29.5%; MIMIC sample 6.2%; below 50% slot coverage |
| | `[core_vital_slots_none]` | observed slots `0` | UW 0.1%; MIMIC sample 1.3% |
| `lab_draw_status` | `[labs_not_drawn]` | `0` real draws among `bicarb,bun,creatinine,wbc` in the eligible DAY window | UW nofill 13.9%; MIMIC sample 3.0%; `[labs_drawn]` is silent/default because it is near-constant in MIMIC |
| `uop_measurement_status` | `[uop_measured]` | uop observed slots `>=6` within the eligible DAY window | UW 32.9%; MIMIC sample 79.6%; enough slots to evaluate 6h low-output rule |
| | `[uop_sparse]` | uop observed slots `1–5` within the eligible DAY window | UW 18.6%; MIMIC sample 8.3%; insufficient for KDIGO 6h output assessment |
| | `[uop_not_measured]` | uop observed slots `0` within the eligible DAY window | UW 48.5%; MIMIC sample 12.1% |

## V1 exclusions

Do not add broad normal/stable tokens such as `[renal_normal]` or `[shock_absent]`. Stability is represented by absence of abnormal/event tokens plus data-quality tokens. Hold fields stay in audit/source artifacts until their registry or gate is validated.
