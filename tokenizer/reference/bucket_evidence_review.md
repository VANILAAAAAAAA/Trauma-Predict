# Bucket Evidence Review

Purpose: bucket tokens are only valid when backed by evidence. UW Cat thresholds define cutpoints; clinical meanings must be marked as interpretation unless an official codebook states them.

## Evidence Levels

| Level | Meaning |
|---|---|
| A | UW threshold + external trauma/clinical literature support |
| B | UW threshold clear, external support partial |
| C | UW threshold clear, meaning is mostly internal/interpretive |
| D | Data-driven: bin thresholds from UW value distribution, no external literature |
| Hold | no sufficient evidence; do not freeze bucket |

## STATIC

| Field | UW threshold | Proposed bucket token | Evidence level | Evidence / note |
|---|---|---|---|---|
| `initial_ed_sbp` | `<=89` | `[initial_ed_sbp_bin_hypotension]` | A | Traditional trauma hypotension threshold is SBP `<90 mmHg`; NTTP uses SBP `<90` as physiologic criterion. |
| `initial_ed_sbp` | `90-110` | `[initial_ed_sbp_bin_borderline_low]` | A/B | NTTP geriatric study: SBP `<110` may represent shock in age >65. |
| `initial_ed_sbp` | `>=111` | `[initial_ed_sbp_bin_not_low]` | A/B | Complement of UW low-SBP categories. |
| `reverse_shock_index` | `<1.1` approx | `[reverse_shock_index_bin_high_risk]` | A/B | rSI = SBP/HR = inverse of Shock Index. rSI <=1 corresponds SI >=1. |
| `reverse_shock_index` | `1.1-1.7` | `[reverse_shock_index_bin_intermediate]` | B | UW middle category. |
| `reverse_shock_index` | `>=1.8` | `[reverse_shock_index_bin_low_risk]` | B | UW high-rSI category. |
| `age` | none accepted | none | Hold → Frozen | User-specified: [age_bin_18_39] through [age_bin_85_89] (6 bins). |

## FIRST48

| Field | UW threshold | Proposed bucket token | Evidence level | Evidence / note |
|---|---|---|---|---|
| `base_def_48` | 0-2.9 / 3-5.9 / 6-9.9 / >=10 | normal / mild / moderate / severe | A/B | Trauma literature stratifies base deficit around 3, 6, 10. |
| `lactate_48` | <=2.9 / 3.0-5.0 / >=5.1 | normal_or_low / elevated / severe | B | Lactate is a trauma mortality/hypoperfusion marker. |
| `crys_48` | 0-1999 / 2000-4992 / 5000-9984 / >=10000 mL | low / moderate / high / very_high | B/C | Jones et al. (2018): ≥5L in first 24h → mortality OR 2.55. |
| `rbc_48` | none in UW Cat table | none | Hold | No UW Cat threshold identified. Do not freeze RBC bucket without external evidence or explicit design decision. |

## HOUR — Treatment Amounts

MIMIC-IV HOUR V1 should not use UW source-scale bins as the primary representation. MIMIC-IV `icu.inputevents` provides interval events with `amount`/`amountuom`; crystalloid/bolus and RBC amounts are represented in mL after interval-to-hour splitting.

| Variable | Source | V1 representation | Evidence level | Evidence |
|---|---|---|---|---|
| `bolus_input_1h` | MIMIC-IV `icu.inputevents` crystalloid/bolus itemids | `[bolus_input_1h] <bolus_input_1h_ml>` | source-data | `inputevents.amountuom` is almost always `ml`; hour value is derived by splitting event intervals into ICU-hour overlap. |
| `rbc_transfusion_1h` | MIMIC-IV `icu.inputevents` RBC itemids 225168/226368/227070 | `[rbc_transfusion_1h] <rbc_transfusion_1h_ml>` | source-data | RBC inputevents are mL; common delivered amounts cluster around one-unit-like volumes but V1 keeps mL numeric. |

UW grouped-wide cumulative deltas are alignment references only:

| UW variable | Source-scale deltas | Status |
|---|---|---|
| `bolusSum` | 0.5 / 1.0 / >1.0 source-scale delta | not a MIMIC HOUR V1 bucket; unit-to-mL conversion unresolved |
| `RBCsum` | 1 / 2 / >2 source-scale delta | not a MIMIC HOUR V1 bucket; mL-equivalent conversion unresolved |

## HOUR — Vital Bins

| Variable | Evidence | Status |
|---|---|---|
| hr, sbp, dbp, map, rr, temp, fio2 | pending | Hold |

## Sources Checked

- Brown JB et al. `Systolic Blood Pressure Criteria in the National Trauma Triage Protocol for Geriatric Trauma: 110 Is the New 90`. J Trauma Acute Care Surg. 2015. PMCID: PMC4620031.
- Vella MA et al. `Acute Management of Traumatic Brain Injury`. Surg Clin North Am. 2017. PMCID: PMC5747306.
- Jones DG et al. `Crystalloid resuscitation in trauma patients: deleterious effect of 5L or more in the first 24h`. BMC Surg. 2018. PMCID: PMC6219036.
- UW grouped-wide data: `/home/vanila/code/ehrtopath-rich-to-structured/patient_trajectories_grouped_wide.csv`.
- UW Cat thresholds: `tokenizer/reference/uw_cat_thresholds.md`.

## Naming Rule

Do not name buckets as if the label is official UW semantics. Prefer neutral labels when evidence is partial:

```text
hypotension / borderline_low / not_low
high_risk / intermediate / low_risk
normal / mild / moderate / severe   # only where literature supports this severity scale
```
