# Bucket Evidence Review

Purpose: bucket tokens are only valid when backed by evidence. UW Cat thresholds define cutpoints; clinical meanings must be marked as interpretation unless an official codebook states them.

## Evidence Levels

| Level | Meaning |
|---|---|
| A | UW threshold + external trauma/clinical literature support |
| B | UW threshold clear, external support partial |
| C | UW threshold clear, meaning is mostly internal/interpretive |
| Hold | no sufficient evidence; do not freeze bucket |

## STATIC

| Field | UW threshold | Proposed bucket token | Evidence level | Evidence / note |
|---|---|---|---|---|
| `initial_ed_sbp` | `<=89` | `[initial_ed_sbp_bin_hypotension]` | A | Traditional trauma hypotension threshold is SBP `<90 mmHg`; NTTP uses SBP `<90` as physiologic criterion. |
| `initial_ed_sbp` | `90-110` | `[initial_ed_sbp_bin_borderline_low]` | A/B | NTTP geriatric study: SBP `<110` may represent shock in age >65; 90-109 had mortality odds similar to <90 in geriatric trauma. For all ages, call it borderline/low-normal, not official mild shock. |
| `initial_ed_sbp` | `>=111` | `[initial_ed_sbp_bin_not_low]` | A/B | Complement of UW low-SBP categories; do not call universally normal for all contexts. |
| `reverse_shock_index` | `<1.1` approx | `[reverse_shock_index_bin_high_risk]` | A/B | rSI = SBP/HR = inverse of Shock Index. rSI <=1 corresponds SI >=1, a common high-risk shock marker. UW uses <=1.0 / 1.1-1.7 / >=1.8 after apparent rounding. |
| `reverse_shock_index` | `1.1-1.7` | `[reverse_shock_index_bin_intermediate]` | B | UW middle category; external SI literature supports risk increases as SI approaches/exceeds 1, but exact rSI 1.1-1.7 category is UW-derived. |
| `reverse_shock_index` | `>=1.8` | `[reverse_shock_index_bin_low_risk]` | B | UW high-rSI category; inverse SI <=~0.56 is reassuring relative to SI>=1, but label remains interpretive. |
| `age` | none accepted | none | Hold | Age bins proposed by user can be used as design strata, but require chosen reference/rationale before being called clinical buckets. |

## FIRST48

| Field | UW threshold | Proposed bucket token | Evidence level | Evidence / note |
|---|---|---|---|---|
| `base_def_48` | 0-2.9 / 3-5.9 / 6-9.9 / >=10 | normal / mild / moderate / severe | A/B | Trauma literature commonly stratifies base deficit around 3, 6, 10 as increasing shock/acidosis severity; exact UW 48h window remains dataset-specific. |
| `lactate_48` | <=2.9 / 3.0-5.0 / >=5.1 | normal_or_low / elevated / severe | B | Lactate is a trauma mortality/hypoperfusion marker; one trauma study reported outcome cutoffs around lactate 3.2 for transfusion and 5.1 for mortality. Exact UW 48h max bins are UW-derived. |
| `crys_48` | 0-1999 / 2000-4992 / 5000-9984 / >=10000 mL | low / moderate / high / very_high | B/C | External trauma evidence supports >=5L crystalloid in first 24h as harmful; 2L and 10L cutpoints are mainly UW/internal volume stratification unless further source found. |
| `rbc_48` | none in UW Cat table | none | Hold | No UW Cat threshold identified. Do not freeze RBC bucket without external evidence or explicit design decision. |

## Sources Checked

- Brown JB et al. `Systolic Blood Pressure Criteria in the National Trauma Triage Protocol for Geriatric Trauma: 110 Is the New 90`. J Trauma Acute Care Surg. 2015. PMCID: PMC4620031.
- Vella MA et al. `Acute Management of Traumatic Brain Injury`. Surg Clin North Am. 2017. PMCID: PMC5747306. Notes historical hypotension threshold SBP `<90 mmHg`.
- Jones DG et al. `Crystalloid resuscitation in trauma patients: deleterious effect of 5L or more in the first 24h`. BMC Surg. 2018. PMCID: PMC6219036.
- UW grouped-wide evidence: `/home/vanila/code/EHR-Predict/tokenizer/reference/uw_cat_thresholds.md`.

## Naming Rule

Do not name buckets as if the label is official UW semantics. Prefer neutral labels when evidence is partial:

```text
hypotension / borderline_low / not_low
high_risk / intermediate / low_risk
normal / mild / moderate / severe   # only where literature supports this severity scale
```
