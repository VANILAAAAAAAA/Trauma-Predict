# Summary Gate Rules — concise summary rule set for DAY/SUMMARY builder

## Format

Each rule has three fields:

```text
Rule: what this gate checks.
Reliability: reliability/strength and why.
Source: evidence IDs and URLs from day_summary_evidence_chain.md.
```

## Hemodynamic

### HEM-01 MAP hypoperfusion burden
- Rule: emit `[map_low_hours]` when any hour MAP `<65` mmHg.
- Reliability: strong. Sepsis-3 consensus defines MAP 65 as perfusion floor; SSC 2021 confirms. Very stable ICU gate.
- Source: HEM-01, HEM-02. https://pubmed.ncbi.nlm.nih.gov/26903338/ ; https://pubmed.ncbi.nlm.nih.gov/34605781/

### HEM-02 SBP low signal
- Rule: emit `[systolic_bp_min]` when day-min SBP `<=100` mmHg, or when age ≥65 and day-min SBP `<110` mmHg.
- Reliability: moderate-plus. SBP ≤100 is a qSOFA screening component (Sepsis-3). Geriatric SBP <110 is supported by trauma triage literature. Pediatric/non-geriatric SBP >90/100 nuance is not yet covered by dedicated trauma hypotension guideline alone.
- Source: HEM-05, HEM-06. https://pubmed.ncbi.nlm.nih.gov/25757122/ ; https://pubmed.ncbi.nlm.nih.gov/26903338/

### HEM-03 HR extreme
- Rule: emit `[heart_rate_max]` when day-max HR `>=131` bpm.
- Reliability: strong. NEWS2 assigns this threshold the highest single-parameter score. Use only for high rate; low-rate extreme (≤40 bpm) is warning-only in current token set.
- Source: HEM-03. NEWS2 official PDF (Royal College of Physicians).

### HEM-04 Shock index
- Rule: no current DAY token. SI elevation (HR/SBP) relates to transfusion need in trauma. Direction must match project convention (reverse shock index = SBP/HR).
- Reliability: moderate. Supported by a large TraumaRegister DGU analysis.
- Source: HEM-04. https://pubmed.ncbi.nlm.nih.gov/23938104/

## Respiratory

### RESP-01 RR extreme
- Rule: emit `[respiratory_rate_max]` when day-max RR `>=25` breaths/min.
- Reliability: strong. NEWS2 high-risk red zone. Low-rate extreme (≤8 breaths/min) is warning-only in current token set.
- Source: RESP-01. NEWS2 official PDF (Royal College of Physicians).

### RESP-02 Ventilation burden
- Rule: emit `[vent_hours]` when any hour of ventilation exists in the day. Optionally emit `[vent_day_index]` as cumulative ventilated-day counter.
- Reliability: strong. Ventilation is a definitive respiratory support marker. Berlin Definition and critical-care context both require ventilation support as context.
- Source: RESP-03. https://pubmed.ncbi.nlm.nih.gov/22797452/ ; https://pubmed.ncbi.nlm.nih.gov/34605781/

### RESP-03 FiO2 high oxygen demand
- Rule: emit `[fio2_max]` when day-max FiO2 `>0.40` fraction.
- Reliability: candidate. FiO2 alone is not an ARDS definition. This is a pragmatic oxygen-demand gate. Stronger when paired with ventilation.
- Source: RESP-04. https://pubmed.ncbi.nlm.nih.gov/22797452/

## Renal / Output

### REN-01 Creatinine AKI criterion
- Rule: emit `[creatinine_change]` when day creatinine rises `>=0.3 mg/dL` within 48 hours, or to `>=1.5×` baseline.
- Reliability: strong. KDIGO AKI stage 1 definition. Widely accepted in ICU and trauma.
- Source: REN-01, REN-02. https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf

### REN-02 Urine output AKI criterion
- Rule: if weight is available, emit `[urine_output_low_hours_kdigo]` when urine output `<0.5 mL/kg/h` for `>=6` consecutive hours. If weight is unavailable, do not emit the KDIGO UOP token.
- Reliability: strong when weight exists; source-dependent otherwise. KDIGO criterion itself is strong, but implementation requires body weight. UW source tables inspected here do not contain weight, so UW-aligned V1 cannot implement the normalized criterion without an added weight source or an explicitly documented proxy.
- Source: REN-03. https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf

## Metabolic / Acid-base

### MET-01 Lactate in hypoperfusion
- Rule: emit `[lactate_48h]` (or daily lactate max if available) when lactate `>2.0 mmol/L`. First-48h token only after window complete.
- Reliability: strong in sepsis/shock context. Sepsis-3 defines lactate >2 as hypoperfusion marker. Use caution: systemic review on lactate in broader acute admissions gives moderate support; specific trauma-lactate literature is thinner.
- Source: MET-01, MET-02, MET-03. https://pubmed.ncbi.nlm.nih.gov/26903338/ ; https://pubmed.ncbi.nlm.nih.gov/34605781/

### MET-02 Base deficit in hypovolemic shock
- Rule: emit `[base_deficit_48h]` (or daily base deficit worst, if available) when base deficit indicates moderate-to-severe metabolic acidosis. First-48h token only after window complete.
- Reliability: moderate-plus. Two large TraumaRegister DGU analyses support base-deficit classification of hypovolemic shock beyond ATLS categories. Exact bin cutpoints depend on source scale.
- Source: MET-04, MET-05. https://pubmed.ncbi.nlm.nih.gov/23497602/ ; https://pubmed.ncbi.nlm.nih.gov/23510230/

### MET-03 Bicarbonate metabolic signal
- Rule: emit `[bicarbonate_min]` when bicarbonate `<22 mEq/L`.
- Reliability: candidate. Plausible metabolic acidosis marker; not yet tied to a single strong guideline-level reference with precise cutpoint for daily summary. Use Stewart/SID framework as supporting context.
- Source: MET-06, https://ccforum.biomedcentral.com/articles/10.1186/cc2908

## Inflammatory / Hematologic

### INF-01 WBC inflammatory signal
- Rule: emit `[wbc_max]` when WBC `>12,000/mm³` or `<4,000/mm³`.
- Reliability: moderate. SIRS-defined thresholds; useful as inflammatory signal, not as modern sepsis diagnosis (Sepsis-3 de-emphasizes SIRS).
- Source: INF-01, INF-02. https://pubmed.ncbi.nlm.nih.gov/1597042/ ; https://pubmed.ncbi.nlm.nih.gov/26903338/

### INF-02 NLR complementary risk marker
- Rule: no formal gate yet. NLR can be computed as derived candidate feature. Use as auxiliary risk marker only; do not freeze gate without cohort validation.
- Reliability: low. Observational biomarker; not a clinical guideline.
- Source: INF-03. https://pmc.ncbi.nlm.nih.gov/articles/PMC6657279/

## Treatment / Resuscitation

### TX-01 Crystalloid volume burden
- Rule: do not emit a DAY crystalloid/bolus token from `>0 mL` alone. Hold `[bolus_daily_total_ml]` until treatment-resuscitation rules separate trauma resuscitation burden from maintenance fluid, drug diluent, and antibiotics-IV carrier volume.
- Reliability: hold. SSC 30 mL/kg is strong in sepsis; trauma-specific crystalloid evidence currently strongest at 24h/48h window (>5L in first 24h), and does not justify a simple `>0 mL` DAY gate.
- Source: TX-01, TX-02. https://pubmed.ncbi.nlm.nih.gov/34605781/ ; https://pubmed.ncbi.nlm.nih.gov/30400852/

### TX-02 RBC transfusion event
- Rule: emit `[rbc_daily_total_present]` whenever RBC transfusion is present (>0 mL). Volumes/buckets should use MIMIC `amount + amountuom` for audit and future bins, but the current DAY token is presence/status only.
- Reliability: strong as treatment event signal. Transfusion presence is important; high-load bin (e.g., >4 PRBC units/hour) is supported by trauma literature but needs unit conversion for MIMIC.
- Source: TX-03, TX-04, TX-05. https://pubmed.ncbi.nlm.nih.gov/26680135/ ; https://pubmed.ncbi.nlm.nih.gov/25647203/

## Data Quality / Missingness

### DQ-01 Informative missingness
- Rule: emit `[labs_measured]` or `[no_labs_measured]` for each completed day. Optionally emit `[low_vital_coverage]` or `[low_output_coverage]` when measurement density is low.
- Reliability: moderate. Informative missingness and measurement density are predictive in EHR time-series. No direct ICU clinical guideline for daily summary coverage tokens; evidence comes from EHR modeling literature.
- Source: DQ-01, DQ-02. https://www.nature.com/articles/s41598-018-24271-9 ; https://www.nature.com/articles/s41597-019-0103-9
