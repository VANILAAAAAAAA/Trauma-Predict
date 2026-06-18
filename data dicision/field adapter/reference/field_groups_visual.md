# Feature Groups

<div style="font-family: Inter, Arial, sans-serif; max-width: 920px; line-height: 1.35;">

<div style="border: 2px solid #8FB6D9; border-radius: 16px; padding: 18px 22px; margin: 18px 0; background: #F4FAFF;">
<h2 style="margin-top: 0; color: #24577A;">G1 — Vital Signs</h2>
<table style="width: 100%; border-collapse: collapse;">
<thead><tr><th align="left">UW Field</th><th align="left">Unit</th><th align="left">Description</th></tr></thead>
<tbody>
<tr><td><code>hr</code></td><td>bpm</td><td>Heart rate</td></tr>
<tr><td><code>sbp</code></td><td>mmHg</td><td>Systolic blood pressure</td></tr>
<tr><td><code>dbp</code></td><td>mmHg</td><td>Diastolic blood pressure</td></tr>
<tr><td><code>map</code></td><td>mmHg</td><td>Mean arterial pressure</td></tr>
<tr><td><code>rr</code></td><td>breaths/min</td><td>Respiratory rate</td></tr>
<tr><td><code>temp</code></td><td>°C</td><td>Temperature</td></tr>
<tr><td><code>fio2</code></td><td>fraction</td><td>Fraction of inspired oxygen</td></tr>
<tr><td colspan="3" style="color:#6B8DA8; font-size:0.92em; padding-top:8px;"><em>→ DAY data quality:</em> <code>core_vital_slots</code> (dense/partial/sparse/none from hr,sbp,dbp,map,rr,temp)</td></tr>
</tbody>
</table>
</div>

<div style="border: 2px solid #A9B8CE; border-radius: 16px; padding: 18px 22px; margin: 18px 0; background: #F7F9FC;">
<h2 style="margin-top: 0; color: #3F5875;">G2 — Static Profile</h2>
<table style="width: 100%; border-collapse: collapse;">
<thead><tr><th align="left">UW Field</th><th align="left">Unit</th><th align="left">Description</th></tr></thead>
<tbody>
<tr><td><code>age</code></td><td>years</td><td>Age at admission</td></tr>
<tr><td><code>male</code></td><td>binary</td><td>Sex indicator; 1 = male, 0 = female</td></tr>
<tr><td><code>MechanismCat</code></td><td>category</td><td>Injury mechanism category</td></tr>
<tr><td><code>transfer</code></td><td>category</td><td>Transfer or arrival context before ICU</td></tr>
<tr><td><code>Initial.ED.SBP</code></td><td>mmHg</td><td>Initial ED systolic blood pressure</td></tr>
<tr><td><code>rSI</code></td><td>ratio</td><td>Reverse shock index = SBP / HR</td></tr>
<tr><td><code>headInjury</code></td><td>binary</td><td>Head injury from ICD (S02/S04/S06/S07/S09 / 800-854)</td></tr>
<tr><td><code>ed_linkage</code></td><td>binary</td><td>ED linkage status; when no, ED SBP and RSI omitted</td></tr>
<tr><td colspan="3" style="color:#7B8FA8; font-size:0.92em; padding-top:8px;"><em>→ STATIC only; no numeric channels. age/Initial.ED.SBP/rSI are bucket tokens:</em> <code>age_bin_*</code> <code>initial_ed_sbp_bin_*</code> <code>reverse_shock_index_bin_*</code></td></tr>
</tbody>
</table>
</div>

<div style="border: 2px solid #9FB0C8; border-radius: 16px; padding: 18px 22px; margin: 18px 0; background: #EEF3FA;">
<h2 style="margin-top: 0; color: #3F5875;">G2* — First 48h Summary</h2>
<table style="width: 100%; border-collapse: collapse;">
<thead><tr><th align="left">UW Field</th><th align="left">Unit</th><th align="left">Description</th></tr></thead>
<tbody>
<tr><td><code>baseDef48</code></td><td>mEq/L</td><td>First-48h base deficit summary</td></tr>
<tr><td><code>lactate48</code></td><td>mmol/L</td><td>First-48h lactate summary</td></tr>
<tr><td><code>RBC48</code></td><td>mL</td><td>First-48h RBC transfusion volume</td></tr>
<tr><td><code>crys48</code></td><td>mL</td><td>First-48h crystalloid volume <span style="color:#C0392B;">— hold</span></td></tr>
<tr><td><code>surg48</code></td><td>count</td><td>First-48h surgery count (1/2/3+) <span style="color:#27AE60;">— V1 FIRST48 bucket</span></td></tr>
<tr><td colspan="3" style="color:#7B8FA8; font-size:0.92em; padding-top:8px;"><em>→ DAY FIRST48 tokens (once, at source_day_index=1). baseDef48 → [base_deficit_48h_bin_*]; lactate48 → [lactate_48h_bin_*]; RBC48 → [rbc_48h_event_present]; surg48 → [surg48_surgeries_bin_*]; crys48 → hold.</em></td></tr>
</tbody>
</table>
</div>

<div style="border: 2px solid #B7A7D6; border-radius: 16px; padding: 18px 22px; margin: 18px 0; background: #FAF7FF;">
<h2 style="margin-top: 0; color: #5C4A7A;">G3 — Treatment / Exposure</h2>
<table style="width: 100%; border-collapse: collapse;">
<thead><tr><th align="left">UW Field</th><th align="left">Unit</th><th align="left">Description</th></tr></thead>
<tbody>
<tr><td><code>bolusSum</code></td><td>mL</td><td>Cumulative crystalloid exposure <span style="color:#C0392B;">— hold</span></td></tr>
<tr><td><code>RBCsum</code></td><td>mL</td><td>Cumulative RBC transfusion exposure <span style="color:#27AE60;">— DAY → [rbc_transfusion_event_present]</span></td></tr>
<tr><td><code>vent</code></td><td>binary</td><td>Current ventilation status <span style="color:#27AE60;">— HOUR sparse → [vent_on]</span></td></tr>
<tr><td><code>ventDaySum</code></td><td>days</td><td>Cumulative ventilation days <span style="color:#27AE60;">— DAY → [vent_hours_bin_*]+[vent_course_bin_*]</span></td></tr>
<tr><td><code>surgHours</code></td><td>hours</td><td>Cumulative surgery hours <span style="color:#C0392B;">— not primary for V1 [surgery_in_window]; batch-updated and unsuitable for hourly/window event detection</span></td></tr>
<tr><td><code>surgSum</code></td><td>days</td><td>Cumulative surgery days <span style="color:#27AE60;">— DAY source for [surgery_in_window] via positive window delta</span></td></tr>
<tr><td><code>abx48</code></td><td>binary</td><td>Antibiotic exposure <span style="color:#27AE60;">— DAY per-window → [antibiotics_in_window]</span></td></tr>
<tr><td colspan="3" style="color:#7B8FA8; font-size:0.92em; padding-top:8px;"><em>→ HOUR sparse: <code>[vent_on]</code> (context, not in vital tensor). DAY: vent → <code>[vent_hours_bin_*]</code> + <code>[vent_course_bin_*]</code>; RBCsum → <code>[rbc_transfusion_event_present]</code>; bolusSum → hold.</em></td></tr>
</tbody>
</table>
</div>

<div style="border: 2px solid #E5B98E; border-radius: 16px; padding: 18px 22px; margin: 18px 0; background: #FFF8F1;">
<h2 style="margin-top: 0; color: #875A2C;">G4 — Laboratory / Output</h2>
<table style="width: 100%; border-collapse: collapse;">
<thead><tr><th align="left">UW Field</th><th align="left">Unit</th><th align="left">Description</th></tr></thead>
<tbody>
<tr><td><code>bicarb</code></td><td>mEq/L</td><td>Bicarbonate <span style="color:#27AE60;">— DAY → [bicarbonate_min_bin_low]</span></td></tr>
<tr><td><code>StrongIon</code></td><td>mEq/L</td><td>Strong ion difference proxy <span style="color:#C0392B;">— hold</span></td></tr>
<tr><td><code>bun</code></td><td>mg/dL</td><td>Blood urea nitrogen</td></tr>
<tr><td><code>creatinine</code></td><td>mg/dL</td><td>Creatinine <span style="color:#27AE60;">— DAY → [creatinine_change_bin_kdigo_delta] + [creatinine_ratio_bin_kdigo_ratio]</span></td></tr>
<tr><td><code>wbc</code></td><td>K/uL</td><td>White blood cell count <span style="color:#27AE60;">— DAY → [wbc_bin_high] / [wbc_bin_low]</span></td></tr>
<tr><td><code>lymphocytes</code></td><td>K/uL</td><td>Absolute lymphocyte count <span style="color:#C0392B;">— hold (NLR)</span></td></tr>
<tr><td><code>neutrophils</code></td><td>K/uL</td><td>Absolute neutrophil count <span style="color:#C0392B;">— hold (NLR)</span></td></tr>
<tr><td><code>uop</code></td><td>mL/h</td><td>Urine output</td></tr>
<tr><td colspan="3" style="color:#7B8FA8; font-size:0.92em; padding-top:8px;"><em>→ DAY data quality: <code>lab_draw_status</code> ([labs_not_drawn] when 0 draws from bicarb,bun,creatinine,wbc) | <code>uop_measurement_status</code> (measured/sparse/not_measured)</em></td></tr>
</tbody>
</table>
</div>

<div style="border: 2px solid #C9A0DC; border-radius: 16px; padding: 18px 22px; margin: 18px 0; background: #FAF5FF;">
<h2 style="margin-top: 0; color: #6B3FA0;">CXR — Chest X-Ray (optional)</h2>
<table style="width: 100%; border-collapse: collapse;">
<thead><tr><th align="left">Field</th><th align="left">Unit</th><th align="left">Description</th></tr></thead>
<tbody>
<tr><td><code>cxr_finding</code></td><td>binary</td><td>CheXpert finding labels from MIMIC-CXR-JPG. 12 retained, positive=1.0 only. 1760/6583 HADM linked.</td></tr>
<tr><td colspan="3" style="color:#8B6FAC; font-size:0.92em; padding-top:8px;"><em>→ CXR event tokens: <code>[cxr_finding_*]</code>. Retained: Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Opacity, No Finding, Pleural Effusion, Pneumonia, Pneumothorax, Support Devices.</em></td></tr>
</tbody>
</table>
</div>

<div style="border: 2px solid #B0B0B0; border-radius: 16px; padding: 18px 22px; margin: 18px 0; background: #FAFAFA;">
<h2 style="margin-top: 0; color: #555;">Structural Tokens</h2>
<table style="width: 100%; border-collapse: collapse;">
<thead><tr><th align="left">Field</th><th align="left">Unit</th><th align="left">Description</th></tr></thead>
<tbody>
<tr><td><code>day_window_len</code></td><td>hours</td><td>24 tokens: [day_window_len_01h]...[day_window_len_24h]. Emitted second in every DAY block.</td></tr>
</tbody>
</table>
</div>

</div>
