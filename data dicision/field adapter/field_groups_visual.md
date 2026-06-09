# Feature Groups

<div style="font-family: Inter, Arial, sans-serif; max-width: 920px; line-height: 1.35;">

<div style="border: 2px solid #8FB6D9; border-radius: 16px; padding: 18px 22px; margin: 18px 0; background: #F4FAFF;">
<h2 style="margin-top: 0; color: #24577A;">G1 — Vital Signs</h2>
<table style="width: 100%; border-collapse: collapse;">
<thead><tr><th align="left">Field</th><th align="left">Unit</th><th align="left">Description</th></tr></thead>
<tbody>
<tr><td><code>hr</code></td><td>bpm</td><td>Heart rate</td></tr>
<tr><td><code>sbp</code></td><td>mmHg</td><td>Systolic blood pressure</td></tr>
<tr><td><code>dbp</code></td><td>mmHg</td><td>Diastolic blood pressure</td></tr>
<tr><td><code>map</code></td><td>mmHg</td><td>Mean arterial pressure</td></tr>
<tr><td><code>rr</code></td><td>breaths/min</td><td>Respiratory rate</td></tr>
<tr><td><code>temp</code></td><td>°C</td><td>Temperature</td></tr>
<tr><td><code>fio2</code></td><td>fraction</td><td>Fraction of inspired oxygen</td></tr>
</tbody>
</table>
</div>

<div style="border: 2px solid #A9B8CE; border-radius: 16px; padding: 18px 22px; margin: 18px 0; background: #F7F9FC;">
<h2 style="margin-top: 0; color: #3F5875;">G2 — Static Profile</h2>
<table style="width: 100%; border-collapse: collapse;">
<thead><tr><th align="left">Field</th><th align="left">Unit</th><th align="left">Description</th></tr></thead>
<tbody>
<tr><td><code>age</code></td><td>years</td><td>Age at admission</td></tr>
<tr><td><code>male</code></td><td>binary</td><td>Sex indicator; 1 = male, 0 = female</td></tr>
<tr><td><code>mechanism_cat</code></td><td>category</td><td>Injury mechanism category</td></tr>
<tr><td><code>transfer</code></td><td>category</td><td>Transfer or arrival context before ICU</td></tr>
<tr><td><code>initial_ed_sbp</code></td><td>mmHg</td><td>Initial ED systolic blood pressure</td></tr>
</tbody>
</table>
</div>

<div style="border: 2px solid #9FB0C8; border-radius: 16px; padding: 18px 22px; margin: 18px 0; background: #EEF3FA;">
<h2 style="margin-top: 0; color: #3F5875;">G2* — First 48h Summary</h2>
<table style="width: 100%; border-collapse: collapse;">
<thead><tr><th align="left">Field</th><th align="left">Unit</th><th align="left">Description</th></tr></thead>
<tbody>
<tr><td><code>base_def_48</code></td><td>mEq/L</td><td>First-48h base deficit summary</td></tr>
<tr><td><code>lactate_48</code></td><td>mmol/L</td><td>First-48h lactate summary</td></tr>
<tr><td><code>rbc_48</code></td><td>mL</td><td>First-48h RBC transfusion volume</td></tr>
<tr><td><code>crys_48</code></td><td>mL</td><td>First-48h crystalloid volume</td></tr>
</tbody>
</table>
</div>

<div style="border: 2px solid #B7A7D6; border-radius: 16px; padding: 18px 22px; margin: 18px 0; background: #FAF7FF;">
<h2 style="margin-top: 0; color: #5C4A7A;">G3 — Cumulative Exposures</h2>
<table style="width: 100%; border-collapse: collapse;">
<thead><tr><th align="left">Field</th><th align="left">Unit</th><th align="left">Description</th></tr></thead>
<tbody>
<tr><td><code>bolus_sum_until_h</code></td><td>mL</td><td>Cumulative crystalloid exposure until current hour</td></tr>
<tr><td><code>rbc_sum_until_h</code></td><td>mL</td><td>Cumulative RBC transfusion exposure until current hour</td></tr>
<tr><td><code>vent_h</code></td><td>binary</td><td>Current ventilation status</td></tr>
<tr><td><code>vent_day_sum_until_h</code></td><td>days</td><td>Cumulative ventilation days until current hour</td></tr>
</tbody>
</table>
</div>

<div style="border: 2px solid #E5B98E; border-radius: 16px; padding: 18px 22px; margin: 18px 0; background: #FFF8F1;">
<h2 style="margin-top: 0; color: #875A2C;">G4 — Laboratory Result</h2>
<table style="width: 100%; border-collapse: collapse;">
<thead><tr><th align="left">Field</th><th align="left">Unit</th><th align="left">Description</th></tr></thead>
<tbody>
<tr><td><code>bicarb</code></td><td>mEq/L</td><td>Bicarbonate</td></tr>
<tr><td><code>strong_ion</code></td><td>mEq/L</td><td>Strong ion difference proxy</td></tr>
<tr><td><code>bun</code></td><td>mg/dL</td><td>Blood urea nitrogen</td></tr>
<tr><td><code>creatinine</code></td><td>mg/dL</td><td>Creatinine</td></tr>
<tr><td><code>wbc</code></td><td>K/uL</td><td>White blood cell count</td></tr>
<tr><td><code>lymphocytes</code></td><td>K/uL</td><td>Absolute lymphocyte count</td></tr>
<tr><td><code>neutrophils</code></td><td>K/uL</td><td>Absolute neutrophil count</td></tr>
<tr><td><code>uop</code></td><td>mL/h</td><td>Urine output</td></tr>
</tbody>
</table>
</div>

</div>
