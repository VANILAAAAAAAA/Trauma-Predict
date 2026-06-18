# EHR2Path-rich Schema-v6 Report

## 1. Purpose

This experiment converts a grouped-wide trauma ICU table into EHR2Path-style structured text and evaluates whether the resulting representation can support next-hour structured prediction.

The conversion keeps the EHR2Path hierarchy as the visible interface:

```text
Emergency Department Stay -> Hospital Stay -> ICU Stay
```

The main prediction target is the next-hour structured state. The output is still text, but each field keeps its original evaluation type: scalar fields are parsed as numbers, and ventilation status is parsed as a bounded categorical state.

## 2. Field Mapping

| EHR2Path-style block | Source field family | Mapped representation | Notes |
|---|---|---|---|
| `Hospital Stay / General` | static demographics and severity | short general-context sentence | age, sex, transfer status, head injury indicator, APACHE score |
| `Emergency Department Stay / Admission Vitals` | ED baseline SBP | admission-vital text | only the available ED vital is used |
| `Hospital Stay / Lab Results` | bicarbonate, BUN, creatinine, WBC, lymphocytes, neutrophils | official-like lab names with units and normal ranges | scalar targets remain evaluated as numeric values |
| `ICU Stay / Chart Events / RoutineVitalSigns` | HR, SBP, DBP, MAP, temperature | last-24h relative-hour history | temperature is rendered in Fahrenheit for EHR2Path-style compatibility |
| `ICU Stay / Chart Events / Respiratory` | respiratory rate and FiO2 | `Respiratory Rate` and `Inspired O2 Fraction` | FiO2 is not mapped to SpO2 |
| `ICU Stay / Procedures` | ventilation status | `Invasive Ventilation` event/state text | binary source can be rendered as a bounded clinical phrase such as `present` or `absent` |
| `ICU Stay / Output` | urine output | ICU output event text | kept separate from laboratory results |
| source-native trauma blocks | cumulative bolus, RBC, ventilator days, surgery count/hours, strong ion difference | source-native sections | retained without pretending to be native MIMIC events |

For binary or simple categorical variables, the intended text representation is not a bare code. A stable form is:

```yaml
'Invasive Ventilation': 'present'
# or
'Invasive Ventilation': 'absent'
```

This is compatible with deterministic parsing as long as the vocabulary is controlled. The current evaluator already accepts numeric `0/1` and words such as `present`, `absent`, `yes`, and `no` for the ventilation field.

## 3. Evaluation Setup

- Base records: 50
- Evaluated target items: 676
- Target families: routine vitals, respiratory, labs, output, and invasive ventilation
- Scalar metrics: coverage, normalized MAE, and skill against carry-forward
- Ventilation metrics: accuracy, balanced accuracy, and carry-forward accuracy

| Mode | Input semantics | What the comparison tests |
|---|---|---|
| `text_only` | recent structured text only | whether the mapped EHR2Path-style text is directly usable |
| `summary_only` | section summary embeddings with minimal text scaffold | whether compressed section summaries can replace recent numeric text |
| `mixed` | section summary embeddings plus recent structured text | whether summaries add value on top of the recent text |

## 4. Main Results

| Mode | Evaluated targets | Scalar coverage | Macro nMAE ↓ | Carry-forward skill ↑ | Invasive ventilation acc | Invasive ventilation balanced acc | Carry-forward acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `text_only` | 676/676 | 1.000 | 0.212 | -0.710 | 0.880 | 0.775 | 0.920 |
| `summary_only` | 676/676 | 1.000 | 0.429 | -1.855 | 0.200 | 0.500 | 0.920 |
| `mixed` | 676/676 | 1.000 | 0.233 | -0.837 | 0.200 | 0.500 | 0.920 |

`text_only` is the strongest current setting. It gives the lowest scalar nMAE and the best invasive-ventilation balanced accuracy. `mixed` is close on several scalar fields but does not improve the overall result. `summary_only` is much weaker for this exact next-hour task because it removes most of the recent numeric text.

The negative carry-forward skill is important: the models generate parseable structured text, but they do not yet beat the simple baseline that copies the last observed value into the next hour.

## 5. Interpretation

- The schema-v6 conversion works as a text interface: all evaluated targets are parseable from generated text.
- The field mapping is close enough to EHR2Path-style input for text-only inference.
- The ventilation variable should be described as an invasive-ventilation status target, not as a generic binary task.
- Bounded natural-language labels for simple classes are usable, but they should stay controlled and parser-backed.
- The current result supports `text_only` as the main baseline for this dataset.

## Appendix A. Original EHR2Path Repo Sample

### A.1 Input structured text

```yaml
Emergency Department Stay:
  General: ambulance patient, female, black/african american
  Chief Complaint: headache
  Admission Vitals:
    temperature: 98F
    heartrate: 77bpm
    resprate: 16breaths/min
    o2sat: 100%
    sbp: '152'
    pain: 0/10
  Medicine Reconciliation: amlodipine, atorvastatin, cholecalciferol (vitamin D3),
    ferrous sulfate, insulin lispro, levothyroxine
  Diagnosis: headache
Hospital Stay:
  General: 'observation admit patient, 86-year old female, insurance: Other, black/african
    american, language: english'
  Patient Location: '24-16: Cardiac Vascular Intensive Care Unit (CVICU),16-0: Neuro
    Intermediate'
  Care Taker: '23-0: Neurologic Medical'
  Lab Results:
    'Anion Gap (mEq/L, normal range: 10.0-18.0)': '5: 14'
    'Bicarbonate (mEq/L, normal range: 22.0-32.0)': '5: 23'
    'Calcium, Total (mg/dL, normal range: 8.4-10.3)': '5: 9.1'
    'Chloride (mEq/L, normal range: 96.0-108.0)': '5: 101'
    'Creatinine (mg/dL, normal range: 0.4-1.1)': '5: 1.5'
    'Glucose (mg/dL, normal range: 70.0-100.0)': '5: 135'
    H: '5: 3'
    'Hematocrit (%, normal range: 34.0-45.0)': '5: 32.1'
    'Hemoglobin (g/dL, normal range: 11.2-15.7)': '5: 10.6'
    I: '5: 1'
    L: '5: 5'
    'MCH (pg, normal range: 26.0-32.0)': '5: 32.1'
    'MCHC (g/dL, normal range: 32.0-37.0)': '5: 33'
    'MCV (fL, normal range: 82.0-98.0)': '5: 97'
    'Magnesium (mg/dL, normal range: 1.6-2.6)': '5: 2'
    'Phosphate (mg/dL, normal range: 2.7-4.5)': '5: 3.8'
    'Platelet Count (K/uL, normal range: 150.0-400.0)': '5: 173'
    'Potassium (mEq/L, normal range: 3.5-5.4)': '5: 4.2'
    'RDW (%, normal range: 10.5-15.5)': '5: 12.6'
    'RDW-SD (fL, normal range: 35.1-46.3)': '5: 45.1'
    'Red Blood Cells (m/uL, normal range: 3.9-5.2)': '5: 3.3'
    'Sodium (mEq/L, normal range: 135.0-147.0)': '5: 138'
    'Urea Nitrogen (mg/dL, normal range: 6.0-20.0)': '5: 26'
    'White Blood Cells (K/uL, normal range: 4.0-10.0)': '5: 7.8'
  Prescriptions:
    levothyroxine sodium: 3-0
    insulin human: 23-0
    sodium chloride: 23-0
    potassium chloride: 23-0
    acetaminophen: 23-0
    lidocaine: 15-0
    aspirin: 23-0
    heparin sodium: 23-0
    nicardipine hydrochloride: 23-18,15-2
    glucagon hydrochloride: 23-0
    ondansetron: 23-0
    magnesium sulfate in water: 23-0
    dextrose monohydrate: 23-0
    levothyroxine sodium anhydrous: 23-16
    hydralazine hydrochloride: 23-0
ICU Stay:
  Stay 0:
    Medication:
      PO Intake: 20,17,15,5
      Insulin - Regular: 22,17
    Output:
      Void(ml): '18: 350'
    Chart Events:
      Cardiovascular:
        LLE Color: '6/3: Normal'
        LLE Temp: '6/3: Warm'
        LUE Color: '6/3: Normal'
        LUE Temp: '6/3: Warm'
        RLE Color: '6/3: Normal'
        RLE Temp: '6/3: Warm'
        RUE Color: '6/3: Normal'
        RUE Temp: '6/3: Warm'
      RoutineVitalSigns:
        Ectopy Type 1: '23: PAC''s'
        Heart Rate(bpm): '23: 85, 22: 76, 21: 88, 20: 76, 19: 74, 18: 76, 17: 80,
          16: 82, 15: 79, 14-13: 78, 11: 75, 9: 88, 7: 78, 5: 97, 4/3: 70, 2: 75,
          1: 69, 0: 68'
        Heart Rhythm: '23-11: SR (Sinus Rhythm), 9/7/5/3/1: SR (Sinus Rhythm)'
        Non Invasive Blood Pressure diastolic(mmHg): '23: 74, 22-21: 69, 19: 77.5,
          18: 77, 17: 68, 16: 90, 15: 90, 11: 67, 7: 81, 3: 73, 0: 71'
        Non Invasive Blood Pressure mean(mmHg): '23: 93, 22: 92, 21: 90, 19: 97.5,
          18: 95, 17: 98, 16: 109, 15: 100, 11: 85, 7: 103, 3: 91, 0: 94'
        Non Invasive Blood Pressure systolic(mmHg): '23: 140, 22: 142, 21: 149, 19:
          150.5, 18: 149, 17: 158, 16: 153, 15: 122, 11: 128, 7: 151, 3: 131, 0: 149'
        Temperature Fahrenheit(°F): '23: 98.5, 19: 98, 15: 97.9, 11: 98.3, 7: 98.4,
          4: 98.2'
        Temperature Site: '23: Temporal, 19/15/11: Oral, 7/4: Axillary'
      Respiratory:
        O2 saturation pulseoxymetry(%): '23: 100, 22-21: 99, 20: 100, 19-18: 98, 17:
          99, 16: 100, 15-14: 97, 13: 100, 11: 98, 9: 96, 7: 99, 5: 100, 4/3: 98,
          2/1/0: 96'
        Respiratory Rate(insp/min): '23: 16, 22: 12, 21: 18, 20: 17, 19: 14, 18: 19,
          17: 18, 3/1: 16'
      Pulmonary:
        Breathing pattern/effort: '6/3: Regular'
        Cough Effort: '3: Strong'
        Cough Type: '3: Non-productive/Congested'
        Current Dyspnea Assessment: '3: None - 0'
        LLL Lung Sounds: '6: Clear, 3: Diminished'
        LUL Lung Sounds: '6/3: Clear'
        RLL Lung Sounds: '6: Clear, 3: Diminished'
        RUL Lung Sounds: '6/3: Clear'
      Skin-Assessment:
        Braden Activity: '6: Chairfast, 3: Bedfast'
        Braden Friction/Shear: '6/3: Potential Problem'
        Braden Mobility: '6: Slight Limitations, 3: Very Limited'
        Braden Moisture: '6/3: Occasionally Moist'
        Braden Nutrition: '6: Adequate, 3: Probably Inadequate'
        Braden Sensory Perception: '6/3: Slight Impairment'
        Skin Color: '6/3: Normal for Race'
        Skin Condition: '6/3: Dry'
        Skin Integrity: '6/3: Intact'
        Skin Temperature: '6/3: Warm'
      Cardiovascular(Pulses):
        Capillary Refill L: '6/3: Normal <3 Seconds'
        Capillary Refill R: '6/3: Normal <3 Seconds'
        Dorsal PedPulse L: '6/3: Easily Palpable'
        Dorsal PedPulse R: '6/3: Easily Palpable'
        Radial Pulse L: '6/3: Easily Palpable'
        Radial Pulse R: '6/3: Easily Palpable'
      Neurological:
        Cerebellar - Finger -> Nose: '23/19/17: Normal'
        Commands: '23-17: Show 2 fingers, 15-11: Stick out tongue, 9/7/5/3/1: Stick
          out tongue'
        Commands Response: '23-11: Consistently, 9: Consistently, 7: Inconsistently,
          5: Consistently, 3/1: Inconsistently'
        Facial Droop: '23-11: No, 9/7/5/3/1: No'
        GCS - Eye Opening: '23-13: Spontaneously, 11: To Speech, 9: To Speech, 7:
          To Pain, 5: Spontaneously, 3/1: To Speech'
        GCS - Motor Response: '23-11: Obeys Commands, 9/7/5/3/1: Obeys Commands'
        GCS - Verbal Response: '23-17: Oriented, 15-11: Confused, 9/7/5/3/1: Confused'
        Neurological Symptoms: '23/19/17: Headache'
        Orientation: '23-17: Year, 15-13: State, 11: Year, 9/7/5/3/1: State'
        Pronator Drift: '15/13: No'
        Pupil Response Left: '23-11: Brisk , 9/7/5/3/1: Brisk'
        Pupil Response Right: '23-11: Brisk , 9/7/5/3/1: Brisk'
        Pupil Size Left(mm): '23-17: 3, 15-13: 4, 11: 2, 9/7/5/3/1: 2'
        Pupil Size Right(mm): '23-17: 3, 15-13: 4, 11: 3, 9/7/5/3/1: 3'
        Shoulder Shrug: '23-11: Normal, 9/7/5/3/1: Normal'
        Speech: '23-11: Normal, 9/7/5/3/1: Normal'
        Strength L Arm: '23-17: Some resistance, 15-11: Full resistance, 9/7/5/3/1:
          Full resistance'
        Strength L Leg: '23-11: Some resistance, 9/7/5/3/1: Some resistance'
        Strength R Arm: '23-17: Some resistance, 15-11: Full resistance, 9/7/5/3/1:
          Full resistance'
        Strength R Leg: '23-11: Some resistance, 9/7/5/3/1: Some resistance'
        Tongue: '23-11: Midline, 9/7/5/3/1: Midline'
        Visual Field Cut: '15-11: No, 9/7/5/3/1: No'
      Alarms:
        Alarms On: '23: 1, 19/15/11/7/3: 1'
        Heart Rate Alarm - Low(bpm): '15/11/7/3: 40'
        Heart rate Alarm - High(bpm): '15/11/7: 130, 3: 110'
        NBP Alarm Source: '15/11/7/3: Systolic'
        Non-Invasive Blood Pressure Alarm - High(mmHg): '15/11/7/3: 160'
        Non-Invasive Blood Pressure Alarm - Low(mmHg): '15/11/7/3: 90'
        O2 Saturation Pulseoxymetry Alarm - High(%): '15/11/7/3: 100'
        O2 Saturation Pulseoxymetry Alarm - Low(%): '15/11/7: 92, 3: 93'
        Parameters Checked: '23: 1, 19/15/11/7/3: 1'
        Resp Alarm - High(insp/min): '15/11/7: 35, 3: 30'
        Resp Alarm - Low(insp/min): '15/11/7/3: 8'
        ST Segment Monitoring On: '23: 1, 19/15/11/7/3: 1'
        SpO2 Desat Limit(%): '15/11/7/3: 85'
      Pain_Sedation:
        CAM-ICU MS Change: '3: No'
        CPOT-Pain Assessment Method: '19: ---, 11/5/3: CPOT'
        Daily Wake Up: '3: No, not sedated'
        Delirium assessment: '3: Negative'
        Goal Richmond-RAS Scale: '23:  0  Alert and calm, 19/15/11/7/3:  0  Alert
          and calm'
        Pain Assessment Method: '22: Patient Verbalized, 15/11: Patient Verbalized,
          7: Non-verbal Cues, 5/3: Patient Verbalized'
        Pain Cause: '22/15/5: At Rest'
        Pain Level Acceptable: '19: Tolerable'
        Pain Level Response: '19: Moderate'
        Pain Location: '22/15/5: Neck'
        Pain Management: '22: Heat Pack, 20/19/15: PO Medication, 11: Not applicable,
          5: PO Medication'
        Pain Present: '22: Yes, 15: Yes, 11/7: No, 5: Yes, 3: No'
        Pain Type: '22/15/5: Aching'
        Richmond-RAS Scale: '23:  0  Alert and calm, 19/15:  0  Alert and calm, 11:
          -1 Awakens to voice (eye opening/contact) > 10 sec, 7/3: -2 Light sedation,
          briefly awakens to voice (eye opening/contact) < 10 sec'
      GI_GU:
        Abdominal Assessment: '6/3: Soft'
        All Medications Tolerated without Adverse Side Effects: '3: Yes'
        Bowel Sounds: '6/3: Present'
        Diet Type: '6/3: House - Regular'
        Flatus: '6/3: Positive'
        Nares L: '6/3: Patent'
        Nares R: '6/3: Patent'
        Oral Cavity: '6/3: Teeth/Tissue WNL'
        Urine Appearance: '3: Clear'
        Urine Color: '6/3: Yellow'
        Urine Source: '6/3: Voids'
```

### A.2 Target / change log

```yaml
Hospital Stay:
  LOS: 751 hours
  Patient Location: Neuro Intermediate
  Care Taker: NMED
  Prescriptions:
  - levothyroxine sodium
  - insulin human
  - sodium chloride
  - potassium chloride
  - acetaminophen
  - lidocaine
  - aspirin
  - heparin sodium
  - glucagon hydrochloride
  - ondansetron
  - magnesium sulfate in water
  - dextrose monohydrate
  - hydralazine hydrochloride
ICU Stay:
  Stay 0:
    LOS: 382 hours
    Medication:
    - Insulin - Regular
    Chart Events:
      RoutineVitalSigns:
        Heart Rate: 74.0
        Heart Rhythm: SR (Sinus Rhythm)
        Non Invasive Blood Pressure diastolic: 105.0
        Non Invasive Blood Pressure mean: 111.0
        Non Invasive Blood Pressure systolic: 116.0
        Temperature Fahrenheit: 98.4
        Temperature Site: Oral
      Respiratory:
        O2 saturation pulseoxymetry: 84.0
        Respiratory Rate: 20.0
      Neurological:
        Commands: Stick out tongue
        Commands Response: Inconsistently
        Facial Droop: 'No'
        GCS - Eye Opening: Spontaneously
        GCS - Motor Response: Obeys Commands
        GCS - Verbal Response: Confused
        Orientation: State
        Pupil Response Left: 'Brisk '
        Pupil Response Right: 'Brisk '
        Pupil Size Left: '2'
        Pupil Size Right: '3'
        Shoulder Shrug: Normal
        Speech: Normal
        Strength L Arm: Full resistance
        Strength L Leg: Some resistance
        Strength R Arm: Full resistance
        Strength R Leg: Some resistance
        Tongue: Midline
        Visual Field Cut: 'No'
      Alarms:
        Alarms On: 1.0
        Heart Rate Alarm - Low: 40.0
        Heart rate Alarm - High: 110.0
        NBP Alarm Source: Systolic
        Non-Invasive Blood Pressure Alarm - High: 160.0
        Non-Invasive Blood Pressure Alarm - Low: 90.0
        O2 Saturation Pulseoxymetry Alarm - High: 100.0
        O2 Saturation Pulseoxymetry Alarm - Low: 93.0
        Parameters Checked: 1.0
        Resp Alarm - High: 30.0
        Resp Alarm - Low: 8.0
        ST Segment Monitoring On: 1.0
        SpO2 Desat Limit: 85.0
      Pain_Sedation:
        Daily Wake Up: No, not sedated
        Goal Richmond-RAS Scale: ' 0  Alert and calm'
        Pain Assessment Method: Patient Verbalized
        Pain Present: 'No'
        Richmond-RAS Scale: ' 0  Alert and calm'
```

## Appendix B. Mapped Schema-v6 Sample

### B.1 Metadata

```yaml
schema_version: schema_v6_final_add
sample_id: redacted
patient_id: redacted
cutoff_hourTally: 24
next_hourTally: 25
horizon: next_hour
```

### B.2 Input structured text

```yaml
'Hospital Stay':
  'General': '17-year old male, transfer patient, head injury documented, APACHE score
    10'
  'Lab Results':
    'Bicarbonate (mEq/L, normal range: 22.0-32.0)': '23-18: 22, 17-12: 23, 11-6: 24,
      5-0: 23'
    'Urea Nitrogen (mg/dL, normal range: 6.0-20.0)': '23-12: 7, 11-6: 6, 5-0: 5'
    'Creatinine (mg/dL, normal range: 0.5-1.2)': '23-18: 0.78, 17-12: 0.83, 11-6:
      0.66, 5-0: 0.75'
    'White Blood Cells (K/uL, normal range: 4.0-10.0)': '23-18: 11.85, 17-12: 11.56,
      11-6: 10.02, 5-0: 9.53'
    'Lymphocytes (%, normal range: 19.0-53.0)': '23-0: 1.8'
    'Neutrophils (%, normal range: 34.0-71.0)': '23-0: 8.5'
  'Source-Native Trauma Scores':
    'Revised Shock Index': '23-0: 0.8'
  'Source-Native Trauma Chemistry':
    'Strong Ion Difference(mEq/L)': '23-18: 29, 17-12: 30, 11-6: 27, 5-0: 31'
  'Source-Native Trauma Exposures':
    'Cumulative IV Fluid Bolus(ml)': '23-0: 0'
    'Cumulative Packed Red Blood Cells(units/count)': '23-0: 0'
    'Cumulative Ventilator Days': '23-0: 2'
    'Cumulative Surgery Count': '23-0: 0'
    'Cumulative Surgery Hours': '23-0: 0'
'Emergency Department Stay':
  'Admission Vitals':
    'sbp': '23-0: 113'
'ICU Stay':
  'Stay 0':
    'Chart Events':
      'RoutineVitalSigns':
        'Heart Rate(bpm)': '23: 71, 22: 70, 21: 68, 20: 67, 19/18: 68, 17/16: 70,
          15: 72, 14: 70, 13/12: 69, 11/10: 67, 9: 66, 8: 70, 7/6: 73, 5: 74, 4: 72,
          3: 70, 2/1: 75, 0: 74'
        'Non Invasive Blood Pressure systolic(mmHg)': '23: 130, 22: 133, 21: 135,
          20: 141, 19: 133, 18: 137, 17: 147, 16: 144, 15: 131, 14: 139, 13: 131,
          12: 135, 11: 127, 10: 135, 9: 134, 8: 125, 7: 131, 6: 119, 5: 122, 4/3:
          126, 2: 129, 1: 126, 0: 128'
        'Non Invasive Blood Pressure diastolic(mmHg)': '23: 55, 22: 57, 21: 60, 20:
          58, 19/18: 59, 17: 65, 16: 62, 15: 51, 14: 65, 13/12: 59, 11: 61, 10: 54,
          9: 56, 8: 53, 7: 60, 6: 55, 5/4: 56, 3: 60, 2: 57, 1/0: 58'
        'Non Invasive Blood Pressure mean(mmHg)': '23: 74, 22: 69, 21: 71, 20: 79,
          19: 74, 18: 73, 17: 84, 16: 82, 15: 67, 14: 87, 13: 78, 12: 72, 11: 77,
          10: 74, 9: 86, 8: 72, 7: 74, 6: 75, 5: 73, 4: 76, 3: 77, 2: 75, 1: 78, 0:
          75'
        'Temperature Fahrenheit(°F)': '23/22: 97.2, 21/20: 97, 19: 96.6, 18: 97, 17/16:
          97.3, 15: 97.5, 14-11: 97.3, 10: 97, 9-7: 97.3, 6/5: 97.7, 4: 98.1, 3: 98.2,
          2: 97.7, 1/0: 98.1'
      'Respiratory':
        'Respiratory Rate(insp/min)': '23-2: 14, 1: 16, 0: 14'
        'Inspired O2 Fraction': '23: 30, 17: 30, 11: 30, 5: 30, 1: 30'
    'Procedures':
      'Invasive Ventilation': '23-0'
    'Output':
      'Urine Output(ml)': '23: 100, 22: 80, 21: 150, 20: 60, 19: 115, 18: 150, 17:
        42, 16/15: 75, 14: 60, 13: 150, 12: 275, 11: 75, 10: 120, 9: 60, 7: 150, 6:
        75, 5: 200, 4: 150, 3: 225, 2: 405, 1: 370, 0: 135'
```

### B.3 Target / change log

```yaml
'ICU Stay':
  'Stay 0':
    'Chart Events':
      'RoutineVitalSigns':
        'Heart Rate': !!int '73'
        'Non Invasive Blood Pressure systolic': !!int '133'
        'Non Invasive Blood Pressure diastolic': !!int '63'
        'Non Invasive Blood Pressure mean': !!int '83'
        'Temperature Fahrenheit': !!float '97.9'
      'Respiratory':
        'Respiratory Rate': !!int '14'
        'Inspired O2 Fraction': !!int '30'
    'Procedures':
      'Invasive Ventilation': 'present'
    'Output':
      'Urine Output': !!int '225'
'Hospital Stay':
  'Lab Results':
    'Bicarbonate': !!int '23'
    'Urea Nitrogen': !!int '5'
    'Creatinine': !!float '0.9'
    'White Blood Cells': !!float '9.63'
    'Lymphocytes': !!float '1.5'
    'Neutrophils': !!float '6.6'
  'Source-Native Trauma Scores':
    'Revised Shock Index': !!float '0.8'
  'Source-Native Trauma Chemistry':
    'Strong Ion Difference': !!int '29'
  'Source-Native Trauma Exposures':
    'Cumulative IV Fluid Bolus': !!int '0'
    'Cumulative Packed Red Blood Cells': !!int '0'
    'Cumulative Ventilator Days': !!int '3'
    'Cumulative Surgery Count': !!int '0'
    'Cumulative Surgery Hours': !!int '0'
```

## Appendix C. Text-only Model Output for the Mapped Sample

### C.1 Parsed prediction state

The text-only run used target-specific prefixes. The following block is the complete parsed prediction state for the mapped sample.

```yaml
ICU Stay:
  Stay 0:
    Chart Events:
      RoutineVitalSigns:
        Heart Rate: 74
        Non Invasive Blood Pressure systolic: 126
        Non Invasive Blood Pressure diastolic: 58
        Non Invasive Blood Pressure mean: 75
        Temperature Fahrenheit: 98.1
      Respiratory:
        Respiratory Rate: 14
        Inspired O2 Fraction: 30
    Procedures:
      Invasive Ventilation: present
    Output:
      Urine Output: 135
Hospital Stay:
  Lab Results:
    Bicarbonate: 23
    Urea Nitrogen: 5
    Creatinine: 0.74
    White Blood Cells: 10
    Lymphocytes: 1
    Neutrophils: 8
```

### C.2 Raw target-wise generations

These are the raw generated text snippets before deterministic parsing.

```yaml
# target: heart_rate
'ICU Stay':
  'Stay 0':
    'Chart Events':
      'RoutineVitalSigns':
        'Heart Rate': 74.0'
        'Heart Rhythm': 'SR (Sinus

# target: nibp_systolic
'ICU Stay':
  'Stay 0':
    'Chart Events':
      'RoutineVitalSigns':
        'Non Invasive Blood Pressure systolic': 126.0'
        'Temperature Fahrenheit': '98.1

# target: nibp_diastolic
'ICU Stay':
  'Stay 0':
    'Chart Events':
      'RoutineVitalSigns':
        'Non Invasive Blood Pressure diastolic': 58.0'
        'Non Invasive Blood Pressure mean': '7

# target: nibp_mean
'ICU Stay':
  'Stay 0':
    'Chart Events':
      'RoutineVitalSigns':
        'Non Invasive Blood Pressure mean': 75.0'
        'Temperature Fahrenheit': '98.1'

# target: temperature_f
'ICU Stay':
  'Stay 0':
    'Chart Events':
      'RoutineVitalSigns':
        'Temperature Fahrenheit': 98.1'
        'Temperature Site': 'Oral'
      '

# target: resp_rate
'ICU Stay':
  'Stay 0':
    'Chart Events':
      'Respiratory':
        'Respiratory Rate': 14.0'
        'Small Volume Neb Drug/Dose #1':

# target: inspired_o2
'ICU Stay':
  'Stay 0':
    'Chart Events':
      'Respiratory':
        'Inspired O2 Fraction': 30.0'
        'O2 Delivery Device(s)': 'High flow

# target: invasive_vent
'ICU Stay':
  'Stay 0':
    'Procedures':
      'Invasive Ventilation': 1.0'
    'Chart Events':
      'RoutineVitalSigns

# target: bicarbonate
'Hospital Stay':
  'Lab Results':
    'Bicarbonate': 23.0'
    'Calcium, Total': '8.2

# target: urea_nitrogen
'Hospital Stay':
  'Lab Results':
    'Urea Nitrogen': 5.0'
    'Vancomycin': '11.1'

# target: creatinine
'Hospital Stay':
  'Lab Results':
    'Creatinine': 0.74'
    'Hematocrit': '30.

# target: white_blood_cells
'Hospital Stay':
  'Lab Results':
    'White Blood Cells': 10.0'
    'pCO2': '34.0

# target: lymphocytes
'Hospital Stay':
  'Lab Results':
    'Lymphocytes': 1.0'
    'Macrophage': '1.0'

# target: neutrophils
'Hospital Stay':
  'Lab Results':
    'Neutrophils': 8.0'
    'PTT': '30.0'

# target: urine_output
'ICU Stay':
  'Stay 0':
    'Output':
      'Urine Output': 135.0'
    'Chart Events':
      'Cardiovascular
```
