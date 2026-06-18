# 官方 EHR2Path input 字段 vs 当前数据字段：代码证据与标量确认

日期：2026-05-04  
用途：桌面展示 / 逐字段检查 / 讨论如何改造当前数据输入  
输出原则：本文件是单独展示文件；机器可读证据覆盖到既有 `sample_cmp.json`。

```text
展示文件：/mnt/c/Users/a6439/Desktop/thesis/ehr2path-rich/docs/input_cmp.md
机器样例：/home/vanila/code/ehrtopath-rich-to-structured/artifacts/sample_cmp.json
样例版本：field_aligned_sample_cmp_v4_code_and_scalar_evidence
当前数据：/home/vanila/code/ehrtopath-rich-to-structured/patient_trajectories_grouped_wide.csv
官方 processed：/mnt/d/Data/mimic-iv-2.2/all_data_24_hours_los_noisy
官方代码：/home/vanila/code/ehrtopath
```

---

## 1. 当前结论

1. `Hospital Stay` 里 lab 很多、`ICU Stay` 里主要是 `Chart Events`，不是从单个 sample 主观猜出来的；这是官方代码结构决定的。
2. 具体出现频率会受 processed subset/timepoint selection 影响，所以本文件同时给出代码证据和 8000 个 official processed sample-timepoints 频率统计。
3. 当前 grouped wide 数据很多字段仿照 MIMIC/EHR2Path，可映射到官方 field names；不要把当前列名直接放进 prompt。
4. 标量单位按全量值域重新确认：能从值域判断的都不再放进待确认项。
5. `*Cat` 字段是 index/code：当前不进入 `desc/change_log` prompt。


---

## 2. 官方 section 不是 sample bias：代码证据

|Evidence|Code location|What it proves|
|---|---|---|
|Wrapper|patient_model/retrieve_patient_model.py:509-546|Adds Patient_ED → Emergency Department Stay; Patient_ADM → Hospital Stay; Patient_ICU → ICU Stay / Stay i.|
|ED sections|mimic_iv_extraction/extract_patient_data.py:1149-1206|Generates General, Chief Complaint, Admission Vitals, Medicine Reconciliation, Vital Measurements, Medication, optional Diagnosis. ED temperature units are hard-coded as F in ED triage/vitals.|
|Hospital sections|mimic_iv_extraction/extract_patient_data.py:646-791|Generates General, Patient Location, Care Taker, Outpatient Measurements, Lab Results, Microbiology Growth Results, Prescriptions, Procedures, Radiology Notes.|
|ICU sections|mimic_iv_extraction/extract_patient_data.py:928-1018|Generates Medication, Output, Procedures, Chart Events; chart event categories come from d_items-style categories.|
|Prompt assembly|patient_model/dataset.py / TextDataset|Model input is yaml.dump(sample["desc"], sort_keys=False); target is yaml.dump(sample["change_log"], sort_keys=False).|

`get_patient_description()` 明确把 ED/Hospital/ICU 三类对象分别包装为 `Emergency Department Stay`、`Hospital Stay`、`ICU Stay / Stay i`。因此 hospital labs 与 ICU chart events 的分布是代码设计，不只是 sample 偏差。ICU 也有 `Medication`、`Output`、`Procedures`，只是 chart events leaf 更丰富。


---

## 3. 官方 processed subset 频率：判断 sample 是否 biased

统计范围：`4553` processed JSON files, `8000` sample-timepoints。


### 3.1 desc top sections

|Top section|Count|Rate|
|---|---|---|
|Hospital Stay|7365|92.1%|
|Emergency Department Stay|3857|48.2%|
|ICU Stay|1248|15.6%|


### 3.2 desc second-level sections, top 18

|Path|Count|
|---|---|
|Hospital Stay/General|7365|
|Hospital Stay/Patient Location|7336|
|Hospital Stay/Care Taker|7320|
|Hospital Stay/Prescriptions|7027|
|Hospital Stay/Lab Results|5659|
|Emergency Department Stay/General|3857|
|Emergency Department Stay/Chief Complaint|3857|
|Emergency Department Stay/Admission Vitals|3857|
|Emergency Department Stay/Diagnosis|3247|
|Emergency Department Stay/Medicine Reconciliation|3010|
|Hospital Stay/Radiology Notes|2558|
|Hospital Stay/Procedures|2069|
|Hospital Stay/Microbiology Growth Results|1366|
|ICU Stay/Stay 0|1066|
|Emergency Department Stay/Vital Measurements|1024|
|Hospital Stay/Outpatient Measurements|933|
|Emergency Department Stay/Medication|739|
|ICU Stay/Stay 1|143|


### 3.3 ICU desc sections

|ICU section|Count among ICU-containing desc|
|---|---|
|Chart Events|1249|
|Medication|1226|
|Output|1178|
|Procedures|1040|


### 3.4 ICU Chart Events categories, top 18

|ICU Chart Events category|Count|
|---|---|
|RoutineVitalSigns|1242|
|Respiratory|1242|
|Pain_Sedation|1214|
|Alarms|1211|
|Pulmonary|1198|
|GI_GU|1197|
|Neurological|1194|
|Skin-Assessment|1166|
|Cardiovascular(Pulses)|1146|
|Cardiovascular|971|
|Skin-Incisions|855|
|Skin-Impairment|625|
|AdmHistory_FHPA|318|
|Hemodynamics|264|
|Cardiovascular(PacerData)|116|
|Dialysis|75|
|Toxicology|39|
|MDProgressNote|18|


### 3.5 Temperature path evidence

|Official processed temperature path|Count|
|---|---|
|Emergency Department Stay/Admission Vitals/temperature|3342|
|ICU Stay/Stay 0/Chart Events/RoutineVitalSigns/Temperature Site|1276|
|ICU Stay/Stay 0/Chart Events/Skin-Assessment/Skin Temperature|1096|
|ICU Stay/Stay 0/Chart Events/RoutineVitalSigns/Temperature Fahrenheit(°F)|972|
|Emergency Department Stay/Vital Measurements/temperature (F)|966|
|Hospital Stay/Lab Results/Temperature|228|
|ICU Stay/Stay 0/Chart Events/RoutineVitalSigns/Temperature Fahrenheit|212|
|ICU Stay/Stay 1/Chart Events/RoutineVitalSigns/Temperature Site|185|
|ICU Stay/Stay 1/Chart Events/Skin-Assessment/Skin Temperature|149|
|Emergency Department Stay/Vital Measurements/temperature|125|
|ICU Stay/Stay 1/Chart Events/RoutineVitalSigns/Temperature Fahrenheit(°F)|118|
|ICU Stay/Stay 0/Chart Events/RoutineVitalSigns/Temperature Celsius(°C)|69|

结论：ED temperature 在官方 ED 代码里硬编码为 F；ICU d_items 同时支持 `Temperature Fahrenheit` 和 `Temperature Celsius`。当前数据若值域是 Celsius，可用官方支持的 `Temperature Celsius(°C)`；若为了更贴近高频官方 Fahrenheit leaf，可做确定性转换并记录 `F=C*9/5+32`。


---

## 4. 本地 MIMIC 字典 / 官方字段单位证据

证据来自本地 MIMIC-IV dictionary：`icu/d_items.csv.gz` 与 `hosp/d_labitems.csv.gz`。

|Source|Label|Unit / official processed unit|Section implication|
|---|---|---|---|
|icu/d_items|Heart Rate|bpm|ICU Chart Events / RoutineVitalSigns|
|icu/d_items|Non Invasive BP + Arterial BP|mmHg|官方区分 NIBP 与 arterial；当前源未区分时不要过度声明|
|icu/d_items|Respiratory Rate|insp/min|ICU Chart Events / Respiratory|
|icu/d_items|Temperature Fahrenheit / Temperature Celsius|°F / °C|Both are official-supported labels|
|icu/d_items|Inspired O2 Fraction|unitname None; processed values are percent-style|ICU Chart Events / Respiratory|
|icu/d_items/outputevents|Urine output/outputevents|mL|ICU Output|
|hosp/d_labitems + processed text|Bicarbonate|mEq/L|Hospital Lab Results|
|hosp/d_labitems + processed text|Urea Nitrogen|mg/dL|Hospital Lab Results|
|hosp/d_labitems + processed text|Creatinine|mg/dL|Hospital Lab Results|
|hosp/d_labitems + processed text|White Blood Cells|K/uL|Hospital Lab Results|
|hosp/d_labitems + processed text|Lactate|mmol/L|Hospital Lab Results|
|hosp/d_labitems + processed text|Base Excess|mEq/L when valueuom available|Not identical to current nonnegative baseDef48 without sign/summary confirmation|


---

## 5. 当前数据标量 / 单位确认 v2：值域驱动

方法：parse all *_series_json arrays and scalar numeric columns; use value domains, quantiles, integer-like proportion, field name, and MIMIC-style unit conventions。统计源：patient_trajectories_grouped_wide.csv full grouped table，`n_patients=2801`。

|Concept|Current field|Full-value evidence|Inferred unit/scalar|Confidence|Official target field|Decision|
|---|---|---|---|---|---|---|
|Heart rate|G1_vitals__hr_series_json|min 20.0, p50 91.0, p99 139.0, max 222.0|bpm|high|Heart Rate(bpm)|direct|
|SBP|sbp_series_json|min 22.0, p50 123.0, p99 173.0, max 239.0|mmHg|high|BP systolic(mmHg)|unit direct; source cautious|
|DBP|G1_vitals__dbp_series_json|min 0.0, p50 70.0, p99 105.0, max 175.0|mmHg|high|BP diastolic(mmHg)|unit direct; source cautious|
|MAP|G1_vitals__map_series_json|min 1.0, p50 87.0, p99 125.0, max 116107.0|mmHg|high|BP mean(mmHg)|unit direct; outlier QC|
|Respiratory rate|G1_vitals__rr_series_json|min 0.0, p50 18.0, p99 38.0, max 98.0|breaths/min|high|Respiratory Rate(insp/min)|direct|
|Temperature|G1_vitals__temp_series_json|min 20.0, p50 37.3, p99 39.1, max 48.5|source °C; model-facing derived °F|high|Temperature Fahrenheit(°F)|convert C→F for repo/model input compatibility; keep source trace|
|FiO2|G1_vitals__fio2_series_json|min 20.0, p50 40.0, p99 100.0, max 100.0|%|high|Inspired O2 Fraction|direct percent-style|
|Bicarbonate|G4_labs__bicarb_series_json|min 5.0, p50 27.0, p99 37.0, max 46.0|mEq/L|high|Bicarbonate (mEq/L)|direct|
|BUN|G4_labs__bun_series_json|min 1.0, p50 18.0, p99 88.0, max 213.0|mg/dL|high|Urea Nitrogen (mg/dL)|direct|
|Creatinine|G4_labs__creatinine_series_json|min 0.19, p50 0.72, p99 4.72, max 17.95|mg/dL|high|Creatinine (mg/dL)|direct|
|WBC|G4_labs__wbc_series_json|min 0.4, p50 11.48, p99 33.97, max 109.67|K/uL|high|White Blood Cells (K/uL)|direct|
|Lactate 48h|lactate48_series_json|min 0.5, p50 3.4, p99 16.0, max 28.0|mmol/L|high|First-48h Lactate (mmol/L)|derived summary; cutoff rule|
|Base deficit 48h|baseDef48_series_json|min 0.0, p50 5.2, p99 21.8, max 29.9|mEq/L deficit magnitude|medium-high|Base deficit magnitude, not direct Base Excess|summary; sign semantics|
|StrongIon|G4_labs__StrongIon_series_json|min 13.0, p50 33.0, p99 44.0, max 56.0|mEq/L|medium-high|Strong ion metric|semantic confirm|
|Urine output|G4_labs__uop_series_json|min 0.0, p50 100.0, p99 1000.0, max 10780.0|mL per recorded hour/interval|high|Output / Urine mL|unit direct; interval confirm|
|RBC48|G2_static__RBC48|min 0.0, p50 0.0, p99 28.0, max 120.0|blood product units/count|medium-high|RBC exposure units/count|confirm product semantics|
|RBCsum|G3_cumulative__RBCsum_series_json|min 0.0, p50 2.0, p99 45.0, max 264.0|blood product units/count|medium-high|Packed RBC derived exposure|confirm unit/count|
|Crystalloid 48h|crys48_series_json|min 0.0, p50 4542.0, p99 17213.0, max 26201.0|mL|high|First-48h crystalloid mL|direct summary; cutoff rule|
|BolusSum|G3_cumulative__bolusSum_series_json|min 0.0, p50 3.0, p99 26.0, max 144.5|likely liters or unit-coded cumulative bolus exposure|medium-high|Bolus exposure cumulative total/delta|appendix + monotonic raw check|
|Ventilation|vent_series_json / ventDaySum|min 0.0, p50 1.0, p99 1.0, max 1.0|binary status + cumulative ventilator days|high|Invasive Ventilation active period / cumulative days|appendix G3 confirms ventDaySum cumulative|
|Surgery|G3_cumulative__surgSum_series_json / surgHours|min 0.0, p50 1.0, p99 7.0, max 14.0|cumulative surgery count / cumulative surgery hours|high|OR/procedure cumulative exposure|appendix G3 confirms cumulative|


### 5.1 关键修正：Temperature

当前 grouped wide 的 `G1_vitals__temp_series_json` 不是 Fahrenheit：

```text
n_values = 346330
min = 20.0, p50 = 37.3, p99 = 39.1, max = 48.5
count >= 45: 1
count >= 60: 0
```

所以当前列判定为 Celsius，并对 `20.0`、`48.5` 做 QC。你举的“如果温度是 76 就不可能是 Celsius”的原则是对的；只是当前 grouped wide 这列没有 76。

但模型输入层面采用 `compat-first`：当前 v4+ schema 将源 °C **确定性转换** 为官方更高频的 `Temperature Fahrenheit(°F)`：

```text
F = C * 9/5 + 32
rounding = 1 decimal
```

依据：本地 official processed subset 中，数值型 temperature leaf 统计为 `Temperature Fahrenheit(°F)=1135` 次、`Temperature Celsius(°C)=80` 次；ED temperature 也按 F。考虑 text-only 模型是低参数 LLM，优先匹配高频 repo input 标量，source_map 保留原始 °C 与转换规则。


### 5.2 现在真正还需要确认的最小集合

|Item|Why still needs confirmation|
|---|---|
|BP source|current SBP/DBP/MAP values are mmHg, but source may be NIBP vs arterial/mixed; prompt should avoid overclaiming NIBP if source unknown.|
|baseDef48|unit and magnitude support base deficit mEq/L; confirm whether value is worst first-48h deficit and whether official Base Excess should use negative sign.|
|StrongIon|unit likely mEq/L and field resembles strong ion difference; formula/clinical interpretation still needs source confirmation before mapping to an official lab name.|
|bolusSum|distribution does not uniquely identify mL vs count/L/unit-coded cumulative exposure; keep as derived exposure until source definition is known.|
|RBCsum/RBC48|distribution supports PRBC units/count; confirm whether units, products, or mL before clinical text beyond "RBC exposure".|
|uop|unit supports mL; confirm whether hourly total, interval total, or rolling total for exact target text.|

不再需要反复确认的单位：HR=bpm、BP=mmHg、RR=breaths/min/insp/min、FiO2=percent-style、Bicarbonate=mEq/L、BUN=mg/dL、Creatinine=mg/dL、WBC=K/uL、Lactate=mmol/L、Crystalloid=mL、Vent binary、Surgery count/hour。

### 5.3 用户提供 appendix feature groups：累计字段规则

用户补充的 appendix 截图给出原始 feature group 定义，可作为当前数据 schema 的直接证据：

|Group|Appendix label|Fields|Current handling|
|---|---|---|---|
|G1|Vital Signs|`hr`, `dbp`, `map`, `rr`, `temp`, `fio2`|动态 vitals；raw table 还含 `sbp`，作为额外 BP signal 保留 source trace。|
|G2|Static Profile|`age`, `male`, `transfer`, `MechanismCat`, `headInjury`, `Initial.ED.SBPCat`, `rSICat`, `baseDef48Cat`, `lactate48Cat`, `RBC48`, `crys48Cat`, `Apache`, `abx48`, `surg48`, `er_dispCat`|静态/profile + first-48h summaries；`*Cat` 为 index/code，不进 prompt；`*48` cutoff<48 禁用。|
|G3|Cumulative Exposures|`bolusSum`, `RBCsum`, `ventDaySum`, `surgSum`, `surgHours`|**必须按累计值处理**；raw monotonic 检查也支持 within-id 非递减。|
|G4|Laboratory Result|`bicarb`, `acidosisCat`, `StrongIon`, `bun`, `creatinine`, `wbc`, `uop`|lab/source group；`acidosisCat` 不进 prompt；`uop` 虽在 G4，但语义和值域是 interval/hourly urine output，EHR2Path prompt 应放 ICU Output/Urine，并在 source_map 保留 G4 来源。|

累计字段处理规则：

```text
cumulative fields: bolusSum, RBCsum, ventDaySum, surgSum, surgHours
not cumulative: uop
first-48h summaries: baseDef48, lactate48, RBC48, crys48, abx48, surg48
```

对 next-state 样本：累计字段要明确是“截至 cutoff 的累计 exposure”。如果作为 target，必须区分预测 `next cumulative value` 还是 `next-hour delta`，不能把累计总量误当瞬时测量。


### 5.4 原始 long table `layers_2012_2019_preprocessed_noimputation.csv` 追加确认

原表位置：

```text
/mnt/d/Download/layers_2012_2019_preprocessed_noimputation.csv
```

同目录小表：

```text
/mnt/d/Download/layers_2012_2019_preprocessed.csv
```

只读聚合检查结果：

```text
raw long table: 776736 rows, 50 columns, 2802 unique ids
hourTally: 1..312
patient-level companion: 2802 rows, 49 columns, no hourTally
```

原表字段本身已经给出更多语义：

```text
hr, sbp, map, dbp, rr, temp, fio2,
bolusSum, RBCsum, surgSum, surgHours, ventDaySum, vent,
bicarb, StrongIon, bun, creatinine, wbc, lymphocytes, neutrophils, uop,
baseDef48, lactate48, RBC48, crys48, abx48, surg48
```

|Field|Raw-table evidence|Decision|
|---|---|---|
|`sbp/map/dbp`|原始列名就是 generic BP；没有 `NIBP` / `arterial` source 列。值域支持 mmHg。|单位已确认 mmHg；source 仍 unknown/mixed，不应强写 NIBP。|
|`temp`|raw long table: min 20.0, p50 37.3, p99 39.1, max 48.5；无 76 这类 Fahrenheit 值。|当前源列是 Celsius；20.0/48.5 是 QC/outlier。|
|`fio2`|20–100，整数为主。|percent-style FiO2；不是 0–1 fraction。|
|`uop`|不是累计：within-id 前后差有正有负；p50 100, p99 1000, max 10780。|mL per hour/recorded interval；应映射 ICU Output/Urine，不应放在 Hospital Lab。|
|`bolusSum`|within-id 非递减，无 negative step；positive delta 主要是 0.5 和 1.0。|累计 bolus exposure confirmed；更像 liter-scale / unit-coded volume，不是 mL-scale。 exact unit 仍需 source definition。|
|`RBCsum` / `RBC48`|`RBCsum` 非递减，positive delta 主要 +1；`RBC48` 是整数 first-48h patient-level summary。|RBC product units/count 高置信；不是 mL。|
|`surgSum/surgHours/surg48`|`surgSum` 非递减，主要 +1；`surgHours` 是累计小时；`surg48` first-48h count。|手术 count/hour 语义确认。|
|`vent` / `ventDaySum`|`vent` binary；`ventDaySum` 非递减，positive delta 全为 +1。|vent active status + cumulative vent days。|
|`baseDef48`|非负 0–29.9，within-id 数值常量，first-48h summary。|base deficit magnitude，单位/量级确认；不是 signed `Base Excess` 直接同义词。|
|`lactate48`|0.5–28，within-id 数值常量，first-48h summary。|mmol/L first-48h lactate summary；exact reducer 如 peak/worst/last 仍不能从列名和值域唯一确认。|
|`crys48`|0–26201，整数 mL scale，within-id 常量。|first-48h crystalloid volume, mL。|
|`StrongIon`|原始列名明确是 `StrongIon`，值域 13–56。|可确认是 strong-ion metric；公式未知，不能硬改名为 official `Anion Gap`。|

因此 raw table 后，剩余真正要确认的范围进一步缩小为：

|Remaining item|Why still unresolved|
|---|---|
|BP source|raw 只有 generic `sbp/map/dbp`；单位 mmHg 已确认，但无法判断 NIBP / arterial / mixed。|
|`baseDef48` reducer/sign|可确认是 first-48h nonnegative base deficit magnitude；若要映射 official `Base Excess`，仍需 sign/reducer rule。|
|`lactate48` reducer|单位和 first-48h window 已确认；peak/worst/last 等 reducer 仍不明。|
|`bolusSum` exact unit|可确认累计 exposure；delta 支持 liter-scale / unit-coded volume，但 exact label 仍需 source definition。|
|`RBCsum/RBC48` exact product label|可确认 units/count；如果要写成 PRBC units 还需 source definition。|
|`StrongIon` formula|raw 字段名确认，但公式/是否可对齐 official lab concept 不明。|

---

## 6. 官方字段 ↔ 当前字段映射

|Official concept/path|Current field(s)|Status|Note|
|---|---|---|---|
|Hospital Stay/General age/sex|G2_static__age, G2_static__male|partial|官方 General 还含 insurance/race/language；当前缺。|
|ED Stay/General, Chief Complaint, Diagnosis|G2_static__transfer, G2_static__headInjury; MechanismCat preserved only as index metadata|weak/partial|No chief complaint/ED diagnosis/medrecon; MechanismCat is an index and does not enter prompt.|
|ED Admission Vitals/sbp|Initial.ED.SBP_series_json; Initial.ED.SBPCat preserved only as index metadata|partial|Numeric ED SBP maps to SBP mmHg; SBPCat is an index and does not enter prompt; ED HR/RR/O2sat/temp/pain/dbp missing.|
|Hospital Lab Results/Bicarbonate|G4_labs__bicarb_series_json|direct|Distribution supports mEq/L; use official Bicarbonate field name.|
|Hospital Lab Results/Urea Nitrogen|G4_labs__bun_series_json|direct|Distribution supports mg/dL; align to official Urea Nitrogen.|
|Hospital Lab Results/Creatinine|G4_labs__creatinine_series_json|direct|Distribution supports mg/dL.|
|Hospital Lab Results/White Blood Cells|G4_labs__wbc_series_json|direct|Distribution supports K/uL.|
|Hospital Lab Results/Anion Gap|none; StrongIon not equivalent|missing|不要把 StrongIon 硬映射成 Anion Gap，除非 codebook/公式确认。|
|Hospital Lab Results/Lactate/Base Excess|lactate48_series_json, baseDef48_series_json|partial/high leakage risk|这是 first-48h summary/series，不等价于逐小时 lab；cutoff<48 不可进 desc。|
|ICU RoutineVitalSigns/Heart Rate|G1_vitals__hr_series_json|direct|用官方字段名 Heart Rate(bpm)。|
|ICU RoutineVitalSigns/NIBP systolic|sbp_series_json|direct-ish|当前未区分 NIBP/arterial BP。|
|ICU RoutineVitalSigns/NIBP diastolic|G1_vitals__dbp_series_json|direct-ish|当前未区分 NIBP/arterial BP。|
|ICU RoutineVitalSigns/NIBP mean|G1_vitals__map_series_json|direct-ish|当前 MAP 有异常高值，需要质控。|
|ICU RoutineVitalSigns/Temperature|G1_vitals__temp_series_json|derived/direct trace|Source distribution supports Celsius; model-facing EHR2Path text converts to `Temperature Fahrenheit(°F)` with `F=C*9/5+32`, while source_map preserves original °C.|
|ICU Respiratory/Respiratory Rate|G1_vitals__rr_series_json|direct|用官方字段名 Respiratory Rate(insp/min)。|
|ICU Respiratory/O2 saturation pulseoxymetry|none|missing|这是官方常见字段；当前无 SpO2。|
|ICU Respiratory/Inspired O2 Fraction|G1_vitals__fio2_series_json|direct|Distribution supports 21-100 percent-style FiO2; align to Inspired O2 Fraction.|
|ICU Respiratory/Ventilator Mode/Type|vent_series_json, ventDaySum|partial|当前只有通气状态/累计天数，缺 mode/type。|
|ICU Procedures/Invasive Ventilation|vent_series_json|partial|二值 active period 可转成官方 procedure period。|
|ICU Medication/Packed Red Blood Cells|G3_cumulative__RBCsum_series_json, RBC48|partial|Integer-like exposure; likely RBC product units/count. Use delta-derived events first; confirm unit before final report.|
|ICU/Hospital Procedures/OR Received or surgery|G3_cumulative__surgSum_series_json, surgHours, surg48|partial|可从 cumulative surgery delta 推断；缺 procedure name/type。|
|Prescriptions/Medication names|abx48 only for antibiotics summary|missing/partial|当前缺药名、start/end time、dose；abx48 只能作 first-48h antibiotic exposure。|
|Outcomes/Sepsis/infection time|Sepsis, infectionDay, infectionHour|metadata only|不能进入 desc；可用于分层或后验分析。|


---

## 7. 当前缺口 / 需要补齐的数据

|Area|Needed|Reason|
|---|---|---|
|ED details|chief complaint, ED diagnosis, arrival transport, med reconciliation, ED HR/RR/O2sat/temp/pain/dbp|提高与官方 ED section 的贴近度。|
|ICU oxygenation|SpO2 / O2 saturation pulseoxymetry|官方 respiratory section 高频字段；当前只有 FiO2。|
|BP source|区分 non-invasive vs arterial BP|官方分别建模 NIBP 和 arterial BP；当前只有通用 SBP/DBP/MAP。|
|CBC/coag/electrolytes|Hgb/Hct/platelets/RBC count/Na/K/Cl/Mg/Ca/Phos/PT/INR/PTT|官方 Hospital Lab Results 覆盖广，当前 lab 较窄。|
|Medication event table|drug name, dose/route/start/end; antibiotics/RBC/fluids|当前大多是累计暴露，不能完全复刻官方 Medication/Prescriptions。|
|Procedure names|procedure label, start/end, OR/procedure type|当前只有手术次数/小时，缺官方 procedure label 粒度。|
|Ventilator details|mode/type/set rate/total rate if available|当前只有 vent status/day summary。|
|Codebook|MechanismCat, rSICat, Initial.ED.SBPCat, acidosisCat, baseDef48Cat, lactate48Cat, crys48Cat, er_dispCat|不确认语义前不应把类别解释为临床文字。|
|Units|temp, FiO2, BUN, creatinine, WBC, uop, bolus/RBC units|决定是否可使用官方字段名和 normal range。|


---

## 8. 当前 official-field-aligned adapted sample

模型 prompt 使用官方 section / field names；当前字段名只在 source_map 中保留。

### 8.1 adapted `desc` YAML

```yaml
Hospital Stay:
  General: 'trauma ICU patient, 46-year old male, non-transfer, head injury indicator:
    1, Apache score: 25'
  Lab Results:
    'Bicarbonate (mEq/L, normal range: 22.0-32.0)': '21-0: 20'
    'Urea Nitrogen (mg/dL, normal range: 6.0-20.0)': '21-0: 8'
    'Creatinine (mg/dL, normal range: 0.5-1.2)': '21-0: 0.91'
    'White Blood Cells (K/uL, normal range: 4.0-11.0)': '21-0: 11.74'
ICU Stay:
  Stay 0:
    Chart Events:
      RoutineVitalSigns:
        Heart Rate(bpm): '23: 71, 22: 73, 21: 67, 20: 70, 19-18: 74, 17: 69, 16: 70,
          15: 74, 14: 76, 13: 80, 12: 70, 11-10: 71, 9: 69, 8-5: 68, 4-3: 70, 2: 69,
          1: 70, 0: 72'
        Non Invasive Blood Pressure systolic(mmHg): '23: 129, 22: 103, 21: 105, 20:
          107, 19: 114, 18: 128, 17: 105, 16: 108, 15: 128, 14: 114, 13: 135, 12:
          108, 11: 126, 10: 124, 9: 112, 8: 105, 7: 104, 6: 102, 5: 114, 4: 106, 3:
          119, 2: 111, 1: 115, 0: 116'
        Non Invasive Blood Pressure diastolic(mmHg): '23: 76, 22: 62, 21: 65, 20:
          61, 19: 74, 18: 77, 17: 61, 16: 71, 15: 81, 14: 62, 13: 87, 12: 59, 11:
          78, 10: 73, 9: 63, 8: 55, 7-6: 61, 5: 69, 4: 59, 3: 70, 2: 66, 1: 67, 0:
          66'
        Non Invasive Blood Pressure mean(mmHg): '23: 94, 22: 72, 21-20: 78, 19: 88,
          18: 105, 17: 76, 16: 83, 15: 94, 14: 76, 13: 105, 12: 73, 11: 91, 10: 90,
          9: 74, 8: 69, 7: 76, 6: 75, 5: 83, 4: 73, 3: 89, 2: 80, 1: 83, 0: 82'
        Temperature Fahrenheit(°F): '23-22: 98.6, 21: 97.9, 20: 98.6, 19: 98.4, 18: 99.3,
          17: 98.2, 16-15: 99.3, 14: 99.7, 13: 99.9, 12: 99.5, 11: 98.6, 10-9: 99.0,
          8: 98.8, 7: 99.9, 6: 98.6, 5: 99.0, 4: 99.7, 3: 100.0, 2: 100.2, 1: 100.0, 0:
          99.9'
      Respiratory:
        Respiratory Rate(insp/min): '23-14: 16, 13: 17, 12-10: 16, 9-0: 19'
        Inspired O2 Fraction: '23-0: 30'
    Procedures:
      Invasive Ventilation: 23-0
```

### 8.2 adapted next-hour `change_log` YAML

```yaml
Hospital Stay:
  Lab Results: {}
ICU Stay:
  Stay 0:
    Chart Events:
      RoutineVitalSigns:
        Heart Rate: 76
        Non Invasive Blood Pressure systolic: 119
        Non Invasive Blood Pressure diastolic: 69
        Non Invasive Blood Pressure mean: 89
        Temperature Fahrenheit: 100.0
      Respiratory:
        Respiratory Rate: 19
        Inspired O2 Fraction: 30
    Procedures:
    - Invasive Ventilation
```

### 8.3 source_map excerpt

|Prompt field|Source field|Status|Note|
|---|---|---|---|
|Hospital Stay/Lab Results/Bicarbonate (mEq/L, normal range: 22.0-32.0)|G4_labs__bicarb_series_json|direct|distribution supports mEq/L; official field name retained|
|Hospital Stay/Lab Results/Urea Nitrogen (mg/dL, normal range: 6.0-20.0)|G4_labs__bun_series_json|direct|distribution supports mg/dL; align to official Urea Nitrogen|
|Hospital Stay/Lab Results/Creatinine (mg/dL, normal range: 0.5-1.2)|G4_labs__creatinine_series_json|direct|distribution supports mg/dL|
|Hospital Stay/Lab Results/White Blood Cells (K/uL, normal range: 4.0-11.0)|G4_labs__wbc_series_json|direct|distribution supports K/uL|
|ICU Stay/Stay 0/Chart Events/RoutineVitalSigns/Heart Rate(bpm)|G1_vitals__hr_series_json|direct||
|ICU Stay/Stay 0/Chart Events/RoutineVitalSigns/Non Invasive Blood Pressure systolic(mmHg)|sbp_series_json|direct-ish|current source has generic sbp, not explicitly NIBP|
|ICU Stay/Stay 0/Chart Events/RoutineVitalSigns/Non Invasive Blood Pressure diastolic(mmHg)|G1_vitals__dbp_series_json|direct-ish|current source has generic dbp, not explicitly NIBP|
|ICU Stay/Stay 0/Chart Events/RoutineVitalSigns/Non Invasive Blood Pressure mean(mmHg)|G1_vitals__map_series_json|direct-ish|current source has generic map, not explicitly NIBP|
|ICU Stay/Stay 0/Chart Events/RoutineVitalSigns/Temperature Fahrenheit(°F)|G1_vitals__temp_series_json|derived_unit_conversion|source is Celsius; model-facing value uses `F=C*9/5+32`, rounded to 1 decimal|
|ICU Stay/Stay 0/Chart Events/Respiratory/Respiratory Rate(insp/min)|G1_vitals__rr_series_json|direct||
|ICU Stay/Stay 0/Chart Events/Respiratory/Inspired O2 Fraction|G1_vitals__fio2_series_json|direct|distribution supports 21-100 percent-style FiO2|
|ICU Stay/Stay 0/Procedures/Invasive Ventilation|vent_series_json|partial|binary ventilation status, no ventilator mode/type|

source_map 不进入 prompt；`*Cat` index 字段也不进入当前 prompt。


---

## 9. 改造规则 v5：experiment-ready schema

这版不再因为少量 semantic uncertainty 阻塞实验。规则是：能确定单位/转换的字段做 deterministic repo-compatible conversion；不能硬贴 official concept 的字段用 derived label + source_map review flag。

1. Section placement follow official code: hospital labs → `Hospital Stay/Lab Results`; ICU vitals/respiratory/hemodynamics → `ICU Stay/Stay 0/Chart Events/...`; ICU outputs → `ICU Stay/Stay 0/Output`.
2. 当前 raw column names 不作为 prompt keys；prompt keys 用 official EHR2Path names 或明确 derived label。
3. Values use official time style: `hours_ago: value`，`0` 是 current cutoff。
4. `*Cat` index/code fields are excluded from prompt unless codebook maps them into validated clinical text。
5. `*48` fields obey leakage rule: cutoff <48 exclude/mask; cutoff >=48 may include as first-48h summaries。
6. G3 fields are cumulative totals at cutoff: `bolusSum`, `RBCsum`, `ventDaySum`, `surgSum`, `surgHours`。如果作为 target，必须定义 next cumulative value vs next-hour delta。
7. Model-facing scalar 优先对齐 repo/MIMIC-IV text distribution；source_map 保留源单位、转换公式、review flag。

### 9.1 Final scalar mapping decisions for first experiment

|Field / group|Decision|Evidence|Review flag, not blocker|
|---|---|---|---|
|`sbp/dbp/map`|Use `Non Invasive Blood Pressure systolic/diastolic/mean(mmHg)` model-facing labels; source_map notes raw source is generic BP.|MIMIC `d_items`: NIBP itemids 220179/220180/220181, mmHg. Official subset NIBP systolic/diastolic/mean ≈994/994/995 leaves vs arterial ≈464/464/466.|NIBP vs arterial/mixed source is not observed. Accept compat-first for experiment.|
|`temp`|Convert source °C to `Temperature Fahrenheit(°F)` using `F=C*9/5+32`, round 1 decimal.|Raw p50≈37.3°C; official subset `Temperature Fahrenheit(°F)=1135` leaves vs `Temperature Celsius(°C)=80`; ED temperature is F.|No blocker; source_map preserves original °C.|
|`baseDef48`|If cutoff≥48, include as `First-48h Base Excess (mEq/L), derived from base deficit`, using `base_excess=-baseDef48`.|MIMIC `Base Excess` itemid 50802; official Base Excess values include negative/positive. Real-world base deficit is positive magnitude of negative base excess.|Exact reducer unknown. If reviewer rejects sign conversion, fallback label is `First-48h Base Deficit magnitude`.|
|`lactate48`|If cutoff≥48, include as `First-48h Lactate (mmol/L)`.|MIMIC `Lactate` itemid 50813; official processed Lactate `(mmol/L, normal range: 0.5-2.0)`; local range 0.5–28 matches mmol/L.|Exact reducer peak/worst/last unknown.|
|`StrongIon`|Include optional derived field `Strong Ion Difference / Strong Ion metric (mEq/L)`; do **not** map to `Anion Gap`.|Raw name explicitly `StrongIon`; local p50≈33, official Anion Gap p50≈14 and normal range around 8–20.|Formula unknown; keep derived/non-core field.|
|`bolusSum`|Treat as cumulative liters; model-facing `Cumulative IV Fluid Bolus (mL) = bolusSum*1000`.|Appendix G3 cumulative; raw monotonic; top positive deltas 0.5/1.0, matching 500 mL / 1 L bolus convention; MIMIC fluid intake labels are mL.|Exact source unit not codebook-confirmed, but enough for first experiment.|
|`RBCsum/RBC48`|Use `Packed Red Blood Cells (units/count)`; do not convert to mL.|Appendix G3 cumulative for `RBCsum`; raw deltas mostly +1; `RBC48` first-48h summary; MIMIC has `Packed Red Blood Cells`, but local source behaves as product units/count.|Exact product label PRBC vs RBC product count unknown.|
|`uop`|Use `ICU Stay/Stay 0/Output/Urine Output (mL)`; not cumulative.|Raw uop has positive and negative steps, p50 100 mL, p99 1000 mL; MIMIC output labels `OR Urine/PACU Urine` are mL.|Exact charting interval wording unknown.|

### 9.2 Remaining review list — not blocking

```text
BP source type if a codebook exists
baseDef48 reducer/sign source definition
lactate48 reducer
StrongIon formula
bolusSum exact stored unit
RBC product label
```

Decision: **GO** for 100–500 sample generation with `schema_v5_experiment_ready`。

---

## 10. 下一步建议

1. 生成 100–500 条 `schema_v5_experiment_ready` EHR2Path-style samples。
2. 对每条样本输出：`desc_yaml`, `change_log_yaml`, `target_next_state`, `source_map`, `scalar_policy`, `leakage_flags`。
3. Validation checks: Cat/index exclusion, `*48` cutoff rule, temperature C→F conversion, bolus L→mL conversion, baseDef48→negative Base Excess only when cutoff≥48, G3 cumulative target semantics。
4. 第一轮不训练；先做 prompt/sample inspection + field coverage + parseability + target missingness。
5. 如果 100–500 条通过，再进入 text-only HF/PEFT inference smoke；不要回到 Unsloth entrypoint。
