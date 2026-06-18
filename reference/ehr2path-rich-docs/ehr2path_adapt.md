# EHR2Path-rich schema-v6 final 简明报告

更新时间：2026-05-06  
定位：这是给讨论/人工检查用的**展示文档**。详细机器证据、完整字段解释和完整样本不再全部塞进正文，而是放在 CSV/JSONL/JSON artifact 中。

## 1. 先前工作一页摘要：方法、问题、启发

| 阶段 | 做法 | 遇到的问题 | 对这次工作的启发 |
|---|---|---|---|
| schema-v5 rich input + zero-shot inference | 把 grouped-wide trauma 表转成 EHR2Path-style text，使用本地 text-only LoRA 推理。 | 模型能生成 MIMIC 风格文本，但经常回到 hospital/location/prescription 模板，不能稳定输出我们的 next-hour scalar target。 | 不能只靠“把字段写进prompt”；必须约束输出结构，并检查 label 是否在 MIMIC/EHR2Path 训练分布内。 |
| output constraint sweep | 比较 instruction-only prompting、few-shot/hard schema 和 assistant-side prefix。 | 自然语言约束基本无效；assistant_prefix 才能把模型带到目标输出轨迹。 | no-finetune baseline 应使用 target-family / target-specific short prefix + deterministic parser。 |
| micro-tune diagnostic | 做 40 train / 10 eval、20-step PEFT 诊断。 | 有小幅 MAE 改善，但 FiO2、vent、cumulative scalar 仍对不上 label space。 | 当前瓶颈先是 schema/label alignment，不是马上扩大 finetuning。 |
| schema-v6 respiratory/labs smoke | 用 official-like `Inspired O2 Fraction`、`Invasive Ventilation`、`Hospital Stay / Lab Results` 做小样本验证。 | broad respiratory prefix 会泄漏到 SpO2；短 target prefix 后明显改善。 | 字段要分层：native 可以直接约束推理，partial/source-native 要单独标记和评价。 |

一句话：前期工作的价值不是“得到了最终模型效果”，而是定位了问题——**要先把字段映射、单位、label space、leakage gate 和输出约束做好，再谈 finetuning。**

## 2. 本次工作目标

这次目标是把 `patient_trajectories_grouped_wide.csv` 的 **55个字段** 固化成一个 schema-v6 final 规范，使它尽量嵌合 EHR2Path repo 使用的 MIMIC-IV processed data：

1. 能用 MIMIC/EHR2Path native label 的字段，使用 native label；
2. trauma 特有但有信息价值的字段，放入 source-native controlled blocks；
3. 分类码、未来 outcome label、first-48h summary 严格防泄漏；
4. 输出不做自由生成，而是 target-specific short prefix + parser；
5. 当前产物先供人工检查，通过后再做 50-row integrated run。

关键产物：

```text
/home/vanila/code/ehrtopath-rich-to-structured/mk_v6_final_add.py
/home/vanila/code/ehrtopath-rich-to-structured/artifacts/schema_v6_final_field_map_add.csv
/home/vanila/code/ehrtopath-rich-to-structured/artifacts/schema_v6_final_samples_add.jsonl
/home/vanila/code/ehrtopath-rich-to-structured/artifacts/schema_v6_final_spec_add.json
/home/vanila/code/ehrtopath-rich-to-structured/artifacts/artifact_index_add.csv
```

## 3. 证据来源

| 证据 | 用途 |
|---|---|
| 本地 official EHR2Path processed data：`/mnt/d/Data/mimic-iv-2.2/all_data_24_hours_los_noisy` | 确认哪些 label 真正在 repo processed text 中出现，例如 `Heart Rate`、`Inspired O2 Fraction`、`Invasive Ventilation`、`Bicarbonate`。 |
| MIMIC-IV `chartevents` 概念 | 支持 vitals、respiratory、chart events 放在 ICU chart events。 |
| MIMIC-IV `labevents` 概念 | 支持 chemistry/CBC 类指标放在 `Hospital Stay / Lab Results`。 |
| MIMIC-IV `outputevents` 概念 | 支持尿量/输出类字段，但源数据缺 Foley/Void device，所以 `uop` 需要 review flag。 |
| MIMIC-IV `procedureevents` 概念 | 支持 `Invasive Ventilation` 作为 procedure/evidence，而不是裸 0/1 label。 |

参考：

```text
https://mimic.mit.edu/docs/IV/modules/icu/chartevents.html
https://mimic.mit.edu/docs/IV/modules/hosp/labevents.html
https://mimic.mit.edu/docs/IV/modules/icu/outputevents.html
https://mimic.mit.edu/docs/IV/modules/icu/procedureevents.html
```

## 4. 最终映射规则

1. **native 优先**：字段含义和单位能对齐 MIMIC/EHR2Path processed label，就用 native label。
2. **单位先转换**：例如源体温是 °C，但模型侧用 `Temperature Fahrenheit(°F)`。
3. **FiO2 不直接暴露为 FiO2**：model-facing label 用 `Inspired O2 Fraction`；不能和 SpO2 混用。
4. **vent binary 转为 evidence**：输入/输出写 `Invasive Ventilation: present/absent`，二值 `Mechanical Ventilation Status` 只放 `derived_targets`。
5. **分类码不进 prompt**：`*Cat` 没有 validated codebook，保留 metadata。
6. **outcome label 不进 prompt**：`Sepsis`、`infectionDay`、`infectionHour` 只用于分层/分析。
7. **first-48h summary gate**：`cutoff_hourTally < 48` 时不出现 `First 48h Summary`。
8. **cumulative scalar 不伪装成 MIMIC events**：bolus/RBC/vent days/surgery 等放 `Source-Native Trauma Exposures`。
9. **推理方式**：不同 target family 分开用 short assistant_prefix 解码，再 deterministic parse。

## 5. 字段分层结论

| 层级 | 字段 | 处理方式 |
|---|---|---|
| A. MIMIC/EHR2Path native 主评价字段 | `G1_vitals__hr_series_json`（心率）, `G1_vitals__dbp_series_json`（舒张压）, `G1_vitals__map_series_json`（平均动脉压）, `G1_vitals__rr_series_json`（呼吸频率）, `G1_vitals__temp_series_json`（体温）, `G1_vitals__fio2_series_json`（吸入氧浓度 FiO2）, `G4_labs__bicarb_series_json`（碳酸氢根）, `G4_labs__bun_series_json`（尿素氮 BUN）, `G4_labs__creatinine_series_json`（肌酐）, `G4_labs__wbc_series_json`（白细胞计数）, `sbp_series_json`（收缩压）, `vent_series_json`（机械通气状态）, `lymphocytes_series_json`（淋巴细胞百分比）, `neutrophils_series_json`（中性粒细胞百分比） | 可进入 no-finetune constrained decoding 主评价。 |
| B. partial / review | `G3_cumulative__RBCsum_series_json`（累计红细胞输注量）, `G4_labs__uop_series_json`（尿量） | 放入最接近 official section，但带 review flag，不能过度声称 native support。 |
| C. source-native rich trauma information | `G2_static__RBC48`（前48小时红细胞输注量）, `G2_static__Apache`（APACHE严重程度评分）, `G2_static__abx48`（前48小时是否使用抗生素）, `G2_static__surg48`（前48小时手术次数/是否手术）, `G3_cumulative__bolusSum_series_json`（累计静脉补液 bolus）, `G3_cumulative__RBCsum_series_json`（累计红细胞输注量）, `G3_cumulative__ventDaySum_series_json`（累计机械通气天数）, `G3_cumulative__surgSum_series_json`（累计手术次数）, `G3_cumulative__surgHours_series_json`（累计手术时长）, `G4_labs__StrongIon_series_json`（强离子差）, `rSI_series_json`（修正休克指数 rSI）, `crys48_series_json`（前48小时晶体液量） | 信息价值高，但不伪装为 MIMIC prescriptions/procedures；单独评价是否带来帮助或 spillover。 |
| D. metadata-only / leakage-sensitive | `id`（患者轨迹编号）, `start_hourTally`（起始小时索引）, `end_hourTally`（结束小时索引）, `observed_hours`（观测小时数）, `start_day`（起始天）, `start_hour`（起始日内小时）, `end_day`（结束天）, `end_hour`（结束日内小时）, `time_axis_mismatch_count`（时间轴不一致计数）, `time_index_json`（小时索引数组）, `G2_static__MechanismCat`（损伤机制分类码）, `G2_static__Initial.ED.SBPCat`（初始急诊收缩压分类码）, `G2_static__rSICat`（修正休克指数分类码）, `G2_static__baseDef48Cat`（前48小时碱缺失分类码）, `G2_static__lactate48Cat`（前48小时乳酸分类码）, `G2_static__crys48Cat`（前48小时晶体液分类码）, `G2_static__er_dispCat`（急诊去向分类码）, `G4_labs__acidosisCat_series_json`（酸中毒分类码）, `Sepsis`（脓毒症标签）, `infectionDay`（感染发生天）, `infectionHour`（感染发生小时） | 用于 traceability、QC、分层或 codebook queue，不进入 prompt。 |

## 6. 55字段轻量检查表

说明：这里保留所有字段的中文名、通俗解释、角色和映射摘要，方便人工快速检查。更长的解释、source dtype、missingness 和完整规则见：

```text
/home/vanila/code/ehrtopath-rich-to-structured/artifacts/schema_v6_final_field_map_add.csv
```

| source column | 中文名/科普解释 | v6角色 | 对齐状态 | 映射去向/规则摘要 |
|---|---|---|---|---|
| `id` | **患者轨迹编号**：每个创伤患者/住院轨迹的一条记录编号。 | 元数据/质控 | metadata key | `metadata.id`；用于追溯样本和 patient-level split |
| `start_hourTally` | **起始小时索引**：该患者轨迹的第一个小时编号。 | 元数据/质控 | metadata time | `metadata.start_hourTally`；用于验证时间轴 |
| `end_hourTally` | **结束小时索引**：该患者轨迹的最后一个小时编号。 | 元数据/质控 | metadata time | `metadata.end_hourTally`；用于验证可观测窗口和 censoring |
| `observed_hours` | **观测小时数**：该患者轨迹包含的小时数量。 | 元数据/质控 | metadata time | `metadata.observed_hours`；用于检查数组长度 |
| `start_day` | **起始天**：轨迹起点所在天数。 | 元数据/质控 | metadata time | `metadata.start_day`；只用于定位，不进入 prompt。 |
| `start_hour` | **起始日内小时**：轨迹起点在当天的小时。 | 元数据/质控 | metadata time | `metadata.start_hour`；只用于定位，不进入 prompt。 |
| `end_day` | **结束天**：轨迹终点所在天数。 | 元数据/质控 | metadata time | `metadata.end_day`；只用于定位，不进入 prompt。 |
| `end_hour` | **结束日内小时**：轨迹终点在当天的小时。 | 元数据/质控 | metadata time | `metadata.end_hour`；只用于定位，不进入 prompt。 |
| `time_axis_mismatch_count` | **时间轴不一致计数**：原始长表聚合后发现的时间轴异常计数。 | 元数据/质控 | metadata QC | `metadata.time_axis_mismatch_count`；质量控制字段 |
| `time_index_json` | **小时索引数组**：保存 hourTally/day/hour/source row 等 aligned arrays。 | 元数据/质控 | metadata time | `metadata.time_index_json`；所有动态字段切片和 next-hour target 的基准 |
| `G2_static__age` | **年龄**：患者年龄，是基础人口学信息。 | 输入背景 | official-like general | `Hospital Stay / General / Age`；official processed 里 General 常包含年龄 |
| `G2_static__male` | **性别**：二值性别编码，1 通常表示男性。 | 输入背景 | official-like general | `Hospital Stay / General / Sex`；转成 male/female 文本，贴近 MIMIC General |
| `G2_static__transfer` | **是否转院/转入**：是否由外院或其他机构转入。 | 输入背景 | official-like general | `Hospital Stay / General / transfer patient`；作为入院背景写成 transfer/non-transfer |
| `G2_static__MechanismCat` | **损伤机制分类码**：创伤机制的分类索引，例如钝性/穿透性等可能类别。 | 元数据/质控 | metadata code | `metadata.codes.MechanismCat`；当前没有验证 codebook，不把整数码写入 prompt |
| `G2_static__headInjury` | **是否头部损伤**：是否记录头部损伤。 | 输入背景 | official-like general | `Hospital Stay / General / head injury documented`；二值临床背景，可转成 head injury documented/not documented。 |
| `G2_static__Initial.ED.SBPCat` | **初始急诊收缩压分类码**：ED 初始 SBP 的分层分类。 | 元数据/质控 | metadata code | `metadata.codes.Initial.ED.SBPCat`；已有连续 Initial.ED.SBP 可用 |
| `G2_static__rSICat` | **修正休克指数分类码**：rSI 的分层分类。 | 元数据/质控 | metadata code | `metadata.codes.rSICat`；连续 rSI 可放 source-native block |
| `G2_static__baseDef48Cat` | **前48小时碱缺失分类码**：base deficit first-48h 的分类。 | 元数据/质控 | metadata code + leakage-sensitive | `metadata.codes.baseDef48Cat`；既是分类码又是 first-48h summary |
| `G2_static__lactate48Cat` | **前48小时乳酸分类码**：lactate first-48h 的分类。 | 元数据/质控 | metadata code + leakage-sensitive | `metadata.codes.lactate48Cat`；不进 prompt |
| `G2_static__RBC48` | **前48小时红细胞输注量**：前48小时 packed RBC 使用量/单位数。 | cutoff≥48后输入背景 | source-native first48 | `Hospital Stay / First 48h Summary / Packed Red Blood Cells(units/count)`；MIMIC 有 Packed Red Blood Cells event，但本字段是 first-48h scalar |
| `G2_static__crys48Cat` | **前48小时晶体液分类码**：crystalloid first-48h 的分类。 | 元数据/质控 | metadata code + leakage-sensitive | `metadata.codes.crys48Cat`；分类码不进 prompt |
| `G2_static__Apache` | **APACHE严重程度评分**：ICU 常用综合严重程度评分，数值越高通常病情越重。 | 输入背景 | source-native general | `Hospital Stay / General / APACHE score`；MIMIC processed 不一定有该字段，但临床背景重要 |
| `G2_static__abx48` | **前48小时是否使用抗生素**：前48小时内是否记录抗生素使用。 | cutoff≥48后输入背景 | source-native first48 | `Hospital Stay / First 48h Summary / Antibiotics`；MIMIC prescriptions 是具体药物/区间 |
| `G2_static__surg48` | **前48小时手术次数/是否手术**：前48小时手术暴露 summary。 | cutoff≥48后输入背景 | source-native first48 | `Hospital Stay / First 48h Summary / Surgery Count`；MIMIC Procedures 是具体 ICD/procedure text |
| `G2_static__er_dispCat` | **急诊去向分类码**：ED disposition 的分类索引。 | 元数据/质控 | metadata code | `metadata.codes.er_dispCat`；未验证 codebook，不进入 prompt |
| `G1_vitals__hr_series_json` | **心率**：每分钟心跳次数，反映循环状态、疼痛/休克/感染等。 | 输入+next-hour目标 | MIMIC/EHR2Path native | `ICU Stay / Stay 0 / Chart Events / RoutineVitalSigns / Heart Rate(bpm)`；official processed 强支持 Heart Rate |
| `G1_vitals__dbp_series_json` | **舒张压**：心脏舒张期血压，单位 mmHg。 | 输入+next-hour目标 | MIMIC/EHR2Path native | `ICU Stay / Stay 0 / Chart Events / RoutineVitalSigns / Non Invasive Blood Pressure diastolic(mmHg)`；official processed 有 NIBP diastolic |
| `G1_vitals__map_series_json` | **平均动脉压**：一个心动周期平均血压，反映灌注压力。 | 输入+next-hour目标 | MIMIC/EHR2Path native | `ICU Stay / Stay 0 / Chart Events / RoutineVitalSigns / Non Invasive Blood Pressure mean(mmHg)`；official processed 有 NIBP mean |
| `G1_vitals__rr_series_json` | **呼吸频率**：每分钟呼吸次数，反映呼吸负荷、疼痛、休克或代谢酸中毒。 | 输入+next-hour目标 | MIMIC/EHR2Path native | `ICU Stay / Stay 0 / Chart Events / Respiratory / Respiratory Rate(insp/min)`；official processed 强支持 |
| `G1_vitals__temp_series_json` | **体温**：源数据为摄氏度；模型侧用华氏度。 | 输入+next-hour目标 | native + 单位转换 | `ICU Stay / Stay 0 / Chart Events / RoutineVitalSigns / Temperature Fahrenheit(°F)`；MIMIC/EHR2Path processed 多为 Fahrenheit |
| `G1_vitals__fio2_series_json` | **吸入氧浓度 FiO2**：患者吸入气体中的氧浓度百分比。 | 输入+next-hour目标 | native + label转换 | `ICU Stay / Stay 0 / Chart Events / Respiratory / Inspired O2 Fraction`；official label 是 Inspired O2 Fraction |
| `G3_cumulative__bolusSum_series_json` | **累计静脉补液 bolus**：截至当前小时累计快速补液量；源单位按既有规则 bolusSum*1000 ml。 | source-native输入+目标 | source-native scalar | `Hospital Stay / Source-Native Trauma Exposures / Cumulative IV Fluid Bolus(ml)`；MIMIC prescriptions 是具体液体/时间区间 |
| `G3_cumulative__RBCsum_series_json` | **累计红细胞输注量**：截至当前小时累计 packed RBC 单位/次数。 | source-native输入+目标 | partial native scalar | `Hospital Stay / Source-Native Trauma Exposures / Cumulative Packed Red Blood Cells(units/count)`；MIMIC 有 Packed Red Blood Cells medication event，但这里是累计 scalar |
| `G3_cumulative__ventDaySum_series_json` | **累计机械通气天数**：截至当前小时累计机械通气暴露天数。 | source-native输入+目标 | source-native scalar | `Hospital Stay / Source-Native Trauma Exposures / Cumulative Ventilator Days`；不要用它代替 Invasive Ventilation event |
| `G3_cumulative__surgSum_series_json` | **累计手术次数**：截至当前小时累计手术次数。 | source-native输入+目标 | source-native scalar | `Hospital Stay / Source-Native Trauma Exposures / Cumulative Surgery Count`；MIMIC procedure list 需要具体 procedure |
| `G3_cumulative__surgHours_series_json` | **累计手术时长**：截至当前小时累计手术小时数。 | source-native输入+目标 | source-native scalar | `Hospital Stay / Source-Native Trauma Exposures / Cumulative Surgery Hours`；无 MIMIC native scalar label |
| `G4_labs__bicarb_series_json` | **碳酸氢根**：血液化学指标，反映酸碱平衡和代谢状态。 | 输入+next-hour目标 | MIMIC/EHR2Path native | `Hospital Stay / Lab Results / Bicarbonate (mEq/L, normal range: 22.0-32.0)`；official Lab Results 强支持 Bicarbonate |
| `G4_labs__acidosisCat_series_json` | **酸中毒分类码**：酸中毒程度/类型的分类索引。 | 元数据/质控 | metadata code | `metadata.codes.acidosisCat`；已有 bicarb/base/lactate 等连续指标 |
| `G4_labs__StrongIon_series_json` | **强离子差**：酸碱分析中的强离子差相关指标。 | source-native输入+目标 | source-native lab scalar | `Hospital Stay / Source-Native Trauma Chemistry / Strong Ion Difference(mEq/L)`；MIMIC 有 Anion Gap 但 Strong Ion Difference 不是同义 |
| `G4_labs__bun_series_json` | **尿素氮 BUN**：反映肾功能、蛋白分解和容量状态的血液化学指标。 | 输入+next-hour目标 | MIMIC/EHR2Path native | `Hospital Stay / Lab Results / Urea Nitrogen (mg/dL, normal range: 6.0-20.0)`；official Lab Results 强支持 Urea Nitrogen。 |
| `G4_labs__creatinine_series_json` | **肌酐**：肾功能指标，升高提示肾功能受损或灌注不足。 | 输入+next-hour目标 | MIMIC/EHR2Path native | `Hospital Stay / Lab Results / Creatinine (mg/dL, normal range: 0.5-1.2)`；official Lab Results 强支持 Creatinine |
| `G4_labs__wbc_series_json` | **白细胞计数**：炎症/感染/应激相关血液指标。 | 输入+next-hour目标 | MIMIC/EHR2Path native | `Hospital Stay / Lab Results / White Blood Cells (K/uL, normal range: 4.0-10.0)`；official Lab Results 强支持 White Blood Cells。 |
| `G4_labs__uop_series_json` | **尿量**：单位时间尿液排出量，反映肾灌注和容量状态。 | 输入+目标/需复核 | partial native output | `ICU Stay / Stay 0 / Output / Urine Output(ml)`；MIMIC outputevents 支持尿/引流输出，processed 常见 Foley(ml)/Void(ml) |
| `Initial.ED.SBP_series_json` | **急诊初始收缩压**：到达 ED 时的初始收缩压，反映早期休克风险。 | 仅输入背景 | official-like ED context | `Emergency Department Stay / Admission Vitals / sbp`；official ED Admission Vitals 有 sbp |
| `rSI_series_json` | **修正休克指数 rSI**：休克相关综合指标，通常由心率和血压组合反映循环不稳定。 | source-native输入+目标 | source-native score | `Hospital Stay / Source-Native Trauma Scores / Revised Shock Index`；非 MIMIC native label，但创伤信息价值高 |
| `baseDef48_series_json` | **前48小时碱缺失**：酸碱状态指标；源是 base deficit 正值，模型侧转为 Base Excess 负值。 | cutoff≥48后输入背景 | native-like first48 derived | `Hospital Stay / First 48h Summary / Base Excess (mEq/L), derived from base deficit`；MIMIC Lab Results 有 Base Excess |
| `lactate48_series_json` | **前48小时乳酸**：组织低灌注/休克相关指标。 | cutoff≥48后输入背景 | native-like first48 | `Hospital Stay / First 48h Summary / Lactate (mmol/L)`；MIMIC Lab Results 有 Lactate |
| `crys48_series_json` | **前48小时晶体液量**：早期复苏使用的晶体液总量。 | cutoff≥48后输入背景 | source-native first48 | `Hospital Stay / First 48h Summary / Crystalloid Fluid(ml)`；不是 MIMIC prescription interval |
| `sbp_series_json` | **收缩压**：心脏收缩期血压，单位 mmHg。 | 输入+next-hour目标 | MIMIC/EHR2Path native | `ICU Stay / Stay 0 / Chart Events / RoutineVitalSigns / Non Invasive Blood Pressure systolic(mmHg)`；official processed 强支持 NIBP systolic |
| `vent_series_json` | **机械通气状态**：每小时是否接受有创机械通气。 | 输入+next-hour目标 | native evidence | `ICU Stay / Stay 0 / Procedures / Invasive Ventilation`；model-facing 用 Invasive Ventilation evidence |
| `lymphocytes_series_json` | **淋巴细胞百分比**：白细胞分类之一，可反映免疫/感染应激状态。 | 输入+next-hour目标 | MIMIC/EHR2Path native | `Hospital Stay / Lab Results / Lymphocytes (%, normal range: 19.0-53.0)`；official Lab Results 支持 Lymphocytes |
| `neutrophils_series_json` | **中性粒细胞百分比**：白细胞分类之一，常与感染/炎症/应激相关。 | 输入+next-hour目标 | MIMIC/EHR2Path native | `Hospital Stay / Lab Results / Neutrophils (%, normal range: 34.0-71.0)`；official Lab Results 支持 Neutrophils |
| `Sepsis` | **脓毒症标签**：后验/研究标签，表示是否发生脓毒症。 | 仅标签/分层元数据 | outcome metadata | `metadata.labels.Sepsis`；当前任务是 next-hour state，不是二分类 |
| `infectionDay` | **感染发生天**：感染/脓毒症发生日期标签。 | 仅标签/分层元数据 | outcome-time metadata | `metadata.labels.infectionDay`；未来事件时间有泄漏风险 |
| `infectionHour` | **感染发生小时**：感染/脓毒症发生小时标签。 | 仅标签/分层元数据 | outcome-time metadata | `metadata.labels.infectionHour`；未来事件时间有泄漏风险 |

## 7. input sample 规范

每条 schema-v6 final 样本包含：

```text
schema_version
sample_id / patient_id          # 仅本地 artifact 保留；展示文档使用 redacted alias
cutoff_hourTally
next_hourTally                  # 必须 next_hourTally - cutoff_hourTally = 1
desc                            # model input
change_log                      # next-hour reference
assistant_prefix_policy
source_map                      # source column / unit / derivation rule
derived_targets                 # 例如 Mechanical Ventilation Status 0/1
metadata                        # Sepsis/infection labels + quarantined category codes
review_flags
```

生成样本验证：

```text
field_mapping_rows = 55
samples = 6
cutoff_counts = {'24': 3, '240': 2, '72': 1}
next_hour_deltas = [1]
forbidden_tokens = []
pre48_first48_leakage_samples = []
avg_desc_chars = 2963.5
```

## 8. compact sample preview（redacted）

展示样本：`S4_post48_redacted`，`cutoff_hourTally=240`，`next_hourTally=241`。完整样本在：

```text
/home/vanila/code/ehrtopath-rich-to-structured/artifacts/schema_v6_final_samples_add.jsonl
```

### desc 结构

| block | 内容 |
|---|---|
| `Hospital Stay / General` | 人口学与创伤背景：年龄、性别、transfer、head injury、APACHE。 |
| `Hospital Stay / Lab Results` | Bicarbonate、Urea Nitrogen、Creatinine、White Blood Cells、Lymphocytes、Neutrophils。 |
| `Hospital Stay / Source-Native Trauma Exposures` | cumulative bolus/RBC/ventilator days/surgery count/surgery hours。 |
| `Hospital Stay / First 48h Summary` | post-48 才出现；pre-48 已验证隐藏。 |
| `ICU Stay / Chart Events / RoutineVitalSigns` | HR、NIBP systolic/diastolic/mean、Temperature Fahrenheit。 |
| `ICU Stay / Chart Events / Respiratory` | Respiratory Rate、Inspired O2 Fraction。 |
| `ICU Stay / Procedures` | Invasive Ventilation evidence。 |
| `ICU Stay / Output` | Urine Output，带 source-device review flag。 |

### desc 片段

```json
{
  "General": "17-year old male, transfer patient, head injury documented, APACHE score 29",
  "Heart Rate(bpm)": "23: 134, 22: 122, 21: 127, 20: 135, 19: 133, 18: 122, 17: 116, 16: 115, 15: 113, 14: 120, 13: 119, 12: 115, 11: 118, 10: 119, 9: 1 ...",
  "Respiratory Rate(insp/min)": "23: 16, 22: 17, 21: 16, 20: 20, 19/18: 16, 17-0: 18",
  "Inspired O2 Fraction": "23: 60, 19: 90, 18: 80, 17: 70, 10/9: 50",
  "Invasive Ventilation": "23-0",
  "Lab Results keys": "Bicarbonate (mEq/L, normal range: 22.0-32.0), Urea Nitrogen (mg/dL, normal range: 6.0-20.0), Creatinine (mg/dL, normal range: 0.5- ...",
  "Source-Native keys": "Cumulative IV Fluid Bolus(ml), Cumulative Packed Red Blood Cells(units/count), Cumulative Ventilator Days, Cumulative Surgery Coun ...",
  "First 48h Summary keys": "Base Excess (mEq/L), derived from base deficit, Lactate (mmol/L), Packed Red Blood Cells(units/count), Crystalloid Fluid(ml), Anti ..."
}
```

### next-hour targets preview

- `ICU Stay / Stay 0 / Chart Events / RoutineVitalSigns / Heart Rate` = `120`
- `ICU Stay / Stay 0 / Chart Events / RoutineVitalSigns / Non Invasive Blood Pressure systolic` = `140`
- `ICU Stay / Stay 0 / Chart Events / RoutineVitalSigns / Non Invasive Blood Pressure diastolic` = `77`
- `ICU Stay / Stay 0 / Chart Events / RoutineVitalSigns / Non Invasive Blood Pressure mean` = `93`
- `ICU Stay / Stay 0 / Chart Events / RoutineVitalSigns / Temperature Fahrenheit` = `98.6`
- `ICU Stay / Stay 0 / Chart Events / Respiratory / Respiratory Rate` = `20`
- `ICU Stay / Stay 0 / Chart Events / Respiratory / Inspired O2 Fraction` = `40`
- `ICU Stay / Stay 0 / Procedures / Invasive Ventilation` = `present`
- `ICU Stay / Stay 0 / Output / Urine Output` = `300`
- `Hospital Stay / Lab Results / Bicarbonate` = `32`
- `Hospital Stay / Lab Results / Urea Nitrogen` = `14`
- `Hospital Stay / Lab Results / Creatinine` = `0.74`
- `Hospital Stay / Lab Results / White Blood Cells` = `31.37`
- `Hospital Stay / Lab Results / Lymphocytes` = `1.8`
- `Hospital Stay / Lab Results / Neutrophils` = `27.6`
- `Hospital Stay / Source-Native Trauma Scores / Revised Shock Index` = `1.1`
- `Hospital Stay / Source-Native Trauma Chemistry / Strong Ion Difference` = `39`
- `Hospital Stay / Source-Native Trauma Exposures / Cumulative IV Fluid Bolus` = `11000`

### review flags

```text
- uop mapped to ICU Output / Urine Output(ml); source device unknown, not forced to Foley or Void
- target-specific short assistant_prefix remains required for model inference
- category codes retained in metadata because codebook is not validated
- source-native trauma blocks are included for richness but should be evaluated separately from official-native targets
```

## 9. schema-v6 final text-only inference 收尾结果

这一步已经实际推理完成；不是只准备输入。使用 text-only local LoRA，对 schema-v6 final 50-row integrated sample 做 target-specific short-prefix decoding。

产物：

```text
/home/vanila/code/ehrtopath-rich-to-structured/v6_eval_add.py
/home/vanila/code/ehrtopath-rich-to-structured/artifacts/schema_v6_textonly_eval50_tasks_add.jsonl
/mnt/d/Data/ehr2path_logs/schema_v6_textonly_eval50_hf_add.jsonl
/home/vanila/code/ehrtopath-rich-to-structured/artifacts/schema_v6_textonly_eval50_metrics_add.json
/home/vanila/code/ehrtopath-rich-to-structured/artifacts/schema_v6_textonly_eval50_summary_add.csv
/home/vanila/code/ehrtopath-rich-to-structured/artifacts/schema_v6_textonly_eval50_detail_add.csv
```

推理规模：

```text
base_records = 50
target_tasks = 676
predictions = 676
model = text-only EHR2Path LoRA, local HF/PEFT fallback
batch_size = 2
max_new_tokens = 16
elapsed = about 471.5 sec
```

### 9.1 评估方式修正

这里不再用“跨字段 raw MAE”作为主结论，因为不同字段单位和量纲完全不同。新的主指标是：

```text
scalar fields:
  coverage
  normalized_mae = MAE / max(eval p90-p10(reference), field_scale_floor)
  carry_forward_skill = 1 - model_MAE / carry_forward_MAE

binary fields:
  coverage
  accuracy
  balanced_accuracy
  carry_forward_accuracy
```

解释：

- `raw MAE` 只保留为字段内单位解释，例如 HR 的 bpm、Temperature 的 °F；不跨字段平均。
- `normalized MAE` 用字段内参考分布或最低临床尺度归一化，适合横向比较误差量级。
- `carry_forward_skill > 0` 才表示模型超过 last-observed carry-forward baseline。

### 9.2 family-level 结果

| family | targets | coverage | normalized MAE | carry-forward skill | binary accuracy/balanced acc | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `labs` | 6 | 1.00 | 0.236 | -1.633 |  | 可解析，但总体明显不如 carry-forward，尤其 lymphocytes；更多像 label-space 可达性验证。 |
| `routine_vitals` | 5 | 1.00 | 0.173 | 0.024 |  | 唯一平均略优于 carry-forward 的 scalar family，但提升很小；仍必须保留 baseline 对照。 |
| `respiratory` | 2 | 1.00 | 0.210 | -0.114 |  | 可解析性好，但 RR/Inspired O2 平均不赢 carry-forward；schema 对齐成功≠预测能力充分。 |
| `procedure` | 1 | 1.00 |  |  | 0.88/0.77 | Invasive Ventilation 可解析，balanced accuracy 尚可，但 carry-forward accuracy 更高。 |
| `partial_output` | 1 | 1.00 | 0.261 | -0.036 |  | Urine Output 可解析但不赢 carry-forward，仍是 partial/review 字段。 |

### 9.3 target-level 结果

| target | n | coverage | raw MAE only within-field | normalized MAE | carry-forward skill / acc |
|---|---:|---:|---:|---:|---:|
| `bicarbonate` | 50 | 1.00 | 2.22 | 0.200 | -0.067 |
| `creatinine` | 50 | 1.00 | 0.199 | 0.147 | -0.001 |
| `heart_rate` | 50 | 1.00 | 6.24 | 0.137 | 0.049 |
| `inspired_o2` | 42 | 1.00 | 11.3 | 0.290 | -0.118 |
| `invasive_vent` | 50 | 1.00 | — | — | acc 0.88, bAcc 0.77, CF acc 0.92 |
| `lymphocytes` | 22 | 1.00 | 2.59 | 0.518 | -9.029 |
| `neutrophils` | 22 | 1.00 | 6.23 | 0.258 | -0.398 |
| `nibp_diastolic` | 50 | 1.00 | 5.74 | 0.202 | 0.043 |
| `nibp_mean` | 50 | 1.00 | 6.3 | 0.175 | 0.100 |
| `nibp_systolic` | 50 | 1.00 | 7.56 | 0.173 | 0.078 |
| `resp_rate` | 50 | 1.00 | 2.2 | 0.129 | -0.111 |
| `temperature_f` | 49 | 1.00 | 0.62 | 0.178 | -0.152 |
| `urea_nitrogen` | 50 | 1.00 | 3.46 | 0.131 | -0.018 |
| `urine_output` | 41 | 1.00 | 57.5 | 0.261 | -0.036 |
| `white_blood_cells` | 50 | 1.00 | 3.8 | 0.163 | -0.282 |

### 9.4 结论

1. **可以开始并已经完成 v6 final text-only inference 收尾**：50-row / 676 target tasks 全部生成并解析成功，说明 schema-v6 final + short prefix 的输出约束是可操作的。
2. **评估结论不能说模型已经强预测**：coverage=1.0 更多说明格式可控；真正预测能力要看 carry-forward skill。
3. **routine vitals 是当前 text-only 最可靠区域**：平均 normalized MAE≈0.173，平均 carry-forward skill≈0.024，只是轻微超过 baseline。
4. **respiratory/labs/urine output 多数不赢 carry-forward**：这些字段的 label-space 对齐和可解析性已经改善，但 next-hour 数值预测能力还不充分。
5. **Invasive Ventilation 可以解析，但仍要看类不平衡**：accuracy=0.88、balanced accuracy=0.775，参考分布 40 present / 10 absent；carry-forward accuracy=0.92，所以不能只看 accuracy。
6. **进入 summary/mixed 之前，必须复用这套 normalized + baseline-relative 评估**，否则会被 raw MAE 或 class imbalance 误导。

### 9.5 对下一阶段的含义

text-only v6 的收尾结论是：

```text
schema-v6 final 解决了“能不能嵌入和约束输出”的问题，
但还没有解决“多数 target 是否明显超过简单时序基线”的问题。
```

因此下一步实验 summary-only / mixed model 时，不能只看生成文本像不像；要使用同一批 schema-v6 tasks 和同一套指标，对比：

```text
text-only vs summary-only vs mixed
normalized_mae
carry_forward_skill
binary balanced_accuracy
format coverage
MIMIC-template spillover
```

## 10. summary-only / mixed 入口构造（待正式推理）

本阶段目标不是立即跑完整推理，而是先按 repo 原始入口把 `summary-only` 和 `mixed` 接到我们已经完成的 `schema-v6 final eval50` 任务上，确保后续运行仍然是 **official-mode semantics + 本地离线路径 + 统一评估协议**。

### 10.1 repo 真实入口

- `summary-only` / `mixed` 都走：`model_code/train_with_summ_embs.py`
- config：
  - `configs/val_pathway_24h_summonly.yaml`
  - `configs/val_pathway_24h_text_summ.yaml`
- 本地权重：
  - summary-only checkpoint：`/mnt/d/Models/ehrtopath/summary_only/summary_input_8_last_int_only_summ_clean2/checkpoint-124000`
  - mixed checkpoint：`/mnt/d/Models/ehrtopath/mixed/summary_input_8_last_int_mixed_24h_clean2_DropAugment_curriculum/checkpoint-125000`
  - summary encoder checkpoint：`/mnt/d/Models/ehrtopath/summary model/summary_full_1M_8token_sumonly_clean/checkpoint-108000`
  - base model：`/mnt/d/Models/ehrtopath/models/unsloth/Qwen2-0.5B-Instruct-bnb-4bit`

### 10.2 新增 bridge runner

已新增 repo-side 脚本：

```text
/home/vanila/code/ehrtopath/scripts/val_summ_modes_add.py
```

它做的事情是：

1. 读取我们已有的 `schema_v6_textonly_eval50_tasks_add.jsonl`；
2. 用 repo 的 `get_sample_sections(...)` + `extract_embs_sample(...)` 从 `desc` 里抽 section summary embeddings；
3. `summary-only`：构造 repo 语义一致的 LOS-only text scaffold；
4. `mixed`：保留 recent text，并与 summary placeholders 一起进入 prompt；
5. 延续当前 target-specific `assistant_prefix` 评估协议，而不是退回 full free-form generation。

### 10.3 当前桥接验证结果

无模型 `--check` 已完成：

- summary-only bridge check：`/mnt/d/Data/ehr2path_logs/summary_only_bridge_check_full_add.json`
- mixed bridge check：`/mnt/d/Data/ehr2path_logs/mixed_bridge_check_full_add.json`

关键结果：

- `n_tasks = 676`
- `n_unique_base_records = 50`
- target families = `labs, partial_output, procedure, respiratory, routine_vitals`
- 在抽查的 12 个 base records 上，`section_prompt_count` 范围是 `6` 到 `7`，均值 `6.917`
- `summary-only` 的 LOS-only scaffold 在抽查样本上都能构造；top-level keys 保持 `Emergency Department Stay / Hospital Stay / ICU Stay`

这说明：**当前 schema-v6 final `desc` 已经足够接入 repo 的 summary embedding 管线，不需要回退到官方全量 processed dirs 才能做我们的 custom eval50 inference。**

### 10.4 进入正式推理前的注意点

1. 这一步只完成了 bridge，不等于已经完成 summary-only / mixed inference。
2. `summary-only` 依赖的是 "summary embeddings + 极少量 LOS-only text scaffold"；因此它和 text-only 的输入语义不同，结果应单独解释。
3. 正式运行后仍用同一套 closeout 指标：
   - scalar：`coverage + normalized_mae + carry_forward_skill`
   - binary：`accuracy + balanced_accuracy + carry_forward_accuracy`
4. 不再把 cross-field raw MAE 当作主结论。

### 10.5 下一步待执行命令

先建议从小批量 smoke 开始，而不是直接 676 task 全跑：

```bash
cd /home/vanila/code/ehrtopath
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline WANDB_DISABLED=true

python scripts/val_summ_modes_add.py \
  --mode summary_only \
  --input-jsonl /home/vanila/code/ehrtopath-rich-to-structured/artifacts/schema_v6_textonly_eval50_tasks_add.jsonl \
  --max-samples 40 \
  --batch-size 1 \
  --max-new-tokens 16 \
  --output-json /mnt/d/Data/ehr2path_logs/schema_v6_summaryonly_eval_smoke_add.json \
  --output-jsonl /mnt/d/Data/ehr2path_logs/schema_v6_summaryonly_eval_smoke_add.jsonl

python scripts/val_summ_modes_add.py \
  --mode mixed \
  --input-jsonl /home/vanila/code/ehrtopath-rich-to-structured/artifacts/schema_v6_textonly_eval50_tasks_add.jsonl \
  --max-samples 40 \
  --batch-size 1 \
  --max-new-tokens 16 \
  --output-json /mnt/d/Data/ehr2path_logs/schema_v6_mixed_eval_smoke_add.json \
  --output-jsonl /mnt/d/Data/ehr2path_logs/schema_v6_mixed_eval_smoke_add.jsonl
```

跑完后可直接复用：

```bash
python /home/vanila/code/ehrtopath-rich-to-structured/v6_eval_add.py eval --pred-jsonl <pred_jsonl> --tasks-jsonl /home/vanila/code/ehrtopath-rich-to-structured/artifacts/schema_v6_textonly_eval50_tasks_add.jsonl --out-prefix <out_prefix>
```

### 10.6 当前结论

- text-only v6 final 已经完成收尾；
- summary-only / mixed 的 repo 入口桥接已经搭好；
- 下一步是先做小批量 smoke inference，再决定是否跑完整 eval50。

## 11. summary-only / mixed 正式推理结果（eval50）

这一步已经不是 bridge check，而是实际跑完两种模式的完整 eval50 inference：

- `summary_only`: 676 / 676 tasks, elapsed = 559.81 sec
- `mixed`: 676 / 676 tasks, elapsed = 687.29 sec

为了避免本机 `Unsloth` summary-embedding 路径在 RTX50 / torch-cu128 栈上 `segfault (-11)`，本次使用 repo-side HF-safe runner：

- runner: `/home/vanila/code/ehrtopath/scripts/val_summ_modes_add.py`
- summary encoder: HF/PEFT forward 提取 section embeddings
- pathway model: HF/PEFT generate + 手工注入 `<SUMMARY>` placeholder embeddings
- 评估仍复用 `v6_eval_add.py` 的 normalized MAE / carry-forward skill / balanced accuracy 协议

### family-level 对比

| family | text-only | summary-only | mixed | 结论 |
|---|---|---|---|---|
| `routine_vitals` | cov=1.00, nMAE=0.173, skill=0.024 | cov=1.00, nMAE=0.372, skill=-1.062 | cov=1.00, nMAE=0.187, skill=-0.067 | text-only 最好；mixed 接近但仍退步 |
| `respiratory` | cov=1.00, nMAE=0.210, skill=-0.114 | cov=1.00, nMAE=0.751, skill=-2.238 | cov=1.00, nMAE=0.198, skill=-0.063 | mixed 最好，略优于 text-only |
| `labs` | cov=1.00, nMAE=0.236, skill=-1.633 | cov=1.00, nMAE=0.390, skill=-2.661 | cov=1.00, nMAE=0.281, skill=-1.879 | mixed 比 summary-only 明显好，但仍不如 text-only |
| `partial_output` | cov=1.00, nMAE=0.261, skill=-0.036 | cov=1.00, nMAE=0.306, skill=-0.213 | cov=1.00, nMAE=0.251, skill=0.007 | mixed 小幅转正，优于 text-only |
| `procedure` | cov=1.00, bAcc=0.775 | cov=1.00, bAcc=0.500 | cov=1.00, bAcc=0.500 | summary/mixed 退到接近常数预测 |

### target-level 关键信号

- `summary_only` 几乎没有明确超过 carry-forward 的 target；唯一略正的是 `heart_rate`（skill ≈ 0.008），整体明显弱于 text-only。
- `mixed` 出现了少量正 skill target：
  - `nibp_systolic` skill ≈ 0.117
  - `nibp_mean` skill ≈ 0.074
  - `nibp_diastolic` skill ≈ 0.040
  - `urine_output` skill ≈ 0.007
- 但 `mixed` 仍在多个 family 上没有全面超过 text-only；尤其 `routine_vitals` 平均上仍弱于 text-only，`procedure` 也明显退化。

### 决策

1. `summary_only` 不建议继续作为主线：它在当前 schema-v6 final eval50 上明显不如 text-only。
2. `mixed` 有局部增益，说明 summary embeddings 不是完全无用，但当前融合方式还不稳定。
3. 目前最合理的主结论是：
   - **text-only 仍是最稳的 baseline**；
   - **mixed 可以作为下一轮结构/提示/融合消融的候选**；
   - **summary-only 可降级为次要对照，不必主推。**
4. 如果继续 mixed，优先检查：
   - summary section 选择是否与 next-hour target 对齐；
   - `<SUMMARY>` 注入位置与 section 数量是否过多；
   - mixed 文本 scaffold 是否稀释了 text-only 已经有效的 target-specific prefix；
   - procedure / partial_output 的模板是否仍需更强约束。

