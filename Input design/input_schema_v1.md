---
title: V1 Model Input Schema
created: 2026-06-08
status: draft
tags: [input, schema, v1, canonical-state]
sources:
  - data dicision/field adapter/uw_style_field_table_v1.md
  - data dicision/field adapter/field_mapping_v1.csv
  - data dicision/field adapter/field_registry_v1.yaml
---

# V1 Model Input Schema

## 设计原则

1. **所有字段来源可追溯**：每个字段对应 registry 中的明确 source/itemid。
2. **时间边界硬切**：input ≤ t，target = t+1（V1）或 t+1:t+12（V2 label）。
3. **LOCF 必须带元数据**：value + observed_flag + time_since_last 三元组。
4. **标签不进 input**：Sepsis / infectionHour / infectionDay / future summary labels 都不出现在 encoder 输入中。
5. **48h conditional 字段**：仅在 completed history block ≥ 48h 时出现。

---

## 输入结构概览

```
每个训练样本是一个 (hadm_id, stay_id, t) 三元组：

input = {
  STATIC:          G2 静态字段（入院已知）
  DAILY_BLOCKS:    [completed 24h blocks]，每个 block 是 24h 的聚合摘要
  HOURLY_BLOCK:    [最近 N 小时的 canonical hourly state]
}
```

| Block | 含义 | 长度 | 分辨率 |
|-------|------|------|--------|
| `STATIC` | 入院已知 / pre-ICU 上下文 | 固定 | 1 条 per admission |
| `DAILY_BLOCKS` | 已完成的历史 24h 摘要 | 可变（≥0） | 每 24h 一条 |
| `HOURLY_BLOCK` | 最近高分辨率状态 | N=24（可配） | 每小时一条，含 LOCF 元数据 |

---

## 1. STATIC Block

每个 field 一行。来源：G2 static / pre-ICU context。

| field | type | range | notes |
|-------|------|-------|-------|
| `age` | float | 18–89+ | `age_at_admit`；>89 截断 |
| `sex_male` | int | {0, 1} | patient.gender == 'M' |
| `mechanism_cat` | int | {1, 2, 3} | 1=blunt, 2=penetrating, 3=other |
| `transfer` | int | {0, 1} | ED arrival_transport 含 transfer/ambulance 等 |
| `initial_ed_sbp` | float | mmHg | ED triage.sbp；缺失用 NA |
| `rsi_ed` | float | ratio | ED HR / ED SBP；SBP=0 则 NA |
| `er_disp_cat` | int | categorical | ED disposition 类别编码 |

---

## 2. DAILY_BLOCKS (Completed 24h Summary Blocks)

每完成一个完整 24h ICU 时段，生成一条 daily summary。不完整的天不产出 block。

### 2.1 基本结构

```text
{block_id, day_index, start_hour, end_hour, fields...}
```

`start_hour` 和 `end_hour` 是 ICU 相对小时数（0-based）。例如 day 0 = hours [0, 23]。

### 2.2 字段

对每个连续变量，summary 包含：

| aggregation | 含义 | 适用字段 |
|-------------|------|---------|
| `_last` | block 内最后一次值 | vitals, labs |
| `_min` | block 内最小值 | MAP, SBP, UOP |
| `_max` | block 内最大值 | HR, RR, FiO2, lactate |
| `_mean` | block 内均值 | vitals |
| `_n_obs` | block 内测量次数 | vitals, labs |
| `_burden_ge_th` | 超过阈值的累计小时数 | MAP<65, SI>1 |
| `_trend` | block 内斜率/方向 | 可选，V1 暂不做 |

### 2.3 字段清单

#### G1 vitals (daily aggregates)

| field | type | aggregations |
|-------|------|-------------|
| `hr` | float | last, min, max, mean, n_obs |
| `sbp` | float | last, min, max, mean, n_obs |
| `dbp` | float | last, min, max, mean, n_obs |
| `map` | float | last, min, max, mean, n_obs, burden_lt65 |
| `rr` | float | last, min, max, mean, n_obs |
| `temp` | float | last, min, max, mean, n_obs |
| `fio2` | float | last, min, max, mean, n_obs |

#### G3 interventions (daily totals)

| field | type | aggregations |
|-------|------|-------------|
| `iv_fluid_total_ml` | float | sum |
| `rbc_total_ml` | float | sum |
| `vent_hours` | float | sum (vent_status=1 的小时数) |
| `surgery_hours` | float | sum |
| `surgery_any` | int | any in_surgery=1 in block? |

#### G4 labs (daily last + count)

| field | type | aggregations |
|-------|------|-------------|
| `lactate_last` | float | last observed value in block |
| `lactate_max` | float | max in block |
| `lactate_n` | int | measurement count |
| `creatinine_last` | float | last observed |
| `creatinine_n` | int | measurement count |
| `bicarb_last` | float | last observed |
| `bicarb_n` | int | measurement count |
| `base_excess_last` | float | last observed |
| `wbc_last` | float | last observed |
| `uop_total_ml` | float | sum |
| `uop_low_hours` | int | UOP < threshold 的小时数 |

#### Conditional: 48h fixed fields

仅在 `total_history_hours >= 48` 时出现：

| field | type |
|-------|------|
| `baseDef48` | float |
| `lactate48` | float |
| `RBC48` | float |
| `crys48` | float |
| `abx48` | int |
| `surg48` | int |

---

## 3. HOURLY_BLOCK (Recent High-Resolution State)

最近 N 小时（默认 N=24），每小时一个 row。

### 3.1 每条 hourly row 的结构

```text
{hour_index, fields...}
```

`hour_index`：ICU 相对小时数（0-based）。

### 3.2 通用元数据字段（每个变量都带）

| suffix | 含义 |
|--------|------|
| `_value` | 当前小时的值（LOCF 后） |
| `_obs` | 当前小时是否有新测量（0/1） |
| `_tsl` | time since last observation（小时），无历史则 max |

### 3.3 字段清单

#### G1 hourly vitals (7 fields × 3 = 21 columns)

```
hr_value, hr_obs, hr_tsl
sbp_value, sbp_obs, sbp_tsl
dbp_value, dbp_obs, dbp_tsl
map_value, map_obs, map_tsl
rr_value, rr_obs, rr_tsl
temp_value, temp_obs, temp_tsl
fio2_value, fio2_obs, fio2_tsl
```

#### G3 hourly interventions (5 fields)

| field | type | 说明 |
|-------|------|------|
| `iv_fluid_ml_1h` | float | 当前小时 IV 晶体量 |
| `rbc_ml_1h` | float | 当前小时 RBC 量 |
| `vent_status` | int {0, 1} | 当前小时是否在机械通气 |
| `in_surgery` | int {0, 1} | 当前小时是否在手术 |
| `surgery_hours_1h` | float | 当前小时手术占用时间 |

注：G3 不需要 LOCF。未干预小时值为 0。

#### G4 labs (3 fields × 11 labs = 33 columns)

```
lactate_value, lactate_obs, lactate_tsl
creatinine_value, creatinine_obs, creatinine_tsl
bicarb_value, bicarb_obs, bicarb_tsl
base_excess_value, base_excess_obs, base_excess_tsl
bun_value, bun_obs, bun_tsl
wbc_value, wbc_obs, wbc_tsl
lymphocytes_value, lymphocytes_obs, lymphocytes_tsl
neutrophils_value, neutrophils_obs, neutrophils_tsl
na_value, na_obs, na_tsl
k_value, k_obs, k_tsl
cl_value, cl_obs, cl_tsl
```

#### G4 derived

| field | type | 说明 |
|-------|------|------|
| `strong_ion_value` | float | (Na + K) − (Cl + bicarb)；仅当所有成分可用 |
| `strong_ion_obs` | int | 所有成分均有测量 |
| `strong_ion_tsl` | float | max(各成分 tsl) |
| `acidosis_cat` | int | 0=normal, 1=mild, 2=moderate, 3=severe；依 bicarb/base_excess |

#### G4 output

| field | type |
|-------|------|
| `uop_value` | float（ml/h） |
| `uop_obs` | int |

---

## 4. 输入字段总数

| Block | 列数 | 说明 |
|-------|------|------|
| STATIC | 8 | G2 静态字段 |
| DAILY_BLOCKS | ~80 | 随 history 长度增加 block 数 |
| HOURLY_BLOCK | 24 × ~65 | 24h × (21 vitals + 5 G3 + 33 labs + 6 derived) |

---

## 5. 训练样本构建规则

### 5.1 时间切片

对每个 ICU stay：

```python
for t in range(0, icu_hours):
    if not sufficient_history(t):
        continue  # 跳过 ICU 早期（< 1h）
    
    sample = {
        'hadm_id': hadm_id,
        'stay_id': stay_id,
        't': t,
        'static': build_static(hadm_id),
        'daily_blocks': build_daily_blocks(hadm_id, stay_id, t),
        'hourly_block': build_hourly_block(hadm_id, stay_id, t, window=24),
    }
    
    # V1 target
    sample['target_next_hour'] = build_next_hour(hadm_id, stay_id, t+1)
    
    # V2 labels (for label building only, NOT model input)
    sample['target_report_12h'] = build_report_label(hadm_id, stay_id, t+1, t+12)
```

### 5.2 sufficient_history 条件

- `t >= 1`：至少有一小时历史
- 不要求完整 24h history；early ICU 只有 short hourly block

### 5.3 LOCF 规则

- Vitals：向前填充，`tsl` 从最后一次实际测量算起
- Labs：向前填充，`tsl` 可超过 24h（陈旧 lab 仍可 LOCF，但 tsl 很大）
- 如果完全没有历史值，`value = NA`，`obs = 0`，`tsl = +inf`（或固定大值如 999）

---

## 6. 排除项

以下数据类型明确不进入 V1 input：

| 排除项 | 原因 |
|--------|------|
| Sepsis / infectionDay / infectionHour | 标签 |
| future summary labels (12h) | 标签 |
| discharge time / LOS total | future leakage |
| hospital_expire_flag | future leakage / label |
| Apache score | MIMIC-IV 无原生数据 |
| headInjury | hold（待用户确认 ICD proxy） |
| Initial_ED_SBPCat / rSICat | hold（待 UW 阈值确认） |
