---
title: 12h Future-Window Summary Report Design
created: 2026-06-06
workspace: summary design/
status: draft
tags: [summary, report, temporal-abstraction, schema]
sources:
  - /home/vanila/code/EHR-Predict/agent-artifact/compiled/uw_mimic_field_registry_v0_20260523.yaml
  - Vincent et al. 1996, SOFA score
  - RCP 2017, NEWS2
  - Shahar 1997, Knowledge-Based Temporal Abstraction
  - Seymour et al. 2016, Sepsis-3
---

# 12h Future-Window Summary Report

## 设计原则

1. **所有字段必须从 $S_{t+1:t+12}$ 确定性计算。** 不依赖自然语言生成。
2. **每个字段有临床依据。** 来源：SOFA、NEWS2、KBTA temporal abstraction。
3. **每个字段是二分类或三分类。** V1 简化，不用连续回归。
4. **report 先不训练模型直接输出。** 先用确定性规则构造 label，再用 probe 验证。

## 可用字段

### 每小时连续变量
```
hr, sbp, dbp, map, rr, temp, fio2
```
来源：registry → hourly_vitals

### 间歇实验室
```
bicarb, bun, creatinine, wbc
```
来源：registry → labs_outputs  
注：均非每小时测量，需处理 missingness

### 累积干预
```
bolusSum (IV fluid cumulative), RBCsum (transfusion cumulative)
vent (hourly ventilation flag)
```
来源：registry → cumulative_interventions

### 输出
```
uop (urine output)
```
来源：registry → labs_outputs

### 标签（不进入模型输入）
```
infectionHour, Sepsis
```
来源：registry → outcome_labels

---

## Report Schema

### 1. 循环系统 (Hemodynamic)

| 字段 | 类型 | 临床依据 | 计算方法 |
|---|---|---|---|
| `hemo_worst_map_bin` | normal/low/severe | SOFA 循环：MAP<70 扣分 | `min(MAP_{t+1:t+12})` → ≥70=normal, 65-69=low, <65=severe |
| `hemo_map_burden_6h` | true/false | MAP<65 累积≥6h 预示恶化 | MAP<65 小时数 ≥ 6 |
| `hemo_pressor_escalation` | true/false | 血管活性药增加是休克进展标志 | bolusSum(t+12) - bolusSum(t) 超过阈值 |

### 2. 呼吸系统 (Respiratory)

| 字段 | 类型 | 临床依据 | 计算方法 |
|---|---|---|---|
| `resp_fio2_increase` | true/false | SOFA 呼吸：FiO2 需求增加 | max(FiO2_{t+1:t+12}) - FiO2(t) > 0.15 |
| `resp_vent_change` | none/started/continued/stopped | SOFA 呼吸最高分：机械通气 | vent flag 在窗口内的变化 |

### 3. 肾脏代谢 (Renal/Metabolic)

| 字段 | 类型 | 临床依据 | 计算方法 |
|---|---|---|---|
| `renal_creatinine_rise` | true/false | SOFA 肾脏：肌酐升高 | last creatinine in window - creatinine(t) > 0.3 mg/dL |
| `renal_uop_low_6h` | true/false | SOFA 肾脏：少尿 | UOP < 0.5 ml/kg/h 的累计小时数 ≥ 6 |

### 4. 组织灌注 (Tissue Perfusion)

| 字段 | 类型 | 临床依据 | 计算方法 |
|---|---|---|---|
| `perf_lactate_high` | true/false | Sepsis-3：乳酸>2 是组织灌注不足标志 | max(lactate_{t+1:t+12}) > 2.0 mmol/L |
| `perf_lactate_rising` | true/false | 乳酸持续升高预示恶化 | last lactate in window > lactate(t) + 0.5 |

### 5. 关键事件 (Critical Events)

| 字段 | 类型 | 临床依据 | 计算方法 |
|---|---|---|---|
| `event_sepsis_onset` | true/false/censored | 感染性事件 | infectionHour 在 (t, t+12] 范围内 |
| `event_death` | true/false | 死亡终点 | death 发生在窗口内 |
| `event_discharge` | true/false | 出院终点 | discharge 发生在窗口内 |

### 6. 综合判断 (Overall)

| 字段 | 类型 | 临床依据 | 计算方法 |
|---|---|---|---|
| `overall_deterioration` | true/false | 至少 2 个系统出现恶化信号 | hemo_worst_map_bin=severe OR resp_vent_change=started OR (perf_lactate_high AND renal_creatinine_rise) OR ≥2 个系统 flag |
| `overall_improving` | true/false | 无系统恶化 + 至少一项改善 | 无任何 deterioration flag AND (MAP 回升 OR FiO2 下降 OR lactate 下降) |

### 7. 窗口完整性 (Censoring)

| 字段 | 类型 | 计算方法 |
|---|---|---|
| `observed_hours` | 0-12 | 窗口内有任何观察数据的小时数 |
| `window_truncated` | true/false | 因死亡/出院导致窗口 < 12h |

---

## 字段构造伪代码

```python
def build_report_label(S_future, patient_kg):
    """S_future: S_{t+1:t+12} canonical hourly states"""
    
    # 循环
    map_vals = [s.map for s in S_future if s.map is not None]
    map_min = min(map_vals) if map_vals else None
    map_low_hours = sum(1 for m in map_vals if m < 65)
    
    label = {
        'hemo_worst_map_bin': 'severe' if map_min and map_min < 65 
                              else 'low' if map_min and map_min < 70 
                              else 'normal',
        'hemo_map_burden_6h': map_low_hours >= 6,
        'hemo_pressor_escalation': _pressor_change(S_future),
        
        # 呼吸
        'resp_fio2_increase': _fio2_change(S_future) > 0.15,
        'resp_vent_change': _vent_change(S_future),
        
        # 肾脏
        'renal_creatinine_rise': _creatinine_delta(S_future) > 0.3,
        'renal_uop_low_6h': _uop_low_hours(S_future, patient_kg) >= 6,
        
        # 灌注
        'perf_lactate_high': _lactate_max(S_future) > 2.0,
        'perf_lactate_rising': _lactate_last(S_future) - _lactate_first(S_future) > 0.5,
        
        # 事件
        'event_sepsis_onset': _sepsis_in_window(S_future),
        'event_death': _death_in_window(S_future),
        'event_discharge': _discharge_in_window(S_future),
        
        # 窗口
        'observed_hours': sum(1 for s in S_future if s.any_observed),
        'window_truncated': len(S_future) < 12,
    }
    
    # 综合
    label['overall_deterioration'] = _compute_deterioration(label)
    label['overall_improving'] = _compute_improving(label)
    
    return label
```

---

## 临床依据速查

| 设计元素 | 参考文献 | 对应内容 |
|---|---|---|
| MAP < 65 为休克阈值 | Vincent 1996 (SOFA)；Sepsis-3 (Seymour 2016) | hemo_worst_map_bin, hemo_map_burden_6h |
| FiO2 需求增加反映呼吸恶化 | SOFA 呼吸组件 | resp_fio2_increase |
| 肌酐升高 0.3 mg/dL 定义 AKI | KDIGO 2012；SOFA 肾脏组件 | renal_creatinine_rise |
| 乳酸 > 2 mmol/L 组织灌注不足 | Sepsis-3；Surviving Sepsis Campaign | perf_lactate_high |
| 少尿 < 0.5 ml/kg/h | SOFA 肾脏组件 | renal_uop_low_6h |
| 多系统恶化需综合判断 | NEWS2 aggregate score；SOFA total score | overall_deterioration |
| 趋势抽象 (rising/stable/falling) | Shahar 1997 (KBTA) | perf_lactate_rising |
| 持续性抽象 (burden hours) | Shahar 1997 (KBTA) | hemo_map_burden_6h, renal_uop_low_6h |

---

## 与 next-state 的关系

```
同一个 future trajectory S_{t+1:t+12}:

next-hour H_{t+1}   = first slice (精确值)
12h report R         = window aggregation (临床摘要)

区别：
  H_{t+1} 是 self-supervised，label 直接是下一小时状态
  R 是构造的，label 通过确定性聚合从同一 trajectory 得到
```

---

## 下一步

1. 确认 schema 字段无遗漏、无冗余
2. 从 canonical hourly state 构造第一批 report label
3. 检查 label 分布（正负样本平衡、censoring 比例）
4. 编写正式 label builder 脚本
