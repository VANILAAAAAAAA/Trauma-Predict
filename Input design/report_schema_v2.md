---
title: V2 Future 12h Summary Report Label Schema
created: 2026-06-08
status: draft
tags: [report, label, v2, summary, temporal-abstraction]
supersedes: summary design/report_schema_v1.md
sources:
  - Vincent et al. 1996, SOFA score
  - RCP 2017, NEWS2
  - Shahar 1997, Knowledge-Based Temporal Abstraction
  - Seymour et al. 2016, Sepsis-3
  - KDIGO 2012, AKI definition
depends:
  - input_schema_v1.md
  - data dicision/field adapter/uw_style_field_table_v1.md
---

# V2 Future 12h Summary Report Label Schema

## 设计原则

1. **所有 label 从 $S_{t+1:t+12}$ 确定性构造**，不依赖模型。
2. **每个 label 有明确临床阈值来源**（SOFA、KDIGO、Sepsis-3、NEWS2、KBTA）。
3. **multi-label binary 为主**，少量 multi-class。
4. **所有 label 带 mask**：censoring/death/discharge/missingness 分层处理。
5. **V2 head 输出概率**，不输出文本。文本由 deterministic renderer 后处理。

---

## Label 输出维度

每条样本 `(hadm_id, stay_id, t)` 产出 23 个 labels + 4 个元信息：

```python
{
    # === Hemodynamic (4) ===
    "hemo_map_lt65_ge2h":        int | masked,   # MAP<65 累计 ≥2h
    "hemo_map_lt65_ge6h":        int | masked,   # MAP<65 累计 ≥6h
    "hemo_map_worst_bin":        int,             # 0=normal, 1=low, 2=severe (multi-class)
    "hemo_si_gt1_any":           int | masked,   # 任意小时 SI>1

    # === Respiratory (3) ===
    "resp_fio2_increase_ge015":  int | masked,   # max(FiO2_window) - FiO2(t) > 0.15
    "resp_vent_started":         int | masked,   # 窗口内新上机
    "resp_vent_continued":       int | masked,   # 窗口内持续通气

    # === Renal (2) ===
    "renal_creatinine_rise_ge03": int | masked,  # Cr 升高 ≥0.3 mg/dL
    "renal_uop_low_ge6h":        int | masked,   # UOP < threshold 累计 ≥6h

    # === Perfusion / Metabolic (4) ===
    "perf_lactate_gt2":          int | masked,   # max(lactate) > 2
    "perf_lactate_gt4":          int | masked,   # max(lactate) > 4
    "perf_lactate_rising_gt05":  int | masked,   # last lactate − first lactate > 0.5
    "perf_be_worse":             int | masked,   # base_excess 恶化（more negative）

    # === Infection / Treatment (3) ===
    "infx_new_abx":              int | masked,   # 窗口内新抗生素
    "infx_new_culture":          int | masked,   # 窗口内新血培养
    "infx_suspected":            int | masked,   # abx + culture 同时新出现

    # === Critical Events / Endpoint (4) ===
    "event_sepsis_onset":        int,             # sepsis onset in window (0/1/censored)
    "event_death_12h":           int,             # 死亡在窗口内
    "event_discharge_12h":       int,             # 出院在窗口内

    # === Composite (2) ===
    "comp_any_deterioration":    int | masked,   # 任一系统恶化
    "comp_multi_system":         int | masked,   # ≥2 系统恶化

    # === Window metadata (4) ===
    "meta_observed_hours":       int,             # 0–12，窗口内有观测的小时数
    "meta_window_truncated":     int,             # 窗口 <12h
    "meta_censored_by_death":    int,             # 死亡截尾
    "meta_censored_by_discharge": int,            # 出院截尾
}
```

---

## 逐字段构造规则

### 1. Hemodynamic

#### `hemo_map_lt65_ge2h` (binary)

```python
map_vals = [s.map for s in S_future if s.map_obs == 1]
hours_lt65 = sum(1 for m in map_vals if m < 65)
label = 1 if hours_lt65 >= 2 else 0
mask = 1 if len(map_vals) >= 6 else 0  # 至少 6h 有效 MAP
```

**临床依据**: Sepsis-3 shock: MAP<65；persistence ≥2h 过滤短暂波动。

#### `hemo_map_lt65_ge6h` (binary)

```python
label = 1 if hours_lt65 >= 6 else 0
mask = 1 if len(map_vals) >= 8 else 0
```

**临床依据**: 持续低 MAP 6h+ 预示血流动力学不稳定。

#### `hemo_map_worst_bin` (multi-class)

```python
map_min = min([s.map for s in S_future if s.map_obs == 1], default=None)
if map_min is None:
    label = 0; mask = 0
elif map_min >= 70:
    label = 0  # normal
elif map_min >= 65:
    label = 1  # low
else:
    label = 2  # severe
mask = 1 if map_min is not None else 0
```

**临床依据**: SOFA 循环: MAP<70 扣 1 分，MAP<65（或需血管活性药）最高分。

#### `hemo_si_gt1_any` (binary)

```python
si_vals = []
for s in S_future:
    if s.hr_obs == 1 and s.sbp_obs == 1 and s.sbp > 0:
        si_vals.append(s.hr / s.sbp)
label = 1 if any(si > 1.0 for si in si_vals) else 0
mask = 1 if len(si_vals) >= 4 else 0
```

**临床依据**: Shock Index = HR/SBP > 1 是创伤常用高风险指标。

---

### 2. Respiratory

#### `resp_fio2_increase_ge015` (binary)

```python
fio2_now = S_t.fio2  # 当前小时值
fio2_max = max([s.fio2 for s in S_future if s.fio2_obs == 1], default=fio2_now)
label = 1 if fio2_max - fio2_now > 0.15 else 0
mask = 1 if S_t.fio2_obs == 1 and any(s.fio2_obs == 1 for s in S_future) else 0
```

**临床依据**: SOFA 呼吸: PaO2/FiO2 比值；FiO2 增加 0.15 反映呼吸支持升级需求。注：我们无 PaO2，用 FiO2 变化代理。

#### `resp_vent_started` (binary)

```python
vent_at_t = S_t.vent_status
vent_in_window = [s.vent_status for s in S_future]
label = 1 if vent_at_t == 0 and any(v == 1 for v in vent_in_window) else 0
# 不 mask——vent_status 总是可观测
```

#### `resp_vent_continued` (binary)

```python
label = 1 if vent_at_t == 1 and all(v == 1 for v in vent_in_window) else 0
```

---

### 3. Renal

#### `renal_creatinine_rise_ge03` (binary)

```python
cr_at_t = S_t.creatinine_value  # latest before t
cr_future = [s.creatinine_value for s in S_future if s.creatinine_obs == 1]
if not cr_future or cr_at_t is None:
    label = 0; mask = 0
else:
    label = 1 if cr_future[-1] - cr_at_t >= 0.3 else 0
    mask = 1
```

**临床依据**: KDIGO AKI Stage 1: Cr rise ≥ 0.3 mg/dL within 48h；12h 窗口为保守子集。

#### `renal_uop_low_ge6h` (binary)

```python
# 需要 patient weight；无体重时用绝对阈值备用
threshold = 0.5 * weight_kg if weight_kg else 30  # ml/h proxy
uop_vals = [s.uop_value for s in S_future if s.uop_obs == 1]
low_hours = sum(1 for u in uop_vals if u < threshold)
label = 1 if low_hours >= 6 else 0
mask = 1 if len(uop_vals) >= 8 else 0
```

**临床依据**: SOFA Renal: UOP < 0.5 ml/kg/h。**权重缺失用绝对阈值 30 ml/h 作为缺省**。

---

### 4. Perfusion / Metabolic

#### `perf_lactate_gt2` (binary)

```python
lac_vals = [s.lactate_value for s in S_future if s.lactate_obs == 1]
label = 1 if lac_vals and max(lac_vals) > 2.0 else 0
mask = 1 if len(lac_vals) >= 1 else 0
```

**临床依据**: Sepsis-3: lactate > 2 mmol/L = tissue hypoperfusion。

#### `perf_lactate_gt4` (binary)

```python
label = 1 if lac_vals and max(lac_vals) > 4.0 else 0
mask = 1 if len(lac_vals) >= 1 else 0
```

**临床依据**: SSC guideline: lactate > 4 = severe；区分中度 vs 重度灌注异常。

#### `perf_lactate_rising_gt05` (binary)

```python
if len(lac_vals) >= 2:
    label = 1 if lac_vals[-1] - lac_vals[0] > 0.5 else 0
    mask = 1
else:
    label = 0; mask = 0
```

**临床依据**: Shahar 1997 KBTA: trend abstraction (rising/stable/falling)。乳酸持续上升是不良预后标志。

#### `perf_be_worse` (binary)

```python
be_at_t = S_t.base_excess_value
be_future = [s.base_excess_value for s in S_future if s.base_excess_obs == 1]
if be_at_t is not None and be_future:
    label = 1 if be_future[-1] < be_at_t - 2 else 0  # BE 下降 2+ mEq/L
    mask = 1
else:
    label = 0; mask = 0
```

**临床依据**: Base excess 恶化反映代谢性酸中毒进展。

---

### 5. Infection / Treatment Escalation

> **注意**: 这些字段依赖 prescriptions / microbiologyevents 表，当前 registry 尚未完全覆盖。V2 初期标记为 `mask=0` 或 `candidate`。

#### `infx_new_abx` (binary)

```python
# 依赖 prescriptions 表
# V2 初期: mask=0（数据待 audit）
label = 0; mask = 0  # placeholder
```

#### `infx_new_culture` (binary)

```python
# 依赖 microbiologyevents 表
# V2 初期: mask=0
label = 0; mask = 0
```

#### `infx_suspected` (binary)

```python
label = infx_new_abx and infx_new_culture  # derivative, same mask
mask = mask_new_abx and mask_new_culture
```

---

### 6. Critical Events / Endpoint

这三个字段不做 mask（事件确定性高）。

#### `event_sepsis_onset` (multi-class: 0/1/censored)

```python
# 来源: infectionHour / Sepsis label
if sepsis_onset_hour in (t, t+12]:
    label = 1
elif discharge_or_death_before_sepsis_check:
    label = 2  # censored
else:
    label = 0
```

#### `event_death_12h` (binary)

```python
label = 1 if death_time and t < death_time <= t+12 else 0
```

#### `event_discharge_12h` (binary)

```python
label = 1 if discharge_time and t < discharge_time <= t+12 else 0
```

---

### 7. Composite

#### `comp_any_deterioration` (binary)

```python
deterioration_signals = [
    hemo_map_lt65_ge2h == 1,
    hemo_si_gt1_any == 1,
    resp_fio2_increase_ge015 == 1,
    resp_vent_started == 1,
    renal_creatinine_rise_ge03 == 1,
    renal_uop_low_ge6h == 1,
    perf_lactate_gt2 == 1,
    perf_lactate_gt4 == 1,
    perf_be_worse == 1,
]
# 只算未被 mask 的 signals
valid = [s for i, s in enumerate(deterioration_signals) if label_masks[i] == 1]
label = 1 if any(valid) else 0
mask = 1 if len(valid) >= 4 else 0  # 至少 4 个系统可评估
```

#### `comp_multi_system` (binary)

```python
# 至少 2 个不同系统出现恶化
systems = {
    'hemo': [hemo_map_lt65_ge2h, hemo_si_gt1_any, hemo_map_worst_bin==2],
    'resp': [resp_fio2_increase_ge015, resp_vent_started],
    'renal': [renal_creatinine_rise_ge03, renal_uop_low_ge6h],
    'perf': [perf_lactate_gt2, perf_lactate_gt4, perf_be_worse],
}
deteriorated_systems = sum(
    1 for sys, flags in systems.items()
    if any(f == 1 for f in flags)
)
label = 1 if deteriorated_systems >= 2 else 0
mask = 1 if len(valid_systems) >= 3 else 0  # 至少 3 个系统可评估
```

---

## 8. Censoring / Mask 统一规则

每个 label 有三个层级：

```
Layer 1: 临床 mask
  → 该小时此指标不可评估（如 lactate 没测、MAP 观测不足）
  → mask_clinical = 0

Layer 2: 死亡截尾
  → 病人在窗口内死亡
  → 死亡后的小时不应参与 label 计算
  → 对非终端 label，mask_death = 0
  → event_death_12h 不受此限制

Layer 3: 出院截尾
  → 病人在窗口内出院
  → mask_discharge = 0
```

最终 mask：

```python
mask = mask_clinical and (not censored_by_death or label_is_terminal) and (not censored_by_discharge)
```

---

## 9. Label 构造伪代码

```python
def build_report_labels(hadm_id, stay_id, t, S_future_12h, patient_kg=None):
    """
    S_future_12h: list of canonical hourly state dicts for t+1 to t+12
    Returns: dict of {label_name: {'value': int, 'mask': int}}
    """
    
    # ── Precompute common ──
    obs_hours = sum(1 for s in S_future_12h if s['any_observed'])
    window_len = len(S_future_12h)
    truncated = window_len < 12
    
    death_in_window = any(s.get('event', {}).get('death', False) for s in S_future_12h)
    discharge_in_window = any(s.get('event', {}).get('discharge', False) for s in S_future_12h)
    
    labels = {}
    
    # ── Hemodynamic ──
    map_vals = [s['map_value'] for s in S_future_12h if s['map_obs']]
    map_lt65_hours = sum(1 for m in map_vals if m < 65)
    map_clinical_mask = len(map_vals) >= 6
    
    labels['hemo_map_lt65_ge2h'] = {
        'value': int(map_lt65_hours >= 2),
        'mask': int(map_clinical_mask)
    }
    labels['hemo_map_lt65_ge6h'] = {
        'value': int(map_lt65_hours >= 6),
        'mask': int(map_clinical_mask and len(map_vals) >= 8)
    }
    
    if map_vals:
        map_min = min(map_vals)
        labels['hemo_map_worst_bin'] = {
            'value': 2 if map_min < 65 else 1 if map_min < 70 else 0,
            'mask': 1
        }
    else:
        labels['hemo_map_worst_bin'] = {'value': 0, 'mask': 0}
    
    # SI
    si_vals = [s['hr_value'] / s['sbp_value'] 
               for s in S_future_12h 
               if s['hr_obs'] and s['sbp_obs'] and s['sbp_value'] > 0]
    labels['hemo_si_gt1_any'] = {
        'value': int(any(si > 1.0 for si in si_vals)),
        'mask': int(len(si_vals) >= 4)
    }
    
    # ... (其余类比)
    
    # ── Censoring metadata ──
    labels['meta_observed_hours'] = {'value': obs_hours, 'mask': 1}
    labels['meta_window_truncated'] = {'value': int(truncated), 'mask': 1}
    labels['meta_censored_by_death'] = {'value': int(death_in_window), 'mask': 1}
    labels['meta_censored_by_discharge'] = {'value': int(discharge_in_window), 'mask': 1}
    
    return labels
```

---

## 10. 当前已知缺口

| 缺口 | 影响 label | 处理 |
|------|-----------|------|
| 无 vasopressor dose/itemid | pressor escalation | 移除，V2 不做 |
| 无 PaO2 | PaO2/FiO2 | 不用，FiO2 change 代理 |
| 无 GCS | SOFA neuro | 不依赖 |
| 无 platelets / bilirubin | SOFA coag/liver | SOFA 不完整 |
| infection/abx/culture 表未 audit | infx_* labels | V2 初期 mask=0 |
| 患者体重不完整 | uop threshold | 缺省 30 ml/h |

---

## 11. 与 report_schema_v1.md 的变化

| 变化 | 原因 |
|------|------|
| 移除 `hemo_pressor_escalation` | registry 无 vasopressor 数据 |
| 新增 `hemo_si_gt1_any` | Shock Index 是可算的高价值指标 |
| 新增 `perf_lactate_gt4` | 区分中度/重度灌注异常 |
| 新增 `perf_be_worse` | Base excess 趋势 |
| 新增 `infx_suspected` | 复合 infection proxy |
| 精确 mask 定义 | 每字段独立 mask |
| meta_ 前缀 | 元信息独立命名空间 |
