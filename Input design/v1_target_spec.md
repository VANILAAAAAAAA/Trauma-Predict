---
title: V1 Next-Hour Prediction Target Specification
created: 2026-06-08
status: draft
tags: [target, v1, self-supervised, next-hour]
depends:
  - input_schema_v1.md
  - data dicision/field adapter/uw_style_field_table_v1.md
---

# V1 Next-Hour Prediction Target

## 核心设计

```
history ≤ t → encoder → z_t → next_hour_head → predict S_{t+1}
```

V1 是纯自监督任务。target 直接来自 canonical hourly state 的 `t+1` 切片，无需人工标注。

---

## 1. 预测目标总览

| 类别 | 字段 | 输出类型 | Loss |
|------|------|---------|------|
| G1 vitals | hr, sbp, dbp, map, rr, temp, fio2 | continuous | masked Huber / nMAE |
| G3 binary flags | vent_status, in_surgery | binary | masked BCE |
| G3 continuous | iv_fluid_ml_1h, rbc_ml_1h, surgery_hours_1h | continuous | masked Huber |
| G4 labs | lactate, creatinine, bicarb, base_excess, bun, wbc, lymphocytes, neutrophils, Na, K, Cl | continuous | masked Huber (仅当 obs=1) |
| G4 lab occurrence | 同上 11 labs | binary | BCE（预测下一小时是否新测量） |
| G4 lab tsl | 同上 11 labs | continuous | Huber（预测 time_since_last） |
| G4 derived | StrongIon, acidosis_cat | continuous / categorical | masked Huber / CE |
| G4 output | uop | continuous | Huber |

---

## 2. 逐字段定义

### 2.1 G1 — Hourly Vitals

每次预测都适用（vitals 是 dense hourly）。

#### 预测头输出

每个 vital k 输出一个实数值：$\hat{v}_{k, t+1}$

#### Loss（masked by observed）

$$\mathcal{L}_{\text{vital},k} = \mathbf{1}_{\text{obs}_{k,t+1}=1} \cdot \ell_{\text{Huber}}(\hat{v}_{k,t+1}, v_{k,t+1}^{\text{true}})$$

`obs_{k,t+1}` = 下一小时是否有该 vital 的 chartevents 测量。

**为什么用 masked 而不是对所有小时 MSE**：LOCF 值不是"真实当前测量"。不应对 LOCF 小时强行回归，否则模型学会的是复制上一小时值。

#### 评估时也 mask

```python
nMAE_k = MAE(v_pred[obs_mask], v_true[obs_mask]) / std(v_true[obs_mask])
```

---

### 2.2 G1 Vitals — 派生阈值评估（不上 loss）

**不训练**以下目标，仅在评估阶段派生：

| 派生事件 | 从什么预测值 | 阈值 |
|---------|-------------|------|
| `map_lt65` | $\hat{\text{map}}_{t+1}$ | < 65 |
| `map_lt70` | $\hat{\text{map}}_{t+1}$ | < 70 |
| `sbp_lt90` | $\hat{\text{sbp}}_{t+1}$ | < 90 |
| `si_gt1` | $\hat{\text{hr}}_{t+1} / \hat{\text{sbp}}_{t+1}$ | > 1.0 |
| `rr_ge22` | $\hat{\text{rr}}_{t+1}$ | ≥ 22 |
| `fio2_ge05` | $\hat{\text{fio2}}_{t+1}$ | ≥ 0.5 |
| `temp_ge38` | $\hat{\text{temp}}_{t+1}$ | ≥ 38 |

评估指标：AUROC, AUPRC, calibration。不进入训练 loss。

---

### 2.3 G3 — Interventions

#### Binary flags（vent_status, in_surgery）

$$\mathcal{L}_{\text{binary}} = -\big[y\log\hat{y} + (1-y)\log(1-\hat{y})\big]$$

不做 mask——vent 和 surgery 的 0/1 状态总是有明确含义（0 = 未通气/未手术，不是缺失）。

#### Continuous（iv_fluid_ml_1h, rbc_ml_1h, surgery_hours_1h）

$$\mathcal{L}_{\text{cont}} = \ell_{\text{Huber}}(\hat{v}, v^{\text{true}})$$

这些值 0 就是 0（未干预），不需要 mask。

---

### 2.4 G4 — Labs

Lab 的关键特征是**间歇测量**。需要同时预测三个维度。

#### 2.4.1 Value prediction（仅当下一小时有测量）

$$\mathcal{L}_{\text{lab\_value},k} = \mathbf{1}_{\text{obs}_{k,t+1}=1} \cdot \ell_{\text{Huber}}(\hat{v}_{k,t+1}, v_{k,t+1}^{\text{true}})$$

#### 2.4.2 Measurement occurrence prediction

预测下一小时是否有该 lab 的新测量：

$$\mathcal{L}_{\text{lab\_obs},k} = -\big[o_{k,t+1}\log\hat{o}_{k,t+1} + (1-o_{k,t+1})\log(1-\hat{o}_{k,t+1})\big]$$

#### 2.4.3 Time-since-last update prediction

预测下一小时该 lab 距上次测量的时间：

$$\mathcal{L}_{\text{lab\_tsl},k} = \ell_{\text{Huber}}(\hat{\text{tsl}}_{k,t+1}, \text{tsl}_{k,t+1}^{\text{true}})$$

#### 为什么不合并成一个 loss？

- `value` 只在 obs=1 时可训练（否则 false target）
- `obs` 总是可训练（观测过程建模）
- `tsl` 总是可训练（时效性建模）

三个维度分别给 loss，让模型学习"这个 lab 什么时候会测、测出来大概多少、现在有多旧"。

---

### 2.5 G4 — Derived fields

#### StrongIon

同 G4 lab value：masked by obs。

#### acidosisCat

Cross-entropy，masked by obs。

---

### 2.6 G4 — UOP

UOP 是每小时输出，不间歇。所有小时都回归：

$$\mathcal{L}_{\text{uop}} = \ell_{\text{Huber}}(\hat{\text{uop}}_{t+1}, \text{uop}_{t+1}^{\text{true}})$$

---

## 3. 总 Loss 组装

$$\mathcal{L}_{\text{V1}} = \lambda_{\text{vitals}} \cdot \frac{1}{|G1|}\sum_{k\in G1} \mathcal{L}_{\text{vital},k} + \lambda_{\text{intv}} \cdot \frac{1}{|G3|}\sum_{k\in G3} \mathcal{L}_{k} + \lambda_{\text{labs}} \cdot \frac{1}{|G4_{\text{lab}}|}\sum_{k\in G4_{\text{lab}}} (\mathcal{L}_{\text{value},k} + \alpha\mathcal{L}_{\text{obs},k} + \beta\mathcal{L}_{\text{tsl},k}) + \lambda_{\text{uop}} \cdot \mathcal{L}_{\text{uop}}$$

初始权重建议：

| weight | value | 理由 |
|--------|-------|------|
| $\lambda_{\text{vitals}}$ | 1.0 | 主任务 |
| $\lambda_{\text{intv}}$ | 1.0 | 主任务 |
| $\lambda_{\text{labs}}$ | 1.0 | 主任务 |
| $\alpha$ | 0.3 | lab occurrence 是辅助 |
| $\beta$ | 0.1 | tsl 是辅助 |
| $\lambda_{\text{uop}}$ | 1.0 | 主任务 |

---

## 4. 输出头架构建议

### 4.1 Head 结构

```text
z_t (shared encoder output, e.g. 256-dim)
  → shared MLP (可选，如 256→128)
    ├─ G1 head: Linear(128 → 7)          # 7 vitals continuous
    ├─ G3 head: Linear(128 → 5)          # 2 binary + 3 continuous
    ├─ G4 value head: Linear(128 → 14)   # 11 labs + StrongIon + uop + acidosisCat
    ├─ G4 obs head:  Linear(128 → 11)    # 11 lab occurrence logits
    └─ G4 tsl head:  Linear(128 → 11)    # 11 lab tsl values
```

### 4.2 输出约束

| Head | 输出 | 激活 | 
|------|------|------|
| G1 vitals | 7 floats | 无（或 ReLU 对非负变量） |
| G3 binary | 2 logits | Sigmoid |
| G3 continuous | 3 floats | ReLU（量非负） |
| G4 value | 14 floats | 无 |
| G4 obs | 11 logits | Sigmoid |
| G4 tsl | 11 floats | ReLU（tsl ≥ 0） |

---

## 5. 评估指标

### 5.1 主要指标

| 指标 | 范围 | 合格线（建议） |
|------|------|--------------|
| nMAE (vitals, observed only) | [0, ∞) | < 0.8 |
| nMAE vs LOCF baseline | ratio | < 1.0 (优于 LOCF) |
| lab value MAE (when observed) | [0, ∞) | 按 lab 分别看 |
| lab obs AUROC | [0.5, 1] | > 0.65 |
| vent status F1 | [0, 1] | > 0.7 |

### 5.2 Gating criteria (是否进入 V2)

1. vitals nMAE **明显**优于 LOCF（至少 3/4 关键 vitals：MAP, HR, RR, FiO2）
2. MAP<65 derived AUROC > 0.65
3. SI>1 derived AUROC > 0.65
4. 至少 2 个 lab obs AUROC > 0.65
5. leakage audit 干净

---

## 6. 与 V2 的关系

V1 的 `next_hour_head` 输出 V2 也保留。V2 新增 `future_summary_head`，两者共享 encoder：

```text
            ┌─ next_hour_head    (train from V1, continue in V2)
z_t ────────┤
            └─ future_summary_head (new in V2)
```

V2 不改变 V1 的训练目标，只增加一个 head。
