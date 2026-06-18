# schema-v6 summary-only / mixed 实验收尾结论

## 一句话结论

本周 schema-v6 eval50 的三模式结果可以收尾：**text-only 是当前最稳主线；mixed 在 scalar 上接近 text-only 但没有总体胜出；summary-only 是负结果/诊断性消融，不应作为主实验结论。** 官方 `all_data_*_los_noisy` 缺失不是当前 general inference 的 blocker；如果直接补 repo-style remaining `LOS`，在 custom next-hour eval 中属于未来端点泄漏，而且 oracle probe 也没有救回 summary/mixed。

## 实验定义

- task file: `/home/vanila/code/ehrtopath-rich-to-structured/artifacts/schema_v6_textonly_eval50_tasks_add.jsonl`
- base records: `50`
- target tasks: `676`
- target families: `routine_vitals`, `respiratory`, `labs`, `partial_output`, `procedure`
- evaluation policy: scalar 看 `coverage + normalized_mae + carry_forward_skill`；binary 看 `accuracy + balanced_accuracy + carry_forward_accuracy`。
- runner:
  - text-only: `/mnt/d/Data/ehr2path_logs/schema_v6_textonly_eval50_hf_add.jsonl`
  - summary-only: `/mnt/d/Data/ehr2path_logs/schema_v6_summaryonly_eval50_hf_prefixed_add.jsonl`
  - mixed: `/mnt/d/Data/ehr2path_logs/schema_v6_mixed_eval50_hf_prefixed_add.jsonl`

## 三模式主结果 eval50

|mode|n|scalar_cov|macro_nMAE↓|CF_skill↑|bin_acc|bin_bacc|bin_CF_acc|
|---|---|---|---|---|---|---|---|
|text_only|676/676|1.000|0.212|-0.710|0.880|0.775|0.920|
|summary_only|676/676|1.000|0.429|-1.855|0.200|0.500|0.920|
|mixed|676/676|1.000|0.233|-0.837|0.200|0.500|0.920|

读法：`CF_skill > 0` 才说明优于 carry-forward。三种模式总体都还没有稳定击败 carry-forward；但 text-only 的 scalar nMAE 最低、binary balanced accuracy 最高。mixed scalar 接近 text-only，summary-only 明显退化。

## summary-only 与 mixed 的输入语义

- `summary_only`：section summary embeddings + `LOS-only` scaffold。当前 schema-v6 task 原始 desc 里没有真实 repo `LOS`，所以该路径更接近“summary embeddings + 空/弱 time scaffold”，不是论文里完整 processed LOS condition。
- `mixed`：section summary embeddings + recent structured text。它保留了最近显式数值，因此 scalar 表现接近 text-only。
- 这不是 checkpoint swap；summary/mixed 跑的是 summary-embedding bridge + HF-safe fallback，保留 repo section/embedding/scaffold 语义，但避开本机 RTX50/cu128 下 Unsloth summary-cache segfault。

## 长历史子集检查

|subset|mode|n|macro_nMAE↓|CF_skill↑|bin_bacc|
|---|---|---|---|---|---|
|>=168h|text_only|168/168|0.192|-0.218|0.917|
|>=168h|summary_only|168/168|0.451|-1.885|0.500|
|>=168h|mixed|168/168|0.230|-0.379|0.500|
|>=240h|text_only|126/126|0.194|-0.224|0.917|
|>=240h|summary_only|126/126|0.472|-1.856|0.500|
|>=240h|mixed|126/126|0.244|-0.422|0.500|

结论：长历史没有救回 summary-only。`>=168h` 和 `>=240h` 子集里 summary-only 仍显著弱于 text-only，说明问题不是简单的“短记录太多”。

## LOS / oracle probe

|run|macro_nMAE↓|CF_skill↑|bin_acc|bin_bacc|bin_CF_acc|
|---|---|---|---|---|---|
|text_only_base|0.212|-0.710|0.880|0.775|0.920|
|summary_only_base|0.429|-1.855|0.200|0.500|0.920|
|summary_only_oracle_los|0.434|-1.877|0.520|0.662|0.920|
|mixed_base|0.233|-0.837|0.200|0.500|0.920|
|mixed_oracle_los|0.233|-0.835|0.180|0.450|0.920|

结论：把 `remaining_hours = end_hourTally - cutoff_hourTally` 构造成 oracle `LOS` 后：

- summary-only scalar 没有改善：nMAE `0.429 -> 0.434`，CF skill `-1.855 -> -1.876`。
- mixed scalar 几乎不变：nMAE `0.234 -> 0.233`。
- binary 有波动但仍低于 carry-forward，且 oracle LOS 是未来端点泄漏，不能作为 primary metric。

因此，缺 official LOS processed artifacts 不应继续作为本周阻塞；它最多是后续“官方 pathway/LOS 复现”实验，而不是当前 schema-v6 general inference 的必要条件。

## 失败模式诊断

|run|target|n|unique_pred|top_pred|top_rate|MAE|CF_MAE|collapse|
|---|---|---|---|---|---|---|---|---|
|summary_only_base|creatinine|50|9|1.0|0.28|0.55|0.20|False|
|summary_only_base|inspired_o2|42|1|100.0|1.00|52.95|10.12|True|
|summary_only_base|invasive_vent|50|1|0.0|1.00|0.80|0.08|True|
|summary_only_base|neutrophils|22|1|1.0|1.00|12.84|4.46|True|
|summary_only_base|nibp_mean|50|4|102.0|0.74|17.42|7.00|False|
|summary_only_base|nibp_systolic|50|11|102.0|0.66|29.38|8.20|False|
|summary_only_base|urine_output|41|3|100.0|0.83|67.32|55.49|True|
|mixed_base|creatinine|50|27|0.5|0.14|0.21|0.20|False|
|mixed_base|inspired_o2|42|9|30.0|0.24|10.48|10.12|False|
|mixed_base|invasive_vent|50|1|0.0|1.00|0.80|0.08|True|
|mixed_base|neutrophils|22|5|1.0|0.77|12.23|4.46|False|
|mixed_base|nibp_mean|50|21|102.0|0.16|6.48|7.00|False|
|mixed_base|nibp_systolic|50|17|102.0|0.16|7.24|8.20|False|
|mixed_base|urine_output|41|9|100.0|0.56|55.12|55.49|False|

主要模式：

1. **summary-only coarse prior / constant-collapse**：`invasive_vent` 预测全 `0`，若干 numeric target 回归到少数常数或粗粒度先验；说明 latent summaries 没有提供 next-hour 精确数值所需的 local numeric precision。
2. **mixed 保留 scalar，但 binary 崩**：mixed 的 BP、urine/output 等局部 scalar 有时接近或略优于 text-only，但 `invasive_vent` 仍全 `0`，总体不胜出。
3. **carry-forward 很强**：next-hour target 对 routine vitals/labs 是强 temporal persistence 场景。没有超过 carry-forward 的模式不能声称临床预测有效。
4. **LOS 不是根因**：oracle remaining-LOS 没有改善 scalar，所以不是简单缺 LOS scaffold。

## 本周可写进 thesis/report 的结论

- **方法结论**：schema-v6 constrained target + prefix-aware parser 已经能让三模式输出 100% parse coverage；评估链路可用。
- **实验结论**：text-only 是当前 schema-v6 next-hour structured prediction 的最强稳定 baseline；mixed 是接近但不胜出的 ablation；summary-only 是 negative result。
- **解释结论**：summary-only 的弱点来自输入语义和任务错配：section summaries + 空/LOS-only scaffold 更适合 coarse trajectory/pathway endpoint，不适合 next-hour exact numeric target。
- **风险结论**：repo-style remaining LOS 是未来信息，不能作为 custom eval 主输入。官方 LOS processed data 缺失不阻塞当前 general inference 收尾。
- **下一步决策**：本周停止追 LOS/summary-only rescue；后续如果继续，只做两个小方向：① 非泄漏 elapsed-time scaffold；② mixed/text-only 的小规模 schema-adaptation 或 calibration，并始终对比 carry-forward。

## 产物索引

- wrapup JSON: `/home/vanila/code/ehrtopath-rich-to-structured/artifacts/schema_v6_summary_mixed_wrapup_add.json`
- overall CSV: `/home/vanila/code/ehrtopath-rich-to-structured/artifacts/schema_v6_summary_mixed_overall_add.csv`
- field metrics CSV: `/home/vanila/code/ehrtopath-rich-to-structured/artifacts/schema_v6_summary_mixed_field_metrics_add.csv`
- failure diagnostics CSV: `/home/vanila/code/ehrtopath-rich-to-structured/artifacts/schema_v6_summary_mixed_failure_diagnostics_add.csv`
- oracle LOS report: `/home/vanila/code/ehrtopath-rich-to-structured/artifacts/schema_v6_oracle_los_probe_report_add.md`
