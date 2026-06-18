# Summary Gate Rules

| 规则 | 所属模块 | 可靠性 |
|---|---|---|
| 任一小时 MAP `<65` mmHg → emit `[map_low_hours_bin_*]` | hemodynamic | strong gate / candidate burden bucket |
| 日最低 SBP `<=100` mmHg 或 age>=65 且 SBP `<110` → emit `[systolic_bp_min_bin_*]` | hemodynamic | moderate-plus |
| 日最高 HR `>=131` bpm → emit `[heart_rate_max_bin_ge_131]` | hemodynamic | strong |
| 日最高 RR `>=25` 次/分 → emit `[respiratory_rate_max_bin_ge_25]` | respiratory | strong |
| 当天有通气 → emit `[vent_hours_bin_*]` 和 `[vent_day_index_bin_*]` | respiratory | strong gate / candidate burden bucket |
| 日最高 FiO2 `>0.40` fraction → emit `[fio2_max_bin_*]` | respiratory | candidate |
| 肌酐 48h 内上升 `>=0.3 mg/dL` 或 `>=1.5×` 基线 → emit `[creatinine_change_bin_ge_0_3]` / `[creatinine_ratio_bin_ge_1_5]` | renal_output | strong |
| 尿量低输出：若有体重则按 `<0.5 mL/kg/h` 持续 `>=6h` → emit `[urine_output_low_hours_kdigo]`；若无体重则不启用该 token | renal_output | strong when weight exists / source-dependent otherwise |
| 乳酸 `>2.0 mmol/L` → emit `[lactate_48h_bin_*]`（仅 48h 窗口完成后） | metabolic | strong |
| 碱缺乏指示中重度代谢性酸中毒 → emit `[base_deficit_48h_bin_*]`（仅 48h 窗口完成后） | metabolic | moderate-plus |
| 碳酸氢盐 `<22 mEq/L` → emit `[bicarbonate_min_bin_lt_22]` | metabolic | candidate |
| WBC `>12,000` 或 `<4,000 /mm³` → emit `[wbc_max_bin_gt_12]` / `[wbc_min_bin_lt_4]` | inflammatory_hematologic | moderate |
| 晶体液/bolus：`>0 mL` 不足以代表 trauma resuscitation burden；暂不默认 emit，等待 treatment 规则重设 | treatment_resuscitation | hold |
| 有 RBC 输注（>0 mL）→ emit `[rbc_daily_total_present]` | treatment_resuscitation | moderate-plus |
| 当天无 lab 观测 → emit `[no_labs_measured]`；有则 `[labs_measured]` | data_quality | moderate |
| 生命体征覆盖不足 → emit `[low_vital_coverage]` | data_quality | moderate |
| 尿量/输出覆盖不足 → emit `[low_output_coverage]` | data_quality | moderate |
