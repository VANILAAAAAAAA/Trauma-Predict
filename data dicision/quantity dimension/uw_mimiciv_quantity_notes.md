| Topic | UW form / risk | MIMIC-IV form / handling |
| --- | --- | --- |
| FiO2 | UW values observed percent-style (20–100 or 0.4-like in processed samples); MIMIC can be percent or fraction | canonical fraction [0,1]; divide values >1 by 100 |
| Temperature | UW raw appears °C; MIMIC sample contains °F item | convert MIMIC °F to °C; apply outlier filter |
| Bolus/crystalloid | UW `bolusSum` is cumulative source-scale; mL conversion unresolved | MIMIC `inputevents.amount` is mL interval event; split to hourly mL; do not use UW delta bins as MIMIC bins |
| RBC | UW `RBCsum/RBC48` behaves like units/count/source-scale, not mL | MIMIC PRBC inputevents are mL; use hourly/daily mL or explicitly convert to units only after rule |
| Ventilation | UW has hourly `vent` plus cumulative `ventDaySum` | MIMIC needs active status reconstructed from chartevents/procedureevents; HOUR emits `[vent_on]`, DAY can summarize `vent_hours` |
| UOP | UW has `uop` mL-like output value but no weight | MIMIC outputevents are mL event/interval; KDIGO mL/kg/h gate requires added weight source; otherwise hold low-uop gate |
| Labs timing | UW table is hourly layer / carried series; exact draw availability not explicit | MIMIC labevents has charttime/storetime; use availability-time if realtime leakage matters |
| Lymphocytes/neutrophils | UW scale must be audited; do not assume absolute count | MIMIC sample itemids are percent; absolute counts require separate itemids or WBC-derived conversion |
| StrongIon | UW has direct StrongIon series | MIMIC must derive `(Na+K)-(Cl+bicarb)` from labs with aligned timing |
| First-48h fields | UW provides fixed first-48h summaries | MIMIC must derive from events/labs; only emit after full 48h history is available |
| Static ED fields | UW has ED SBP/rSI static values | MIMIC requires ED linkage; absent linkage must be missing, not ICU proxy unless explicit |
| Category fields | UW categories can be code/index without codebook semantics | MIMIC categories must be derived from continuous/source rules; do not assign unsupported clinical labels |
