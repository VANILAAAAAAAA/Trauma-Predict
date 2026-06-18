# UW Detail — quantity/forms for current fields

| Group | Field | UW source | Data form | Canonical unit/scale | 5-sample form | Note |
| --- | --- | --- | --- | --- | --- | --- |
| G2 | age | UW raw column `age` | static scalar / repeated per hourly row | years | {46, 46, 46, 46, 46} |  |
| G2 | male | UW raw column `male` | static scalar / repeated per hourly row | binary | {1, 1, 1, 1, 1} |  |
| G2 | mechanism_cat | UW raw column `MechanismCat` | static scalar / repeated per hourly row | category | {1, 1, 1, 1, 1} |  |
| G2 | transfer | UW raw column `transfer` | static scalar / repeated per hourly row | category | {0, 0, 0, 0, 0} |  |
| G2 | initial_ed_sbp | UW raw column `Initial.ED.SBP` | static scalar / repeated per hourly row | mmHg | {131, 131, 131, 131, 131} |  |
| G2 | rsi | UW raw column `rSI` | static scalar / repeated per hourly row | ratio SBP/HR | {2.3, 2.3, 2.3, 2.3, 2.3} |  |
| G2 | head_injury | UW raw column `headInjury` | static scalar / repeated per hourly row | binary | {1, 1, 1, 1, 1} |  |
| G1 | hr | UW raw column `hr` | hourly series value | bpm | {64, 60, 61, 58, 58} |  |
| G1 | sbp | UW raw column `sbp` | hourly series value | mmHg | {92, 100, 114, 112, 100} |  |
| G1 | dbp | UW raw column `dbp` | hourly series value | mmHg | {50, 57, 77, 66, 62} |  |
| G1 | map | UW raw column `map` | hourly series value | mmHg | {67, 72, 91, 82, 77} |  |
| G1 | rr | UW raw column `rr` | hourly series value | breaths/min | {16, 16, 16, 16, 16} |  |
| G1 | temp | UW raw column `temp` | hourly series value | degC | {37, 36.8, 36.5, 36.5, 36.5} |  |
| G1 | fio2 | UW raw column `fio2` | hourly series value | fraction [0,1] | {NA, 43, NA, 45, 45} | UW observed percent-style in raw/wide; canonical fraction requires /100 when >1. |
| G3 | vent_h | UW raw column `vent` | hourly binary/status series | binary | {1, 1, 1, 1, 1} |  |
| G3 | vent_day_sum_until_h | UW raw column `ventDaySum` | hourly cumulative series | days | {2, 2, 2, 2, 2} |  |
| G3 | bolus_sum_until_h | UW raw column `bolusSum` | hourly cumulative series | UW source-scale cumulative / MIMIC mL events | {1, 1, 1, 1, 1} | Cumulative source scale; positive deltas often 0.5/1.0; mL conversion unresolved. |
| G3 | rbc_sum_until_h | UW raw column `RBCsum` | hourly cumulative series | UW source-scale cumulative / MIMIC mL events | {0, 0, 0, 0, 0} | Source behaves like product units/count, not mL. |
| G4 | bicarb | UW raw column `bicarb` | hourly series value | mEq/L | {NA, NA, NA, 24, 24} |  |
| G4 | strong_ion | UW raw column `StrongIon` | hourly series value | mEq/L | {NA, NA, NA, 31, 31} |  |
| G4 | bun | UW raw column `bun` | hourly series value | mg/dL | {NA, NA, NA, 16, 16} |  |
| G4 | creatinine | UW raw column `creatinine` | hourly series value | mg/dL | {NA, NA, NA, 1.17, 1.17} |  |
| G4 | wbc | UW raw column `wbc` | hourly series value | K/uL | {NA, NA, NA, 6.95, 6.95} |  |
| G4 | lymphocytes | UW raw column `lymphocytes` | hourly series value | UW/MIMIC unit audit; MIMIC raw sample percent | {NA, NA, NA, NA, NA} | Unit/scale must be audited; do not silently mix percent and absolute counts. |
| G4 | neutrophils | UW raw column `neutrophils` | hourly series value | UW/MIMIC unit audit; MIMIC raw sample percent | {NA, NA, NA, NA, NA} | Unit/scale must be audited; do not silently mix percent and absolute counts. |
| G4 | uop | UW raw column `uop` | hourly series value | mL per recorded hour/interval | {NA, NA, NA, NA, NA} | Not cumulative; mL per recorded hour/interval; no weight field in UW tables. |
| G2* | base_def_48 | UW raw column `baseDef48` | static scalar / repeated per hourly row | mEq/L deficit magnitude | {6.3, 6.3, 6.3, 6.3, 6.3} | Only usable after full 48h history is available. |
| G2* | lactate_48 | UW raw column `lactate48` | static scalar / repeated per hourly row | mmol/L | {5.2, 5.2, 5.2, 5.2, 5.2} | Only usable after full 48h history is available. |
| G2* | rbc_48 | UW raw column `RBC48` | static scalar / repeated per hourly row | UW source-scale units/count; MIMIC mL-derived if rebuilt | {0, 0, 0, 0, 0} | Source behaves like product units/count, not mL. Only usable after full 48h history is available. |
| G2* | crys_48 | UW raw column `crys48` | static scalar / repeated per hourly row | mL | {5206, 5206, 5206, 5206, 5206} | Only usable after full 48h history is available. |
