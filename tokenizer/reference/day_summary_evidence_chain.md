# DAY/SUMMARY Evidence Chain

Verified: 2026-06-14

Purpose: external evidence for future DAY/SUMMARY salience gates, threshold candidates, and evidence levels. This file is reference only. It does not freeze the DAY token design.

## Evidence Levels

| Level | Meaning | Use |
|---|---|---|
| G | Guideline / consensus definition | may support a strong emit gate |
| T | Trauma / critical-care cohort, RCT, or validated clinical study | may support domain-specific gate or bucket candidate |
| M | EHR modeling / missingness / summarization methodology | supports representation or evaluation choice |
| C | Observational / physiology / candidate support | candidate only; do not freeze alone |
| X | broken, wrong, or mismatched source | do not cite |

## Hemodynamic

| ID | Topic | Candidate gate / use | Level | Source |
|---|---|---|---|---|
| HEM-01 | MAP target / hypotension burden | `map_low_hours` around MAP `<65 mmHg` | G | Sepsis-3, JAMA 2016. PubMed: https://pubmed.ncbi.nlm.nih.gov/26903338/ |
| HEM-02 | Initial resuscitation MAP target | MAP target `65 mmHg`; lactate-guided resuscitation context | G | Surviving Sepsis Campaign 2021. PubMed: https://pubmed.ncbi.nlm.nih.gov/34605781/ ; journal page: https://journals.lww.com/ccmjournal/Fulltext/2021/11000/Surviving_Sepsis_Campaign__International.21.aspx |
| HEM-03 | NEWS2 HR extreme | HR `>=131` or `<=40` as high-risk NEWS2 single-parameter threshold | G | Royal College of Physicians NEWS2 official PDF: https://www.rcp.ac.uk/media/a4ibkkbf/news2-final-report_0_0.pdf |
| HEM-04 | Trauma shock index | SI = HR/SBP; SI elevation relates to transfusion need in trauma | T | `The Shock Index revisited`, Critical Care 2013. PubMed: https://pubmed.ncbi.nlm.nih.gov/23938104/ |
| HEM-05 | Geriatric trauma SBP threshold | SBP `<110` can be important in geriatric trauma; useful for older-age SBP context | T | `Systolic blood pressure criteria ... 110 is the new 90`. PubMed: https://pubmed.ncbi.nlm.nih.gov/25757122/ ; PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC4620031/ |
| HEM-06 | qSOFA bedside prompt | SBP `<=100`, RR `>=22` are qSOFA components; screening context only | G | Sepsis-3, JAMA 2016. PubMed: https://pubmed.ncbi.nlm.nih.gov/26903338/ |

Notes:
- MAP `<65` is strong for perfusion/hypotension burden.
- Shock index uses `HR/SBP`; project STATIC uses UW reverse shock index `SBP/HR`, so direction must be explicit.
- `SBP <90` is common trauma hypotension context, but use current project token thresholds only after reference/bucket review.

## Respiratory

| ID | Topic | Candidate gate / use | Level | Source |
|---|---|---|---|---|
| RESP-01 | RR extreme | RR `>=25` or `<=8` supports `respiratory_rate_max` salience | G | Royal College of Physicians NEWS2 official PDF: https://www.rcp.ac.uk/media/a4ibkkbf/news2-final-report_0_0.pdf |
| RESP-02 | ARDS oxygenation severity | PaO2/FiO2 categories: mild `200<P/F<=300`, moderate `100<P/F<=200`, severe `P/F<=100`; requires PEEP/CPAP context | G | Berlin Definition, JAMA 2012. PubMed: https://pubmed.ncbi.nlm.nih.gov/22797452/ ; JAMA: https://jamanetwork.com/journals/jama/fullarticle/1160659 |
| RESP-03 | Ventilation as support marker | `vent_hours >0` is clinically salient as respiratory support; summarize burden, not just start event | G/T | Berlin Definition respiratory support context: https://pubmed.ncbi.nlm.nih.gov/22797452/ ; SSC critical care context: https://pubmed.ncbi.nlm.nih.gov/34605781/ |
| RESP-04 | FiO2 high oxygen demand | `fio2_max` candidate gate around `>0.40`; evidence indirect unless paired with oxygenation/ventilation support | C | Berlin Definition uses P/F ratio rather than FiO2 alone: https://pubmed.ncbi.nlm.nih.gov/22797452/ |

Notes:
- FiO2 alone is not an ARDS definition. Treat standalone FiO2 thresholds as pragmatic oxygen-demand gates until a direct oxygen-support reference is added.
- `vent_hours` is suitable as a DAY summary burden token because HOUR already records `[vent_on]` when active.

## Renal / Output

| ID | Topic | Candidate gate / use | Level | Source |
|---|---|---|---|---|
| REN-01 | AKI creatinine criterion | creatinine increase `>=0.3 mg/dL within 48h` | G | KDIGO AKI guideline page: https://kdigo.org/guidelines/acute-kidney-injury/ ; 2012 PDF: https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf |
| REN-02 | AKI creatinine ratio | creatinine `>=1.5x baseline` | G | KDIGO AKI guideline PDF: https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf |
| REN-03 | AKI urine output | urine output `<0.5 mL/kg/h for >=6h` | G | KDIGO AKI guideline PDF: https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf |
| REN-04 | BUN physiology | BUN and BUN/Cr support pre-renal physiology interpretation; candidate only | C | NCBI Bookshelf Clinical Methods BUN chapter: https://www.ncbi.nlm.nih.gov/books/NBK303/ |

Notes:
- `creatinine_change` should define the comparison baseline explicitly: previous day max/last, prior 48h minimum, or admission baseline.
- `urine_output_low_hours` requires weight if using mL/kg/h. Without weight, use a documented fallback such as absolute mL/h only as candidate.

## Metabolic / Acid-base

| ID | Topic | Candidate gate / use | Level | Source |
|---|---|---|---|---|
| MET-01 | Lactate in septic shock | lactate `>2 mmol/L` in septic shock construct with vasopressor/MAP context | G | Sepsis-3, JAMA 2016: https://pubmed.ncbi.nlm.nih.gov/26903338/ |
| MET-02 | Lactate-guided resuscitation | lactate is prognostic and used to guide resuscitation in sepsis | G | SSC 2021: https://pubmed.ncbi.nlm.nih.gov/34605781/ |
| MET-03 | Lactate general acute mortality | lactate prognostic value in acute hospital admissions; broad support, not trauma-specific | C | Systematic review protocol / acute hospital mortality context: https://pubmed.ncbi.nlm.nih.gov/22202128/ |
| MET-04 | Base deficit trauma classification | base deficit-based classification for hypovolemic shock in trauma | T | Critical Care 2013. PubMed: https://pubmed.ncbi.nlm.nih.gov/23497602/ |
| MET-05 | Base deficit and hypovolemic shock | base deficit supports recognition of hypovolemic shock beyond ATLS categories | T | Critical Care 2013. PubMed: https://pubmed.ncbi.nlm.nih.gov/23510230/ |
| MET-06 | Stewart/SID acid-base framework | SID and bicarbonate support acid-base interpretation; threshold candidates require local/reference confirmation | C | Critical Care clinical review: https://ccforum.biomedcentral.com/articles/10.1186/cc2908 |

Notes:
- `lactate_48h` and `base_deficit_48h` are only safe after the 48h window is complete.
- `bicarbonate_min <22` and SID cutpoints are clinically plausible but should remain candidate until tied to a direct normal-range/guideline source or local lab reference.

## Inflammatory / Hematologic

| ID | Topic | Candidate gate / use | Level | Source |
|---|---|---|---|---|
| INF-01 | SIRS WBC | WBC `>12,000/mm3` or `<4,000/mm3`; inflammatory signal, not modern sepsis diagnosis | G historical | ACCP/SCCM 1992 consensus. PubMed: https://pubmed.ncbi.nlm.nih.gov/1597042/ |
| INF-02 | Sepsis-3 de-emphasizes SIRS | SIRS is not sufficient for sepsis; use WBC as inflammatory marker only | G | Sepsis-3, JAMA 2016: https://pubmed.ncbi.nlm.nih.gov/26903338/ |
| INF-03 | NLR risk marker | neutrophil-to-lymphocyte ratio as mortality/risk marker in critical illness; candidate derived feature | C | PMC article: https://pmc.ncbi.nlm.nih.gov/articles/PMC6657279/ |
| INF-04 | RBC transfusion threshold context | hemoglobin/RBC transfusion decisions; general guideline if hemoglobin is later added | G | AABB 2023 RBC transfusion guideline. PubMed: https://pubmed.ncbi.nlm.nih.gov/37824153/ |
| INF-05 | Trauma bleeding/coagulopathy | platelet/hemorrhage management context if platelet is later added | G/T | European trauma bleeding/coagulopathy guideline, 6th edition: https://ccforum.biomedcentral.com/articles/10.1186/s13054-023-04327-7 |

Notes:
- Current UW-aligned V1 may not include hemoglobin/platelets. Keep INF-04/INF-05 as optional expansion evidence.
- NLR is useful but observational; do not promote to a frozen V1 gate without cohort validation.

## Treatment / Resuscitation

| ID | Topic | Candidate gate / use | Level | Source |
|---|---|---|---|---|
| TX-01 | Crystalloid 30 mL/kg | sepsis-induced hypoperfusion initial crystalloid reference; indirect for trauma | G | SSC 2021: https://pubmed.ncbi.nlm.nih.gov/34605781/ |
| TX-02 | Crystalloid volume in trauma | `>=5L` crystalloid in first 24h associated with worse outcomes; supports high-volume burden context | T | BMC Surgery 2018. PubMed: https://pubmed.ncbi.nlm.nih.gov/30400852/ ; PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC6219036/ |
| TX-03 | Massive transfusion hourly definition | `>4 U PRBC/hour` proposed as improved massive transfusion definition | T | J Trauma Acute Care Surg 2015. PubMed: https://pubmed.ncbi.nlm.nih.gov/26680135/ |
| TX-04 | Balanced transfusion in severe trauma | RBC/plasma/platelet transfusion ratios in severe trauma; supports transfusion as critical treatment signal | T/RCT | PROPPR trial, JAMA 2015. PubMed: https://pubmed.ncbi.nlm.nih.gov/25647203/ |
| TX-05 | Major bleeding/coagulopathy after trauma | resuscitation, coagulopathy, platelet/RBC/plasma context | G/T | European guideline 6th edition: https://ccforum.biomedcentral.com/articles/10.1186/s13054-023-04327-7 |
| TX-06 | RBC transfusion general threshold | restrictive vs liberal transfusion thresholds; general inpatient guideline, not trauma-specific summary gate | G | AABB 2023. PubMed: https://pubmed.ncbi.nlm.nih.gov/37824153/ |

Notes:
- For MIMIC-IV, use `inputevents.amount`/`amountuom` and interval-to-hour splitting for `bolus_input_1h_ml` and `rbc_transfusion_1h_ml`.
- `bolus_daily_total_ml` buckets should not be frozen from SSC alone; SSC is sepsis context. Trauma-specific volume evidence currently stronger at first-24h/48h burden than single-day gate.
- RBC `>0` is a reasonable emit rule as treatment event. High-load bins require mL/unit conversion or explicit PRBC-unit definition.

## Data Quality / Missingness / Representation

| ID | Topic | Use | Level | Source |
|---|---|---|---|---|
| DQ-01 | Informative missingness | missingness and time gaps can be predictive; supports coverage/missingness tokens | M | GRU-D / missing values in time series. Nature Scientific Reports: https://www.nature.com/articles/s41598-018-24271-9 ; arXiv: https://arxiv.org/abs/1606.01865 |
| DQ-02 | Clinical time-series benchmark masks | MIMIC time-series benchmark uses value/mask/time-series structure; supports explicit masks/coverage | M | Harutyunyan et al., Scientific Data 2019: https://www.nature.com/articles/s41597-019-0103-9 |
| DQ-03 | Clinical summarization evaluation | factual agreement, critical omissions, unsupported inference should be separate evaluation axes | M | Local wiki synthesis: `/home/vanila/wiki/concepts/clinical-note-summarization-for-daily-blocks.md` and raw source listed there |

Notes:
- `labs_measured`, `no_labs_measured`, `low_vital_coverage`, and `low_output_coverage` are justified as representation/evaluation controls, not diagnosis claims.
- Missingness tokens should record evidence quality; they should not imply clinical normality.

## Longitudinal / Temporal Dependence References

| ID | Topic | Use | Level | Source |
|---|---|---|---|---|
| TIME-01 | Event self-excitation / decay analogy | supports considering residual state/trend features as candidate, not V1 architecture requirement | C/M | Hawkes process overview: https://en.wikipedia.org/wiki/Hawkes_process ; use primary literature before formal citation |
| TIME-02 | Time-aware attention / decay as modeling idea | candidate for later ablation, not required for V1 | M | GRU-D decay formulation: https://www.nature.com/articles/s41598-018-24271-9 |

Notes:
- V1 should prefer explicit `DAY_REL` time embeddings plus selected change/burden tokens before custom attention-bias mechanisms.

## Rejected / Corrected Links from `对话.md`

| Original claim | URL | Status | Replacement |
|---|---|---|---|
| RCP NEWS2 web page | https://www.rcp.ac.uk/guidelines-and-policy/national-early-warning-score-news2/ | X: 404 | Official PDF: https://www.rcp.ac.uk/media/a4ibkkbf/news2-final-report_0_0.pdf |
| NEWS2 PubMed evidence | https://pubmed.ncbi.nlm.nih.gov/31167367/ | X: wrong article (`Trueperella pyogenes`) | Use RCP NEWS2 PDF above |
| Shock Index revisited | https://pubmed.ncbi.nlm.nih.gov/23506935/ | X: wrong article (ophthalmology reply) | Correct PubMed: https://pubmed.ncbi.nlm.nih.gov/23938104/ |
| Shock Index PMC | https://pmc.ncbi.nlm.nih.gov/articles/PMC3672535/ | X: wrong article (A-FABP in critical illness) | Correct PubMed: https://pubmed.ncbi.nlm.nih.gov/23938104/ |
| Base deficit trauma | https://ccforum.biomedcentral.com/articles/10.1186/cc12542 | X: ARDS/EVLWI letter, not base deficit | Use https://pubmed.ncbi.nlm.nih.gov/23497602/ and https://pubmed.ncbi.nlm.nih.gov/23510230/ |
| ACS TQIP massive transfusion PDF | https://www.facs.org/media/x1pfz2o5/massivetransfusion.pdf | X: 404 during verification | Use TX-03/TX-05 until a current ACS URL is found |
| PubMed crystalloid trauma ID previously mentioned as `30068354` | https://pubmed.ncbi.nlm.nih.gov/30068354/ | X: wrong article (sports nutrition review) | Correct PubMed: https://pubmed.ncbi.nlm.nih.gov/30400852/ |

## Use Policy for Summary Design

When adding a DAY/SUMMARY token or gate:

```text
1. Point to at least one evidence ID above.
2. If evidence is G or strong T: gate may become candidate-freeze after source/unit audit.
3. If evidence is C/M only: keep as candidate or evaluation feature.
4. If source is X: do not cite.
5. If field source/unit is unresolved: keep continuous value or hold bucket.
```

Do not convert this reference file directly into token vocabulary without checking current field constructibility, unit normalization, and cutoff leakage.
