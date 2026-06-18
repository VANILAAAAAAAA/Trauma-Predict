# CXR VLM Pipeline Design

> Draft 2026-06-18. This pipeline is upstream of final CXR input design.
> The goal is to derive image-based CXR text/labels first, then decide which
> CXR tokens or structured-text fields belong in EHR-Predict.

## Why This Is Needed

MIMIC-CXR CheXpert labels are derived from radiology reports, not directly from
raw CXR images. They are useful as a reference and weak label source, but they
do not solve the core input-design question:

```text
What image-derived CXR information is available at prediction time, and which
parts are useful for trauma ICU state prediction?
```

CXR may contain trauma-relevant findings that are not fully captured by the
standard CheXpert label set, such as fracture, tube/device position, lung
collapse, pneumothorax, hemothorax-like pleural opacity patterns, or other
support-related context.

Therefore the CXR branch should be designed after a VLM-derived representation
audit, not only from the existing CheXpert table.

## Current CXR Data Anchor

Use the C4 trauma cohort linkage:

```text
C4 denominator: 6,583 HADM
L3 ICU-window CXR: 1,760 HADM / 9,608 studies / 11,004 image rows
default image list: download_prep/c4_cxr_download_list_L3_icu_window.txt
```

The current default image list is exact L3 image rows, not all images for L3
subjects.

## Pipeline Order

```text
1. Select CXR event scope
2. Download a small pilot image subset
3. Run a CXR-capable VLM
4. Normalize VLM output into structured findings
5. Compare against CheXpert labels
6. Audit trauma-relevant finding coverage
7. Redesign CXR input tokens / structured text fields
8. Only then scale to all L3 images
```

Do not start with full 10.7 GB image download unless the pilot proves the
representation is useful and the output schema is stable.

## Event Unit

Primary event unit:

```text
study_id
```

Reason: MIMIC-CXR labels and reports are study-level, while the linkage table is
DICOM/image-row-level. Multiple images can belong to the same study. The VLM
pipeline may run per image, but the model-facing CXR event should aggregate to
one study-level representation.

Aggregation rule to freeze later:

```text
image-level VLM outputs -> study-level finding set
```

For multi-view studies, positive or high-confidence findings can be unioned
across images, with view metadata retained for audit.

## Candidate VLM Models

Initial Hugging Face candidates:

| Model | Role | Notes |
|---|---|---|
| `StanfordAIMI/CheXagent-2-3b-srrg-findings` | first pilot | CXR-specific image-to-text findings model; best fit for short findings. |
| `StanfordAIMI/CheXagent-2-3b` | VQA / controlled extraction | Can be prompted for yes/no finding table or concise findings. |
| `StanfordAIMI/CheXagent-8b` | stronger but heavier baseline | More expensive; use after 2-3B pilot if needed. |
| `ChantalPellegrini/RaDialog-interactive-radiology-report-generation` | report-style baseline | Useful comparison, but may produce longer report-like prose. |

General-purpose VLMs are not first choice because CXR interpretation needs
radiology-specific training.

## Required VLM Output Format

Do not store only free-form prose. Each VLM run should produce both:

```text
short_text: concise findings/impression text
structured_findings: controlled yes/no/uncertain table
```

Draft JSON shape:

```json
{
  "subject_id": "...",
  "hadm_id": "...",
  "study_id": "...",
  "dicom_ids": ["..."],
  "cxrtime": "...",
  "model": "StanfordAIMI/CheXagent-2-3b-srrg-findings",
  "prompt_version": "cxr_vlm_pilot_v1",
  "short_text": "...",
  "findings": {
    "atelectasis": "yes|no|uncertain|not_mentioned",
    "cardiomegaly": "yes|no|uncertain|not_mentioned",
    "consolidation": "yes|no|uncertain|not_mentioned",
    "edema": "yes|no|uncertain|not_mentioned",
    "fracture": "yes|no|uncertain|not_mentioned",
    "lung_opacity": "yes|no|uncertain|not_mentioned",
    "pleural_effusion": "yes|no|uncertain|not_mentioned",
    "pneumonia": "yes|no|uncertain|not_mentioned",
    "pneumothorax": "yes|no|uncertain|not_mentioned",
    "support_devices": "yes|no|uncertain|not_mentioned",
    "trauma_relevant_other": "free short phrase or null"
  }
}
```

The controlled table lets Method 1 map findings to tokens. The short text can
support the later structured-text input design.

## Prompt Direction

Pilot prompt should force concise, structured output:

```text
You are reviewing a chest radiograph for ICU trauma state modeling.
Return only:
1. one short findings sentence
2. a JSON object with yes/no/uncertain/not_mentioned labels for the requested
   findings.
Do not infer clinical history beyond the image.
```

The finding list should include both CheXpert-style categories and
trauma-relevant additions discovered during pilot review.

## CheXpert Role

CheXpert labels remain useful for:

```text
sanity check
coverage comparison
weak supervision
backward-compatible baseline tokens
```

CheXpert labels should not be treated as the final CXR input source because they
are report-derived and may not be available at image acquisition time.

## Leakage Rule

For image-derived VLM labels:

```text
cxrtime <= observed_until_t
```

For report-derived CheXpert labels:

```text
report_or_label_available_time <= observed_until_t
```

If report availability time is unknown, CheXpert-as-input is an approximation
and must be marked as such.

## Pilot Selection

Pilot should be small and stratified:

```text
positive pneumothorax by CheXpert
positive fracture by CheXpert
positive support devices by CheXpert
positive pleural effusion / lung opacity
CheXpert no-finding
uncertain/blank-heavy studies
```

Suggested first pilot size:

```text
50-100 studies, study-level, exact L3 ICU-window CXR only
```

Do not print raw patient rows in audit reports. Aggregate counts and redacted
examples only.

## Output Artifacts

Suggested locations:

```text
data dicision/trauma cohort/cxr_linkage/vlm_pilot/
  cxr_vlm_pilot_manifest.json
  cxr_vlm_pilot_outputs.jsonl
  cxr_vlm_pilot_aggregate_audit.md
```

These are pilot artifacts, not final sample-builder outputs.

## Downstream Design Questions

The VLM pilot should answer:

```text
1. Are the 12 CheXpert-style tokens enough?
2. Which trauma-relevant CXR findings need new tokens?
3. Should CXR enter Method 1 as controlled tokens, Method 2 as structured text,
   or both?
4. How many CXR events per patient are useful before the sequence becomes noisy?
5. Should multiple studies in the same window be kept separately or summarized?
6. Does VLM output agree enough with CheXpert to use CheXpert as a large-scale
   fallback when images are unavailable?
```

Final CXR token design should be updated only after this pilot audit.
