# CXR Token Vocabulary

> Draft 2026-06-18. CXR is an optional sparse event view. The current V1
> interface is token-based and should not require downloading JPG images unless
> a VLM-derived label pipeline is explicitly enabled.

## Scope

CXR provides time-stamped chest-imaging findings aligned to the same patient
trajectory as STATIC, DAY, and HOUR.

Current C4 linkage:

```text
C4 denominator: 6,583 HADM
L3 ICU-window CXR: 1,760 HADM / 9,608 studies / 11,004 image rows
```

CXR events are optional. Patients without observed CXR before the prediction
time emit no finding tokens; missing modality should be handled by a modality
mask or by absence of CXR_EVENT blocks, not by normal-finding imputation.

## Event Unit

The CXR event unit is one `study_id`, not one `dicom_id`.

Reason:

```text
MIMIC-CXR CheXpert labels are study-level: subject_id + study_id.
The C4 linkage table is image-row/DICOM-level and may contain multiple images
for the same study. Emitting per image would duplicate the same label set.
```

## Sequence

```text
[CXR_EVENT_T_*]
[cxr_finding_*] ...
[CXR_SEP]
```

`[CXR_EVENT_T_*]` is a relative-time structural token. The exact time-bucket
scheme is not frozen yet; it must be derived from CXR acquisition or label
availability time relative to `observed_until_t`.

Example:

```text
[CXR_EVENT_T_minus_52h]
[cxr_finding_cardiomegaly] [cxr_finding_edema]
[CXR_SEP]
```

## Source Options

### Option A: MIMIC-CXR CheXpert labels

Source:

```text
/mnt/d/Data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv.gz
```

Join keys:

```text
subject_id
study_id
```

Label rule:

```text
1.0  -> emit [cxr_finding_*]
0.0  -> no token
-1.0 -> no token in V1
empty -> no token
```

Important leakage note:

```text
CheXpert labels are derived from radiology reports, not raw images.
If used as model input, the label availability time must be <= observed_until_t.
CXR acquisition time alone is sufficient for image availability, but may be too
early for report-derived label availability.
```

Until report availability is audited, CheXpert-input use should be marked as a
pragmatic approximation or used only in an ablation.

### Option B: VLM-derived image labels

Raw CXR images can be passed through a vision-language or radiology image model
to produce a short structured representation. This avoids depending on future
radiology report text, but requires downloading images and running a separate
inference pipeline.

VLM output must be normalized back to the same token interface:

```text
raw image -> VLM findings -> controlled finding vocabulary -> [cxr_finding_*]
```

Do not feed free-form VLM prose directly into the Method 1 token-slot pipeline.
If free text is retained, it belongs to the later structured-text input path.

## Finding Tokens

Retained V1 finding tokens:

```text
[cxr_finding_atelectasis]
[cxr_finding_cardiomegaly]
[cxr_finding_consolidation]
[cxr_finding_edema]
[cxr_finding_enlarged_cardiomediastinum]
[cxr_finding_fracture]
[cxr_finding_lung_opacity]
[cxr_finding_no_finding]
[cxr_finding_pleural_effusion]
[cxr_finding_pneumonia]
[cxr_finding_pneumothorax]
[cxr_finding_support_devices]
```

Excluded from V1:

```text
[cxr_finding_lung_lesion]
[cxr_finding_pleural_other]
```

These were removed because their positive rates are low in the current C4
ICU-window linkage audit.

## Time Boundary

For any prediction landmark `observed_until_t`:

```text
emit CXR event only if cxr_event_available_time <= observed_until_t
```

Candidate time definitions:

| Time source | Meaning | Status |
|---|---|---|
| `cxrtime` from metadata | image acquisition / study time | usable for image-derived VLM labels |
| report time | report-derived label availability | required for strict CheXpert-as-input use |
| future window CXR | after `observed_until_t` | target-side only, never input |

## Model Encoding

CXR uses sparse token slots:

```text
time_emb[cxr_relative_time]
+ segment_emb[CXR]
+ token_emb[cxr_finding_id]
```

The interface should remain stable if V1 starts with CheXpert labels and later
switches to VLM-derived labels. Only the label source changes; the model-facing
tokens stay controlled.

## V1 Holds

```text
raw image embedding
free-form CXR caption text
uncertain-label tokens
negative-label tokens
report-generation targets
```

These can be revisited after the CXR availability audit and VLM label pipeline
are defined.
