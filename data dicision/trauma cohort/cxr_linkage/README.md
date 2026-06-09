# C4 Cohort → MIMIC-CXR Linkage

Data: `mimic-cxr-jpg/2.0.0/` metadata CSV only. Local JPG images are **not downloaded yet**.

Scripts: `pipeline/02_cxr_linkage/`

| Script | Purpose |
|---|---|
| `link_c4_cxr.py` | Link C4 cohort to CXR metadata → C4-scoped layer counts, HADM membership CSV, DICOM/image-row CSV |
| `gen_cxr_download_list.py` | Generate exact filtered JPG URL list from `c4_cxr_dicom_layers.csv` |

## Corrected results

C4 denominator: **6,583 HADM**.

| Layer | Definition | HADM | Studies | Image rows |
|---|---|---:|---:|---:|
| L2 | C4 HADM with same-admission or ICU-window CXR | 2,059 | 12,480 | 14,947 |
| L3 | C4 HADM with ICU-window CXR | 1,760 | 9,608 | 11,004 |
| L4 | C4 HADM with ICU first312h CXR | 1,760 | 8,534 | 9,762 |
| L5 | C4 HADM with pre-ICU CXR and no ICU-window CXR | 101 | 134 | 167 |

Validation:

```text
L3 ⊂ L2  ✓
L4 ⊂ L3  ✓
L5 ⊂ L2  ✓
L5 ∩ L3 = ∅  ✓
```

## Important correction

Earlier download prep used **L3 subjects** and would have downloaded all CXR images for those subjects, including unrelated admission/time-window images. That was too broad.

Current download prep uses exact DICOM/image rows from:

```text
c4_cxr_dicom_layers.csv
```

Default download layer:

```text
L3_icu_window_cxr
```

Result:

```text
1,760 HADM
1,730 subjects
9,608 studies
11,004 JPG files
estimated size ≈ 10.7 GB
```

## Download entry point

Generated files:

```text
download_prep/c4_cxr_download_list.txt                 # default exact L3 URL list
download_prep/c4_cxr_download_list_L3_icu_window.txt   # same as default
download_prep/c4_cxr_download_summary.json
download_prep/download_c4_cxr_l3.sh                    # executable wrapper
```

The first URL was checked with `wget --spider` and returned `200 OK` using PhysioNet authentication.

### Run later, after approval

```bash
cd "/home/vanila/code/EHR-Predict/data dicision/trauma cohort"
bash "cxr_linkage/download_prep/download_c4_cxr_l3.sh"
```

The script:

- reads credentials from `/home/vanila/code/Physionet Oath.txt`
- downloads only exact L3 image URLs
- resumes partial files with `wget -c`
- writes log to `download_prep/c4_cxr_download.log`
- downloads to `/mnt/d/Data/mimic-cxr-jpg-c4/`

## Notes

- L3 and L4 have identical HADM count, but L4 has fewer studies/images. Meaning: every ICU-window positive HADM has at least one first312h CXR, but later ICU studies/images are removed in L4.
- CXR should remain an optional multimodal channel. It covers about 31% of C4 on same-admission/ICU basis and 26.7% on ICU-window HADM basis.
- CXR metadata/labels are local; JPG image files still require explicit download approval.
