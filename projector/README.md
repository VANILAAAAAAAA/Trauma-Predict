# EHR-Predict Vital Numeric Projector

This folder prototypes the separated numeric-projection side of the EHR-Predict input design.

Scope:

```text
HOUR vitals only:
  hr, sbp, dbp, map, rr, temp, fio2

Input tensors:
  vital_values [N, T, 7]
  vital_mask   [N, T, 7]

Target tensors:
  target_values [N, 7]
  target_mask   [N, 7]
```

Non-vital DAY/STATIC/REPORT tokens are intentionally out of scope. This tests whether the 7-vital numeric projection and next-hour regression path can run independently.

## Files

```text
build_vital_dataset.py  # extract fixed [T,7] windows from raw MIMIC chartevents
model.py                # VitalValueProjector + HourStateEncoder + NextVitalHead
train_smoke.py          # masked Huber next-hour vital training smoke test
run_pipeline.py         # build dataset + train smoke end-to-end
requirements.txt        # dependency declaration; current environment already has torch/numpy
artifacts/              # generated .npz and smoke result JSON
```

## Run

```bash
python3 projector/run_pipeline.py --workdir /home/vanila/code/EHR-Predict
```

Outputs:

```text
projector/artifacts/vital_dataset_sample.npz
projector/artifacts/vital_dataset_manifest.json
projector/artifacts/train_smoke_result.json
```

## Contract

- no LOCF;
- missing values use `vital_mask=0` and value placeholder `0` only after standardization/masking;
- FiO2 remains fixed slot 6 even when sparse;
- loss is masked by `target_mask`;
- split is stay-level/patient-level for the small selected-stay smoke test.
