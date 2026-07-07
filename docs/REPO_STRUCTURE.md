# Repository Structure

The repository is organized around a standard ML lifecycle while keeping restricted data outside Git.

```text
configs/
  accelerate/      Accelerator configs for Kaggle and local fallback.
  dataset/         Dataset artifact path and required-field configs.
  train/           Training run configs.
docs/              Human-readable repository policy and runbooks.
notebooks/kaggle/  Kaggle-specific entrypoints.
schemas/           Machine-readable data artifact contracts.
src/trauma_predict/
  data/            Manifest and split utilities.
  eval/            Metrics and evaluation helpers.
  training/        Training config and checkpoint helpers.
tests/             No-data tests for repo hygiene and schema contracts.
tools/             Maintenance tools.
```

## Design Rules

1. No source or derived clinical data is committed.
2. No `agent-artifact/` directory is committed.
3. Every tracked file must be listed in `docs/FILE_INDEX.md`.
4. Training code reads data through config paths, environment variables, or mounted datasets.
5. Patient split is keyed by `subject_id`; samples are keyed by `(subject_id, hadm_id, stay_id, prediction_hour)`.
