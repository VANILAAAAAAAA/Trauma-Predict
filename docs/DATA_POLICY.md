# Data Policy

Trauma-Predict is a code-only repository. It must remain safe to push to GitHub without exposing restricted clinical data or generated derivatives.

## Allowed

| Category | Examples |
| --- | --- |
| Source code | Python package, tests, Kaggle launchers, maintenance tools. |
| Configuration | YAML configs with environment-variable paths. |
| Schemas | JSON Schema files that describe manifests and samples. |
| Documentation | Repository runbooks, structure notes, index tables. |

## Not Allowed

| Category | Examples |
| --- | --- |
| Restricted source data | MIMIC-IV, MIMIC-CXR, ED, CXR linkage exports. |
| Derived patient data | Sample JSONL shards, manifests with patient rows, feature tables. |
| Training artifacts | Checkpoints, predictions, metrics from restricted samples. |
| Agent workspace state | `agent-artifact/`, archived project state, local Codex run outputs. |

## Upstream Boundary

Return to `/home/vanila/code/EHR-Predict` for:

- cohort extraction evidence,
- field adapter development,
- sample builder source decisions,
- historical design documents,
- agent-managed project state.

This repository only consumes finalized data artifacts through external paths.
