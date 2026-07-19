# GRU-D H1 Joint-M4 Baseline

This baseline tests whether the proposed multi-resolution and registered-relation method improves the same forecasting task over a recognized recurrent missing-data model. It is not intended to reproduce the main architecture with a smaller encoder.

The persisted sample identity is unchanged: the same 50,350 C4 anchors, patient split, visibility boundary, and full-r9 target hashes are used. The input representation is the already audited H1 sidecar. Each sample becomes 118 hourly value, observation-mask, and elapsed-time channels over at most 312 hours. An omitted tuple is missing; an emitted zero is observed. CXR study slots become per-label hourly counts.

The encoder is GRU-D. Its final history state initializes an ordinary recurrent target decoder. The decoder advances in the frozen block-major order over 6 M4 blocks and 29 field processes, for 174 causal positions. During training, position p receives only the registered target feedback from positions before p. During free-running, it receives its own previously sampled feedback. The existing typed V2 emission heads, 414-factor registry, raw joint NLL, and target projections are reused unchanged.

The baseline deliberately excludes FAR/MIDDLE/NEAR input, Transformer layers, the 52 target-target edges, the 39 input-target edges, relation bias, target attention, and eight-field temporal fusion. These are the method components being tested. The causal recurrent decoder is only the adapter required to express the same joint six-block task.

Formal training is a fresh 4,000-step single-P100 Kaggle run. The training Notebook performs only input binding, the offline P100 runtime restore, training, checkpoint selection, and export. Full 6,309-anchor teacher-forced evaluation and 100-trajectory free-running evaluation are separate jobs so an evaluation failure cannot invalidate a completed training run.
