# GRU-D H1 Baseline Sample

This stage builds a new input-only sidecar for a matched classic GRU-D baseline. It reuses the
frozen 50,350 C4 sample IDs, prediction anchors, patient split, exact clinical values, availability
gate, and r9 six-M4 targets. The only representation change is that historical input uses H1 blocks
only; M4 and F24 history are absent.

Each sample covers ICU hour 0 through its prediction anchor with consecutive one-hour blocks. Its
events keep the registered five-tuple form
`[field_id, operator_id, condition_id, exact_value, block_id]`. The 37-field registry exposes 118
legal H1 channels. Values are not normalized in the sample artifact.

The target is referenced by `sample_id`, target content hash, target shard, and target line rather
than copied. A later GRU-D adapter will deterministically convert sparse H1 events into channel
values, observation masks, and elapsed-time deltas. Model, optimizer, training, and evaluation code
are intentionally outside this sample-building stage.

Build through `tools/build_grud_h1_baseline_samples.py`. Source and output roots are supplied by
arguments or the environment variables named in `configs/dataset/grud_h1_baseline_c4.yaml`;
generated clinical artifacts remain outside Git.

The full external artifact completed with 50,350 samples in 52 shards, split
37,734/6,309/6,307 for train/validation/test. It contains 5,197,428 H1 blocks and 213,422,601
emitted registered events. The dataset-manifest SHA-256 is
`2d30bdd75071f50b1631639087c2338e69ae346ec1facad13c6a8285e70288cf` and the consolidated
sample-manifest SHA-256 is
`6762897d5f516dc3442a7a206bc3bf19c3e43e32a2444f2807a475d3db61412b`.
