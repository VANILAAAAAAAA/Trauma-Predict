# GRU-D H1 Sample Contract

This bundle freezes the input sidecar for the matched classic GRU-D baseline. It vendors the
unchanged V1 field, ID, aggregation, unit, missingness, and visibility registries, then restricts
the input view to the 118 registered H1 channels over the same frozen anchors and patient split.

The sidecar stores exact sparse five-tuples. A later model adapter may densify them into GRU-D
value, observation-mask, and elapsed-time tensors; that adapter must not change sample identity,
input visibility, target identity, or the r9 six-M4 prediction task.

No patient records or generated sample shards belong in this directory.
