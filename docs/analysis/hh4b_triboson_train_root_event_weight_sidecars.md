# HH4b train-only triboson ROOT event-weight sidecars

PN-c3c recovered signed nominal ROOT-weight sums for all six triboson campaigns, but the
per-event numerator had not yet been transported into the candidate layer.

PN-c3d operates only on the 12 train source shards from the three scaleout campaigns. It
writes an immutable external sidecar for every source ROOT event with:

- `event`;
- `generator_nominal_weight`;
- `generator_weight_sign`.

The `event` column in the frozen HH4b candidate builder is the original Delphes entry
index. Each existing train candidate file is checksum-verified and opened only for that
column. The gate proves that every candidate event index is unique, lies within the mapped
ROOT file, and joins exactly to the external event-weight sidecar.

The existing candidate files are never changed. The three validation candidate files are
recorded but remain unopened.

Once the 12 sidecar sums reproduce the PN-c3c campaign sums, the signed normalization
denominators are frozen for the three train scaleout campaigns. No validation denominator,
external reference cross section, physical event weight, yield, model, or threshold is
authorized.
