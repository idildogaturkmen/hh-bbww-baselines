# HH4b real-Delphes four-b parity canary

## Purpose

This checkpoint compares the immutable frozen HH4b four-b candidate builder
with the new builder's explicit `fourb-parity` mode on a deterministic,
train-only set of real Delphes canaries.

## Result

- Status: `hh4b_4b_real_delphes_parity_pass`
- Selected and processed canaries: 8
- Passing canaries: 8
- Failing canaries: 0
- Candidate rows compared: 2304
- Column comparisons: 576
- Total parity mismatches: 0
- Zero-row canaries: 1
- Temporary files remaining: 0

Selected member indices: 1, 18, 99, 123, 184, 413, 443, 543.

The selection was generated from declarative rules using the lowest train
member index per rule, followed by deduplication. Both builders read the same
extracted ROOT file and used identical explicit reconstruction parameters.
Arrow types, nullability, row ordering, values, provenance indices, and
zero-row behavior were compared. No physics-shape plots were used.

## Safety record

- Validation canaries: 0
- Test canaries: 0
- ROOT files opened in three-b mode: 0
- Three-b candidates reconstructed: 0
- Physics yields calculated: 0
- Normalizations calculated: 0
- Transfer factors calculated: 0
- Significances calculated: 0

All downloaded bundles, nested archives, extracted ROOT files, and temporary
Parquet outputs were deleted after their individual comparison.

## Reproducibility

- Source commit: `7f511757e8888f6f93cbe4e130cf62548a778a7e`
- Source-map SHA-256: `e797ab8cc878ea1fbb84c05cd52fb8919c6e67bb56bab07bef46701615a29f60`
- Frozen builder SHA-256: `4d7eb8e3400ec50c1c254744e6f3a2b11bd033a4525c13ec22e85247ecf59c57`
- Parity builder SHA-256: `d81a27538437d90878f98d7f7760e5aa574814b9af6da99810642d76c6c2afa8`
- Configuration SHA-256: `a38402161807392f36c583571642c6685730f8b96c6d2f54c15101e1026bb0db`
- Validation script SHA-256: `6ad1bfddf6c5eed23d26fa7751408c943dcddc71810f59157a625ecf9f831b36`

## Next gate

`run_hh4b_3b_control_train_canary`
