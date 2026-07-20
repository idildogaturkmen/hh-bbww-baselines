# Frozen-v2 ZZ and ZH pilots

These records describe the exact-lineage ZZ→bbbb and ZH→bbbb frozen-v2 Delphes pilots promoted to EOS and independently verified by an EOS round-trip audit.

## Status

- Local lineage and artifact audit: valid
- EOS promotion: valid
- Independent EOS round trip: valid
- Total generated events: 20,000
- Total reconstructed candidate rows: 441
- Dataset role: train/validation pilots
- Test usage authorized: no

## Samples

| Sample | Seed | Generator cross section [pb] | Events | Candidates | Acceptance |
|---|---:|---:|---:|---:|---:|
| ZH4b | 715002 | 0.06882 | 10000 | 315 | 0.031500 |
| ZZ4b | 715001 | 0.19298 | 10000 | 126 | 0.012600 |

The ROOT, HepMC, Parquet, and tar.gz payloads are not stored in Git. Their local and EOS identities are recorded through SHA-256, Adler-32, sizes, manifests, receipts, and the canonical pilot registry.

These samples must not be treated as sealed test data.
