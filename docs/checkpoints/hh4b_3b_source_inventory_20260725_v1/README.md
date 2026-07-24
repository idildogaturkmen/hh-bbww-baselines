# HH4b 3b-control source inventory

## Purpose

This checkpoint inventories all available source-provenance channels
needed to reconstruct a three-b-tag control category from the frozen
HH4b development membership.

It deliberately does not assume that the candidate-output registries
contain one complete ROOT-path field.

## Failure corrected

The original preflight required a single signal-registry ROOT field with
complete coverage. No such field exists because ROOT provenance is
distributed across registries, candidate provenance columns, production
receipts, and cache/source manifests.

The absence of one complete registry field is not evidence that source
ROOT files are missing.

## Frozen membership

- Development members: 581
- Signal members: 107
- Background members: 474
- Train members: 458
- Validation members: 123
- Development candidate rows: 70,071
- Test members considered: 0

## Metadata findings

- Nonzero candidate Parquet metadata files opened:
  361
- Zero-row candidate files skipped:
  220
- Files with 72 columns:
  264
- Files with 75 columns:
  97
- Candidate files whose schema contains `source_root`:
  97
- Candidate event arrays read: 0
- ROOT files opened: 0

The 97 files with 75 columns contain the provenance columns
`source_root`, `source_root_index`, and `analysis_sample`.

## Interpretation

This checkpoint establishes the available evidence channels but does
not yet claim a complete member-level ROOT-source map.

The next stage must resolve each of the 581 development members using a
documented hierarchy:

1. direct member-level ROOT fields in authoritative manifests;
2. `source_root` provenance values in compatible candidate Parquets;
3. production receipts and materialization manifests;
4. cache/source mapping records.

No test source may be accessed.

## Reproducibility

- Source commit: `8a6ad6e1c1d501a1bd22e6e4b60e10658107c40a`
- Development manifest SHA-256:
  `b3e42a0af56594445e297e2347ccfd3bd799fd107a578af844c9733bfd2ebe6a`
- Frozen builder SHA-256:
  `4d7eb8e3400ec50c1c254744e6f3a2b11bd033a4525c13ec22e85247ecf59c57`
- Frozen policy SHA-256:
  `4b2a951dba7ac8bd0e2e982e3447e998e07b86d63390c248600976b3f4912b68`

## Next gate

`build_member_level_root_source_resolution_map`
