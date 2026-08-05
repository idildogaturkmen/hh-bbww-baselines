# Complete train common-table freeze

## Status

The complete Track A train common table is frozen at repository head
`4d0b6616cf4ee03279f9d204b8ec45d1f944f13c` before this checkpoint commit.

The production comprises seven byte-identically reused pilot sources and
457 successful HTCondor production jobs, for 464 sources total.

## Frozen closure

- Condor production jobs: 457/457 completed with exit code zero
- Accounting rows: 3,799,873
- Resolved rows: 1,171,072
- Non-broad rows: 2,628,801
- Assignment-matchable rows: 30,397
- Primary sources/events: 441 / 3,569,873
- Auxiliary QCD sources/events: 23 / 230,000
- Global event UID uniqueness: pass
- Resolved rows equal the exact broad accounting subset: pass
- Fold-local comparison-weight closure: pass
- Diagnostic physical-weight arithmetic: pass
- Historical reconstruction parity: pass
- Physical-weight application authorization: false
- Validation payloads opened: 0
- Test payloads opened: 0
- Models trained: 0

The Parquet payloads remain outside Git. This checkpoint contains only
manifests, receipts, checksums, summaries, and audit evidence needed to
reproduce and verify the freeze.

## Next gate

Run the dedicated physical-normalization authorization and Run-2
138 fb^-1 yield-closure gate. The unchanged historical
`R_HH^(125,125) < 34` comparator must not be evaluated with physical
yields until that authorization passes.

## Git whitespace policy

The immutable copied production-plan TSV contains 457
lines ending in tab bytes because its terminal columns are empty. The
checkpoint copy remains byte-identical to the audited source, with
SHA-256 `039d3d4227be8ccc597e79d71f86df78b02045ec78ad93131f91874efd73f0b2`.

A checkpoint-local `.gitattributes` rule disables only the
`trailing-space` diagnostic for that exact evidence file. No TSV bytes
were normalized, and every other Git whitespace check remains enabled.
The diagnosis and byte-identity proof are preserved in
`evidence/git_whitespace_policy_receipt.json`.
