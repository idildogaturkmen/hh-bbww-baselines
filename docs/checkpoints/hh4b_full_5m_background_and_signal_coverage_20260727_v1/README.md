# Full 5M background and unique-signal source coverage audit

Status: `hh4b_full_5m_background_and_signal_coverage_audit_pass`.

This is a source-level metadata audit only. It opened no candidate Parquet,
ROOT, model, prediction, or sealed-test content.

## Coverage result

- Background: 520 / 520 members,
  5,000,000 generated events.
- Signal: 110 / 110 unique members,
  200,000 generated events.
- Combined: 630 / 630 members,
  5,200,000 generated events.
- Candidate-row metadata accounted: 71,262.

The 27 canonical legacy-ttbar members are fully represented at source level
but remain a separate schema hold. They are not silently mixed into the
frozen-v2 canonical72 development population.

## Development reconciliation

The frozen development manifest matches exactly 581
non-test, common72-compatible members (474
background and 107 signal), with
70,071 candidate rows in its already-frozen
metadata. The 49 non-development source members are exactly 27 legacy-ttbar
schema holds, 19 sealed background-test members, and 3 sealed ggF-test
members.

## Sealed-test boundary

All 22 test members are covered by frozen
source metadata only. Their 193
candidate rows are accounting metadata; candidate files opened and candidate
rows read by this audit are both zero.

## Canonical-v1 preservation

The completed validation checkpoint remains unchanged:
`127d59487078411a15e43e694f2671e900bf9c9a77a4bd8e39610094c22a92dc`. All
88 entries in its `SHA256SUMS` manifest were
verified byte-for-byte.

## Authorization boundary

This audit does not authorize candidate-content access, full-5M table
evaluation, model training/scoring/evaluation, physical normalization, or
physical significance.

Candidate-reconstruction coverage status:
`blocked_on_27_ttbar_schema_holds`. Physical
normalization ready: `false`.
Unresolved coverage count: 27. Next gate:
`reconstruct_hh4b_ttbar27_canonical72_candidates`.
