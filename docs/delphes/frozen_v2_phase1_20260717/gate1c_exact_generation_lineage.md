# Gate 1C: exact Phase 1 generation lineage

Date: 2026-07-17

## Result

Status:

`GATE1C_EXACT_LINEAGE_VALID`

Validated:

- 25 manifest rows
- 25 exact lineage rows
- 25 unique MG5 run names
- 25 unique immutable run banners
- 25 unique HepMC SHA256 hashes
- 25 unique generation seeds
- 25 banner-to-metadata cross-section comparisons

Maximum relative difference between the MG5 banner integrated
cross section and the value propagated through the frozen-v2 metadata:

`5.486613340678164e-08`

## Process counts

- qcd_bbbb_general: 5 shards
- qcd_bbbb_iht400to600: 10 shards
- ttbar: 5 shards
- zbbbb: 5 shards

## QCD overlap rule

The general targeted QCD-bbbb sample has:

- `ihtmin = 0`
- `ihtmax = -1`

The targeted QCD-bbbb IHT slice has:

- `ihtmin = 400`
- `ihtmax = 600`

The general sample therefore contains the IHT 400–600 region.
These two samples must never be added as independent physical yields.

## Scientific interpretation

Exact lineage validates provenance and reproducibility. It does not
automatically validate final physical normalization.

- Targeted QCD-bbbb remains enrichment/diagnostic material.
- Targeted Zbbbb remains enrichment material until overlap with an
  inclusive Z+jets prescription is resolved.
- Inclusive ttbar is a candidate physical component, but its external
  cross-section normalization and decay prescription still require a
  final normalization contract.
- No additional production is authorized by this gate.
