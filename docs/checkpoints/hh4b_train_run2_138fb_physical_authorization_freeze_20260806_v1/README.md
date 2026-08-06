# Run-2 physical-normalization authorization freeze

## Status

This checkpoint freezes the successful physical-normalization
authorization at repository head `f01df61d4bf69487a7044d1a1d42c84568996aa2` before this commit.

The authorization is for a Run-2 13 TeV expected-yield projection with
an integrated luminosity of 138 fb^-1.

## Authorized scope

- Primary physical sources: 441
- Auxiliary QCD sources excluded from physical use: 23
- Primary events: 3,569,873
- Accounting rows: 3,799,873
- Resolved rows: 1,171,072
- Signed negative-generator-weight rows: 70
- Source/event weight closure: pass
- Process, campaign, class, fold, and transport yield closure: pass
- Physical-weight application authorization: true for primary sources
- Auxiliary-QCD physical authorization: false
- Validation payloads opened: 0
- Test payloads opened: 0
- Models trained: 0
- Nominal cut threshold selected: false

The common-table Parquet payloads were not modified. Authorization is
carried by an immutable external sidecar that aliases the previously
audited stored columns for physical use.

## Pre-cut resolved yields

- Signal: 754.78495657184658
- Background: 62294487511.196976

These are broad resolved pre-cut yields, not the historical RHH34
baseline and not a final analysis selection.

## Next gate

Evaluate the unchanged historical `r_hh_125_125 < 34` comparator using
the authorized primary-source resolved weights. Do not optimize a new
threshold yet; that remains a separate nested five-fold source-group
OOF gate.
