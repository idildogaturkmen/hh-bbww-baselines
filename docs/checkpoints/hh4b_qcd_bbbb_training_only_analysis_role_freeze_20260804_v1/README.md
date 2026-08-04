# QCD bbbb training-only analysis-role freeze

This checkpoint resolves the 23-source difference between the complete
train manifest and the physically normalized PN-c7 source registry.

## Inventory

- Members: 23
- Generated events: 230000
- Candidate rows: 22457
- Candidate file existence, row counts, and SHA-256 checks: exact
- Remote source bundles present: 23/23

## Generator result

Both source families use the hard process:

    p p > b b~ b b~ QED=0

The representative `qcd_bbbb_general` source is inclusive in partonic HT:

    ihtmin = 0.0
    ihtmax = -1.0
    generator cross section = 347.012401 pb

The representative `qcd_bbbb_iht400to600` source uses:

    ihtmin = 400.0
    ihtmax = 600.0
    generator cross section = 9.7152351 pb

The 400--600 GeV sample is therefore a subset of the inclusive sample.

## Frozen analysis role

These samples are not authorized as independent physical-yield
contributions and must not be summed with each other or with the stitched
`qcd_hardqcd` prediction.

Their allowed role is auxiliary train-only QCD enrichment.

They must not contribute to:

- Run-2 expected yields;
- category optimization metrics;
- significance calculations;
- validation or test metrics;
- the final likelihood.

Auxiliary model variants must be compared against the no-enrichment
variant using the same physically authorized source-group OOF evaluation.

The machine-readable contract is in `analysis_role_contract.json`.
