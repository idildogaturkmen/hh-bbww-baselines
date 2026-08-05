# Broad-feature extractor and normalization integration freeze

This checkpoint does not redo the 5.2M physical normalization.

The source-group split, generated-event accounting, generator-weight
transport, cross-section coefficients, Run-2 luminosity convention, and
physical-weight formulas remain those of the existing physical-normalization
checkpoints.

This checkpoint freezes the integration of those identities into the new
rich broad-ML representation.

## Train production accounting

- 464 train sources
- 3,799,873 train events
- 441 primary physical sources and 3,569,873 events
- 23 classification-only auxiliary QCD sources and 230,000 events
- 442 remote bundle sources
- 22 direct ROOT sources

Primary transport classes:

- 241 ordinary uniform constants
- 14 ordinary signed exact-event sidecars
- 59 hard-QCD uniform constants
- 127 hard-QCD variable frozen fragments

The 23 auxiliary `qcd_bbbb` sources are not physical-yield samples.

## Important separation

The extractor stores generator nominal weights and source identities.
Run-2 physical weights are applied later by joining the already-frozen
process/source coefficient and multiplying it by the generator nominal
weight.

Validation and test remain sealed.

## Next gate

Build the resumable single-source production worker and release the complete
464-source train extraction.
