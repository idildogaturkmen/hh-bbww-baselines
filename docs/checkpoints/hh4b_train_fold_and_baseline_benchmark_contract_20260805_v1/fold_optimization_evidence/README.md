# Deterministic K=5 train-group fold optimization

The simple hash candidate was not frozen because its max-to-min ratios were
1.760049 for total broad rows, 2.633496 for signal, 1.770562 for background,
and 2.465159 for assignment supervision.

This preflight built per-source b-tag strata from the 464 immutable training
Parquet products and optimized a deterministic, capacity-constrained K=5
source-group assignment. Each of the 464 groups remains wholly inside one
fold.

## Optimized max-to-min ratios

- broad rows: 1.023325
- signal broad rows: 1.081991
- background broad rows: 1.026255
- assignment-matchable rows: 1.009916
- auxiliary-QCD broad rows: 1.360775

## Status

All acceptance gates passed: False

This mapping is still a preflight candidate. It must be frozen together with
the common table schema, sampler strata, training-loss weights, and validation
opening protocol before model training begins.

Physical luminosity weights remain unauthorized. Validation and test remain
sealed. No model was trained.
