# Adaptive physical-QCD Wave 2 review

Campaign:

`qcd_hardqcd_importance_adaptive500k_phys_wave2_20260717`

## Reviewed configuration

- 500,000 generated events
- 56 whole-shard jobs
- eight disjoint inclusive-HardQCD pTHat strata
- unique seeds and shard IDs
- frozen-v2 detector card
- no use of the existing final test outcomes in allocation

## Test design

Wave 1 test coverage contained bins 0, 2, and 3.

Wave 2 predeclares sealed test shards in missing bins 1, 4, 5, 6,
and 7 before event generation.

Combined test coverage therefore contains all eight strata.

Wave 2 contains 140,000 test events. This larger-than-standard test
fraction is intentional because the Wave 1 physical test set was too
small for stable rare-tail evaluation.

## Authorization

The exact reviewed manifest is authorized for one submission only,
after payload, EL9 compile, storage, identity, EOS, and Condor dry-run
checks pass.

No 5M campaign is authorized.
