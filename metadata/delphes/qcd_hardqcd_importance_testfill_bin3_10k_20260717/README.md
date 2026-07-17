# QCD bin-3 sealed test-fill campaign

Campaign:

`qcd_hardqcd_importance_testfill_bin3_10k_20260717`

## Reason

A pre-outcome manifest audit found that the combined Wave 1 and
reviewed Wave 2 test set contained only 4,543 events in pTHat bin 3
(200-300 GeV).

This 10,000-event shard was declared before inspection of Wave 2
physics outcomes.

## Role

- inclusive Pythia8 HardQCD physical background;
- pTHat bin 3 only;
- sealed test split only;
- not used for allocation, tuning, model selection, or training.

After completion, combined bin-3 test statistics will contain
14,543 generated events.

The supplement must be combined with the other physical-QCD shards
using stratum-level physical normalization. It must not be normalized
as an independent QCD component.
