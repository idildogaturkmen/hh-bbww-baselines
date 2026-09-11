# Adaptive physical-QCD 500K decision

Campaign:

`qcd_hardqcd_importance_adaptive500k_phys_20260717_0022`

## Technical result

Status: production valid

- 54 jobs completed successfully
- 500,000 generated events
- 54 verified EOS bundles
- 4,266 cross-layer checks passed
- frozen-v2 detector-card SHA256 verified
- no missing receipts
- no failed jobs
- no partial bundles

## Statistical result

Status: signal-like tail not converged

- HH-like rows: 27
- rHH < 80 rows: 11
- rHH < 50 rows: 5
- HH-like ESS: 3.188
- rHH < 80 ESS: 3.169
- rHH < 50 ESS: 1.991
- HH-like maximum event fraction: 0.411098
- rHH < 50 maximum event fraction: 0.593906
- HH-like bootstrap relative standard deviation: 50.55%
- rHH < 80 bootstrap relative standard deviation: 50.70%
- rHH < 50 bootstrap relative standard deviation: 66.76%

The loose exactly-2b and exactly-3b controls improved substantially,
but the four-b-tag signal-like physical estimator is not sufficiently
stable for final inference.

## Test-set limitation

The current sealed test split contains pTHat strata 0, 2, and 3 only.

A future wave must predeclare sealed test shards for strata:

- 1: 75-100 GeV
- 4: 300-500 GeV
- 5: 500-700 GeV
- 6: 700-1000 GeV
- 7: 1000-inf GeV

No current test event may be reassigned.

## Decision

Authorize preparation and review of one additional adaptive physical
inclusive-HardQCD wave.

Do not authorize:

- a 5M production campaign;
- final cut/BDT/DNN/SPA-Net inference;
- use of targeted QCD-bbbb as an additive physical component.

Targeted QCD-bbbb and other heavy-flavor samples remain ML enrichment.
