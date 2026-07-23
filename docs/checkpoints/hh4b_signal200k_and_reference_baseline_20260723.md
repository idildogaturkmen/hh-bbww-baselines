# HH4b canonical signal-200k and reference-baseline checkpoint

Date: 2026-07-23

## Combined canonical signal registry

Canonical registry:

`metadata/production_plans/hh4b_canonical_signal200k_registry_20260723.tsv`

Canonical summary:

`metadata/production_plans/hh4b_canonical_signal200k_registry_20260723.json`

Registry builder:

`scripts/production/build_hh4b_canonical_signal200k_registry.py`

Validated state:

- canonical members: 110
- generated events: 200,000
- candidate rows: 12,395
- ggF generated events: 100,000
- VBF generated events: 100,000
- ggF candidate rows: 6,112
- VBF candidate rows: 6,283
- train generated events: 160,000
- validation generated events: 37,000
- sealed-test generated events: 3,000
- train candidate rows: 9,975
- validation candidate rows: 2,246
- sealed-test candidate rows: 174
- common analysis schema: 72 columns
- ggF signal schema: 75-column compatible superset
- VBF signal schema: exact common 72 columns
- VBF test split: absent
- ggF test split: sealed
- physics-yield authorization: no

ggF and VBF remain separate production modes.

They must not be combined by generated sample sizes.

## Frozen reference baseline

Configuration:

`configs/baselines/hh4b_cms_run2_resolved_rich_v2_reference_v1.yaml`

Reference selection:

- begin from the rich-v2 candidate denominator
- require all four candidate jets to have pT greater than 40 GeV
- require all four candidate jets to have absolute eta below 2.4
- analysis region: `r_hh_125_120 < 55`
- signal region: `r_hh_125_120 < 30`
- control region: `30 <= r_hh_125_120 < 55`
- low-mHH category: `mhh < 450 GeV`
- high-mHH category: `mhh >= 450 GeV`

This reference selection is immutable.

It must not be modified after inspecting scan results.

## Current authorization

Authorized:

- signal-only descriptive cutflow
- generated-event efficiency reporting
- candidate-relative efficiency reporting
- mode-separated signal efficiency reporting

Not yet authorized:

- full signal-versus-background cutflow
- cut optimization
- physical yields
- S over B
- significance

The complete signal-versus-background baseline remains blocked by:

- 493 remote background candidate materializations
- 27 local ttbar rich-v2 rebuilds
- final unified 520-member background audit
