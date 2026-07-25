# HH4b exactly-3b control train canary

## Purpose

This checkpoint validates the first bounded real-Delphes reconstruction of
the exactly-three-tag HH4b control category. A deterministic train-only
canary set was reconstructed in both `threeb-control` and `fourb-parity`
modes from each identical ROOT member.

## Result

- Status: `hh4b_3b_control_train_canary_pass`
- Canaries selected and passed: 13
- Background canaries: 10
- Signal reconstruction-QA canaries: 3
- Exactly-3b candidate rows: 10613
- Four-b parity candidate rows: 4146
- Three-b integrity failures: 0
- Four-b integrity failures: 0
- Duplicate event rows: 0
- Three-b/four-b event overlap: 0
- QA plots written: 13
- Temporary files remaining: 0

Every promoted jet matches its candidate-jet record and is the highest-pT
selected untagged jet according to the six authorized ROOT branches. The
three-b and four-b event categories are exactly disjoint within every source
member. Event-level Parquets remain only in the ignored runtime directory and
are not included in this checkpoint.

## Scientific interpretation boundary

All counts and plots are raw, unweighted train-canary reconstruction QA. They
are not physical yields or a background prediction. No cross-section or
luminosity normalization, 3b/4b ratio, transfer factor, closure prediction,
significance, optimized signal region, or signal-based bin optimization was
performed. Signal and background QA plots are separate.

## Safety record

- Validation members considered: 0
- Test members considered: 0
- Weighted events calculated: 0
- Physics yields calculated: 0
- Transfer factors calculated: 0
- Closure predictions calculated: 0
- Significances calculated: 0

## Reproducibility

- Source commit: `1976e57d414f8a62e6bf37d70de9623d0589a022`
- Source-map SHA-256: `e797ab8cc878ea1fbb84c05cd52fb8919c6e67bb56bab07bef46701615a29f60`
- Builder SHA-256: `d81a27538437d90878f98d7f7760e5aa574814b9af6da99810642d76c6c2afa8`
- Configuration SHA-256: `ce21650290eb4c114ef5cbccfef1741d55f814ca3ae7b8e973a18f40327e3491`
- Runner SHA-256: `2b12cf1517a104af64dce3a2b9fbe923116058dce04f20eb8f5b97b4afa55d83`

## Next gate

`freeze_hh4b_3b_4b_common_mass_region_geometry`
