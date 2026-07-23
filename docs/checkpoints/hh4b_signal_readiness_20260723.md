# HH4b canonical signal readiness checkpoint

Date: 2026-07-23

Branch: `delphes-hh4b-production`

## Analysis scope

This repository currently supports a CMS-inspired resolved HH to 4b
analysis using Delphes and reduced candidate tables.

It is not an exact reproduction of the full CMS object-level analysis.

## Frozen common candidate schema

The common classifier and cut-baseline schema contains 72 columns.

Schema SHA-256:

`4008b937032acad6aac87c2fea1df12a110d19a3876d1f8008496e9ca0c85cf4`

Frozen reconstruction policy:

`configs/production/hh4b_rich_v2_reconstruction_policy_v1.yaml`

Rich-v2 candidate builder:

`scripts/delphes/reconstruct_hh4b_candidates_v2.py`

## Canonical ggF HH to 4b signal

Final audit directory:

`outputs/agent_runs/ggf_hh4b_canonical100k_registry_audit_20260723_v2`

Final audit script:

`scripts/production/audit_ggf_hh4b_canonical100k_registry_v2.py`

Validated state:

- canonical members: 100
- generated events: 100,000
- candidate rows: 6,112
- train generated events: 80,000
- validation generated events: 17,000
- sealed-test generated events: 3,000
- train candidate rows: 4,893
- validation candidate rows: 1,045
- sealed-test candidate rows: 174
- live EOS bundle checks: 100
- candidate schema: uniform 75-column signal schema
- common analysis compatibility: all common 72 columns are present
- reconstruction required: no
- physics-yield authorization: no

The three additional ggF columns are provenance fields:

- `analysis_sample`
- `source_root`
- `source_root_index`

They must not be used as classifier inputs.

## Canonical VBF HH to 4b signal

Generated-event registry audit:

`outputs/agent_runs/vbf_hh4b_sm_canonical100k_registry_audit_20260723_v1`

Final materialization audit:

`outputs/agent_runs/vbf_hh4b_canonical100k_materialization_audit_20260723_v1`

Final audit script:

`scripts/production/audit_vbf_hh4b_canonical100k_materialization.py`

Canonical EOS classifier location:

`/store/user/iturkmen/hh4b_delphes/run2_13tev/frozen_v2/classifier/vbf_hh4b_canonical100k_v1`

Validated state:

- canonical members: 10
- generated events: 100,000
- candidate rows: 6,283
- train generated events: 80,000
- validation generated events: 20,000
- train candidate rows: 5,082
- validation candidate rows: 1,201
- test split: absent
- exact common-72 candidate schema: yes
- valid materialization receipts: 10
- valid live EOS candidates: 10
- reconstruction required: no
- new generation required: no
- physics-yield authorization: no

The native VBF train and validation split must be preserved.

A VBF test split must not be invented.

## Combined canonical signal target

The two modes together provide:

- canonical members: 110
- generated events: 200,000
- candidate rows: 12,395
- train generated events: 160,000
- validation generated events: 37,000
- sealed-test generated events: 3,000
- train candidate rows: 9,975
- validation candidate rows: 2,246
- sealed-test candidate rows: 174

ggF and VBF must remain separate production modes in registries,
efficiency tables, plots, and normalization.

They must not be combined using generated sample sizes.

A physical combination requires official mode-specific production
cross sections and branching fractions.

## Canonical background state

Canonical generated-event audit:

`outputs/agent_runs/final_background_unified5m_registry_audit_20260723_v2`

Classifier-source audit:

`outputs/agent_runs/final_background_5m_classifier_source_audit_20260723_v1`

Current state:

- generated events: exactly 5,000,000
- canonical members: 520
- expected candidate rows: 58,867
- local candidate sources: 27
- remote bundle sources: 493
- 493 remote candidate materializations: pending
- 27 local ttbar rich-v2 rebuilds: pending
- physics-yield authorization: no

No additional background generation is authorized.

## Fixed reference baseline

Reference name:

`hh4b_cms_run2_resolved_rich_v2_reference_v1`

Planned immutable selection:

1. Begin from the rich-v2 candidate denominator.
2. Require all four candidate jets to have pT greater than 40 GeV.
3. Require all four candidate jets to have absolute eta below 2.4.
4. Define the analysis region by `r_hh_125_120 < 55`.
5. Define the signal region by `r_hh_125_120 < 30`.
6. Define the control region by `30 <= r_hh_125_120 < 55`.
7. Report `mhh < 450 GeV` and `mhh >= 450 GeV` categories.

The reference selection must be frozen before examining optimization
or threshold-scan results.

## Statistical rules

Before normalization is frozen:

- report generated-event efficiencies
- report candidate-relative efficiencies
- report ggF and VBF separately
- report every background family separately
- report background rejection
- use train and validation only for cut optimization
- keep the sealed ggF test untouched
- do not report final physical yields
- do not report final S over B
- do not report final significance

After normalization is frozen:

- use official mode-specific HH production cross sections
- apply BR(H to bb) squared separately
- retain process-specific background weights
- include finite-simulation uncertainty
- include declared systematic uncertainties
- use a declared significance definition

## Immediate next operations

1. Build and audit the mode-preserving combined signal registry.
2. Freeze the immutable reference baseline configuration.
3. Run a signal-only descriptive cutflow as a pipeline test.
4. Materialize the 493 remote background candidates.
5. Rebuild the 27 local ttbar members into rich-v2.
6. Audit all 520 background members.
7. Run the first complete unweighted cutflow.
8. Perform train-only scans and validation-based cut selection.
9. Freeze normalization before physical yields or significance.
