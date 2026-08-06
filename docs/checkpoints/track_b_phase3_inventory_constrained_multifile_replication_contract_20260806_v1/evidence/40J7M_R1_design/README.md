# Track B Phase 3 bounded multifile replication design

This metadata-only preflight designs the next execution stage after the
published four-file Sophon adapter canary.

## Question

Does the released Sophon response remain stable across additional files from
the same QCD, ordinary ttbar, Z+jets, and ZZ sample groups used in the canary?

## Proposed scope

Two new deterministic files are selected for QCD, ordinary ttbar, and
Z+jets. The frozen `ZZ_ntuple` inventory contains only one eligible
non-canary file, so one new ZZ file is selected. The proposed execution
would therefore process seven new files and target 128 preprocessing-valid
events per file, for 896 new unified event rows.

The prior canary file remains a reference for each lane. QCD, ttbar, and
Z+jets would have three files per lane after replication; ZZ would have two.

## Boundaries

This output does not authorize execution. It does not authorize the full
44-file balanced plan, 46-file tail plan, or 90-row combined plan. Track-B
physical weighting remains blocked. No SPA-Net, ParT, or JP-JEPA work is
authorized here.

## Next gate

`40J7N_freeze_bounded_multifile_replication_contract`
