# Track B Phase 3 inventory-constrained multifile replication contract

This checkpoint freezes the metadata-only replication design produced by
`40J7M-R1`.

## Schema adjudication

The source contract correctly encoded the asymmetric `2/2/2/1` allocation,
but omitted a redundant `selection.inventory_constraint` object. The
constraint is instead established authoritatively by the candidate-audit,
selected-file, cumulative-lane, and design-summary artifacts. This checkpoint
derives and records that constraint without changing the scientific or
execution scope.

## Replication scope

The design retains the same sample group used by the published canary in each
lane and excludes all published canary paths.

The frozen allocation is:

- QCD: two new files
- ordinary ttbar: two new files
- Z+jets: two new files
- ZZ: one new file

The ZZ allocation is smaller because `ZZ_ntuple` contains only one eligible
non-canary file in the frozen selection. The design does not borrow a file
from another sample group, preserving the within-group stability question.

The proposed execution contains seven new files, targets 128 valid events per
file, and would produce 896 new unified event rows.

## Resource controls

Downloads must be sequential. File size and Adler-32 identity must validate
before ROOT I/O. Each ONNX model must run in a separate short-lived
single-threaded subprocess with memory arenas disabled. Temporary ROOT files
must be deleted after each file.

## Boundaries

This checkpoint freezes design metadata only. It does not authorize execution,
the 44-file balanced study, the 46-file tail study, or the full 90-selection
plan. Track-B physical weighting remains blocked. No SPA-Net, ParT, or
JP-JEPA work is authorized here.

## Next gate

`40J7O_review_and_commit_inventory_constrained_replication_contract`
