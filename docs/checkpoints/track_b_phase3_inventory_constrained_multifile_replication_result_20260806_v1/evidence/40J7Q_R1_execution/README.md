# Track B Phase 3 inventory-constrained Sophon replication

This output executes the published seven-file replication contract using the
memory-bounded released Sophon ensemble adapter.

## Scope

Seven new files are processed: two each for QCD, ordinary ttbar, and Z+jets,
and one for ZZ. Published canary paths are excluded. Exactly 128
preprocessing-valid events are required per file, producing 896 unique new
unified event rows.

## Execution controls

Files are downloaded and processed sequentially. Each ONNX model runs in a
separate short-lived single-threaded subprocess with the CPU memory arena and
memory pattern disabled. File size and Adler-32 identities are verified before
ROOT I/O, and each temporary ROOT file is deleted after its file-level outputs
are complete.

## Interpretation

The file-level and lane-level values are unweighted released-model response
stability diagnostics. They are not efficiencies, calibrated probabilities,
expected yields, AUC, significance, or sensitivity. Track-B physical weights
remain blocked. No SPA-Net, ParT, or JP-JEPA model is run or trained.

## Next gate

`40J7R_review_and_freeze_inventory_constrained_multifile_replication`
