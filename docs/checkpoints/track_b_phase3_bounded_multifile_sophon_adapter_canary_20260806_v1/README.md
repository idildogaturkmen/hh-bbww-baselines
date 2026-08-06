# Track B Phase 3 bounded multifile Sophon adapter canary

This checkpoint freezes the successful memory-bounded repair of the first
Phase-3 Sophon adapter execution canary and the subsequent probability-sum
tolerance adjudication.

## Execution result

The original `40J7I` attempt ended with wrapper return code 137. The repaired
`40J7I-R1` retained the exact four-file and 256-event scientific scope while
running each ONNX model in a separate short-lived subprocess.

One deterministic file was processed for each of QCD, ordinary ttbar, Z+jets,
and ZZ. Each lane produced exactly 64 valid events. All 256 event identifiers
were reproducible and unique, all file identities were verified, no
preprocessing failures occurred, and all temporary ROOT files were removed.

## Probability-sum adjudication

The initial freeze review used an absolute tolerance of `1e-8` on three
aggregate probabilities serialized from float32 ONNX outputs. A read-only
diagnosis measured a global maximum absolute deviation from one of
`2.0622746889826e-7`, with no nonfinite values, no probability-bound
violations, and no row exceeding `1e-6`. This checkpoint therefore records an
absolute serialized-aggregate sum tolerance of `1e-6`. The frozen Sophon score
contract remains the authoritative matrix validator.

## Scientific interpretation

The recorded lane means and threshold fractions are unweighted released-model
response diagnostics. They are not efficiencies, AUC values, yields,
significance, sensitivity, calibrated masses, or mass resolutions.

Track-B physical weighting remains blocked. No SPA-Net, ParT, or JP-JEPA
model was run or trained.

## Execution boundary

This checkpoint validates the bounded four-file adapter canary only. It does
not authorize the 44-file balanced plan, 46-file tail plan, or combined
90-selection-row execution.

## Next gate

`40J7K_review_and_commit_bounded_multifile_sophon_adapter_canary`
