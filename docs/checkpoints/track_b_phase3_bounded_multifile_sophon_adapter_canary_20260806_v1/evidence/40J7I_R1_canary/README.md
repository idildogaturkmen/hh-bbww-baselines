# Track B Phase 3 bounded multifile Sophon adapter canary — memory-bounded repair

This output repairs the exit-137 execution canary under the frozen Phase-3 sampling
and unified event-table contracts.

## Scope

One deterministic size-bounded file was selected from each of four
representative lanes: QCD, ordinary ttbar, Z+jets, and ZZ. Exactly 64
preprocessing-valid events were required from each file, producing 256 unique
unified event rows.

## Interpretation

This is an unweighted released-model response canary. ONNX models were run one at a time in isolated subprocesses. The recorded fractions
are control-response diagnostics, not efficiencies or physics sensitivity.

Track-B physical weights remain blocked. No SPA-Net, ParT, or JP-JEPA model
was run or trained.

## Next gate

`40J7J_freeze_bounded_multifile_sophon_adapter_canary`
