# Track B Phase 3 deterministic multifile sampling preflight

This bundle freezes deterministic file selection and the unified event-table
contract for later Track B control studies.

## Sampling modes

- `balanced_control_multifile_v1`: up to four deterministic files per each of
  the 18 accessible groups, with an equal target of 256 preprocessing-valid
  events per selected file.
- `high_score_tail_stability_v1`: up to eight deterministic files per QCD,
  ttbar, Z+jets, and ZZ group, with 512 valid events per selected file.

The selection is a metadata-only plan. No remote file was accessed or
downloaded by this step.

## Model scope

The released Sophon ensemble is ready for a bounded multifile adapter canary.
Native SPA-Net and its ParT/JP-JEPA variants remain deferred to the frozen
Track-A maturity gates. Direct Track-B SPA-Net use remains blocked pending a
validated jet and truth-assignment bridge.

## Weight scope

Track-B physical weights remain blocked. Raw `gen_weight` payloads may only be
preserved structurally until author clarification is frozen.

## Next gate

`40J7F_freeze_phase3_deterministic_sampling_and_event_table_contract`
