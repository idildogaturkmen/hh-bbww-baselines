# V2 safe event-feature two-BDT HH4b baseline

This output uses v2 reconstructed HH4b candidates, with enlarged QCD samples from `$HH4B_STORE/parquet_v2_nominal_plus_extra`.

It reuses the validated safe two-BDT event-feature training/scanning logic, but loads v2 candidate parquets directly.
No truth-level jet flavor columns, source identifiers, raw indices, selected indices, or event IDs are used as BDT features.

# Safe event-feature two-BDT HH4b baseline

This is the corrected event-feature version. It verifies that merging event-level features does not change the number of candidate rows.

All-candidate SM-normalized signal yield at 450/fb: 322.646760 events.

The previous non-safe event-feature output is invalid because merging on non-unique event IDs duplicated rows.
