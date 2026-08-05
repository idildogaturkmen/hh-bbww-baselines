# Targeted weight-transport coverage augmentation

The original canary passed all 70 hard scientific review gates. Its small
320-row sample missed three event-level weight-transport coverage checks:

- `signed_transport_negative_weight_coverage`
- `variable_qcd_weight_variation`
- `uniform_member_constant_constant_weight_behavior`

This augmentation searched only immutable train Parquets and selected
deterministic event rows that exercise the missing behavior.

- scanned train sources: 5;
- targeted rows: 7;
- negative signed event weight found: True;
- positive signed event weight found: True;
- variable hard-QCD event weights found: True;
- constant transport behavior verified: True;
- auxiliary physical weights remain null: True;
- byte-identical deterministic rerun: true.

All original hard gates remain passed and all seven combined coverage gates
now pass. Physical weight application remains unauthorized. Validation and
test remain sealed. No model was trained.
