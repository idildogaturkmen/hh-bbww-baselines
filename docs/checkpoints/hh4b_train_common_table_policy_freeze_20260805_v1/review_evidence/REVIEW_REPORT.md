# Train-only common-table canary review

- hard gates: 70 passed;
- coverage gates: 4 passed, 3 failed;
- decision: `targeted_canary_augmentation_required`;
- physical weights remain unauthorized;
- validation and test remained sealed;
- models trained: 0.

The scientific hard gates passed, but the following event-level coverage checks need a small targeted augmentation:

- `signed_transport_negative_weight_coverage`: observed `0`, expected `>=1`.
- `variable_qcd_weight_variation`: observed `1`, expected `>=2`.
- `uniform_member_constant_constant_weight_behavior`: observed `3`, expected `1`.
