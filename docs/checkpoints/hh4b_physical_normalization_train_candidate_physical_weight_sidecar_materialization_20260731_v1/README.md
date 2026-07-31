# HH4b train candidate Run-2 physical-weight sidecar

This checkpoint materializes one immutable train-only Parquet sidecar with
30705 rows and the composite identity
`(population_kind, transport_id, event)`.

Each row contains the frozen nominal generator weight, the frozen
cross-section coefficient, the 138000 pb^-1 Run-2 coefficient, and

    run2_candidate_physical_weight =
        run2_yield_coefficient_per_generator_weight
        * generator_nominal_weight

PN-c7i v4 proved every selected train source event maps to exactly one
candidate row, so no candidate-multiplicity division is applied.

The sidecar is stored outside Git at:

    /uscms/home/iturkmen/physical_weight_sidecars/train_candidate_run2_physical_weights_20260731_v1/train_candidate_run2_physical_weights.parquet

Its SHA-256 and complete 441-source row-range registry are frozen in this
checkpoint.

No selected yield, cutflow, S/B, significance, model score or threshold is
calculated here. Validation and final test remain sealed.

Source commit: `4647ffc50cc5436b73430897680613e76c327d94`
