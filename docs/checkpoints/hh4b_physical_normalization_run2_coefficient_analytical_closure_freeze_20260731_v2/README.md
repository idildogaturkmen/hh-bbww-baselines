# HH4b Run-2 physical coefficient analytical closure freeze

This checkpoint freezes the 13 TeV CMS Run-2-equivalent 138 fb^-1
normalization coefficients for the authoritative ordinary and nominal
hard-QCD populations.

## Ordinary normalization

For each of 22 ordinary processes:

    cross_section_coefficient = effective_reference_factor_pb / process_sumw
    run2_yield_coefficient = 138000_pb_inverse * cross_section_coefficient
    event_yield_weight = run2_yield_coefficient * generator_nominal_weight

The 47 frozen campaign denominators close exactly to their process
denominators. The process and class analytical yields close to
138000_pb_inverse times the frozen effective reference cross sections.

## Hard-QCD normalization

For each of 260 nominal production shards:

    cross_section_coefficient =
        (N_shard / N_exclusive_bin) * (sigmaGen_shard_pb / shard_sumw)

    run2_yield_coefficient =
        138000_pb_inverse * cross_section_coefficient

The source coefficients close to all eight frozen exclusive pTHat-bin
cross sections. The three production extensions are pooled within each
exclusive bin and are never treated as independent full-cross-section
samples.

## Exclusions

- 300000 overlapping qcd_bbbb events have nominal coefficient zero.
- The 10000-event diagnostic hard-QCD testfill has nominal coefficient zero.

## Safety state

No candidate Parquet, ROOT event payload or generator-weight sidecar payload
was opened. No candidate-level physical weight was materialized. Inclusive
analytical coefficient closure is complete, but selected candidate yields
remain unauthorized until train event multiplicity and join semantics are
proven.
