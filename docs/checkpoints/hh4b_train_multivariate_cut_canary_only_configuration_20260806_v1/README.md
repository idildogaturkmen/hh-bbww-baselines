# HH4b multivariate-cut canary-only configuration freeze

This checkpoint freezes the resolution of the committed-code canary
search-budget discrepancy.

The archived temporary diagnostic thresholds reproduce the archived metrics
when replayed on the immutable committed canary tables. The effective sample
and weight semantics are therefore closed. The remaining difference is the
optimizer implementation path: all four audited core optimizer functions
differ between the temporary diagnostic implementation and the committed
repository-native implementation.

The committed implementation at
`11f8bc445983bf47bcca33827e138ca87a7e5a39` is authoritative.

A separate canary-only configuration is introduced with a bounded search
budget of 7 quantiles, beam width 16, 4 refinement starts, and 2 coordinate
passes. This configuration is only for deterministic software validation on
the bounded canary tables. It is not a physics configuration and is forbidden
for HTCondor production, family ranking, or cut selection.

The production configuration remains unchanged at 63 quantiles, beam width
128, 16 refinement starts, 8 coordinate passes, target signal efficiency
0.585957, and zero target tolerance.

At this freeze, no production thresholds have been scanned, no family has
been ranked, no cut has been selected, and validation and test remain sealed.

Next: push this commit, rerun the six-sentinel committed canary twice using the
canary-only configuration and immutable fold tables, freeze that passing
evidence, and then prepare a full-production-settings HTCondor pilot.
