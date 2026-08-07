# Train-only nested-OOF multivariate-cut aggregation

This checkpoint freezes the first train-only aggregation after the complete
270-job nested-OOF production audit.

For each of five outer folds and each retained b-tag category, the winning
family was selected from the 27 predeclared candidates using pooled true
inner-OOF metrics only. Outer-fold physical results were not used for family
selection.

The exact3tag modal family is radial mass plus HT candidate jets, selected in
2/5 outer folds. The ge4tag modal family is radial mass plus mHH and absolute
Higgs-pair delta-eta, also selected in 2/5 outer folds. All five ge4tag
outer-fold winners use a radial mass family, but their optional kinematic cuts
vary.

Both categories trigger the predeclared selection-stability escalation because
their modal-family frequency is 0.40 and optional-variable inclusion is
ambiguous. Therefore these modal families and their fold-specific refit
thresholds are not final deployable cuts.

Validation and test remain sealed. No final deployable thresholds are selected
in this checkpoint.
