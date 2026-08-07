# Train-only multivariate-cut deployment candidate

This checkpoint freezes the nominal train-only deployment candidate under the
predeclared modal-family plus coordinate-wise-median threshold rule.

## exact3tag

Selected structure:

`radial_mass__category__plus_ht_candidate_jets`

Candidate cuts:

- `r_hh_125_125 < 36.4081401963986`
- `ht_candidate_jets > 176.545806884766 GeV`

This family was selected in 2/5 outer folds. Its two
fold-specific refit threshold sets are retained in the JSON checkpoint.

## ge4tag

Selected structure:

`radial_mass__category__plus_mhh_and_abs_h_delta_eta`

Candidate cuts:

- `r_hh_125_125 < 33.9280891780496`
- `mhh > 164.737080966897 GeV`
- `abs_h_delta_eta < 6.90430216447399`

This family was selected in 2/5 outer folds. Its two
fold-specific refit threshold sets are retained in the JSON checkpoint.

## Interpretation

These are the nominal train-only deployment-candidate cuts defined by the
frozen contract. Both categories triggered the predeclared selection-stability
escalation, so the candidate remains subject to the separate reoptimization
stability diagnostic.

The stability escalation changes uncertainty precision, not the nominal
selection rule. Validation and test remain sealed and have not been applied.
