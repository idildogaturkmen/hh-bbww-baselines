# Small train-only common-table canary

## Full accounting retained

- train generated events: 3,799,873;
- primary physical train events: 3,569,873;
- auxiliary QCD train events: 230,000;
- full-train resolved model rows: 1,171,072;
- non-broad rows retained in accounting: 2,628,801.

The canary does not discard the non-broad population. It samples both broad
and non-broad events and assigns a zero resolved-selection contribution to
primary events that fail the four-selected-jet requirement.

## Representative coverage

- selected sources: 7;
- folds covered: [0, 1, 2, 3, 4];
- transport classes covered: ['not_applicable_auxiliary_qcd', 'qcd_uniform_constant', 'qcd_variable_frozen_fragment', 'signed_exact_event_sidecar', 'uniform_member_constant'].

## Reconstruction proposal

The generalized reconstruction considers up to eight selected jets, first
maximizes the number of tagged jets in the four-jet candidate, and then uses
the historical mass-pairing objectives. For events with at least four tagged
jets, the canary requires numerical parity with the frozen rich-v2
reconstruction.

## Weighting

All 441 primary source coefficients were joined. All 23 auxiliary-QCD source
coefficients remain null. Event-level Run-2 weights are computed only as a
diagnostic and remain unauthorized.

Fold-local comparison weights are derived from the complete train metadata,
not from the canary sample. The auxiliary-QCD share within the qcd_multijet
family is held fixed across outer folds using the full-train source fraction.
This policy remains a canary proposal until reviewed and frozen.

## Determinism

The canary was executed twice and all 5 canonical artifacts
were byte-identical.

Validation and test remained sealed. No model was trained.
