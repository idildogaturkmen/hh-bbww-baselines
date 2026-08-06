# Track B Phase 2 empirical mass-grid consistency

## Scope

This checkpoint freezes the complete local empirical consistency program from
Steps 40J5A through 40J5D for the released jet-free HH4b Sophon ensemble.

It establishes:

- direct event-level generator masses from `gen_higgs1_mass` and
  `gen_higgs2_mass`, interpreted exchange-symmetrically;
- complete coverage of all 561 unordered 5 GeV generator-mass pairs and all
  136 released 10 GeV signal-grid cells;
- consistency between the retained public `calculateClsIndex` behavior and the
  released output-index-to-cell mapping;
- a 272-event, 136-cell balanced three-model ONNX canary;
- an independent-file replication *within the training-signal namespace* using
  840 events from five additional files;
- reproducible but imperfect signal-grid localization;
- reproducibly weaker diagonal-cell localization.

## Central replication result

The five-file ensemble replication contains 840 selected events:

- 600 off-diagonal events;
- 240 diagonal events;
- no preprocessing failures;
- no event weights;
- no DCB fits.

The ensemble results are:

- mean signal probability: 0.902901;
- fraction with signal probability above 0.9:
  0.780952;
- exact target-cell top-1 fraction:
  0.105952;
- target in top-3:
  0.272619;
- target in top-5:
  0.382143;
- within Chebyshev distance 1:
  0.465476;
- within Chebyshev distance 2:
  0.630952.

All predeclared replication gates passed.

## Diagonal behavior

Diagonal cells remain materially weaker than off-diagonal cells:

- diagonal top-1: 0.033333;
- off-diagonal top-1: 0.135000;
- diagonal top-5: 0.287500;
- off-diagonal top-5: 0.420000;
- diagonal mean Chebyshev distance:
  3.808333;
- off-diagonal mean Chebyshev distance:
  2.883333.

## Permitted interpretation

> The released three-model ensemble exhibits reproducible empirical
> consistency between its 136-cell signal-grid response and generator
> mass-cell assignments across a balanced canary and a bounded five-file
> variable-mass training-signal replication.

## Boundaries

This checkpoint does **not** establish:

- independent test-set performance;
- the original unavailable Weaver ordered label list;
- calibrated reconstructed Higgs masses;
- unbiased mass resolution;
- physical `gen_weight` semantics;
- cross-section-weighted efficiencies or expected yields;
- significance or sensitivity.

The ntuples contain no event-identity branch, so training/test independence
cannot be established from these files alone.

## Files

- `scientific_summary.json`: compact scientific decision record.
- `checkpoint_receipt.json`: source and repository provenance.
- `remaining_author_clarification_draft.txt`: concise unresolved questions.
- `evidence/`: compact machine-readable source evidence.
- `SHA256SUMS`: hashes of every checkpoint file except itself.

The large temporary ROOT files and binary NPZ score payloads are intentionally
not included. The aggregate event-level TSV preserves the selected-event
diagnostics needed for review.
