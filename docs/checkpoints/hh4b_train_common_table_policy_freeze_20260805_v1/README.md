# Common train-table reconstruction, feature, and weighting policy freeze

This checkpoint freezes the approved train-only common-table policy after:

1. a deterministic representative event-level canary;
2. a 70-gate hard scientific review;
3. a targeted weight-transport augmentation that closed every remaining
   event-level coverage gate.

## Accounting

All 3,799,873 train events remain represented in the accounting view:

- 3,569,873 primary physical events;
- 230,000 auxiliary `qcd_bbbb` classification-only events;
- 1,171,072 events in the resolved model view;
- 2,628,801 non-broad events retained in acceptance and normalization
  accounting.

## Frozen now

- broad resolved-event preselection;
- deterministic generalized four-jet reconstruction;
- historical rich-v2 parity for events with at least four tagged jets;
- exact 34-feature and 30-feature conventional schemas;
- five-fold source-group comparison-weight policy;
- auxiliary-QCD loss-budget treatment;
- source-level physical coefficient join keys and null auxiliary coefficients.

## Not authorized yet

Physical event-weight application remains diagnostic only. Authorization
requires full train-table scale-out followed by source-level and event-level
closure across all 3,799,873 accounting rows and all 1,171,072 resolved rows.

## Cut optimization

The old 34 GeV thresholds are historical references only. The new nominal cut
will be rederived on the full-production train resolved view using nested
five-fold source-group OOF after full table materialization and physical-weight
authorization. Validation and test remain sealed.

The next gate is a pilot of the full train-only common-table scale-out.

## Git-safe TSV serialization

The original review artifact is preserved outside Git at:

    /uscms_data/d3/iturkmen/hh4b_delphes/analysis_preflight/hh4b_train_common_table_canary_review_v2_20260805T203259Z/weight_transport_diagnostics.tsv

The committed `review_evidence/weight_transport_diagnostics.tsv` appends one
nonempty provenance column, `checkpoint_serialization_status`, to every row.
This prevents rows with empty final fields from ending in whitespace while
preserving every original TSV field exactly. The original and committed hashes
and the round-trip proof are recorded in
`review_evidence/weight_transport_diagnostics_git_serialization_receipt.json`.
