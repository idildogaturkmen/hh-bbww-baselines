# Common-table join-key and full-train population selection

This checkpoint freezes the passed metadata-only gate that prepares the
train-only common resolved-baseline table.

## Full train accounting

- 3,799,873 train generated-event rows are retained in the analysis accounting.
- 3,569,873 are primary physical train events.
- 230,000 are auxiliary `qcd_bbbb` classification-only events.
- Auxiliary `qcd_bbbb` events never enter physical yields or significance.

## Resolved model view

- 1,171,072 train rows satisfy the frozen minimal resolved-event preselection:
  at least four selected AK4 jets with pT > 30 GeV and |eta| < 2.5.
- These rows form the common resolved-model view for cuts, BDT, DNN, LBN-DNN
  and resolved SPA-Net classification.
- The remaining 2,628,801 rows are not discarded. They remain in generated
  acceptance, cutflow, source-provenance and physical-normalization accounting.
  Candidate-dependent resolved features are undefined for those events.

## Cut optimization

The old 34 GeV thresholds are historical references only. The new nominal
resolved cut has not been selected. It must be rederived on the new full
train-production resolved view using nested five-fold source-group OOF after
the broad physical coefficient join is authorized.

## Normalization metadata

The gate identified the leading frozen coefficient-table candidates:

- ordinary campaign Run-2 allocation registry;
- hard-QCD shard Run-2 coefficient registry;
- ordinary process Run-2 coefficient registry;
- zero-coefficient exclusion registry.

No event payload rows were opened. Validation and test remained sealed. No
model was trained.

The next gate is a small train-only common-table canary.

## Git whitespace normalization

The original metadata-selection log is preserved outside Git at:

    /uscms_data/d3/iturkmen/hh4b_delphes/analysis_preflight/hh4b_common_table_join_key_selection_v1_20260805T200213Z/selection.log

Its original SHA-256 is recorded in `source_SHA256SUMS` and in
`log_sanitization_receipt.json`. The committed `selection.log` differs only by
removing trailing ASCII spaces or tabs from lines 52 through 60 so that the
repository whitespace gate passes. No scientific text, field, count or result
was changed.
