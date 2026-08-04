# Track B Phase 2: combined four-control comparison

This directory compares the frozen 64-event ZH, ZZ, QCD, and inclusive
ttbar SophonHH control outputs. No ONNX model was loaded and inference
was not rerun.

## Scope

The comparison is unweighted and diagnostic. It is not a cross-section-
weighted performance measurement, an ROC result, a significance result,
or a calibrated mistag estimate.

## Outputs

- `four_control_summary.tsv`: principal numerical summary.
- `four_control_top1_response.tsv`: grouped top-1 counts and fractions.
- `four_control_probability_quantiles.tsv`: output-group quantiles.
- `four_control_threshold_counts.tsv`: signal and target thresholds.
- `four_control_mean_probabilities.svg`: mean output composition.
- `four_control_top1_response.svg`: grouped top-1 composition.
- `four_control_signal_probability_ecdf.svg`: signal-probability ECDF.
- `four_control_comparison.json`: machine-readable receipt.

## Interpretation

QCD and inclusive ttbar show strong concentration in their dedicated
background output classes. ZH and ZZ show substantial cumulative signal
probability distributed over the 136 signal-grid classes. These 64-event
canaries validate qualitative released-model behavior but do not replace
larger control samples or the pending independent-HH evaluation.
