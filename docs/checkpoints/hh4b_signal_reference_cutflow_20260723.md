# HH4b signal reference-cutflow checkpoint

Date: 2026-07-23

## Reference selection

Baseline:

`hh4b_cms_run2_resolved_rich_v2_reference_v1`

Configuration:

`configs/baselines/hh4b_cms_run2_resolved_rich_v2_reference_v1.yaml`

Cutflow runner:

`scripts/analysis/run_hh4b_signal_reference_cutflow.py`

Audit output:

`outputs/agent_runs/hh4b_signal_reference_cutflow_20260723_v1`

## Scope

This was a signal-only descriptive pipeline validation.

It was not a cut optimization and did not report physical yields,
S over B, or significance.

## Input membership

Processed:

- members: 107
- generated events: 197,000
- candidate rows: 12,221
- splits: train and validation only
- signal modes: ggF and VBF reported separately

Skipped and not accessed:

- sealed ggF test members: 3
- sealed ggF test generated events: 3,000
- sealed ggF test candidate rows: 174

## ggF signal-region efficiency

Train:

- selected events: 1,454
- efficiency relative to generated events: 0.018175
- efficiency relative to candidates: 0.297159

Validation:

- selected events: 313
- efficiency relative to generated events: 0.018412
- efficiency relative to candidates: 0.299522

## VBF signal-region efficiency

Train:

- selected events: 1,488
- efficiency relative to generated events: 0.018600
- efficiency relative to candidates: 0.292798

Validation:

- selected events: 367
- efficiency relative to generated events: 0.018350
- efficiency relative to candidates: 0.305579

## Interpretation

The fixed reference selection has consistent train and validation
efficiencies for both signal modes.

The different low- and high-mHH compositions of ggF and VBF are retained.
The modes must remain separate until mode-specific physical normalization
is frozen.

## Next production gates

1. Inspect and freeze the 520-member background materialization plan.
2. Extract 493 existing candidates from canonical bundles.
3. Rebuild 27 local legacy ttbar candidates using rich-v2.
4. Audit the unified 520-member candidate layer.
5. Run the complete unweighted train/validation reference cutflow.
6. Freeze physical normalization.
7. Evaluate the sealed test only after all choices are fixed.
