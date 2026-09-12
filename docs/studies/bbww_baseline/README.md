# bbWW baseline (HH → bbWW, superseded)

## Question

Can we reconstruct and classify HH → bbWW events — pairing jets to the H → bb
candidate and reconstructing the H → WW side from leptons and missing transverse
energy — well enough to build a full analysis baseline on this channel?

## Method

- b-jet energy regression and a pT-binned mass correction, to improve the H → bb
  invariant-mass resolution.
- A symmetry-aware jet-pairing network (SPA-Net) trained to assign jets to the
  H → bb candidate.
- A lepton+MET pairing DNN, evaluated separately, for the H → WW side.
- A classical gradient-boosted baseline for comparison.
- A dedicated diagnostic characterizing how often the WW-side decay products are
  actually reconstructable at all, before investing further in a WW-side model.

## Dataset

A Delphes-simulated HH → bbWW sample (event definitions and training options
checked into this repository; the underlying generated events are external, not
committed to Git).

## Main result

H → bb jet-pairing reached a full-sample assignment purity of 0.558 (SPA-Net,
2-jet-category purity 1.000, degrading with jet multiplicity). The H → WW side was
the limiting factor: in the fully hadronic WW decay mode, only about 2.7% of events
had all four W-side quarks matched to retained AK4 jets, and even the more
favorable semileptonic modes matched both hadronic W-side quarks in only about 22%
of events. This motivated **not** forcing full WW-side jet assignment, and
ultimately motivated the project's pivot to the cleaner, more tractable resolved
**HH → 4b** channel described in [`../hh4b_simulation/`](../hh4b_simulation/README.md).

## Key figures/tables

- `Results Summaries/RESULTS_HHBBWW_WW_TARGET_DECISION.md` — the WW-side
  matchability study and decision.
- `Results Summaries/RESULTS_SPANET_HBB_V2.md` — the H → bb SPA-Net purity table.
- `Results Summaries/RESULTS_HBB_BJET_REGRESSION.md`,
  `RESULTS_HBB_PAIR_DNN.md`,
  `RESULTS_HBB_RECONSTRUCTION_COMPARISON.md` — supporting b-jet regression and
  pairing-model comparisons.

## Code/config pointers

- `scripts/train_hbb_bjet_regression.py`, `scripts/make_hbb_bjet_regression_dataset.py`
- `scripts/make_spanet_hbb_assignment_dataset.py`, `scripts/convert_hbb_dataset_to_spanet_format.py`
- `scripts/train_pre_spa_pair_baselines.py`, `scripts/train_pre_spa_pair_dnn_baseline.py`, `scripts/train_pair_dnn.py`
- `scripts/train_lbn_fourvector_dnn_v0.py`, `scripts/make_lbn_fourvector_input.py`
- `event_files/hh_bbww_hbb.yaml`, `event_files/hh_bbww_hbb_explicit.yaml`
- `options_files/hh_bbww_hbb/*.json`
- `data/spanet_hbb/*.metadata.csv`

## Frozen artifact

`Results Summaries/` — 24 markdown result write-ups from this phase of the project
(kept at its current path, including its literal directory name, unchanged).

## Provenance

- `notes/bbww_baseline_summary_and_pivot.md` — the full pivot rationale, written at
  the time the project moved from bbWW to resolved HH → 4b (2026-07-01).
- `RESULTS_HBB_BJET_REGRESSION.md` (repository root) — an early frozen result
  write-up from this phase, kept at its original root-level path.
