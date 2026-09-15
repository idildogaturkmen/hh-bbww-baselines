# Study branch: 01 — HH→bbWW baseline

This is a reader-facing **study branch**: a curated snapshot of `main`
letting a visitor land directly on one scientific study without first
learning the project's historical Track A/B branch names. It is not an
independent line of development — it is `main` (full history preserved,
nothing squashed) plus this file. For the complete SURF repository,
including all other studies, see `main`.

## Scientific question

Before any channel decision was made, could resolved **HH→bbWW** (one Higgs
to b-quarks, one to WW) support a workable reconstruction and classification
pipeline — b-jet energy correction, H→bb jet-pairing, and (if feasible) a
WW-side target — well enough to build a full analysis on?

## Dates

2026-05 → 2026-07-01 (superseded by the channel pivot to HH→4b on 2026-07-01;
see [study 03](docs/studies/03_channel_pivot_and_hh4b_simulation/README.md)).

## Relevant code

`scripts/train_hbb_bjet_regression.py`, `make_hbb_bjet_regression_dataset.py`,
`check_hbb_matching.py`, `diagnose_hbb_decay_modes.py`,
`diagnose_hbb_detector_response.py`, `check_ttbar_jet_activity_robustness.py`;
`data/spanet_hbb/`, `options_files/hh_bbww_hbb/`, `event_files/` (SPA-Net-for-Hbb
configs/metadata; HDF5 payloads gitignored, not in this branch either).

## Relevant data / provenance

`Results Summaries/` (24 dated write-ups, root of repo, 2026-05/06) — the
primary source documents for every finding below. `RESULTS_HBB_BJET_REGRESSION.md`
(root — a live script write-target, not moved). `docs/methods/ml_baselines/multiclass_dnn_comparison.md`.
Figure archive: `outputs/plots/signal_validation/`,
`outputs/plots/hhbbww_status23_ww_targets/`, `outputs/plots/spanet_hbb_assignment/`.

## Main results

- Pairing-baseline ladder: top-2 b-tag (36.7% accuracy) → Pair-DNN (57.3%) →
  corrected-feature BDT (69.78%) → corrected-feature Pair-DNN (69.98%,
  AUC 0.9394) — the last two statistically indistinguishable.
- ttbar rejection BDT: AUC 0.7122 (jet-multiplicity-artifact corrected).
  Z→bb rejection BDT: AUC 0.9403.
- Best LBN variant (obj+aux) reached Z≈0.00664, comparable to but not
  clearly exceeding the corrected-recoMET BDT reference (Z≈0.00676) — no
  systematics/fit model yet at this stage, stated as a caveat in the
  source documents themselves.
- Bootstrap resampling of fixed test sets put typical metric uncertainty
  at ±0.01–0.02 (e.g. Pair-DNN top-1 accuracy 0.5731±0.0140).

## Superseded / negative results (preserved, not erased)

- **First SPA-Net-for-Hbb attempt underperformed the classical ladder**
  (0.546–0.558 full-H-purity vs. ≈0.70 for BDT/Pair-DNN), bottlenecked by
  the ≥4-jet combinatorial category. This negative result is the direct
  scientific ancestor of the governing HH→4b SPA-Net study — kept because
  it explains *why* later SPA-Net work took the input-representation
  questions it did.
- Only ~2.7% of fully-hadronic events had all four W-daughter quarks
  truth-matched (best semileptonic mode ~22%) — the WW-side reconstruction
  target was judged infeasible and dropped in favor of H→bb-only assignment.
- Truth-matched dijet mass sits well below 125 GeV (mean≈108.7,
  median≈103.2 GeV); five sequential diagnostics converged on low-pT jet
  response as the dominant cause, and pT-cut-only mass recovery was shown
  to discard 85–90% of usable signal — a genuine dead end, not a solved
  problem.

**None of these numbers are HH→4b results** — a different sample, channel,
and selection. Do not cite them against later HH→4b-channel numbers.

## Relationship to main

This branch **is** `main` at the point this file was added (no divergent
commits, no rebasing, no squashing) — every commit, script, and dataset
reference on `main` is present here unchanged. The curation is additive:
this one file. `main` remains the single complete, canonical SURF
repository; this branch exists purely as a study-scoped entry point. See
[`docs/studies/01_hh_bbww/README.md`](docs/studies/01_hh_bbww/README.md) on
`main` for the full narrative this file summarizes.

## Source historical branches incorporated

None — this study's content was always part of the project's mainline
history (predates the branch-per-topic era) and required no cross-branch
merge.
