# Study 01 — HH→bbWW baseline

## Question

Before any channel decision was made, could resolved **HH→bbWW** (one Higgs
to b-quarks, one to WW) support a workable reconstruction and classification
pipeline — b-jet energy correction, H→bb jet-pairing, and (if feasible) a
WW-side target — well enough to build a full analysis on?

## Dataset / samples

The COLLIDE-1M public dataset (see [study 02](../02_collide_dataset_studies/README.md)
for what that dataset actually is), restricted to its HH→bbWW signal files
(2 files) plus ttbar and Z→bb backgrounds. Truth labels were built from
generator status-23 b-quarks and W-daughter quarks matched to reconstructed
AK4 jets (ΔR-matching). This is explicitly a **COLLIDE-1M / simplified**
result, not a full CMS-style reproduction — several source documents say so
directly (e.g. the b-jet regression dataset is flagged "not a full CMS
b-jet regression reproduction").

## Method

A sequential ladder, each stage motivated by the previous one's findings:

1. **Truth-matching and MAX_JETS convention** — established a 99.5%
   truth-pair retention rate at `MAX_JETS=12`; baseline pairing heuristics
   (top-2 b-tag, closest-to-125-mass) got 30–37% accuracy.
2. **WW-side target feasibility** — checked whether a WW-side assignment
   target could be built alongside H→bb; found only ~2.7% of fully-hadronic
   events have all four W-daughter quarks matched (best semileptonic mode
   ~22%), so the decision was made to target **H→bb assignment only**,
   keeping WW decay mode as metadata.
3. **H→bb mass-shift diagnosis** — the truth-matched dijet mass sits well
   below 125 GeV (mean ≈108.7, median ≈103.2 GeV); five sequential
   diagnostics converged on low-pT jet response as the dominant cause (weak
   η and semileptonic-neutrino effects), and showed that recovering the
   correct mass via pT cuts alone discards 85–90% of usable signal.
4. **DNN b-jet response regression** — a per-jet DNN correcting
   `log(GenJet pT / RecoJet pT)`, evaluated with and without a correction-cap
   (capping at 1.25 was the best tested compromise).
5. **Pre-SPA-Net pairing baseline ladder** — top-2 b-tag (36.7%) →
   Pair-DNN (57.3%) → corrected-feature BDT (69.78%) → corrected-feature
   Pair-DNN (69.98%, AUC 0.9394) — the last two statistically
   indistinguishable.
6. **First SPA-Net-for-Hbb application** — a permutation-aware,
   symmetry-preserving assignment model, tested against the ladder above.
7. **Background rejection** — dedicated BDTs against ttbar (AUC 0.7122
   corrected for jet-multiplicity artifacts) and Z→bb (AUC 0.9403).
8. **Model-architecture comparison** — a tabular multiclass DNN vs. four
   Lorentz Boost Network (LBN) variants vs. a corrected-recoMET BDT
   reference (`docs/methods/ml_baselines/multiclass_dnn_comparison.md`).

## Main findings

- The pairing-baseline ladder saturated around AUC≈0.94 / accuracy≈0.70
  before SPA-Net was tried; **SPA-Net underperformed it** at this stage
  (0.546–0.558 full-H-purity vs. ≈0.70 for the BDT/Pair-DNN baseline), with
  the ≥4-jet combinatorial category identified as the main bottleneck — an
  early negative result for SPA-Net that is preserved here, not erased,
  because it directly motivated later input-representation work.
- The best LBN variant (obj+aux, no explicit pair features) reached a
  broad-region significance (Z≈0.00664) comparable to, but not clearly
  exceeding, the corrected-recoMET BDT reference (Z≈0.00676) — normalization
  was still rough at this stage (no systematics, no fit model); the source
  document states this caveat itself.
- Z→bb rejection (AUC 0.9403) was strong despite mass-degenerate m_bb with
  the HH signal, driven mostly by non-mass kinematic features.
- Bootstrap resampling of the fixed test sets (no retraining) put typical
  metric uncertainties at the ±0.01–0.02 level (e.g. Pair-DNN top-1 accuracy
  0.5731±0.0140), giving a sense of how much of the ladder's apparent
  progress is signal vs. noise.

## Figures

- [`figures/bbww_truth_matched_higgs_mbb.png`](figures/bbww_truth_matched_higgs_mbb.png) — the truth-matched H→bb dijet mass peak (the earliest validation figure in the whole repository, 2026-06-05).
- [`figures/bbww_ww_decay_mode_breakdown.png`](figures/bbww_ww_decay_mode_breakdown.png) — W-decay-mode composition behind the WW-side target decision.
- [`figures/bbww_spanet_hbb_precursor_dataset_composition.png`](figures/bbww_spanet_hbb_precursor_dataset_composition.png) — the first SPA-Net-for-Hbb dataset (9,973 usable events).

## Reproducible code

- `Results Summaries/` — the 24 dated write-ups this page summarizes (one per stage above).
- `docs/methods/ml_baselines/multiclass_dnn_comparison.md` — the DNN/LBN/BDT architecture comparison.
- `data/spanet_hbb/`, `options_files/hh_bbww_hbb/`, `event_files/` — SPA-Net-for-Hbb configs and metadata (HDF5 payloads are gitignored).
- `scripts/train_hbb_bjet_regression.py`, `scripts/make_hbb_bjet_regression_dataset.py`, `scripts/check_hbb_matching.py`, `scripts/diagnose_hbb_decay_modes.py`, `scripts/diagnose_hbb_detector_response.py`, `scripts/check_ttbar_jet_activity_robustness.py` — the pipeline scripts behind the ladder above.

## Relationship to the final HH→4b study

This is the project's **original channel**, entirely superseded by the
[channel pivot](../03_channel_pivot_and_hh4b_simulation/README.md) on
2026-07-01. It is preserved in full because: (1) the pairing-baseline
methodology (truth-matching convention, response-regression, ladder
evaluation) carried over directly into the HH→4b classical-ML work; (2) the
first SPA-Net-for-Hbb attempt is the direct scientific ancestor of the
[governing HH→4b SPA-Net study](../05_spanet_reconstruction/README.md),
including its early failure mode; and (3) the pivot decision itself cannot
be understood without knowing what bbWW's reconstruction actually looked
like. **None of these numbers are HH→4b results** — do not cite them
against later HH→4b-channel numbers, which use a different sample and
selection.

## Provenance

Source write-ups: `Results Summaries/RESULTS_HHBBWW_BASELINE.md`,
`RESULTS_HHBBWW_STATUS23_WW_TARGET_DIAGNOSTICS.md`,
`RESULTS_HHBBWW_WW_TARGET_DECISION.md`, `RESULTS_HBB_DETECTOR_RESPONSE.md`,
`RESULTS_HBB_DECAY_MODES.md`, `RESULTS_MBB_DIAGNOSTICS.md`,
`RESULTS_HBB_PT_THRESHOLD_TRADEOFF.md`, `RESULTS_MIN_BJET_PT_CUT_SCAN.md`,
`RESULTS_HBB_BJET_REGRESSION_DATASET.md`, `RESULTS_BJET_RESPONSE_CLIPPING.md`,
`RESULTS_HBB_MASS_SCALING_AND_EVENT_INSPECTION.md`, `RESULTS_HBB_PAIR_DNN.md`,
`RESULTS_HBB_RECONSTRUCTION_COMPARISON.md`, `RESULTS_PRE_SPA_PAIR_BASELINES.md`,
`RESULTS_PRE_SPA_PAIR_BASELINES_WITH_DNN.md`,
`RESULTS_SPANET_HBB_ASSIGNMENT_DATASET.md`, `RESULTS_SPANET_HBB_FIRST_RUN.md`,
`RESULTS_SPANET_HBB_RUNS.md`, `RESULTS_SPANET_HBB_V2.md`,
`RESULTS_TTBAR_BDT_BASELINE.md`, `RESULTS_TTBAR_JET_ACTIVITY_ROBUSTNESS.md`,
`RESULTS_ZBB_REJECTION.md`, `RESULTS_ZBB_MBB_DIAGNOSTICS.md`,
`RESULTS_RESAMPLING.md` (all in `Results Summaries/`, root of repo, dated
2026-05/06; not moved this stage — see
`docs/provenance/REPOSITORY_CONTENT_MAP.md`). `RESULTS_HBB_BJET_REGRESSION.md`
(root, not moved — live script consumer). Root-level outputs archive:
`outputs/plots/signal_validation/`, `outputs/plots/hhbbww_status23_ww_targets/`,
`outputs/plots/spanet_hbb_assignment/`.
