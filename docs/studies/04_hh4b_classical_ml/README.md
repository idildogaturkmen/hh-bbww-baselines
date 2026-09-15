# Study 04 — HH→4b classical ML

## Question

On the project's own physically-normalized HH→4b production (see
[study 03](../03_channel_pivot_and_hh4b_simulation/README.md)), how far do
progressively more expressive classical models — a rectangular cut, a BDT,
a dense DNN, an LBN-style network — get, and how stable are their
selections under resampling?

## Dataset / samples

The physically-normalized HH→4b Delphes production (study 03), evaluated
under a strict **train-only** discipline throughout: every number in this
study is a train-source-group out-of-fold (OOF) estimate or a
nested-cross-validation statistic. No validation or test payload was ever
opened for the classical-ML results reported here (see "Main findings"
below for the one exception, which failed before opening any events).

## Method

A four-stage ladder, in chronological order:

1. **Cut-based baseline** — a coarse→fine→exact grid scan over
   $R_{HH}(125,125)$, $p_T$, and $|\eta|$ cuts, frozen at
   `pT_min=30 GeV, |eta|_max=2.5, R_HH<34` (0.221% from the exact optimum,
   rounded for reproducibility).
2. **BDT model-choice campaign** — built a global mass-aware BDT (`v1`) and
   a CMS-inspired categorized alternative (`v2`), compared them train-only
   under a predeclared validation-selection rule, and chose `v1` as the
   final nominal model after `v2`'s apparent category-level gain did not
   survive a source-member bootstrap stability check.
3. **Broad feature extractor / common table build-out ("Track A")** — a
   schema-frozen, HTCondor-scaled unified training table spanning the full
   ~5.2M-event production, replacing several earlier per-campaign tables.
4. **Rigorous cut-baseline re-run with a 1000-replica selection-stability
   study** — re-derived the cut baseline on the new common table with a
   nested 5-fold outer-OOF methodology; discovered the nominal multivariate
   cut family was selected in only 2/5 outer folds (0.40 frequency, below
   the predeclared stability threshold), triggering a predeclared escalation
   from a 200-replica bootstrap pilot to the full 1000-replica budget before
   accepting any result as stable. A subsequent one-time blind validation
   attempt (Condor cluster `3795859`, 116 jobs) failed at a pre-execution
   runtime probe (`ModuleNotFoundError: awkward`) before any validation
   event was opened — recorded as an infrastructure non-result, not
   resubmitted.
5. **Dense-DNN and LBN comparisons** on the qcdplus (enlarged QCD HT-slice)
   candidate dataset, following the same evaluation contract as the bbWW-era
   comparison in [study 01](../01_hh_bbww/README.md).

## Main findings

- **Model hierarchy** (best-region S/√B, qcdplus dataset): mass-aware
  BDT-v3 (0.199) > topology-only BDT (0.142, robustness control) >
  mass-aware DNN (0.154) > topology-only DNN (0.105) > best LBN variant
  (0.123). The mass-aware BDT measurably pulls background candidates toward
  lower $R_{HH}$-like mass regions relative to the topology-only control.
- **Final nominal BDT**: `global_v1_mass_aware`, train-only weighted OOF
  AUC = 0.7737 (vs. 0.7537/0.8053 by category for the non-selected `v2`
  alternative).
- **Cut-baseline selection is genuinely unstable** under source-group
  resampling (modal frequency well below a stable-majority threshold even
  at the full 1000-replica budget) — this instability is reported as a
  finding in its own right, not smoothed into a single confident number.
- **Validation was never achieved for the cut baseline.** Both the July and
  August cut-baseline packages are train-only; the one accepted validation
  submission failed at the infrastructure level before opening any event.
  This is a **scientific non-result, not a negative result** — see
  `docs/provenance/REPOSITORY_CONTENT_MAP.md`, section D.
- Two independently-frozen "cut baseline" result bundles exist at different
  maturity levels (`results/hh4b_cut_baseline_20260724_v1`, early/narrower;
  `artifacts/hh4b_cut_baseline/`, later/physically-normalized) — both are
  kept, see the content map for how they differ.

## Figures

- [`figures/cut_baseline_vs_historical_rhh34_comparison.pdf`](figures/cut_baseline_vs_historical_rhh34_comparison.pdf) — optimized multivariate cut vs. the historical simple R_HH<34 cut.
- [`figures/cut_baseline_asimov_significance_stat_only.pdf`](figures/cut_baseline_asimov_significance_stat_only.pdf) — the primary stat-only Asimov significance figure.
- [`figures/cut_baseline_all1000_structure_frequency_matrix.pdf`](figures/cut_baseline_all1000_structure_frequency_matrix.pdf) — the full 1000-replica cut-family selection-frequency landscape (the core selection-instability evidence).
- [`figures/bdt_v1_global_oof_roc.pdf`](figures/bdt_v1_global_oof_roc.pdf) — OOF ROC of the final-chosen global BDT.
- [`figures/cut_baseline_v1_cutflow_efficiency.svg`](figures/cut_baseline_v1_cutflow_efficiency.svg) — cutflow efficiency from the earlier (July) frozen cut-baseline package.
- [`figures/classical_ml_full_model_hierarchy_s_over_sqrtB.png`](figures/classical_ml_full_model_hierarchy_s_over_sqrtB.png) — the capstone figure ranking cut/BDT/DNN/LBN by best-region S/√B.

## Reproducible code

- `scripts/analysis/fit_hh4b_bdt_candidates_and_select_validation.py`,
  `freeze_hh4b_bdt_model_choice.py` — the BDT model-choice campaign.
- `scripts/analysis/build_hh4b_cut_baseline_final_report.py`,
  `build_hh4b_cut_baseline_blocked_final_report.py` — cut-baseline report
  builders (root-level write targets, see
  `docs/provenance/REPOSITORY_CONTENT_MAP.md`, section A).
- `scripts/analysis/audit_hh4b_all1000_selection_stability.py`,
  `aggregate_hh4b_all1000_selection_stability.py` — the selection-stability
  bootstrap.
- `configs/baselines/hh4b_bdt_model_choice_v1.yaml`,
  `hh4b_bdt_v1_grouped_cv.yaml`, `hh4b_bdt_v2_cms_inspired_grouped_cv.yaml`,
  `hh4b_optimized_cut_scan_*.json` — model/protocol configs.
- `artifacts/hh4b_cut_baseline/` — the frozen, citable classical cut-baseline
  bundle (tables, publication figures, the blocked-validation report).

## Relationship to the final HH→4b study

This is the **classical-ML rung of the model-comparison ladder** that
motivated moving to a symmetry-aware architecture
([study 05](../05_spanet_reconstruction/README.md)): every classical model
here plateaus below what SPA-Net later achieves, and the cut baseline's
validation never completed at all. See
`docs/results/model_landscape.md` for how these numbers sit alongside the
SPA-Net/ParT/ZERO20 results — **on a different dataset/evaluation contract**
(Run-2 physical-yield projection here vs. raw matched-cohort AUC there); they
are not directly comparable point-for-point.

## Provenance

`docs/checkpoints/hh4b_cut_*`, `hh4b_bdt_*`, `hh4b_broad_*`,
`hh4b_common_table_*`, `hh4b_train_common_table_*`, `hh4b_seven_class_*`
(classical-ML era, 2026-07-24 → 08-11); `docs/checkpoints/hh4b_train_multivariate_cut_*`,
`hh4b_train_cut_baseline_*`, `hh4b_train_fold_*`, `hh4b_train_run2_*`,
`hh4b_train_nested_oof_*` (38 directories, selection-stability campaign,
2026-08-05 → 08-11) — full campaign breakdown in
`docs/provenance/REPOSITORY_CONTENT_MAP.md`, section B.
`HH4B_CUT_BASELINE_FINAL_STATUS.{md,json}` (root, not moved — live script
consumer). `results/hh4b_cut_baseline_20260724_v1/`,
`artifacts/hh4b_cut_baseline/`. Two later, methodologically distinct
classical-ML/tail-reliability studies exist **unmerged** on
`bdt-apples-to-apples-v1` and `cms-resolved-sensitivity-gap-v1` — see
[`docs/history/BRANCH_GUIDE.md`](../../history/BRANCH_GUIDE.md).
