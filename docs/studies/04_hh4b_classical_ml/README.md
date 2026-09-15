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
- **Two later, methodologically distinct classical-ML studies remain
  unmerged** (`bdt-apples-to-apples-v1`, `cms-resolved-sensitivity-gap-v1`);
  their headline content, incorporated additively this pass:
  - **`bdt-apples-to-apples-v1`** (8 commits, 2026-08-11, tip `753e1df3`) —
    a Harvey-facing "apples-to-apples" nested-outer-OOF rejection audit of
    the global mass-aware/mass-plane-blind BDTs against the CMS HIG-24-010
    resolved SR4b guide points. At the CMS-comparable working point
    (ε_S≈0.4), this project's BDT achieves background rejection ≈13
    (mass-plane-blind) / ≈13.2 (mass-aware), versus CMS HIG-24-010's guide
    value of ≈100 at a similar signal efficiency — a real, order-of-
    magnitude gap. Diagnosis: `CLASSIFIER_AND_QCD_LIMITING` — the current
    feature representation lacks continuous b-tag scores, a 5th b-tag-
    ranked jet, and alternative pairing kinematics, **and** QCD Monte
    Carlo support thins sharply at high score (Neff drops from ~1.2k at
    ε_S=0.6 to ~32-14 at ε_S=0.1), so the gap cannot be attributed to the
    classifier alone. Explicitly not treated as apples-to-apples with CMS
    in scope (different preselection), and no new production was
    authorized to close it. Full artifact tree:
    `artifacts/hh4b_bdt_apples_to_apples/` on that branch — not copied
    here (large figure/table tree); see
    [`docs/history/BRANCH_GUIDE.md`](../../history/BRANCH_GUIDE.md).
  - **`cms-resolved-sensitivity-gap-v1`** (5 commits, 2026-08-11, tip
    `dedf3bbe`) — extends the R_HH<34 cut baseline into an actual expected-
    limit number: pooled μ95=74.015 (cross-validated against 1.96/Z_A=74.010),
    versus CMS HIG-24-010's expected μ95=5.9 (real collision data, full
    detector reconstruction, data-driven multibin background model — **not
    a like-for-like comparison**, but the scale of the gap is real). Keeping
    the exact-3-tag/≥4-tag categories separate improves the pooled number to
    μ95≈44.3, confirming category structure carries real information. A
    validated 2b→3b QCD-transfer closure (yield ratio 1.005, held-fold
    ratios 0.82–1.15) exists, but no ≥4-tag transfer validates (predicted/
    target ratios 0.29–4.4, inconsistent across folds) — this, not a lack
    of raw lower-tag events (1.04M available), is diagnosed as the actual
    bottleneck to closing the sensitivity gap. No new QCD production was
    authorized on this branch. See
    [`docs/history/BRANCH_GUIDE.md`](../../history/BRANCH_GUIDE.md); a
    related SPA-Net-side tail-reliability result is in
    [study 06](../06_scaling_and_tail_reliability/README.md).

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
`bdt-apples-to-apples-v1` (tip `753e1df3`) and
`cms-resolved-sensitivity-gap-v1` (tip `dedf3bbe`) — headline findings
incorporated above; full artifact trees remain on those branches only —
see [`docs/history/BRANCH_GUIDE.md`](../../history/BRANCH_GUIDE.md).
