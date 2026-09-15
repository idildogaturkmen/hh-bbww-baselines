# Study branch: 04 — HH→4b classical ML

This is a reader-facing **study branch**: a curated snapshot of `main`
letting a visitor land directly on one scientific study without first
learning the project's historical Track A/B branch names. It is `main`
(full history preserved, nothing squashed) plus two merged branches and
this file. For the complete SURF repository, including all other studies,
see `main`.

## Scientific question

On the project's own physically-normalized HH→4b production (study 03), how
far do progressively more expressive classical models — cut, BDT, DNN, LBN
— get, how stable are their selections under resampling, and how does the
best of them compare to CMS's own published resolved-HH4b sensitivity?

## Dates

2026-07-24 → 2026-08-11.

## Relevant code

`scripts/analysis/fit_hh4b_bdt_candidates_and_select_validation.py`,
`freeze_hh4b_bdt_model_choice.py`, `build_hh4b_cut_baseline_final_report.py`,
`audit_hh4b_all1000_selection_stability.py`; `configs/baselines/hh4b_bdt_model_choice_v1.yaml`,
`hh4b_bdt_v1_grouped_cv.yaml`, `hh4b_bdt_v2_cms_inspired_grouped_cv.yaml`.
**Newly merged into this branch:** `scripts/analysis/analyze_hh4b_bdt_harvey_roc_train_only.py`,
`finalize_hh4b_bdt_apples_to_apples.py`, `hh4b_bdt_apples_to_apples_common.py`,
`run_hh4b_bdt_apples_to_apples_outer.py`, `scripts/production/prepare_hh4b_bdt_apples_*.py`
(from `bdt-apples-to-apples-v1`); `scripts/analysis/audit_hh4b_cms_gap_qcd_tail.py`,
`audit_hh4b_existing_lower_tag_train.py`, `audit_hh4b_qcd_pthat_support.py`,
`build_hh4b_cms_gap_limit_ladder.py`, `build_hh4b_support_aware_likelihood.py`,
`hh4b_expected_limit.py`, `prototype_hh4b_lower_tag_transfer.py` (from
`cms-resolved-sensitivity-gap-v1`).

## Relevant data / provenance

`artifacts/hh4b_cut_baseline/`, `results/hh4b_cut_baseline_20260724_v1/`;
`docs/checkpoints/hh4b_cut_*`, `hh4b_bdt_*`, `hh4b_broad_*` (2026-07-24 →
08-11, 38+ dirs). **Newly merged:** `artifacts/hh4b_bdt_apples_to_apples/{harvey_roc_train_only_v1,train_only_final_v1}/`
(full tables/figures/models — 105 files) and `docs/provenance/analysis_notes/hh4b_bdt_apples_to_apples_v1_*`
(from `bdt-apples-to-apples-v1`); `artifacts/hh4b_cms_sensitivity_gap_{v1,v2,v3}/`
and `docs/provenance/analysis_notes/{cms_hh4b_resolved_public_benchmark_v1,hh4b_cms_gap_*,hh4b_cms_sensitivity_gap_harvey_brief_*,hh4b_qcd_additional_generation_decision_v1}.*`
(from `cms-resolved-sensitivity-gap-v1`; landed at `docs/provenance/analysis_notes/`
following `main`'s own `docs/analysis/`→`docs/provenance/analysis_notes/` rename).

## Main results

- Model hierarchy (best-region S/√B, qcdplus): mass-aware BDT-v3 (0.199) >
  topology-only BDT (0.142) > mass-aware DNN (0.154) > topology-only DNN
  (0.105) > best LBN (0.123). Final nominal BDT `global_v1_mass_aware`:
  train-only weighted OOF AUC=0.7737.
- **Newly incorporated in full — `bdt-apples-to-apples-v1`**: a Harvey-facing
  nested-outer-OOF rejection audit against CMS HIG-24-010 guide points. At
  ε_S≈0.4: background rejection ≈13 (mass-plane-blind) / ≈13.2 (mass-aware)
  vs. CMS's guide value ≈100 — an order-of-magnitude gap. Diagnosis:
  `CLASSIFIER_AND_QCD_LIMITING` (both the feature representation and QCD
  Monte Carlo tail support are implicated; not attributable to either
  alone).
- **Newly incorporated in full — `cms-resolved-sensitivity-gap-v1`**:
  extends the R_HH<34 cut baseline to an expected-limit number: pooled
  μ95=74.015 (cross-validated against 1.96/Z_A=74.010) vs. CMS HIG-24-010's
  μ95=5.9 (not a like-for-like comparison, but the scale of the gap is
  real). Category-separated pooling improves this to μ95≈44.3. A validated
  2b→3b QCD-transfer closure exists (yield ratio 1.005); no ≥4-tag transfer
  validates (ratios 0.29–4.4, inconsistent across folds) — diagnosed as the
  actual bottleneck, not a lack of raw lower-tag events (1.04M available).

## Superseded / negative results

- Cut-baseline validation was never achieved: the one accepted validation
  submission (Condor cluster 3795859, 116 jobs) failed at a pre-execution
  runtime probe before any event was opened — an infrastructure
  non-result, not resubmitted, and never described as "validated" anywhere
  in this project.
- Cut-baseline selection is genuinely unstable under source-group
  resampling (modal frequency 0.40, below the stable-majority threshold
  even at the full 1000-replica budget) — reported as a finding, not
  smoothed over.
- Neither `bdt-apples-to-apples-v1` nor `cms-resolved-sensitivity-gap-v1`
  closed their respective gaps to CMS; both explicitly declined to
  authorize new production to chase the gap further without first
  resolving the diagnosed bottleneck (feature representation / QCD
  transfer validation, respectively).

## Relationship to main

This branch is `main` plus two merge commits bringing in
`origin/bdt-apples-to-apples-v1` (8 commits, fully unique) and
`origin/cms-resolved-sensitivity-gap-v1` (5 commits, fully unique) — both
previously unmerged anywhere, both scientifically classical-ML/cut-baseline
extensions. Both merges were clean and conflict-free (150 files added, 0
modified, 0 deleted, verified against `main`) — original commit authorship
on the incoming history is unchanged; only the two merge commits are newly
authored. `main` remains the single complete, canonical SURF repository.
See [`docs/studies/04_hh4b_classical_ml/README.md`](docs/studies/04_hh4b_classical_ml/README.md)
on `main` for the full narrative.

## Source historical branches incorporated

- `bdt-apples-to-apples-v1` (tip `753e1df3`, 8 commits, 2026-08-11) —
  merged in full.
- `cms-resolved-sensitivity-gap-v1` (tip `dedf3bbe`, 5 commits, 2026-08-11)
  — merged in full.

Neither source branch was deleted or altered by this merge.
