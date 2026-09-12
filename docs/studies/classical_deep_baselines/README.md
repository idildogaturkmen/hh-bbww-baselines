# Classical / deep baselines

## Question

How well do simple, well-understood models — a multivariate cut-based selection
and gradient-boosted decision trees (with and without flavor-tag information) —
separate resolved HH → 4b signal from QCD/ttbar background? These are the reference
point every later model (SPA-Net, ParT-augmented SPA-Net, future representation
learning) must beat to justify its added complexity.

## Method

- **Cut-based selection**: a multivariate cut-selection search over kinematic and
  b-tag-count variables, validated with a nested out-of-fold stability protocol
  across escalating bootstrap-replica budgets (200 → 1000 replicas) to confirm the
  selected cut structure is not an artifact of a particular fold or replica draw.
- **BDT-K**: gradient-boosted trees on 52 purely kinematic features (per-jet
  pT/eta/phi/mass over up to 10 jets, jet count, HT).
- **BDT-KF**: the same kinematic features plus continuous per-jet flavor-tag
  probabilities (probB/probC/probL), 82 features total.
- All three use the same shared, physically-normalized event table (see
  [`../physical_normalization/`](../physical_normalization/README.md)) and grouped
  cross-validation to avoid event-level leakage across folds.

## Dataset

The train-only common event table shared by every classical and deep baseline in
this project (train/validation split only; no sealed test set was opened for this
study).

## Main result

Adding flavor-tag information is the single largest lever among classical features:
in the physically-normalized four-model comparison (see
[`../physical_normalization/`](../physical_normalization/README.md)), BDT-KF reaches
a background-rejection significance (ZA, cross-checked) of 0.364–0.385 across the
10–20% SM-signal-efficiency working points tested, versus 0.039–0.054 for BDT-K
alone — roughly an order of magnitude better. The nested cut-selection search
converges to a stable cut structure across all 1000 bootstrap replicas tested.

## Key figures/tables

- `artifacts/hh4b_cut_baseline/publication_train_only/figures/` and
  `figure_data/*.tsv` — the cut-baseline's full publication figure set (nested
  fold winners, cut-family frequency, stability, significance).
- `docs/paper/jhep_hh4b_ml/track_b_physical_normalization/MAIN_MODEL_COMPARISON.md` —
  the BDT-K vs. BDT-KF vs. SPA-Net comparison table (see the physical-normalization
  study for the SPA-Net rows).

## Code/config pointers

- `scripts/analysis/train_hh4b_bdt_v1_grouped_cv.py`,
  `train_hh4b_bdt_v2_cms_inspired_grouped_cv.py`
- `scripts/analysis/scan_hh4b_cut_baseline.py`, `scan_hh4b_cut_grid.py`,
  `scan_hh4b_exact_rhh.py`
- `configs/baselines/hh4b_bdt_v1_grouped_cv.yaml`,
  `hh4b_bdt_v2_cms_inspired_grouped_cv.yaml`,
  `hh4b_optimized_cut_scan_{coarse,fine,exact_rhh}_v1.json`
- `tests/test_hh4b_bdt_v1_common.py`, `test_hh4b_bdt_v2_common.py`,
  `test_hh4b_expanded_cut_baseline.py`

## Frozen artifact

- `artifacts/hh4b_cut_baseline/` — the current, most complete bundle
  (`publication_train_only/`, `all1000_stability/`, `train_performance/`).
- `results/hh4b_cut_baseline_20260724_v1/` — the earliest frozen cut-baseline
  bundle (kept at its original path).

## Provenance

- `HH4B_CUT_BASELINE_FINAL_STATUS.{md,json}` (repository root).
- `docs/checkpoints/hh4b_bdt_*_20260725_v1/`, `hh4b_bdt_v2_*_20260726_v1/`,
  `hh4b_bdt_validation_*_20260727_v1/` — BDT protocol checkpoints.
- `docs/checkpoints/hh4b_cut_*_20260725_v1/`,
  `hh4b_cut_baseline_master_train_only_freeze_20260810_v1` (and `_v2`, `_v3`),
  `hh4b_cut_baseline_validation_*_20260810_v1`/`_20260811_v1..v3` — cut-baseline
  freeze and validation-campaign checkpoints.
- `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_*` (≈30
  checkpoint directories, 2026-08-06 through 2026-08-10) — the nested stability
  escalation campaign (initial200 → all1000 → escalation800).
