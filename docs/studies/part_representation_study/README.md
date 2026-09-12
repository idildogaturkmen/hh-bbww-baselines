# ParT representation study (frozen active20 augmentation)

## Question

Does appending a small number of frozen, pretrained Particle Transformer (ParT)
embedding dimensions to native SPA-Net's input improve classification or
reconstruction — or does it introduce unintended harm?

## Method

Append 20 TRAIN-only, variance-selected, **frozen** ParT embedding dimensions to
the same 2M native-SPA-Net input used in
[`../spanet_reconstruction_classification/`](../spanet_reconstruction_classification/README.md)
(identical architecture family; only the input width changes), evaluate under the
same permutation-safe matched-comparison protocol used for the training-scale
study, then run a read-only root-cause audit of the resulting performance change.

## Dataset

The same matched 2M training cohort / 400k evaluation cohort as the SPA-Net and
training-scale studies, plus a frozen ParT embedding production for that cohort.

## Main result

**Harmful, not neutral or beneficial.** All-background AUC fell from 0.969281
(native) to **0.944222** (ΔAUC = −0.0251, paired 95% CI [−0.0255, −0.0246]), and
exact-event HH-reconstruction collapsed from 0.866588 to **0.496015**
(Δ = −0.3706, 95% CI [−0.3739, −0.3672]).

A read-only root-cause audit ruled out native-feature corruption, jet-slot
misalignment, preprocessing implementation errors, unintended
hyperparameter/architecture drift, and checkpoint corruption. The diagnosis
narrowed the likely causes to two compounding effects: (1) changed optimization
dynamics from widening SPA-Net's input embedding, and (2) a variance-only ParT
feature-selection rule that discarded several more discriminative embedding
dimensions than it kept.

This is a matched development/validation result, not a final blind-test result.

## Key figures/tables

- `artifacts/hh4b_spanet_part_20260911/plots/auc_comparison_bar.svg`,
  `delta_auc_forest_plot.svg`, `jet_multiplicity_stratified_auc.svg`
- `artifacts/hh4b_spanet_part_20260911/tables/auc_summary.tsv`,
  `mcnemar_reconstruction.tsv`, `paired_bootstrap_deltas.tsv`,
  `rejection_at_fixed_efficiency.tsv`

## Code/config pointers

No standalone, reusable training/evaluation script for this study exists outside
the frozen checkpoint pipelines recorded under Provenance below.

## Frozen artifact

`artifacts/hh4b_spanet_part_20260911/` — the complete bundle: `README.md`
(headline result in plain language), `metrics/`, `tables/`, `plots/`, `training/`,
`diagnosis/` (`ROOT_CAUSE_DIAGNOSIS.md`, `FOLLOWUP_DESIGNS.md`, `RESULT_FREEZE.md`),
and `SHA256SUMS` recording the external large-file hashes this result depends on.

## Provenance

- `docs/checkpoints/track_f_postproduction_pipeline_20260908_v1/` — the
  preprocess/train/eval/join/postflight pipeline used to produce this comparison.
- `docs/checkpoints/track_f_evaluation_readiness_protocol_20260909_v1/` — the
  permutation-safe evaluation protocol (preregistration, schemas, bootstrap
  utilities) this comparison follows.
- `FULL144_PART_SPANET_SCALE_FEASIBILITY_20260826.md` (repository root) — the
  feasibility audit for scaling this comparison to the full ~144M-event
  population.
- `docs/analysis_contracts/SPANET_PART_RESOURCE_AWARE_COMPARISON.md.save` — the
  original resource-aware ParT/SPA-Net comparison contract this study implements.

(Internal codenames attached to some of these checkpoint directory names are not
used in this study's narrative; they are retained above only as exact,
reproducibility-relevant paths.)
