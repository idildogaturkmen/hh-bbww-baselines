# Training-scale study (native SPA-Net, 2M vs. 10M)

## Question

Does training the native SPA-Net model (see
[`../spanet_reconstruction_classification/`](../spanet_reconstruction_classification/README.md))
on 5x more data — 10 million vs. 2 million events — meaningfully improve its
classification or reconstruction performance, or has the native representation
already saturated?

## Method

Train the identical native-SPA-Net architecture at 2M and 10M event scale, holding
architecture, input features, and the evaluation protocol fixed, then compare both
models on the same matched 400k development cohort — a controlled scaling
experiment in which training-set size is the only variable changed.

## Dataset

Two training cohorts (2M and 10M events) drawn from the same underlying HH → 4b
production population (see
[`../hh4b_simulation/`](../hh4b_simulation/README.md)); a shared matched
400k-event evaluation cohort.

## Main result

Performance is effectively flat: all-background AUC **0.969281 (2M) vs. 0.969272
(10M)**. The 10M model shows a small ttbar-specific AUC gain (0.975392 → 0.977502)
but no meaningful gain overall. Five times more training data does not move the
native-feature performance ceiling — this saturation is the direct motivation for
exploring additional input representations (see
[`../part_representation_study/`](../part_representation_study/README.md) and
[`../future_representation_learning/`](../future_representation_learning/README.md)).

## Key figures/tables

- `docs/track_b/development_snapshot_20260821/spanet_10m_scaling/scaling_study_2M_vs_10M/plots/`
  (`epoch_vs_classification_loss.png`, `epoch_vs_val_loss_total_loss.png`,
  `epoch_vs_validation_average_jet_accuracy.png`) — the original scaling-curve
  plots.
- `artifacts/hh4b_spanet_part_20260911/training/training_native_spanet_2M_seed0_result.json`
  and `classification_native_spanet_10M_seed0_result.json` — the matched,
  refined comparison this page's headline numbers are drawn from.

## Code/config pointers

No standalone, reusable training script for this exact comparison exists outside
the frozen pipeline recorded under Provenance below — the 2026-09 postproduction
pipeline that produced the matched comparison was built and run inside a dated
checkpoint directory rather than in `scripts/`. Treat the Provenance paths as the
authoritative, runnable record.

## Frozen artifact

`artifacts/hh4b_spanet_part_20260911/` — both training-result JSON files listed
above; this bundle is shared with the SPA-Net and ParT-representation studies.

## Provenance

- `docs/track_b/development_snapshot_20260821/spanet_10m_scaling/` — the original
  (2026-08-21) 2M-vs-10M scaling study, including its own `SHA256SUMS` and
  `SOURCE_PROVENANCE.tsv`.
- `docs/checkpoints/track_b_harvey_fp32_score_precision_20260906_v1/`,
  `_20260906_v2_governing10m/`, `_20260907_v3_exact_harvey_thresholds/` — the
  floating-point score-precision cross-check performed against the governing 10M
  model (retained here only as an exact path citation; see
  [`../tail_numerical_reliability/`](../tail_numerical_reliability/README.md) for
  what this cross-check was for).
- `docs/checkpoints/track_f_postproduction_pipeline_20260908_v1/` — the ready-to-run
  pipeline used to produce the matched 2026-09-11 comparison, in particular
  `docs/checkpoints/track_f_postproduction_pipeline_20260908_v1/scripts/train/launch_spa2m_part.py`
  and
  `docs/checkpoints/track_f_postproduction_pipeline_20260908_v1/scripts/eval/evaluate_matched_2m.py`.
