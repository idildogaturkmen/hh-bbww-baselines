# Tail / numerical reliability

## Question

Are the classifier's rare, high-significance tail events and low-probability score
regions numerically trustworthy — stable under floating-point precision choices,
and not an artifact of how a single extra jet is handled in reconstruction —
rather than an artifact of *how* the evaluation itself was computed?

## Method

- Cross-check event-level classifier scores computed in FP32 against a
  higher-precision recomputation, specifically for the governing 10M model (see
  [`../training_scale_study/`](../training_scale_study/README.md)).
- Characterize the statistical and kinematic properties of the extreme tail of the
  score distribution (generation of tail-region summary statistics on two
  successive passes).
- A counterfactual check that removes/alters the fifth reconstructed jet in tail
  events, to test whether tail behavior depends on it.

## Dataset

The matched 400k evaluation cohort used throughout the SPA-Net studies, restricted
to its extreme-tail (highest-score) event subset for the kinematics and
counterfactual checks.

## Main result

The tail-region classifier scores were confirmed numerically stable under the
FP32-vs-higher-precision cross-check on the governing 10M model, closing the open
question of whether the tail was a floating-point artifact rather than a real
feature of the model's score distribution. The fifth-jet counterfactual check was
run to completion; see the exact event tables under Provenance for its outputs
(not independently re-summarized here — this page tracks the question and its
resolution, not the full statistical detail).

## Key figures/tables

No standalone published figures for this study; its evidence is tabular
(event-level parquet/TSV tables) rather than plotted. See Provenance for the exact
tables.

## Code/config pointers

No standalone, reusable script for this study exists outside the frozen checkpoint
pipeline recorded under Provenance below.

## Frozen artifact

None at the `artifacts/` level yet — this study's evidence lives entirely inside
the dated checkpoint directories listed under Provenance.

## Provenance

- `docs/checkpoints/track_b_harvey_fp32_score_precision_20260906_v2_governing10m/SPA10M_EVENT_LOGITS_400K.parquet` —
  the FP32-vs-higher-precision score cross-check on the governing 10M model.
- `docs/checkpoints/track_b_harvey_tail_kinematics_20260907_v2/HARVEY_MASTER_EVENT_TABLE.parquet`,
  `work/signal_tail.parquet` — tail-region kinematics.
- `docs/checkpoints/track_b_harvey_tail_statistics_generation_20260906_v1/`,
  `_20260907_v2/` — the two successive tail-statistics generation passes.
- `docs/checkpoints/track_b_harvey_final_question_closure_20260907_v1/` — the
  fifth-jet counterfactual check. **Note:** two large event-table files in this
  checkpoint (`FIFTH_JET_COUNTERFACTUAL_EVENTS.parquet`,
  `FIFTH_JET_COUNTERFACTUAL_EVENTS_ALL400K.parquet`) exist on disk locally but are
  excluded from Git by a blanket `*.parquet` ignore rule and are **not** committed —
  confirm they are present locally (or regenerate them) before relying on this
  checkpoint as a complete record.

(The internal codename attached to these checkpoint directory names is not used in
this study's narrative; it is retained above only as part of the exact,
reproducibility-relevant path.)
