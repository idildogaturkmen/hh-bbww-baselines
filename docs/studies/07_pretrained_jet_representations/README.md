# Study: pretrained jet representations for SPA-Net

This study asks a single question: does giving SPA-Net's jet-assignment
and classification network access to a **frozen, pretrained per-jet
embedding** — computed once by an external model and never fine-tuned —
improve resolved HH→4b reconstruction and classification beyond the
native 7-feature representation (kinematics + b/c/light-flavor tag
probabilities)?

The answer, for the specific integration tested so far, is **no, and the
follow-up ablation shows why not**. Two documents cover the two frozen
results in this line of work; a third looks forward.

## Contents

- **[`part_active20.md`](part_active20.md)** — the first representation
  tested: 20 frozen dimensions from a pretrained Particle Transformer
  (ParT) embedding, selected by training-set variance and appended to the
  native input. Result: **significant harm** to both classification and
  reconstruction.
- **[`zero20_width_control.md`](zero20_width_control.md)** — the
  controlled follow-up: replace the same 20 input slots with a **constant
  zero**, keeping architecture and training contract otherwise identical.
  Result: **no harm** — indistinguishable from the native model on both
  metrics, at the pre-registered practical scale. This isolates the cause:
  widening the input layer alone does not explain the harm above; the
  specific embedding content (and/or its preprocessing/optimization
  interaction) does.
- **[`next_experiments.md`](next_experiments.md)** — what these two
  results imply about which follow-up experiment is actually informative
  now, updated in light of the ZERO20 outcome (some previously-proposed
  follow-ups are now lower priority than they looked before ZERO20 ran).

## One-paragraph summary, if you read nothing else

Appending 20 frozen ParT embedding dimensions to SPA-Net's native input
significantly degraded both event classification (all-background AUC
−0.025) and HH-reconstruction (exact-event rate −0.371) relative to the
native model, at matched 2M-event training scale. A controlled ablation —
identical architecture and training contract, but with those same 20
input slots held at zero instead of holding ParT values — reproduced
**none** of that harm (ΔAUC −0.00017, ~30× below the pre-registered
0.005-AUC practical-effect floor; reconstruction statistically
indistinguishable, McNemar p=0.52). **This rules out "the wider input
layer alone is the problem"** and points instead at the specific selected
ParT values and/or how they interact with this integration's
preprocessing and optimization — not at pretrained jet representations
being unable to help in general. See `next_experiments.md` for what
distinguishes those two remaining explanations and how to test between
them.

## How these results were produced (brief; full detail in each document)

Both results share one exact training/evaluation contract: SPA-Net, seed
0, batch size 2048, 50 epochs, AdamW (lr 0.00659, weight decay 0.000374,
gradient clip 0.425), on the identical 2,000,000-event training / 400,000-
event matched-validation cohort used throughout this project's SPA-Net
work. The only thing that differs between native, ParT-active20, and
ZERO20 is the content of 20 additional input columns (absent / frozen ParT
values / constant zero). Every comparison below is a **paired** bootstrap
on the same held-out event cohort, so per-event variance cancels between
models — this is what makes a ΔAUC as small as 0.00017 statistically
resolvable at all, which is exactly why this study is careful to separate
statistical from practical significance throughout (see
`zero20_width_control.md`).

The ParT checkpoint used throughout this study is the **official
JetClass-supervised** Particle Transformer — not a self-supervised model.
Its exact provenance is frozen in
`docs/analysis_contracts/SPANET_PART_RESOURCE_AWARE_COMPARISON.md.save`
(kept at that exact, unusual filename because 8 already-frozen provenance
documents cite it by that literal path — see
`docs/provenance/REPOSITORY_CONTENT_MAP.md`). JP-JEPA (mentioned in
`next_experiments.md` and under `docs/checkpoints/track_g_jpjepa_*`) is a
**different, genuinely self-supervised** method explored only as exploratory
compatibility groundwork — do not conflate the two.

## Figures

- [`figures/part_active20_delta_auc_forest.svg`](figures/part_active20_delta_auc_forest.svg) — the ParT active20 harm result as a paired-AUC-delta forest plot.
- [`figures/zero20_delta_auc_forest.svg`](figures/zero20_delta_auc_forest.svg) — the same forest-plot presentation for ZERO20's null result; side by side with the one above, these two make the isolation argument visually.
- [`figures/zero20_roc_all_four_models.svg`](figures/zero20_roc_all_four_models.svg) — the combined ROC across native 2M/10M, ParT active20, and ZERO20 (two-panel, full-range-plus-zoom; see `zero20_width_control.md`).

## Scale feasibility and future work (not a result)

[`full144_scale_feasibility_20260826.md`](full144_scale_feasibility_20260826.md)
is a planning/audit document (2026-08-26, no production launched) assessing
whether the ParT+SPA-Net comparison can scale to 10M or the full ~144M-event
training population. It recommends scaling next to **10M**, not jumping to
144M, and flags that the ParT checkpoint's own domain-mismatch verdict
(`REVIEW_NEEDED`/`DOMAIN_SHIFT_REQUIRES_VALIDATION`) is still open and gates
scientific validity at *any* scale — this is the same checkpoint used in
active20/ZERO20 above. The "All-128" sensitivity run proposed in
`next_experiments.md` (sometimes referred to informally as "full128") is
**not authorized or running** as of this reorganization; per
`docs/results/model_landscape.md`'s rule of only listing completed results,
neither it nor full144 appears in that table.
