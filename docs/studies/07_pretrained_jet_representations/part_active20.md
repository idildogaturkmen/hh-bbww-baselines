# Frozen ParT "active20" augmentation — a harm result

**Source bundle:** `artifacts/hh4b_spanet_part_20260911/` (frozen
2026-09-11; see that bundle's own `README.md` and
`diagnosis/ROOT_CAUSE_DIAGNOSIS.md` for full statistical and diagnostic
detail — this page summarizes it in public-facing terms).

## Setup

A frozen, pretrained Particle Transformer (ParT) — a transformer trained
externally on generic jet-tagging data, never fine-tuned here — was used
as a per-jet feature extractor. Its 128-dimensional embedding was reduced
to 20 dimensions by keeping only the dimensions with the highest variance
across the training population, then standardized (zero mean, unit
variance, on TRAIN statistics only) and appended to SPA-Net's native
7-feature per-jet input (kinematics + b/c/light-flavor tag probabilities),
widening the input from 7 to 27 features per jet. Everything else —
architecture, optimizer, learning rate, batch size, epoch budget, random
seed, training/validation cohort — was held identical to the native model.

## Result

| Metric | Native SPA-Net (2M) | + ParT active20 (2M) | Δ (paired, 95% CI) |
|---|---:|---:|---|
| All-background AUC | 0.969281 | 0.944222 | −0.0251 [−0.0255, −0.0246] |
| Exact-event HH reconstruction | 0.866588 | 0.496015 | −0.3706 [−0.3739, −0.3672] |

Both differences are large and statistically decisive (paired 95% CIs far
from zero, McNemar p≈0 for reconstruction) — this is not sampling noise.
Appending this specific frozen representation, in this way, **significantly
degraded** the model on both tasks it is asked to do, with reconstruction
affected far more severely (in relative terms) than classification.

## What was ruled out

A read-only audit of the training data, jet-to-embedding join, checkpoint
weights, and training/architecture configuration found:

- Native input features were bit-identical between the native and
  ParT-augmented training data (no corruption).
- Every ParT embedding was joined to the correct event and the correct jet
  slot (checked directly against the extraction pipeline's own records,
  not merely inferred).
- No unintended architecture or hyperparameter difference existed beyond
  the intended 7→27 input-width change (confirmed at both the
  configuration level and the trained-weight-tensor level).
- The trained checkpoint itself was numerically healthy — no NaN/Inf, no
  saturated layers.

Two things were **not** ruled out and instead identified as the likely
mechanism:

1. **The wider input layer changes optimization dynamics.** The
   degradation is present from the very first training epoch (not
   something that develops gradually), is much larger for the
   assignment/reconstruction task than for classification, and co-occurs
   with training-instability episodes (transient loss spikes) that the
   native model does not show.
2. **The dimension-selection rule discarded more-useful dimensions than it
   kept.** Selecting embedding dimensions purely by population variance is
   not the same as selecting by usefulness: several *dropped* raw ParT
   dimensions were measurably more discriminative (for both signal/
   background separation and correct-jet identification) than any of the
   20 *retained* dimensions.

Both are plausible, non-exclusive contributors. Separating them required a
controlled ablation — see [`zero20_width_control.md`](zero20_width_control.md).

**This is a matched development/validation result, not a final blind-test
result** — stated explicitly in the source bundle and repeated here.
