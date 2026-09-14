# ZERO20 — a width-control ablation, final result

**Source bundle:**
`artifacts/hh4b/pretrained_jet_representations/zero20_20260914/` (frozen
2026-09-14; see that bundle's `README.md` for the full table set,
figures, and machine-readable statistics this page summarizes).

## Why this experiment

The [ParT active20 result](part_active20.md) showed two candidate
mechanisms for the harm it measured: (1) widening SPA-Net's input
embedding layer from 7 to 27 features changes optimization dynamics in a
way that hurts the model, independent of what the extra features contain;
or (2) the specific frozen ParT values selected (and/or their
preprocessing and interaction with training) are the problem. These
predict different outcomes for a specific ablation: keep everything about
the architecture and training contract identical, including the 7→27
input-width change, but replace the 20 ParT-derived columns with a
**constant zero**. If mechanism (1) is the whole story, this "ZERO20"
model should be about as harmed as the ParT model was. If mechanism (2)
dominates, ZERO20 should look like the native model.

## Result

| Model | AUC (all-bg) | AUC (QCD) | AUC (ttbar) | Exact-event reco. | Per-Higgs reco. |
|---|---:|---:|---:|---:|---:|
| Native SPA-Net (2M) | 0.969281 | 0.968155 | 0.975392 | 0.866588 | 0.895299 |
| SPA-Net + ParT active20 (2M) | 0.944222 | 0.941220 | 0.960516 | 0.496015 | 0.513756 |
| **SPA-Net + ZERO20 (2M)** | **0.969116** | 0.967868 | 0.975886 | **0.866201** | 0.895051 |

ZERO20 lands almost exactly on top of the native model, on every metric —
and nowhere near the ParT-active20 result, despite sharing ParT-active20's
architecture, input width, and training contract exactly. Paired
differences confirm this is not a coincidence of the point estimates:

| Comparison | ΔAUC (all-bg), 95% CI | Δexact-event reconstruction, 95% CI | McNemar p |
|---|---|---|---:|
| ParT20 − Native | −0.0251 [−0.0255, −0.0246] | −0.3706 [−0.3739, −0.3672] | ≈0 |
| **ZERO20 − Native** | **−0.00017 [−0.00029, −0.00004]** | **−0.00039 [−0.00153, 0.00075]** | **0.515** |
| ZERO20 − ParT20 | +0.0249 [0.0245, 0.0253] | +0.3702 [0.3669, 0.3735] | ≈0 |

## Statistical significance is not the same question as practical relevance

The ZERO20-vs-native AUC difference has a 95% confidence interval that
technically **excludes zero** — with 400,000 paired validation events,
even a very small, consistent difference is statistically resolvable.
Taken alone, "the CI excludes zero" sounds like a positive finding. It
is not, here: this project pre-registered a **practical-effect floor of
0.005 AUC** for deciding whether an effect is large enough to act on — set
before this ablation existed, at roughly 2.5× the largest previously
measured "pure noise" scaling effect (native SPA-Net moved by only ~0.002
AUC when trained on 5× more data). The ZERO20-vs-native difference
(0.00017) is **about 30 times smaller** than that floor. On
reconstruction, the difference is not even statistically resolvable
(McNemar p = 0.52, consistent with pure chance). By the standard this
project set for itself in advance, **ZERO20 is native-like**, full stop —
the statistically-nonzero AUC difference is a large-cohort-size effect,
not a physically meaningful one.

## Causal-branch classification — mechanically derived, not assumed

Applying the pre-registered decision rule (in
`ZERO20_DIAGNOSTIC_INTERPRETATION_CONTRACT.md`, written and frozen before
any ZERO20 result existed) to these final numbers gives, on both
classification and reconstruction independently:

# Branch B

**ZERO20 recovers to native-like performance while ParT active20 remains
harmed. This falsifies the pure-width explanation.** Widening the input
embedding from 7 to 27 channels is not, by itself, sufficient to
reproduce the harm the ParT integration showed. The harm is tied to the
actual selected ParT values and/or how they interact with this
integration's preprocessing and optimization — not to input width alone.

**This does not mean pretrained jet representations are inherently unable
to help.** It means *this specific* integration — this dimension-selection
rule, this standardization, this training recipe — hurts, and the width of
the input layer is not why. See [`next_experiments.md`](next_experiments.md)
for what would distinguish "the selected ParT dimensions themselves carry
misleading information" from "the preprocessing/optimization interaction
is the problem, independent of which dimensions are used."

## Seeing it directly: ROC and background rejection

Tables and paired-bootstrap numbers are precise, but the four-model ROC
comparison on the exact common 400,000-event cohort
(`.../zero20_20260914/plots/roc_all_background_four_models.svg`) shows the
Branch B pattern at a glance: Native 2M, Native 10M, and ZERO20 trace
essentially one curve, while ParT active20 sits visibly below all three
across the full efficiency range. The figure is a two-panel layout (not
an inset): a full-range panel and a second, plain zoomed panel over the
exact same curve data (εB∈[0,0.08], εS∈[0.72,1.0]), with the three
native-like curves given distinct line styles (solid/dashed/dash-dot), not
color alone, since they sit close enough that color-only encoding was not
legible where one line overlaps another.

The companion background-rejection figure
(`.../plots/background_rejection_four_models.svg`, signal efficiency vs.
1/εB, log scale) makes the same point in the language of a physics
working point rather than an ROC curve, while being explicit about where
the underlying Monte Carlo statistics run out: each curve stops at its own
last real background survivor (no fitted or extrapolated tail beyond that
point), and the segment resting on fewer than 10 raw background events is
drawn dashed with hollow markers. ParT active20 reaches implausibly low
signal efficiencies (down to εS≈0.002) only because its background
contamination is so much worse that a handful of background events still
survive there; the three native-like models run out of background
statistics much earlier (εS≈0.03–0.05), a data-availability fact, not
itself a performance ranking at those specific points.

## Training-history confirmation

The same pattern is visible from the very first training epoch (epoch-0
values shown; see `plots/epoch_assignment_loss.svg` and
`plots/epoch_jet_accuracy.svg` for the full 50-epoch trajectories):

| | assignment loss (h1+h2), epoch 0 | validation jet accuracy, epoch 0 |
|---|---:|---:|
| Native | 1.111 | 0.466 |
| ParT active20 | 2.572 | 0.164 |
| ZERO20 | 0.998 | 0.460 |

ZERO20 tracks native closely from the start; ParT active20 diverges
immediately and never recovers, plateauing roughly 15 epochs earlier than
native at a jet accuracy of ~0.29, far below native's ~0.49 plateau. The
harm is not something that develops gradually during training — it is an
immediate consequence of what the ParT integration adds, not of the wider
input layer it also happens to introduce.

## What this bundle does not do

- It does not authorize or run ALL128, JP-JEPA, additional seeds, or a
  10M-scale ParT production — those remain separate decisions.
- It does not claim pretrained jet embeddings cannot help SPA-Net in
  general — only that this specific 20-dimension, variance-selected,
  z-scored integration does not, and that the reason is not input width.

**This is a matched development/validation result, not a final blind-test
result.**
