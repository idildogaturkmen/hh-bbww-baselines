# Study 06 — training-scale and tail reliability

## Question

Two related reliability questions about the governing native SPA-Net
checkpoint ([study 05](../05_spanet_reconstruction/README.md)): (1) does 5×
more training data (10M vs. 2M events) actually improve it? (2) can its
rare-event (high-signal-efficiency) tail behavior be trusted numerically
(FP32 precision) and statistically (finite QCD Monte Carlo)?

## Dataset / samples

The same fixed, matched 2M-train/400k-validation HH→4b cohort as
[study 05](../05_spanet_reconstruction/README.md), plus a 10M-event training
run on the identical native 7-feature representation, evaluated on the same
400k validation set.

## Method

1. **2026-08-21 development-scale check** — SPA-Net 10M (seed 0, 50 epochs)
   trained and evaluated for the first time against the 400k cohort.
2. **2026-09-11 refined, matched comparison** — the checksummed, governing
   2M-vs-10M comparison (`artifacts/hh4b_spanet_part_20260911/`).
3. **2026-09-14 reconstruction addendum** — Native 10M's exact-event/per-Higgs
   reconstruction, computed directly from the frozen `native10m_eval_400k.npz`
   export using the exact same `match_higgs_pairs()`/`reconstruction_metrics()`
   code as every other reconstruction number in this project (verbatim copy,
   not reimplemented), hard-gated on bit-for-bit reproduction of the
   already-published Native 2M numbers before being trusted.
4. **Harvey FP32 score-precision investigation (2026-09-06/07, v1→v2→v3)** —
   built a full FP32 unique-score census and threshold-crossing lattice on
   the governing 10M checkpoint's 400k event logits, checking whether
   float32 quantization changes which events pass/fail near tight decision
   thresholds.
5. **Harvey tail-statistics and tail-kinematics characterization
   (2026-09-06/07)** — estimated the additional QCD Monte Carlo needed to
   populate the high-efficiency tail adequately (N_eff vs. QCD-multiplier
   curves), and built a 1D/2D/4D kinematic feature-ranking table for the
   tail population, including a fifth-jet assignment analysis.
6. **Fifth-jet causal counterfactual test** — not merely observational: every
   one of 88,854 signal events with ≥5 selected jets was **re-scored with
   jets ranked ≥5 masked out**, directly measuring the causal effect of the
   extra jet(s) on the classification score.
7. **Harvey final question closure (2026-09-07)** — a terminal wrap-up
   mapping each open reliability question to its resolving evidence.

## Main findings

- **Classification has saturated, not improved**, with 5× more training
  data: all-background AUC 0.969281 (2M) vs. 0.969272 (10M) — statistically
  indistinguishable at this scale (2026-08-21 pass: AUC change −0.000009).
- **Reconstruction shows a small point-estimate improvement** at 10M
  (+0.0056 exact-event, +0.0035 per-Higgs) — no paired-bootstrap CI was
  computed for this delta, so it is reported as a raw point estimate, not a
  statistically significant claim.
- **FP32 quantization is kinematically unremarkable.** Score-lattice
  compression exists near tight decision thresholds (as expected from
  float32 precision alone), but events sitting in the compressed bins show
  no special kinematic mode — they sit smoothly on the same trend as
  neighboring populated bins.
- **The fifth-jet effect is real and directional, not negligible.** 0% of
  qualifying events are unaffected by masking jets ranked ≥5; median
  Δu_logit = −0.090; 58.1% of events are net-helped by the extra jet(s),
  41.9% net-hurt — extra jets matter, in both directions, not as pure noise.
- **QCD tail statistics are adequate at moderate tail depth, not deeper.**
  Empirically stable (not merely projected) across the two existing QCD
  production lanes at u>3.5 (p=0.90 consistency test); at u>4.5 there are
  only 12 raw events — explicitly reported as **too few to judge either
  way**, not forced to a conclusion.
- **An earlier, unmerged compressed-tail package** (`delphes-hh4b-production`,
  tip commit `e17fdbf9`, 2026-09-02 — predates and directly foreshadows the
  Harvey tail-statistics work above) introduces the same u=-log10(1-score)
  transform used throughout this study's tail characterization and applies
  it to the governing SPA-Net 10M checkpoint's score≥0.9997/0.99997
  populations: n=100 raw events (Neff=81.0, tier `FINITE_SUPPORT_CAUTION`)
  at ≥0.9997, dropping to n=15 (Neff=14.0, `EXTREMELY_LIMITED`) at
  ≥0.99997 — QCD supplies 61% of raw events but 83.8% of the weighted
  yield at ≥0.9997, rising to 100% of raw events by ≥0.99999. No single
  kinematic mode explains the tail (|Spearman ρ|≤0.21 for every variable
  tested against u). Preserved verbatim at
  [`spanet_10m_compressed_tail_20260902.md`](spanet_10m_compressed_tail_20260902.md);
  this content was not linked from any study before this pass — see
  [`docs/history/BRANCH_GUIDE.md`](../../history/BRANCH_GUIDE.md).
- A related, methodologically distinct classical-baseline sensitivity-gap
  result (`cms-resolved-sensitivity-gap-v1`) is documented in
  [study 04](../04_hh4b_classical_ml/README.md) — it uses the same
  QCD-Neff/tail-support diagnostic style but targets the R_HH<34 cut
  baseline's expected limit rather than SPA-Net.

## Figures

- [`figures/governing_10m_fp32_probability_tail_census.png`](figures/governing_10m_fp32_probability_tail_census.png) — the FP32 unique-score census in the high-efficiency tail for the governing 10M checkpoint.
- [`figures/qcd_tail_neff_vs_mc_multiplier.png`](figures/qcd_tail_neff_vs_mc_multiplier.png) — effective sample size vs. QCD Monte Carlo multiplier, the central plot for deciding how much additional tail MC is needed.
- [`figures/tail_region_1d_feature_ranking.png`](figures/tail_region_1d_feature_ranking.png) — kinematic features ranked by their power to characterize the QCD-tail population.

## Reproducible code

- `artifacts/hh4b/pretrained_jet_representations/zero20_20260914/code/compute_native10m_reconstruction.py` — the hard-gated Native 10M reconstruction cross-check.
- `docs/checkpoints/track_b_harvey_fp32_score_precision_20260906_v1/`, `_v2_governing10m/`, `_v3_exact_harvey_thresholds/` — the FP32 investigation, in chronological revision order.
- `docs/checkpoints/track_b_harvey_tail_statistics_generation_20260906_v1/`, `_20260907_v2/` and `track_b_harvey_tail_kinematics_20260907_v2/` — tail-statistics and tail-kinematics code.
- `docs/checkpoints/track_b_harvey_final_question_closure_20260907_v1/` — the terminal closure package, including the fifth-jet causal test.

## Relationship to the final HH→4b study

This study answers "can the governing SPA-Net checkpoint be trusted?" before
it is used as the baseline for the pretrained-representation comparison in
[study 07](../07_pretrained_jet_representations/README.md). The saturation
finding (10M doesn't help) also directly informed the practical-effect floor
used to interpret the ZERO20 result: the ~0.002 AUC 2M→10M scaling "noise
ceiling" measured here is the empirical basis for that study's pre-registered
0.005 AUC practical-significance threshold.

## Provenance

`artifacts/hh4b_spanet_part_20260911/` (governing 2M-vs-10M comparison);
`docs/track_b/development_snapshot_20260821/spanet_10m_scaling/` (earlier
development pass); `docs/checkpoints/track_b_harvey_*` (10 directories,
2026-09-06 → 09-07); `docs/checkpoints/track_b_phase3_bounded_multifile_sophon_adapter_canary_20260806_v1/`
(the paused Sophon-adapter thread this study's tail-reliability framing
grew out of — see [study 05](../05_spanet_reconstruction/README.md)).
`delphes-hh4b-production` (tip `e17fdbf9`, 2026-09-02 compressed-tail
package, preserved above). A later, methodologically distinct tail/
sensitivity-gap study (`cms-resolved-sensitivity-gap-v1`) exists
**unmerged**, documented in [study 04](../04_hh4b_classical_ml/README.md) —
see [`docs/history/BRANCH_GUIDE.md`](../../history/BRANCH_GUIDE.md).
