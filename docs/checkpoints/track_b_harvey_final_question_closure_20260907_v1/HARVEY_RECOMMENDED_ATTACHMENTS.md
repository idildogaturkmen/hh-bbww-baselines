# Recommended attachments (of the ~30 figures produced across Tracks A/B/C/D)

Capped at 6, chosen for information density and direct correspondence to a
specific Harvey question.

1. **`figures/threshold_crossings_counterfactual.png`** (Task 1) — the
   single most important new figure this pass produced. Shows, in one
   glance, how many signal events actually cross the u_logit=3.5/4.5
   working-point boundaries when the 5th+ jet is masked out — the direct,
   quantitative answer to "does the fifth jet actually perturb the
   classification."

2. **`figures/delta_u_logit_distribution.png`** (Task 1) — the full
   distribution of the causal score shift under masking, for every signal
   event with ≥5 selected jets. Shows the effect is real but modest and
   two-sided (some events become more signal-like, some less), not a
   one-directional artifact.

3. **`figures/qcd_u35_yield_stability.png`** (Task 2) — the running QCD
   tail-survival rate with Poisson error bars vs. cumulative generated
   exposure, converging toward the final rate. Directly answers "is the
   result stable as statistics increase," with the actual existing data
   rather than a projection.

4. **`figures/fp32_spike_feature_comparison.png`** (Task 3) — median
   kinematics vs. float32 lattice bin, target bins k=2/k=4 marked with
   stars. Shows visually that the spike bins sit smoothly on the same trend
   as their neighbors — the kinematic complement to Track A's precision
   finding.

5. **`figures/ttbar_lane_validation_pTH1.png`** (Task 4) — pT_H1
   distributions for signal and both independent ttbar lanes, frozen
   candidate cut overlaid. Shows both the real qualitative separation and
   the lane-to-lane dispersion that drives the `NOT_REPRODUCED` verdict.

6. **`figures/qcd1_vs_qcd2_tail_composition.png`** (Task 2) — HT/mHH/
   n_btag_loose distributions for the two independent QCD lanes'
   u>3.5 survivors overlaid. Directly shows the two productions agree in
   shape as well as rate.

**Not recommended for the email** (available in this package if requested):
`delta_u_logit_vs_pt5.png`, `delta_u_logit_vs_minDR5.png` (supporting detail
for Fig. 1-2, useful for a technical appendix but not for the headline
email), `qcd_tail_rate_vs_generated_events.png` (raw-count version of Fig.
3, less informative than the rate plot), `fp32_spike_ht_mhh.png` /
`fp32_spike_fifth_jet_fraction.png` (supporting detail for Fig. 4).
