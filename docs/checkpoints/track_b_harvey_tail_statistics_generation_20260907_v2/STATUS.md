# Status

**Package:** `track_b_harvey_tail_statistics_generation_20260907_v2`
**Date (UTC):** 2026-09-07
**Type:** Read-only reconciliation / post-processing of already-frozen
physical-background artifacts. No new computation of any model output.

## What was done

1. Reconciled exact raw/weighted governing physical-background tail counts
   at `u>3.5`, `score>=0.9997`, `u>4.5`, and `score>=0.99997`, independently
   re-derived directly from `FULL_QCD_SURVIVOR_SIDECAR.h5` and the frozen
   non-QCD `job_summary_*.json` per-event survivor lists (Sec.1).
2. Reported naive-vs-correct (N_eff-based) finite-MC uncertainty at the
   existing governing eps_S=4.0% WP and at the two u-thresholds (Sec.2).
3. Projected 2x/4x/10x QCD-only MC-statistics scaling at fixed physical
   normalization, for four populations (Sec.3), with figures.
4. Estimated CPU-slot-hours / illustrative wall time / storage / scoring
   cost for 2x/4x/10x using only measured receipts (Sec.4).
5. Compared brute-force vs. targeted (importance-sampled) QCD generation
   conceptually, using existing kinematic findings as motivation only, with
   an explicit closure-test requirement for any targeted scheme (Sec.5).
6. Assessed stability of the existing SPA-Net advantage using the
   **corrected** (v3) 9-bin score-space likelihood decomposition, not the
   superseded v1 bin-folding bug (Sec.6).
7. Wrote Harvey-facing summary and email paragraph.
8. Independently recomputed every headline number from source
   (`work/cross_check_recompute.py`, `work/cross_check_output.txt`) —
   all assertions passed, all values matched already-published independent
   packages to the digit.

## What was explicitly NOT done

- No SPA-Net training, no ParT, no EAF/GPU access.
- No access to `holdout_B` or Stage-C.
- No new MC generation, no Condor production launch.
- No package installation or environment change.
- No modification to any existing package (`track_b_harvey_tail_
  characterization_20260825_v1`, `track_b_harvey_complete_3to5pct_
  study_20260825_v1`, `track_b_harvey_score_tail_diagnostics_20260829_v*`,
  `track_b_harvey_followup_finescan_likelihood_20260829_v1`,
  `track_b_harvey_spanet_compressed_tail_20260903_v2`,
  `track_b_qcd_tail_generation_strategy_20260902_v1`,
  `track_b_harvey_final_physics_bridge_20260825_v1`, or the repo's own
  `track_b_harvey_tail_statistics_generation_20260906_v1` package from the
  prior day) — all read only.
- No use of the 400k FP32/logit-precision development cohort anywhere in
  this package's physics numbers.

## Errors caught during this package's own adversarial self-check

- Section 2's threshold-tier table initially mislabeled the eps_S=4.0% WP
  (n=23) as `EXTREMELY_LIMITED`; the frozen source
  (`spa10m_finescan_rows.json`, field `support_tier_overall`) literally
  stores `LIMITED` for this row (the 20-99-event band). Corrected before
  finalizing.

## Result

`READY_FOR_HARVEY = YES`
