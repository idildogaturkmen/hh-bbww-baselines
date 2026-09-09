# Stability study: tail statistics as cumulative QCD generation increases

**Scope.** This is a design document. No block has been generated, no
Condor/EAF job launched. `NEW_JOB_LAUNCHED = NO`. It is written to be
executed against **new, independent QCD blocks** — never against
`inference_qcd_1`/`inference_qcd_2`, which remain the untouched closure
reference throughout (per `VALIDATION_AND_CLOSURE_CONTRACT.md` item 2,
reused unchanged here).

## 1. Block definition and independence

- **Block size**: start with the existing, already-designed 20,000-event
  canary (10,000 unbiased control + 10,000 `bias2Selection`-biased,
  `CANARY_PRODUCTION_PLAN.md`) as **block 0** (mechanism-validation only,
  not counted toward physics statistics). Subsequent physics blocks should
  be sized geometrically (e.g. 0.5M, 1M, 2M, 4M, 8M... generated events) so
  that early blocks are cheap sanity checks and later blocks dominate total
  statistics without requiring a fixed number of blocks to be decided in
  advance.
- **Independence (binding, reused from `VALIDATION_AND_CLOSURE_CONTRACT.md`
  item 1)**: every block uses a freshly-drawn seed range, recorded and
  cross-checked against every seed used in `training_qcd`,
  `inference_qcd_1`, `inference_qcd_2`, and every prior block in this
  study. A block that cannot prove seed-independence does not enter the
  cumulative total.
- **Per-block receipt (item 3 of the closure contract, reused)**: events
  attempted / passed generator-level cuts / passed the embedded production
  filter / passed Delphes / passed the SPA-Net-10M 7.5% WP survivor
  boundary — frozen before scoring, per block.
- **Weight bookkeeping**: if blocks are produced by ordinary (Design A,
  flat-weight) generation, each block's QCD events carry the same
  `weight_450fb = sigma_eff x L / N_generated_cumulative` (recomputed each
  time N_generated_cumulative grows, per the fixed-physical-normalization
  convention in `HARVEY_MC_DOUBLING_ESTIMATE.md` Sec.4). If any block uses
  `bias2Selection` (Design B), that block's events keep **per-event**
  `Info::weight()`-derived weights, never averaged to a stratum flat value
  (closure contract item 5) — the cumulative `N_eff`/`sum_w2` machinery
  below already handles mixed flat- and per-event-weighted blocks
  correctly as long as every event's own weight is retained.

## 2. What is tracked at every cumulative point

At each cumulative generated-event count `N_cum` (after each new block is
individually closure-checked per Sec.3 and admitted):

1. **Raw tail counts** at u>3.5 and u>4.5, separately for QCD / ttbar (if
   ttbar blocks are also being extended) / other / all-background.
2. **Weighted yield B** at 450 fb-1 (and at 1000/3000/4000 fb-1 by the same
   scale factor, per the existing `NORMALIZATION_CONTRACT.md` convention).
3. **N_eff = (sum wi)^2 / sum(wi^2)** and **sum_w2**, computed exactly (not
   approximated as raw N) — mandatory once any biased block is mixed in
   (closure contract item 8).
4. **Relative MC statistical uncertainty** = 1 / sqrt(N_eff).
5. **Process composition**: QCD / ttbar / other raw-count and B fractions.
6. **Kinematic distributions** of the tail population (HT, mHH, R_HH,
   pT(H1)/pT(H2), DeltaR(bb), leading-jet pT, n_selected_jets) — summary
   statistics (median, IQR) plus a KS or coarse-binned chi-squared
   comparison of the **cumulative** distribution against the **previous**
   cumulative distribution (not against a fixed reference, since the
   population itself is what's accumulating) and, separately, against the
   frozen `inference_qcd_1`/`inference_qcd_2` distribution in the
   well-populated overlap region (closure contract item 6 — a hard-stop
   check).
7. **score / u distribution**: full histogram of `u = -log10(1-score)` for
   the cumulative tail population, log-y, to visually track whether new
   blocks are filling in the same shape or revealing a new mode.
8. **Top preselection-feature distributions** (from
   `HARVEY_PRESELECTION_FEATURE_RANKING.csv`): min-probB(leading 4),
   probB_leading4_mean, HT/mHH — tracked the same way as item 6, since a
   drift in *these* specific variables would most directly signal a
   chain-configuration problem (they are the variables that most determine
   whether an event reaches the tail at all).

## 3. Per-block admission gate (reuses `VALIDATION_AND_CLOSURE_CONTRACT.md`
items 1, 4, 6, 7, 9, 11 as hard gates)

A new block is added to the cumulative total **only if**:

- Seed-independence is provable (item 1).
- Cross-section/filter-efficiency/bias-configuration receipts exist,
  frozen before scoring (item 4).
- The block's own weighted HT/jet-pT/jet-multiplicity distribution, in the
  region overlapping `inference_qcd_1`/`inference_qcd_2`'s well-populated
  range (reconstructed HT 300-700 GeV), is statistically consistent with
  that frozen reference (item 6) — **any failure here is a hard stop**,
  investigated before the block is admitted, exactly as specified in the
  existing contract.
- The cumulative weighted yield in any region where the frozen inclusive
  sample also has adequate statistics still reproduces the frozen sample's
  yield there within MC uncertainty (item 7).
- The block's own tail-population kinematics are not systematically
  discrepant from the existing tail characterization in
  `track_b_harvey_spanet_compressed_tail_20260903_v2` in any overlapping
  stratum (item 9).
- No `holdout_B`/Stage-C access anywhere in the chain (item 11).

A block that fails any gate is **excluded from the cumulative total and
investigated**, not silently down-weighted or dropped without record.

## 4. Convergence plots (to be produced once real blocks exist; templates
described here, illustrative axes shown using the current single frozen
point in `figures/neff_vs_qcd_multiplier.png` and `figures/relunc_vs_qcd_
multiplier.png` in this package)

- **B vs cumulative generated events** (u>3.5 and u>4.5, separate panels):
  under the fixed-normalization convention, the *central* B should be flat
  (by construction, for Design A) or converge from block-to-block scatter
  toward a flat asymptote (for Design B, where each block's own finite
  statistics still fluctuate before averaging) — plotted with a shrinking
  MC-uncertainty band (`+/- sqrt(sum_w2)`) around it. **A visible trend
  (not just scatter) in the central value as N grows is itself a red flag**
  — it would mean either a normalization bug or a genuine, previously
  unsampled kinematic region opening up.
- **N_eff vs cumulative generated events**: expected to grow
  approximately linearly with N for flat-weight blocks (Design A) once QCD
  dominates sum_w2, and sub-linearly (bounded above the current
  minor-background floor) if minor backgrounds are not also being
  extended — exactly the behaviour already demonstrated analytically in
  `HARVEY_MC_DOUBLING_ESTIMATE.md` Sec.4 and shown in
  `figures/neff_vs_qcd_multiplier.png` for the QCD-only-scaling case.
- **Tail efficiency vs cumulative generated events** (survivors / generated,
  at u>3.5 and u>4.5): this is the quantity that should genuinely
  *converge* to a stable value — it is the underlying physical acceptance
  being measured, not an artifact of normalization convention. Plotted
  with binomial error bars shrinking as `1/sqrt(N)`.

## 5. Quantitative stability criterion

Declare the cumulative tail statistics **stable** at a given region
(u>3.5 or u>4.5) once **all** of the following hold simultaneously:

1. **Precision target reached**: relative MC uncertainty on the combined
   all-background B falls at or below a pre-declared target (e.g. the
   10-12% band already used elsewhere in this project's fine-scan work,
   `OPTIMAL_ALLOCATION.md` Sec. "Applying both to concrete targets") —
   currently 10.58% at u>3.5 (already inside the band) and 26.73% at
   u>4.5 (needs approximately 4x QCD statistics per
   `HARVEY_MC_DOUBLING_ESTIMATE.md` Sec.4 to reach ~14%).
2. **No residual drift**: the central B estimates from the last two
   doubling-sized blocks (i.e. comparing N_cum and 2 x N_cum) agree within
   their combined 1-sigma MC uncertainty — `|B_2 - B_1| < sqrt(sigma_1^2 +
   sigma_2^2)`.
3. **Tail efficiency has converged**: the survivors/generated ratio at the
   two most recent doubling points agrees within its own binomial
   uncertainty (same form as item 2, applied to the efficiency instead of
   B).
4. **Kinematic and process-composition stability**: the KS/chi-squared
   comparisons in Sec.2 items 6-8, between the two most recent cumulative
   snapshots, do not reject compatibility (e.g. p > 0.01, Bonferroni-
   corrected across the small number of tracked variables) — a rejection
   here means the population's *character*, not just its size, is still
   changing, which should block a "stable" declaration even if items 1-3
   already pass.
5. **No admission-gate failure remains unresolved** (Sec.3) for any block
   contributing to the cumulative total.

**If item 1 is reached while items 2-4 are still failing**, this must be
reported explicitly as "precision target met but not yet demonstrated
stable" — the two are logically distinct and must not be conflated (a
precise but still-drifting estimate is not a stable one).

## 6. Relationship to existing infrastructure (reused, not duplicated)

This plan is a direct application of `VALIDATION_AND_CLOSURE_CONTRACT.md`
(binding gates), `CANARY_PRODUCTION_PLAN.md` (block 0 / mechanism
validation), and `OPTIMAL_ALLOCATION.md` (per-stratum allocation, if Design
B/C targeted generation is used instead of, or alongside, ordinary Design A
blocks) — all three already exist, frozen, in
`track_b_qcd_tail_generation_strategy_20260902_v1/` and are not modified or
re-derived here. This document's contribution is the specific
**cumulative-tracking and stability-declaration protocol** (Sec.2-5), which
did not previously exist as a standalone specification.

`NEW_JOB_LAUNCHED = NO`
