# Technical response to Harvey — final closure pass

Question-by-question. Every number below states its source dataset/sample
size and any caveat. Read-only additive package; no frozen Track A/B/C
output was modified.

## 1. Does the fifth (or any extra) jet actually perturb the classification?

**Yes — demonstrated causally, not just observed.** A genuine same-event
A/B counterfactual was run through the unchanged, checksum-verified
governing SPA-Net 10M checkpoint over the full frozen 400,000-event
`production_2M_val.h5` cohort: (A) native input, (B) identical event with
jet slots ≥4 (rank ≥5) masked, leading four preserved exactly.

Population: 88,854 signal events with ≥5 selected jets (of 193,358 total).
Every one of them changes score under masking (0% unchanged). Median
Δu_logit (masked − native) = **−0.090** (68% CI [−0.093, −0.088]); mean =
**−0.148**. 58.1% of events become less signal-like when the extra jets
are removed (the extra jets were net-helping), 41.9% become more
signal-like (the extra jets were net-hurting). At the working points
Harvey has used throughout this project, masking **gains** far more tail
membership than it **loses**: u_logit>3.5, 2,578 events gain vs. 330 lose
(native pass count 5,374 → masked 7,622); u_logit>4.5, 3,441 gain vs. 610
lose (3,334 → 6,165). The effect grows monotonically with the extra jet's
pT (median Δu_logit from −0.061 at pT5 30–50 GeV to −0.563 at pT5≥150 GeV)
and depends non-monotonically on its angular separation from the leading
four, peaking (most negative, i.e. most helpful when present) at ΔR
1.0–1.5, and reversing sign (harmful when present) at ΔR≥2.0.

Not attempted: HH-origin-truth stratification (would require an unproven
cross-sample join to a different signal population) and SPA-Net
assignment-output recovery (technically feasible — a real assignment
target exists in this HDF5 — but not run in this pass to keep it
tractable). Full detail: `FIFTH_JET_COUNTERFACTUAL_FINAL.md`.

## 2. Comprehensive signal/background kinematic features; 1D/2D/4D correlations

Unchanged from `track_b_harvey_tail_kinematics_20260907_v2`. PRIMARY
population (n_bkg=110, QCD+ttbar+other, u>3.5): best 1D feature
`n_btag_loose`; best 2D `n_btag_loose+jet1_pt` (CV AUC 0.8931±0.0393); best
4D adds `jet2_pt+pT_H2` (CV AUC 0.8938±0.0392, but the 2D→4D improvement,
Δ≈0.001, is smaller than the fold-to-fold CV standard deviation and is not
considered credible). Nested CV confirms no meaningful selection-bias
inflation (naive vs. nested AUC differ by ≤0.0009).

## 3. Remaining ttbar suppression

The exploratory candidate `pT_H1 < 406.1153676240428 GeV` (98.6% signal
efficiency, 51.9% pooled ttbar rejection on 27 raw events) was validated
this task against the two independent ttbar lanes recorded in
`HARVEY_MASTER_EVENT_TABLE.parquet` (`inference_ttbar_1`, n=11;
`inference_ttbar_2`, n=16 — confirmed disjoint by `event_uid`). Applying
the frozen cut per-lane gives 36.4% rejection (lane 1) vs. 62.5% (lane 2).
A leave-one-lane-out test (derive the median-based threshold on one lane,
evaluate on the other) yields thresholds of 337.0 GeV and 549.5 GeV — a
1.63× spread, exceeding this task's 1.15× reproducibility bar. The
qualitative direction (surviving ttbar is kinematically harder/more
boosted than signal) holds in both lanes for `pT_H1` and `jet1_pt` (both
≥1.5× the signal median) but not as strongly for `HT`, `pT_H2`, `jet2_pt`.
**Verdict: `TTBAR_SUPPRESSION = NOT_REPRODUCED`.** The cut remains a
legitimate candidate observation, not a validated selection; it should not
be described as final or publication-ready. Full detail:
`TTBAR_SIMPLE_CUT_VALIDATION_FINAL.md`.

## 4. Fifth-jet pT and nearest-ΔR dependence

Observational finding unchanged from `HARVEY_FIFTH_JET_SUMMARY.md` §3
(signal, u>3.5: median u=4.66 with a 5th jet, n=17,520, vs. 5.18 without,
n=9,819) — now reinforced by the causal counterfactual in item 1 above,
which shows the same qualitative pT/ΔR dependencies appear when the extra
jet is actually removed from the model's input, not merely when comparing
different events.

## 5. Score formulation / compressed-near-one behavior

Unchanged from Track A. The near-1.0 score comb is a float32
representation artifact of the two-class softmax; the raw logit margin
Δ=z1−z0 remains continuous and non-saturating through it. Not re-derived
here.

## 6. Do the u≈6.6/6.9 spikes contain "hundreds" of events?

In Track A's 400,000-event development cohort: k=2 (u=6.9237) has 99
signal events, k=4 (u=6.6227) has 100 — combined, "hundreds," though
neither individually reaches 3 digits in that specific sample. In the
*physical* signal sample used for kinematic characterization this task
(a different, larger population, `HARVEY_MASTER_EVENT_TABLE.parquet`,
u>3.5, n=27,339 signal), the same two float32 bins hold **255** (k=2) and
**228** (k=4) events respectively — genuinely hundreds in that sample. All
counts are zero background at both k values, both samples.

## 7. Interesting physical/kinematic features of the spike events

**None found — `SPIKE_KINEMATICS = NO_CLEAR_SPECIAL_MODE`.** Comparing k=2
(n=255) and k=4 (n=228) against their nearest populated neighboring
float32 bins across 13 kinematic features (HT, mHH, R_HH, pT_H1/H2, jet
pTs, ΔR(bb), fifth-jet presence, etc.) using bootstrap medians and Cliff's
delta: 0 of 26 feature/bin comparisons reach a medium effect size
(|δ|≥0.33); the largest is δ=0.27 (pT5 at k=2). Kinematics vary smoothly
through the float32 comb; the spike-bin events are an unremarkable random
draw from the surrounding tail population, exactly as expected given
Track A's established quantization mechanism. Full detail:
`FP32_SPIKE_KINEMATICS_FINAL.md`.

## 8. Does single precision compromise the result?

No, materially. Unchanged from Track A v2/v3: at score>0.9997, 0/9,170
threshold-crossing disagreements between the FP32-stored score and the
exact raw-logit decision; at score>0.99997, 2/6,368 (0.031%) disagree, both
boundary-adjacent (true confidence within 4×10⁻⁸ of the literal threshold
in score space). 92.4%–100% of tail events share a stored FP32 probability
with at least one other event, but 99.9%+ retain a fully distinct raw
logit margin — the classifier's internal ranking is preserved; only the
*displayed* probability is compressed.

## 9. How long/costly to double statistics at u>3.5 / u>4.5?

Unchanged from `HARVEY_MC_DOUBLING_ESTIMATE.md`/`HARVEY_GENERATION_
RESOURCE_ESTIMATE.md`: doubling either region under ordinary (non
-importance-sampled) generation requires a full second production of the
existing QCD sample's size (+278,360,000,000 generated, +87,623,306
selected events) — the requirement is identical for both thresholds, since
ordinary generation cannot decouple one score region's statistics from
another's. Estimated cost: ~7.5M CPU-slot-hours (allocated-wall basis).

## 10. Which preselection variables isolate the tails?

Reorganized (not recomputed) into three explicit stages: (1) analysis-level
post-Delphes variables — b-tag mistag rate is the strongest lever found
(AUC up to 0.99) but cannot reduce generation cost, only downstream
analysis cost; (2) variables available before SPA-Net scoring — the same
set, minus the score itself; (3) generator-level variables — only `pTHat`
(native Pythia8 `bias2Selection`) is structurally available early enough to
reduce *generation* cost itself, and remains **PROMISING BUT CANARY NOT YET
RUN**, unchanged. Full table: `PRESELECTION_ANSWER_FOR_HARVEY.md`.

## 11. Is the result stable as QCD statistics increase?

**Empirically checked for the first time using the actual existing
production** (not a projection). Per-file generated-event exposure is
provably uniform for lane 1 (`inference_qcd_1`: 17,600 jobs / 176 files =
exactly 100 jobs/file) but not provable for lane 2 (`inference_qcd_2`:
38,072/381 = 99.93, not an integer; no per-file job manifest exists) — this
blocker is disclosed, and lane 2's per-file exposure is reported as an
efficiency-scaled approximation, not an audited count.

At u>3.5 (68 raw QCD survivors across the full existing sample): a
constant-rate dispersion test across 8 geometric cumulative checkpoints
gives χ²/dof=0.96 — **`STABLE_WITHIN_CURRENT_MC`**. The two lanes'
survival rates are compatible (22 of 68 in lane 1 vs. 46 in lane 2; exact
binomial p=0.897 against the generated-exposure-weighted expectation).

At u>4.5 (12 raw QCD survivors total): **`TOO_FEW_EVENTS_TO_JUDGE`** —
disclosed rather than forced into a verdict the data cannot support
(binomial lane-compatibility p=0.536, for what limited power it has).

This tests stability *within* the two existing production lanes only; it
does not substitute for validation against a future, independently
generated sample. Full detail: `QCD_EXISTING_STATS_STABILITY_FINAL.md`.

## 12. Is the extreme result single-event or one-process dominated?

Unchanged from `HARVEY_PHYSICAL_TAIL_STATS_FINAL.md` Sec.6: not
single-event dominated (the terminal ≥3.0%-efficiency population, 12 raw
events, carries 77.1% of the total significance-squared; the next-largest
piece, 12.4%, is a 3-event shell). QCD dominates raw support and weighted
background at every threshold examined (61–90% of raw events, 84–90% of
weighted B).

## 13. The 75-epoch question

Unchanged: the 75-epoch SPA-Net 10M run completed cleanly but did not
improve on the frozen 50-epoch checkpoint on either validation average jet
accuracy (0.49258 vs. 0.49284) or validation loss (0.49618 vs. 0.49178).
The 50-epoch checkpoint remains governing.

## 14. Characterization of the extreme QCD/ttbar survivors

QCD1-vs-QCD2 composition (this task, u>3.5 survivors): HT medians 590 GeV
(lane 1, n=22) vs. 515 GeV (lane 2, n=46); mHH 811 vs. 725 GeV; R_HH 143
vs. 151 — broadly consistent shapes across the two independent
productions. ttbar survivors are systematically boosted relative to signal
(pT_H1, jet1_pt, HT roughly 2–3× the signal median in both lanes). The
strongest QCD-tail discriminator (from the pre-existing proxy-tail study)
is b-tag mistag rate (AUC up to 0.99): QCD only reaches the extreme tail by
simultaneously looking kinematically plausible *and* mistagging light/gluon
jets as b-jets.

## 15. Any earlier direct question not yet superseded/resolved

None found beyond the items above, after reviewing the working-points,
BDT-convergence, and background-suppression packages in this project's
Harvey-facing lineage — all superseded by later, more complete packages
already cited in the closure matrix.
