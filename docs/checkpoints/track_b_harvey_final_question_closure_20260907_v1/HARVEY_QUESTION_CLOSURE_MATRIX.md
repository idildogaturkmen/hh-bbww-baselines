# Harvey question-closure matrix (Track D, final pass)

Every substantive Harvey point, its status before this task, the exact
package/file that answers it, and (Section 6, appended after Tasks 1-5 run)
its final status. This package is purely additive: it does not modify Track
A/B/C outputs, the governing checkpoint, or any prior package.

## Status before this task (Section 0)

| # | Harvey point | status before this task | evidence (exact package/file) |
|---|---|---|---|
| 1 | Comprehensive signal/background kinematic features | ANSWERED_ALREADY | `track_b_harvey_tail_kinematics_20260907_v2/HARVEY_KINEMATICS_EXECUTIVE_SUMMARY.md`, `HARVEY_1D_FEATURE_RANKING.csv` |
| 2 | 1D, 2D, 4D correlations | ANSWERED_ALREADY | same package, `HARVEY_1D/2D/4D_FEATURE_RANKING.csv` (n_bkg=110 PRIMARY population; best 2D `n_btag_loose+jet1_pt` CV AUC 0.8931±0.0393; 4D improvement not credible) |
| 3 | Remaining ttbar suppression | PARTIALLY_ANSWERED | `HARVEY_TTBAR_FEATURE_SUMMARY.md` establishes the exploratory `pT_H1<406.115 GeV` candidate (98.6% sig eff, 51.9% ttbar rejection) but explicitly flags it as chosen-and-evaluated on the same 27-event sample — no independent validation existed |
| 4 | Fifth-jet pT and nearest-ΔR dependence | ANSWERED_ALREADY (observational) | `HARVEY_FIFTH_JET_SUMMARY.md` §3 (median u 4.66 with a 5th jet vs 5.18 without, conditioned on u>3.5) |
| 5 | Whether the fifth jet actually perturbs signal classification/assignment | **UNANSWERED** | `HARVEY_FIFTH_JET_SUMMARY.md` §5: "leading-four-only counterfactual... PLAN ONLY, NOT EXECUTED" |
| 6 | Score formulation / compressed-near-one behavior | ANSWERED_ALREADY | `track_b_harvey_fp32_score_precision_20260906_v2_governing10m/HARVEY_FP32_10M_FINAL_REPORT.md` + v3 exact-threshold package |
| 7 | Whether the u~6.6 and u~6.9 spikes contain hundreds of events | ANSWERED_ALREADY (counts only) | governing10m `receipt.json`: k=2 (u=6.9237) n=99 signal, k=4 (u=6.6227) n=100 signal, 0 background, in the 400k dev cohort |
| 8 | Whether single precision compromises the result | ANSWERED_ALREADY | v2/v3 FP32 packages: logit ranking preserved (99.9%+ of tail events retain distinct Δ even when FP32-tied); threshold-crossing disagreement 0/9258 at u>3.5, 3/6454 (0.046%) at u>4.5 |
| 9 | Interesting physical/kinematic features of events in the spikes | **UNANSWERED** | no prior package characterizes k=2/k=4 kinematics — only their existence/count was established |
| 10 | How long/costly to double statistics at u>3.5 and u>4.5 | ANSWERED_ALREADY | `HARVEY_MC_DOUBLING_ESTIMATE.md`, `HARVEY_GENERATION_RESOURCE_ESTIMATE.md` (+278.36B generated events either way; 2x/4x/10x N_eff/rel.unc. tables) |
| 11 | Which preselection variables isolate those tails | ANSWERED_ALREADY (needs reorganizing) | `HARVEY_PRESELECTION_CANDIDATES.md` — b-tag mistag rate dominant (AUC up to 0.99) but stage-1-only; `pTHat` bias is the only generation-cost lever, canary not run |
| 12 | Whether the result is stable as QCD statistics increase | **PARTIALLY_ANSWERED** (projection only) | `track_b_harvey_tail_statistics_generation_20260907_v2/HARVEY_PHYSICAL_TAIL_STATS_FINAL.md` Sec.3 models a *hypothetical future* 2x/4x/10x rescaling under a fixed-weight assumption — this is a projection, not an empirical read of the two lanes as they actually accumulated |
| 13 | Whether the extreme result is single-event or one-process dominated | ANSWERED_ALREADY | same package, Sec.6: terminal ≥3.0%-efficiency population = 12 raw events carries 77.1% of Z_A²; not single-event; QCD dominates raw support and weighted B at every threshold |
| 14 | The 75-epoch question | ANSWERED_ALREADY | `track_b_harvey_spanet_compressed_tail_20260903_v2/PART_E_75EPOCH_TWO_SENTENCES.md`: 75-epoch run did not improve on the frozen 50-epoch checkpoint on either metric; 50-epoch remains governing |
| 15 | Characterization of the extreme QCD/ttbar survivors | ANSWERED_ALREADY (mostly) | `HARVEY_PRESELECTION_CANDIDATES.md` §1 (3-event u_2M>3.5 QCD descriptive characterization) + `HARVEY_TTBAR_FEATURE_SUMMARY.md` (ttbar boosted kinematics) |
| 16 | Any earlier direct question in the quoted thread not superseded/resolved | RESOLVED_EARLIER (spot-checked) | `HARVEY_WORKING_POINTS.{tsv,json}`, `HARVEY_BDT_CONVERGENCE_INTERPRETATION.md`, `HARVEY_BACKGROUND_SUPPRESSION_BRIEF.md` — all superseded by later, more complete packages in the same lineage; no live open item found beyond #3, #5, #9, #12 above |

## What this task executes (only the genuinely remaining items: #5, #9, #12, #3-validation, #11-reorganization)

1. **Task 1** — fifth-jet leading-four-only counterfactual, actually run (item #5).
2. **Task 2** — empirical (not projected) QCD cumulative-statistics stability check (item #12).
3. **Task 3** — FP32 spike-bin kinematic characterization (item #9).
4. **Task 4** — ttbar cut leave-one-lane-out validation (item #3).
5. **Task 5** — preselection three-stage reconciliation table (item #11, reorganization only).

Items #1, #2, #4, #6, #7, #8, #10, #13, #14, #15, #16 are **not redone** — they
are cited above and carried into the final response unchanged.

## Final status (Section 6 — filled in after Tasks 1-5 completed)

| # | Harvey point | final status | result |
|---|---|---|---|
| 1 | Comprehensive kinematic features | RESOLVED_EARLIER | see above |
| 2 | 1D/2D/4D correlations | RESOLVED_EARLIER | see above |
| 3 | Remaining ttbar suppression | ANSWERED_WITH_LIMITATION | Task 4: `TTBAR_SUPPRESSION = NOT_REPRODUCED` under leave-one-lane-out (derived thresholds 337 GeV vs 549 GeV, ratio 1.63x; rejection 36.4% vs 62.5% applying the frozen cut per-lane) — the qualitative "boosted ttbar" direction survives for pT_H1/jet1_pt only (2/5 features), not as a validated cut |
| 4 | Fifth-jet pT/ΔR dependence | RESOLVED_EARLIER | see above |
| 5 | Whether fifth jet perturbs classification/assignment | ANSWERED_WITH_RESULT | Task 1: genuine A/B counterfactual executed on the frozen governing checkpoint over the full 400k cohort — see `FIFTH_JET_COUNTERFACTUAL_FINAL.md` |
| 6 | Score formulation | RESOLVED_EARLIER | see above |
| 7 | u~6.6/6.9 spikes: hundreds of events? | RESOLVED_EARLIER | 99 + 100 = 199 signal events in the 400k dev cohort (both packages agree; "hundreds" combined, not individually) |
| 8 | Single precision compromise | RESOLVED_EARLIER | see above |
| 9 | Physical/kinematic features of spike events | ANSWERED_WITH_RESULT | Task 3: `SPIKE_KINEMATICS = NO_CLEAR_SPECIAL_MODE` — see `FP32_SPIKE_KINEMATICS_FINAL.md` |
| 10 | Cost to double statistics | RESOLVED_EARLIER | see above |
| 11 | Preselection variables | ANSWERED_WITH_RESULT | Task 5: `PRESELECTION_ANSWER_FOR_HARVEY.md` — reorganized three-stage table; `pTHat` bias remains PROMISING BUT CANARY NOT YET RUN |
| 12 | Stability vs QCD statistics | ANSWERED_WITH_RESULT | Task 2: `QCD_EXISTING_STATS_STABILITY_RESULT_U35 = STABLE_WITHIN_CURRENT_MC` (χ²/dof=0.96 across 8 geometric checkpoints; lane1-vs-lane2 rate-compatibility binomial p=0.897), `QCD_EXISTING_STATS_STABILITY_RESULT_U45 = TOO_FEW_EVENTS_TO_JUDGE` (n=12 total) |
| 13 | Single-event/one-process dominance | RESOLVED_EARLIER | see above |
| 14 | 75-epoch question | RESOLVED_EARLIER | see above |
| 15 | Extreme survivor characterization | RESOLVED_EARLIER (+ Task 2/3 add QCD1-vs-QCD2 composition and spike-bin kinematics) | see above |
| 16 | Any other unresolved thread item | RESOLVED_EARLIER | none found beyond items above |
