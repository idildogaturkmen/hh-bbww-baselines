# Harvey follow-up: comprehensive tail kinematics — executive summary (v2, corrected)

**Read-only throughout. No SPA-Net training. No frozen artifact modified. No holdout_B/Stage-C access. No new MC generation. This v2 package applies four methodological corrections to `track_b_harvey_tail_kinematics_20260906_v1` (preserved byte-for-byte, untouched); v1 remains on disk as the original, uncorrected reference.**

**Interpretation note, flagged as instructed:** Harvey's "u>3/5" is interpreted as **u>3.5** throughout this package. Exact `u` thresholds are used (u>3.5, u>4.5), not rounded score thresholds.

## What changed relative to v1 (read this first)

1. **MET/QCD population mismatch fixed.** `met_pt` is NaN for every QCD row, so any v1 model containing it silently evaluated on a 42-event non-QCD-only population while being compared to results on the full 110-event population as if identical. v2 splits this into a **PRIMARY** all-background (n=110) study that excludes `met_pt`, and a separately-labeled **AUXILIARY_NONQCD_MET** study (n=42, QCD absent) that is never compared numerically to the PRIMARY models.
2. **ttbar rejection arithmetic/prose fixed.** v1 said "51.9% (13 of 27 removed)" — 13 is the number of ttbar events *surviving*, not removed. Correct: **14/27 removed = 51.85%, 13/27 remaining**. The underlying computed fraction (0.518519) was always correct in v1; only the prose was wrong.
3. **Signal truth-matching audited and corrected.** v1's matching (nearest-jet-per-truth-object, no removal from the candidate pool) is many-to-one, not one-to-one. A one-to-one **exact** (maximum-cardinality, minimum-total-ΔR) bipartite matcher was implemented, unit-tested, and run over the full 27,339-event population (a second read-only pass over the same already-authorized files, initially blocked by an expired IHEP proxy, resumed after the user renewed it). It differs from a one-to-one greedy alternative in 3 of 27,339 events (assignment), 1 (j5 flag), 2 (leading-4 count), 0 (all-4 flag) — corrected fractions: **j5 match = 59.17% (10,367/17,520)**, **all-4-in-leading-4 = 0.4509% (79/17,520)**, both numerically close to but formally superseding v1's 59.1%/0.45%.
4. **Fifth-jet counterfactual plan finalized**, explicitly named a "leading-four-only counterfactual," specifying the exact `u_logit = softplus(Δ)/ln(10)` definition (corrected from an earlier draft's large-Δ approximation `Δ/ln(10)`) as the metric — still not executed.

## Population sizes (exact, every number below traces to these — unchanged from v1)

| | signal | QCD | ttbar | other background | total |
|---|---:|---:|---:|---:|---:|
| u>3.5 | 27,339 | 68 | 27 | 15 | 27,449 |
| u>4.5 | 18,155 | 12 | 2 | 1 | 18,170 |

**u>4.5 background support is extremely thin (12 QCD, 2 ttbar, 1 other) — every u>4.5 result in this package is explicitly labeled descriptive-only, never used for feature selection or cut optimization.**

## PRIMARY 1D/2D/4D (Corrected, Correction 1) — n_bkg=110 (QCD+ttbar+other), met_pt excluded

All three of 1D/2D/4D use the **exact same 110-background rows**; `met_pt` and any feature not defined for all 3 background classes is excluded from the candidate list.

- **Best 1D**: `n_btag_loose`
- **Best 2D**: `n_btag_loose + jet1_pt`, **CV AUC = 0.8931 ± 0.0393**
- **Best 4D**: `n_btag_loose + jet1_pt + jet2_pt + pT_H2`, **CV AUC = 0.8938 ± 0.0392**
- **Nested CV (selection-bias check)**: 2D nested CV AUC = 0.8931 ± 0.0393 (identical to naive — the same pair, `n_btag_loose`+`jet1_pt`, was selected in all 5 outer folds); 4D nested CV AUC = 0.8929 ± 0.0389 (vs. naive 0.8938, Δ=0.0009 — **not a meaningful search-induced optimism** at this population size). Both the naive and nested numbers are honestly reported side by side in `HARVEY_2D_FEATURE_RANKING.csv` / `HARVEY_4D_FEATURE_RANKING.csv` / `work/primary_v2_summary.json`.
- **The 4D improvement over 2D (Δ≈0.001) is not credible** — far smaller than the fold-to-fold CV standard deviation.

Full ranking: `HARVEY_1D_FEATURE_RANKING.csv` (+`_full.csv`), `HARVEY_2D_FEATURE_RANKING.csv` (+`_full.csv`), `HARVEY_4D_FEATURE_RANKING.csv`. Produced by `work/analysis_primary_v2.py`.

## AUXILIARY_NONQCD_MET (Correction 1) — QCD ABSENT, n=42, NOT an all-background result

`met_pt` is undefined for QCD, so it cannot enter the PRIMARY ranking above. Evaluated **only** against the 42 non-QCD backgrounds (27 ttbar + 15 other_background) it is available for:

- n_signal (met available) = 27,337; n_nonqcd_background = 42
- met_pt AUC (raw) = 0.260; bootstrap mean = 0.257, 68% CI = [0.219, 0.301]
- median signal met_pt = 49.2 GeV vs. median non-QCD background met_pt = 106.6 GeV

**This number is NOT comparable to the PRIMARY 110-background AUCs above — different, smaller, QCD-absent population.** Produced by `work/analysis_auxiliary_met.py` → `work/auxiliary_met_summary.json`.

## ttbar (Task 5, Corrections 2) — EXPLORATORY, not validated

Surviving ttbar (n=27 at u>3.5) is systematically boosted relative to signal (2–2.5× higher pT_H1/jet pT/HT). Candidate cut **`pT_H1 < 406.115 GeV`** (the median of ttbar's own pT_H1 distribution — selected and evaluated on the same 27-event sample, hence exploratory):

- Signal efficiency: 98.6% (26,951/27,339 retained)
- ttbar rejection: **51.85% (14/27 removed, 13/27 remaining)**
- All-background rejection (raw count): 24.5% (83/110) — **weighted rejection is only 18.5%** (480.2/588.9 weighted-450fb survive), since this ttbar-tuned cut rejects QCD (which dominates the weighted background) far less efficiently (11.8%) than ttbar (51.9%).
- Full process-by-process raw+weighted before/after table, and exploratory downstream Z_A/B95 (Z_A nominal = 0.58, using each background's own Poisson-Gamma 95%-upper-mean: Z_A = 0.51), in `HARVEY_TTBAR_FEATURE_SUMMARY.md`.

## Fifth jet (Task 6) — v2 Corrections 3/3a/4 applied

**Observational, well-populated (tens of thousands of signal events) finding, conditioned on u>3.5**: signal events with a 5th selected jet score systematically lower (median u 4.66 vs. 5.18 without one).

**v1's truth-matching algorithm was audited and found to be many-to-one (not one-to-one/bijective)**: a reco jet could be the nearest match for more than one HH-origin b-hadron with no reassignment mechanism. An exact one-to-one, maximum-cardinality/minimum-total-ΔR bipartite matcher was implemented, unit-tested (`work/exact_bipartite_match.py`), and run over the full population (`work/audit_correction3a_exact_vs_greedy.py`, 200/200 files, 27,339 events, 107s — initially blocked 2026-09-07 by an expired IHEP proxy, `voms-proxy-info -all` showed `timeleft: 0:00:00`; resumed after the user renewed it and `voms-proxy-info -timeleft` confirmed a positive lifetime). **Corrected (EXACT) results: j5 HH-origin-truth-match fraction = 59.17% (10,367/17,520 events with ≥5 selected jets); all-4-in-leading-4 fraction = 0.4509% (79/17,520)** — both numerically close to v1's original 59.1%/0.45%, since the non-bijective collisions the audit found (3 of 27,339 events differ in full assignment) were rare, but the EXACT values are the ones now reported as final. Full detail and the differential event-by-event breakdown: `HARVEY_FIFTH_JET_SUMMARY.md` §2.

For QCD (genuine model output, n=28, extremely limited, unaffected by the truth-matching fix since QCD has no HH-origin truth objects), the 5th jet enters SPA-Net's actual assignment 71.4% of the time.

v1's ΔR-bin ISR/FSR/pileup physical-origin interpretation is **removed in v2** — no branch-level truth-origin information supports attributing specific ΔR bins to those specific physical processes; the recomputed heatmap (`figures/fifth_jet_signal_truth_heatmap.png`, EXACT matching) is reported purely descriptively.

**This is association, not a tested causal perturbation.** A precisely-specified "leading-four-only counterfactual" (mask all jet slots ≥4, re-score with the unchanged frozen checkpoint, using raw logits and `u_logit` rather than the FP32-saturating `u`) is fully planned in `HARVEY_FIFTH_JET_SUMMARY.md` §5 but explicitly **not executed** in this task.

## What is proxy-based / descriptive vs. genuine, stated plainly

- **Genuine SPA-Net assignment**: QCD only (`spanet_10m_assignment_indices`, frozen). Signal and all non-QCD backgrounds use the **leading-four-by-pT geometric convention**, explicitly not a model output.
- **Signal "truth" fields**: this analysis's own ΔR<0.4-to-`gen_bhadron_fromhh` convention, not verified identical to any SPA-Net-training-time convention. v1 used a non-bijective matching algorithm; v2 (Correction 3) replaced it with an exact one-to-one bipartite matcher, run over the full population (see above).
- **u>4.5 tables/figures**: every one is labeled descriptive-only given 12–15-event background support.
- **u>6 FP32-spike region**: descriptive tables only produced where requested (`work/` companion files); **not interpreted as physical**, per instruction, pending the separate precision audit.

## Verification performed before finalizing

- v1 preserved byte-for-byte: `sha256sum -c SHA256SUMS` against the v1 package re-verified during v2 work.
- Master table: `event_uid` uniqueness asserted; all rows confirmed u>3.5; component counts (signal+QCD+ttbar+other) reconciled exactly against the merge (27,449 / 18,170) — unchanged from v1, re-verified.
- PRIMARY 110-background population verified via NaN-scan (zero NaN across all candidate PRIMARY features) and an in-script `assert n_bkg_check == 110`.
- Frozen ttbar cut threshold (406.1153676240428) independently re-derived as the median of ttbar's own pT_H1 distribution and confirmed to reproduce n_ttbar_before=27/after=13 exactly.
- Exact bipartite matcher unit-tested against an adversarial case (greedy provably under-matches), a tied-cardinality case (exact correctly minimizes total ΔR), trivial cases, and a synthetic worst-case timing benchmark, before being run at full population scale (200/200 files, 27,339 events, 107s, zero file failures).
- 1D/2D/4D AUC implementations unit-tested against known-answer cases (perfect separation→1.0, random→0.5, brute-force cross-check to 1e-9) before use (carried over from v1, unchanged).
