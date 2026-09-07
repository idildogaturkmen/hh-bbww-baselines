# Harvey's fifth-jet question (Task 6) — v2, Correction 3 / 3a / 4 applied

**Terminology change (v2): "HH-origin b-hadron" is used throughout instead of "b-quark" or "Higgs-decay b-quark" — the underlying branch is `gen_bhadron_*` with a `fromhh` flag, a hadron-level truth object, not a parton-level quark record. No branch-provenance evidence supports the stronger "b-quark" language used in v1.**

**Raw support: signal 17,520 events with ≥5 selected jets (of 27,339 at u>3.5) and 9,819 with exactly 4; QCD 28 events with ≥5 selected jets (of 68 at u>3.5); ttbar 19 events with ≥5 selected jets (of 27 at u>3.5) — QCD and ttbar numbers here are EXTREMELY LIMITED, reported descriptively only.**

## 0) v1 truth-matching algorithm — exact pseudocode, and why it is not one-to-one

v1's `extract_signal.py` computed truth-matching as:

```
for each event i:
    truth_objects = [ t for t in gen_bhadron if gen_bhadron_fromhh[t] == True ]
    n_truth_bhadrons_fromhh[i] = len(truth_objects)
    for t in truth_objects:                       # loop order: per TRUTH object, independently
        for j in selected_jet_slots[0 : n_sel(i)]:
            dR[j] = DeltaR(truth_eta[t], truth_phi[t], jet_eta[i,j], jet_phi[i,j])
        j_star = argmin_j dR[j]
        if dR[j_star] < 0.4:
            truth_matched_slot[i, j_star] = True   # sets a boolean flag; j_star is NOT removed from the pool
```

**Bijectivity audit:**
- Can one HH-origin b-hadron match multiple reco jets? **No** — each truth-object loop iteration selects a single argmin jet.
- Can one reco jet be the nearest match for multiple HH-origin b-hadrons? **Yes** — nothing removes `j_star` from later truth objects' candidate pool, so two (or more) truth objects can independently choose the same jet slot as nearest.
- **Conclusion: v1's matching is many-to-one, not one-to-one (not bijective).** When a collision occurs, the boolean output (`truth_matched_slot`) cannot distinguish "2 truth objects matched to 1 jet" from "1 truth object matched to 1 jet" — it silently undercounts the true multiplicity of the collision, and the jet is never reassigned to whichever truth object it would have gone to under an optimal one-to-one assignment.

## 1) Distribution of `n_truth_bhadrons_fromhh` over signal events (u>3.5, n=27,339)

| n_truth_bhadrons_fromhh | n events |
|---:|---:|
| 4 | 27,099 |
| 5 | 3 |
| 6 | 235 |
| 8 | 2 |

**Fraction with exactly 4 HH-origin b-hadrons: 99.12% (27,099 / 27,339).** The small population with 5/6/8 flagged truth objects (240 events, 0.88%) is not further characterized here; it likely reflects gluon-splitting/fragmentation producing more than one `gen_bhadron` record traceable to the HH system in a small subset of events, not a data error (all these events pass the same `fromhh==True` selection criterion, unmodified from v1).

## 2) Correction 3a — GREEDY is not proven globally optimal; EXACT bipartite matching implemented

The v1 algorithm above is not one-to-one. A **corrected one-to-one (bijective)** replacement was implemented as the standard "sort all valid truth-jet edges (ΔR<0.4) globally, greedily accept the smallest-ΔR edge whose truth object and jet are both still free" procedure. This is one-to-one by construction, but **a global-greedy edge acceptance is not guaranteed to maximize the number of matched pairs, nor to minimize the total ΔR among maximum-cardinality matchings** — a real gap, correctly flagged rather than asserted away.

**Exact solver implemented** (`work/exact_bipartite_match.py`): a bitmask dynamic program over `(truth_index, bitmask_of_used_jets)` that exhaustively enumerates every valid one-to-one assignment respecting ΔR<0.4, with lexicographic objective (1) maximize number of matched truth objects, (2) minimize total ΔR among matchings achieving that maximum. This is exact by construction (full enumeration via memoized recursion, no heuristic step) — not an approximation. State space is at most `n_truth × 2^n_jets ≤ 8 × 2^10 = 8,192` per event (this dataset: n_truth≤8, n_jets≤10), and the typical event (n_truth=4, n_jets≤6) is far smaller.

**Unit-tested before use** against:
- An adversarial case constructed so greedy strictly under-matches (greedy achieves cardinality 1, exact achieves cardinality 2) — exact correctly beats greedy.
- A tied-cardinality case where two maximum matchings exist with different total ΔR — exact correctly picks the minimum-total-ΔR one.
- Trivial 1×1 and no-valid-edge cases.
- A synthetic worst-case 8×10 random instance — solves in ~1.7 ms, confirming tractability was correctly benchmarked before committing to a full-population run (per instruction).

**Full-population GREEDY-vs-EXACT comparison: COMPLETE.** (Initially blocked 2026-09-07 by an expired IHEP VOMS/CMS proxy — `voms-proxy-info -all` showed `timeleft: 0:00:00`, producing XRootD error 3010 `kXR_NotAuthorized` on the first benchmark file. The user renewed the proxy; `voms-proxy-info -timeleft` was re-verified positive (43,154s) before resuming. `work/audit_correction3a_exact_vs_greedy.py` then ran to completion: 200/200 files, 27,339 events, 107s.)

**GREEDY vs EXACT, event-by-event (n=27,339 signal events at u>3.5):**

| Comparison | n events differing |
|---|---:|
| Full per-truth-object assignment | 3 |
| j5 truth-match flag | 1 |
| n truth objects matched in leading-4 | 2 |
| all-4-in-leading-4 flag | 0 |

**These differences are real but rare (≤3 of 27,339 events, ≤0.011%)** — inspecting them directly confirms the mechanism: e.g. event (file_index=152, entry_index=544), greedy assigns truth-object-0 to jet1 (its individual nearest jet) leaving one fewer truth object matched overall (2 of 4), while the exact solver reassigns truth-object-0 to jet4 (jet5) instead, freeing jet1 for a different truth object and matching 3 of 4 truth objects total — one more than greedy, and specifically landing the extra match on jet slot 4, flipping that event's j5-match flag from False (greedy) to True (exact). This is exactly the class of error the many-to-one v1 algorithm and the unproven-optimal greedy replacement were both vulnerable to; the exact solver's lexicographic (max-cardinality, then min-total-ΔR) objective corrects it.

**Because differences exist (not zero), the EXACT matching result is used everywhere in this final v2 package, per instruction — not the greedy or v1 result.**

**Final EXACT-matching signal quantities:**

| Quantity | Denominator | n | Fraction | v1 value (non-bijective, for reference only) |
|---|---:|---:|---:|---:|
| Signal j5 HH-origin-truth-match fraction | 17,520 (events with ≥5 selected jets) | 10,367 | **59.17%** | 59.1% |
| All-4-in-leading-4 fraction (given ≥5 jets) | 17,520 | 79 | **0.4509%** | 0.45% |

**The EXACT-matching fractions are numerically almost identical to v1's original (non-bijective) figures** — the rare collisions found above (≤3 events) were not numerous enough to materially shift either aggregate fraction at this population size. This independently confirms, via the exact algorithm rather than by assumption, that v1's headline 59.1%/0.45% figures were fortuitously close to correct despite the flawed algorithm — but the EXACT values (59.17%, 0.4509%) are the ones now reported as final, and the audit trail above (3/1/2/0 differing events) is retained as evidence rather than asserted away.

## 3) Observational association — what does NOT depend on the truth-matching fix

### Signal: does having a 5th jet associate with lower score? (u>3.5 only, observational)

| | n | median u | frac(u>4.5) |
|---|---:|---:|---:|
| ≥5 selected jets | 17,520 | 4.66 | 58.6% |
| =4 selected jets | 9,819 | 5.18 | 80.3% |

**A clear, well-populated (tens of thousands of events) association, conditioned on u>3.5**: signal events with a 5th selected jet have systematically lower u, both in median and in the fraction reaching the tighter u>4.5 tail. **This is observational and conditioned on the u>3.5 selection already applied to build this population — it is not a claim about the relationship between jet count and score in the full (unconditioned) signal sample, and it is not a tested causal effect** (see §5, Correction 4).

### QCD (genuine SPA-Net assignment — unaffected by the truth-matching fix, since QCD has no HH-origin truth objects)

- **71.4%** of QCD events with a 5th jet have that jet **actually included in SPA-Net's real 4-jet assignment** (i.e. the model itself dropped one of the leading-4-by-pT jets in favor of jet 5).
- Only **14.3%** of these events have the assignment exactly equal to the leading-four set.
- **n=28 raw events** — every per-bin heatmap cell (`figures/fifth_jet_qcd_assignment_heatmap.png`) has 0–4 events; **explicitly EXTREMELY_LIMITED, not a statistically precise map**, shown for completeness only.

### ttbar

Descriptive kinematics only (median pT5≈40.6 GeV, median minΔR5≈1.16, n=19) — **no genuine assignment and no HH-origin truth-matching exist for ttbar**; not characterized further given the tiny support.

## 4) Removed in v2: unsupported ΔR-bin physical-origin interpretation

v1 stated that the heatmap's non-monotonic ΔR structure was "physically sensible: a jet very close to an already-selected jet is more likely overlapping radiation, one very far away is more likely unrelated ISR/pileup, while the true 'other Higgs' jet naturally sits at an intermediate angular separation." **This interpretive claim is removed in v2**: no branch-level truth-origin information (e.g., a parton-shower history flag distinguishing ISR/FSR/pileup jets) was used or is available in this analysis to support attributing specific ΔR bins to those specific physical origins. The recomputed heatmap (`figures/fifth_jet_signal_truth_heatmap.png`, EXACT matching, n=17,520) is reported here purely descriptively: which (pT5, minΔR5) bins have higher or lower `gen_bhadron_fromhh` match rate, with no causal or origin-based explanation attached beyond what the data directly shows. The bin structure is unchanged from v1 (same edges); the per-bin match-rate values differ from v1 by at most the single j5-flag flip found in §2 above, e.g. the lowest-pT5, lowest-ΔR bin moves from 0.4988 (v1, n=846) to 0.5012 (v2 exact, n=846). Full bin table: `work/fifth_jet_signal_heatmap_truth_EXACT.csv`.

## 5) Correction 4 — leading-four-only counterfactual: PLAN ONLY, NOT EXECUTED

**Purpose**: distinguish the *observational* association in §3 (events with a 5th jet score lower) from a genuine *causal* test of whether that specific 5th jet's presence in SPA-Net's input moves the model's score for the *same* event.

**Explicit naming requirement**: this is a **"leading-four-only counterfactual"** — for events with more than 5 selected jets (n_selected_jets > 5; some events have up to 10), masking "jet slot ≥4" removes *all* jets beyond the leading four simultaneously, not just jet 5 alone. **This plan does NOT claim to isolate jet 5 in isolation for events with >5 selected jets** — it isolates "the leading four only" as input (A) vs. "the leading four plus jet 5 plus everything else it originally had" as input (B). A true jet-5-only isolation (holding jets 6+ fixed, removing only slot 4) is a different, narrower experiment not specified here.

**A/B design**:
- **(A) Original, full input**: the exact SPA10M native input tensor as originally scored (all selected jet slots up to `n_sel`, capped at the model's native slot count), unchanged frozen 50-epoch checkpoint.
- **(B) Leading-four-only input**: identical event, identical jet ordering and kinematics for slots 0–3, with `mask[:, 4:] = False` (slots 4 and beyond hidden from the model), same unchanged checkpoint.
- Both (A) and (B) re-scored through the identical frozen inference code path (`class_logits` / `score_pair`), producing raw **logits** (not just the final softmax score) for both classes.

**Critical precision requirement — use `u_logit`, not the FP32-saturated `u`**:
- `u = -log10(1 - score)` saturates at FP32 precision once `score` rounds to `1.0` (empirically ≈7.22 in this project's prior precision audits) — many high-confidence signal events in this u>3.5/u>4.5 tail are expected to sit at or near that floor, which would make naive `u`-based before/after deltas meaningless (both A and B could show identical saturated `u` despite a real, measurable difference in the underlying model output).
- **Exact definition** (corrected — an earlier draft of this plan stated `u_logit = Delta / ln(10)`, which is only the large-positive-Delta *approximation*, not the exact quantity): with `Delta = z1 - z0` (the raw pre-softmax class-1 minus class-0 logit margin),
  ```
  u_logit = softplus(Delta) / ln(10) = log10(1 + exp(Delta))
  ```
  which is **exactly** `-log10(1 - sigmoid(Delta))` — i.e. the same `-log10(1-p)` transform as `u`, applied exactly (not approximately) to the logit margin via `p = sigmoid(Delta)`, consistent with Track A's definition. It reduces to the sigmoid/softmax-based `score` and hence to `u` itself whenever `score` has not yet saturated, and remains well-defined and non-saturating (limited only by float64, not float32) once `u` would otherwise saturate. Verified numerically: `softplus(Delta)/ln(10)` and `-log10(1-sigmoid(Delta))` agree to double-precision (e.g. Delta=10 → 4.342965 both ways). **The bare `Delta/ln(10)` approximation only agrees with the exact value for large positive Delta** (Delta=20: exact 8.68589 vs. approx 8.68589 — agree; Delta=1: exact 0.57034 vs. approx 0.43429 — diverge badly; Delta=-5: exact 0.00292 vs. approx -2.17147 — sign-wrong), so it must not be used as if exact, especially for events near or below the classification boundary.
- Both raw logits (`z0`, `z1`) must be saved for every (event, A/B) pair — not just the derived score — so `u_logit` (and any other downstream re-derivation) can be computed without re-running inference.

**Metrics to compute once executed** (not computed here):
- Δ(logit margin) = `(z1_B - z0_B) - (z1_A - z0_A)` per event.
- Δu_logit = `u_logit(B) - u_logit(A)` per event.
- Fraction of events where the score increases vs. decreases under masking (sign of Δu_logit).
- Threshold-crossing counts: how many events cross u_logit-equivalent-of-3.5 or 4.5 in either direction between A and B.
- Change in the geometric leading-four-by-pT assignment itself is not applicable here (masking only affects which jets the *model* sees, not the geometric pairing convention used elsewhere in this analysis) — but any change in which pairing (of the 3 `PAIRINGS`) is argmin-selected under (B) vs (A), if the assignment convention used for that event's kinematic features is affected, should be logged.

**Stratifications to compute once executed**:
- By `pt5` and `min_dR_j5_leading4` (already available, unaffected by the truth-matching fix).
- By the audited EXACT-matching signal j5 HH-origin-truth-match flag (§2: 59.17% match rate, n=17,520).
- By whether, for QCD events, jet 5 entered the original genuine SPA-Net assignment (`spanet_10m_assignment_indices`) — QCD is the only class with a real (non-proxy) assignment to stratify by.

**Explicitly NOT executed in this v2 correction task** (per instruction: finish the observational package first): no new inference is launched, no new logits are produced, no scores are recomputed. `COUNTERFACTUAL_READY = YES` means this plan is fully specified and ready to hand to the next task, not that it has been run.
