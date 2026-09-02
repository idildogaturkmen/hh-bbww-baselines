# SPA-Net 10M compressed tail: score ≥ 0.9997 vs ≥ 0.99997

**Scope and safety.** Read-only analysis of already-frozen artifacts only.
No Condor/EAF job launched, no model inference run, no training, no
holdout_B/Stage-C access. Governing checkpoint: frozen SPA-Net 10M
50-epoch primary (`fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c307102fa6943de5`).
The completed 75-epoch run is **not** substituted (Part E). This package
does not modify v1–v5 of the prior Harvey packages, the frozen fine-scan
package, the 50-epoch checkpoint, or the 75-epoch run.

**Coverage boundary, stated once, used throughout.** The SPA10M
7.5%-signal-efficiency threshold is **0.998958945274353**. QCD and
non-QCD survivor archives are complete (every event that passed is
retained, with its exact score) only for score ≥ this value. Both
thresholds Harvey has asked about are inside this boundary:

| threshold | vs. 0.998958945274353 | coverage |
|---|---|---|
| 0.9997 | above | **complete — exact answer from frozen data** |
| 0.99997 | above | **complete — exact answer from frozen data** |
| 0.997 (for reference) | **below** | **incomplete — see Part G** |

## Part A — exact ≥ 0.9997 / ≥ 0.99997 tail from frozen data

### A.1 Signal (holdout_A, non-Stage-C; n_total = 508,168)

| threshold | n | fraction of total |
|---|---:|---:|
| ≥0.999 | 37,351 | 7.350% |
| ≥0.9997 | 27,128 | 5.338% |
| ≥0.9999 | 23,525 | 4.629% |
| ≥0.99997 | 17,869 | 3.516% |
| ≥0.99999 | 11,556 | 2.274% |
| exact float32 = 1.0 | 173 | 0.0340% |

Signal support is large at every threshold checked; no small-N concern
on the signal side.

### A.2 Background — all frozen processes, all five thresholds

| threshold | total raw n | B (450 fb⁻¹) | N_eff | rel. MC unc. | support tier |
|---|---:|---:|---:|---:|---|
| ≥0.999 | 2,598 | 14,558.35 | 2,128.56 | 2.17% | PRIMARY_ADEQUATE |
| ≥0.9997 | 100 | 532.36 | 81.01 | 11.11% | FINITE_SUPPORT_CAUTION |
| ≥0.9999 | 40 | 221.35 | 33.29 | 17.33% | LIMITED_DIAGNOSTIC_B95_ORIENTED |
| ≥0.99997 | 15 | 97.26 | 13.99 | 26.73% | EXTREMELY_LIMITED |
| ≥0.99999 | 9 | 65.80 | 9.00 | 33.33% | EXTREMELY_LIMITED |

`N_eff = (Σwᵢ)²/Σwᵢ²`, `B = Σ(n_k·w_k)` under the existing frozen flat
per-process 450 fb⁻¹ weight convention (unchanged from every prior
package). **All five rows are exact, not extrapolated** — every event
counted here is inside the complete-coverage region.

### A.3 The ≥0.9997 population in detail (n=100 raw, B=532.36)

Process-by-process:

| process | n | weight (450 fb⁻¹) | B_k | % of total raw | % of total B |
|---|---:|---:|---:|---:|---:|
| QCD | 61 | 7.3113 | 445.99 | 61.0% | 83.8% |
| inference_ttbar_2 | 16 | 2.3214 | 37.14 | 16.0% | 7.0% |
| inference_ttbar_1 | 9 | 2.3214 | 20.89 | 9.0% | 3.9% |
| ZJetsToQQ | 8 | 1.4103 | 11.28 | 8.0% | 2.1% |
| SingleTop | 2 | 4.8846 | 9.77 | 2.0% | 1.8% |
| TTbarW | 2 | 0.3354 | 0.67 | 2.0% | 0.1% |
| TW | 1 | 6.2271 | 6.23 | 1.0% | 1.2% |
| TTbarZ | 1 | 0.3866 | 0.39 | 1.0% | 0.1% |

Full machine-readable version: `work/tail_counts_results.json`,
`work/event_table_ge0p9997.json` (100-row per-event table).

### A.4 The two populations with n < 20 — every event listed individually

**≥0.99997 (n=15): 12 QCD + 2 ttbar (inference_ttbar_2) + 1 SingleTop.**
**≥0.99999 (n=9): 9 QCD** (a strict subset of the 15 above).

Full per-event kinematics for both: `work/individual_events_ge0p99997.json`
and `work/individual_events_ge0p99999.json` (also reproduced in
`HARVEY_INDIVIDUAL_EVENTS_LT20_TABLE.md`). These 15/9-event counts and
the specific process split reproduce, independently and exactly, the
same population already reported in the prior score-tail package
(v1–v3) at this identical score cut — no discrepancy found.

### A.5 Score meaning, resolved rigorously

The stored `score_spanet_10m` is **`torch.softmax(classification_logits, dim=-1)[:, 1]`**
from the frozen scoring path (`run_spanet_physical_scan.py::class_logits`/`score_pair`,
independently re-inspected for this package, unchanged since prior
packages). It is the **class-1 two-class softmax output**, not a
calibrated probability — **no post-hoc calibration has ever been applied
or demonstrated** in this project. It should be called a "softmax
score" or "class-1 softmax output," not a "signal probability."

**Why it compresses near 1.** For two-class softmax with logits
`(z0, z1)`: `softmax(z)_1 = sigmoid(z1 - z0)`. As the logit difference
`z1 - z0` grows, `sigmoid` saturates exponentially close to 1, so linear
differences in score become invisible near 1 while the underlying model
output (the logit difference) can still vary substantially. This is a
**property of the sigmoid/softmax link function**, not evidence of
overfitting or miscalibration by itself.

**The u-transform**, defined purely algebraically:

```
u = -log10(1 - score)
```

is a **monotonic re-expression of the score** that undoes exactly this
compression (score=0.999→u=3, 0.9997→u≈3.523, 0.99997→u≈4.523,
0.99999→u=5, score→1⁻→u→∞). Larger u means a larger, more extreme
softmax output — **it is not a calibrated confidence level, a
p-value, or a likelihood ratio**; it is a relabeling of the same score
that makes near-1 differences visible on a plot.

**float32 score = 1.0, handled explicitly.** In float32, `1 - score`
underflows to exactly 0 once `score` rounds to `1.0f` (i.e. once
`score > 1 - 2⁻²⁴ ≈ 1 − 5.96×10⁻⁸`, `u` is mathematically infinite. This
is **not a broken calculation** — it correctly reflects a genuine
float32 representability limit, not a division-by-zero bug. Signal has
173 such events (0.034% of 508,168); **QCD has zero** in the frozen
sidecar (max QCD score = 0.99999917, still finite in float32); **no
non-QCD background event reaches exact 1.0 either** (checked directly,
0 found). Every plot below either excludes `u=∞` points from a finite
axis or notes their count explicitly — none are silently dropped.

### A.6 Figures (`figures/`)

All three are **raw MC event counts, not weighted by cross-section or
luminosity** — stated on every axis, per instruction not to normalize
away the small-support issue.

- `signal_vs_background_u.png` — signal (508,168 total; u≥1 shown) vs.
  total background (n=2,907, the complete ≥7.5%-WP union), both as
  step histograms, log-y. Vertical lines at score=0.9997 and 0.99997.
  The first/last bins are open-ended overflow bins (u<1 and u≥7 folded
  in) — stated here explicitly since a plot cannot show that on its
  own.
- `process_separated_background_tail_u.png` — every one of the 2,907
  complete-coverage background events plotted as an **individual point**
  (never a smoothed density), grouped by process on the y-axis. This is
  the honest picture of how thin background support actually is at high
  u: QCD has the most points and reaches furthest right; several minor
  processes contribute only a handful of points near the 7.5% boundary
  and none past u≈4.5.
- `raw_score_near_one_companion.png` — the same two populations in raw
  score space (not u-transformed), log-y, restricted to [0.999, 1.0].
  Included specifically to show *why* the u-transform is useful: in raw
  score space essentially everything piles into the last visible pixel.

## Part B — kinematics vs. transformed score (≥0.9997 population, n=100)

Descriptive only (Spearman rank correlation between u and each
variable; **no smoothing, no medians quoted for n<20 subpopulations** —
the individual-event tables in A.4 are the correct reference for those).

| variable | Spearman ρ(u, ·) | range | median |
|---|---:|---|---:|
| HT | −0.006 | [338, 4911] GeV | 637 GeV |
| mHH (leading-4) | +0.211 | [393, 5219] GeV | 791 GeV |
| R_HH (leading-4 pairing) | +0.210 | [4.7, 1626] GeV | 158 GeV |
| min ΔR(bb) | +0.152 | [0.42, 3.12] | 1.06 |
| max ΔR(bb) | +0.147 | [0.50, 4.64] | 2.27 |
| pT(H1) | +0.020 | [32, 2018] GeV | 293 GeV |
| pT(H2) | +0.067 | [10, 2518] GeV | 236 GeV |
| leading jet pT | +0.082 | [82, 1922] GeV | 256 GeV |
| subleading jet pT | +0.082 | [72, 1657] GeV | 168 GeV |
| jet multiplicity (n_selected) | +0.034 | [4, 7] | 4 |

**What is visible:** every correlation is weak (|ρ| ≤ 0.21). The two
largest, mHH and R_HH, point the same direction (higher-u events trend
slightly toward *larger* mHH and *larger* R_HH), but the effect is small
and R_HH is **not** concentrated near the signal-like region — the R_HH
range at ≥0.9997 spans from 4.7 GeV (genuinely Higgs-mass-consistent)
to 1,626 GeV (very far from it), median 158 GeV. **The highest-scoring
background events are not, as a population, kinematically
"signal-like" by simple dijet-mass-consistency geometry.**

**What is not visible / not claimed:** no strong monotonic trend in HT,
jet pT, or ΔR(bb); no evidence of a single dominant kinematic mode; no
causal claim of any kind. `work/partB_kinematics_results.json` carries
the full numeric detail; individual scatter data is in
`work/event_table_ge0p9997.json` for any further cut the reader wants
to make directly (no re-derivation needed).

## Part C — extra / "noise" jet question (≥0.9997 population)

**QCD (n=61) — genuine frozen SPA-Net assignment indices used, not a
geometric proxy:**

| | n | % |
|---|---:|---:|
| exactly 4 selected jets | 37 | 60.7% |
| ≥5 selected jets (has "extra" jets) | 24 | 39.3% |

Leading extra-jet pT (n=24): min 30.4, median 45.1, max 84.9 GeV.
Leading extra-jet probB (n=24): min 0.003, median 0.035, max 1.000.

**A genuinely new check enabled by the frozen assignment field:** does
SPA-Net's real predicted 4-jet assignment even match the leading-four
geometric convention used elsewhere in this project? **No, not always —
only 41/61 (67%) QCD events have `spanet_10m_assignment_indices` equal
to the leading-four-by-pT jets {0,1,2,3}; the other 20/61 (33%) have the
model picking at least one non-leading jet into its assignment.** This
is reported as a direct, measured fact about the frozen model output —
not evidence of any particular mechanism.

**Non-QCD (n=39) — no frozen SPA-Net assignment field exists for any
non-QCD process anywhere in this project.** Stated explicitly, as
required, rather than substituted with a geometric proxy presented as
equivalent: the "extra jet" numbers below use the same leading-four-by-
pT selected-jet count as every other topology table in this project
(`njet_selected` from `nonqcd_tail_topology.json`), but this is **not**
a model assignment for non-QCD, only a jet-counting convention.

| | n | % |
|---|---:|---:|
| exactly 4 selected jets | 17 | 43.6% |
| ≥5 selected jets | 22 | 56.4% |

Leading extra-jet pT (n=22): min 31.2, median 40.1, max 108.2 GeV.
Leading extra-jet probB (n=22): min 0.003, median 0.011, max 0.998.

**No ISR/FSR or heavy-flavor origin is claimed for any extra jet.** No
generator-level truth matching was performed in this package (none was
authorized or available read-only); the pT/probB ranges above are
compatible with hard non-b-like radiation but are not, by themselves,
evidence of any specific truth-level origin.

## Part D — what drives the ≥0.9997 result?

**Distinct from the 9-bin working-point significance decomposition.**
The already-corrected 9-bin accounting (`track_b_harvey_score_tail_diagnostics_20260829_v3`)
decomposes the *fine-scan working-point* significance (Z_A at fixed
target signal efficiencies) into score shells. That is a different
quantity from the population examined here, which is a **fixed literal
score cut** (≥0.9997), not tied to any signal-efficiency target. The two
should not be conflated; this package does not recompute or restate the
9-bin Z_A decomposition.

**Answering Harvey's question directly, using A.2–A.3 above:**

- **Not one or a few isolated events.** At ≥0.9997, n=100 raw events
  (N_eff=81.0, tier **FINITE_SUPPORT_CAUTION**) — a real, if thin,
  population, not a single-event artifact.
- **It does get MC-limited quickly as the cut tightens.** By ≥0.99997,
  n drops to 15 (N_eff=14.0, tier **EXTREMELY_LIMITED**); by ≥0.99999,
  n=9 (N_eff=9.0, same tier). The support-tier degradation with
  tightening cut is itself the headline finding, not any single event.
- **Concentrated in one process, yes: QCD.** QCD supplies 61% of raw
  events but **83.8% of the weighted yield** at ≥0.9997 (rising to 80%
  of raw events by ≥0.99997, 100% of raw events by ≥0.99999). This
  concentration is real and reproducible, not an artifact of one
  extreme-weight event — QCD's weight (7.31) is in fact the **highest**
  of the contributing processes (TW's is 6.23, SingleTop's is 4.88), so
  its dominance of the weighted yield is not surprising by itself; the
  point is that this is *compounded* by QCD also supplying the most raw
  MC events (61/100), not that a single outlier weight is doing the
  work alone.
- **Not concentrated in one identifiable kinematic mode.** Part B shows
  weak, inconsistent correlations and an R_HH range spanning genuinely
  Higgs-mass-consistent (4.7 GeV) to very far from it (1,626 GeV). No
  single topology explains the population.

**Explicitly not claimed:** `s_i/b_i` or `log(1+s_i/b_i)` values are
**not** presented anywhere in this package as a per-event posterior or
Matrix-Element-Method probability — they are not computed here at all;
this package works entirely in raw/weighted event counts.

## Part E — 75-epoch result (two sentences, per instruction)

See `PART_E_75EPOCH_TWO_SENTENCES.md` for the exact text used in the
email draft; full technical detail is in the existing, unmodified
`75EPOCH_READONLY_AUDIT.md` (read-only, not reproduced here).

## Part F — background-statistics update (from v5, not v4)

See `PART_F_BACKGROUND_STATS_FROM_V5.md`. No new computation — this
section only restates already-corrected v5 numbers for inclusion in the
Harvey-facing summary.

## Part G — does exact ≥0.997 require new scoring?

See `EAF_0P997_SCORING_PREFLIGHT.md`, `EAF_0P997_SCORING_COMMANDS.sh`
(prepared, **not executed**), and `EAF_0P997_SCORING_RECEIPT.json`.

**Proof, in one line:** the frozen survivor archives are complete only
for score ≥ 0.998958945274353 (the 7.5%-efficiency threshold); 0.997 <
0.998958945274353, so events with `0.997 ≤ score < 0.998958945274353`
were never retained anywhere (QCD scores in that range were counted
once and discarded; non-QCD events in that range were never selected as
survivors at any frozen target). **EXACT_0P997_REQUIRES_NEW_SCORING = YES.**
