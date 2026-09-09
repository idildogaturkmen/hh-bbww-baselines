# Tail-isolation / preselection variables and generation-level applicability

**Scope.** Read-only analysis of already-frozen event-level data. No Condor/
EAF job launched, no new scoring, no frozen physics result modified.

## 0. Data used and its limitation (stated up front)

The literal SPA-Net-10M tails (u>3.5: 68 QCD events; u>4.5: 12 QCD events,
see `HARVEY_TAIL_STATS_CURRENT.csv`) are **too small, and have no companion
generator-level branches, to fit or cross-validate a preselection model
directly.** No per-event SPA-Net-10M score exists for any bulk/non-survivor
QCD population (only survivors above the 7.5% WP are retained anywhere).

This package instead uses the **matched-400k development cohort**
(`matched_400k_features.npz`, SHA256 `9588d0fe79e32d455467c719eaaa57541fce
ad9835b8c32bebb9d68b96a2d30`, 174,485 real QCD events with full 82-column
kinematics + a genuine **SPA-Net-2M** per-event score, `topology_full.npz`
for HT/mHH/R_HH/pT_H1,2/DeltaR_bb/m_bb1,2) as the ranking substrate. SPA-Net
2M and 10M have near-identical background AUC (all-background AUC changes
by -0.000009 between them, `PROJECT_STATUS_20260821.md`), so SPA-Net-2M's
own extreme tail is used as a **statistically tractable proxy** for
"SPA-Net-10M-tail-like." In this cohort, only **3 QCD events** exceed
u_2M>3.5 and **0** exceed u_2M>4.5 (out of 174,485) -- too few for any
cross-validated statistic on their own, so the ranking below instead uses
proxy tail definitions at the **top 1%, 0.5%, and 0.1%** of the QCD score
distribution (n_pos = 1,745 / 873 / 175), with the literal 3-event u>3.5
population shown separately, descriptively only. Direction and ranking
should transfer reasonably to the true extreme tail (the top-0.1% proxy's
dominant variable, b-tag mistag rate, is the *same* variable independently
identified, from the real SPA-Net-10M tail itself, in Part D of
`track_b_harvey_spanet_compressed_tail_20260903_v2/HARVEY_COMPRESSED_TAIL_
REPORT.md` -- QCD supplies 61% of raw events but 83.8% of weighted yield at
the real >=0.9997 cut, and is the highest-weight process precisely because
its rate of faking a well-tagged 4b final state is what drives it there).
**Precise numerical thresholds should not be over-trusted at n<20.**

## 1. Univariate ranking (top-0.1% QCD score proxy, n_pos=175; full table for
all three proxy fractions in `HARVEY_PRESELECTION_FEATURE_RANKING.csv`)

| rank | variable | AUC | direction | generation-level availability |
|---:|---|---:|---|---|
| 1 | probB_leading4_mean / sum probB (leading 4) | 0.988 | higher -> more tail-like | **B** (post-Delphes b-tag) |
| 2 | probL_leading4_mean | 0.977 | lower -> more tail-like | **B** |
| 3 | min probB (leading 4, "weakest-tagged jet") | 0.875 | higher -> more tail-like | **B** |
| 4 | HT / mHH | 0.830 | higher -> more tail-like | **C** (needs mHH) |
| 5 | max deltaR(bb) | 0.758 | lower -> more tail-like | **C** |
| 6 | max probB (leading 4) | 0.744 | higher -> more tail-like | **B** |
| 7 | pT(H2) (2nd dijet pair pT) | 0.702 | higher -> more tail-like | **C** |
| 8 | 3rd-jet pT | 0.697 | higher -> more tail-like | **B** |
| 9 | pT(H1) | 0.691 | higher -> more tail-like | **C** |
| 10 | min deltaR(bb) | 0.674 | lower -> more tail-like | **C** |
| 11 | probC_leading4_mean | 0.674 | lower -> more tail-like | **B** |
| 12 | 2nd-jet pT | 0.671 | higher -> more tail-like | **B** |
| 13 | R_HH | 0.666 | lower -> more tail-like | **C** |
| 14 | 4th-jet pT | 0.660 | higher -> more tail-like | **B** |
| 15 | HT (reco) | 0.658 | higher -> more tail-like | **B** |
| 16 | mHH | 0.625 | lower -> more tail-like (weak, non-monotonic across fractions) | **C** |
| 17 | 5th-jet pT | 0.618 | higher -> more tail-like | **B** |
| 18 | leading-jet pT | 0.579 | higher -> more tail-like | **B** |
| 19 | mass asymmetry | 0.566 | lower -> more tail-like | **C** |
| 20 | n_selected_jets | 0.526 | ~flat (weakest of all variables tested) | **B** |

**Headline finding: b-tag mistag rate dominates everything else by a wide
margin** (AUC 0.87-0.99 vs 0.53-0.83 for every kinematic variable). A QCD
event only reaches the SPA-Net tail if, in addition to looking kinematically
plausible, it also **simultaneously mistags multiple light/gluon jets as
b-jets** -- this is confirmed independently from the real >=0.9997
SPA-Net-10M tail itself (Part C of `HARVEY_COMPRESSED_TAIL_REPORT.md`: the
unassigned/extra jets in tail QCD events are uniformly **low**-probB, i.e.
the four *assigned* jets are the ones carrying the anomalously high tags).

**The 3 literal u_2M>3.5 QCD events (n=3, purely descriptive):** all three
have HT in [662,764] GeV (cohort median 413), mHH in [688,811] GeV (median
633), **R_HH in [19,49]** (median 107 -- markedly more compact/Higgs-like),
deltaR(bb) both pairs well below median, and min-probB(leading 4) of
0.99/0.63/1.00 (2 of 3 have *all four* leading jets tagged b-like at
>=0.93). This is fully consistent with the univariate ranking above.

## 2. Simple cut scans (QCD only, top-0.1% proxy tail, n_pos=175 of 174,485)

| cut | bulk-QCD pass fraction | tail efficiency | bulk rejection | tail enrichment factor |
|---|---:|---:|---:|---:|
| min-probB(lead.4) > 0.50 | 0.209% | 45.7% | 99.79% | **219x** |
| min-probB(lead.4) > 0.70 | 0.156% | 38.9% | 99.84% | 248x |
| min-probB(lead.4) > 0.90 | 0.105% | 32.0% | 99.90% | 305x |
| min-probB(lead.4) > 0.99 | 0.033% | 13.1% | 99.97% | 395x |
| HT > 400 GeV | 55.9% | 75.4% | 44.1% | 1.35x |
| HT > 500 GeV | 24.3% | 44.6% | 75.7% | 1.8x |
| HT > 600 GeV | 11.2% | 29.7% | 88.8% | 2.6x |
| HT > 700 GeV | 5.5% | 15.4% | 94.5% | 2.8x |
| min-probB>0.90 AND HT>500 | 0.025% | 10.3% | 99.98% | **417x** |
| min-probB>0.50 AND HT>400 | 0.109% | 29.7% | 99.89% | 272x |

**Pairwise combination adds real value over either variable alone**
(417x vs 305x/1.8x individually), because b-tag mistag rate and HT are only
weakly correlated (the two variables carry complementary information).

## 3. Compact interpretable models (4 features, 5-fold cross-validated)

Feature set: `{probB_leading4_mean, min-probB(lead.4), HT/mHH, max-deltaR(bb)}`.

| tail proxy | n_pos | shallow tree (depth<=3) CV AUC | logistic regression CV AUC |
|---|---:|---:|---:|
| top 1% | 1,745 | 0.934 +/- 0.010 | 0.973 +/- 0.004 |
| top 0.5% | 873 | 0.939 +/- 0.009 | 0.983 +/- 0.003 |
| top 0.1% | 175 | 0.964 +/- 0.016 | **0.994 +/- 0.002** |

The fitted depth-3 tree's dominant split is `probB_leading4_mean <= 0.78`
(everything below this is bulk); within the surviving branch, the
`HT/mHH > 0.85` split further separates tail-like events. This is a simple,
auditable rule that recovers most of the discrimination in the full ranking.
Full tree text and code: `work/rank2.py` output archived in this package's
receipt.

## 4. Be cautious at u>4.5

At the literal u_2M>4.5 level, **zero** QCD events exist anywhere in the
174,485-event matched cohort (max observed u_2M = 3.77). The real SPA-Net-10M
u>4.5 population is only 12 QCD events total (`HARVEY_TAIL_STATS_CURRENT.csv`).
**No preselection variable ranking can be cross-validated at this population
size with any real cohort available in this project.** The `STRATUM_TAIL_
TABLE.csv` conditional-efficiency analysis in the existing
`track_b_qcd_tail_generation_strategy_20260902_v1` package (built directly
from the real, if thin, SPA-Net-10M tail) is the best available evidence at
this extremity, and it is explicitly hedged there for the same reason (its
own >=1000 GeV HT stratum's 3.125% efficiency estimate rests on 96 pilot
events, 3 of which are the entire population above 1000 GeV).

## 5. Generation-level applicability matrix

| variable | availability | notes |
|---|---|---|
| **pTHat** (true generator hard-process scale) | **A -- generator level** | Not stored per-event in the frozen ntuples, but this is exactly what native Pythia8 `PhaseSpace:bias2Selection`/`bias2SelectionPow` biases on directly, with an exact compensating weight. Structurally applicable here because the governing QCD sample is confirmed **direct Pythia**, not MG5/LHE (`CURRENT_QCD_GENERATION_AUDIT.md`). This is the single most promising **early** lever: it can reduce generation cost *before* any Delphes simulation or scoring is run. |
| inclusive generator-level HT (parton-level, `iht`) | **A -- generator level** | Already used successfully as an importance-sampling variable, but only in the *separate*, locally-runnable MG5 `pp->bbbb` chain (`iht100to200`...`iht600plus` slices) -- **not the governing Pythia HardQCD sample**, and demonstrably shifts the reconstructed R_HH/avg-mbb shape away from the signal region when used alone (`qcd_iht_closure_metrics_2026_07_08.csv`: R_HH<30 under-populated by -16.8% relative to inclusive). Relevant precedent, not directly transferable to sample (A). |
| reconstructed HT | **B -- after Delphes, before HH-pairing** | Weak-moderate discriminator alone (AUC 0.63-0.66; stratified conditional-tail-efficiency gradient is stronger, 4-14x from 500-700 GeV to >=1000 GeV per `STRATUM_TAIL_TABLE.csv`, but only *conditional on already being in the 7.5% WP*). The best available **reconstructed proxy** for the generator-level pTHat lever above -- cannot itself be cut on before generation, but justifies *where* to bias pTHat. |
| n_selected_jets | **B -- after Delphes** | Weak alone in the direct ranking (AUC 0.53) but a real conditional gradient exists in the tail-survivor stratum table (2.2% at 4 jets -> 15.9% at 6 jets). A rough generator-level proxy (final-state parton multiplicity before showering) exists and is exactly what Design C (`THREE_DESIGNS_COMPARISON.md`) proposes combining with pTHat in a custom bias hook -- not yet built. |
| leading/subleading/3rd/4th/5th jet pT | **B -- after Delphes** | Individually weak-moderate (AUC 0.58-0.70); correlated with, but a noisier proxy for, pTHat than reconstructed HT itself. |
| b-tag probabilities (probB/C/L) | **B -- after Delphes (Sophon tagger output)** | **The single strongest discriminator found (AUC up to 0.99)**, but fundamentally a detector/tagging-level quantity -- **cannot** be used to bias or filter events before Delphes+tagging are run, and therefore **cannot reduce generation cost**, only analysis-level scoring/storage cost downstream. |
| mHH, R_HH, pT(H1)/pT(H2), DeltaR(bb), m_bb1/m_bb2, mass asymmetry | **C -- only after HH-pairing/reconstruction** | Requires jet-pairing/Higgs-candidate assignment (minimal-R_HH combinatorics or the SPA-Net assignment head itself); moderate discriminators (AUC 0.63-0.83 for the best of these, HT/mHH); cannot be used before generation or even before the full reconstruction step. |
| SPA-Net score / u | excluded by task instruction | The quantity being predicted, not a candidate preselection variable. |

**Most promising early-generation lever:** native Pythia8 `PhaseSpace:
bias2Selection` biasing on `pTHat`, exactly as already proposed (not yet run)
in `track_b_qcd_tail_generation_strategy_20260902_v1/HARVEY_BACKGROUND_
PRODUCTION_SUMMARY.md` ("Design B", recommended there). It is structurally
compatible with the governing chain (confirmed direct-Pythia), requires no
new code (native run-card flag), and concentrates new raw statistics in the
region `STRATUM_TAIL_TABLE.csv` shows has the highest tail-conditional
efficiency, while its exact compensating weight (`Info::weight()`) keeps the
inclusive cross-section unbiased by construction. **This is the only
variable in the table above that is genuinely available early enough
(category A) to reduce generation cost itself**, rather than only analysis
cost.

## 6. Candidate preselections, generation-efficiency-gain framing

| candidate | category | tail eff. | bulk rejection | enrichment | reduces generation cost? |
|---|---|---:|---:|---:|---|
| min-probB(lead.4)>0.5 AND HT>400 | B (analysis-level only) | 29.7% | 99.89% | 272x | **No** -- both inputs require Delphes+tagging to already have run |
| min-probB(lead.4)>0.9 AND HT>500 | B | 10.3% | 99.98% | 417x | **No**, same reason; best pure analysis-level cut found |
| reconstructed HT > 500-1000 GeV region (motivates pTHat bias) | B, motivates A | -- | -- | 1.8-2.6x locally; up to ~4x in the >=1000 GeV stratum conditional on 7.5% WP | **No by itself**, but directly informs `bias2SelectionRef` choice for the pTHat-level bias (A) that *does* reduce generation cost |
| Pythia `bias2Selection` on pTHat, `Ref`~500 GeV, `Pow`~4 (per `CANARY_PRODUCTION_PLAN.md`, not yet run) | **A** | not yet measured (canary required) | -- | expected several-x, per Part 2 stratum evidence | **Yes** -- the only candidate here that acts before Delphes/scoring |

**Bottom line:** the strongest *statistical* preselection (b-tag mistag
rate) cannot cut generation cost because it is only knowable after the full
detector simulation and tagging chain runs. The only lever that can reduce
*generation* cost is a generator-level bias on `pTHat` (category A),
already designed (Design B) but explicitly **not yet run** anywhere in this
project (`NEW_JOB_LAUNCHED = NO` in the source package's own receipt, and
unchanged here).
