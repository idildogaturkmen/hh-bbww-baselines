# Physical-background tail statistics, MC scaling, and next-step recommendation

**Scope and safety.** Read-only reconciliation of already-frozen artifacts
in this repository and the Track-B analysis workspace. No SPA-Net training,
no ParT, no EAF/GPU, no `holdout_B`/Stage-C access, no new MC generation, no
Condor production launch, no package installation. Every existing package
referenced below is unmodified.

**Relationship to the separate FP32/logit precision audit (Track A).**
Track A established that the score comb visible near 1.0 is a float32
representation artifact of the two-class softmax and that the raw logit
margin remains continuous through it. **This document does not use, repeat,
or depend on that finding.** It answers a different question entirely: how
much real, physical Monte Carlo statistical support underlies the QCD/
ttbar/minor-background tail, independent of how the score happens to be
stored. The **400k development cohort** used for the FP32/logit numerical
audit is explicitly **not** used anywhere in this document — every number
here comes from the much larger physical-background production
(`inference_qcd_1`+`inference_qcd_2`, 87,623,306 QCD events; the 16-process,
non-QCD inference population) that underlies the frozen fine-scan and
background-statistics packages.

## 0. Score/u semantics used throughout

`score` = SPA-Net 10M checkpoint `fd9ea100825ecc0412a2666f01fd19d1a91b25a5
0109b958c307102fa6943de5`'s class-1 softmax output. `u = -log10(1-score)`.
Four distinct threshold definitions are used below, and their exact
relationships are stated explicitly rather than assumed:

| threshold | exact meaning | population identity |
|---|---|---|
| `u > 3.5` | `score > 1 - 10^-3.5 = 0.9996837722339832` (strict `>`) | its own, slightly looser than the next row |
| `score >= 0.9997` | `u >= 3.522878745...` | **tighter than** `u>3.5` by construction |
| `u > 4.5` | `score > 1 - 10^-4.5 = 0.9999683772233983` (strict `>`) | **identical population** to the next row (verified, Sec.1) |
| `score >= 0.99997` | `u >= 4.522878745...` | **identical population** to `u>4.5` -- zero events fall in the sliver between them |

`u>3.5` and `score>=0.9997` are **not the same population** (they differ by
10 background events, Sec.1); `u>4.5` and `score>=0.99997` **are** the same
population (zero events lie in `[0.9999683772233983, 0.99997)`), confirmed
directly, not assumed, from the frozen per-event counts.

## 1. Reconciled governing physical tail counts (exact, from frozen production data)

Source: `FULL_QCD_SURVIVOR_SIDECAR.h5` (QCD, complete coverage above the
SPA10M 7.5%-efficiency boundary `score>=0.998958945274353`) and the frozen
non-QCD `job_summary_*.json` per-event survivor lists (same coverage
boundary), both from `track_b_harvey_tail_characterization_20260825_v1` /
`track_b_harvey_complete_3to5pct_study_20260825_v1`; cross-checked against
the independently-published `track_b_harvey_spanet_compressed_tail_
20260903_v2/HARVEY_COMPRESSED_TAIL_REPORT.md` and the fine-scan package
`track_b_harvey_followup_finescan_likelihood_20260829_v1`. All four
populations below were **independently recomputed for this package**
(Sec.8), not merely copied.

### u > 3.5 (score > 0.9996837722339832)

| process | raw N | weight (450 fb-1) | B_450 | sum w^2 | N_eff | rel. MC unc. | frac. of B |
|---|---:|---:|---:|---:|---:|---:|---:|
| QCD | 68 | 7.3112875413 | 497.168 | 3634.935 | 68.00 | 12.13% | 84.4% |
| ttbar | 27 | 2.3214135412 | 62.678 | 145.502 | 27.00 | 19.25% | 10.6% |
| other (14 minor) | 15 | (mixed) | 29.097 | 103.360 | 8.19 | 34.94% | 4.9% |
| **all background** | **110** | -- | **588.943** | **3883.797** | **89.31** | **10.58%** | 100% |

Signal S at exactly `u>3.5` is not on any pre-tabulated grid point (signal is
never MC-limited here); bracketing frozen fine-scan values are
S_450=66.089 (at u=3.4246, eps_S=5.5%) and S_450=60.084 (at u=3.7521,
eps_S=5.0%) -- S is not the limiting quantity at this threshold.

### score >= 0.9997 (a tighter cut than u>3.5)

| process | raw N | B_450 | sum w^2 | N_eff | rel. MC unc. | frac. of B |
|---|---:|---:|---:|---:|---:|---:|
| QCD | 61 | 445.99 | 3260.99 | -- | -- | 83.8% |
| ttbar (both lanes) | 25 | 58.03 | 134.72 | -- | -- | 10.9% |
| other (5 of 14 minor) | 14 | 28.34 | 102.78 | -- | -- | 5.3% |
| **all background** | **100** | **532.36** | **3498.49** | **81.01** | **11.11%** | 100% |

Signal: `score>=0.9997` -> n=27,128/508,168 (5.338%) ->
**S_450 = 64.143**.

### u > 4.5 (score > 0.9999683772233983) = score >= 0.99997 (identical population, verified)

| process | raw N | weight (450 fb-1) | B_450 | sum w^2 | N_eff | rel. MC unc. | frac. of B |
|---|---:|---:|---:|---:|---:|---:|---:|
| QCD | 12 | 7.3112875413 | 87.736 | 641.459 | 12.00 | 28.87% | 90.2% |
| ttbar | 2 | 2.3214135412 | 4.643 | 10.778 | 2.00 | 70.71% | 4.8% |
| other (SingleTop only) | 1 | 4.8846004265 | 4.885 | 23.859 | 1.00 | 100.00% | 5.0% |
| **all background** | **15** | -- | **97.263** | **676.096** | **13.99** | **26.73%** | 100% |

Signal: `score>=0.99997` -> n=17,869/508,168 (3.516%) -> **S_450 = 42.251**.

## 2. Finite-MC uncertainty, naive vs. correct, at the existing governing ~4% WP and at u>3.5 / u>4.5

The project's existing governing fine-scan working point closest to "the
~4% signal-efficiency region" is `eps_S = 4.0%` (achieved 4.0024%,
threshold `score=0.9999514818191528`, `u=4.3141`):

| quantity | value | source |
|---|---:|---|
| N_raw background | 23 (17 QCD + 4 ttbar + 1 ZJetsToQQ + 1 SingleTop) | `spa10m_finescan_rows.json` |
| N_eff (correct) | **20.4618** | `track_b_harvey_score_tail_diagnostics_20260829_v5_background_stats_audit` |
| B (450 fb-1) | 139.872454 | same |
| sigma_MC = sqrt(sum w^2) | 30.92 | same |
| sigma_MC / B (**correct**, via N_eff) | **22.11%** | same |
| naive 1/sqrt(N_raw=23) | 20.85% | v3 (superseded number, kept only as the "naive" comparison) |

**The naive treatment understates the true uncertainty by about 6% relative
(1.26 points absolute) at this WP**, because N_eff < N_raw whenever
contributing processes carry heterogeneous weights (0.34-7.31 across the 4
processes present) -- confirmed directly from the frozen per-event weights,
not assumed. QCD supplies 88.9% of B but 95.0% of `sum w^2` at this WP,
which is why the naive approximation, while still wrong, is not wildly off:
it degrades further as looser WPs let more heterogeneous minor backgrounds
contribute.

At `u>3.5` (n=110, correct N_eff=89.31, correct rel.unc.=10.58% vs naive
1/sqrt(110)=9.53%) and at `u>4.5`/`score>=0.99997` (n=15, correct
N_eff=13.99, correct rel.unc.=26.73% vs naive 1/sqrt(15)=25.82%), the same
qualitative gap holds, most pronounced (largest naive-vs-correct gap in
relative terms) at the loosest of the four thresholds examined and
essentially negligible at `u>4.5` (QCD is 90.2% of B and its own N_eff there
already equals its raw count exactly, since it is the only process with
more than one contributing event of comparable weight).

**Support-tier labels** (contract: N>=1000 PRIMARY_ADEQUATE, 100-999
FINITE_SUPPORT_CAUTION, 20-99 DIAGNOSTIC/B95-ORIENTED, 1-19
EXTREMELY_LIMITED, 0 upper-limit-only; reused unchanged from `common.py`
across this project):

| threshold | N_raw (all-bkg) | support tier |
|---|---:|---|
| eps_S=4.0% WP | 23 | LIMITED (`support_tier_overall` as literally stored in `spa10m_finescan_rows.json`; the 20-99 band, elsewhere in this project also called `DIAGNOSTIC_B95_ORIENTED`) |
| u>3.5 | 110 | FINITE_SUPPORT_CAUTION |
| score>=0.9997 | 100 | FINITE_SUPPORT_CAUTION (boundary case) |
| u>4.5 / score>=0.99997 | 15 | EXTREMELY_LIMITED -- **descriptive, not a precision claim** |

## 3. QCD statistics scaling: 2x / 4x / 10x (physical yield held fixed)

**Assumption, stated explicitly:** additional QCD MC is generated at the
*same* cross-section/luminosity/selection as the existing sample, i.e.
future QCD events are assumed statistically equivalent (same flat weight
convention) to the existing ones -- no new kinematic regime, no importance
sampling. Under this assumption, generating N x more QCD MC does **not**
increase the physical B_QCD: the per-event weight scales down by 1/N while
the raw count scales up by N, holding B_QCD (and the combined B) exactly
fixed while only the precision (`N_eff`, `sum w^2`) improves. This is the
same corrected model established in
`track_b_harvey_score_tail_diagnostics_20260829_v5_background_stats_audit`
(v4's model, which incorrectly let B grow with the raw count, is superseded
and not used here).

| population | 1x N_eff / rel.unc. | 2x | 4x | 10x |
|---|---|---|---|---|
| eps_S=4.0% WP (B=139.87, fixed) | 20.46 / 22.11% | 38.99 / 16.01% | 71.25 / 11.85% | 141.49 / 8.41% |
| u>3.5 (B=588.94, fixed) | 89.31 / 10.58% | 167.86 / 7.72% | 299.63 / 5.78% | 566.42 / 4.20% |
| score>=0.9997 (B=532.36, fixed) | 81.01 / 11.11% | 151.73 / 8.12% | 269.22 / 6.10% | 502.87 / 4.46% |
| u>4.5 = score>=0.99997 (B=97.26, fixed) | 13.99 / 26.73% | 26.62 / 19.38% | 48.51 / 14.36% | 95.77 / 10.22% |

(1x row of each population independently reproduces the frozen v5-audit /
compressed-tail-report values to the digit -- see Sec.8.)

Figures: `figures/neff_vs_qcd_multiplier.png`, `figures/relunc_vs_qcd_
multiplier.png` (all four populations plotted together).

**QCD's contribution to variance** dominates every population above (84-95%
of `sum w^2` depending on threshold, Sec.1 tables) -- this is *why*
QCD-only scaling is effective: the minor-background floor (`sum w^2` from
ttbar+other, held fixed because those samples are not being regenerated)
is small relative to QCD's own contribution at every threshold studied, so
N_eff improves substantially (though sub-linearly, since the fixed floor
does not vanish) as QCD statistics grow. At looser thresholds (eps_S=4.0%
specifically) the floor is a slightly larger fraction (5.0% of `sum w^2`)
than at `u>3.5` (1.7%), which is why the eps_S=4.0% row's N_eff improvement
saturates slightly sooner in relative terms (10x gives 6.9x the 1x N_eff,
vs 6.3x for u>3.5).

**Expected reduction relative to current uncertainty:**

| population | 2x reduction | 4x reduction | 10x reduction |
|---|---:|---:|---:|
| eps_S=4.0% WP | 27.5% | 46.4% | 62.0% |
| u>3.5 | 27.1% | 45.4% | 60.3% |
| score>=0.9997 | 26.9% | 45.1% | 59.9% |
| u>4.5/0.99997 | 27.5% | 46.3% | 61.8% |

(reduction = `1 - rel.unc.(Nx)/rel.unc.(1x)`; consistently ~27% at 2x,
~46% at 4x, ~60-62% at 10x across all four populations -- this consistency
is itself evidence the calculation is dominated by the same QCD-scaling
mechanism at every threshold, not a coincidence of any one population's
particular composition.)

## 4. Compute / wall-clock resource estimate (measured receipts only)

**Generation chain, confirmed:** the governing QCD sample is direct Pythia8
HardQCD-style multijet, single inclusive `pTHat>75 GeV` slice -- **not**
MG5/LHE (`CURRENT_QCD_GENERATION_AUDIT.md`, verified via `gen_weight=1.0`
and absence of LHE multi-weight structure). There is no MG5 stage in this
chain.

**Measured rate** (the only real Pythia8+Delphes QCD generation receipt in
this project): `track_a_targeted_qcd_pilot_arm1_200to500_50k_20260814_v1` --
1.3517 allocated-core-hours (1.2461 CPU-busy-hours) for 50,000 events, 6
concurrent `request_cpus=1` Condor jobs. Rate = 2.7034e-5 CPU-slot-hr/event
(allocated-wall basis) / 2.4922e-5 (CPU-busy basis). **Caveat, carried
forward from this project's own prior estimate and not softened here:**
this pilot used a different `pTHat` slice (200-500 GeV) than the governing
sample's inclusive `pTHat>75 GeV` production, and the real IHEP/author
chain's cost could differ materially -- this is a scale proxy, not an
author-chain benchmark.

Current QCD generation total (author-confirmed): 278,360,000,000 events
(`AUTHOR_QCD_NORMALIZATION_AND_DATASET_USAGE_CLARIFICATION.md`); current
selected/inference total: 87,623,306.

| multiplier | additional generated events | additional selected events | CPU-slot-hours (allocated-wall) | CPU-slot-days |
|---:|---:|---:|---:|---:|
| 2x | 278,360,000,000 | 87,623,306 | 7,525,184 | 313,549 |
| 4x | 835,080,000,000 | 262,869,918 | 22,575,553 | 940,648 |
| 10x | 2,505,240,000,000 | 788,609,754 | 67,726,658 | 2,821,944 |

**Cross-check:** the 4x and 10x figures agree with this project's own
independently-published `TENFOLD_BACKGROUND_FEASIBILITY_MEMO.md` (~22.5M
and ~67.6M CPU-hours respectively) to within 0.4%, since both use the same
underlying pilot rate.

**These are single-CPU-*slot*-hours, not wall time and not independently
verified physical-core-hours** (whether one HTCondor slot maps to one
physical core was never verified at LPC). Illustrative wall time at stated
concurrency (division only, not a resource commitment):

| multiplier | @100 slots | @500 slots | @1000 slots | @5000 slots |
|---:|---|---|---|---|
| 2x | 3,135.5 days (8.58 yr) | 627.1 days (1.72 yr) | 313.5 days (0.86 yr) | 62.7 days (0.17 yr) |
| 4x | 9,406.5 days (25.75 yr) | 1,881.3 days (5.15 yr) | 940.6 days (2.58 yr) | 188.1 days (0.52 yr) |
| 10x | 28,219.4 days (77.26 yr) | 5,643.9 days (15.45 yr) | 2,821.9 days (7.73 yr) | 564.4 days (1.55 yr) |

None of these concurrency levels has been sustained by a real QCD-specific
Condor campaign in this project (the largest QCD-generation-specific
submission observed is 150 jobs, `qcd_hardqcd_importance_adaptive1500k_
phys_wave3_20260718`); they are illustrative divisions, exactly as used in
this project's own prior audits, not resource commitments.

**Scoring cost, separated from generation:**

- **BDT-KF (CPU, measured):** rate = 788,609,754 rows / 2.16 h =
  365,097,108 rows/hour (`TENFOLD_BACKGROUND_FEASIBILITY_MEMO.md` Route 3,
  "No GPU is required"). At this rate: 2x=0.24h, 4x=0.72h, 10x=2.16h.
- **SPA-Net 10M inference-only throughput: UNAVAILABLE.** No inference-only
  (as opposed to training) throughput measurement exists anywhere in this
  project (`CPU_GPU_RESOURCE_TABLE.csv` flags this explicitly). Marked
  unavailable per instruction, not estimated.

## 5. Is brute-force more QCD the best use of resources?

The kinematic findings below (already established in prior, unmodified
packages) are used **only as motivation for what generator-level quantity
might help** -- not as a license to design a cut on the SPA-Net score
itself, and not as evidence that any targeted scheme is already validated.

### A. Brute-force inclusive QCD scaling

- **Pro:** zero new methodology risk -- same flat-weight convention, same
  closure trivially satisfied (it *is* the existing sample, just more of
  it), immediately usable in every existing downstream calculation.
- **Con:** as Sec.4 shows, reaching even a modest (4x) precision
  improvement costs ~22.6M CPU-slot-hours -- years at any concurrency this
  project has actually sustained for QCD generation. Nearly all of that
  compute is spent on phase space far below the classifier's tail (the
  current sample's own `score>=0.9997`-passing fraction is ~1 in 900,000).

### B. Targeted / importance-sampled QCD generation

**Not presented as solved.** Any targeted scheme changes the sampling
density of generated events relative to the true cross-section-weighted
distribution, which **requires an exact compensating per-event weight and a
demonstrated closure test** before its output may enter any physics result
-- this is a design requirement, not yet met by anything run in this
project (`track_b_qcd_tail_generation_strategy_20260902_v1/CANARY_
PRODUCTION_PLAN.md`: prepared, not executed).

| candidate generator-level lever | why it might enrich the tail | physics bias it could create | correction/reweighting required | mandatory closure test |
|---|---|---|---|---|
| Native Pythia8 `PhaseSpace:bias2Selection` biased on **pTHat** (the true generator hard-process scale) | Reconstructed-HT-conditional tail efficiency rises 4-14x from 500-700 GeV to >=1000 GeV, *conditional on already passing the 7.5% WP* (`STRATUM_TAIL_TABLE.csv`); pTHat is the generator-level quantity that most directly drives reconstructed HT, and the governing sample is confirmed direct-Pythia (structurally compatible, not LHE-restricted) | Oversamples high-pTHat phase space; if the compensating weight is dropped or miscoded anywhere downstream, silently reintroduces exactly the bias the method is meant to remove | Every event's own `Info::weight()` must be retained and propagated through every downstream `N_eff`/`sum w^2`/significance calculation (currently all of them assume QCD's flat weight) | Closure of the *unbiased control run's* HT/jet spectrum against the existing inclusive sample; closure of the *biased run's* `sum Info::weight()` against the same inclusive cross-section, within MC uncertainty; a measurable, correctly-directed shift in raw event density toward high HT/pTHat |
| A new, separate high-`pTHatMin` inclusive slice, stitched by cross section | Same STRATUM_TAIL_TABLE motivation, simpler mechanism (flat weight preserved within the new slice) | Double-counts the `pTHat>pTHatMin` phase space already inside the existing inclusive sample unless explicitly re-sliced/excluded | A rigorous accounting of which phase space belongs to which sample (no true `pTHat` branch exists in the existing sample to do this precisely) | Closure of the inclusive spectrum in the well-populated overlap region; explicit non-overlap bookkeeping |
| Reconstructed HT / jet multiplicity itself as a generation-time bias | **Explicitly excluded.** These are post-Delphes, post-reconstruction quantities; biasing generation on them is not physically implementable (the hard-process event does not yet have a reconstructed HT when Pythia's `biasSelectionBy` hook fires) and would risk sculpting the very distribution the tail statistics are meant to measure | -- | -- | -- |

**Concrete recommendation (not launched):** run the already-specified,
smallest mechanism-validation canary -- 10,000 unbiased control + 10,000
`bias2Selection`-biased events (`bias2SelectionPow=4`,
`bias2SelectionRef~500 GeV`), single-digit CPU-slot-hours, <1 GB storage
(`CANARY_PRODUCTION_PLAN.md`). **Before any full-scale campaign**, the
canary must demonstrate: (1) the unbiased run's weighted HT spectrum
matches the existing inclusive sample in the well-populated 300-700 GeV
region; (2) the biased run's `sum Info::weight()` reproduces the same
inclusive cross-section within MC uncertainty; (3) the biased run's *raw*
event density is measurably shifted toward high HT/pTHat while its
*weighted* spectrum still matches the inclusive sample; (4) no seed
collision against `training_qcd`/`inference_qcd_1`/`inference_qcd_2`.
**Any one failure stops the plan before scale-up** -- this document does
not claim the method already works.

## 6. Stability of the existing SPA-Net advantage

Reusing the **corrected** 9-bin score-space likelihood decomposition
(`track_b_harvey_score_tail_diagnostics_20260829_v3`, which fixed a real v1
bug where the terminal `score>=threshold(3.0%)` population's raw support
was folded incorrectly into a displayed bin, making it look like 15 events
instead of the true exclusive/terminal split of 3+12 -- the v2/v3-corrected
version is used here, not the superseded v1 numbers):

| bin | kind | raw n | support tier | % of full 9-bin Z_A^2 |
|---|---|---:|---|---:|
| >= eps_S=3.0% (terminal, cumulative) | terminal | 12 | EXTREMELY_LIMITED | **77.10%** |
| 3.5%-3.0% (exclusive shell) | transition | 3 | EXTREMELY_LIMITED | 12.43% |
| 4.0%-3.5% | transition | 8 | EXTREMELY_LIMITED | 4.47% |
| 4.5%-4.0% | transition | 11 | EXTREMELY_LIMITED | 3.48% |
| 5.0%-4.5% | transition | 21 | DIAGNOSTIC_B95_ORIENTED | 1.96% |
| 5.5%-5.0% and looser (3 bins) | transition | 115, 442, 572 | FINITE_SUPPORT_CAUTION | 0.65% combined |

**Not single-event dominated, but concentrated in a small population.** The
single largest contribution (77.1% of the total significance-squared) comes
from the terminal >=3.0%-efficiency population -- 12 raw events, not one.
The next-largest single piece (12.4%) is the exclusive 3.5%->3.0% shell (3
events: 1 QCD + 1 ttbar + 1 SingleTop). Combined, the top two pieces (89.5%
of the total) rest on only 15 raw events total (the same 15 as the `u>4.5`/
`score>=0.99997` population in Sec.1, reconciled exactly: 3+12=15).

**Process role:**
- **QCD**: dominates raw support and weighted B at every threshold examined
  (61-90% of raw events, 84-90% of weighted B depending on threshold,
  Sec.1) and supplies 84-95% of `sum w^2` (Sec.3) -- QCD statistics are the
  single lever that improves the combined uncertainty the most per unit of
  regenerated MC.
- **ttbar**: a real, non-negligible but always-subdominant contributor
  (5-19% of raw events, 5-11% of B) -- not regenerated in the QCD-only
  scaling scenario of Sec.3, so its fixed variance becomes an increasingly
  visible floor as QCD statistics grow.
- **Minor backgrounds (14 processes)**: individually and collectively
  small at every threshold (1-15% of B), with 8-11 of the 14 contributing
  zero observed events at the tightest thresholds studied.

**Where finite MC is the dominant limitation:** at every threshold examined
above `eps_S~5%` (equivalently `u>~3.7`), raw background support is below
100 events and support tiers are FINITE_SUPPORT_CAUTION or
EXTREMELY_LIMITED. The tightest populations (`u>4.5`, `score>=0.99997`, the
eps_S<=4.5% transition shells, and the terminal >=3.0% bin) are explicitly
**descriptive**, not precision claims -- any headline significance number
that leans on these populations should be reported with its N_eff/rel.unc.
attached, never as a bare central value.

## 7. Cross-checks performed (Sec.8 has the full list)

Every headline number in Sections 1-3 was independently recomputed in this
package's `work/` scripts from the same frozen inputs already used
elsewhere in this project, and separately cross-checked against the
already-published, independent values in
`track_b_harvey_spanet_compressed_tail_20260903_v2`,
`track_b_harvey_score_tail_diagnostics_20260829_v3`/`v5`, and
`track_b_harvey_followup_finescan_likelihood_20260829_v1`. No discrepancy
was found; every match is to the digit or to floating-point precision.

`NEW_MC_GENERATION_LAUNCHED = NO`. `NEW_TRAINING_LAUNCHED = NO`.
