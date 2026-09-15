# Mechanism Results: Paper Draft (Track A + Track B)

Status: draft, read-only synthesis. Written 2026-08-17 from already-frozen,
checksum-verified Track-A and Track-B artifacts only (see
`SOURCE_PROVENANCE.tsv`). No model was trained, retrained, or evaluated to
produce this document or any figure/table in this package. `holdout_A` (Track
B) was not reopened -- it was already opened exactly once, by Phase-4M, on
2026-08-13. `holdout_B`, `holdout_Q`, `inference_qcd_1`, `inference_qcd_2`
were not opened. Track A's own validation/test partitions were not opened by
this document either (Track A's authoritative source,
`post_training_evaluation_20260815_v2`, itself never opened them).

An independent validation script (`scripts/independent_validate.py`)
re-derived every plotted/tabulated number from the raw frozen source files in
a fresh code path and cross-checked it against this package's own tables and
figures data. Result: 181/181 checks pass (`VALIDATION_REPORT.json`).

---

## 1. Measured result

### Track B (external `jetfree-hh4b` Delphes benchmark, `holdout_A`, one-time)

Three arms of a **CMS-inspired five-jet pairwise event Transformer**
(project-specific implementation; not SPA-Net, not canonical ParT, not an
official CMS reproduction) were compared, 3 seeds each, on `holdout_A`
(`tight_exact4` population, no mass-plane cut):

- **E0** (five-jet kinematics only) vs. **E1** (E0 + continuous reconstructed
  SophonAK4 `probB/probC/probL`): E1 **fully and reproducibly** exceeds E0.
  At every in-region working point (`eps_S` from 0.60 down to 0.20), all 3
  seeds agree in direction; seed-mean `R_B(E1)/R_B(E0)` ranges from **+23%**
  (`eps_S=0.60`) to **+62%** (`eps_S=0.25`). This matches the direction and
  approximate magnitude already seen on the internal development split
  (Phase-4L).
- **E1** vs. **E2** (E1 + the frozen, project-defined pairwise attention bias
  using `[m_ij, |Delta eta_ij|, |Delta phi_ij|, Delta R_ij]`): E2 does **not**
  improve on E1. At `eps_S >= 0.25`, all 3 seeds agree that `R_B(E2) <
  R_B(E1)` (a **degradation**, seed-mean ratio 0.56-0.69 across
  `eps_S in {0.60,...,0.25}`). At `eps_S=0.20`, 2 of 3 seeds still favor E1;
  one seed flips, at exactly the point where E2's own raw surviving-QCD count
  is 10 -- the boundary of the "limited but reportable" support tier. This is
  reported as a genuine discrepancy, not smoothed over (see `phase4M`'s own
  `HOLDOUT_A_ADJUDICATION_RESULT.md`).

See Figure 1 (`figures/trackb_holdoutA_mechanism_rejection.*`), Figure 2
(`figures/trackb_mechanism_gain.*`), Table A (`TRACKB_MECHANISM_TABLE.tex`),
Table D (`TRACKB_MECHANISM_GAIN_TABLE.tex`).

### Track A (Delphes simulation, this project's own TRAIN population)

Authoritative source: `post_training_evaluation_20260815_v2` (v1 is
preserved but **not** authoritative -- v2 documents and corrects v1's
arithmetic/labeling errors and passed its own independent validation,
`VALIDATION_REPORT.json`, 25/25 checks).

At `eps_S=0.40`, point-estimate ordering: cut baseline (`R_HH<34`, `R_B =
1.95`) $\ll$ BDT `CONTROL_0` (13.21) < dense DNN (13.29 $\pm$ 0.26) <
**CMS-inspired five-jet pairwise event Transformer** (14.00 $\pm$ 0.30) <
`FiveJetParTNet` (14.15, frozen reference, not retrained here; a late-fusion
jet-level Transformer, **not** canonical ParT) < BDT `NEW_C` (14.71). Only
the two newly-trained models (DNN, transformer) have a bootstrap confidence
interval computed anywhere in this package.

The paired (same-resampled-event-multiset) bootstrap comparison of the
transformer against the DNN, at every measured `eps_S`, shows a
**working-point-dependent** result, not a uniform win: the transformer is
favored (68% CI excludes zero) at `eps_S in {0.60, 0.585957, 0.50}`, favored
only at 68% (not 95%) at `eps_S=0.40`, a statistical coin flip at
`eps_S=0.25` (48.9%/51.0%), and **disfavored** at `eps_S=0.10` (64.1% of
replicates favor the DNN there, `fraction_replicates_dnn_gt_transformer =
0.641` in `EVALUATION_RESULTS.json`, matching that package's own Table 5 --
v2's `RESULTS_SUMMARY.md` prose paragraph rounds this to a value about half
a percentage point lower at this one spot, a minor prose-only transcription
slip against its own table and JSON in the same document; caught
independently here by `scripts/independent_validate.py`, not corrected in
the v2 package itself per this project's no-silent-edit-of-a-frozen-file
convention) -- though that interval still includes zero at both 68% and 95%
confidence.

See Figure 3 (`figures/tracka_epsS040_model_comparison.*`), Figure 4
(`figures/tracka_dnn_vs_transformer_rejection.*`), Table B
(`TRACKA_MECHANISM_TABLE.tex`).

---

## 2. Statistical limitation

### Track B

Support tiers follow the pre-frozen convention: raw surviving QCD $\geq100$
= well-supported; 10-99 = limited but reportable; $<10$ = exploratory only,
not a primary quantitative claim. On `holdout_A`'s 20,501-event QCD
population, **only `eps_S=0.60` is well-supported for all three arms**;
`eps_S` from 0.50 down to 0.20 is mostly in the limited-but-reportable tier
(a few points, e.g. E0/E2 at `eps_S=0.50`, are well-supported); `eps_S=0.10`
is exploratory for all three arms (raw QCD 2.7-4.7) and is **not** a primary
quantitative claim anywhere in this package, despite E2's numerically large
point estimate there (10250 $\pm$ 7248 -- a relative std of 71%, i.e.
numerically larger, not statistically distinguishable). See Table C
(`SUPPORT_LIMITS_TABLE.tex`) for Track A's analogous fixed-`eps_B` boundary
and Table A for Track B's per-point support tier.

### Track A

Support gate (all three required): `raw_background_rows>=100 AND
qcd_raw_rows>=10 AND qcd_Neff>=10`. On the fixed-`eps_S` grid actually
plotted (0.10-0.60), **every point passes the gate for both models, at every
one of the 3 seeds** -- the grid never enters unsupported territory. Support
only collapses on the separate fixed-`eps_B` slicing (primary seed only):
`eps_B=0.01` (`R_B=100`) still passes; `eps_B=0.001` (`R_B~1000`) and
`eps_B=0.0001` (`R_B~10,000`) both **fail** (`qcd_Neff` 1.1-6.6, both models)
and are not reported as supported physical claims anywhere in this package
(Figure 4 marks this region explicitly; Table C lists it).

---

## 3. Interpretation

**Central finding, stated conservatively:**

> On the high-statistics Track-B benchmark, continuous reconstructed flavor
> information produces a large and reproducible increase in QCD rejection
> relative to jet kinematics alone, whereas the tested project-defined
> pairwise attention bias does not provide an additional benefit.
>
> On Track A, pairwise event modeling gives only a modest,
> working-point-dependent improvement over a dense DNN, while finite-QCD
> support and the absence of continuous flavor information limit access to
> the extreme-rejection regime.

Three separable reasons the two tracks show a different-looking picture for
"does pairwise/attention modeling help," per the already-frozen Track-A
mechanism interpretation (`mechanism_interpretation_20260817_v1`):

1. **Track A's transformer carries both a kinematic/pairwise-attention change
   AND (unlike Track B) never had access to continuous flavor input at all**
   -- Track A's own b-tag input is Delphes' binary flag, not a continuous
   score, so Track A cannot isolate "flavor vs. pairwise attention" the way
   Track B's E0/E1/E2 ladder does by construction. Track A's transformer
   result is a single combined architecture effect, not decomposed.
2. **Track B's E1-vs-E2 result and Track A's DNN-vs-transformer result are
   not the same comparison.** Track B isolates the pairwise-attention-bias
   mechanism specifically (E1 already has flavor; E2 only adds the pairwise
   bias) and finds it does not help. Track A compares an entire architecture
   (kinematics + pairwise attention, no flavor) against a plain DNN (same
   kinematics, no flavor, no pairwise structure) and finds a modest,
   working-point-dependent edge. These are compatible findings, not
   contradictory ones: they answer different, narrower questions.
3. **Finite QCD support caps how deep either track's claim can go**,
   independent of which classifier is used -- Track A's own supported ceiling
   (`R_B~100-170`) and Track B's well-supported ceiling on `holdout_A`
   (`R_B~100`, `eps_S=0.60` only) are both properties of sample size, not of
   model quality.

**Track-B mechanism-gain answers (per the pre-registered mechanism
questions):**

1. **E0->E1 rejection gain** at `eps_S=0.60/0.50/0.40/0.10`: **+23% / +37% /
   +54% / +79%** (seed-paired mean ratio; the `eps_S=0.10` figure is
   exploratory-tier only, not a primary claim). All well-supported/limited
   points show the same direction across all 3 seeds.
2. **E1->E2 additional effect** at the same points: **-32% / -39% / -44% /
   +29%** (again, `eps_S=0.10` exploratory only -- the sign flips back to
   positive only in the unsupported tail, consistent with Figure 2 and
   `VALIDATION_REPORT.json`). This is a **degradation**, not a gain,
   everywhere the sample supports a claim.
3. **Is the E2 tail statistically supported or merely numerically larger?**
   Merely numerically larger. E2's largest point estimates live entirely in
   the exploratory tier (raw QCD 1-3), with seed-to-seed relative spread of
   33%-71%.
4. **Are all three seeds qualitatively consistent?** Yes at every
   well-supported or limited-but-reportable point (E1>E0 always; E1>E2
   always) with exactly one documented exception: `eps_S=0.20`, seed 0,
   where E2 edges out E1 at the raw-QCD=10 support boundary.
5. **Does E2 show a reproducible benefit over E1?** No. The frozen dev-stage
   finding (E2 underperforms E1) reproduces on the independent one-time
   `holdout_A` sample, with one boundary-case exception reported, not
   suppressed.

---

## 4. What remains to be tested

- **Track B**: the E1-vs-E2 `eps_S=0.20` seed-0 discrepancy is unresolved --
  plausibly ordinary statistical fluctuation at a low-count boundary, but not
  confirmed, and `holdout_A` may not be reopened to investigate it further
  under the current authorization (it is a spent resource for this
  adjudication).
- **Track B / next model families**: native official SPA-Net
  (`Alexanders101/SPANet`) is CPU-integration-verified end to end but has not
  completed a CLI-level training canary or any full-scale training; a GPU
  canary (Phase-4U, FNAL EAF A100) is prepared but had not been executed as
  of the latest inspected artifact (Phase-4V, 2026-08-15). Frozen pretrained
  canonical ParT remains blocked on checkpoint domain-mismatch and
  unverified normalization constants. `holdout_B` remains sealed, reserved
  for a later, one-time adjudication once the principal model families
  (including SPA-Net) are frozen -- this document does not open it and does
  not recommend opening it yet.
- **Track A**: whether the transformer's modest edge over the DNN is itself
  driven by the pairwise-attention mechanism specifically (as opposed to
  parameter count, ~29x larger) is not disentangled by this benchmark; Track
  A has no E0/E1/E2-style ablation of its own.
- **Cross-track transfer**: Track B's E1 mechanism (continuous flavor input)
  is confirmed **not** transferable to Track A as-is (Track A has no
  continuous jet-flavor discriminator materialized anywhere in its
  pipeline); closing that gap would require either running SophonAK4 on
  Track A's own constituents (unverified compatibility) or training a
  Track-A-native continuous flavor tagger (unscoped). Neither is attempted
  here.

---

## 5. Explicit non-claims

This document, and every figure/table in this package, does **not** claim:

- **CMS equivalence.** CMS HIG-24-010's public description reports
  approximately two orders of magnitude of background rejection at
  `eps_S~40%` and three orders at `eps_S~10%`, for its full analysis chain.
  Neither track attempts or approaches this number: both differ from CMS in
  simulation (Delphes/Pythia8 vs. full CMS simulation+reconstruction),
  background model (single inclusive-QCD Delphes sample vs. CMS's
  data-driven multi-process model), and, for Track A, tagger (binary Delphes
  flag vs. continuous CMS taggers).
- **Reproduction of CMS significance.** Neither track computes a
  CMS-comparable expected-signal-strength limit.
- **Official CMS architecture reproduction.** The CMS-inspired five-jet
  pairwise event Transformer matches only the *publicly documented* facts of
  HIG-24-010 (five leading b-tagged jets, pairwise features modifying
  attention, eight Transformer blocks); every other implementation choice is
  a project choice, explicitly not claimed to match CMS's actual,
  undisclosed hyperparameters (`PROJECT_CHOICE_PREPERFORMANCE`/
  `PROJECT_CHOICE_NOT_CMS` in the frozen model source).
- **Track-B physical yields.** Track B's `gen_weight`-based counts are not
  treated as an authoritative physical normalization; Track B is a
  methodology-validation benchmark, independent of Track A's Run-2 138
  fb$^{-1}$ convention.
- **Physically normalized training-QCD significance.** No number in this
  package is a signal-significance or discovery-significance claim.
