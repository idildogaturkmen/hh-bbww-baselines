# Track B development results — interpretation

**DEVELOPMENT ONLY. This document does not report a final, governing,
or officially reviewed physics result. It is not independent Stage-C
inference. No physical (cross-section / luminosity / generator-weight)
normalization has been applied anywhere in this snapshot. No final
model ranking should be claimed from it.**

This snapshot freezes, for internal record-keeping ahead of the final
144M-population BDT fits, a compact comparison of four already-trained
DEVELOPMENT models — BDT-K, BDT-KF, SPA-Net (2M events), and SPA-Net
(10M events) — scored on the identical fixed 400,000-event training-
production development cohort (signal 193,358 / QCD 174,485 / ttbar
32,157). Every number quoted below is copied, unmodified, from
already-frozen, hash-verified source artifacts; see
`master_results/PROVENANCE_MAP.tsv` and `SOURCE_PROVENANCE.tsv` for the
exact source file and SHA256 behind each figure and table.

## 1. Continuous flavor information produces the dominant K→KF gain

Adding the three Sophon AK4 flavor-tag probability channels
(probB/probC/probL) to the 52 purely kinematic BDT-K features (→ 82
features for BDT-KF) produces, by a wide margin, the largest single
improvement seen anywhere in this comparison:

- AUC vs. all-background: **0.8366 → 0.9682** (Δ ≈ +0.132).
- At fixed εS = 0.50, background rejection improves from R_B ≈ 10.8 to
  R_B ≈ 225.3 — roughly a **21× reduction** in surviving background at
  the same signal efficiency, from adding flavor information alone.
- The effect grows at tighter working points (εS = 0.10: R_B 163.6 →
  11,480, a ≈70× improvement).

This is the dominant lever in the entire model comparison: continuous
flavor-tag information contributes far more discriminating power than
the kinematics-only BDT-K feature set, and more than the 2M→10M
SPA-Net training-population increase discussed below.

## 2. Native SPA-Net is competitive with — and directionally stronger
   than — KF in some QCD-tail working points

In this development comparison, both SPA-Net checkpoints (2M and 10M)
have a small, consistent edge over BDT-KF on overall AUC and, more
robustly, on QCD rejection specifically:

- Overall (vs. all-background) AUC: SPA-Net 2M 0.9693 / SPA-Net 10M
  0.9693, vs. BDT-KF 0.9682 (Δ ≈ +0.001, SPA-Net ahead in both cases).
- vs. QCD (the higher-statistics background, 174,485 events): SPA-Net
  is ahead of BDT-KF at *every* tested working point, and the margin
  widens as εS tightens. At εS = 0.60, R_QCD is 114.0 (BDT-KF) vs.
  129.1 (SPA-Net 2M) vs. 124.5 (SPA-Net 10M) — SPA-Net 2M ≈1.13× KF,
  SPA-Net 10M ≈1.09× KF. By εS = 0.10, R_QCD is 9,693.6 (BDT-KF) vs.
  17,448.5 (SPA-Net 2M, ≈1.80× KF) vs. 24,926.4 (SPA-Net 10M, ≈2.57×
  KF) — see `TABLE_WORKING_POINTS` for the full set. QCD is the
  highest-statistics background in this cohort, so this is the most
  statistically robust part of the comparison and the clearest
  evidence that SPA-Net's edge over KF here is real, not a statistical
  fluctuation. Note that SPA-Net 2M is modestly *ahead* of SPA-Net 10M
  on QCD rejection at every tested point (Section 3) — the SPA-Net vs.
  KF gap is not monotonically widening with SPA-Net training-population
  size.
- This edge is modest in absolute AUC terms and should be read as
  *directional*, not as a settled ranking — see Section 4 and the
  caveats below.

## 3. 2M→10M SPA-Net scaling does not produce a clear
   event-classification improvement

Scaling the SPA-Net training population 5× (2,000,000 → 10,000,000
events, same architecture, same seed, same batch size, same 50-epoch
budget, same validation population, same checkpoint-selection rule)
produces a small, direction-mixed change in classification AUC, not a
clear improvement:

| | SPA-Net 2M | SPA-Net 10M | Δ |
|---|---|---|---|
| AUC all-background | 0.96928 | 0.96927 | −0.00001 |
| AUC QCD | 0.96816 | 0.96776 | −0.00040 |
| AUC ttbar | 0.97539 | 0.97750 | +0.00211 |

All-background and QCD AUC are essentially flat (QCD is marginally
*lower* at 10M); only ttbar AUC moves up, by ≈0.002, itself within the
range that this comparison's own finite-support caveats (Section 5)
make hard to fully trust at the tightest operating points. This is in
sharp contrast to the jet-*assignment* metric
(`validation_average_jet_accuracy`), which does improve modestly with
scale (2M primary 0.49123 at epoch 49 → 10M primary 0.49284 at epoch
47) — but that is a jet-assignment metric, not an event-classification
metric, and should not be read as evidence of better event
classification. **The 5× training-population increase tested here does
not, by itself, translate into a clear event-classification gain** —
in this development setting, adding flavor-tag information (Section 1)
is a far larger lever than 5× more training statistics at the current
architecture and epoch budget.

## 4. Tight ttbar working points are finite-support limited for every model

At εS ≤ 0.25, raw ttbar-background survivor counts collapse to single
digits or zero for BDT-KF, SPA-Net 2M, and SPA-Net 10M alike (only
BDT-K, whose overall rejection is far weaker, retains >100 ttbar
survivors down to εS = 0.10). Concretely:

- BDT-KF: 10 survivors at εS = 0.25, 6 at εS = 0.20, **0** at εS = 0.10
  (rejection undefined at that point; only a lower bound R_B > 32,157
  is supported).
- SPA-Net 2M: 10 survivors at εS = 0.25, 6 at εS = 0.20, 1 at εS = 0.10.
- SPA-Net 10M: 10 survivors at εS = 0.25, 3 at εS = 0.20, **0** at
  εS = 0.10.

The full-curve ttbar AUC nominally favors BDT-KF over both SPA-Net
checkpoints (KF 0.9793 vs. SPA-Net 2M 0.9754 / SPA-Net 10M 0.9775), but
at the specific εS working points actually tested, these survivor
counts are Poisson-dominated. **This comparison is not capable of
resolving which classifier is actually better on ttbar at tight
working points**, regardless of what the full-curve AUC integral
suggests. Read the ttbar-AUC ranking as suggestive, not decisive; read
any R_ttbar value at or below εS ≈ 0.25 in `TABLE_WORKING_POINTS` as
carrying large, unquantified statistical uncertainty rather than as a
precise rejection estimate. Figure `fig_C_signal_eff_vs_ttbar_rejection`
marks every such point with an open-circle finite-support annotation.

## 5. These are NOT independent Stage-C results and are NOT physically normalized

- This entire snapshot uses the **training-production development
  population**, not the sealed, independent Stage-C inference/test
  population. BDT-K and BDT-KF may have encountered some or all of
  these 400,000 physical events during their own train/val split; no
  leakage control was applied for this comparison (none was
  requested). SPA-Net's own training/eval population relationship to
  this exact cohort has likewise not been independently re-verified as
  part of this specific comparison step (though by construction, the
  frozen 400k validation population is disjoint from the 2M/10M SPA-Net
  training populations under the project's nested `D_2M ⊂ D_5M ⊂
  D_10M` design).
- BDT-K and BDT-KF are **interim** models trained on an interim
  ~10.47M-event staged population (n_train = 8,377,425), not the final
  144M-population fit — both hit a hard 6000-boosting-round cap without
  their early-stopping criterion (50 non-improving rounds) ever firing.
  Both val-logloss curves are essentially flat in their final ~15
  rounds (*de facto* plateaued), but strictly speaking neither
  converged by the formal early-stopping rule.
- **No cross-section, luminosity, or generator-weight normalization**
  has been applied to any number in this snapshot. All populations are
  natural source-file mixtures (flat per-event weighting), not a
  physically normalized analysis sample.
- The Stage-C independent-inference contract has not been designed or
  previewed anywhere in this project as of this snapshot.

## 6. No final model ranking should be claimed yet

Taken together: flavor-tag information is the dominant, unambiguous
lever between BDT-K and BDT-KF; native SPA-Net shows a small,
directionally consistent edge over BDT-KF in this development
comparison (most robustly on QCD, the higher-statistics background);
5× more SPA-Net training data does not by itself produce a clear
event-classification improvement; and the ttbar comparison across all
models is inconclusive at tight operating points due to low
statistics. None of this constitutes a final ranking of BDT-KF vs.
native SPA-Net, nor a final judgment on whether further SPA-Net
scaling is worthwhile — both questions require the eventual final
144M-population BDT fit and a properly designed, independent Stage-C
inference evaluation, neither of which exists yet. This snapshot exists
to freeze the current DEVELOPMENT-stage evidence base for internal
reference before that work begins, not to substitute for it.
