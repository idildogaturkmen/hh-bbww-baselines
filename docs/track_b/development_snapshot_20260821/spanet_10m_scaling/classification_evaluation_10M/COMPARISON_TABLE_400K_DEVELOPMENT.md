# Compact 400k classification-evaluation comparison -- DEVELOPMENT ONLY

**Not a final/governing physics result. Not independent Stage-C
inference. No physical normalization applied.** All four rows below
are scored on the exact same frozen 400,000-event cohort (n_signal =
193,358, n_qcd = 174,485, n_ttbar = 32,157), with cohort identity
established via a hash-gated provenance chain (this project's own
`build_event_provenance_bridge.py`, PASS 15/15, feeding
`build_matched_400k_features.py` in the BDT study for BDT-K/BDT-KF;
the exact same `production_2M_val.h5`, sha256
`3c94bf8300d1dc3324e2cf25f9b6c618dffb94819cb3a297a43ac06320d5c2a2`, for
both SPA-Net rows).

## Primary comparison table

| Model | AUC all-background | AUC QCD | AUC ttbar |
|---|---|---|---|
| BDT-K (52 features, kinematics-only, interim 10.47M-population) | 0.8366047 | 0.8270892 | 0.8882361 |
| BDT-KF (82 features, + Sophon AK4 flavor tags, interim 10.47M-population) | 0.9681764 | 0.9661312 | 0.9792733 |
| SPA-Net 2M seed-0 (primary, epoch 49, frozen) | 0.9692812 | 0.9681550 | 0.9753918 |
| **SPA-Net 10M seed-0 (primary, epoch 47, this phase)** | **0.9692723** | **0.9677557** | **0.9775018** |

10M primary source: `classification_evaluation_10M/work/classification_evaluation_10M_result.json`,
checkpoint sha256 `fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c307102fa6943de5`.

## Working points, SPA-Net 10M primary (development cohort, n=400,000)

| target eps_S | threshold | achieved eps_S | R_all-bg | R_QCD | R_ttbar | n_sig pass | n_bg pass | n_QCD pass | n_ttbar pass |
|---|---|---|---|---|---|---|---|---|---|
| 0.60 | 0.9514 | 0.6000 | 132.8 | 124.5 | 207.5 | 116015 | 1556 | 1401 | 155 |
| 0.50 | 0.9733 | 0.5000 | 260.3 | 245.4 | 387.4 | 96679 | 794 | 711 | 83 |
| 0.40 | 0.9854 | 0.4000 | 510.2 | 476.7 | 824.5 | 77343 | 405 | 366 | 39 |
| 0.25 | 0.9947 | 0.2500 | 1895.8 | 1762.5 | 3215.7 | 48341 | 109 | 99 | 10 |
| 0.20 | 0.9964 | 0.2000 | 4132.8 | 3712.4 | 10719.0 | 38672 | 50 | 47 | 3 |
| 0.10 | 0.9987 | 0.1000 | 29520.3 | 24926.4 | null (0 survivors) | 19336 | 7 | 7 | 0 |

**Finite-support caveat**: at eps_S=0.10, ttbar has **zero** surviving
background events out of 32,157 -- `ttbar_rejection` is undefined
(`null`), matching the same low-count-tail pattern already documented
for the frozen 2M SPA-Net evaluation. At eps_S=0.20, ttbar survivors
drop to 3; at eps_S=0.25, to 10. Any R_ttbar value at or below
eps_S~0.25 should be read as having large, unquantified statistical
uncertainty from small-number counting, not a precise rejection
estimate.

## Caveats carried forward verbatim from the BDT-K/BDT-KF study (Step2j, `matched_400k/`)

1. BDT-K and BDT-KF may have encountered some or all of these 400,000
   physical events during their own train/val split (natural source
   mixture -- no leakage control was applied for this comparison, none
   was requested). SPA-Net's own training/eval population relationship
   to this exact 400k cohort is likewise not independently established
   here (2M/10M training populations are disjoint-by-construction from
   the fixed validation population per the nested `D_2M subset D_5M
   subset D_10M` design, but this has not been re-verified as part of
   this specific comparison step).
2. Per `TASK_D_EVALUATION_BOUNDARY_FREEZE.md`: this is a DEVELOPMENT
   matched-cohort comparison, not a final evaluation; the Stage-C
   independent-inference contract has not been designed or previewed
   anywhere in this project.
3. BDT-K/BDT-KF are interim 10.47M-population models
   (`n_train=8,377,425`) that hit a 6000-round hard cap without
   early-stopping triggering -- not the eventual final-144M-population
   fit.
4. On ttbar specifically, BDT-KF nominally edges both SPA-Net rows
   (0.9793 vs. 0.9754 [2M] / 0.9775 [10M]) at the AUC level, but tight
   working-point survivor counts collapse to single digits/zero for
   all three classifiers alike -- this comparison is not capable of
   resolving which classifier is actually better on ttbar at tight
   operating points.

## Diagnostic-only: 10M primary vs. secondary checkpoint

Purpose: demonstrate that checkpoint choice was not cherry-picked. This
diagnostic result **does not** change checkpoint selection -- the
governing checkpoint remains the primary (maximum
`validation_average_jet_accuracy`), per the project's fixed selection
rule.

| Checkpoint | Epoch | AUC all-background | AUC QCD | AUC ttbar |
|---|---|---|---|---|
| Primary (governing, max `validation_average_jet_accuracy`) | 47 | 0.9692723 | 0.9677557 | 0.9775018 |
| Secondary (diagnostic, min `val/loss/total_loss`) | 49 | 0.9694593 | 0.9678947 | 0.9779489 |
| Difference (secondary - primary) | -- | +0.0001870 | +0.0001391 | +0.0004471 |

The two checkpoints are statistically indistinguishable at this
precision (AUC differences ~1-4e-4 across all three background
definitions) -- the same primary/secondary near-identical-plateau
pattern already established for the frozen 2M run. This confirms the
epoch-47 primary was not a cherry-picked outlier: essentially any
late-training checkpoint in this run performs comparably on event
classification, and the small residual difference does not warrant
overriding the governing selection rule.

Full working-point tables for the secondary:
`work/classification_evaluation_10M_secondary_diagnostic_result.json`
(checkpoint sha256 `0a7f5504c0ab11434bc8712bff8ff90cf45c30b4daa073e00cf37fb5e2e24ca1`,
`threshold_accuracy_0p5=0.91037`).
