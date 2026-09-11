# Multiclass DNN Comparison for HH→bbWW Semileptonic Baseline

This note summarizes the multiclass DNN studies using corrected recoMET inputs for the one-lepton HH→bbWW-like selection. All quoted yields use the same rough physics weights and the same train/validation/test split. These numbers are diagnostic only because the normalization is still rough and no systematics or fit model are included.

## Models compared

| Short name | Inputs used | Meaning |
|---|---|---|
| Tabular multiclass DNN | Engineered topology/recoMET features | CMS-inspired 3-class tabular DNN baseline |
| LBN v0 full | `X_obj + X_pair + X_aux` | b1, b2, lepton, MET objects; pair features; auxiliary scalar features |
| LBN v0 obj+pair only | `X_obj + X_pair` | Object branch plus manual pair features, no auxiliary scalars |
| LBN v0 obj only | `X_obj` | Raw object-level b1, b2, lepton, MET features only |
| LBN v0 obj+aux no-pair | `X_obj + X_aux` | Object branch plus auxiliary scalars, no manual pair branch |

The LBN-style v0 models are called “v0” because they only contain four true objects: b1, b2, the leading lepton, and MET. They do not yet include full non-b jet four-vectors, so they cannot fully model the hadronic-W/top side as object-level inputs.

## Main HH-vs-background performance

| Model | Train HH weighted AUC | Val HH weighted AUC | Test HH weighted AUC | Comment |
|---|---:|---:|---:|---|
| Tabular multiclass DNN | 0.913 | 0.697 | 0.611 | Strong train performance, noticeable overfitting |
| LBN v0 full | 0.797 | 0.714 | 0.625 | Best test HH AUC among the DNN variants so far |
| LBN v0 obj+pair only | 0.766 | 0.697 | 0.598 | Pair features help vs object-only, but not enough |
| LBN v0 obj only | 0.904 | 0.591 | 0.507 | Severe overfitting; nearly random weighted test HH AUC |
| LBN v0 obj+aux no-pair | 0.734 | 0.689 | 0.618 | Best category-level performance; close to full v0 AUC |

## Argmax-HH category performance on test split

| Model | Test S_raw | Test B_raw | Test S_weighted | Test B_weighted | Test S/B weighted | Test Z | Test background Neff |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tabular multiclass DNN | 101 | 1526 | 6.038 | 1.175e6 | 5.14e-6 | 0.00557 | 13.37 |
| LBN v0 full | 140 | 2258 | 8.369 | 2.151e6 | 3.89e-6 | 0.00571 | 28.37 |
| LBN v0 obj+pair only | 139 | 2528 | 8.309 | 1.972e6 | 4.21e-6 | 0.00592 | 30.75 |
| LBN v0 obj only | 109 | 2059 | 6.516 | 2.205e6 | 2.96e-6 | 0.00439 | 29.08 |
| LBN v0 obj+aux no-pair | 157 | 2439 | 9.386 | 2.032e6 | 4.62e-6 | 0.00658 | 32.08 |

## Stable global-HH score region on test split

| Model | Threshold | Test S_raw | Test B_raw | Test S_weighted | Test B_weighted | Test S/B weighted | Test Z | Test background Neff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tabular multiclass DNN | 0.125 | 213 | 4416 | 12.733 | 4.067e6 | 3.13e-6 | 0.00631 | 60.04 |
| LBN v0 full | 0.340 | 156 | 2654 | 9.326 | 2.597e6 | 3.59e-6 | 0.00579 | 38.03 |
| LBN v0 obj+pair only | 0.275 | 188 | 3778 | 11.239 | 3.297e6 | 3.41e-6 | 0.00619 | 55.33 |
| LBN v0 obj only | 0.060 | 210 | 4997 | 12.554 | 4.311e6 | 2.91e-6 | 0.00605 | 73.94 |
| LBN v0 obj+aux no-pair | 0.270 | 207 | 3959 | 12.375 | 3.478e6 | 3.56e-6 | 0.00664 | 60.07 |

## Comparison to corrected recoMET BDT reference

| Reference BDT region | Test S/B weighted | Test Z | Test background Neff | Interpretation |
|---|---:|---:|---:|---|
| Corrected recoMET simple-topology BDT high-purity | 4.66e-5 | 0.00646 | 25.96 | Much higher purity, but lower signal acceptance |
| Corrected recoMET simple-topology BDT broad stable | 3.49e-6 | 0.00676 | 58.68 | Broad stable reference region |

The best LBN-style multiclass DNN result so far is the **obj+aux no-pair** model. Its argmax-HH category gives test Z ≈ 0.00658 with background Neff ≈ 32.1, and its broad global-HH region gives test Z ≈ 0.00664 with background Neff ≈ 60.1. This is close to the broad corrected recoMET BDT reference, but it does not clearly beat the BDT. The BDT still has a much more signal-enriched high-purity category.

## Interpretation

The ablation study shows:

1. **Object-only is insufficient.** The b1, b2, lepton, and MET object inputs alone overfit badly. The object-only model reaches train HH weighted AUC ≈ 0.904 but only test HH weighted AUC ≈ 0.507.
2. **Pair features help compared with object-only.** Adding manual pair features improves test HH weighted AUC from about 0.507 to about 0.598.
3. **Auxiliary scalar features are very important.** The obj+aux no-pair model reaches test HH weighted AUC ≈ 0.618 and gives the best argmax-HH category among the LBN-style ablations.
4. **Manual pair features do not clearly improve over auxiliary features in the current v0 setup.** The full v0 model has the best test HH weighted AUC, but the obj+aux no-pair model has the best category-level test Z. This suggests the current pair branch is either partly redundant with existing auxiliary features or not yet using enough information to add a clear gain.
5. **The missing ingredient is likely full non-b jet four-vectors.** The v0 input has true objects only for b1, b2, lepton, and MET. For semileptonic HH→bbWW, the hadronic-W/top side is important. A stronger v2 model should add non-b jet four-vectors: j1, j2, j3, j4, with pt/eta/phi/mass.

## Recommended conclusion

A CMS-style multiclass DNN was tested in both tabular and v0 LBN-style forms. The LBN-style full model modestly improves the test HH weighted AUC relative to the tabular multiclass DNN, but the best category-level result comes from the object+auxiliary no-pair ablation. None of the multiclass DNNs clearly outperforms the corrected recoMET BDT baseline. The current result supports using the corrected recoMET BDT as the nominal event-level baseline, while treating the LBN-style DNN as a useful diagnostic. The next meaningful improvement should be a v2 object-level model with full non-b jet four-vectors rather than more tuning of the current v0 architecture.

## Next steps

1. Keep the corrected recoMET BDT as the nominal baseline.
2. Save the v0 LBN-style DNN study as an ablation result.
3. Run a process-composition diagnostic for the best DNN model, especially the obj+aux no-pair model.
4. Search upstream files for full jet object branches such as `FullReco_Jet_PT`, `FullReco_Jet_Eta`, `FullReco_Jet_Phi`, and `FullReco_Jet_Mass`.
5. If full jet branches exist upstream, build v2 inputs with b1, b2, lepton, MET, and the leading non-b jets as true four-vector objects.
