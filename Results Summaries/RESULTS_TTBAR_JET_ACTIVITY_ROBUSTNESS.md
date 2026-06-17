# TTbar Jet-Activity Robustness Check

This study addresses whether the corrected HH-vs-semileptonic-ttbar BDT is still sensitive to jet multiplicity or to the fixed `MAX_JETS = 12` preprocessing choice.

## Command

```bash
/tmp/hh-bbww-venv/bin/python scripts/check_ttbar_jet_activity_robustness.py --seed 42
```

## Method

- The script reuses the fixed `outputs/ttbar_classifier/features_{train,val,test}.npz` files; it does not rescan COLLIDE-1M.
- The nominal model uses the same `GradientBoostingClassifier` configuration as the current ttbar baseline.
- `drop_n_jets` removes the explicit jet-count input.
- `njet_reweighted_training` gives training events weights so the HH and ttbar `n_jets` spectra match.
- `njet_balanced` evaluation uses validation/test weights that equalize the HH and ttbar `n_jets` spectra. This asks how much performance remains after the test metric stops rewarding jet-count differences.

## Test AUC Summary

| Variant | Natural test AUC | n_jets-balanced test AUC | Change |
|---|---:|---:|---:|
| all_features_nominal | 0.7122 | 0.7040 | -0.0082 |
| drop_n_jets | 0.7107 | 0.7003 | -0.0105 |
| njet_reweighted_training | 0.7057 | 0.7061 | +0.0004 |
| drop_n_jets_njet_reweighted_training | 0.7059 | 0.7000 | -0.0059 |

## Working Points

| Variant | Evaluation | Target signal eff. | Test signal eff. | Test ttbar eff. | Test ttbar rejection |
|---|---|---:|---:|---:|---:|
| all_features_nominal | natural | 0.50 | 0.5107 | 0.2151 | 4.65 |
| all_features_nominal | natural | 0.70 | 0.7020 | 0.4039 | 2.48 |
| all_features_nominal | natural | 0.90 | 0.9097 | 0.7184 | 1.39 |
| all_features_nominal | njet_balanced | 0.50 | 0.5130 | 0.2299 | 4.35 |
| all_features_nominal | njet_balanced | 0.70 | 0.7087 | 0.4296 | 2.33 |
| all_features_nominal | njet_balanced | 0.90 | 0.9160 | 0.7415 | 1.35 |
| drop_n_jets | natural | 0.50 | 0.5082 | 0.2053 | 4.87 |
| drop_n_jets | natural | 0.70 | 0.7028 | 0.4023 | 2.49 |
| drop_n_jets | natural | 0.90 | 0.9122 | 0.7250 | 1.38 |
| drop_n_jets | njet_balanced | 0.50 | 0.5140 | 0.2333 | 4.29 |
| drop_n_jets | njet_balanced | 0.70 | 0.6967 | 0.4243 | 2.36 |
| drop_n_jets | njet_balanced | 0.90 | 0.9167 | 0.7550 | 1.32 |
| njet_reweighted_training | natural | 0.50 | 0.5131 | 0.2217 | 4.51 |
| njet_reweighted_training | natural | 0.70 | 0.7020 | 0.4204 | 2.38 |
| njet_reweighted_training | natural | 0.90 | 0.8933 | 0.7069 | 1.41 |
| njet_reweighted_training | njet_balanced | 0.50 | 0.5211 | 0.2271 | 4.40 |
| njet_reweighted_training | njet_balanced | 0.70 | 0.7036 | 0.4205 | 2.38 |
| njet_reweighted_training | njet_balanced | 0.90 | 0.8960 | 0.7100 | 1.41 |
| drop_n_jets_njet_reweighted_training | natural | 0.50 | 0.4992 | 0.2077 | 4.81 |
| drop_n_jets_njet_reweighted_training | natural | 0.70 | 0.6897 | 0.4154 | 2.41 |
| drop_n_jets_njet_reweighted_training | natural | 0.90 | 0.9048 | 0.7266 | 1.38 |
| drop_n_jets_njet_reweighted_training | njet_balanced | 0.50 | 0.5128 | 0.2270 | 4.40 |
| drop_n_jets_njet_reweighted_training | njet_balanced | 0.70 | 0.6951 | 0.4375 | 2.29 |
| drop_n_jets_njet_reweighted_training | njet_balanced | 0.90 | 0.9077 | 0.7411 | 1.35 |

## Feature Importance Notes

| Variant | Most important features |
|---|---|
| all_features_nominal | mbb_top2_btag (0.280), subleading_jet_pt (0.195), HT (0.180), second_highest_btag (0.126) |
| drop_n_jets | mbb_top2_btag (0.291), subleading_jet_pt (0.199), HT (0.174), deltaR_top2_btag (0.130) |
| njet_reweighted_training | mbb_top2_btag (0.301), subleading_jet_pt (0.222), second_highest_btag (0.131), deltaR_top2_btag (0.125) |
| drop_n_jets_njet_reweighted_training | mbb_top2_btag (0.326), subleading_jet_pt (0.234), deltaR_top2_btag (0.144), second_highest_btag (0.139) |

## Interpretation

On the natural test split, the mean capped jet multiplicity is about 9.29 for HH and 9.61 for semileptonic ttbar. Therefore, jet activity is a real difference between the samples, but it is also exactly the kind of difference that can make a classifier look better if not checked carefully.

If the natural AUC and the n_jets-balanced AUC are close, the BDT is not mainly being rewarded for the class-level jet-count spectrum. If the balanced AUC is much lower, then the separation depends strongly on jet multiplicity or features correlated with it.

This is different from the earlier `MAX_JETS` retention scan. That scan asked whether a 12-jet cap loses truth-matchable H->bb signal events. This check asks whether the event-level classifier changes when the HH and ttbar jet-multiplicity spectra are made comparable.

## Outputs

- `outputs/ttbar_jet_activity_robustness/variant_metrics.csv`
- `outputs/ttbar_jet_activity_robustness/working_points.csv`
- `outputs/ttbar_jet_activity_robustness/feature_importance.csv`
- `outputs/ttbar_jet_activity_robustness/njet_distributions.csv`
- `outputs/ttbar_jet_activity_robustness/summary.json`
- `outputs/plots/ttbar_jet_activity_robustness/*.png`
