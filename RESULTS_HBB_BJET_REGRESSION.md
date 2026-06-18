# H->bb B-Jet Response Regression

This study trains simulation-level b-jet response corrections for truth-matched H->bb AK4 jets in COLLIDE-1M HH->bbWW.

The target is:

`target = log(GenJetAK4 pT / RecoJetAK4 pT)`

The correction is applied as:

`corrected pT = reco pT * exp(predicted target)`

For the m_bb evaluation, the matched reco-jet four-vector is scaled by the same factor, keeping eta and phi unchanged.

## Dataset configuration

- gen_b_to_reco_ak4_dr: 0.4
- reco_ak4_to_genjet_ak4_dr: 0.4
- max_reco_jets: -1
- min_reco_pt: 20.0
- max_abs_eta: 2.5

## Methods

- **Uncorrected:** no response correction.
- **pT/eta binned correction:** median target in reco pT and |eta| bins, fit using the training split only.
- **DNN regression:** MLP trained with Huber/SmoothL1 loss and selected using validation loss.

## Test-set per-jet response closure

| Method | Jets | Target MAE | Target RMSE | Closure median | Closure 16% | Closure 84% | Closure width68 | Correction median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Uncorrected | 1978 | 0.1559 | 0.2128 | 0.9315 | 0.7753 | 1.0729 | 0.1488 | 1.0000 |
| pT/eta binned correction | 1978 | 0.1255 | 0.1808 | 0.9972 | 0.8784 | 1.1527 | 0.1371 | 1.0709 |
| DNN regression | 1978 | 0.1006 | 0.1452 | 0.9970 | 0.8977 | 1.1091 | 0.1057 | 1.0575 |

Here, response closure is `corrected pT / GenJetAK4 pT`; a perfect correction has median closure near 1.

## Test-set event-level m_bb

Events skipped because they did not have exactly two rows in the test split: 0

| Method | Events | Mean [GeV] | Median [GeV] | 16% [GeV] | 84% [GeV] | Width68 [GeV] | Frac 90-140 | Frac 100-150 | |Median-125| [GeV] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Uncorrected | 989 | 112.04 | 105.39 | 83.03 | 130.85 | 23.91 | 0.6542 | 0.5035 | 19.61 |
| pT/eta binned correction | 989 | 122.17 | 115.75 | 94.03 | 139.71 | 22.84 | 0.7250 | 0.6714 | 9.25 |
| DNN regression | 989 | 120.32 | 114.64 | 93.90 | 134.88 | 20.49 | 0.7563 | 0.6754 | 10.36 |

## DNN training

- best epoch: 40
- best validation loss: 0.061370
- hidden width: 128
- dropout: 0.1
- learning rate: 0.001
- weight decay: 0.0001
- Huber beta: 0.1

## Important caveat

This is a simulation-level, CMS-inspired response correction study. It does not reproduce the full CMS jet energy calibration chain and does not include explicit secondary-vertex variables.

## Conclusion

Using all selected reco AK4 jets with pT > 20 GeV and |eta| < 2.5, the uncorrected matched H→bb mass has a median of 105.4 GeV on the test set. A simple pT/η binned correction shifts the median to 115.7 GeV, while the DNN regression gives a median of 114.6 GeV. Although the binned correction gives a slightly better median mass position, the DNN gives the best per-jet response prediction and the best mass resolution: the 68% half-width improves from 23.9 GeV uncorrected to 20.5 GeV with the DNN, and the fraction of events in 90–140 GeV increases from 65.4% to 75.6%.

## Outputs

- `outputs/hbb_bjet_regression/results.json`
- `outputs/hbb_bjet_regression/perjet_metrics_test.csv`
- `outputs/hbb_bjet_regression/event_mass_summary_test.csv`
- `outputs/hbb_bjet_regression/event_masses_test.csv`
- `outputs/hbb_bjet_regression/predictions_test.csv`
- `outputs/hbb_bjet_regression/binned_correction_table.csv`
- `outputs/hbb_bjet_regression/dnn_training_history.csv`
- `outputs/plots/hbb_bjet_regression/*.png`
