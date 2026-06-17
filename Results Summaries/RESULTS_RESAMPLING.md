# Bootstrap Test-Set Resampling

This study estimates sampling uncertainty from the fixed held-out test sets by resampling test events with replacement. It does not retrain any model and does not change validation-defined working-point thresholds.

## Command

```bash
/tmp/hh-bbww-venv/bin/python scripts/bootstrap_resampling.py --n-bootstrap 1000 --seed 20260616
```

## Inputs Used

- Pair-DNN: `outputs/hbb_pair_dnn/predictions.csv`, using the test-set `target_rank` column.
- HH-vs-ttbar BDT: `outputs/ttbar_classifier/scores.npz` and validation-defined thresholds in `working_points.csv`.
- HH-vs-ZJetsTobb full BDT: `outputs/zbb_rejection/scores.npz` and validation-defined full-BDT thresholds in `working_points.csv`.

## Main Results

| Metric | Nominal | Bootstrap mean | Std | 16-84% | 95% interval |
|---|---:|---:|---:|---:|---:|
| Pair-DNN Top-1 exact pair accuracy | 0.5731 | 0.5730 | 0.0140 | 0.5591-0.5862 | 0.5443-0.6018 |
| Pair-DNN Top-3 pair accuracy | 0.8218 | 0.8220 | 0.0108 | 0.8112-0.8325 | 0.7997-0.8440 |
| Pair-DNN Top-5 pair accuracy | 0.9056 | 0.9056 | 0.0082 | 0.8974-0.9138 | 0.8883-0.9212 |
| HH-vs-ttbar BDT test AUC | 0.7122 | 0.7121 | 0.0106 | 0.7015-0.7221 | 0.6909-0.7328 |
| ttbar_bdt: ttbar rejection at validation target 90% | 1.3920 | 1.3910 | 0.0243 | 1.3662-1.4153 | 1.3453-1.4381 |
| ttbar_bdt: ttbar rejection at validation target 70% | 2.4756 | 2.4780 | 0.0870 | 2.3904-2.5647 | 2.3236-2.6563 |
| ttbar_bdt: ttbar rejection at validation target 50% | 4.6489 | 4.6646 | 0.2576 | 4.4064-4.9193 | 4.2135-5.1992 |
| HH-vs-ZJetsTobb full-BDT test AUC | 0.9403 | 0.9404 | 0.0045 | 0.9359-0.9448 | 0.9313-0.9493 |
| zbb_full_bdt: ZJetsTobb rejection at validation target 90% | 6.0297 | 6.0569 | 0.3873 | 5.6756-6.4444 | 5.3530-6.9135 |
| zbb_full_bdt: ZJetsTobb rejection at validation target 70% | 15.8182 | 16.0373 | 1.8552 | 14.2736-17.7965 | 12.9558-20.0230 |
| zbb_full_bdt: ZJetsTobb rejection at validation target 50% | 36.9091 | 38.2046 | 7.0373 | 31.6235-45.0370 | 27.6270-54.7273 |

## How To Interpret This

The existing binomial uncertainties in the reconstruction and working-point tables estimate counting uncertainty for simple pass/fail efficiencies. They are useful for quantities like pair accuracy or a single signal/background efficiency.

The ttbar BDT random-seed stability study varies the BDT `random_state` on fixed train/validation/test feature files. That probes model-training randomness for the corrected ttbar baseline, but it does not resample events or change the data split.

This bootstrap study keeps all trained models, fixed splits, and validation-selected thresholds unchanged. It asks: if the finite test set were a slightly different sample drawn from the same population, how much would the reported test metrics fluctuate? This is a test-set sampling uncertainty, not a full analysis uncertainty.

A future repeated split/retraining study would be stronger and more expensive: regenerate train/validation/test splits, retrain models, reselect thresholds on each validation split, and evaluate each corresponding test split. That would mix data-split variance, training variance, and test-set sampling variance.

## Outputs

- `outputs/resampling/bootstrap_replicates.csv`
- `outputs/resampling/bootstrap_summary.csv`
- `outputs/resampling/bootstrap_summary.json`
- `outputs/plots/resampling/*.png`
