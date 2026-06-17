# H->bb Reconstruction Comparison

This comparison uses the fixed HH -> bbWW train/validation/test split in `outputs/hbb_npz/`. The goal is to establish transparent reconstruction baselines before SPA-Net training.

Input format:

- `jets`: event x jet x feature, shape `(N, 12, 6)`
- `mask`: event x jet, shape `(N, 12)`
- `labels`: event x 2 truth-matched H -> bb AK4 jet indices
- jet features: `[pt, eta, phi, mass, btag, btagPhys]`
- `MAX_JETS = 12`, retaining 99.5% of truth-matchable H -> bb events while limiting each event to 66 unordered jet pairs

## Test-set metrics

| Method | Status | Top-1 exact pair accuracy | Top-3 accuracy | Top-5 accuracy | Median selected m_jj | 90 < m_jj < 140 |
|---|---|---:|---:|---:|---:|---:|
| Top-2 b-tag | done | 0.3670 +/- 0.0138 | 0.4589 +/- 0.0143 | 0.5066 +/- 0.0143 | 99.39 GeV | 0.362 |
| Closest m_jj to 125 GeV | done | 0.0928 +/- 0.0083 | 0.2479 +/- 0.0124 | 0.4015 +/- 0.0140 | 124.87 GeV | 0.974 |
| Combined b-tag + mass | done | 0.3079 +/- 0.0132 | 0.5411 +/- 0.0143 | 0.6954 +/- 0.0132 | 123.33 GeV | 0.925 |
| Pair-DNN | done | 0.5731 +/- 0.0142 | 0.8218 +/- 0.0110 | 0.9056 +/- 0.0084 | 103.17 GeV | 0.594 |
| SPA-Net | planned | TBD | TBD | TBD | TBD | TBD |
| Jet-embedded SPA-Net | planned | TBD | TBD | TBD | TBD | TBD |

The Pair-DNN is a lightweight learned baseline. It scores each unordered jet pair with a shared MLP using per-jet features plus pair-level features. It is not yet SPA-Net: it does not use a global transformer/event encoder or explicit permutation-group assignment machinery.

## Current interpretation

- The Pair-DNN improves the test exact pair accuracy from 36.7% for the legacy top-2-btag baseline to 57.3%.
- The Pair-DNN reaches 82.2% Top-3 and 90.6% Top-5 accuracy, which is useful for later studies of candidate ranking or beam-search-style reconstruction.
- The closest-to-125 mass heuristic has a Higgs-like mass distribution by construction but only 9.3% exact truth-pair accuracy, so selected m_jj alone is not a reliable reconstruction metric.
- The Pair-DNN selected m_jj median is near the truth-matched median rather than being forced to 125 GeV, consistent with the matched m_bb diagnostic.

## Reproducibility commands

```bash
/tmp/hh-bbww-venv/bin/python scripts/evaluate_hbb_reconstruction.py
/tmp/hh-bbww-venv/bin/python scripts/train_pair_dnn.py --epochs 30 --patience 8 --batch-size 2048 --lr 0.001 --weight-decay 0.0001 --hidden 96 --dropout 0.10 --seed 42
```

## Outputs

- `outputs/hbb_reconstruction/baseline_metrics.csv`
- `outputs/hbb_reconstruction/model_comparison_metrics.csv`
- `outputs/plots/hbb_reconstruction/test_accuracy_comparison_with_pair_dnn.png`
- `outputs/hbb_pair_dnn/metrics.csv`
- `outputs/hbb_pair_dnn/training_history.csv`
- `outputs/plots/hbb_pair_dnn/training_loss.png`
- `outputs/plots/hbb_pair_dnn/training_accuracy.png`
- `outputs/plots/hbb_pair_dnn/test_selected_mbb.png`

## Notes for SPA-Net comparison

Future SPA-Net and jet-embedded SPA-Net runs should report the same split-level metrics and append rows with the same schema as `model_comparison_metrics.csv`. The table should not mix validation-selected numbers with test metrics: use validation for model/epoch selection, then report test once.
