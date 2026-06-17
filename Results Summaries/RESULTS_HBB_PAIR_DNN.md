# Pair-DNN H->bb Assignment Baseline

This is a lightweight learned baseline between the hand-built heuristics and SPA-Net. It scores each unordered AK4 jet pair independently with a shared MLP.

## Inputs

- Up to 12 AK4 jets per event
- Per-jet features: `[pt, eta, phi, mass, btag, btagPhys]`
- Pair extras: `[mjj, deltaR, pt_sum, pt_abs_diff, mass_sum, btag_sum, btag_max, btag_min]`
- Target: unordered truth-matched H -> bb jet pair

## Training

- Seed: `42`
- Epochs requested: `30`
- Best validation epoch: `30`
- Batch size: `2048`
- Learning rate: `0.001`
- Weight decay: `0.0001`

## Test Metrics

| Metric | Value |
|---|---:|
| Top-1 exact pair accuracy | 0.5731 +/- 0.0142 |
| Top-3 pair accuracy | 0.8218 +/- 0.0110 |
| Top-5 pair accuracy | 0.9056 +/- 0.0084 |
| Median selected m_jj | 103.17 GeV |
| Fraction with 90 < m_jj < 140 GeV | 0.594 |

The model is trained on truth-pair labels, not on closeness to 125 GeV. The selected m_jj distribution is therefore a diagnostic, not the optimization target.
