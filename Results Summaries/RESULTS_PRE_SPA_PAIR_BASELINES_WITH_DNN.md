# Pre-SPA-Net H→bb Pair Baselines with Pair-DNN

This comparison adds pair-level neural-network classifiers to the existing physics and BDT pre-SPA-Net baselines.

## Test metrics

| Selector | Events | Pair accuracy | Acc. err | m_bb median [GeV] | m_bb width68 [GeV] | Frac 90-140 | Frac 100-150 | Pair AUC | Pair AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top2_btag | 1969 | 0.5307 | 0.0114 | 117.27 | 78.61 | 0.4530 | 0.3819 |  |  |
| closest_mass_uncorrected | 1969 | 0.2367 | 0.0097 | 123.60 | 9.99 | 0.9213 | 0.9152 |  |  |
| closest_mass_corrected | 1969 | 0.2671 | 0.0098 | 124.21 | 7.91 | 0.9426 | 0.9578 |  |  |
| btag_mass_uncorrected_tuned | 1969 | 0.6105 | 0.0105 | 112.91 | 22.09 | 0.7268 | 0.6379 |  |  |
| btag_mass_corrected_tuned | 1969 | 0.6333 | 0.0103 | 121.70 | 21.08 | 0.7288 | 0.7344 |  |  |
| bdt_uncorrected_features | 1969 | 0.6856 | 0.0108 | 104.59 | 19.45 | 0.7288 | 0.5526 | 0.9329 | 0.6070 |
| bdt_corrected_features | 1969 | 0.6978 | 0.0105 | 119.22 | 21.59 | 0.7461 | 0.7359 | 0.9366 | 0.6157 |
| pair_dnn_uncorrected_features | 1969 | 0.6897 | 0.0109 | 105.63 | 20.89 | 0.7151 | 0.5510 | 0.9344 | 0.6109 |
| pair_dnn_corrected_features | 1969 | 0.6998 | 0.0110 | 119.67 | 22.55 | 0.7318 | 0.7141 | 0.9394 | 0.6255 |

## Interpretation

- `pair_dnn_uncorrected_features` is the neural-network analogue of the uncorrected-feature BDT.
- `pair_dnn_corrected_features` is the neural-network analogue of the corrected-feature BDT.
- The fair comparison is BDT corrected features vs Pair-DNN corrected features, since both use the same candidate-pair table and deterministic event split.
- SPA-Net should later be compared against the strongest pre-SPA-Net method from this table.

## Results
the pair-level DNN is now your strongest pre-SPA-Net baseline, but only by a small margin over the corrected-feature BDT.

The best method is: pair_dnn_corrected_features

pair accuracy = 0.6998 ± 0.0110
pair AUC      = 0.9394
pair AP       = 0.6255

Compared to the best BDT:
bdt_corrected_features:
pair accuracy = 0.6978 ± 0.0105
pair AUC      = 0.9366
pair AP       = 0.6157


So the corrected-feature Pair-DNN is slightly better than the corrected-feature BDT in:

truth-pair accuracy
ROC AUC
average precision

But the difference in pair accuracy is tiny:

0.6998 - 0.6978 = 0.0020

That is much smaller than the bootstrap uncertainty of about 0.011, so not a dramatic improvement.

## Corrected features

For both BDT and Pair-DNN, adding the DNN b-jet correction features helps.

BDT:

bdt_uncorrected_features accuracy = 0.6856
bdt_corrected_features   accuracy = 0.6978

Pair-DNN:

pair_dnn_uncorrected_features accuracy = 0.6897
pair_dnn_corrected_features   accuracy = 0.6998

DNN-corrected b-jet response features improve H→bb pair assignment for both tree-based and neural pair classifiers.

## Conclusion 
The Pair-DNN performs comparably to the corrected-feature BDT, with a slight improvement in pair-level classifier metrics.


## validation-loss plot

The Pair-DNN validation-loss plot looks healthy.

uncorrected Pair-DNN best epoch = 12
corrected Pair-DNN best epoch = 18

The corrected-feature model reaches a lower validation loss:

uncorrected best val loss = 0.6099
corrected best val loss   = 0.5868

That means the corrected features make the pair-classification problem easier for the neural network.

## pair accuracy vs selected mass
Pair-DNN corrected has the best accuracy, but its selected mass is not the narrowest:

pair_dnn_corrected_features:
m_bb width68 = 22.55 GeV

closest_mass_corrected:
m_bb width68 = 7.91 GeV

This is expected. The mass-only selector artificially chooses pairs close to 125 GeV, even if they are wrong. Pair-DNN tries to choose the true pair, not to sculpt the mass peak.

Mass-only methods produce narrow selected-mass distributions but low truth-pair accuracy. Pair classifiers give much higher assignment accuracy, at the cost of broader selected-mass distributions.

### Next step
The strongest pre-SPA-Net baseline to beat is: pair_dnn_corrected_features with accuracy about: 0.700 ± 0.011

## Summary
The corrected-feature Pair-DNN gives the strongest pre-SPA-Net performance, with a truth-pair accuracy of 0.6998 ± 0.0110, AUC 0.9394, and AP 0.6255. Its performance is comparable to the corrected-feature BDT, which reaches 0.6978 ± 0.0105 accuracy. Both BDT and Pair-DNN improve when DNN b-jet response correction features are included, showing that the response correction is useful for H→bb assignment. The next step is to move from pair-level classification to event-level permutation-aware assignment with SPA-Net.