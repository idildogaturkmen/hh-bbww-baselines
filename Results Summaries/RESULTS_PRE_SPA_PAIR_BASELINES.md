# Pre-SPA-Net H→bb Pair Baselines

This study uses the candidate-pair table from the DNN-corrected b-jet response workflow.
Events are split deterministically by event_id into train/validation/test.

## Validation-tuned lambda values

- btag_mass_uncorrected: lambda = 100
- btag_mass_corrected: lambda = 200

## Test metrics

| Selector | Lambda | Events | Pair accuracy | Acc. err | m_bb median [GeV] | m_bb width68 [GeV] | Width err | Frac 90-140 | Frac 100-150 | Pair AUC | Pair AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top2_btag |  | 1969 | 0.5307 | 0.0114 | 117.27 | 78.61 | 4.52 | 0.4530 | 0.3819 |  |  |
| closest_mass_uncorrected |  | 1969 | 0.2367 | 0.0097 | 123.60 | 9.99 | 0.33 | 0.9213 | 0.9152 |  |  |
| closest_mass_corrected |  | 1969 | 0.2671 | 0.0098 | 124.21 | 7.91 | 0.24 | 0.9426 | 0.9578 |  |  |
| btag_mass_uncorrected_tuned | 100 | 1969 | 0.6105 | 0.0105 | 112.91 | 22.09 | 0.48 | 0.7268 | 0.6379 |  |  |
| btag_mass_corrected_tuned | 200 | 1969 | 0.6333 | 0.0103 | 121.70 | 21.08 | 0.59 | 0.7288 | 0.7344 |  |  |
| bdt_uncorrected_features |  | 1969 | 0.6856 | 0.0108 | 104.59 | 19.45 | 0.46 | 0.7288 | 0.5526 | 0.9329 | 0.6070 |
| bdt_corrected_features |  | 1969 | 0.6978 | 0.0105 | 119.22 | 21.59 | 0.52 | 0.7461 | 0.7359 | 0.9366 | 0.6157 |
