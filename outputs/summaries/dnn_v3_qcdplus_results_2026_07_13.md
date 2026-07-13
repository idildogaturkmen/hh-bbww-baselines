# DNN-v3 qcdplus results summary

Date: 2026-07-13  
Branch: `delphes-hh4b-production`

## Purpose

This note documents the ordinary dense neural-network baseline trained after the BDT-v3 qcdplus study.

Two DNNs were trained:

1. **DNN mass-aware**
   - Uses the same 46 reconstructed features as the mass-aware BDT-v3 qcdplus baseline.
   - Includes direct candidate mass information.

2. **DNN topology-only**
   - Uses the same 33 topology-only features as the topology-only BDT-v3 qcdplus baseline.
   - Removes direct mass variables.

Both DNNs use the same enlarged-QCD samples and the same train/test split strategy as the BDT-v3 qcdplus studies.

## Main conclusion

The ordinary dense DNN does **not** outperform the BDT baseline. The BDT remains the best current tabular baseline. The DNN is still useful as a controlled neural-network baseline before moving to LBN-DNN and SPA-Net.

## BDT vs DNN AUC comparison

| model             |   QCD weighted AUC |   top weighted AUC |
|:------------------|-------------------:|-------------------:|
| BDT mass_aware    |             0.8355 |             0.809  |
| BDT topology_only |             0.7923 |             0.7852 |
| DNN mass_aware    |             0.7731 |             0.7462 |
| DNN topology_only |             0.7346 |             0.7151 |

## BDT vs DNN best-region comparison

For the BDT, the main region is CAT0+CAT1+CAT2. For the DNN, the best stable rectangle from the DNN score scan is used because DNN scores are not calibrated the same way as BDT scores.

| model             | region                                       |   signal_events_450fb |   background_events_450fb |   S_over_B |   S_over_sqrtB |   S_over_10pctB |   n_signal_test_rows |   n_background_test_rows |
|:------------------|:---------------------------------------------|----------------------:|--------------------------:|-----------:|---------------:|----------------:|---------------------:|-------------------------:|
| BDT mass_aware    | CAT0+CAT1+CAT2                               |                92.113 |                  214886   |   0.000429 |       0.198708 |        0.004287 |                  144 |                      389 |
| BDT topology_only | CAT0+CAT1+CAT2                               |                52.444 |                  136390   |   0.000385 |       0.142005 |        0.003845 |                   66 |                      223 |
| DNN mass_aware    | best stable rectangle: qcd>=0.95, top>=0.575 |                45.67  |                   87898.8 |   0.00052  |       0.154042 |        0.005196 |                   68 |                      146 |
| DNN topology_only | best stable rectangle: qcd>=0.8, top>=0.5    |                69.074 |                  432190   |   0.00016  |       0.10507  |        0.001598 |                  117 |                      672 |

## DNN fixed-category yields

These use the same numerical category definitions as the BDT category audit. They are useful for consistency checks, but the DNN best-rectangle scan is more important for performance comparison.

| run           | category                    |   n_signal_test_rows |   n_background_test_rows |   signal_events_450fb |   background_events_450fb |   S_over_B |   S_over_sqrtB |   S_over_10pctB |   neff_signal |   neff_background |
|:--------------|:----------------------------|---------------------:|-------------------------:|----------------------:|--------------------------:|-----------:|---------------:|----------------:|--------------:|------------------:|
| mass_aware    | CAT0_high_purity_diagnostic |                  110 |                      416 |                67.302 |                  264553   |   0.000254 |       0.130848 |        0.002544 |      55.5324  |           74.0816 |
| mass_aware    | CAT1_tight                  |                   23 |                       91 |                18.915 |                   77248   |   0.000245 |       0.068055 |        0.002449 |      15.0747  |           17.7983 |
| mass_aware    | CAT2_medium_tight           |                   14 |                      171 |                 8.432 |                  106518   |   7.9e-05  |       0.025837 |        0.000792 |       6.97299 |           37.1537 |
| mass_aware    | CAT3_medium                 |                   28 |                      238 |                15.643 |                  120785   |   0.00013  |       0.04501  |        0.001295 |      13.0791  |          110.081  |
| mass_aware    | CAT4_loose                  |                   34 |                      398 |                22.224 |                  269270   |   8.3e-05  |       0.042828 |        0.000825 |      18.1772  |           61.2553 |
| topology_only | CAT0_high_purity_diagnostic |                   62 |                      335 |                37.867 |                  220783   |   0.000172 |       0.080589 |        0.001715 |      31.2527  |           48.4494 |
| topology_only | CAT1_tight                  |                   35 |                      169 |                18.637 |                  124690   |   0.000149 |       0.05278  |        0.001495 |      15.701   |           36.3664 |
| topology_only | CAT2_medium_tight           |                   20 |                      168 |                12.57  |                   86717.7 |   0.000145 |       0.042685 |        0.00145  |      10.334   |           68.9194 |
| topology_only | CAT3_medium                 |                   16 |                      316 |                 8.59  |                  200732   |   4.3e-05  |       0.019172 |        0.000428 |       7.22689 |           44.595  |
| topology_only | CAT4_loose                  |                   33 |                      533 |                28.254 |                  402206   |   7e-05    |       0.044552 |        0.000702 |      22.4302  |           80.3015 |

## DNN background mass-sculpting summary

| run           | category                    |   background_yield_450fb |   neff_background |   mbb1_median |   mbb2_median |   avg_mbb_median |   mhh_median |   r_hh_median |   avg_mbb_distance_from_125 |
|:--------------|:----------------------------|-------------------------:|------------------:|--------------:|--------------:|-----------------:|-------------:|--------------:|----------------------------:|
| mass_aware    | CAT0_high_purity_diagnostic |                 264553   |            74.082 |       119.242 |       112.892 |          116.472 |      338.285 |        26.058 |                       8.528 |
| mass_aware    | CAT1_tight                  |                  77248   |            17.798 |       121.166 |       117.33  |          120.744 |      327.109 |        32.227 |                       4.256 |
| mass_aware    | CAT2_medium_tight           |                 106518   |            37.154 |       117.412 |       115.591 |          115.893 |      314.175 |        25.923 |                       9.107 |
| mass_aware    | CAT3_medium                 |                 120785   |           110.081 |       118.714 |       117.177 |          117.86  |      320.855 |        27.156 |                       7.14  |
| mass_aware    | CAT4_loose                  |                 269270   |            61.255 |       119.515 |       118.042 |          117.334 |      321.239 |        28.689 |                       7.666 |
| topology_only | CAT0_high_purity_diagnostic |                 220783   |            48.449 |       123.423 |       112.911 |          119.814 |      388.736 |        35.2   |                       5.186 |
| topology_only | CAT1_tight                  |                 124690   |            36.366 |       123.576 |       120.838 |          124.042 |      352.464 |        35.79  |                       0.958 |
| topology_only | CAT2_medium_tight           |                  86717.7 |            68.919 |       119.972 |       115.174 |          117.152 |      321.676 |        36.719 |                       7.848 |
| topology_only | CAT3_medium                 |                 200732   |            44.595 |       121.927 |       114.794 |          119.691 |      327.628 |        36.992 |                       5.309 |
| topology_only | CAT4_loose                  |                 402206   |            80.302 |       122.653 |       118.705 |          121.708 |      322.55  |        36.737 |                       3.292 |

## Interpretation

### DNN mass-aware

The mass-aware DNN performs better than the topology-only DNN, but worse than the mass-aware BDT. Its best stable rectangle has S/sqrt(B) around 0.154, compared with about 0.199 for the mass-aware BDT CAT0+CAT1+CAT2 region.

### DNN topology-only

The topology-only DNN is the weakest of the tabular models studied so far. This confirms that both mass information and the BDT model class are important for the current performance.

### Model hierarchy so far

Current performance ordering:

1. Mass-aware BDT-v3 qcdplus
2. Topology-only BDT-v3 qcdplus
3. Mass-aware DNN-v3 qcdplus
4. Topology-only DNN-v3 qcdplus

The DNN result supports moving next to a more physics-structured neural network, such as LBN-DNN, rather than spending too much time on a plain dense DNN.

## Plots

Generated plots are stored in:

`outputs/plots/dnn_v3_qcdplus_summary_2026_07_13`

Important plots:
- `bdt_vs_dnn_weighted_auc.png`
- `bdt_vs_dnn_best_s_over_sqrtB.png`
- `dnn_fixed_category_s_over_sqrtB.png`
- `dnn_category_background_composition.png`

## Next steps

1. Commit this DNN baseline and summary.
2. Train an LBN-DNN using the four selected jet four-vectors.
3. Compare LBN-DNN against BDT and ordinary DNN using the same metrics.
4. Move to SPA-Net only after the tabular and structured-DNN baselines are documented.
