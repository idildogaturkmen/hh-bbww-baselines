# LBN-DNN v3 qcdplus results summary

Date: 2026-07-13  
Branch: `delphes-hh4b-production`

## Purpose

This note documents the lightweight LBN-style neural-network baselines for the HH→4b Delphes analysis.

The LBN-DNN uses the same BDT-v3 qcdplus candidate dataset as the BDT and ordinary DNN studies. The input dataset contains four selected candidate jet four-vectors in `[E, px, py, pz]` format.

Modes evaluated:

1. `lbn_p4_only`: four candidate jet four-vectors only.
2. `lbn_p4_plus_topology`: four-vectors plus topology-only scalar features.
3. `lbn_p4_plus_topology_btag`: four-vectors plus topology-only features plus candidate b-tag scores.
4. `lbn_p4_plus_massaware_btag`: four-vectors plus mass-aware scalar features plus candidate b-tag scores. This is an upper-bound, mass-aware LBN mode.

## Main conclusion

The LBN-DNN does not beat the BDT baseline. The best LBN mode by AUC is `lbn_p4_plus_massaware_btag`, but the best LBN mode by expected sensitivity is `lbn_p4_plus_topology`.

Adding b-tag and mass-aware auxiliary information improves the global AUC but does not improve the best stable S/sqrt(B). Therefore, further LBN optimization is not the highest-priority next step.

## LBN AUC summary

| mode                       | classifier   | negative_class               |   unweighted_auc |   physics_weighted_auc |   n_train_task |   n_test_task |
|:---------------------------|:-------------|:-----------------------------|-----------------:|-----------------------:|---------------:|--------------:|
| lbn_p4_only                | LBN_QCD      | QCD bbbb HT slices           |           0.657  |                 0.6329 |          24729 |         13359 |
| lbn_p4_only                | LBN_top      | ttbar / top-like backgrounds |           0.5424 |                 0.538  |           1366 |           706 |
| lbn_p4_plus_topology       | LBN_QCD      | QCD bbbb HT slices           |           0.746  |                 0.7185 |          24729 |         13359 |
| lbn_p4_plus_topology       | LBN_top      | ttbar / top-like backgrounds |           0.6668 |                 0.6692 |           1366 |           706 |
| lbn_p4_plus_topology_btag  | LBN_QCD      | QCD bbbb HT slices           |           0.7344 |                 0.7251 |          24729 |         13359 |
| lbn_p4_plus_topology_btag  | LBN_top      | ttbar / top-like backgrounds |           0.6853 |                 0.6787 |           1366 |           706 |
| lbn_p4_plus_massaware_btag | LBN_QCD      | QCD bbbb HT slices           |           0.7718 |                 0.7425 |          24729 |         13359 |
| lbn_p4_plus_massaware_btag | LBN_top      | ttbar / top-like backgrounds |           0.7053 |                 0.7095 |           1366 |           706 |

## LBN best stable regions

|   qcd_threshold |   top_threshold |   signal_events_450fb |   background_events_450fb |   S_over_B |   S_over_sqrtB |   S_over_10pctB |   n_signal_test_rows |   n_background_test_rows |   neff_signal |   neff_background | mode                       |
|----------------:|----------------:|----------------------:|--------------------------:|-----------:|---------------:|----------------:|---------------------:|-------------------------:|--------------:|------------------:|:---------------------------|
|           0.55  |           0.5   |                70.93  |                    712715 |   0.0001   |       0.084018 |        0.000995 |                   94 |                     1211 |       57.0374 |           177.844 | lbn_p4_only                |
|           0.525 |           0.8   |                89.253 |                    549416 |   0.000162 |       0.120412 |        0.001624 |                  125 |                      812 |       72.2186 |           133.363 | lbn_p4_plus_massaware_btag |
|           0.625 |           0.5   |                78.377 |                    408549 |   0.000192 |       0.122621 |        0.001918 |                  111 |                      716 |       63.5009 |           131.262 | lbn_p4_plus_topology       |
|           0.525 |           0.775 |                66.436 |                    383817 |   0.000173 |       0.107236 |        0.001731 |                   99 |                      663 |       54.1586 |           142.118 | lbn_p4_plus_topology_btag  |

## LBN fixed-category yields

| mode                       | category                    |   n_signal_test_rows |   n_background_test_rows |   signal_events_450fb |   background_events_450fb |   S_over_B |   S_over_sqrtB |   S_over_10pctB |   neff_signal |   neff_background |
|:---------------------------|:----------------------------|---------------------:|-------------------------:|----------------------:|--------------------------:|-----------:|---------------:|----------------:|--------------:|------------------:|
| lbn_p4_only                | CAT0_high_purity_diagnostic |                   25 |                      303 |                14.185 |                  198594   |   7.1e-05  |       0.031831 |        0.000714 |      11.8324  |           39.8366 |
| lbn_p4_only                | CAT1_tight                  |                   17 |                      141 |                15.999 |                   60430.4 |   0.000265 |       0.065084 |        0.002648 |      12.5934  |           61.1938 |
| lbn_p4_only                | CAT2_medium_tight           |                   18 |                      127 |                12.413 |                   74209.5 |   0.000167 |       0.045565 |        0.001673 |      10.085   |           45.5142 |
| lbn_p4_only                | CAT3_medium                 |                   15 |                      276 |                10.955 |                  156292   |   7e-05    |       0.02771  |        0.000701 |       8.84109 |           43.0408 |
| lbn_p4_only                | CAT4_loose                  |                   23 |                      482 |                17.693 |                  294911   |   6e-05    |       0.032581 |        0.0006   |      14.1982  |           56.5161 |
| lbn_p4_plus_topology       | CAT0_high_purity_diagnostic |                   58 |                      348 |                42.44  |                  166778   |   0.000254 |       0.10392  |        0.002545 |      34.2438  |          120.537  |
| lbn_p4_plus_topology       | CAT1_tight                  |                   16 |                       53 |                 9.812 |                   29313.5 |   0.000335 |       0.057306 |        0.003347 |       8.09325 |           17.2598 |
| lbn_p4_plus_topology       | CAT2_medium_tight           |                   15 |                       91 |                 8.511 |                   70184.2 |   0.000121 |       0.032126 |        0.001213 |       7.09941 |           11.1689 |
| lbn_p4_plus_topology       | CAT3_medium                 |                   14 |                      141 |                 9.654 |                   84489   |   0.000114 |       0.033214 |        0.001143 |       7.84389 |           24.7069 |
| lbn_p4_plus_topology       | CAT4_loose                  |                   16 |                      250 |                12.255 |                  152033   |   8.1e-05  |       0.03143  |        0.000806 |       9.8389  |           43.05   |
| lbn_p4_plus_topology_btag  | CAT0_high_purity_diagnostic |                   69 |                      342 |                43.305 |                  219509   |   0.000197 |       0.09243  |        0.001973 |      35.6086  |           72.4875 |
| lbn_p4_plus_topology_btag  | CAT1_tight                  |                   10 |                       40 |                 6.896 |                   28724.5 |   0.00024  |       0.040688 |        0.002401 |       5.60278 |           13.7336 |
| lbn_p4_plus_topology_btag  | CAT2_medium_tight           |                    4 |                       52 |                 3.98  |                   43508.4 |   9.1e-05  |       0.019082 |        0.000915 |       3.11841 |            7.8326 |
| lbn_p4_plus_topology_btag  | CAT3_medium                 |                    8 |                      102 |                 5.517 |                   44806.4 |   0.000123 |       0.026062 |        0.001231 |       4.48223 |           40.5382 |
| lbn_p4_plus_topology_btag  | CAT4_loose                  |                   16 |                      223 |                 8.59  |                   90626.3 |   9.5e-05  |       0.028533 |        0.000948 |       7.22689 |           92.5126 |
| lbn_p4_plus_massaware_btag | CAT0_high_purity_diagnostic |                   88 |                      432 |                60.683 |                  323608   |   0.000188 |       0.106674 |        0.001875 |      49.3045  |           63.672  |
| lbn_p4_plus_massaware_btag | CAT1_tight                  |                    3 |                       42 |                 1.458 |                   26757.3 |   5.4e-05  |       0.008912 |        0.000545 |       1.2475  |           14.7446 |
| lbn_p4_plus_massaware_btag | CAT2_medium_tight           |                    7 |                       93 |                 4.216 |                   63514.1 |   6.6e-05  |       0.01673  |        0.000664 |       3.4865  |           15.7844 |
| lbn_p4_plus_massaware_btag | CAT3_medium                 |                   13 |                      119 |                 7.132 |                   84869.9 |   8.4e-05  |       0.024481 |        0.00084  |       5.97981 |           23.951  |
| lbn_p4_plus_massaware_btag | CAT4_loose                  |                   25 |                      213 |                19.072 |                  125790   |   0.000152 |       0.053775 |        0.001516 |      15.3186  |           48.2167 |

## All-model best-region comparison

| model                      | region                                        |   signal_events_450fb |   background_events_450fb |   S_over_B |   S_over_sqrtB |   S_over_10pctB |
|:---------------------------|:----------------------------------------------|----------------------:|--------------------------:|-----------:|---------------:|----------------:|
| BDT mass_aware             | CAT0+CAT1+CAT2                                |                92.113 |                  214886   |   0.000429 |       0.198708 |        0.004287 |
| DNN mass_aware             | best stable rectangle: qcd>=0.95, top>=0.575  |                45.67  |                   87898.8 |   0.00052  |       0.154042 |        0.005196 |
| BDT topology_only          | CAT0+CAT1+CAT2                                |                52.444 |                  136390   |   0.000385 |       0.142005 |        0.003845 |
| lbn_p4_plus_topology       | best stable rectangle: qcd>=0.625, top>=0.5   |                78.377 |                  408549   |   0.000192 |       0.122621 |        0.001918 |
| lbn_p4_plus_massaware_btag | best stable rectangle: qcd>=0.525, top>=0.8   |                89.253 |                  549416   |   0.000162 |       0.120412 |        0.001624 |
| lbn_p4_plus_topology_btag  | best stable rectangle: qcd>=0.525, top>=0.775 |                66.436 |                  383817   |   0.000173 |       0.107236 |        0.001731 |
| DNN topology_only          | best stable rectangle: qcd>=0.8, top>=0.5     |                69.074 |                  432190   |   0.00016  |       0.10507  |        0.001598 |
| lbn_p4_only                | best stable rectangle: qcd>=0.55, top>=0.5    |                70.93  |                  712715   |   0.0001   |       0.084018 |        0.000995 |

## Interpretation

### Four-vector-only LBN

The four-vector-only model is too weak, especially for top-background rejection. Fixed candidate jet four-vectors alone are not enough.

### LBN plus topology

Adding topology features gives the strongest LBN sensitivity. This shows that global event topology is important.

### LBN plus topology and b-tags

Adding b-tags improves the weighted AUC slightly but worsens the best stable S/sqrt(B). It does not improve the analysis-level performance.

### LBN plus mass-aware features and b-tags

This upper-bound mode gives the best LBN AUC, but not the best LBN sensitivity. It remains below the mass-aware BDT and mass-aware DNN baselines.

## Current conclusion

The current fixed-candidate LBN-DNN models do not outperform the BDT baseline. The next architecture worth trying is SPA-Net, because SPA-Net can address the jet-assignment and permutation problem directly instead of only classifying already chosen candidate jets.

## Next steps

1. Commit this final LBN summary.
2. Stop optimizing LBN for now.
3. Begin the SPA-Net dataset audit or SPA-Net training workflow.
4. In parallel, begin ZZ→4b and ZH→4b Delphes validation samples.
