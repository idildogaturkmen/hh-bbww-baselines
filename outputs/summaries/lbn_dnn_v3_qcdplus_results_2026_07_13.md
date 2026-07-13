# LBN-DNN v3 qcdplus results summary

Date: 2026-07-13  
Branch: `delphes-hh4b-production`

## Purpose

This note documents the lightweight LBN-style neural-network baseline for the HH→4b Delphes analysis.

The LBN-DNN uses the same BDT-v3 qcdplus candidate dataset as the BDT and ordinary DNN studies. The input dataset contains four selected candidate jet four-vectors in `[E, px, py, pz]` format.

Two modes are evaluated:

1. **lbn_p4_only**
   - Uses only the four candidate jet four-vectors.
2. **lbn_p4_plus_topology**
   - Uses the four candidate jet four-vectors plus topology-only scalar auxiliary features.

## Main conclusion

The LBN-DNN does not beat the BDT baseline. The four-vector-only model is weak, especially for the top-background classifier. Adding topology features improves performance substantially, but the result remains below the BDT and ordinary mass-aware DNN baselines.

The LBN-DNN is still useful because it provides a physics-structured neural-network baseline between the plain dense DNN and SPA-Net.

## LBN AUC summary

| mode                 | classifier   | negative_class               |   unweighted_auc |   physics_weighted_auc |   n_train_task |   n_test_task |
|:---------------------|:-------------|:-----------------------------|-----------------:|-----------------------:|---------------:|--------------:|
| lbn_p4_only          | LBN_QCD      | QCD bbbb HT slices           |           0.657  |                 0.6329 |          24729 |         13359 |
| lbn_p4_only          | LBN_top      | ttbar / top-like backgrounds |           0.5424 |                 0.538  |           1366 |           706 |
| lbn_p4_plus_topology | LBN_QCD      | QCD bbbb HT slices           |           0.746  |                 0.7185 |          24729 |         13359 |
| lbn_p4_plus_topology | LBN_top      | ttbar / top-like backgrounds |           0.6668 |                 0.6692 |           1366 |           706 |

## LBN best stable regions

|   qcd_threshold |   top_threshold |   signal_events_450fb |   background_events_450fb |   S_over_B |   S_over_sqrtB |   S_over_10pctB |   n_signal_test_rows |   n_background_test_rows |   neff_signal |   neff_background | mode                 |
|----------------:|----------------:|----------------------:|--------------------------:|-----------:|---------------:|----------------:|---------------------:|-------------------------:|--------------:|------------------:|:---------------------|
|           0.55  |             0.5 |                70.93  |                    712715 |   0.0001   |       0.084018 |        0.000995 |                   94 |                     1211 |       57.0374 |           177.844 | lbn_p4_only          |
|           0.625 |             0.5 |                78.377 |                    408549 |   0.000192 |       0.122621 |        0.001918 |                  111 |                      716 |       63.5009 |           131.262 | lbn_p4_plus_topology |

## LBN fixed-category yields

| mode                 | category                    |   n_signal_test_rows |   n_background_test_rows |   signal_events_450fb |   background_events_450fb |   S_over_B |   S_over_sqrtB |   S_over_10pctB |   neff_signal |   neff_background |
|:---------------------|:----------------------------|---------------------:|-------------------------:|----------------------:|--------------------------:|-----------:|---------------:|----------------:|--------------:|------------------:|
| lbn_p4_only          | CAT0_high_purity_diagnostic |                   25 |                      303 |                14.185 |                  198594   |   7.1e-05  |       0.031831 |        0.000714 |      11.8324  |           39.8366 |
| lbn_p4_only          | CAT1_tight                  |                   17 |                      141 |                15.999 |                   60430.4 |   0.000265 |       0.065084 |        0.002648 |      12.5934  |           61.1938 |
| lbn_p4_only          | CAT2_medium_tight           |                   18 |                      127 |                12.413 |                   74209.5 |   0.000167 |       0.045565 |        0.001673 |      10.085   |           45.5142 |
| lbn_p4_only          | CAT3_medium                 |                   15 |                      276 |                10.955 |                  156292   |   7e-05    |       0.02771  |        0.000701 |       8.84109 |           43.0408 |
| lbn_p4_only          | CAT4_loose                  |                   23 |                      482 |                17.693 |                  294911   |   6e-05    |       0.032581 |        0.0006   |      14.1982  |           56.5161 |
| lbn_p4_plus_topology | CAT0_high_purity_diagnostic |                   58 |                      348 |                42.44  |                  166778   |   0.000254 |       0.10392  |        0.002545 |      34.2438  |          120.537  |
| lbn_p4_plus_topology | CAT1_tight                  |                   16 |                       53 |                 9.812 |                   29313.5 |   0.000335 |       0.057306 |        0.003347 |       8.09325 |           17.2598 |
| lbn_p4_plus_topology | CAT2_medium_tight           |                   15 |                       91 |                 8.511 |                   70184.2 |   0.000121 |       0.032126 |        0.001213 |       7.09941 |           11.1689 |
| lbn_p4_plus_topology | CAT3_medium                 |                   14 |                      141 |                 9.654 |                   84489   |   0.000114 |       0.033214 |        0.001143 |       7.84389 |           24.7069 |
| lbn_p4_plus_topology | CAT4_loose                  |                   16 |                      250 |                12.255 |                  152033   |   8.1e-05  |       0.03143  |        0.000806 |       9.8389  |           43.05   |

## All-model best-region comparison

| model                | region                                       |   signal_events_450fb |   background_events_450fb |   S_over_B |   S_over_sqrtB |   S_over_10pctB |
|:---------------------|:---------------------------------------------|----------------------:|--------------------------:|-----------:|---------------:|----------------:|
| BDT mass_aware       | CAT0+CAT1+CAT2                               |                92.113 |                  214886   |   0.000429 |       0.198708 |        0.004287 |
| DNN mass_aware       | best stable rectangle: qcd>=0.95, top>=0.575 |                45.67  |                   87898.8 |   0.00052  |       0.154042 |        0.005196 |
| BDT topology_only    | CAT0+CAT1+CAT2                               |                52.444 |                  136390   |   0.000385 |       0.142005 |        0.003845 |
| lbn_p4_plus_topology | best stable rectangle: qcd>=0.625, top>=0.5  |                78.377 |                  408549   |   0.000192 |       0.122621 |        0.001918 |
| DNN topology_only    | best stable rectangle: qcd>=0.8, top>=0.5    |                69.074 |                  432190   |   0.00016  |       0.10507  |        0.001598 |
| lbn_p4_only          | best stable rectangle: qcd>=0.55, top>=0.5   |                70.93  |                  712715   |   0.0001   |       0.084018 |        0.000995 |

## Interpretation

### lbn_p4_only

The four-vector-only model performs poorly. This suggests that the current LBN implementation and four selected candidate jet four-vectors alone do not capture enough information to compete with the tabular baselines.

### lbn_p4_plus_topology

Adding topology-only auxiliary variables improves the LBN significantly. This confirms that global event topology and candidate-level scalar features are important. However, the model still underperforms the BDT.

### Current model ordering

Using best stable S/sqrt(B), the current ordering is:

1. Mass-aware BDT-v3 qcdplus
2. Mass-aware DNN-v3 qcdplus
3. Topology-only BDT-v3 qcdplus
4. LBN-DNN p4 plus topology
5. Topology-only DNN-v3 qcdplus
6. LBN-DNN p4 only

## Next steps

1. Try one improved LBN variant with explicit b-tag auxiliary inputs.
2. If the improved LBN still does not approach BDT performance, stop optimizing LBN.
3. Move to SPA-Net as the next architecture baseline.
4. In parallel, begin ZZ→4b and ZH→4b Delphes validation samples.
