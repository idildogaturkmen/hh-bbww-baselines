# SPA-Net qcdplus HH→4b results

Date: 2026-07-14  
Branch: `delphes-hh4b-production`

## Dataset

The SPA-Net qcdplus dataset uses ROOT-backed leading-8 jet inputs with features `pt, eta, phi, mass, btag`.

Final NPZ:
`$HH4B_STORE/spanet_npz/qcdplus_btag4_v1/hh4b_spanet_leading8_all.npz`

Dataset size:
- total selected events: 42,322
- signal events: 1,331
- background events: 40,991
- signal events with complete HH assignment labels: 924

## Model

This first SPA-Net qcdplus baseline uses:
- jet-set transformer encoder
- event classification head
- pair-assignment head
- assignment loss on matched signal events
- classification loss on all events

## Test summary

|   test_auc_all_unweighted |   test_auc_all_weighted |   test_qcd_weighted_auc |   test_top_weighted_auc |   test_assignment_mask_events |   best_threshold |   best_signal_events_450fb |   best_background_events_450fb |   best_S_over_B |   best_S_over_sqrtB |   best_n_signal_test_rows |   best_n_background_test_rows |
|--------------------------:|------------------------:|------------------------:|------------------------:|------------------------------:|-----------------:|---------------------------:|-------------------------------:|----------------:|--------------------:|--------------------------:|------------------------------:|
|                  0.743684 |                  0.7231 |                0.758687 |                 0.45955 |                           140 |              0.8 |                    85.5456 |                         275263 |     0.000310778 |            0.163051 |                        51 |                           197 |

## Best-threshold composition

| sample                                        |   selected_test_rows |   expected_events_450fb_scaled | group      |
|:----------------------------------------------|---------------------:|-------------------------------:|:-----------|
| ggF_HH4b_SMnorm                               |                   26 |                       81.1871  | signal     |
| VBF_HH4b_SMnorm                               |                   25 |                        4.35847 | signal     |
| ttbar_200k                                    |                   25 |                   190182       | background |
| Zbbbb_100k                                    |                   13 |                     2691.36    | background |
| qcd_bbbb_iht100to200_20000                    |                    0 |                        0       | background |
| qcd_bbbb_iht200to400_combined220k_rootkeep_v1 |                   29 |                    49978.7     | background |
| qcd_bbbb_iht400to600_combined120k_rootkeep_v1 |                  103 |                    25178.2     | background |
| qcd_bbbb_iht600plus_20000                     |                   27 |                     7232.82    | background |

## Comparison to current baselines

| model             |   best_S_over_sqrtB | role                   |
|:------------------|--------------------:|:-----------------------|
| BDT mass-aware    |            0.198708 | best current baseline  |
| SPA-Net qcdplus   |            0.163051 | first full SPA-Net     |
| DNN mass-aware    |            0.154042 | plain NN baseline      |
| BDT topology-only |            0.142005 | mass-sculpting control |
| LBN p4 + topology |            0.122621 | best LBN sensitivity   |
| DNN topology-only |            0.10507  | topology NN control    |

## Interpretation

The first full qcdplus SPA-Net baseline is successful and improves over the smoke run. It reaches a best S/sqrt(B) of about 0.163, which is competitive with the plain DNN baseline but still below the mass-aware BDT-v3 qcdplus baseline.

The main weakness is top-background rejection: the top weighted AUC is about 0.460. This suggests that the next architecture should not only deepen the transformer, but also add separate QCD and top classification heads.

## Next steps

1. Make ROC, score, and composition plots.
2. Implement Deep/two-head SPA-Net:
   - QCD head: signal vs QCD-like backgrounds
   - top head: signal vs ttbar/top-like backgrounds
   - assignment head: HH jet assignment
   - optional global event features
3. Compare Deep SPA-Net to BDT/DNN/LBN and this first SPA-Net.
