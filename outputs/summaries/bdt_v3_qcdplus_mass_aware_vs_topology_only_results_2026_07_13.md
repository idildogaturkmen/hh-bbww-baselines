# BDT-v3 qcdplus results summary: mass-aware vs topology-only

Date: 2026-07-13  
Branch: `delphes-hh4b-production`

## Purpose

This note summarizes the current BDT-v3 qcdplus baseline for the HH→4b Delphes analysis before moving to DNN, LBN-DNN, and SPA-Net.

Two BDT versions are compared:

1. **Mass-aware BDT-v3 qcdplus**
   - Uses candidate mass variables such as `mbb1`, `mbb2`, `avg_mbb`, `delta_mbb`, `r_hh`, `r_hh_125_125`, `r_hh_125_120`, `mhh`, and jet masses.
   - This is the strongest current BDT baseline.

2. **Topology-only BDT-v3 qcdplus**
   - Removes direct Higgs-candidate mass variables.
   - Used as a mass-sculpting control.

Both use the same enlarged-QCD sample set and the same train/test split logic.

## Main conclusion

The mass-aware BDT-v3 is the strongest current baseline. It gives higher AUC and higher expected sensitivity than the topology-only BDT. However, the mass-aware BDT also pulls background events closer to the Higgs-like mass region, so it is mass-sculpting. The topology-only BDT is weaker but provides an important robustness control.

## AUC summary

| model                | classifier                                           | negative class               |   unweighted AUC |   physics-weighted AUC |   train rows |   test rows |
|:---------------------|:-----------------------------------------------------|:-----------------------------|-----------------:|-----------------------:|-------------:|------------:|
| mass-aware BDT-v3    | BDT_QCD_event_features_safe_v3_qcdplus               | QCD bbbb HT slices           |           0.8625 |                 0.8355 |        24756 |       13332 |
| mass-aware BDT-v3    | BDT_top_event_features_safe_v3_qcdplus               | ttbar / top-like backgrounds |           0.7851 |                 0.809  |         1347 |         725 |
| topology-only BDT-v3 | BDT_QCD_event_features_safe_v3_qcdplus_topology_only | QCD bbbb HT slices           |           0.8206 |                 0.7923 |        24756 |       13332 |
| topology-only BDT-v3 | BDT_top_event_features_safe_v3_qcdplus_topology_only | ttbar / top-like backgrounds |           0.7578 |                 0.7852 |         1347 |         725 |

## Exclusive category yields

| model                | category                    |   n_signal_test_rows |   n_background_test_rows |   signal_events_450fb |   background_events_450fb |   S_over_B |   S_over_sqrtB |   S_over_10pctB |   neff_background |
|:---------------------|:----------------------------|---------------------:|-------------------------:|----------------------:|--------------------------:|-----------:|---------------:|----------------:|------------------:|
| mass-aware BDT-v3    | CAT0_high_purity_diagnostic |                   34 |                       43 |                20.071 |           19177.4         |   0.001047 |       0.144932 |        0.010466 |           9.86593 |
| mass-aware BDT-v3    | CAT1_tight                  |                   63 |                      156 |                48.475 |          107869           |   0.000449 |       0.147593 |        0.004494 |          48.0873  |
| mass-aware BDT-v3    | CAT2_medium_tight           |                   47 |                      190 |                23.568 |           87839.4         |   0.000268 |       0.079519 |        0.002683 |          91.7433  |
| mass-aware BDT-v3    | CAT3_medium                 |                   58 |                      483 |                41.859 |          320385           |   0.000131 |       0.073953 |        0.001307 |          89.8074  |
| mass-aware BDT-v3    | CAT4_loose                  |                   86 |                     1268 |                76.415 |          810152           |   9.4e-05  |       0.084897 |        0.000943 |         203.752   |
| topology-only BDT-v3 | CAT0_high_purity_diagnostic |                    4 |                        7 |                 2.801 |            2650.24        |   0.001057 |       0.054407 |        0.010569 |           4.14384 |
| topology-only BDT-v3 | CAT1_tight                  |                   34 |                      105 |                30.037 |           68137           |   0.000441 |       0.115069 |        0.004408 |          29.8814  |
| topology-only BDT-v3 | CAT2_medium_tight           |                   28 |                      111 |                19.606 |           65602.8         |   0.000299 |       0.076549 |        0.002989 |          35.8031  |
| topology-only BDT-v3 | CAT3_medium                 |                   61 |                      429 |                34.617 |          261739           |   0.000132 |       0.067663 |        0.001323 |          94.5774  |
| topology-only BDT-v3 | CAT4_loose                  |                  126 |                     1771 |                93.212 |               1.06576e+06 |   8.7e-05  |       0.09029  |        0.000875 |         287.23    |

## Inclusive CAT0+CAT1+CAT2 comparison

The region CAT0+CAT1+CAT2 corresponds approximately to the best supported inclusive BDT-v3 rectangle: `qcd_score >= 0.800` and `top_score >= 0.500`.

| run           | region         |   n_signal_test_rows |   n_background_test_rows |   signal_events_450fb |   background_events_450fb |   S_over_B |   S_over_sqrtB |   S_over_10pctB |
|:--------------|:---------------|---------------------:|-------------------------:|----------------------:|--------------------------:|-----------:|---------------:|----------------:|
| mass_aware    | CAT0+CAT1+CAT2 |                  144 |                      389 |                92.113 |                    214886 |   0.000429 |       0.198708 |        0.004287 |
| topology_only | CAT0+CAT1+CAT2 |                   66 |                      223 |                52.444 |                    136390 |   0.000385 |       0.142005 |        0.003845 |

## Top background components by category

Only the top three background components per category are shown.

| model                | category                    | background sample                       |   test rows |   yield at 450/fb |   fraction of category B |   neff sample |
|:---------------------|:----------------------------|:----------------------------------------|------------:|------------------:|-------------------------:|--------------:|
| mass-aware BDT-v3    | CAT0_high_purity_diagnostic | ttbar_200k                              |           3 |          9877.67  |                   0.5151 |             3 |
| mass-aware BDT-v3    | CAT0_high_purity_diagnostic | qcd_bbbb_iht200to400_combined220k_bdtv2 |           8 |          5931.37  |                   0.3093 |             8 |
| mass-aware BDT-v3    | CAT0_high_purity_diagnostic | qcd_bbbb_iht400to600_combined120k_bdtv2 |          20 |          2080.57  |                   0.1085 |            20 |
| mass-aware BDT-v3    | CAT1_tight                  | ttbar_200k                              |          20 |         65851.1   |                   0.6105 |            20 |
| mass-aware BDT-v3    | CAT1_tight                  | qcd_bbbb_iht200to400_combined220k_bdtv2 |          44 |         32622.5   |                   0.3024 |            44 |
| mass-aware BDT-v3    | CAT1_tight                  | qcd_bbbb_iht400to600_combined120k_bdtv2 |          53 |          5513.5   |                   0.0511 |            53 |
| mass-aware BDT-v3    | CAT2_medium_tight           | qcd_bbbb_iht200to400_combined220k_bdtv2 |          92 |         68210.7   |                   0.7765 |            92 |
| mass-aware BDT-v3    | CAT2_medium_tight           | ttbar_200k                              |           3 |          9877.67  |                   0.1125 |             3 |
| mass-aware BDT-v3    | CAT2_medium_tight           | qcd_bbbb_iht400to600_combined120k_bdtv2 |          68 |          7073.93  |                   0.0805 |            68 |
| mass-aware BDT-v3    | CAT3_medium                 | qcd_bbbb_iht200to400_combined220k_bdtv2 |         246 |        182390     |                   0.5693 |           246 |
| mass-aware BDT-v3    | CAT3_medium                 | ttbar_200k                              |          18 |         59266     |                   0.185  |            18 |
| mass-aware BDT-v3    | CAT3_medium                 | qcd_bbbb_iht100to200_20000              |           4 |         56934.1   |                   0.1777 |             4 |
| mass-aware BDT-v3    | CAT4_loose                  | qcd_bbbb_iht200to400_combined220k_bdtv2 |         684 |        507132     |                   0.626  |           684 |
| mass-aware BDT-v3    | CAT4_loose                  | qcd_bbbb_iht100to200_20000              |          13 |        185036     |                   0.2284 |            13 |
| mass-aware BDT-v3    | CAT4_loose                  | ttbar_200k                              |          19 |         62558.6   |                   0.0772 |            19 |
| topology-only BDT-v3 | CAT0_high_purity_diagnostic | qcd_bbbb_iht200to400_combined220k_bdtv2 |           3 |          2224.26  |                   0.8393 |             3 |
| topology-only BDT-v3 | CAT0_high_purity_diagnostic | qcd_bbbb_iht600plus_20000               |           2 |           233.005 |                   0.0879 |             2 |
| topology-only BDT-v3 | CAT0_high_purity_diagnostic | qcd_bbbb_iht400to600_combined120k_bdtv2 |           1 |           104.028 |                   0.0393 |             1 |
| topology-only BDT-v3 | CAT1_tight                  | ttbar_200k                              |          13 |         42803.2   |                   0.6282 |            13 |
| topology-only BDT-v3 | CAT1_tight                  | qcd_bbbb_iht200to400_combined220k_bdtv2 |          25 |         18535.5   |                   0.272  |            25 |
| topology-only BDT-v3 | CAT1_tight                  | qcd_bbbb_iht400to600_combined120k_bdtv2 |          41 |          4265.16  |                   0.0626 |            41 |
| topology-only BDT-v3 | CAT2_medium_tight           | qcd_bbbb_iht200to400_combined220k_bdtv2 |          40 |         29656.8   |                   0.4521 |            40 |
| topology-only BDT-v3 | CAT2_medium_tight           | ttbar_200k                              |           9 |         29633     |                   0.4517 |             9 |
| topology-only BDT-v3 | CAT2_medium_tight           | qcd_bbbb_iht400to600_combined120k_bdtv2 |          31 |          3224.88  |                   0.0492 |            31 |
| topology-only BDT-v3 | CAT3_medium                 | qcd_bbbb_iht200to400_combined220k_bdtv2 |         202 |        149767     |                   0.5722 |           202 |
| topology-only BDT-v3 | CAT3_medium                 | ttbar_200k                              |          19 |         62558.6   |                   0.239  |            19 |
| topology-only BDT-v3 | CAT3_medium                 | qcd_bbbb_iht100to200_20000              |           2 |         28467.1   |                   0.1088 |             2 |
| topology-only BDT-v3 | CAT4_loose                  | qcd_bbbb_iht200to400_combined220k_bdtv2 |         947 |        702125     |                   0.6588 |           947 |
| topology-only BDT-v3 | CAT4_loose                  | qcd_bbbb_iht100to200_20000              |          16 |        227736     |                   0.2137 |            16 |
| topology-only BDT-v3 | CAT4_loose                  | ttbar_200k                              |          17 |         55973.4   |                   0.0525 |            17 |

## Background mass-sculpting check for CAT0-CAT2

This table focuses on background events because mass sculpting is mainly a background-modeling concern.

| run           | category                    |   yield_450fb |   neff |   mbb1_median |   mbb2_median |   avg_mbb_median |   mhh_median |   r_hh_median |   avg_mbb_distance_from_125 |
|:--------------|:----------------------------|--------------:|-------:|--------------:|--------------:|-----------------:|-------------:|--------------:|----------------------------:|
| mass_aware    | CAT0_high_purity_diagnostic |      19177.4  |  9.866 |       119.842 |       112.141 |          115.535 |      395.351 |        18.818 |                       9.465 |
| topology_only | CAT0_high_purity_diagnostic |       2650.24 |  4.144 |       102.878 |       109.091 |          106.679 |      309.234 |        32.057 |                      18.321 |
| mass_aware    | CAT1_tight                  |     107869    | 48.087 |       119.823 |       114.514 |          115.308 |      313.013 |        19.094 |                       9.692 |
| topology_only | CAT1_tight                  |      68137    | 29.881 |       118.077 |       104.858 |          108.943 |      268.769 |        37.131 |                      16.057 |
| mass_aware    | CAT2_medium_tight           |      87839.4  | 91.743 |       119.385 |       113.216 |          115.674 |      334.766 |        20.682 |                       9.326 |
| topology_only | CAT2_medium_tight           |      65602.8  | 35.803 |       114.875 |       114.954 |          113.92  |      323.185 |        36.712 |                      11.08  |

## Interpretation

### Mass-aware BDT-v3

The mass-aware BDT-v3 gives the best current sensitivity. CAT0 has the highest purity, but it is MC-stat limited. CAT1 is the most defensible tight category. The inclusive CAT0+CAT1+CAT2 region gives the best supported signal-enriched region.

### Topology-only BDT-v3

The topology-only BDT-v3 has lower AUC and lower expected sensitivity. This confirms that direct mass information is important for the current BDT performance. However, topology-only separation is nonzero, so event topology does contain useful discrimination power.

### Mass sculpting

The mass-aware BDT pulls background candidate masses closer to the Higgs-like region than the topology-only BDT, especially in CAT0 and CAT1. This is expected because the mass-aware BDT directly uses mass variables. This does not invalidate the mass-aware BDT, but it means any final paper-style study should report both mass-aware performance and topology-only control results.

## Current baseline choice

For future comparisons, use:

- **Main baseline:** mass-aware BDT-v3 qcdplus
- **Robustness/control baseline:** topology-only BDT-v3 qcdplus
- **Main signal-enriched region:** CAT0+CAT1+CAT2
- **Most defensible tight exclusive category:** CAT1
- **Diagnostic only:** CAT0, unless more MC statistics are produced

## Next analysis steps

1. Train DNN-v3 using the same mass-aware and topology-only feature sets.
2. Compare DNN-v3 to BDT-v3 using AUC, category yields, S/B, S/sqrt(B), background composition, and mass sculpting.
3. Train LBN-DNN after ordinary DNN.
4. Move to SPA-Net only after the tabular baselines are frozen and understood.
