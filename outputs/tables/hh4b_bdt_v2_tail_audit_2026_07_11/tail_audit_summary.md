# HH4b BDT-v2 tail audit


Date: 2026-07-11


This audit retrains the two BDT-v2 classifiers seed-by-seed, saves event-level BDT scores, and studies the most signal-like tails.


The goal is to identify which few high-score events dominate the highest-purity region and which backgrounds need more MC statistics.


## AUC summary


|   qcd_bdt_weighted_auc_mean |   qcd_bdt_weighted_auc_std |   top_bdt_weighted_auc_mean |   top_bdt_weighted_auc_std |
|----------------------------:|---------------------------:|----------------------------:|---------------------------:|
|                    0.832653 |                  0.0128467 |                    0.788759 |                  0.0168122 |


## Tail region summary


| tail_region                |   background_tail_fraction |   tail_score_cut_median |   signal_rows_median |   background_rows_median |   S_450fb_mean |   B_450fb_mean |   S_over_B_mean |   S_over_sqrtB_mean |   S_over_sqrtB_std |   asimov_Z_mean |   background_neff_median |   largest_background_weight_fraction_median |
|:---------------------------|---------------------------:|------------------------:|---------------------:|-------------------------:|---------------:|---------------:|----------------:|--------------------:|-------------------:|----------------:|-------------------------:|--------------------------------------------:|
| top_0.1pct_background_tail |                      0.001 |                0.786526 |                   21 |                        7 |        8.97548 |        5711.19 |     0.00165953  |            0.120798 |          0.0464498 |        0.12076  |                  4.15748 |                                   0.305322  |
| top_0.2pct_background_tail |                      0.002 |                0.765249 |                   29 |                       14 |       13.0448  |       11978.5  |     0.00112826  |            0.12068  |          0.0362645 |        0.120655 |                  6.80641 |                                   0.256985  |
| top_0.5pct_background_tail |                      0.005 |                0.734117 |                   46 |                       34 |       22.2856  |       30130.4  |     0.000750936 |            0.129165 |          0.0339813 |        0.129148 |                 17.2036  |                                   0.110047  |
| top_1.0pct_background_tail |                      0.01  |                0.699194 |                   65 |                       68 |       34.0785  |       63308.6  |     0.000546531 |            0.136294 |          0.0253643 |        0.136281 |                 35.6366  |                                   0.0544443 |
| top_2.0pct_background_tail |                      0.02  |                0.652926 |                   93 |                      136 |       52.6301  |      131216    |     0.000404611 |            0.145829 |          0.0211448 |        0.145819 |                 73.6201  |                                   0.0262872 |
| top_5.0pct_background_tail |                      0.05  |                0.571924 |                  145 |                      339 |       91.5038  |      338382    |     0.000271365 |            0.157525 |          0.018911  |        0.157518 |                156.157   |                                   0.0366244 |


## Tightest background tail composition


| tail_region                | analysis_sample                   |   rows_median |   expected_events_450fb_mean |   fraction_of_tail_background_mean |
|:---------------------------|:----------------------------------|--------------:|-----------------------------:|-----------------------------------:|
| top_0.1pct_background_tail | ttbar_200k                        |             1 |                     3731.57  |                          0.54945   |
| top_0.1pct_background_tail | qcd_bbbb_iht200to400_combined120k |             3 |                     3419.18  |                          0.616118  |
| top_0.1pct_background_tail | qcd_bbbb_iht400to600_20000        |             2 |                     1137.04  |                          0.2165    |
| top_0.1pct_background_tail | qcd_bbbb_iht600plus_20000         |             2 |                      263.2   |                          0.0531067 |
| top_0.1pct_background_tail | Zbbbb_100k                        |             1 |                      123.716 |                          0.0228683 |


## Interpretation checklist


- Low `background_neff_median` means the tail is MC-statistics limited.


- Large `largest_background_weight_fraction_median` means a few weighted events dominate the estimate.


- The next generated background should be whichever sample dominates the tightest tail.


- This is diagnostic and does not change the training inputs.

