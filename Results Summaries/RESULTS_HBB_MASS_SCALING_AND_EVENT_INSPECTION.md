# H→bb Mass Scaling and Event Inspection Study

This quick follow-up adds Harvey-requested normalized mass metrics and event candidates for inspection.

## Outputs

- `outputs/hbb_mass_scaling_and_event_inspection/test_normalized_mass_metrics.csv`
- `outputs/hbb_mass_scaling_and_event_inspection/test_event_inspection_candidates.csv`
- `outputs/plots/hbb_mass_scaling_and_event_inspection/test_relative_width68_percent.png`
- `outputs/plots/hbb_mass_scaling_and_event_inspection/test_raw_vs_scaled_high_tail.png`

## Definitions

- `width68_percent_of_median = 100 * width68 / median`
- `scale_to_125_from_median = 125 / median`
- `scaled_width68_gev = width68 * 125 / median`
- `scaled_tail_high_140` is computed after scaling each mass distribution so its median is 125 GeV.

## No-cut metrics

| version      |   median |   width68_gev |   width68_percent_of_median |   scaled_width68_gev |   raw_tail_high_140 |   scaled_tail_high_140 |   raw_frac_90_140 |   scaled_frac_90_140 |
|:-------------|---------:|--------------:|----------------------------:|---------------------:|--------------------:|-----------------------:|------------------:|---------------------:|
| uncorrected  | 106.4414 |       23.9926 |                     22.5407 |              28.1758 |              0.1193 |                 0.3052 |            0.6770 |               0.6155 |
| dnn_original | 122.0810 |       26.5253 |                     21.7276 |              27.1596 |              0.2676 |                 0.3017 |            0.6607 |               0.6399 |
| dnn_cap_1p25 | 116.2253 |       22.3506 |                     19.2304 |              24.0380 |              0.1569 |                 0.2534 |            0.7359 |               0.6775 |
| dnn_cap_1p50 | 119.4143 |       23.2228 |                     19.4472 |              24.3090 |              0.2057 |                 0.2707 |            0.7075 |               0.6648 |
| dnn_cap_1p75 | 121.0168 |       25.3271 |                     20.9286 |              26.1607 |              0.2453 |                 0.2885 |            0.6790 |               0.6516 |
| dnn_cap_2p00 | 121.6534 |       26.0954 |                     21.4506 |              26.8132 |              0.2626 |                 0.3032 |            0.6663 |               0.6394 |
| dnn_cap_2p50 | 122.2716 |       26.5564 |                     21.7192 |              27.1490 |              0.2702 |                 0.3022 |            0.6602 |               0.6399 |

## Event categories selected for inspection

| inspection_category                  |   count |
|:-------------------------------------|--------:|
| lowest_min_bjet_pt                   |      10 |
| near_30gev_threshold_below           |      10 |
| near_30gev_threshold_above           |      10 |
| already_high_uncorrected             |      10 |
| new_high_after_original_dnn          |      10 |
| largest_original_dnn_ratio           |      10 |
| cap1p25_fixes_original_dnn_high_tail |      10 |

