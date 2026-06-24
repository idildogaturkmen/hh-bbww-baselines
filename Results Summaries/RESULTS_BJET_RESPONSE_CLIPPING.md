# B-jet Response Correction Clipping Study

## Goal

The original DNN b-jet response correction improved the low-mass side of the H→bb mass distribution, but it also produced a high-mass tail. This study tests whether capping the DNN response correction can reduce the high-mass tail while keeping the useful upward correction.

The clipping is applied to the per-jet DNN response correction factor before recomputing the corrected H→bb mass.

## Correction versions tested

| Version | Meaning |
|---|---|
| `uncorrected` | No DNN correction applied |
| `dnn_original` | Original DNN correction with no clipping |
| `dnn_cap_1p25` | DNN response correction capped at 1.25 |
| `dnn_cap_1p50` | DNN response correction capped at 1.50 |
| `dnn_cap_1p75` | DNN response correction capped at 1.75 |
| `dnn_cap_2p00` | DNN response correction capped at 2.00 |
| `dnn_cap_2p50` | DNN response correction capped at 2.50 |

## No-cut performance summary

| Version | Median m_bb [GeV] | Width68 [GeV] | Fraction in 90–140 | Low tail <90 | High tail >140 |
|---|---:|---:|---:|---:|---:|
| `uncorrected` | 106.4 | 24.0 | 0.677 | 0.204 | 0.119 |
| `dnn_original` | 122.1 | 26.5 | 0.661 | 0.072 | 0.268 |
| `dnn_cap_1p25` | 116.2 | 22.4 | 0.736 | 0.107 | 0.157 |
| `dnn_cap_1p50` | 119.4 | 23.2 | 0.707 | 0.087 | 0.206 |
| `dnn_cap_1p75` | 121.0 | 25.3 | 0.679 | 0.076 | 0.245 |
| `dnn_cap_2p00` | 121.7 | 26.1 | 0.666 | 0.071 | 0.263 |
| `dnn_cap_2p50` | 122.3 | 26.6 | 0.660 | 0.070 | 0.270 |

## Main observation

The DNN correction shifts the H→bb mass upward. This helps recover low-mass events, but the original uncapped correction also creates a large high-mass tail.

The response cap at **1.25** gives the best tested compromise:

| Metric | Original DNN | DNN cap 1.25 |
|---|---:|---:|
| Median m_bb [GeV] | 122.1 | 116.2 |
| Width68 [GeV] | 26.5 | 22.4 |
| Fraction in 90–140 | 0.661 | 0.736 |
| Low tail <90 | 0.072 | 0.107 |
| High tail >140 | 0.268 | 0.157 |

Compared with the original DNN correction, the 1.25 cap reduces the high-mass tail from **26.8% to 15.7%**, improves the 90–140 GeV window fraction from **66.1% to 73.6%**, and improves the width68 from **26.5 GeV to 22.4 GeV**.

## Event category comparison

### Original DNN correction

| Category | Events | Fraction | Interpretation |
|---|---:|---:|---|
| `already_high_uncorrected` | 235 | 0.119 | Events already above 140 GeV before correction |
| `new_high_after_correction` | 307 | 0.156 | Events pushed above 140 GeV by the correction |
| `fixed_low_to_window` | 241 | 0.122 | Events moved from below 90 GeV into the 90–140 GeV window |
| `stays_in_window` | 1045 | 0.531 | Events that stayed in the 90–140 GeV window |
| `stays_low` | 136 | 0.069 | Events that stayed below 90 GeV |

### DNN correction capped at 1.25

| Category | Events | Fraction | Interpretation |
|---|---:|---:|---|
| `already_high_uncorrected` | 235 | 0.119 | Events already above 140 GeV before correction |
| `new_high_after_correction` | 89 | 0.045 | Events pushed above 140 GeV by the correction |
| `fixed_low_to_window` | 195 | 0.099 | Events moved from below 90 GeV into the 90–140 GeV window |
| `stays_in_window` | 1239 | 0.629 | Events that stayed in the 90–140 GeV window |
| `stays_low` | 206 | 0.105 | Events that stayed below 90 GeV |

The original DNN correction fixes more low-mass events, but it also creates many more high-mass-tail events. The 1.25 cap fixes slightly fewer low-mass events, but it substantially reduces overcorrection and keeps more events inside the 90–140 GeV window.

## Interpretation

The DNN correction is useful, but the uncapped correction is not stable because large response corrections can overcorrect some jets and push events into the high-mass tail.

A response correction cap around **1.25** appears to control the high-mass tail while preserving useful correction behavior. This suggests that the correction should not be used blindly in its original uncapped form. A clipped or calibrated version may be more appropriate for the H→bb mass reconstruction study.

## Plots

The following plots were produced:

- `outputs/plots/bjet_response_clipping/test_mbb_clipping_overlay.png`
- `outputs/plots/bjet_response_clipping/test_width68_vs_response_cap.png`
- `outputs/plots/bjet_response_clipping/test_high_tail_vs_response_cap.png`
- `outputs/plots/bjet_response_clipping/test_frac_90_140_vs_response_cap.png`
