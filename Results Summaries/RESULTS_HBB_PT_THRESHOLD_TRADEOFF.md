# H->bb pT Threshold Tradeoff Study

This is a signal-only follow-up to the detector-response diagnostic. It studies how cuts on the lower-pT truth-matched H->bb b jet change the reconstructed m_bb scale and the retained signal fraction.

**Important caveat:** this uses truth-matched b jets, so it is not yet a directly data-applicable selection. For a background/data-like version, the analogous observable would need to be something like the lower pT of the selected b-tagged jet pair.

## Input

- Input CSV: `outputs/hbb_detector_response/event_response_diagnostics.csv`
- Total usable truth-matched HH events in input: 12174

## Main takeaway

- With no additional lower-b-jet pT cut, the median matched m_bb is 103.18 GeV.
- The tested cut with median m_bb closest to 125 GeV is `lower_pT_ge_70`, with median m_bb = 125.42 GeV and signal retention = 0.137.
- The first tested threshold where the median m_bb reaches or exceeds 125 GeV is `lower_pT_ge_70`, retaining 0.137 of usable HH events.

This supports the interpretation that harder lower-b-jet pT requirements can recover the m_bb scale, but the efficiency cost can be large.

## Summary table

| Cut | n events | Signal retention | Mean m_bb [GeV] | Median m_bb [GeV] | Bootstrap median 16-84% [GeV] | 90 < m_bb < 140 | 100 < m_bb < 150 | Median lower response |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_cut | 12174 | 1.000 | 108.71 | 103.18 | 102.95–103.45 | 0.615 | 0.477 | 0.765 |
| lower_pT_ge_20 | 10971 | 0.901 | 111.43 | 105.39 | 105.16–105.73 | 0.647 | 0.509 | 0.787 |
| lower_pT_ge_25 | 9603 | 0.789 | 114.59 | 108.03 | 107.75–108.26 | 0.679 | 0.545 | 0.812 |
| lower_pT_ge_30 | 8226 | 0.676 | 117.77 | 110.28 | 109.98–110.62 | 0.702 | 0.580 | 0.836 |
| lower_pT_ge_40 | 5781 | 0.475 | 124.47 | 114.89 | 114.59–115.23 | 0.715 | 0.632 | 0.881 |
| lower_pT_ge_50 | 3828 | 0.314 | 132.01 | 119.34 | 118.99–119.76 | 0.689 | 0.645 | 0.913 |
| lower_pT_ge_60 | 2489 | 0.204 | 139.55 | 122.91 | 122.33–123.41 | 0.646 | 0.618 | 0.931 |
| lower_pT_ge_70 | 1664 | 0.137 | 147.37 | 125.42 | 124.75–126.11 | 0.621 | 0.600 | 0.944 |
| lower_pT_ge_80 | 1186 | 0.097 | 153.46 | 127.44 | 126.56–128.62 | 0.608 | 0.595 | 0.958 |
| lower_pT_ge_90 | 880 | 0.072 | 158.72 | 128.91 | 127.89–131.07 | 0.597 | 0.575 | 0.968 |
| lower_pT_ge_110 | 490 | 0.040 | 170.22 | 132.86 | 131.85–134.09 | 0.553 | 0.545 | 0.987 |

## How to interpret

- If the median m_bb increases with the lower-b-jet pT cut, then the low inclusive mass is strongly tied to soft matched b jets.
- If the signal retention drops quickly, then a hard pT cut is not a simple fix, even if it improves the mass peak.
- If the response also increases with the pT cut, this supports the detector/reconstruction-response explanation.
- The mean can become much larger than the median at high pT cuts because the remaining sample has broad high-mass tails, so the median is usually the more stable mass-scale diagnostic.

## Outputs

- `outputs/hbb_pt_threshold_tradeoff/pt_threshold_summary.csv`
- `outputs/hbb_pt_threshold_tradeoff/summary.json`
- `outputs/plots/hbb_pt_threshold_tradeoff/median_mean_mbb_vs_pt_cut.png`
- `outputs/plots/hbb_pt_threshold_tradeoff/signal_retention_vs_pt_cut.png`
- `outputs/plots/hbb_pt_threshold_tradeoff/mass_window_fraction_vs_pt_cut.png`
- `outputs/plots/hbb_pt_threshold_tradeoff/response_vs_pt_cut.png`
- `outputs/plots/hbb_pt_threshold_tradeoff/mbb_distributions_for_selected_pt_cuts.png`

## Conclusion
A lower matched-b-jet pT requirement can recover the reconstructed m_bb scale, but only at a large signal-efficiency cost. The cut that gives median m_bb closest to 125 GeV, lower pT ≥ 70 GeV, retains only 13.7% of usable HH events. Therefore, a hard lower-b-jet pT cut is useful as a diagnostic, but it is probably not an acceptable final solution. This motivates studying response correction/regression instead of simply cutting away low-pT events.