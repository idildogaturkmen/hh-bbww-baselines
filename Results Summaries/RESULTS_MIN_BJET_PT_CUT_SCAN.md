# Lower-b-jet pT Cut Scan for H→bb Mass Reconstruction

## Goal

This study checks how the H→bb mass reconstruction changes when requiring the softer of the two truth-matched H→bb jets to pass a minimum pT threshold. This directly addresses the question of how efficiency, mass resolution, and tail fractions change when removing events with a low-pT second b jet.

## Definition of the cut

For each event, define:

min_bjet_pt = min(pT_b1, pT_b2)

where b1 and b2 are the two truth-matched reconstructed jets assigned to H→bb.

The scan applies cuts at 0, 30, 40, 50, and 60 GeV.

## Dataset

- Input file: data/spanet_hbb/spanet_hbb_test.h5
- Number of test events: 1969
- Compared mass definitions:
  - uncorrected H→bb mass
  - DNN-corrected H→bb mass

## Efficiency

Efficiency is the fraction of truth-matched H→bb events that survive the lower-b-jet pT cut.

| min b-jet pT cut [GeV] | Events kept | Efficiency |
|---:|---:|---:|
| 0 | 1969 | 1.000 |
| 30 | 1521 | 0.772 |
| 40 | 1111 | 0.564 |
| 50 | 740 | 0.376 |
| 60 | 483 | 0.245 |

## Mass resolution

The resolution is measured using the central 68% half-width of the m_bb distribution:

width68 = (84th percentile - 16th percentile) / 2

A smaller value means a narrower reconstructed H→bb mass peak.

| min b-jet pT cut [GeV] | Uncorrected width68 [GeV] | DNN-corrected width68 [GeV] |
|---:|---:|---:|
| 0 | 23.99 | 26.53 |
| 30 | 23.08 | 26.68 |
| 40 | 24.15 | 28.52 |
| 50 | 28.85 | 32.21 |
| 60 | 33.90 | 36.84 |

## Tail fractions

The main mass window used here is 90–140 GeV. The tail fraction is the fraction of events outside this window.

| min b-jet pT cut [GeV] | Version | Fraction in 90–140 | Outside 90–140 | Low tail <90 | High tail >140 |
|---:|---|---:|---:|---:|---:|
| 0 | uncorrected | 0.677 | 0.323 | 0.204 | 0.119 |
| 0 | DNN corrected | 0.661 | 0.339 | 0.072 | 0.268 |
| 30 | uncorrected | 0.724 | 0.276 | 0.132 | 0.144 |
| 30 | DNN corrected | 0.655 | 0.345 | 0.049 | 0.296 |
| 40 | uncorrected | 0.734 | 0.266 | 0.080 | 0.185 |
| 40 | DNN corrected | 0.628 | 0.372 | 0.032 | 0.339 |
| 50 | uncorrected | 0.704 | 0.296 | 0.055 | 0.241 |
| 50 | DNN corrected | 0.573 | 0.427 | 0.024 | 0.403 |
| 60 | uncorrected | 0.646 | 0.354 | 0.052 | 0.302 |
| 60 | DNN corrected | 0.518 | 0.482 | 0.023 | 0.460 |

## Main observations

1. The lower-b-jet pT requirement has a large efficiency cost. A 30 GeV threshold keeps about 77% of events, while a 60 GeV threshold keeps only about 25%.

2. The uncorrected m_bb median moves closer to 125 GeV as the lower-b-jet pT threshold increases.

3. The DNN correction shifts the mass scale upward, reducing the low-mass tail but creating a significant high-mass tail.

4. The best compromise in the uncorrected case appears to be around the 30–40 GeV threshold, where the efficiency remains moderate and the 90–140 GeV window fraction is highest.

5. In this SPA-Net test-file diagnostic, the DNN-corrected mass distribution is broader than expected from the earlier DNN-response study, so the corrected-feature construction and/or dataset split should be cross-checked against the original DNN-evaluation setup.

## Plots

Generated plots:

- outputs/plots/min_bjet_pt_cut_scan/test_min_bjet_pt_distribution.png
- outputs/plots/min_bjet_pt_cut_scan/test_efficiency_vs_min_bjet_pt_cut.png
- outputs/plots/min_bjet_pt_cut_scan/test_mbb_uncorrected_vs_dnn_corrected.png
- outputs/plots/min_bjet_pt_cut_scan/test_mbb_width68_vs_min_bjet_pt_cut.png
- outputs/plots/min_bjet_pt_cut_scan/test_tail_fraction_vs_min_bjet_pt_cut.png
- outputs/plots/min_bjet_pt_cut_scan/test_corrected_mbb_by_min_bjet_pt_cut.png

## High-mass tail investigation
The event scan shows that the high-mass tail is not a single effect. Some events already have very large uncorrected m_bb, often with large ΔR between the matched H→bb jets. These should be inspected for matching or topology issues. A second class of events has moderate/high uncorrected m_bb but receives a large response correction on one jet, especially the softer jet, which pushes corrected m_bb further into the high-mass tail.