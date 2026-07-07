# HH→4b Delphes + ML progress summary — 2026-07-07

## Bottom-line messages

1. I now have a working LPC workflow for generating HH→4b signal and background samples through MG5/Pythia8/Delphes.
2. I validated two 10k signal samples: VBF HH→4b SM and ggF HH→4b HEFT/effective approximation.
3. I validated three 10k background samples: QCD bbbb, Zbbbb, and ttbar.
4. The candidate-level dataset contains 2104 HH4b candidate rows: 1335 signal and 769 background.
5. A first candidate-level BDT sanity check gives an unweighted AUC of 0.748.
6. This is not yet a final analysis result; it is a validation milestone showing that the simulation, reconstruction, feature extraction, and first ML baseline are functioning.

## Current 10k signal/background samples

| sample        |   events |   xsec_pb_median |   n_ge4_jets |   n_ge4_bjets |   candidate_rows |   median_mbb1 |   median_mbb2 |   median_avg_mbb |   median_delta_mbb |   median_mhh |
|:--------------|---------:|-----------------:|-------------:|--------------:|-----------------:|--------------:|--------------:|-----------------:|-------------------:|-------------:|
| VBF HH4b      |    10000 |      0.000947731 |         6576 |           694 |              694 |       125.323 |       115.304 |          120.842 |            24.7081 |      439.885 |
| ggF HEFT HH4b |    10000 |      0.000993955 |         6040 |           641 |              641 |       125.115 |       110.517 |          117.028 |            22.7038 |      322.517 |
| QCD bbbb      |    10000 |    345.156       |         3220 |           400 |              400 |       131.515 |       109.857 |          123.265 |            38.0273 |      368.023 |
| Zbbbb         |    10000 |      6.91164     |         3204 |           335 |              335 |       129.338 |       109.858 |          122.127 |            42.0768 |      346.476 |
| ttbar         |    10000 |    512.398       |         6688 |            34 |               34 |       128.836 |       117.889 |          126.689 |            44.7345 |      436.192 |

## Candidate-level training table

| process       |   label | process_group   |   rows |   median_mbb1 |   median_mbb2 |   median_avg_mbb |   median_delta_mbb |   median_mhh |       xsec_pb |
|:--------------|--------:|:----------------|-------:|--------------:|--------------:|-----------------:|-------------------:|-------------:|--------------:|
| ggF HEFT HH4b |       1 | signal          |    641 |       125.115 |       110.517 |          117.028 |            22.7038 |      322.517 |   0.000993955 |
| QCD bbbb      |       0 | background      |    400 |       131.515 |       109.857 |          123.265 |            38.0273 |      368.023 | 345.156       |
| ttbar         |       0 | background      |     34 |       128.836 |       117.889 |          126.689 |            44.7345 |      436.192 | 512.398       |
| VBF HH4b      |       1 | signal          |    694 |       125.323 |       115.304 |          120.842 |            24.7081 |      439.885 |   0.000947731 |
| Zbbbb         |       0 | background      |    335 |       129.338 |       109.858 |          122.127 |            42.0768 |      346.476 |   6.91164     |

## First BDT sanity check

- Input table: `/uscms_data/d3/iturkmen/hh4b_delphes/ml/hh4b_candidate_training_10k_v0.parquet`
- Total candidates: 2104
- Train candidates: 1472
- Test candidates: 632
- Signal rows in test: 401
- Background rows in test: 231
- Unweighted AUC: 0.748

Features used:
n_selected_bjets, mbb1, mbb2, avg_mbb, delta_mbb, mhh, drbb1, drbb2, j1_pt, j2_pt, j3_pt, j4_pt

## Caveats

- The ggF sample used for bulk production is HEFT/effective ggF, not full SM loop-induced ggF.
- A 20-event SM loop-induced ggF smoke test has passed, but it is not yet used for bulk production.
- QCD bbbb and Zbbbb samples are preselected/fiducial heavy-flavor samples, not inclusive QCD.
- ttbar has very low four-b-tag candidate yield at 10k, as expected for ordinary ttbar.
- Current BDT is only a candidate-level sanity check, not a final optimized classifier.
- For larger production, ROOT and HepMC cannot be kept for every event due to storage limits. Future 100k/1M production should keep compact parquet/features and only retain small ROOT validation subsets.

## Useful plots

Plots are saved in:

`/uscms_data/d3/iturkmen/hh4b_delphes/plots/harvey_update_2026_07_07`

Recommended plots to show first:
- `candidate_rows_by_process.png`
- `avg_mbb_signal_vs_background.png`
- `delta_mbb_signal_vs_background.png`
- `mhh_signal_vs_background.png`
- `bdt_score_signal_vs_background.png`
- `bdt_roc_curve.png`

## Next steps

1. Produce storage-safe 100k samples using shard-level parquet output.
2. Add event-level features and possibly jet-level arrays, not only HH candidate features.
3. Compare cut baseline, BDT, DNN, and SPA-Net-style assignment-aware models.
4. Use simulated backgrounds to test ABCD closure before claiming any data-driven background estimate.
5. Keep SM loop-induced ggF as a validation/shape cross-check unless it scales efficiently.
