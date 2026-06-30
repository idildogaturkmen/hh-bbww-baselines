# Rough-normalized one-lepton HH→bbWW baseline status

This note summarizes the current rough-normalized one-lepton HH→bbWW baseline after correcting the MET feature extraction to use reconstructed MET consistently.

## Source tables and output locations

Main corrected recoMET feature tables:

- Lepton/MET features: `outputs/lepton_met_features_lepemu_rough_recoMET/onelep_pre_mbb_classifier_input_with_lepton_met_features.parquet`
- Simple topology features: `outputs/topology_features_lepemu_rough_recoMET/onelep_pre_mbb_classifier_input_with_topology_features.parquet`
- Hadronic-W/top features: `outputs/hadronic_w_top_features_lepemu_rough_recoMET/onelep_pre_mbb_classifier_input_with_hadronic_w_top_features.parquet`

Main corrected BDT outputs:

- recoMET lepton/MET BDT: `outputs/bdt_weighted_baseline_lepemu_rough_augmented_recoMET/`
- recoMET simple-topology BDT: `outputs/bdt_weighted_baseline_lepemu_rough_topology_recoMET/`
- recoMET hadronic-W/top BDT: `outputs/bdt_weighted_baseline_lepemu_rough_hadronic_w_top_recoMET/`

Summary comparison outputs:

- Model comparison: `outputs/bdt_recoMET_model_comparison.csv`
- Stability-constrained threshold comparison: `outputs/bdt_recoMET_stable_thresholds.csv`
- recoMET lepton/MET cut scan: `outputs/lepton_met_features_lepemu_rough_recoMET/recoMET_lepton_met_cut_scan.csv`
- DNN-ready input: `outputs/dnn_inputs_lepemu_rough_topology_recoMET/tabular_dnn_inputs.npz`

## Important MET correction

The old BDT studies used an inconsistent MET treatment in the added lepton/MET angular features.

The corrected script now uses this MET priority:

1. `FullReco_MET_MET` / `FullReco_MET_Phi`
2. `FullReco_PUPPIMET_MET` / `FullReco_PUPPIMET_Phi`
3. `FullReco_GenMissingET_MET` / `FullReco_GenMissingET_Phi` only as fallback

In the corrected run, all selected events used `FullReco_MET`:

| MET source | Events |
|---|---:|
| FullReco_MET | 33912 |

The direct old-vs-new feature comparison showed:

| Feature | New - old summary | Correlation |
|---|---|---:|
| `met` | mean = 0.0, std = 0.0 | 1.000 |
| `met_phi` | mean = 0.002, std = 2.049 | 0.364 |
| `mt_lep_met` | mean = 18.111, std = 35.534 | 0.743 |
| `dphi_lep_met` | mean = 0.292, std = 0.985 | 0.477 |

Interpretation: the MET magnitude was unchanged, but the MET direction changed substantially. Therefore the main correction affects angular and transverse-mass features such as:

- `met_phi`
- `dphi_lep_met`
- `mt_lep_met`
- `dphi_bb_met`
- `min_dphi_met_b`
- `mt_bb_met`

The old BDT result should therefore be treated as diagnostic only. The corrected recoMET results are the realistic baseline.

## Rough-normalized event sample

The recoMET DNN input preparation used:

| Quantity | Value |
|---|---:|
| Input rows | 33912 |
| Rows with safe rough physics weights | 33679 |
| Excluded rows | 233 |
| Number of DNN features | 57 |

Source: `outputs/dnn_inputs_lepemu_rough_topology_recoMET/split_summary.csv`

| Split | Raw events | Signal raw | Background raw | Signal weighted | Background weighted |
|---|---:|---:|---:|---:|---:|
| train | 20224 | 714 | 19510 | 42.683 | 1.5385e7 |
| val | 6609 | 263 | 6346 | 15.722 | 5.6746e6 |
| test | 6846 | 248 | 6598 | 14.826 | 5.1733e6 |

Source: `outputs/dnn_inputs_lepemu_rough_topology_recoMET/multiclass_summary.csv`

Approximate CMS-inspired classes prepared for the tabular DNN:

| Split | Class | Raw events | Weighted events |
|---|---|---:|---:|
| train | HH_ggF_like | 714 | 42.683 |
| train | Top_Higgs | 11809 | 4.0957e6 |
| train | WJets_Other | 7701 | 1.1290e7 |
| val | HH_ggF_like | 263 | 15.722 |
| val | Top_Higgs | 3840 | 1.3813e6 |
| val | WJets_Other | 2506 | 4.2933e6 |
| test | HH_ggF_like | 248 | 14.826 |
| test | Top_Higgs | 4024 | 1.3773e6 |
| test | WJets_Other | 2574 | 3.7960e6 |

## Corrected recoMET BDT comparison

Source: `outputs/bdt_recoMET_model_comparison.csv`

| Model | Split | Raw AUC | Weighted AUC | AP raw | Threshold | S weighted | B weighted | S/B | Asimov Z | Bkg Neff |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old simple topology, Gen/MET-phi diagnostic | val | 0.754 | 0.814 | 0.154 | 0.705 | 3.467 | 4.12e4 | 8.4e-5 | 0.0171 | 52.2 |
| old simple topology, Gen/MET-phi diagnostic | test | 0.747 | 0.747 | 0.137 | 0.705 | 2.690 | 3.92e4 | 6.9e-5 | 0.0136 | 51.8 |
| recoMET lepton/MET | val | 0.717 | 0.744 | 0.089 | 0.440 | 10.880 | 1.53e6 | 7.1e-6 | 0.00881 | 52.9 |
| recoMET lepton/MET | test | 0.706 | 0.676 | 0.076 | 0.440 | 9.445 | 1.98e6 | 4.8e-6 | 0.00671 | 30.1 |
| recoMET simple topology | val | 0.723 | 0.756 | 0.106 | 0.655 | 2.750 | 3.33e4 | 8.3e-5 | 0.0151 | 47.7 |
| recoMET simple topology | test | 0.700 | 0.672 | 0.076 | 0.655 | 1.793 | 1.64e5 | 1.1e-5 | 0.00443 | 6.18 |
| recoMET hadronic-W/top | val | 0.738 | 0.767 | 0.114 | 0.740 | 0.120 | 59.2 | 2.0e-3 | 0.0155 | 1.00 |
| recoMET hadronic-W/top | test | 0.711 | 0.691 | 0.081 | 0.740 | 0.060 | 961.7 | 6.2e-5 | 0.00193 | 3.14 |

Interpretation:

- After correcting the MET direction, BDT performance drops substantially relative to the old diagnostic result.
- The old result is not the nominal baseline because it used inconsistent MET angular information.
- The recoMET hadronic-W/top BDT has the best AUC among corrected models, but its validation-best high-score threshold is not usable because it selects only 2 signal and 3 background raw validation events, with background Neff about 1.
- The corrected nominal baseline must be chosen using stability-constrained thresholds, not the raw validation-best threshold.

## Stability-constrained threshold comparison

Source: `outputs/bdt_recoMET_stable_thresholds.csv`

The stability scan required both validation and test to satisfy minimum background Neff, minimum raw signal count, and minimum raw background count.

### Candidate 1: recoMET simple-topology high-purity category

Chosen threshold: `BDT >= 0.685`

Reason for choosing it:

- It is the best high-purity corrected recoMET option with both validation and test background Neff above 10.
- It gives the highest useful S/B among the corrected recoMET models without the extreme sparse-tail behavior of the hadronic-W/top validation-best threshold.
- It is not fully robust enough to be the only result, because test signal raw count is only 15 and test background Neff is about 26.

| Split | S raw | B raw | S weighted | B weighted | S/B | Z | Bkg Neff |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 26 | 101 | 1.554 | 1.270e4 | 1.22e-4 | 0.0138 | 23.8 |
| test | 15 | 100 | 0.897 | 1.924e4 | 4.66e-5 | 0.00646 | 26.0 |

Conclusion: use this as the corrected recoMET high-purity category, with the caveat that statistics are limited.

### Candidate 2: recoMET simple-topology broad stable category

Chosen threshold: `BDT >= 0.275`

Reason for choosing it:

- It passes background Neff >= 50 in both validation and test.
- It is useful as a broad/stable baseline or control-like region.
- It is not high-purity; S/B is only about 3.5e-6 on test.

| Split | S raw | B raw | S weighted | B weighted | S/B | Z | Bkg Neff |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 238 | 4070 | 14.228 | 3.348e6 | 4.25e-6 | 0.00778 | 50.5 |
| test | 219 | 4149 | 13.092 | 3.753e6 | 3.49e-6 | 0.00676 | 58.7 |

Conclusion: use this as a conservative stable recoMET reference point, not as the main signal-like region.

### Candidate 3: recoMET lepton/MET broad stable category

Chosen threshold: `BDT >= 0.440`

Reason for choosing it:

- It passes background Neff >= 25 in both validation and test.
- It has consistent validation/test behavior.
- It does not improve purity; it keeps large background yields.

| Split | S raw | B raw | S weighted | B weighted | S/B | Z | Bkg Neff |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 182 | 2258 | 10.880 | 1.526e6 | 7.13e-6 | 0.00881 | 52.9 |
| test | 158 | 2267 | 9.445 | 1.983e6 | 4.76e-6 | 0.00671 | 30.1 |

Conclusion: lepton/MET alone is stable but not sufficiently discriminating.

### Candidate 4: recoMET hadronic-W/top moderate category

Chosen threshold: `BDT >= 0.550`

Reason for not choosing it as nominal:

- It gives the best stable-ish hadronic-W/top point for Neff >= 10.
- It has better S/B than broad lepton/MET selections.
- However, it fails Neff >= 25 on the test split and has no threshold passing Neff >= 50 in both validation and test.
- Therefore the W/top model is useful as a feature study, but not currently the nominal corrected baseline.

| Split | S raw | B raw | S weighted | B weighted | S/B | Z | Bkg Neff |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 118 | 957 | 7.054 | 4.822e5 | 1.46e-5 | 0.0102 | 22.3 |
| test | 91 | 978 | 5.440 | 4.352e5 | 1.25e-5 | 0.00825 | 19.0 |

Conclusion: W/top variables improve AUC and have physics motivation, but the high-score tail remains statistics-limited.

## recoMET lepton/MET cut scan

Source: `outputs/lepton_met_features_lepemu_rough_recoMET/recoMET_lepton_met_cut_scan.csv`

Base selection:

- `ht >= 100`
- `80 <= mbb_pt_binned_scaled <= 150`

Best rough-Z row:

| mt_min | reliso_max | S raw | B raw | S weighted | B weighted | S/B | Z | Bkg Neff | Top backgrounds |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | none | 845 | 14078 | 50.514 | 9.880e6 | 5.1e-6 | 0.0161 | 152.4 | dy_zjets, ttbar, wjets, diboson |

Best isolation-only row among the listed cuts:

| mt_min | reliso_max | S raw | B raw | S weighted | B weighted | S/B | Z | Bkg Neff | Top backgrounds |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.2 | 725 | 11356 | 43.341 | 7.421e6 | 5.8e-6 | 0.0159 | 103.5 | dy_zjets, ttbar, wjets, diboson |

Effect of `mT > 30` with `reliso < 0.2`:

| mt_min | reliso_max | S raw | B raw | S weighted | B weighted | S/B | Z | Bkg Neff | Top backgrounds |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 30 | 0.2 | 536 | 8232 | 32.042 | 5.408e6 | 5.9e-6 | 0.0138 | 65.7 | dy_zjets, ttbar, wjets, diboson |

Interpretation:

- With corrected recoMET, simple lepton/MET cuts do not improve the rough weighted significance.
- Increasing mT reduces background but also removes signal, so Z decreases.
- Isolation cuts modestly improve S/B but do not improve Z enough to become the baseline.
- The dominant backgrounds after these cuts remain DY/Z+jets, ttbar, W+jets, and diboson.

## Excluded-background caveat

Before rough weighting, QCD/gamma/minbias/upsilon were excluded from the weighted yields because reliable rough cross-section metadata are not yet included.

Raw overlap check after the one-lepton and bb-like region showed:

Mass window only:

| Process group | Sample | Raw events |
|---|---|---:|
| qcd | QCD_HT50tobb | 72 |
| gamma | gamma_V | 6 |
| qcd | QCD_HT50toInf | 5 |
| gamma | gamma | 1 |

Mass window plus `mT > 30`:

| Process group | Sample | Raw events |
|---|---|---:|
| qcd | QCD_HT50tobb | 4 |
| gamma | gamma_V | 1 |

Interpretation:

- These excluded samples are still a normalization caveat.
- However, after recoMET mT requirements, their raw overlap becomes small.
- They should be revisited once sample metadata or better normalization information is available.

## Current corrected nominal interpretation

The corrected nominal baseline is the recoMET simple-topology BDT, but it should be presented as two regions:

1. Broad stable category: `recoMET simple topology, BDT >= 0.275`
   - Background Neff >= 50 in both validation and test.
   - Low S/B, useful as a stable reference/control-like region.

2. High-purity statistics-limited category: `recoMET simple topology, BDT >= 0.685`
   - Better S/B.
   - Limited signal and background effective statistics.

The recoMET hadronic-W/top BDT is not selected as nominal because its validation-best threshold is a sparse-tail artifact, and its more stable thresholds do not pass stronger Neff requirements.

## DNN preparation status

A DNN-ready tabular input has been prepared from the corrected recoMET simple-topology table.

Output:

- `outputs/dnn_inputs_lepemu_rough_topology_recoMET/tabular_dnn_inputs.npz`

The input contains:

- 33679 events with safe rough weights
- 57 standardized features
- binary signal/background labels
- approximate CMS-inspired multiclass labels:
  - `HH_ggF_like`
  - `Top_Higgs`
  - `WJets_Other`
- physics weights
- train/val/test split labels
- event metadata

This is ready for a controlled weighted tabular DNN comparison against the corrected recoMET BDT. A particle-level or LBN-style DNN should come after this controlled comparison.

## Likely next steps

1. Keep the old Gen/MET-phi BDT only as a diagnostic reference.
2. Use the corrected recoMET simple-topology BDT as the current nominal baseline.
3. Train a weighted tabular DNN on the corrected recoMET inputs to test whether a DNN improves over the BDT with the same features, weights, and splits.
4. If the tabular DNN does not improve clearly, focus on normalization/control regions rather than a more complex architecture.
5. If Harvey specifically wants a CMS-like DNN, move toward a multiclass DNN and later an LBN-style four-vector input, but keep the corrected recoMET BDT as the baseline reference.
