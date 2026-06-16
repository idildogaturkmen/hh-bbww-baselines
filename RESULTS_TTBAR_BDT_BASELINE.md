# HH -> bbWW vs semileptonic ttbar BDT baseline

## Dataset

Signal:
- HH -> bbWW from COLLIDE-1M
- Usable matched HH events: 12,171
- Train/validation/test split: 9,736 / 1,217 / 1,218
- Signal features are built from the first `MAX_JETS = 12` AK4 jets saved in `outputs/hbb_npz/*.npz`.

Background:
- Semileptonic ttbar from COLLIDE-1M
- Source files:
  - `tt0123j_5f_ckm_LO_MLM_semiLeptonic-NEVENT10000-RS30000001.parquet`
  - `tt0123j_5f_ckm_LO_MLM_semiLeptonic-NEVENT10000-RS30000002.parquet`
- Balanced to the HH sample in each split.
- The ttbar feature builder now applies the same `MAX_JETS = 12` truncation as the HH preprocessing before computing `n_jets`, HT, b-tag ordering, and dijet observables.

The `MAX_JETS = 12` choice is data-informed by `scripts/diagnose_max_jets.py`: it retains 12,171 / 12,232 = 99.5% of truth-matchable H -> bb events while limiting unordered jet-pair combinatorics to 66 candidate pairs per event.

## Input Features

The BDT uses 8 event-level AK4 jet features:

1. Number of AK4 jets after the 12-jet cap
2. Leading jet pT
3. Subleading jet pT
4. Highest b-tag score
5. Second-highest b-tag score
6. Invariant mass of the two highest-b-tag jets
7. DeltaR between the two highest-b-tag jets
8. HT, the scalar sum of retained AK4 jet pT

## Model

- Classifier: `GradientBoostingClassifier`
- `n_estimators = 200`
- `learning_rate = 0.05`
- `max_depth = 3`
- `subsample = 0.8`
- nominal `random_state = 42`

The model is trained on the training split. Working-point thresholds are chosen using the validation split and evaluated on the held-out test split.

## Corrected Performance

After applying consistent `MAX_JETS = 12` treatment to both signal and ttbar, the nominal BDT performance is:

| Split | AUC |
|---|---:|
| Train | 0.7200 |
| Validation | 0.7012 |
| Test | 0.7122 |

The previous test AUC was 0.7623. The drop to 0.7122 after consistent jet truncation indicates that the earlier model likely benefited from a preprocessing mismatch, especially through jet multiplicity. This corrected result is the more defensible baseline.

## Working Points Evaluated On Test

| Validation target signal efficiency | Test signal efficiency | Test ttbar efficiency | Test ttbar rejection |
|---:|---:|---:|---:|
| 0.90 | 0.9097 +/- 0.0082 | 0.7184 +/- 0.0129 | 1.39 |
| 0.70 | 0.7020 +/- 0.0131 | 0.4039 +/- 0.0141 | 2.48 |
| 0.50 | 0.5107 +/- 0.0143 | 0.2151 +/- 0.0118 | 4.65 |

## Feature Importance

| Feature | Importance |
|---|---:|
| mbb_top2_btag | 0.2799 |
| subleading_jet_pt | 0.1948 |
| HT | 0.1798 |
| second_highest_btag | 0.1259 |
| deltaR_top2_btag | 0.1217 |
| leading_jet_pt | 0.0600 |
| n_jets | 0.0198 |
| highest_btag | 0.0181 |

The corrected feature ranking is substantially different from the earlier result: `n_jets` is no longer dominant once ttbar and HH are treated with the same jet cap.

## Seed Stability

`scripts/run_bdt_seed_stability.py` was added and run over seeds `42, 123, 2024, 2025, 2026` using the corrected fixed train/validation/test feature files. This varies the BDT `random_state`; it is not yet a full resampling or split-stability study.

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| Train AUC | 0.7194 | 0.0005 | 0.7189 | 0.7200 |
| Validation AUC | 0.7008 | 0.0013 | 0.6989 | 0.7024 |
| Test AUC | 0.7110 | 0.0012 | 0.7101 | 0.7124 |
| Test ttbar rejection at ~90% signal efficiency | 1.38 | 0.01 | 1.37 | 1.39 |
| Test ttbar rejection at ~70% signal efficiency | 2.46 | 0.06 | 2.39 | 2.53 |
| Test ttbar rejection at ~50% signal efficiency | 4.70 | 0.06 | 4.63 | 4.78 |

Outputs:
- `outputs/ttbar_classifier/seed_stability.csv`
- `outputs/ttbar_classifier/seed_stability_summary.json`
- `outputs/plots/ttbar_classifier/seed_stability.png`

## Interpretation

The corrected BDT provides moderate HH -> bbWW vs semileptonic ttbar separation using simple AK4 observables. The stability scan suggests the result is not sensitive to the BDT random seed on the fixed feature split. However, the lower corrected AUC and the reduced importance of `n_jets` show why consistent preprocessing is essential before comparing reconstruction or classification methods.

This baseline is intended as an interpretable reference model, not the final CMS-style event classifier. It uses COLLIDE-1M samples only, limited backgrounds, no cross-section weighting, and no systematic uncertainties.
