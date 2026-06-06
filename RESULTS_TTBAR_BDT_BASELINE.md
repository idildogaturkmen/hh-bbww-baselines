# HH → bbWW vs semileptonic ttbar BDT baseline

## Dataset

Signal:
- HH → bbWW from COLLIDE-1M
- Usable matched HH events: 12,171
- Train/validation/test split: 9,736 / 1,217 / 1,218

Background:
- Semileptonic ttbar from COLLIDE-1M
- Source files:
  - `tt0123j_5f_ckm_LO_MLM_semiLeptonic-NEVENT10000-RS30000001.parquet`
  - `tt0123j_5f_ckm_LO_MLM_semiLeptonic-NEVENT10000-RS30000002.parquet`
- Balanced to the HH sample in each split.

## Input features

The BDT uses 8 event-level AK4 jet features:

1. Number of AK4 jets
2. Leading jet pT
3. Subleading jet pT
4. Highest b-tag score
5. Second-highest b-tag score
6. Invariant mass of the two highest-b-tag jets
7. DeltaR between the two highest-b-tag jets
8. HT, the scalar sum of AK4 jet pT

## Model

- Classifier: `GradientBoostingClassifier`
- `n_estimators = 200`
- `learning_rate = 0.05`
- `max_depth = 3`
- `subsample = 0.8`
- `random_state = 42`

The model is trained on the training split. Working-point thresholds are chosen using the validation split and evaluated on the held-out test split.

## Performance

AUC:

| Split | AUC |
|---|---:|
| Train | 0.7656 |
| Validation | 0.7555 |
| Test | 0.7623 |

The similar train, validation, and test AUC values indicate no obvious severe overtraining.

## Working points evaluated on test

| Validation target signal efficiency | Test signal efficiency | Test ttbar efficiency | Test ttbar rejection |
|---:|---:|---:|---:|
| 0.90 | 0.8998 ± 0.0086 | 0.5936 ± 0.0141 | 1.68 |
| 0.70 | 0.7011 ± 0.0131 | 0.3374 ± 0.0135 | 2.96 |
| 0.50 | 0.5082 ± 0.0143 | 0.1773 ± 0.0109 | 5.64 |

## Feature importance

| Feature | Importance |
|---|---:|
| n_jets | 0.4457 |
| mbb_top2_btag | 0.1742 |
| subleading_jet_pt | 0.1122 |
| second_highest_btag | 0.0888 |
| HT | 0.0716 |
| deltaR_top2_btag | 0.0680 |
| leading_jet_pt | 0.0307 |
| highest_btag | 0.0088 |

## Interpretation

The BDT provides a moderate but useful event-level separation between HH → bbWW and semileptonic ttbar using simple AK4 observables. The dominant feature is jet multiplicity, followed by the invariant mass of the two highest-b-tag jets and subleading jet pT. This suggests that semileptonic ttbar can be partially suppressed using global event topology and b-jet-pair kinematics, but more powerful models or richer inputs are needed for stronger background rejection.

This baseline is intended as an interpretable reference model, not the final CMS-style event classifier.
