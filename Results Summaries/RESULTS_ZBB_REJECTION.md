# HH -> bbWW vs Z->bb Rejection Baseline

This study tests whether a dedicated rejection of Z -> bb can help. It uses COLLIDE-1M HH -> bbWW as signal and `ZJetsTobb_13TeV-madgraphMLM-pythia8` as background.

## Command

```bash
/tmp/hh-bbww-venv/bin/python scripts/run_zbb_rejection.py --progress-every 1000 --stream-attempts 5
```

## Inputs

- HH signal: existing fixed `outputs/hbb_npz/{train,val,test}.npz` split.
- Z background: separate ZJetsTobb shards for train, validation, and test.
- Both samples use the first `MAX_JETS = 12` AK4 jets before feature construction.
- Labels: HH = 1, ZJetsTobb = 0.

Features:

- `n_jets`
- `leading_jet_pt`
- `subleading_jet_pt`
- `highest_btag`
- `second_highest_btag`
- `mbb_top2_btag`
- `deltaR_top2_btag`
- `HT`

## Dataset Counts

| Split | HH | ZJetsTobb | Total |
|---|---:|---:|---:|
| train | 9736 | 9736 | 19472 |
| val | 1217 | 1217 | 2434 |
| test | 1218 | 1218 | 2436 |

## Test AUC

| Method | Train AUC | Validation AUC | Test AUC |
|---|---:|---:|---:|
| mass_only_top2_btag_mbb | 0.5741 | 0.5640 | 0.5578 |
| full_bdt | 0.9550 | 0.9369 | 0.9403 |
| no_mbb_bdt | 0.9526 | 0.9338 | 0.9364 |

## Validation-Chosen Working Points Evaluated On Test

| Method | Target HH eff on validation | Test HH eff | Test Z eff | Test Z rejection |
|---|---:|---:|---:|---:|
| mass_only_top2_btag_mbb | 0.90 | 0.8998 +/- 0.0086 | 0.7980 +/- 0.0115 | 1.25 |
| mass_only_top2_btag_mbb | 0.70 | 0.7110 +/- 0.0130 | 0.5222 +/- 0.0143 | 1.92 |
| mass_only_top2_btag_mbb | 0.50 | 0.4803 +/- 0.0143 | 0.3949 +/- 0.0140 | 2.53 |
| full_bdt | 0.90 | 0.9015 +/- 0.0085 | 0.1658 +/- 0.0107 | 6.03 |
| full_bdt | 0.70 | 0.6921 +/- 0.0132 | 0.0632 +/- 0.0070 | 15.82 |
| full_bdt | 0.50 | 0.5312 +/- 0.0143 | 0.0271 +/- 0.0047 | 36.91 |
| no_mbb_bdt | 0.90 | 0.9039 +/- 0.0084 | 0.1650 +/- 0.0106 | 6.06 |
| no_mbb_bdt | 0.70 | 0.6790 +/- 0.0134 | 0.0640 +/- 0.0070 | 15.62 |
| no_mbb_bdt | 0.50 | 0.5074 +/- 0.0143 | 0.0312 +/- 0.0050 | 32.05 |

## Full-BDT Feature Importance

| Feature | Importance |
|---|---:|
| HT | 0.8948 |
| leading_jet_pt | 0.0221 |
| n_jets | 0.0212 |
| subleading_jet_pt | 0.0146 |
| mbb_top2_btag | 0.0136 |
| highest_btag | 0.0132 |
| second_highest_btag | 0.0110 |
| deltaR_top2_btag | 0.0094 |

## Interpretation

The mass-only score is a deliberately simple one-sided test using the top-2-btag dijet mass, with larger values treated as more HH-like. It gives only weak Z rejection. The trained BDTs give much stronger rejection, and the no-mbb BDT is nearly as good as the full BDT. This means the current rejection is driven mostly by global event kinematics such as HT, not by a clean m_bb-only Z veto.

Physically, a dedicated ZJetsTobb rejection can help in this COLLIDE-1M baseline, but the result should be treated as a first stress test. ZJetsTobb is inclusive here, the classes are balanced, and the study does not include cross-section weights, full event selections, additional backgrounds, or systematic uncertainties.

## Outputs

- `outputs/zbb_rejection/features_{train,val,test}.npz`
- `outputs/zbb_rejection/metrics.csv`
- `outputs/zbb_rejection/working_points.csv`
- `outputs/zbb_rejection/dataset_summary.json`
- `outputs/plots/zbb_rejection/roc_curve_test.png`
- `outputs/plots/zbb_rejection/background_rejection_test.png`
