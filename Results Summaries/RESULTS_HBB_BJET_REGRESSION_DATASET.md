# H->bb B-Jet Regression Dataset

This dataset is built for a CMS-inspired b-jet response regression study on COLLIDE-1M HH->bbWW.

Each row is one truth-matched H->bb reconstructed AK4 b jet. The correction target is the matched GenJetAK4 response:

`target = log(GenJetAK4 pT / RecoJetAK4 pT)`

The corrected pT can later be computed as:

`corrected pT = reco pT * exp(model prediction)`

## Important caveat

This is simulation-only and CMS-inspired, not a full CMS b-jet regression reproduction. The dataset includes PF-candidate and jet-constituent information, but does not include explicit secondary-vertex variables.

## Reco AK4 jet object selection

- max reco jets: -1 (`-1` means all selected reco AK4 jets)
- minimum reco jet pT: 20.0 GeV
- maximum |eta|: 2.5 (`-1` means no eta cut)

## Event and row counts

- raw HH events scanned: 17967
- usable events with two matched reco jets and two matched GenJetAK4 targets: 9883
- usable b-jet rows: 19766
- unmatched gen b -> reco AK4: 7779
- unmatched reco AK4 -> GenJetAK4: 84
- same reco jet matched to both gen b quarks: 40
- same GenJetAK4 matched to both reco jets: 6

## Split counts

| Split | Jet rows | Unique events |
|---|---:|---:|
| train | 15812 | 7906 |
| val | 1976 | 988 |
| test | 1978 | 989 |

## Target summary

| Quantity | n | Mean | Median | Std | 16% | 84% |
|---|---:|---:|---:|---:|---:|---:|
| log response target | 19766 | 0.0884 | 0.0702 | 0.1964 | -0.0637 | 0.2520 |
| GenJet/reco response | 19766 | 1.1150 | 1.0727 | 0.2554 | 0.9383 | 1.2866 |

## Uncorrected event-level m_bb summary

| n events | Mean [GeV] | Median [GeV] | Std [GeV] | 16% [GeV] | 84% [GeV] |
|---:|---:|---:|---:|---:|---:|
| 9883 | 111.65 | 105.48 | 41.74 | 83.65 | 131.92 |

## Feature groups

- Reco AK4 jet kinematics and b-tag quantities
- Event-level HT, MET, PUPPIMET, and primary vertex quantities
- PF-candidate constituent composition and shape features
- PF muon/electron-in-jet features
- Tight muon/electron proximity features

## Dataset configuration 
max_reco_jets = -1
min_reco_pt = 20 GeV
max_abs_eta = 2.5
DR gen b -> reco AK4 = 0.4
DR reco AK4 -> GenJetAK4 = 0.4

For the b-jet response regression dataset, all reconstructed AK4 jets passing pT > 20 GeV and |eta| < 2.5 were considered. Each H→bb b quark was matched to the nearest selected reco AK4 jet within ΔR < 0.4, and each matched reco jet was then matched to the nearest GenJetAK4 within ΔR < 0.4. This produced 9883 usable HH events and 19766 matched b-jet rows.

## Outputs

- `outputs/hbb_bjet_regression_dataset/train.npz`
- `outputs/hbb_bjet_regression_dataset/val.npz`
- `outputs/hbb_bjet_regression_dataset/test.npz`
- `outputs/hbb_bjet_regression_dataset/jets_all.csv`
- `outputs/hbb_bjet_regression_dataset/feature_names.json`
- `outputs/hbb_bjet_regression_dataset/metadata.json`
- `outputs/hbb_bjet_regression_dataset/response_profile_vs_reco_pt.csv`
- `outputs/plots/hbb_bjet_regression_dataset/*.png`
