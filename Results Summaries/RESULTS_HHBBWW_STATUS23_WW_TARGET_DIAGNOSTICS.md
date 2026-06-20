# HH→bbWW status-23 WW-side target diagnostics

This diagnostic classifies the WW side using status-23 hard-process particles, because the available GenPart parent/daughter links are not reliable for following W decays.

## Counters

- raw_events: 17967
- events_with_two_status23_bquarks: 17967
- events_with_truth_hbb_reco_pair: 9973
- events_with_hadronic_ww_targets: 16381
- events_with_all_hadronic_target_quarks_matched: 2028

## WW decay-mode counts

| WW mode | Events | Fraction |
|---|---:|---:|
| had_had | 8059 | 0.4485 |
| unknown | 6308 | 0.3511 |
| lep_had_e | 1284 | 0.0715 |
| lep_had_mu | 1215 | 0.0676 |
| lep_had_tau | 813 | 0.0452 |
| lep_lep | 288 | 0.0160 |

## Hadronic target matching by mode

| WW mode | Events | Hbb reco-pair matched fraction | Mean hadronic target quarks | Per-quark match fraction | All hadronic targets matched fraction |
|---|---:|---:|---:|---:|---:|
| had_had | 8059 | 0.5558 | 4.00 | 0.4555 | 0.0272 |
| lep_had_e | 1284 | 0.5467 | 2.00 | 0.4907 | 0.2181 |
| lep_had_mu | 1215 | 0.5572 | 2.00 | 0.4930 | 0.2280 |
| lep_had_tau | 813 | 0.5683 | 2.00 | 0.4711 | 0.2177 |
| lep_lep | 288 | 0.5208 | 0.00 |  | 0.0000 |
| unknown | 6308 | 0.5553 | 1.68 | 0.4976 | 0.1704 |

## Interpretation

- `had_had` means four status-23 light quarks were found after excluding the two H→bb b quarks.
- `lep_had_*` means two status-23 light quarks plus a charged lepton and neutrino were found.
- The hadronic target match fraction tells us whether W-side quarks can be associated with retained reco AK4 jets after excluding the truth H→bb jets.
- This is the right diagnostic for deciding whether to build a semileptonic, fully hadronic, or mixed bbWW assignment target.

