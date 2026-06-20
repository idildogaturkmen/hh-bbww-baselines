# HH→bbWW WW-side target decision

The status-23 WW-side diagnostic was run on the full two-file HH→bbWW COLLIDE-1M signal sample.

## Full-sample counts

- raw events: 17,967
- events with two status-23 H→bb b quarks: 17,967
- events with truth H→bb reco pair: 9,973
- events with hadronic WW-side targets: 16,381
- events with all hadronic WW-side target quarks matched to retained AK4 jets: 2,028

## WW decay-mode composition

| WW mode | Events | Fraction |
|---|---:|---:|
| had_had | 8,059 | 0.4485 |
| unknown | 6,308 | 0.3511 |
| lep_had_e | 1,284 | 0.0715 |
| lep_had_mu | 1,215 | 0.0676 |
| lep_had_tau | 813 | 0.0453 |
| lep_lep | 288 | 0.0160 |

## W-side target matchability

| WW mode | Per-quark match fraction | All target quarks matched fraction |
|---|---:|---:|
| had_had | 0.4555 | 0.0272 |
| lep_had_e | 0.4907 | 0.2181 |
| lep_had_mu | 0.4930 | 0.2280 |
| lep_had_tau | 0.4711 | 0.2177 |
| unknown | 0.4976 | 0.1704 |

## Interpretation

The fully hadronic WW mode is not a robust immediate assignment target with the current retained AK4 jets: only about 2.7% of had_had events have all four hadronic W-side target quarks matched. The semileptonic modes are more promising, but even there only about 22% of events have both hadronic W-side quarks matched.

Therefore, the next model comparison should not force full WW assignment. The immediate next modeling step should be SPA-Net H→bb assignment inside HH→bbWW events, using the same target as the BDT and Pair-DNN baselines. WW decay-mode information should be kept as diagnostic metadata and revisited later for a semileptonic bbℓνqq′ extension.
