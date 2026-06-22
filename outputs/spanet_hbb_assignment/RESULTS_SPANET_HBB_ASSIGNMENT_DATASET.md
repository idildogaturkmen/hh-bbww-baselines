# SPA-Net H→bb Assignment Dataset

This dataset converts HH→bbWW events into a padded event-level jet dataset for SPA-Net-style H→bb assignment.

The target is the pair of reconstructed AK4 jets matched to the two status-23 H→bb b quarks. WW decay mode is stored as metadata only.

## Counters

- raw_events_scanned: 17967
- events_with_truth_target_in_pair_table: 9973
- usable_events: 9973
- skipped_not_in_targets: 7994
- skipped_too_few_reco_jets: 0
- skipped_target_not_in_reco_jets: 0
- skipped_target_truncated_by_max_jets: 0
- max_jets: 20
- n_features: 10

## Dataset shape

- events: 9973
- jets: `(9973, 20, 10)`
- mask: `(9973, 20)`
- target: `(9973, 2)`
- features: `['log_pt', 'eta', 'sin_phi', 'cos_phi', 'log_mass', 'btag', 'btag_phys', 'response_corr', 'log_corrected_pt', 'log_corrected_mass']`

## Split counts

| Split | Events |
|---|---:|
| train | 6019 |
| val | 1985 |
| test | 1969 |

## WW-mode metadata counts

| WW mode | Events |
|---|---:|
| had_had | 4479 |
| unknown | 3503 |
| lep_had_e | 702 |
| lep_had_mu | 677 |
| lep_had_tau | 462 |
| lep_lep | 150 |

## Jet multiplicity

- min n_jets: 2
- median n_jets: 6.0
- max n_jets: 15

## Interpretation

This is the first event-level dataset for comparing SPA-Net against the existing physics, BDT, and Pair-DNN H→bb assignment baselines.

Among the 9973 usable H→bb events:

had_had:      4479
unknown:      3503
lep_had_e:     702
lep_had_mu:    677
lep_had_tau:   462
lep_lep:       150

Approximate fractions:

had_had:      44.9%
unknown:      35.1%
lep_had_e:     7.0%
lep_had_mu:    6.8%
lep_had_tau:   4.6%
lep_lep:       1.5%

This means the dataset still comes from full HH→bbWW events, but the model target is currently only:

Which two jets are H→bb?

## Current dataset
Input:
  all retained AK4 jets in the event
  padded to 20 jets
  10 features per jet

Target:
  the two jet positions corresponding to H→bb

Metadata:
  event_id
  train/val/test split
  WW decay mode
  original reco jet indices


event-level ML dataset:
jets:   (9973, 20, 10)
mask:   (9973, 20)
target: (9973, 2)