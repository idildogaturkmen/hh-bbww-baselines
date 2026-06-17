# H->bb Decay-Mode m_bb Diagnostic

This diagnostic tests whether the low truth-matched reconstructed H->bb mass is correlated with semileptonic-like b decays.

## Command

```bash
/tmp/hh-bbww-venv/bin/python scripts/diagnose_hbb_decay_modes.py --n-files 2 --max-events -1
```

## Definition

- The H->bb b quarks are selected with the same practical convention as the current HH preprocessing: status-23 generator b quarks, matched to distinct retained AK4 jets with DeltaR < 0.4.
- A matched b jet is called `strict_semileptonic_like` when both a generator charged lepton and a generator neutrino are found within DeltaR < 0.4 of the matched AK4 jet axis.
- A matched b jet is called `neutrino_like` when a generator neutrino is found within DeltaR < 0.4 of the matched AK4 jet axis.
- These are cone-based labels, not perfect b-hadron ancestry labels. They can be contaminated by nearby W-decay leptons/neutrinos, so the result should be interpreted as a diagnostic trend.

## Event Counts

- scanned events: 17967
- usable matched events: 12174
- no two status-23 b quarks: 0
- unmatched b quark: 5755
- both b quarks matched to the same AK4 jet: 38

## Overall Matched m_bb

| n | Mean [GeV] | Median [GeV] | 16-84% [GeV] | 90 < m_bb < 140 |
|---:|---:|---:|---:|---:|
| 12174 | 108.71 | 103.18 | 80.39-129.66 | 0.615 |

## Strict Semileptonic-Like Categories

| Category | n | Mean [GeV] | Median [GeV] | 16-84% [GeV] | 90 < m_bb < 140 |
|---|---:|---:|---:|---:|---:|
| 0_strict_semileptonic_like_bjets | 5525 | 113.36 | 108.26 | 85.65-133.63 | 0.672 |
| 1_strict_semileptonic_like_bjet | 5302 | 106.15 | 100.35 | 78.13-126.39 | 0.589 |
| 2_strict_semileptonic_like_bjets | 1347 | 99.66 | 93.12 | 73.11-121.43 | 0.478 |

## Nearby-Neutrino Categories

| Category | n | Mean [GeV] | Median [GeV] | 16-84% [GeV] | 90 < m_bb < 140 |
|---|---:|---:|---:|---:|---:|
| 0_neutrino_like_bjets | 4962 | 113.86 | 108.98 | 86.56-133.90 | 0.680 |
| 1_neutrino_like_bjet | 5579 | 106.51 | 100.87 | 78.52-126.55 | 0.597 |
| 2_neutrino_like_bjets | 1633 | 100.56 | 93.59 | 73.02-123.14 | 0.477 |

## Interpretation

If the hadronic-like category moves closer to 125 GeV while semileptonic-like categories shift lower, that supports semileptonic b-decay neutrino energy loss as one contributor to the low reconstructed m_bb. If all categories remain similarly low, then detector/jet response, AK4 containment, pT thresholds, or simplified COLLIDE calibration are likely also important.

This diagnostic cannot prove the full cause because the available truth record does not provide a clean b-hadron decay chain for every nearby lepton/neutrino. It is still a useful first physics cross-check because it tests whether invisible energy near the matched b jets correlates with the mass shift.

## Outputs

- `outputs/hbb_decay_modes/hbb_decay_mode_event_diagnostics.csv`
- `outputs/hbb_decay_modes/hbb_decay_mode_summary.csv`
- `outputs/hbb_decay_modes/hbb_decay_mode_summary.json`
- `outputs/plots/hbb_decay_modes/matched_mbb_by_strict_decay_category.png`
- `outputs/plots/hbb_decay_modes/matched_mbb_by_neutrino_category.png`
- `outputs/plots/hbb_decay_modes/matched_mbb_vs_nearby_neutrino_pt_sum.png`
