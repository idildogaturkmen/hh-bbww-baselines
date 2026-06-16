# Z->bb m_bb Diagnostics

This note compares the truth-matched reconstructed bb mass in the COLLIDE-1M `ZJetsTobb_13TeV-madgraphMLM-pythia8` sample to the current HH -> bbWW truth-matched H -> bb mass diagnostic.

## Command

```bash
python scripts/diagnose_zbb_mbb.py --n-files 1 --target-usable 3000 --max-scan 10000 --max-jets 12 --max-dr 0.4
```

## Matching Definition

- Z sample: status-23 generator b quarks are selected, preferring candidates with a Z-boson ancestor in the generator record.
- The two selected generator b quarks are matched to distinct retained AK4 jets with DeltaR < max_dr.
- The same MAX_JETS cap is used as in the HH preprocessing.
- The reconstructed mass is computed from the matched AK4 jet four-vectors, not from the generator b quarks.

## Event Counts

- scanned events: 5892
- usable matched Z->bb events: 3000
- events with two selected b quarks: 5819
- events with two matched b quarks: 3004
- events with distinct matched AK4 jets: 3000
- label-mode counts: `{'status23_no_z_ancestor_fallback': 5806, 'status23_with_z_ancestor': 13}`

Truth-record caveat: the script tries to identify b quarks with a resolvable Z-boson ancestor, but the flattened COLLIDE generator record does not always make `M1/M2` usable as direct array indices. When the ancestor cannot be resolved, it falls back to the same practical convention used for the HH preprocessing: the two status-23 b quarks.

## Truth-Matched m_bb Comparison

| Sample | Mean [GeV] | Median [GeV] | 16-84% [GeV] | Fraction 80-120 | Fraction 90-140 |
|---|---:|---:|---:|---:|---:|
| HH truth-matched H->bb | 108.71 | 103.18 | 80.40-129.67 | 0.594 | 0.615 |
| ZJetsTobb truth-matched Z->bb | 75.01 | 72.03 | 57.29-90.39 | 0.305 | 0.153 |

Relative to the current HH truth-matched distribution, the Z->bb matched mean shifts by -33.70 GeV and the median shifts by -31.15 GeV.

## Z Pair Selection Cross-Checks

| Method | Mean [GeV] | Median [GeV] | 16-84% [GeV] | Truth-pair accuracy |
|---|---:|---:|---:|---:|
| truth_matched | 75.01 | 72.03 | 57.29-90.39 | 1.000 |
| top2_btag | 142.28 | 79.10 | 42.17-224.25 | 0.272 |
| closest_z_mass | 87.27 | 89.54 | 77.03-96.55 | 0.284 |
| closest_h_mass | 115.80 | 122.20 | 91.05-133.05 | 0.142 |

## Interpretation

A Z-mass veto or Z-vs-H classifier should be evaluated as a signal-efficiency tradeoff, not assumed to be safe. The current HH truth-matched H->bb mass is already shifted below 125 GeV, so any rejection using only m_bb can remove genuine HH signal as well as Z->bb background.

## Outputs

- `outputs/zbb_diagnostics/zbb_event_diagnostics.csv`
- `outputs/zbb_diagnostics/zbb_mbb_comparison.csv`
- `outputs/zbb_diagnostics/zbb_mbb_summary.json`
- `outputs/plots/zbb_diagnostics/hh_vs_zbb_truth_matched_mbb.png`
- `outputs/plots/zbb_diagnostics/zbb_pair_method_mbb.png`
