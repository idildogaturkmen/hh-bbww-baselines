# HH → bbWW Signal Preprocessing and Baseline Results

## Dataset

Selected COLLIDE-1M signal files:

- `HH_bbWW/HH_bbWW-NEVENT10000-RS23000001.parquet`
- `HH_bbWW/HH_bbWW-NEVENT10000-RS23000002.parquet`

Truth-labeling procedure:

- Select generator-level b quarks with `abs(PID) == 5` and `status == 23`.
- Match each selected b quark to the nearest reconstructed AK4 jet with ΔR < 0.4.
- Keep events only if both b quarks match to two distinct AK4 jets within the first 12 padded jets.
- Saved jet features: `[pt, eta, phi, mass, btag, btagPhys]`.

## Preprocessing yield

- Scanned events: 17,967
- Usable matched events: 12,171
- Usable fraction: 67.7%

Train/validation/test split:

- Train: 9,736
- Validation: 1,217
- Test: 1,218

Mean AK4 jet multiplicity in usable events:

- Train: 9.347
- Validation: 9.277
- Test: 9.288

## Simple H → bb jet-assignment baselines

Test-set accuracies:

| Method | Accuracy |
|---|---:|
| Top-2 b-tag jets | 0.3670 |
| Dijet mass closest to 125 GeV | 0.0928 |
| Combined b-tag + mass heuristic | 0.3079 |

The strongest simple baseline is currently the top-2 b-tag method.

## Signal validation plots

Plots are saved in:

`outputs/plots/signal_validation/`

Included plots:

- AK4 jet multiplicity
- Leading AK4 jet pT
- Subleading AK4 jet pT
- All-jet b-tag score distribution
- Matched H → bb jet b-tag score distribution
- Matched H → bb jet pT distribution
- Matched reconstructed mbb distribution
- Matched ΔRbb distribution

Summary from plotting script:

- Number of usable events plotted: 12,171
- Mean AK4 jet multiplicity: 9.334
- Mean matched mbb: 108.710
- Median matched mbb: 103.176
- Mean matched ΔRbb: 2.082
