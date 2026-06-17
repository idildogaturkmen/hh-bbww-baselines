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


## MAX_JETS choice

The fixed-size H -> bb reconstruction inputs use `MAX_JETS = 12`. This is now supported by a raw-event diagnostic scan in `scripts/diagnose_max_jets.py`. Before applying any jet cap, 12,232 scanned HH -> bbWW events have both H -> bb b quarks matched to two distinct AK4 jets. Keeping only the first 12 AK4 jets retains 12,171 of these events, or 99.5%, while limiting the unordered pair search to C(12, 2) = 66 candidate jet pairs per event.

Increasing the cap gives only a small retention gain at higher combinatoric cost: `MAX_JETS = 14` retains 12,221 events, or 99.9%, but requires 91 pairs per event; `MAX_JETS = 20` retains all 12,232 matchable events but requires 190 pairs per event. The current choice is therefore a pragmatic balance between H -> bb truth-pair retention and pair-assignment complexity.

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
