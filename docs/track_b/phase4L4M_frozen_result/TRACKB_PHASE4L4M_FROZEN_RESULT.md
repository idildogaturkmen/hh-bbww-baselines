# Track B Phase-4L/4M Frozen Result Note

Date: 2026-08-14

Scope: concise paper-facing record of the frozen Track-B Phase-4L/4M
HH->4b raw-count ML benchmark. This note contains no CMS collision-data
artifacts and no internal CMS material.

## Dataset and population

Track B uses the high-statistics external Sophon / jet-free HH->4b simulation
dataset from Congqiao's study. The QCD background used here is the frozen
Track-B `training_qcd` lane, and the signal benchmark is `gghh_kl1`.

The reported rejection values are conditional on the frozen Track-B
`tight_exact4` population: all four of the first four selected jets pass the
Track-B SophonAK4 tight working point. They are not a CMS SR4b reproduction,
not a CMS collision-data measurement, and not a physically normalized yield or
sensitivity result. The useful paper statement is therefore about controlled
classifier rejection within this external simulation population.

## Model definitions

| Label | Input meaning |
|---|---|
| E0 | frozen event classifier using kinematics only |
| E1 | E0 plus continuous Sophon B/C/L flavor information |
| E2 | E1 plus the frozen pairwise-attention augmentation |

No model was retrained, retuned, filtered after seeing holdout data, or changed
for the holdout_A result.

## Fixed working point result

At `epsilon_S=0.40`, the frozen development and untouched holdout_A results are:

| Split | E0 R_B | E1 R_B | E2 R_B | E1 vs E0 |
|---|---:|---:|---:|---:|
| development | \(283.1 \pm 27.2\) | \(370.6 \pm 26.8\) | \(216.3 \pm 18.0\) | +30.9% |
| holdout_A | \(285.7 \pm 43.9\) | \(439.5 \pm 38.4\) | \(244.6 \pm 19.3\) | +53.8% |

Here `R_B = 1 / epsilon_B`, using raw-count background efficiency inside the
frozen Track-B population.

A concise paper-facing statement is:

> On the development sample, continuous learned Sophon flavor information
> increased QCD rejection at 40% signal efficiency from \(R_B=283.1 \pm 27.2\)
> to \(370.6 \pm 26.8\). On a fully untouched holdout_A sample, the corresponding
> rejection values were \(285.7 \pm 43.9\) and \(439.5 \pm 38.4\). The improvement
> reproduced across all three seeds and all adequately supported working
> points. Adding the frozen pairwise-attention augmentation did not provide an
> additional improvement in this configuration.

## Reproducibility and caveats

- The protocol, working-point grid, support labels, and reproducibility
  criterion were frozen before opening holdout_A.
- The predeclared checkpoints and scripts were hash-verified against the frozen
  Phase-4L artifacts before holdout_A event data was read.
- holdout_A was opened once for this adjudication and is now spent for this
  result.
- holdout_B, inference_qcd_1, and inference_qcd_2 remain unopened for this
  adjudication.
- The primary reproduced conclusion is `E1 > E0`: it generalizes across all
  adequately supported working points, using the predeclared raw-QCD support
  threshold of at least 10 events.
- The `E1 > E2` comparison partially reproduces: it holds for all seeds at
  `epsilon_S` in `{0.60, 0.50, 0.40, 0.25}`, but one seed flip occurs at
  `epsilon_S=0.20`, exactly at the raw-QCD=10 support boundary.
- Tail working points should be read with finite-count caution. This result is
  a raw-count classifier-rejection benchmark, not a final normalized physics
  sensitivity estimate.

## Interpretation

This is a controlled ablation result: adding continuous Sophon B/C/L flavor
information to kinematics substantially improves rejection in both development
and untouched holdout_A. The direction strengthens rather than disappears on
unseen data, with the holdout_A improvement at `epsilon_S=0.40` about 54%
compared with about 31% on development.

The result does not show an additional gain from the specific frozen E2
pairwise-attention augmentation. That should be reported as a configuration
result, not as a general claim that pairwise attention cannot help.
