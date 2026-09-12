# Project story

## The physics goal

This project studies resolved **HH → 4b** — Higgs pair production where both Higgs
bosons decay to a b-quark pair, giving a four-b-jet final state — in
Delphes-simulated, CMS-Run-2-like events. The aim is to compare, on equal physical
footing, how well different modeling approaches (a cut-based selection, gradient-
boosted trees, a symmetry-aware jet-assignment neural network, and increasingly
ambitious learned representations) separate this signal from its QCD and ttbar
backgrounds, and to understand *why* any given approach does or doesn't work — not
just to report a single headline number.

## Where it started: HH → bbWW

The project's earliest work targeted a different final state: **HH → bbWW**, with
one Higgs decaying to b-quarks and the other to a W-boson pair. This required
reconstructing two very different objects at once — a b-jet pair for the H → bb
side, and a lepton-plus-missing-energy or fully hadronic system for the H → WW
side. A dedicated diagnostic found that the WW side was the real obstacle: even in
the most favorable decay modes, only about a fifth of events had both hadronic
W-side quarks matched to reconstructed jets, and in the fully hadronic mode that
fraction was closer to one in forty. The H → bb side, by contrast, was
tractable — a symmetry-aware jet-pairing network reached a respectable assignment
purity there. Full detail:
[`docs/studies/bbww_baseline/`](docs/studies/bbww_baseline/README.md).

## The pivot

On 2026-07-01, the project pivoted to resolved **HH → 4b**: a channel where every
final-state object is a b-jet, sidestepping the WW-side reconstruction problem
entirely while keeping the same core question (can a symmetry-aware network do
better than a classical baseline?) in a cleaner setting. The reasoning is recorded
in full in the checkpoint this pivot produced. This repository's name predates the
pivot and still describes the superseded channel — see "On the repository name"
below.

## Building the foundation: simulation, baselines, and normalization

Before any model comparison could mean anything, three things had to be built:

1. A **simulated dataset** large enough and well-enough understood to support it —
   ultimately a ~144-million-event resolved HH → 4b population with full generator
   provenance traced for every source
   ([`docs/studies/hh4b_simulation/`](docs/studies/hh4b_simulation/README.md)).
2. **Classical reference baselines** — a multivariate cut-based selection (stability-
   tested across 1,000 bootstrap replicas) and gradient-boosted trees, with and
   without flavor-tag information
   ([`docs/studies/classical_deep_baselines/`](docs/studies/classical_deep_baselines/README.md)).
3. A **physical-normalization contract** translating raw simulated counts into
   expected event yields at a real integrated luminosity (138 fb⁻¹, later
   cross-checked at 450 fb⁻¹), so that "better AUC" could be restated as "better
   expected significance"
   ([`docs/studies/physical_normalization/`](docs/studies/physical_normalization/README.md)).

That normalization work also produced the project's first four-way model
comparison — two BDT variants against native SPA-Net at two training scales — which
showed flavor-tag information was worth roughly an order of magnitude in
significance for the classical models, and that SPA-Net led all four.

## The neural baseline: SPA-Net, and does more data help?

**SPA-Net**, a symmetry-aware jet-assignment network, was trained on native
per-jet kinematic and flavor-tag features and evaluated on both classification
(AUC) and reconstruction (correct jet-pairing rate) —
[`docs/studies/spanet_reconstruction_classification/`](docs/studies/spanet_reconstruction_classification/README.md).
It comfortably beat the classical baselines. The natural next question — does more
training data help further? — got a controlled answer: training on 10 million
events instead of 2 million left classification AUC essentially unchanged
(0.969281 → 0.969272). The native representation had saturated
([`docs/studies/training_scale_study/`](docs/studies/training_scale_study/README.md)).
Because that comparison depended on the extreme tail of the score distribution
being numerically trustworthy, a separate reliability pass cross-checked the tail
under different floating-point precision and confirmed it was not a numerical
artifact
([`docs/studies/tail_numerical_reliability/`](docs/studies/tail_numerical_reliability/README.md)).

## Trying a richer representation: ParT, and a harm result

If native features had saturated, the next lever was a richer input
representation. Twenty frozen, pretrained Particle Transformer (ParT) embedding
dimensions were appended to the same native SPA-Net input — same architecture
family, same training population, only the input width changed. The result was not
a modest improvement or a null result: it was a clear **harm**. Classification AUC
fell by 0.025, and — more strikingly — the correct jet-pairing rate collapsed from
0.87 to 0.50. A careful root-cause audit ruled out the boring explanations (data
corruption, misaligned jet slots, a preprocessing bug, an accidental
hyperparameter change, a corrupted checkpoint) and narrowed the cause to something
more interesting: widening SPA-Net's input embedding changed its optimization
dynamics, compounded by a feature-selection rule that kept the 20
*highest-variance* ParT dimensions rather than the most *discriminative* ones —
plausibly discarding the dimensions that would have helped
([`docs/studies/part_representation_study/`](docs/studies/part_representation_study/README.md)).

## What's next: isolating the cause, and a cleaner representation-learning path

Two lines of follow-up are in motion. The most direct is a controlled ablation —
**ZERO20** — that repeats the exact same experiment with 20 *zeroed* dimensions
instead of ParT features, isolating "wider input embedding" from "ParT content
specifically" as the cause of the harm above. It is designed but not yet run
([`docs/studies/zero20_causal_ablation/`](docs/studies/zero20_causal_ablation/README.md)).

In parallel, the project is preparing a second attempt at a richer representation —
this time via a joint-embedding predictive architecture, **JP-JEPA** — informed
directly by what the ParT result taught. An earlier, separate line of work had
already validated an external released transfer-learning model on real inference
workloads as a faithfulness check; JP-JEPA compatibility preparation now follows
the same discipline the ParT contract used, plus a numerical audit the ParT study
didn't have the chance to do first. That preparation is complete, with one open,
honestly-documented caveat: a bounded but not-yet-fully-explained CPU-vs-GPU
numerical discrepancy in the frozen embedding. No JP-JEPA-augmented model has been
trained yet
([`docs/studies/future_representation_learning/`](docs/studies/future_representation_learning/README.md)).

## On the repository name

This repository is named `hh-bbww-baselines`. By every measure — files, data
volume, and active development — it is now an HH → 4b project with a preserved
bbWW-era baseline kept as history, not the reverse. The name is not changed here;
see [`RESULTS_OVERVIEW.md`](RESULTS_OVERVIEW.md) and `docs/studies/` for what the
project actually is today.
