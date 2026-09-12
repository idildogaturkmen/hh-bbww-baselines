# HH → 4b: ML baselines and representation studies

This repository holds the simulation, baseline modeling, and representation-learning
pipeline for a CMS-inspired study of resolved **HH → 4b** (di-Higgs → four b-quarks)
using Delphes-simulated events, together with the earlier **HH → bbWW** baseline the
project moved on from in July 2026.

- **New here?** Start with [`PROJECT_STORY.md`](PROJECT_STORY.md) — the narrative of
  what this project is, why it pivoted from bbWW to HH → 4b, and where the science
  stands today.
- **Want the numbers?** [`RESULTS_OVERVIEW.md`](RESULTS_OVERVIEW.md) is an
  at-a-glance table of every study's headline result.
- **Want the detail on one topic?** [`docs/studies/`](docs/studies/README.md) is a
  curated, one-README-per-topic index — Question / Method / Dataset / Main result /
  Key figures/tables / Code+config pointers / Frozen artifact / Provenance — for
  each of the ten scientific threads in this project:

  1. [bbWW baseline](docs/studies/bbww_baseline/README.md) (superseded)
  2. [HH→4b simulation](docs/studies/hh4b_simulation/README.md)
  3. [Classical/deep baselines](docs/studies/classical_deep_baselines/README.md)
  4. [Physical normalization](docs/studies/physical_normalization/README.md)
  5. [SPA-Net reconstruction/classification](docs/studies/spanet_reconstruction_classification/README.md)
  6. [Training-scale study](docs/studies/training_scale_study/README.md)
  7. [Tail/numerical reliability](docs/studies/tail_numerical_reliability/README.md)
  8. [ParT representation study](docs/studies/part_representation_study/README.md)
  9. [ZERO20 causal ablation](docs/studies/zero20_causal_ablation/README.md) (planned)
  10. [Future representation learning](docs/studies/future_representation_learning/README.md)

Each study page links out to its exact frozen artifact/provenance directory rather
than duplicating that evidence — this README and the study pages are curated entry
points, not a replacement for the underlying frozen record.

## Where things live

| What | Where |
|---|---|
| Simulation, production, and preprocessing scripts | `scripts/delphes/`, `scripts/production/` |
| Model training and evaluation scripts | `scripts/analysis/` |
| Legacy bbWW-era scripts | flat files under `scripts/` (see the bbWW baseline study page) |
| Run/experiment configuration | `configs/`, `config/` |
| Physics generator cards | `cards/` |
| Unit tests | `tests/` |
| Frozen, citable, lightweight result bundles | `artifacts/` |
| Self-contained public data-release snapshot | `data_release/` |
| Curated scientific study pages (start here for any topic) | `docs/studies/` |
| Full dated provenance ledger (hash-stamped, one directory per decision point) | `docs/checkpoints/` |
| Paper components (figures, tables, captions) | `docs/paper/` |

## A note on external artifacts

Large binary artifacts — training datasets (HDF5), full model checkpoints, and
event-level NPZ exports — are deliberately not committed to this repository; they
live in an external working area and are regenerable from the committed configs and
scripts. Every frozen result under `artifacts/` and `docs/checkpoints/` records the
hashes of the external files it depended on (typically in a `SHA256SUMS` file
alongside it), so external artifacts can always be verified against what a frozen
result actually used.
