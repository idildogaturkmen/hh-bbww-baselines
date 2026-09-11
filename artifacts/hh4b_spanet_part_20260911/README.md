# HH→4b SPA-Net + frozen ParT representation study — 2026-09-11 snapshot

This directory freezes the lightweight publication/evaluation artifacts from the
controlled SPA-Net representation study on the exact matched 400k development
cohort.

## Models

- **Native SPA-Net 2M**: 7 native reconstructed-jet features.
- **Native SPA-Net 10M**: same native representation, 5× larger training sample.
- **SPA-Net + ParT active20 2M**: same 2M cohort and training contract as native
  SPA2M, with 20 TRAIN-only variance-selected frozen ParT dimensions appended.

## Main result

| Model | All-background AUC | QCD AUC | ttbar AUC |
|---|---:|---:|---:|
| Native SPA-Net (2M) | 0.969281 | 0.968155 | 0.975392 |
| Native SPA-Net (10M) | 0.969272 | 0.967756 | 0.977502 |
| SPA-Net + ParT active20 (2M) | 0.944222 | 0.941220 | 0.960516 |

For ParT-active20 minus native-2M:

- all-background ΔAUC = -0.025059
- paired 95% CI = [-0.025504, -0.024619]

Exact-event HH reconstruction:

- Native SPA-Net (2M): 0.866588
- SPA-Net + ParT active20 (2M): 0.496015
- Δ = -0.370573
- paired 95% CI = [-0.373936, -0.367229]

This is a matched development/validation result, not final blind-test performance.

## Interpretation

The active20 frozen-ParT augmentation significantly degraded both event
classification and symmetry-aware HH reconstruction.

A subsequent read-only root-cause audit ruled out native-feature corruption,
major jet-slot misalignment, preprocessing implementation errors, accidental
hyperparameter/architecture differences beyond the intended input-width change,
and checkpoint corruption.

The diagnosis narrowed the likely causes to:

1. changed optimization dynamics from widening the SPA-Net input embedding; and
2. a variance-only ParT feature-selection rule that discarded several more
   discriminative embedding dimensions.

A ZERO20 training ablation is the next controlled diagnostic and is not included
in this snapshot.

## Directory layout

- `metrics/` — machine-readable matched comparison and frozen GO/NO-GO result
- `tables/` — publication-ready TSV result tables
- `plots/` — SVG figures
- `training/` — lightweight training/evaluation metadata only
- `diagnosis/` — root-cause audit, follow-up design, provenance freeze, hashes

Large HDF5 datasets, model checkpoints, and event-level NPZ exports are
intentionally not tracked in Git. Their hashes and provenance are recorded in
`diagnosis/RESULT_FREEZE.md` and `diagnosis/SHA256SUMS`.
