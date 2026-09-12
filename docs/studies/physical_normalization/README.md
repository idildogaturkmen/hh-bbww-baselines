# Physical normalization

## Question

How do we convert raw simulated event counts into physically meaningful expected
yields — correct per-process cross sections, generator weights, and an integrated
luminosity — so that comparisons between models mean something in real
collider-physics terms (expected events, significance) rather than only a relative
classifier ranking?

## Method

- Per-process generator-weight and cross-section provenance extraction from
  MadGraph5 LHE/banner metadata, for every signal and background source
  individually.
- Construction and closure-testing of a Run-2, 13 TeV, 138 fb⁻¹
  expected-yield-projection contract (source/event weight closure, process/campaign/
  class/fold/transport yield closure).
- A dedicated four-model physically-normalized comparison that translates each
  classifier's raw output score into an expected signal and QCD-background yield at
  chosen Standard-Model signal efficiencies and an integrated luminosity, rather
  than reporting only ROC AUC.

## Dataset

Per-source generator banners and cross sections for every signal/background sample
in [`../hh4b_simulation/`](../hh4b_simulation/README.md); the frozen common event
table with physical weights attached.

## Main result

- An authorized Run-2 138 fb⁻¹ physical-normalization contract: 441 primary
  physical sources, ~3.57M primary events, with source/event weight closure and
  process/campaign/class/fold/transport yield closure all passing.
- A four-model comparison (BDT-K, BDT-KF, native SPA-Net 2M, native SPA-Net 10M) at
  450 fb⁻¹, QCD-only, statistics-only: native SPA-Net 10M reaches the highest
  cross-checked significance (ZA(B95) = 0.540 at 10% SM signal efficiency), ahead of
  SPA-Net 2M (0.525) and both BDT variants (BDT-KF 0.375, BDT-K 0.043), though the
  SPA-Net 2M vs. 10M gap is modest — consistent with the saturation finding in
  [`../training_scale_study/`](../training_scale_study/README.md).

## Key figures/tables

- `docs/paper/jhep_hh4b_ml/track_b_physical_normalization/MAIN_MODEL_COMPARISON.{md,csv}`
- `docs/paper/jhep_hh4b_ml/track_b_physical_normalization/CANONICAL_FOUR_MODEL_NORMALIZED_RESULTS.{md,csv}`
- `docs/paper/jhep_hh4b_ml/track_b_physical_normalization/{qcd_rejection_vs_actual_sm_efficiency,za_b95_vs_actual_sm_efficiency}.pdf`

## Code/config pointers

- `scripts/analysis/parse_hh4b_generator_provenance.py`,
  `review_hh4b_generator_configuration.py`,
  `extract_hh4b_normalization_provenance.py` (see repository-wide search for the
  full `*normalization*`/`*provenance*` script family under `scripts/analysis/`).
- `docs/normalization_assumptions.md` — the stable statement of normalization
  assumptions used throughout.

## Frozen artifact

- `docs/paper/jhep_hh4b_ml/track_b_physical_normalization/` — the four-model
  physically-normalized comparison bundle, with its own `SHA256SUMS`,
  `NORMALIZATION_CONTRACT.md`, and `MODEL_FREEZE.md` recording exact model hashes.
- `docs/checkpoints/hh4b_train_run2_138fb_physical_authorization_freeze_20260806_v1/` —
  the Run-2 138 fb⁻¹ authorization freeze.

## Provenance

- `docs/checkpoints/hh4b_physical_normalization_*_2026072{9,9,9}_v1/` and
  `_20260731_v*` — the full 41-checkpoint provenance-extraction and freeze series
  (generator artifact locations, QCD/ttbar/triboson/VBF weight transport,
  run-2 luminosity contract, and the global ordinary/hard-QCD weight registries).
- `docs/paper/jhep_hh4b_ml/track_b_physical_normalization/` — note this directory's
  path retains its original internal-track naming; the study itself is described
  above without that naming.
