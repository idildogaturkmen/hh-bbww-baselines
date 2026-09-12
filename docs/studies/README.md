# Scientific studies index

This directory is the curated entry point into the project's science, organized by
topic rather than by when or how the work was done. Each subdirectory below has one
short `README.md` answering: what question was being asked, what method was used,
what dataset it ran on, what the main result was, where the key figures/tables and
code live, and — separately, in a **Provenance** subsection — the exact frozen
directory this study's full evidence trail lives in.

These pages are **curated summaries that link out**, not a replacement for the
underlying frozen evidence. Nothing the underlying evidence directories contain has
been moved, renamed, or altered to produce these pages.

Read in this order for the project's narrative arc; jump directly to any topic
otherwise.

1. [`bbww_baseline/`](bbww_baseline/README.md) — the original HH→bbWW baseline and
   why the project moved on from it
2. [`hh4b_simulation/`](hh4b_simulation/README.md) — the resolved HH→4b simulated
   dataset every later study is built on
3. [`classical_deep_baselines/`](classical_deep_baselines/README.md) — cut-based,
   BDT, and DNN reference baselines
4. [`physical_normalization/`](physical_normalization/README.md) — turning raw
   simulated counts into physically meaningful expected yields
5. [`spanet_reconstruction_classification/`](spanet_reconstruction_classification/README.md) —
   the native SPA-Net jet-assignment and classification model
6. [`training_scale_study/`](training_scale_study/README.md) — does 5x more training
   data improve native SPA-Net?
7. [`tail_numerical_reliability/`](tail_numerical_reliability/README.md) — is the
   extreme-score tail numerically trustworthy?
8. [`part_representation_study/`](part_representation_study/README.md) — appending
   frozen ParT features to SPA-Net: a harm result, diagnosed
9. [`zero20_causal_ablation/`](zero20_causal_ablation/README.md) — the planned
   follow-up isolating *why* the ParT augmentation harmed performance
10. [`future_representation_learning/`](future_representation_learning/README.md) —
    external-model transfer, and JP-JEPA compatibility preparation

For the full narrative and current headline numbers, see `PROJECT_STORY.md` and
`RESULTS_OVERVIEW.md` at the repository root.
