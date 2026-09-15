# Study branch: 02 — COLLIDE dataset studies

This is a reader-facing **study branch**: a curated snapshot of `main`
letting a visitor land directly on one scientific study without first
learning the project's historical Track A/B branch names. It is not an
independent line of development — it is `main` (full history preserved,
nothing squashed) plus this file. For the complete SURF repository,
including all other studies, see `main`.

## Scientific question

What does the external **COLLIDE-1M** public sample (used to support the
original HH→bbWW study) actually contain, and is it adequate as a working
background/signal source for early ML development?

## Dates

2026-05 → 2026-06, run in parallel with [study 01](docs/studies/01_hh_bbww/README.md).

## Relevant code

`scripts/build_collide_inventory.py` (background inventory builder),
`scripts/check_collide_normalization_metadata.py`,
`scripts/build_physics_weights.py` (normalization metadata checks),
`docs/studies/02_collide_dataset_studies/inspect_collide2v.ipynb` (86-cell
exploratory notebook, moved here from the repo root during the 2026-09
reorganization — nothing referenced its old root-level path),
`scripts/analysis/hh4b_plot_style.py` (shared plotting style).

## Relevant data / provenance

`Results Summaries/RESULTS_COLLIDE_BACKGROUND_INVENTORY.md`;
`config/collide_sample_metadata.csv` (adopted) and
`config/collide_sample_metadata_rough.csv` (its own filename says "rough" —
kept distinct, not merged, since the two represent different maturity
levels of the same normalization exercise); `outputs/plots/ttbar_classifier/`,
`outputs/summaries/` (COLLIDE-era entries); `docs/project_overview/physics_goal_and_pivot.md`
cites COLLIDE as the dataset motivating the eventual channel pivot.

## Main results

- Background inventory: 156 parquet files (~208 GB) across 12 process
  groups (HH_bbWW signal, HH other, ttbar family, DY/Z+jets, W+jets,
  diboson, triboson, single Higgs, QCD, gamma, minbias, upsilon).
  Corrected a regex false positive that had mislabeled `ttW_incl` as
  single-top.
- A `GradientBoostingClassifier` separating HH→bbWW signal from
  COLLIDE-sourced ttbar background (8 event-level features) reached
  AUC_test=0.712 (train 0.720, val 0.701) — at 90% signal efficiency this
  only rejects ttbar to 72% efficiency (1.39× rejection): usable but
  coarse separation.
- This background source is what study 01's ttbar/Z→bb rejection numbers
  are built on — not the project's own later Delphes production.

## Superseded / negative results

None specific to this study beyond the general finding that COLLIDE is
"usable but coarse" — the dataset itself was superseded wholesale (not
because of a negative result, but because the channel pivot to HH→4b
moved to a dedicated, purpose-built simulation instead). COLLIDE-era
numbers should never be cited as HH→4b results.

## Relationship to main

This branch **is** `main` at the point this file was added (no divergent
commits, no rebasing, no squashing) — every commit and script on `main` is
present here unchanged. The curation is additive: this one file. `main`
remains the single complete, canonical SURF repository. See
[`docs/studies/02_collide_dataset_studies/README.md`](docs/studies/02_collide_dataset_studies/README.md)
on `main` for the full narrative this file summarizes.

## Source historical branches incorporated

None — this study's content was always part of the project's mainline
history and required no cross-branch merge.
