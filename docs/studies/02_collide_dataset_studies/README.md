# Study 02 — COLLIDE dataset studies

## Question

What does the **COLLIDE-1M** public sample (the external dataset used to
support the original [HH→bbWW study](../01_hh_bbww/README.md)) actually
contain, and is it adequate as a working background/signal source for early
ML development?

## Dataset / samples

COLLIDE-1M, a public simulated-event cache, accessed as a local expanded
subset after a storage (PVC) expansion. This is explicitly **not** the
project's own Delphes production (see
[study 03](../03_channel_pivot_and_hh4b_simulation/README.md)) — it is an
external, coarser-grained resource used only in the project's earliest
phase.

## Method

1. **Background inventory** — catalogued the expanded local subset:
   156 parquet files (~208 GB) across 12 process groups (HH_bbWW signal, HH
   other, ttbar family, DY/Z+jets, W+jets, diboson, triboson, single Higgs,
   QCD, gamma, minbias, upsilon). Corrected a regex false positive in the
   inventory script that had mislabeled `ttW_incl` as single-top.
2. **Cross-section/normalization metadata** — built fiducial cross-section,
   filter-efficiency, and k-factor lookup tables for the sample's physics
   processes (`config/collide_sample_metadata.csv`,
   `config/collide_sample_metadata_rough.csv` — the `_rough` suffix is the
   source's own label, kept as such rather than smoothed away).
3. **A first classifier on the raw sample** — a `GradientBoostingClassifier`
   separating HH→bbWW signal from a COLLIDE-sourced ttbar background using 8
   event-level features.
4. **Exploratory inspection** — `inspect_collide2v.ipynb`, an 86-cell
   notebook doing ad hoc exploration of the "collide2v" variant of the
   sample.

## Main findings

- The dataset is usable but coarse: the ttbar-vs-HH classifier reached
  AUC_test=0.712 (train 0.720, val 0.701) — at 90% signal efficiency this
  only rejects ttbar to 72% efficiency (a 1.39× rejection factor), i.e.
  modest separation with simple event-level features.
- Two normalization metadata files exist side by side —
  `collide_sample_metadata.csv` (adopted) and `collide_sample_metadata_rough.csv`
  (its own filename says "rough") — kept as two distinct files rather than
  merged, since they represent different maturity levels of the same
  normalization exercise; do not treat the `_rough` file as equivalent to
  the adopted one.
- This background source is what the [bbWW study](../01_hh_bbww/README.md)'s
  ttbar/Z→bb rejection numbers are built on, not the project's own later
  production.

## Figures

- [`figures/collide_ttbar_classifier_roc.png`](figures/collide_ttbar_classifier_roc.png) — the headline ROC (AUC 0.712) for the COLLIDE-sourced ttbar-vs-HH classifier.
- [`figures/collide_ttbar_classifier_feature_importance.png`](figures/collide_ttbar_classifier_feature_importance.png) — which of the 8 event-level features drove that separation.

## Reproducible code

- `scripts/build_collide_inventory.py` — the background inventory builder.
- `scripts/check_collide_normalization_metadata.py`,
  `scripts/build_physics_weights.py` — normalization metadata checks.
- `inspect_collide2v.ipynb` (moved here from repo root this stage — no
  script referenced it by its old root-level path).
- `scripts/analysis/hh4b_plot_style.py` — shared plotting style used across
  studies including this one.

## Relationship to the final HH→4b study

COLLIDE-1M was used **only** as external background support for the
original [bbWW study](../01_hh_bbww/README.md); the project's later HH→4b
work uses its own dedicated MG5→Pythia8→Delphes production instead (see
[study 03](../03_channel_pivot_and_hh4b_simulation/README.md)), which is
**not** a COLLIDE derivative. COLLIDE-era numbers should never be cited as
HH→4b results. The main lasting influence is methodological: the
inventory/normalization-metadata discipline established here (distinguish
adopted vs. rough conventions, record exact process-group provenance) was
carried forward into the much larger HH→4b physical-normalization campaign.

## Provenance

`Results Summaries/RESULTS_COLLIDE_BACKGROUND_INVENTORY.md`;
`config/collide_sample_metadata.csv`, `config/collide_sample_metadata_rough.csv`;
`outputs/plots/ttbar_classifier/`, `outputs/summaries/` (COLLIDE-era entries);
`docs/project_overview/physics_goal_and_pivot.md` (cites COLLIDE as the
dataset motivating the eventual pivot).
