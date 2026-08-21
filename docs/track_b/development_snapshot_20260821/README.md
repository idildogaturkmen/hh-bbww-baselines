# Track-B development snapshot — 2026-08-21

> **DEVELOPMENT ONLY — not a final or governing physics result.**

This branch is a small public reproducibility snapshot of paper-relevant
Track-B development evidence. It was assembled additively from existing,
frozen LPC artifacts. No training, rescoring, submission, inference, or
scientific-result modification was performed.

The archive deliberately excludes raw events, ROOT/HDF5/NPZ datasets,
checkpoints and model binaries, TensorBoard/Condor payloads, logs, virtual
environments, credentials, and large staging products. The six residual-
background SVGs larger than 20 MB are also omitted; equivalent PDF and PNG
renderings are included.

## Contents

- `model_comparison/`: four-model 400k-cohort results, tables, figures, and
  the small self-contained builders from Source A.
- `residual_backgrounds/`: residual-composition/topology prototype, tables,
  figures smaller than 20 MB, and scripts from Source B.
- `bdt_convergence/`: 200/1000/4000/6000-round diagnostics from Source C.
- `spanet_10m_scaling/`: final 10M x 50 adjudication, 2M-vs-10M scaling
  evidence, and final classification summaries from Source D.
- `harvey_update/`: records the search outcome for Source E.
- `provenance/`: source paths, archive decisions, and manifest hashes.

## Frozen source locations

The original LPC trees are:

- `/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_development_snapshot_20260821_v1`
- `/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_development_residual_backgrounds_20260821_v1`
- `/uscms_data/d3/iturkmen/hh4b_delphes/track_b_harvey_bdt_working_points_20260818_v1/step2m_harvey_round_budget_diagnostics_20260821_v1`
- `/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4AI_spanet_10M_scaling_seed0_20260821_v1`

Each archived component carries its original `SOURCE_PROVENANCE.tsv` and
`SHA256SUMS`. Their manifest hashes and the omitted-artifact inventory are in
`provenance/ARCHIVE_PROVENANCE.md`.

The principal omitted large SPA-Net dataset is on EOS at
`/eos/uscms/store/user/iturkmen/spanet_10M_scaling/production_hdf5_build_10M_v1/work/production_10M_train.h5`
(3,380,013,912 bytes; SHA256
`2186ff1221eb7d42b26852167c29d61b91a646eaf4494314e5d55d427cd97790`).
The reused validation dataset is
`/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4AF_spanet_partial_events_population_correction_20260819_v1/production_hdf5_build_v1/work/production_2M_val.h5`
(SHA256 `3c94bf8300d1dc3324e2cf25f9b6c618dffb94819cb3a297a43ac06320d5c2a2`).

This snapshot is for provenance and remote access only. Interpret all
numbers with the caveats in `PROJECT_STATUS_20260821.md` and the component
interpretation documents.
