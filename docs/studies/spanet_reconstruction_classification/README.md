# SPA-Net reconstruction/classification

## Question

Can a symmetry-aware jet-assignment neural network (SPA-Net) both correctly
identify which jets belong to which Higgs candidate (reconstruction) and classify
signal vs. background (classification) more effectively than the classical
baselines, using only native per-jet kinematic and flavor-tag features?

## Method

Train SPA-Net on seven native per-jet features (pT, eta, phi, mass, and the three
flavor-tag probabilities probB/probC/probL), under a permutation-safe evaluation
protocol that accounts for jet-assignment symmetry when scoring both the
classification and reconstruction tasks.

## Dataset

A 2-million-event training cohort built from the resolved HH → 4b simulation (see
[`../hh4b_simulation/`](../hh4b_simulation/README.md)), evaluated on a matched
400,000-event development cohort.

## Main result

Native SPA-Net (2M) reaches an all-background classification AUC of **0.969281**
(QCD AUC 0.968155, ttbar AUC 0.975392) and an exact-event HH-reconstruction
(jet-pairing) match rate of **0.866588** — clearly ahead of both classical BDT
baselines in the physically normalized comparison (see
[`../physical_normalization/`](../physical_normalization/README.md)).

## Key figures/tables

- `artifacts/hh4b_spanet_part_20260911/tables/auc_summary.tsv`,
  `reconstruction_summary.tsv`
- `artifacts/hh4b_spanet_part_20260911/training/training_native_spanet_2M_seed0_result.json`
- `docs/paper/jhep_hh4b_ml/single_head_spanet/` — paper figures/tables for this
  model.

## Code/config pointers

- `scripts/delphes/build_hh4b_spanet_npz.py`, `build_hh4b_spanet_qcdplus_npz.py` —
  input construction.
- `scripts/delphes/train_hh4b_spanet_qcdplus.py` — the delphes-stage training
  entry point.
- `scripts/analysis/pn_c7x_spanet_model.py`, `pn_c7x_spanet_common.py`,
  `run_pn_c7x_single_head_spanet.py`, `publish_pn_c7x_single_head_spanet.py`.
- `tests/test_pn_c7x_spanet_model.py`, `test_pn_c7x_spanet_common.py`,
  `test_pn_c7x_spanet_evaluation.py`, `test_run_pn_c7x_single_head_spanet.py`.

## Frozen artifact

`artifacts/hh4b_spanet_part_20260911/` — the native-SPA-Net rows of this bundle
specifically (`training/training_native_spanet_2M_seed0_result.json`,
corresponding `tables/` and `plots/` entries). This bundle is shared with the
training-scale and ParT-representation studies; see those pages for the rest of
its contents.

## Provenance

- `docs/checkpoints/track_b_phase2_*_20260804_v1/`,
  `track_b_phase2_inference_namespace_closure_20260806_v1/` — permutation-safe
  evaluation and inference-namespace groundwork this model's evaluation protocol
  relies on.
- `docs/paper/jhep_hh4b_ml/single_head_spanet/` — retains its original internal
  naming in its path; the study itself is described above without that naming.
