# Study branch: 05 — SPA-Net reconstruction and classification

This is a reader-facing **study branch**: a curated snapshot of `main`
letting a visitor land directly on one scientific study without first
learning the project's historical Track A/B branch names. It is `main`
(full history preserved, nothing squashed) plus this file — no branch
merge was needed here (see below). For the complete SURF repository,
including all other studies, see `main`.

## Scientific question

Does a symmetry-preserving, permutation-aware jet-assignment network
(SPA-Net) both classify HH→4b events and correctly pair jets into two
Higgs candidates better than the classical-ML ladder (study 04) plateaued
at? Separately: does an externally released representation model
(Sophon/jetfree-hh4b) offer a useful comparison point?

## Dates

2026-07-13 (first SPA-Net attempt) → 2026-08-24 (four-model
physical-normalization archive).

## Relevant code

`docs/paper/jhep_hh4b_ml/single_head_spanet/` (single-head SPA-Net paper
package); `docs/checkpoints/track_b_phase2_*` (11 dirs — Sophon
released-model validation scripts/contracts).

## Relevant data / provenance

`docs/checkpoints/track_b_phase2_*` (2026-08-01 → 08-06), `track_b_phase3_*`
(paused 2026-08-06); `docs/track_b/development_snapshot_20260821/`;
`docs/paper/jhep_hh4b_ml/track_b_physical_normalization/` (2026-08-24
four-model archive); `artifacts/hh4b_spanet_part_20260911/` (frozen
native-SPA-Net evaluation bundle); `docs/plans/track_b_track_a_integration_plan_20260801_v1.md`.
`docs/studies/05_spanet_reconstruction/LITERATURE_RESOURCES.md` (Chiang et
al. 2024 for native SPA-Net; Yang & Li 2508.15048 for the Sophon/jetfree-hh4b
dataset/framework — copied in during the 2026-09 documentation
reorganization from `track-b-literature-resources-20260819`, itself now
incorporated in full on `study/07-pretrained-representations`, since that
literature review is most directly about the pretrained-representation
line of work rather than native SPA-Net).

## Main results

- First SPA-Net attempt (S/√B=0.163) sat between DNN (0.154) and BDT
  (0.199) baselines, with weak top-background rejection (weighted
  AUC=0.460) — motivated a two-head redesign.
- Continuous flavor information (K→KF) was the **dominant** gain in the
  four-model development snapshot; SPA-Net gave a further, statistically
  robust gain over KF in the QCD-tail-supported region.
- **Governing native SPA-Net (2M)**: all-background AUC 0.969281,
  exact-event HH reconstruction 0.866588 — the baseline every later
  scaling (study 06) and pretrained-representation (study 07) result is
  measured against.

## Superseded / negative results

- The Sophon/jetfree-hh4b released model reproduced correctly and
  localized physical resonance masses as expected, but its adapter-
  integration extension was explicitly **paused** (2026-08-06) rather than
  pursued to completion — an early representation-learning direction that
  did not continue past a feasibility canary. Its later, still-provisional
  working-manuscript continuation (quantitative result: +30.9%/+53.8%
  QCD-rejection improvement from continuous SophonAK4 flavor info,
  dev/holdout, at ε_S=0.4) is documented and now fully merged on
  branch `study/07-pretrained-representations`, not here — it is scientifically a representation-learning result, not a
  native-SPA-Net one.

## Relationship to main

This branch **is** `main` at the point this file was added — no merges
were performed here. Both `track-b-development-snapshot-20260821` and
`track-b-physical-normalization-20260824` were verified (via `git
rev-list --left-right --count` against `main`) to have **zero commits
unique** to them — their content is already fully an ancestor of `main`,
so merging them would be a no-op. `spanet-part-resource-aware` was
deliberately **not** merged: its scientific content (the governing
SPA-Net/ParT results) is already an ancestor of `main` (verified: its
base commit `73516e9` is an ancestor of `main`), and its one commit not
on `main` is a superseded, differently-named parallel `docs/studies/`
reorganization attempt (predating this one) whose root-level files
(`README.md`, `PROJECT_STORY.md`, `RESULTS_OVERVIEW.md`) would directly
path-conflict with `main`'s actual files — merging it would overwrite or
conflict with, not add to, this project's current documentation. That
branch/worktree is also explicitly flagged as potentially hosting active
SPA-Net-adjacent development and was left untouched on that basis too.
`main` remains the single complete, canonical SURF repository. See
[`docs/studies/05_spanet_reconstruction/README.md`](docs/studies/05_spanet_reconstruction/README.md)
on `main` for the full narrative.

## Source historical branches incorporated

None merged. Verified and intentionally left unmerged:
- `track-b-development-snapshot-20260821` — 0 unique commits vs. `main`.
- `track-b-physical-normalization-20260824` — 0 unique commits vs. `main`.
- `spanet-part-resource-aware` — 1 unique commit, superseded/conflicting
  content, not merged (see above); branch kept untouched.
