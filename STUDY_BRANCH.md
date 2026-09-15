# Study branch: 06 — training-scale and tail reliability

This is a reader-facing **study branch**: a curated snapshot of `main`
letting a visitor land directly on one scientific study without first
learning the project's historical Track A/B branch names. It is `main`
(full history preserved, nothing squashed) plus one merged branch and
this file. For the complete SURF repository, including all other studies,
see `main`.

## Scientific question

Does 5× more training data (10M vs. 2M events) actually improve the
governing native SPA-Net checkpoint (study 05)? And can its rare-event
(high-signal-efficiency) tail behavior be trusted numerically (FP32
precision) and statistically (finite QCD Monte Carlo)?

## Dates

2026-08-21 (first 10M development check) → 2026-09-14 (10M reconstruction
addendum); the newly merged compressed-tail package is dated 2026-09-02,
chronologically between and directly foreshadowing the Harvey FP32/tail
investigation (2026-09-06/07).

## Relevant code

`artifacts/hh4b/pretrained_jet_representations/zero20_20260914/code/compute_native10m_reconstruction.py`;
`docs/checkpoints/track_b_harvey_fp32_score_precision_20260906_v1/`, `_v2_governing10m/`,
`_v3_exact_harvey_thresholds/`; `track_b_harvey_tail_statistics_generation_20260906_v1/`,
`_20260907_v2/`, `track_b_harvey_tail_kinematics_20260907_v2/`;
`track_b_harvey_final_question_closure_20260907_v1/` (fifth-jet causal test).

## Relevant data / provenance

`artifacts/hh4b_spanet_part_20260911/` (governing 2M-vs-10M comparison);
`docs/track_b/development_snapshot_20260821/spanet_10m_scaling/`.
**Newly merged into this branch** (previously only on the unmerged
`delphes-hh4b-production` branch, its one commit not otherwise on `main`):
`docs/analysis/hh4b_spanet_compressed_tail_20260902_v1/` (README, 3
figures, 2 CSVs — the raw original package). A curated copy of its text
already existed on `main` at
`docs/studies/06_scaling_and_tail_reliability/spanet_10m_compressed_tail_20260902.md`
(added during the 2026-09-15 documentation pass); this merge additionally
brings in the original figures/CSVs that curated copy did not duplicate.

## Main results

- Classification has **saturated, not improved**, with 5× more training
  data: all-background AUC 0.969281 (2M) vs. 0.969272 (10M).
- Reconstruction shows a small point-estimate improvement at 10M (+0.0056
  exact-event, +0.0035 per-Higgs) — no paired-bootstrap CI computed, so
  reported as a point estimate, not a significant claim.
- FP32 quantization is kinematically unremarkable — compressed score bins
  near tight thresholds show no special kinematic mode.
- **Newly incorporated in full — the 2026-09-02 compressed-tail package**:
  introduces the u=-log10(1-score) transform later reused throughout this
  study's tail characterization. At score≥0.9997: n=100 raw events
  (Neff=81.0, `FINITE_SUPPORT_CAUTION`); at ≥0.99997: n=15 (Neff=14.0,
  `EXTREMELY_LIMITED`). QCD supplies 61% of raw events but 83.8% of the
  weighted yield at ≥0.9997, rising to 100% of raw events by ≥0.99999. No
  single kinematic mode explains the tail (all |Spearman ρ|≤0.21).
- The fifth-jet effect is real and directional: median Δu_logit=−0.090;
  58.1% of qualifying events net-helped by extra jet(s), 41.9% net-hurt.

## Superseded / negative results

- QCD tail statistics are adequate at moderate depth (u>3.5, p=0.90
  consistency), not deeper (u>4.5: only 12 raw events — explicitly "too
  few to judge either way," not forced to a conclusion).
- A related, methodologically distinct classical-baseline sensitivity-gap
  study (`cms-resolved-sensitivity-gap-v1`) uses the same QCD-Neff/tail-
  support diagnostic style but targets the R_HH<34 cut baseline rather
  than SPA-Net — merged in full on `study/04-hh4b-classical-ml` instead,
  not duplicated here.

## Relationship to main

This branch is `main` plus one merge commit bringing in
`origin/delphes-hh4b-production`. That branch's history is almost entirely
already an ancestor of `main` (verified: only 1 of its commits is unique);
merging it therefore added only that one commit's content — the
compressed-tail package — with zero modifications or deletions to
existing files. Original commit authorship on the incoming history is
unchanged; only the merge commit is newly authored. Note this deviates
from an earlier suggested mapping of `delphes-hh4b-production` to
`study/03-hh4b-simulation`: verification showed its only unique content is
a SPA-Net score-tail diagnostic, scientifically a study-06 topic, not a
simulation-production one — the branch's simulation-era history is
already fully on `main` independent of this merge. `main` remains the
single complete, canonical SURF repository. See
[`docs/studies/06_scaling_and_tail_reliability/README.md`](docs/studies/06_scaling_and_tail_reliability/README.md)
on `main` for the full narrative.

## Source historical branches incorporated

- `delphes-hh4b-production` (tip `e17fdbf9`, 1 unique commit, 2026-09-02) —
  merged in full (which, given its history, means only its one unique
  commit).

Not merged: `track-b-development-snapshot-20260821` and
`track-b-physical-normalization-20260824` — verified 0 unique commits vs.
`main`, so nothing to add. The source branch was not deleted or altered by
this merge.
