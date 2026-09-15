# Study branch: 07 — pretrained jet representations

This is a reader-facing **study branch**: a curated snapshot of `main`
letting a visitor land directly on one scientific study without first
learning the project's historical Track A/B branch names. It is `main`
(full history preserved, nothing squashed) plus two merged branches and
this file. For the complete SURF repository, including all other studies,
see `main`.

## Scientific question

Does giving SPA-Net's jet-assignment and classification network access to
a frozen, pretrained per-jet embedding — computed once by an external
model and never fine-tuned — improve resolved HH→4b reconstruction and
classification beyond the native 7-feature representation?

## Dates

2026-08-14 (Sophon manuscript draft) → 2026-09-14 (ZERO20 final result);
2026-08-17 (Track A/B mechanism-interpretation freeze); 2026-08-19
(literature review); 2026-09-08 → 09-11 (ParT active20/ZERO20, native to
this branch's history already).

## Relevant code

`docs/checkpoints/track_f_*` (postproduction pipeline, evaluation
protocol, ParT active20 integration); `docs/checkpoints/track_g_jpjepa_*`
(JP-JEPA/SPA-Net compatibility prep). **Newly merged into this branch**
(previously only on the unmerged `track-b-sophon-transfer` branch):
`docs/track_a/mechanism_interpretation_20260817_v1/build_json.py` and its
companions; `docs/paper/jhep_hh4b_ml/mechanism_results_20260817/scripts/`
(figure/table builders, checksum verifier, independent validator).

## Relevant data / provenance

`docs/analysis_contracts/SPANET_PART_RESOURCE_AWARE_COMPARISON.md.save`
(frozen pre-registration contract — kept at this exact filename; 8+
frozen documents cite it by this literal path);
`artifacts/hh4b/pretrained_jet_representations/zero20_20260914/`.
**Newly merged in full:** `docs/paper_draft/hh4b_representation_learning_manuscript.md`
(the working manuscript itself — previously only summarized by quote on
`main`, now the full document is present); `docs/track_a/mechanism_interpretation_20260817_v1/`,
`docs/track_b/phase4H_benchmark_readiness_report/`,
`docs/track_b/phase4L4M_frozen_result/`, `docs/track_b/phase4W_historical_reconciliation_20260817_v1/`;
`docs/paper/jhep_hh4b_ml/mechanism_results_20260817/` (figures, LaTeX
tables, validation report); `docs/updates/hh4b_external_sophon_and_mechanism_status_2026_08_14.md`;
`docs/checkpoints/track_b_frozen_score_pilot_*` (2026-08-01); and
`docs/paper/track_b/LITERATURE_RESOURCES.md` (from `track-b-literature-resources-20260819` —
a second copy already existed at
`docs/studies/05_spanet_reconstruction/LITERATURE_RESOURCES.md` from the
2026-09 documentation pass; both kept, same content).

## Main results

- **Governing result**: appending 20 frozen ParT embedding dimensions to
  SPA-Net's native input significantly **harmed** both classification
  (ΔAUC −0.0251) and reconstruction (Δexact-event −0.371) at matched
  2M-event scale.
- **ZERO20 controlled ablation** (final, 2026-09-14): replacing the same
  20 input slots with constant zero reproduced **none** of that harm
  (ΔAUC −0.00017, ~30× below the pre-registered 0.005 practical floor) —
  ruling out "wider input layer alone" as the cause; the specific ParT
  embedding content and/or its preprocessing/optimization interaction is
  implicated instead.
- The ParT checkpoint used throughout is the **official JetClass-supervised**
  Particle Transformer — explicitly **not** a self-supervised model. Do not
  describe it otherwise.
- **Newly incorporated in full — the Sophon/jetfree-hh4b manuscript**
  (working draft, 2026-08-14, explicitly self-labeled provisional — "must
  be frozen before a submission draft is prepared"): on the external
  Sophon/jetfree-hh4b sample, adding continuous SophonAK4 b/c/light-flavor
  information to an otherwise kinematics-only classifier raised QCD
  rejection at ε_S=0.4 from 283.1±27.2 to 370.6±26.8 in development data
  (+30.9%) and from 285.7±43.9 to 439.5±38.4 in a held-out sample
  (+53.8%), reproducing across all three seeds tested. A frozen
  pairwise-attention augmentation did **not** add further improvement in
  that configuration.

## Superseded / negative results

- The manuscript's native-vs-representation-enhanced SPA-Net comparison is
  explicitly **not yet stated** pending its own blind protocol — do not
  treat any number implying that conclusion as final.
- JP-JEPA compatibility prep is exploratory future work only, explicitly
  "not a production authorization." Full128/full144 scale-feasibility is
  planning, not a completed result — a scale-feasibility audit recommends
  10M next, not full144, pending resolution of the ParT checkpoint's own
  still-open domain-mismatch verdict.

## Relationship to main

This branch is `main` plus two merge commits bringing in
`origin/track-b-sophon-transfer` (12 commits, fully unique — the largest
single incorporation in this reorganization) and
`origin/track-b-literature-resources-20260819` (1 commit). Both merges
were clean and conflict-free (73 files added, 0 modified, 0 deleted,
verified against `main`) — original commit authorship on the incoming
history is unchanged; only the two merge commits are newly authored.
`spanet-part-resource-aware` was **deliberately not merged here** (see
`study/05-spanet-reconstruction`'s STUDY_BRANCH.md for the full reasoning:
its scientific content is already an ancestor of `main`, and its one
unique commit is a superseded, path-conflicting parallel reorganization
attempt, not new science; that branch/worktree may also host active
SPA-Net-adjacent development and was left untouched on that basis). `main`
remains the single complete, canonical SURF repository. See
[`docs/studies/07_pretrained_jet_representations/README.md`](docs/studies/07_pretrained_jet_representations/README.md)
on `main` for the full narrative.

## Source historical branches incorporated

- `track-b-sophon-transfer` (tip `8e43e92d`, 12 commits, 2026-07-31 →
  08-17) — merged in full.
- `track-b-literature-resources-20260819` (tip `34d261fa`, 1 commit,
  2026-08-19) — merged in full.

Neither source branch was deleted or altered by this merge.
`spanet-part-resource-aware` (also considered for this study) was
evaluated and intentionally left unmerged and untouched.
