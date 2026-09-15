# Branch guide

This repository's history is spread across many branches rather than linear
commits to `main` — a consequence of how the SURF project was actually run
(each new topic frequently got its own branch, not always merged back). This
guide exists so a reader can tell, for any branch, what it represents,
whether it holds unique work, and what to read instead if it doesn't.

**As of 2026-09-15 (afternoon), `main` is the complete, canonical SURF
repository** — `repo-reorg/2026-09` was promoted into `main` by a
fast-forward-only merge (no squash, no rebase, tip
`87720169b28855330268ca006634897cfc063f0f` on both). A tag,
`pre-surf-reorg-main-20260915`, marks exactly where `main` stood
immediately before that promotion, for anyone who needs the pre-reorg
state. `repo-reorg/2026-09` is kept (identical to `main`) as the named
provenance branch for the reorganization itself, not as a separate
"more current" branch. **No branch is deleted, renamed, or merged away by
this reorganization** — this guide is documentation only. Commit counts
below are "commits reachable from this branch but not from `main`"
(computed 2026-09-15, before the promotion; `repo-reorg/2026-09` and
`main` are interchangeable for this purpose since the promotion).

## Branch categories

A conceptual grouping (documentation only — no branch below is renamed or
moved to conform to this taxonomy):

| Category | Branches |
|---|---|
| **Canonical / current** | `main` (== `repo-reorg/2026-09`, kept as the provenance branch for this reorganization) |
| **Simulation / production** | `delphes-hh4b-production` |
| **SPA-Net development** | `origin/recover-hh4b-spanet`, `track-b-development-snapshot-20260821`, `track-b-physical-normalization-20260824`, `spanet-part-resource-aware` |
| **Normalization / sensitivity** | `cms-resolved-sensitivity-gap-v1`, `bdt-apples-to-apples-v1` |
| **Representation-learning experiments** | `track-b-sophon-transfer`, `track-b-literature-resources-20260819` (its literature-review companion) |
| **Recovery / snapshots** | `origin/pivot-channel-scouting` (pre-`recover-hh4b-spanet` snapshot of the same early post-pivot work) |
| **Agent scratch (non-scientific)** | `worktree-agent-a2ed451390d084359` |

## How to read this table

- **Governing** — the branch (or `repo-reorg/2026-09`, which absorbed most
  branches' content) is the one to actually read for that topic.
- **Historical** — superseded by later work, but the commits are a genuine,
  non-duplicated record of an earlier stage.
- **Unique, unmerged** — contains real content not present anywhere on
  `repo-reorg/2026-09` or `main`. Worth knowing about even though nothing in
  this stage moves it.
- **Agent scratch** — created by an automated worktree/agent process, not
  meaningful project history.

| Branch | Scientific stage | Timeframe | Unique vs. `repo-reorg/2026-09` | Status | Read instead |
|---|---|---|---|---|---|
| `main` | **The complete, canonical SURF repository** (promoted from `repo-reorg/2026-09` 2026-09-15 by fast-forward; identical tip) | 2026-05 → 2026-09-15 | — | **canonical** | — |
| `repo-reorg/2026-09` | This reorganization's working branch; identical to `main` since the 2026-09-15 promotion | 2026-07 → 2026-09-15 | — | **provenance (kept, == `main`)** | `main` |
| `origin/pivot-channel-scouting` (remote-only) | The literal HH→bbWW→HH→4b **pivot decision**: `notes/channel_scouting_decision.md` (300 lines) compares candidate channels and picks resolved HH→4b; also holds the first resolved-HH4b BDT/DNN/LBN training scripts | 2026-07-01 → 07-02 | 5 commits, never merged anywhere | **unique, unmerged** | See recommendation below |
| `origin/recover-hh4b-spanet` (remote-only) | Builds on `pivot-channel-scouting`; first resolved-HH4b SPA-Net dataset builder and a GenPart-truth diagnostic, immediately after the pivot | 2026-07-02 → 07-03 | 9 commits, never merged anywhere | **unique, unmerged** | See recommendation below |
| `track-b-sophon-transfer` | Early "Track B" = validating the external PKU-HEP Sophon/jetfree-HH4b released model, plus a **386-line draft paper manuscript** (`docs/paper/track_b/hh4b_representation_learning_manuscript.md`) and Phase-4 interpretation-freeze provenance | 2026-08-14 → 08-17 | 12 commits, never merged anywhere | **unique, unmerged** | See recommendation below |
| `track-b-literature-resources-20260819` | One literature-review document for the Track B / Sophon line | 2026-08-19 | 1 commit (`docs/paper/track_b/LITERATURE_RESOURCES.md`) | unique, unmerged (small) | trivial to fold in later; see recommendation |
| `track-b-development-snapshot-20260821` | The 2026-08-21 four-model (BDT-K/KF, SPA-Net 2M/10M) development snapshot | 2026-08-21 | 0 commits — tip is already an ancestor of `repo-reorg/2026-09` | historical, fully absorbed | `docs/studies/05_spanet_reconstruction/`, `docs/studies/06_scaling_and_tail_reliability/` |
| `track-b-physical-normalization-20260824` | The four-model physical-normalization freeze (`docs/paper/jhep_hh4b_ml/track_b_physical_normalization/`) | 2026-08-24 | 0 commits — tip is already an ancestor of `repo-reorg/2026-09` | historical, fully absorbed | `docs/studies/05_spanet_reconstruction/` |
| `delphes-hh4b-production` | HH→4b Delphes production line; tip adds SPA-Net "compressed-tail diagnostics" (3 figures + tables) not present elsewhere | 2026-07 → 2026-09-02 | 1 commit (`e17fdbf`, "Add SPA-Net compressed-tail diagnostics") | mostly absorbed; **tip commit unique** | tip commit content not yet linked from any study — see recommendation |
| `spanet-part-resource-aware` | The **other active worktree**'s branch (`/uscms_data/d3/iturkmen/repos/hh-bbww-baselines-bdt`) — do not touch, per this task's explicit instructions | 2026-09 (ongoing) | 1 commit (`5e8387e`, in progress on that worktree) | **active, external to this task** | leave alone |
| `bdt-apples-to-apples-v1` | A distinct, later BDT methodology variant — "apples-to-apples" nested-OOF production comparison against the primary BDT and Harvey's ROC-tail audit, with its own scripts/tables/tests | 2026-08 → 08-11 | 8 commits, ~105 files, never merged | **unique, unmerged** | see recommendation |
| `cms-resolved-sensitivity-gap-v1` | A diagnostic study of whether the QCD tail has enough Monte Carlo support to close a "CMS resolved-sensitivity gap" — expected-limit ladder, lower-b-tag transfer feasibility, remediation decision | 2026-08 → 08-11 | 5 commits, never merged | **unique, unmerged** | see recommendation |
| `worktree-agent-a2ed451390d084359` | Auto-generated by a Claude Code agent worktree at some point in this project's history; identical to `main`, zero unique commits | unknown | 0 commits (identical to `main`) | **agent scratch, no unique content** | ignore |

## Branches with unique, unmerged content — status

Five branches/remote-refs hold real scientific content that existed
**nowhere else**, including not on `repo-reorg/2026-09`, as of 2026-09-15
morning. **As of 2026-09-15 afternoon, their headline content and/or small
primary-source documents have been incorporated additively** into the
relevant `docs/studies/` READMEs (full text copies, not summaries alone,
for the three smallest/highest-value documents) — **the source branches
themselves are untouched, unmerged, undeleted, and still hold the complete
raw artifact trees** (large figure/table sets were deliberately not
duplicated, per the "prefer links over duplicate gigabytes" policy):

1. `origin/pivot-channel-scouting` — the actual pivot-decision document,
   `notes/channel_scouting_decision.md`, is now preserved verbatim at
   [`docs/studies/03_channel_pivot_and_hh4b_simulation/channel_scouting_decision.md`](../studies/03_channel_pivot_and_hh4b_simulation/channel_scouting_decision.md)
   and linked from that study's README. Its first post-pivot BDT/DNN/LBN
   training scripts (also on `origin/recover-hh4b-spanet`, which is a
   superset — see below) remain unmerged; they are superseded in
   substance by the governing classical-ML ladder in
   `docs/studies/04_hh4b_classical_ml/`, so were not additionally copied.
2. `origin/recover-hh4b-spanet` — a superset of (1) plus the first
   resolved-HH4b SPA-Net dataset-builder script and a GenPart-truth
   diagnostic; superseded in substance by the governing SPA-Net line in
   `docs/studies/05_spanet_reconstruction/`. Kept for provenance; not
   additionally copied (early/superseded code, not a decision document).
3. `track-b-sophon-transfer` — its manuscript's headline quantitative
   result (SophonAK4 continuous-flavor QCD-rejection improvement:
   +30.9%/+53.8% dev/holdout at ε_S=0.4) is now stated directly in
   [`docs/studies/05_spanet_reconstruction/README.md`](../studies/05_spanet_reconstruction/README.md),
   correctly labeled as a provisional working manuscript, not a frozen
   result. The manuscript itself and the Track A/B mechanism-
   interpretation packages remain unmerged (large, LaTeX-figure-heavy
   trees) — not copied.
4. `bdt-apples-to-apples-v1` and `cms-resolved-sensitivity-gap-v1` — both
   summarized with exact headline numbers and diagnoses in
   [`docs/studies/04_hh4b_classical_ml/README.md`](../studies/04_hh4b_classical_ml/README.md).
   Full artifact trees (tables/figures/models) remain unmerged, not
   copied.
5. `delphes-hh4b-production` (tip commit only) — its SPA-Net compressed-
   tail package is preserved verbatim at
   [`docs/studies/06_scaling_and_tail_reliability/spanet_10m_compressed_tail_20260902.md`](../studies/06_scaling_and_tail_reliability/spanet_10m_compressed_tail_20260902.md)
   and linked from that study's README; its 3 figures were not copied
   (the text package is self-contained without them).
6. `track-b-literature-resources-20260819` — its one document is preserved
   verbatim at
   [`docs/studies/05_spanet_reconstruction/LITERATURE_RESOURCES.md`](../studies/05_spanet_reconstruction/LITERATURE_RESOURCES.md).

**What remains a genuine follow-up, if ever wanted:** a normal `git merge`
or targeted `git cherry-pick`/`git checkout <branch> -- <path>` of the
*full* artifact trees (figures, tables, model files, scripts) behind items
1–5 above, if a future reader decides the raw reproducibility artifacts
(not just the documented headline findings, which are now on
`repo-reorg/2026-09`) are worth carrying forward too. Not done here,
consistent with "prefer links/documentation over duplicate gigabytes of
data."

**Curated archival branches:** given the above, 1–2 curated archival
branches could materially help a future reader — e.g. a single
`archive/pre-repo-reorg-unmerged-work` branch that merges
`pivot-channel-scouting`, `recover-hh4b-spanet`, `track-b-sophon-transfer`,
`bdt-apples-to-apples-v1`, and `cms-resolved-sensitivity-gap-v1` together
(preserving all five without picking a "winner"), so a reader has one place
to look for "unmerged work" instead of five branch names. **This is a
proposal only — it was not created**, per this task's instruction not to
create new branches automatically.
