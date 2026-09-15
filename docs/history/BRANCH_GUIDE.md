# Branch guide

This repository's history is spread across many branches rather than linear
commits to `main` — a consequence of how the SURF project was actually run
(each new topic frequently got its own branch, not always merged back). This
guide has two parts: the small set of branches an ordinary reader needs, and
the full historical-development inventory for anyone doing provenance work.

**`main` is the complete, canonical SURF repository.** `repo-reorg/2026-09`
(the 2026-09 documentation/provenance reorganization's working branch) was
promoted into `main` by a fast-forward-only merge on 2026-09-15 (no squash,
no rebase). Once identical to `main` with zero unique content of its own,
`repo-reorg/2026-09` was retired on 2026-09-16 — its exact tip is preserved
by the pushed annotated tag `archive/branches/repo-reorg-2026-09-20260916`
(→ `87720169b28855330268ca006634897cfc063f0f`), and `pre-surf-reorg-main-20260915`
marks exactly where `main` stood immediately before the promotion. **No
other branch is deleted, renamed, or merged away by this reorganization.**

## Reader-facing branches

For ordinary navigation, you need only these. Each `study/*` branch is
`main` (full history, nothing squashed or removed) plus one root-level
`STUDY_BRANCH.md`; three of them additionally merge in real commit history
from old topic branches (marked below) — see the per-branch detail further
down for exactly what and why.

| Branch | What it is |
|---|---|
| `main` | The complete, canonical SURF repository. Start here. |
| `study/01-hh-bbww` | Study 01 (HH→bbWW baseline) entry point. |
| `study/02-collide` | Study 02 (COLLIDE dataset studies) entry point. |
| `study/03-hh4b-simulation` | Study 03 entry point. **Merges in** `recover-hh4b-spanet` (and transitively `pivot-channel-scouting`). |
| `study/04-hh4b-classical-ml` | Study 04 entry point. **Merges in** `bdt-apples-to-apples-v1` and `cms-resolved-sensitivity-gap-v1`. |
| `study/05-spanet-reconstruction` | Study 05 entry point. No merge needed (candidates already fully absorbed or deliberately excluded — see below). |
| `study/06-scaling-tail-reliability` | Study 06 entry point. **Merges in** `delphes-hh4b-production`. |
| `study/07-pretrained-representations` | Study 07 entry point. **Merges in** `track-b-sophon-transfer` and `track-b-literature-resources-20260819`. |

## Archived historical development branches

Everything below this point is for provenance/audit purposes, not ordinary
navigation. Commit counts are "commits reachable from this branch but not
from `main`," computed 2026-09-15/16.

- **Governing** — read `main` (or the relevant `study/*` branch) instead.
- **Merged into a study branch** — its content is now reachable from that
  `study/*` branch via a real `git merge` (original authorship preserved),
  in addition to remaining on its own original branch (which is untouched).
- **Historical, fully absorbed** — zero commits unique vs. `main`; nothing
  to merge, kept only as a named historical pointer.
- **Kept, not merged** — evaluated and deliberately left out of every
  `study/*` branch, with a stated reason; branch itself untouched.
- **Agent scratch** — created by an automated worktree/agent process, not
  meaningful project history.

| Branch | Scientific stage | Timeframe | Unique vs. `main` | Status | Destination |
|---|---|---|---|---|---|
| `origin/pivot-channel-scouting` (remote-only) | The literal HH→bbWW→HH→4b **pivot decision**: `notes/channel_scouting_decision.md` (300 lines) compares candidate channels and picks resolved HH→4b; also holds the first resolved-HH4b BDT/DNN/LBN training scripts | 2026-07-01 → 07-02 | 5 commits | **merged into a study branch** (transitively, via `recover-hh4b-spanet`) | `study/03-hh4b-simulation` |
| `origin/recover-hh4b-spanet` (remote-only) | Builds on `pivot-channel-scouting`; first resolved-HH4b SPA-Net dataset builder and a GenPart-truth diagnostic | 2026-07-02 → 07-03 | 9 commits | **merged into a study branch** | `study/03-hh4b-simulation` |
| `track-b-sophon-transfer` | Validating the external PKU-HEP Sophon/jetfree-HH4b released model, plus a working draft manuscript and Phase-4 interpretation-freeze provenance | 2026-08-14 → 08-17 | 12 commits | **merged into a study branch** | `study/07-pretrained-representations` |
| `track-b-literature-resources-20260819` | One literature-review document for the Sophon/representation-learning line | 2026-08-19 | 1 commit | **merged into a study branch** | `study/07-pretrained-representations` |
| `track-b-development-snapshot-20260821` | The 2026-08-21 four-model (BDT-K/KF, SPA-Net 2M/10M) development snapshot | 2026-08-21 | 0 commits | **historical, fully absorbed** | `docs/studies/05_spanet_reconstruction/`, `docs/studies/06_scaling_and_tail_reliability/` on `main` |
| `track-b-physical-normalization-20260824` | The four-model physical-normalization freeze | 2026-08-24 | 0 commits | **historical, fully absorbed** | `docs/studies/05_spanet_reconstruction/` on `main` |
| `delphes-hh4b-production` | HH→4b Delphes production line; tip adds SPA-Net "compressed-tail diagnostics" | 2026-07 → 2026-09-02 | 1 commit (`e17fdbf`) | **merged into a study branch** (tip commit only — its simulation-era history is otherwise already on `main`) | `study/06-scaling-tail-reliability`, not `study/03` (verified from the diff: its unique content is a tail diagnostic, not simulation work) |
| `bdt-apples-to-apples-v1` | A distinct, later BDT methodology variant — "apples-to-apples" nested-OOF comparison against the primary BDT and a Harvey ROC-tail audit against CMS | 2026-08 → 08-11 | 8 commits, ~105 files | **merged into a study branch** | `study/04-hh4b-classical-ml` |
| `cms-resolved-sensitivity-gap-v1` | A diagnostic study of whether the QCD tail has enough Monte Carlo support to close a "CMS resolved-sensitivity gap" — expected-limit ladder, lower-b-tag transfer feasibility | 2026-08 → 08-11 | 5 commits | **merged into a study branch** | `study/04-hh4b-classical-ml` |
| `spanet-part-resource-aware` | An active development branch/worktree (`/uscms_data/d3/iturkmen/repos/hh-bbww-baselines-bdt`) with its own SPA-Net/ParT scientific history | 2026-09 (ongoing) | 1 commit vs. `main` | **kept, not merged** — its scientific content is already an ancestor of `main`; its one unique commit is a superseded, path-conflicting parallel `docs/studies/` reorganization attempt, not new science. May host active follow-on SPA-Net work — left completely untouched | none (intentionally excluded) |
| `worktree-agent-a2ed451390d084359` | Auto-generated by a Claude Code agent worktree; identical to `main` | unknown | 0 commits | **agent scratch, no unique content** | none |
| `repo-reorg/2026-09` | *(retired 2026-09-16)* The 2026-09 reorganization's working branch, promoted into `main` | 2026-07 → 2026-09-15 | 0 commits | **deleted; archived** | tag `archive/branches/repo-reorg-2026-09-20260916` |

## What "merged into a study branch" means, concretely

For `study/03-hh4b-simulation`, `study/04-hh4b-classical-ml`, and
`study/07-pretrained-representations`, the listed source branches were
merged in with `git merge` (directory-rename conflicts resolved by
following `main`'s own already-adopted renames, e.g.
`docs/analysis/`→`docs/provenance/analysis_notes/`) — a real merge commit,
not a squash or a copy-paste. Every incoming commit keeps its original
author/committer identity; only the merge commits themselves are newly
authored. Each merge was verified conflict-free and additions-only (no
existing file on `main` was modified or deleted) before being pushed. The
source branches are untouched, unmerged-into-`main`, and undeleted — they
remain independently readable at their original tips.

`study/05-spanet-reconstruction` and `study/06-scaling-tail-reliability`
each also carry `STUDY_BRANCH.md` explanations of why one branch/candidate
was deliberately *not* merged in (`spanet-part-resource-aware` for 05; the
`cms-resolved-sensitivity-gap-v1` cross-reference for 06, whose full
content lives on `study/04` instead to avoid the same commits appearing to
originate from two different curated branches).
