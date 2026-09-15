# Branch guide

This repository's history is spread across many branches rather than linear
commits to `main` — a consequence of how the SURF project was actually run
(each new topic frequently got its own branch, not always merged back). This
guide has two parts: the small set of branches an ordinary reader needs, and
the full historical-development inventory (now mostly retired, archive-
tagged branches) for anyone doing provenance work.

**`main` is the complete, canonical SURF repository.** `repo-reorg/2026-09`
(the 2026-09 documentation/provenance reorganization's working branch) was
promoted into `main` by a fast-forward-only merge on 2026-09-15 (no squash,
no rebase), then retired on 2026-09-16 once confirmed identical to `main`
with zero unique content — its exact tip is preserved by the pushed
annotated tag `archive/branches/repo-reorg-2026-09-20260916` (→
`87720169b28855330268ca006634897cfc063f0f`), and `pre-surf-reorg-main-20260915`
marks exactly where `main` stood immediately before the promotion.

**On 2026-09-16, 9 further historical topic branches were retired from the
remote branch list** (`git push origin --delete`) after each one's complete
commit history was verified reachable (`git merge-base --is-ancestor`) from
its destination `study/*` branch or `main`, and only after an annotated
archive tag for its exact original tip was created and confirmed on the
remote. **No history was rewritten, rebased, squashed, or force-pushed at
any point** — every one of these branches' commits, with original
authorship intact, remains reachable today from either `main` or a
`study/*` branch, or from its `archive/branches/*` tag if nowhere else.
Four of the nine still exist as **local-only** branches (not on GitHub) in
this repository's other active worktrees, which were left completely
untouched — see the table below for which.

## Reader-facing branches

For ordinary navigation, you need only these — the remote branch list is
now exactly this set plus `spanet-part-resource-aware` (kept live; see
below).

| Branch | What it is |
|---|---|
| `main` | The complete, canonical SURF repository. Start here. |
| `study/01-hh-bbww` | Study 01 (HH→bbWW baseline) entry point. |
| `study/02-collide` | Study 02 (COLLIDE dataset studies) entry point. |
| `study/03-hh4b-simulation` | Study 03 entry point. **Merges in** `pivot-channel-scouting` and `recover-hh4b-spanet` (both now retired/archived — see below). |
| `study/04-hh4b-classical-ml` | Study 04 entry point. **Merges in** `bdt-apples-to-apples-v1` and `cms-resolved-sensitivity-gap-v1` (both now retired/archived). |
| `study/05-spanet-reconstruction` | Study 05 entry point. No merge needed (candidates already fully absorbed or deliberately excluded — see below). |
| `study/06-scaling-tail-reliability` | Study 06 entry point. **Merges in** `delphes-hh4b-production` (now retired/archived). |
| `study/07-pretrained-representations` | Study 07 entry point. **Merges in** `track-b-sophon-transfer` and `track-b-literature-resources-20260819` (both now retired/archived). |
| `spanet-part-resource-aware` | Kept live and untouched — associated with current SPA-Net/Track K/Track L work. See below; do not retire without explicit confirmation that work has concluded. |

## Retired historical branches (preserved by archive tag)

These branches' remote copies (`origin/<branch>`) were deleted on
2026-09-16. Every one is fully preserved: reachable from the listed
destination branch, and additionally pinned at its exact original tip by
a pushed annotated tag. **Local copies remain, untouched, in the four
worktrees that still have them checked out** (noted per row) — deleting a
remote branch does not affect a local branch or its worktree.

| Retired branch | Original tip | Destination (full history verified reachable) | Archive tag | Local copy still exists? |
|---|---|---|---|---|
| `pivot-channel-scouting` | `6bf35191` | `study/03-hh4b-simulation` (transitively, via `recover-hh4b-spanet`) | `archive/branches/pivot-channel-scouting-20260916` | No — was remote-only even before retirement |
| `recover-hh4b-spanet` | `652db2b5` | `study/03-hh4b-simulation` | `archive/branches/recover-hh4b-spanet-20260916` | No — was remote-only even before retirement |
| `bdt-apples-to-apples-v1` | `753e1df3` | `study/04-hh4b-classical-ml` | `archive/branches/bdt-apples-to-apples-v1-20260916` | No — not worktree-pinned, deleted locally too |
| `cms-resolved-sensitivity-gap-v1` | `dedf3bbe` | `study/04-hh4b-classical-ml` | `archive/branches/cms-resolved-sensitivity-gap-v1-20260916` | **Yes** — checked out at `/uscms_data/d3/iturkmen/repos/hh-bbww-baselines-cms-gap` |
| `delphes-hh4b-production` | `e17fdbf9` | `study/06-scaling-tail-reliability` (tip commit only; its simulation-era history is otherwise already on `main`) | `archive/branches/delphes-hh4b-production-20260916` | **Yes** — checked out at `/uscms_data/d3/iturkmen/repos/hh-bbww-baselines` |
| `track-b-sophon-transfer` | `8e43e92d` | `study/07-pretrained-representations` | `archive/branches/track-b-sophon-transfer-20260916` | **Yes** — checked out at `/uscms_data/d3/iturkmen/repos/hh-bbww-baselines-docs` |
| `track-b-literature-resources-20260819` | `34d261fa` | `study/07-pretrained-representations` | `archive/branches/track-b-literature-resources-20260819-20260916` | **Yes** — checked out at `/uscms_data/d3/iturkmen/repos/hh-bbww-baselines-literature-20260819` |
| `track-b-development-snapshot-20260821` | `e82f5ab9` | `main` (0 unique commits — already a strict ancestor) | `archive/branches/track-b-development-snapshot-20260821-20260916` | No — not worktree-pinned, deleted locally too |
| `track-b-physical-normalization-20260824` | `2edb4cd9` | `main` (0 unique commits — already a strict ancestor) | `archive/branches/track-b-physical-normalization-20260824-20260916` | No — not worktree-pinned, deleted locally too |

## Branches kept, not retired

| Branch | Why kept |
|---|---|
| `spanet-part-resource-aware` (tip `5e8387ee`) | Explicitly kept live and untouched: associated with current SPA-Net/Track K/Track L work per direct instruction. Its already-frozen scientific content is an ancestor of `main`; its one commit not on `main` is a superseded, path-conflicting parallel `docs/studies/` reorganization attempt (not new science) — evaluated for merging into `study/05-spanet-reconstruction` and `study/07-pretrained-representations` and deliberately excluded from both, but the branch itself is left alone regardless. |
| `worktree-agent-a2ed451390d084359` (tip `cb38072d`) | Zero unique content (strict ancestor of `main`) and never pushed to origin (nothing to simplify on GitHub), but it is still a **registered git worktree** (`/uscms_data/d3/iturkmen/repos/hh-bbww-baselines/.claude/worktrees/agent-a2ed451390d084359`, last modified 2026-08-25 — stale but not deregistered). Deleting its local branch would require first removing that worktree registration, an action beyond branch cleanup that wasn't requested; left untouched rather than guessed at. |

## What "merged into a study branch" means, concretely

For `study/03-hh4b-simulation`, `study/04-hh4b-classical-ml`, and
`study/07-pretrained-representations`, the retired source branches above
were merged in with `git merge` (directory-rename conflicts resolved by
following `main`'s own already-adopted renames, e.g.
`docs/analysis/`→`docs/provenance/analysis_notes/`) — a real merge commit,
not a squash or a copy-paste. Every incoming commit keeps its original
author/committer identity; only the merge commits themselves are newly
authored. Each merge was verified conflict-free and additions-only (no
existing file on `main` was modified or deleted) before being pushed.

`study/05-spanet-reconstruction` and `study/06-scaling-tail-reliability`
each also carry `STUDY_BRANCH.md` explanations of why one branch/candidate
was deliberately *not* merged in (`spanet-part-resource-aware` for 05; the
`cms-resolved-sensitivity-gap-v1` cross-reference for 06, whose full
content lives on `study/04` instead to avoid the same commits appearing to
originate from two different curated branches).

## Recovering a retired branch, if ever needed

Every retired branch can be fully restored from its archive tag:
`git push origin refs/tags/archive/branches/<name>-20260916:refs/heads/<name>`
recreates `origin/<name>` at its exact original tip. Nothing was lost —
only the live remote branch pointer was removed.
