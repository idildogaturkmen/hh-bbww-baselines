# Study branch: 03 — channel pivot and HH→4b simulation

This is a reader-facing **study branch**: a curated snapshot of `main`
letting a visitor land directly on one scientific study without first
learning the project's historical Track A/B branch names. It is `main`
(full history preserved, nothing squashed) plus one merged branch and this
file. For the complete SURF repository, including all other studies, see
`main`.

## Scientific question

Having found HH→bbWW's WW-side reconstruction largely infeasible (study 01),
which channel should the project switch to, and can a CMS-realistic,
physically-normalized, large-statistics simulation of it be built from
scratch?

## Dates

2026-07-01 (pivot decision) → 2026-08-04 (full 5.2M-event coverage +
physical normalization complete).

## Relevant code

`cards/mg5/`, `cards/delphes/` (MadGraph5/Delphes cards); `config/production/hh4b_final_production_v1.yaml`,
`configs/production/`; `scripts/analysis/audit_hh4b_3b_delphes_branches.py`,
`build_hh4b_3b_root_source_map.py`, `audit_hh4b_denominator_evidence_and_sources.py`,
`extract_hh4b_normalization_provenance.py`, `audit_hh4b_generator_artifact_locations.py`.
**Newly merged into this branch** (previously only on the unmerged
`recover-hh4b-spanet`/`pivot-channel-scouting` branches):
`scripts/channel_scouting/scout_hh_channels.py` (the channel-scouting scan
itself), `scripts/hh4b_baseline/{train_hh4b_resolved_bdt,train_hh4b_resolved_bdt_v2,train_hh4b_resolved_dnn_v1,train_hh4b_resolved_lbn_dnn_v2,build_hh4b_resolved_features,bdt_v2_stability_resampling,make_hh4b_model_comparison_summary}.py`
(the first post-pivot classical-ML scripts — early versions superseded in
result by study 04's governing ladder, kept here for provenance),
`scripts/hh4b_spanet/{build_hh4b_resolved_spanet_dataset,diagnose_hh4b_genpart_truth}.py`
(the first resolved-HH4b SPA-Net dataset builder and GenPart-truth
diagnostic — the direct ancestor of study 05's governing SPA-Net work).

## Relevant data / provenance

`data_release/hh4b_delphes_analysis_v0_2026_07_08/` (self-contained,
checksummed release snapshot); `metadata/delphes/`, `metadata/production_plans/`;
`docs/checkpoints/hh4b_reference_baseline_*`, `hh4b_3b_*`, `hh4b_4b_*`,
`hh4b_ttbar*`, `hh4b_expanded_cut_*`, `hh4b_full_5m*` (26 dirs, physical
normalization: 41 dirs, 2026-07-29 → 07-31);
`docs/project_overview/physics_goal_and_pivot.md` (pivot rationale). **The
original pivot-decision document** is preserved verbatim at
`docs/project_overview/channel_scouting_decision.md` (landed here by the
merge below, following `main`'s own `notes/`→`docs/project_overview/`
rename) — a second curated copy also lives at
`docs/studies/03_channel_pivot_and_hh4b_simulation/channel_scouting_decision.md`;
both are the same content, kept at both locations deliberately (one
following the merge's natural path, one at the curated study location).

## Main results

- The pivot rationale held up: HH→4b's fully hadronic final state avoided
  the WW-side combinatorics that stalled bbWW.
- Channel-scouting decision (now merged in full, see below): compared 8
  candidate regions by rough Asimov significance and picked
  `HH4b_resolved_4b_basic` (Z_A≈0.079, ~3.6× the bbWW-reference region's
  Z_A≈0.022).
- The simulation pipeline is fully accounted for: 630/630 generated
  members traced to a registry entry; every physical-weight number traces
  through a checksum-sealed chain to an authoritative cross-section/BR
  source; 5,000,000 background + 200,000 signal events, 138 fb⁻¹-equivalent.
- The ttbar production-recovery saga (8 of 27 members lost to a stalled
  HTCondor cluster) was closed by exact regeneration + cross-checked
  reconstruction-only recovery, not by silently dropping or substituting
  the missing members.

## Superseded / negative results

- CMS-realistic AK4(R=0.4)/AK8(R=0.8) conventions were adopted mid-project
  (2026-07-14 → 07-16), replacing an earlier R=0.5 jet card — results built
  on the earlier card are not directly comparable to later AK4/AK8 numbers.
- The first post-pivot classical-ML scripts (merged in from
  `recover-hh4b-spanet`/`pivot-channel-scouting`) are superseded in result
  by study 04's governing classical-ML ladder; kept here as the historical
  first attempt, not as a citable result.

## Relationship to main

This branch is `main` plus one merge commit bringing in
`origin/recover-hh4b-spanet` (which is itself a strict superset of
`origin/pivot-channel-scouting` — verified via `git merge-base --is-ancestor`
before merging, so no redundant merge was needed) — a branch that had never
been merged anywhere and held real, unique scientific history (the actual
pivot-decision document and the first post-pivot modeling scripts) not
otherwise reachable from `main`. The merge was a clean, conflict-free
addition (15 files, 6,473 lines, 0 modifications to existing files, 0
deletions) — original commit authorship on the incoming history is
unchanged; only the merge commit itself is newly authored. `main` remains
the single complete, canonical SURF repository. See
[`docs/studies/03_channel_pivot_and_hh4b_simulation/README.md`](docs/studies/03_channel_pivot_and_hh4b_simulation/README.md)
on `main` for the full narrative.

## Source historical branches incorporated

- `recover-hh4b-spanet` (remote-only, tip `652db2b5`, 9 commits, 2026-07-02
  → 07-03) — merged in full via `git merge`.
- `pivot-channel-scouting` (remote-only, tip `6bf35191`, 5 commits,
  2026-07-01 → 07-02) — transitively included, since it is an ancestor of
  `recover-hh4b-spanet`.

Neither source branch was deleted or altered by this merge.
