# Study 03 — channel pivot and HH→4b simulation

## Question

Having found HH→bbWW's WW-side reconstruction largely infeasible (see
[study 01](../01_hh_bbww/README.md)), which channel should the project
switch to, and can a CMS-realistic, physically-normalized, large-statistics
simulation of it be built from scratch?

## Dataset / samples

The project's **own** MG5→Pythia8→Delphes production (not COLLIDE — see
[study 02](../02_collide_dataset_studies/README.md)) for resolved
**HH→4b** (di-Higgs to four b-quarks): ggF/VBF HH4b signal, plus QCD bbbb
(multiple $\hat{p}_T$-sliced samples), ttbar/ttbb, Z(bb)bb, ZH4b, ZZ4b
backgrounds. Final production: 630 members, 5,000,000 background + 200,000
signal generated events, at CMS Run-2-like AK4/AK8 conventions.

## Method

1. **Channel pivot decision (2026-07-01/02)** — compared candidate channels
   and chose resolved HH→4b for its cleaner, fully hadronic final state,
   more directly comparable to CMS's own resolved-HH4b search than bbWW's
   lepton+MET/WW combinatorics. Rationale:
   `docs/project_overview/physics_goal_and_pivot.md`. The original
   decision document, `notes/channel_scouting_decision.md`, exists on the
   unmerged `origin/pivot-channel-scouting` branch — see
   [Branch guide](../../history/BRANCH_GUIDE.md).
2. **First validated samples (2026-07-06 → 07-10)** — 10k-event ggF-HEFT
   and VBF-SM signal, and QCD/Z/ttbar background pilots; a frozen
   `data_release/hh4b_delphes_analysis_v0_2026_07_08/` snapshot.
3. **CMS-convention upgrade (2026-07-14 → 07-16)** — moved from an earlier
   R=0.5 jet card to CMS-realistic AK4 (R=0.4)/AK8 (R=0.8) cards; an audit
   concluded Delphes b-tag fields must never be described as
   DeepJet/ParticleNet discriminators absent a calibrated discriminator.
4. **3b/4b control-category builder + Delphes source/branch audit
   (2026-07-25)** — verified 581/581 Delphes branch members resolved with
   zero missing, then built an exactly-3-tag control category reusing the
   same branches as the immutable 4b builder.
5. **ttbar production-recovery saga (2026-07-27 → 07-28)** — 8 of 27 ttbar
   members were lost to a stalled HTCondor cluster; exact regeneration
   (ttbar8) plus reconstruction-only recovery of 7 retained ROOT files
   (ttbar7) closed the full 27-member, 270,000-event hold.
6. **Full coverage audit + expanded development cache (2026-07-27 → 08-04)**
   — confirmed all 630/630 members accounted for (5.2M events), then built a
   464/121/45 train/val/final-eval grouped split.
7. **Physical normalization (2026-07-29 → 07-31, 41 checkpoints)** —
   converted raw simulated yields to Run-2-equivalent (138 fb⁻¹) physical
   yields: per-process denominator/generator-weight freezes, authoritative
   cross-section/branching-ratio sourcing for all 22 ordinary processes
   (4 batches), then a capstone chain assembling the global registry, the
   138 fb⁻¹ luminosity contract, and the final per-event physical weight and
   selected-yield freeze for both the 4b signal region and a 3b control.

## Main findings

- The pivot rationale held up: HH→4b's fully hadronic final state avoided
  the WW-side combinatorics that stalled the bbWW study.
- The simulation pipeline is fully accounted for and reproducible: every one
  of 630 generated members is traced to a registry entry, and every
  physical-weight number traces back through a checksum-sealed, fail-closed
  chain to an authoritative cross-section/branching-ratio source (see
  `docs/provenance/REPOSITORY_CONTENT_MAP.md`, section B, for the full
  41-checkpoint campaign breakdown).
- The ttbar production-recovery saga is preserved as an example of honest
  infrastructure trouble handled without shortcuts: rather than substitute
  or drop the 8 missing members, they were exactly regenerated and
  cross-checked against non-sealed legacy files before being counted.
- CMS-realistic AK4/AK8 conventions were adopted deliberately mid-project
  (not from day one) — downstream studies built on the earlier R=0.5 card
  should not be silently assumed comparable to later AK4/AK8-card results.

## Figures

- [`figures/expanded_cut_signal_background_efficiency_plane.png`](figures/expanded_cut_signal_background_efficiency_plane.png) — efficiency plane for the nominal R_HH<34 cut on the full expanded production cache.
- [`figures/threeb_fourb_control_mbb_plane_background.png`](figures/threeb_fourb_control_mbb_plane_background.png) — the 3b-control vs. 4b-signal m_bb plane validating the control-category builder.
- [`figures/ttbar27_member_recovery_status.png`](figures/ttbar27_member_recovery_status.png) — per-member status map for the full ttbar production-recovery saga.

## Reproducible code

- `cards/mg5/`, `cards/delphes/` — MadGraph5 process cards and Delphes detector cards (including the frozen `delphes_card_CMS_lpc_ak4ak8_run2_frozen_v2.tcl`).
- `config/production/hh4b_final_production_v1.yaml`, `configs/production/` — production configuration.
- `data_release/hh4b_delphes_analysis_v0_2026_07_08/` — a self-contained, checksummed release snapshot with its own `scripts_snapshot/` of the pipeline as it stood 2026-07-08.
- `scripts/analysis/audit_hh4b_3b_delphes_branches.py`, `build_hh4b_3b_root_source_map.py`, `audit_hh4b_denominator_evidence_and_sources.py`, `extract_hh4b_normalization_provenance.py`, `audit_hh4b_generator_artifact_locations.py` — the simulation/normalization audit scripts.
- `metadata/delphes/`, `metadata/production_plans/` — production-stage bookkeeping (not moved this stage; live script consumers).

## Relationship to the final HH→4b study

This is the **foundation** every later HH→4b result (classical ML, SPA-Net,
ParT/ZERO20) is built on: it is where the channel itself was decided and
where the dataset those later studies evaluate against was produced and
physically normalized. It is not itself a "result" in the sense of a
model comparison — it is the provenance layer that makes the later
model-comparison numbers physically meaningful (expressible in Run-2
fb⁻¹-equivalent yields rather than raw simulated counts).

## Provenance

Pivot rationale: `docs/project_overview/physics_goal_and_pivot.md`.
Simulation checkpoints: `docs/checkpoints/hh4b_reference_baseline_20260724_v1/`,
`hh4b_3b_*`, `hh4b_4b_*`, `hh4b_ttbar*`, `hh4b_expanded_cut_*`, `hh4b_full_5m*`
(26 directories, 2026-07-24 → 08-04). Physical normalization: `docs/checkpoints/hh4b_physical_normalization_*`
(41 directories, 2026-07-29 → 07-31), summarized in
`docs/provenance/analysis_notes/hh4b_physical_normalization_provenance.md`.
Early status updates: `docs/updates/hh4b_validation_update_2026_07_07.md`,
`docs/harvey_update_2026_07_08/`. Background-compatibility rules:
`docs/research/background_inventory_and_ml_readiness_20260720.md`. Full
inventory-vs-reuse audit: `outputs/audits/existing_hh4b_assets_2026_07_16/`.
Original pivot-decision document (unmerged branch, not moved into this
tree): `notes/channel_scouting_decision.md` on `origin/pivot-channel-scouting`
— see [`docs/history/BRANCH_GUIDE.md`](../../history/BRANCH_GUIDE.md).
