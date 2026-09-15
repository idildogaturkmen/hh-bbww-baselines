# Repository content map

A thematic inventory of this repository's content, built for the
`repo-reorg/2026-09` history-preserving reorganization. This is a
**provenance document**, not an onboarding guide — start at the root
`README.md`, `docs/project_overview/PROJECT_STORY.md`,
`docs/history/SURF_TIMELINE.md`, and `docs/studies/` instead. This page
exists so a future maintainer (or this same process, run again) can see what
exists, where, and why it was or wasn't moved, without re-deriving it from
scratch.

Scientific-study keys used below match `docs/studies/`: `01_hh_bbww`,
`02_collide_dataset_studies`, `03_channel_pivot_and_hh4b_simulation`,
`04_hh4b_classical_ml`, `05_spanet_reconstruction`,
`06_scaling_and_tail_reliability`, `07_pretrained_jet_representations`.
`cross_cutting_infra` denotes shared code/config, not one study.

## How to read "status"

- **governing** — the authoritative version of that result/campaign.
- **historical** — real, non-duplicated content from an earlier stage,
  superseded by later work but preserved as history.
- **superseded** — an earlier attempt at the same thing a later item
  replaced; kept for provenance, not for citation.
- **exploratory** — a genuine attempt with no strong conclusion yet, or
  explicitly self-labeled provisional/rough/proxy by its own source text.
- **blocked/incomplete** — an honest non-result (infrastructure failure,
  not a physics finding); see [D. Scientific integrity notes](#d-scientific-integrity-notes).

## A. Root-level files

| Item | Study | Type | Live consumers | Decision |
|---|---|---|---|---|
| `HH4B_CUT_BASELINE_FINAL_STATUS.{md,json}` | 04 | result | `docs/project_overview/PROJECT_STORY.md`, `docs/results/RESULTS_OVERVIEW.md` cite it by this exact root path; `scripts/analysis/build_hh4b_cut_baseline_final_report.py` and `build_hh4b_cut_baseline_blocked_final_report.py` hardcode it as their default write target; checksummed in `artifacts/hh4b_cut_baseline/final_status_blocked_validation/SHA256SUMS` | **left in place** — moving would require editing two scripts and re-checksumming in the same commit; deferred, same policy as `docs/checkpoints/`/`metadata/`/`Results Summaries/` below |
| `RESULTS_HBB_BJET_REGRESSION.md` | 01 | result | `scripts/train_hbb_bjet_regression.py` defaults `--summary-md` to this exact root path | **left in place** |
| `FULL144_PART_SPANET_SCALE_FEASIBILITY_20260826.{md,json}` | 07 | provenance/planning | none found (`git grep` repo-wide, self-references only) | **moved** → `docs/studies/07_pretrained_jet_representations/full144_scale_feasibility_20260826.{md,json}` |
| `inspect_collide2v.ipynb` | 02 | notebook | none found | **moved** → `docs/studies/02_collide_dataset_studies/inspect_collide2v.ipynb` |
| `docs/analysis_contracts/SPANET_PART_RESOURCE_AWARE_COMPARISON.md.save` | 07 | provenance (frozen pre-registration contract) | cited **by its exact `.md.save` path** from 8 separate already-frozen `docs/checkpoints/track_b_harvey_*` and `docs/checkpoints/track_f_*`/`track_g_*` documents, including one that explicitly calls it "superseded" | **left exactly as named** — the unusual extension looks like an accidental editor-save artifact, but the file's content is unique and real (the 2026-09-03 SPA-Net+ParT comparison contract), and renaming it would break citations inside documents that are themselves frozen provenance and should not be edited retroactively |
| `notes/` | — | empty directory | none (untracked, not in `git ls-files`) | left as-is; not part of the committed tree, nothing to move |

## B. `docs/checkpoints/` — the dated decision ledger (not moved, 170+ entries)

`docs/checkpoints/` holds this project's hash-stamped, dated "lab notebook":
one directory per frozen decision or audit step, each with a README, a
JSON/TSV contract, and usually a `SHA256SUMS`. **None of it was moved this
stage** — `scripts/`, `configs/baselines/`, and other files hardcode this
path as a read/write default (see root `README.md`'s provenance-policy
section for specific examples). What follows is a campaign-level map, not a
170-row enumeration — directories are grouped by the scientific campaign
they belong to; a `v1→v2→...` suffix sequence is a genuine revision chain
within one campaign, not duplication (version counters are independent
per-campaign, confirmed via `git log --reverse` chronology, not directory
naming).

| Campaign | Study | Dates | ~Dirs | Outcome |
|---|---|---|---:|---|
| Reference baseline + 3b/4b control-category builder + Delphes source audit | 03 | 2026-07-24 → 07-25 | 7 | Frozen CMS-inspired cutflow; 3b/4b builder validated exactly disjoint |
| ttbar8/ttbar7/ttbar27 production-recovery saga | 03 | 2026-07-27 → 07-28 | 11 | 27-member, 270,000-event ttbar hold closed after a multi-stage HTCondor recovery |
| Full 5M-background/200k-signal coverage + expanded cache + 5.2M workflow-role contract | 03 | 2026-07-27 → 08-04 | 5 | 630/630 members accounted for; train/val/test roles assigned |
| Expanded-cut classical baseline protocol | 03 / 04 | 2026-07-28 | 3 | Nominal R_HH<34 cut + BDT protocol on the expanded cache |
| Physical normalization: inventory/provenance audit | 03 | 2026-07-29 | 10 | Fail-closed inventory; zero weights assigned yet (pure reconnaissance) |
| Physical normalization: per-process denominator/weight freezes (QCD/ttbar/triboson/VBF/ggF) | 03 | 2026-07-29 → 07-30 | 13 | Each process family's denominator/weight-transport registry frozen |
| Physical normalization: authoritative cross-section/BR reference sourcing (Batches A–D) | 03 | 2026-07-30 → 07-31 | 8 | All 22 ordinary processes' references frozen |
| Physical normalization: **global registry, Run-2 luminosity, coefficient closure, train physical-yield freeze (governing)** | 03 | 2026-07-31 | 10 | `hh4b_physical_normalization_train_fourb_selected_yield_closure_20260731_v1` is the authoritative deliverable; 138 fb⁻¹ contract frozen |
| Cut-based baseline: coarse/fine/exact R_HH scan | 04 | 2026-07-24 → 07-25 | 4 | R_HH<34 frozen (0.221% from the exact optimum, rounded for reproducibility) |
| BDT model-choice & validation-selection campaign | 04 | 2026-07-25 → 07-27 | 7 | `global_v1_mass_aware` chosen as final nominal BDT over a categorized v2 alternative |
| Broad feature extractor & common train-table build-out ("Track A") | 04 | 2026-08-04 → 08-05 | 9 | Unified, schema-frozen, HTCondor-scaled train table from the full ~5.2M-event production |
| Cut-baseline re-run + one-time blind validation-access protocol | 04 | 2026-08-10 → 08-11 | 11 | Train-side frozen; validation batch failed at pre-execution probe (0 payloads opened) — **honest non-result**, not resubmitted |
| Baseline-ladder governance freezes (fold map, historical comparator, Run-2 authorization, nested-OOF contract) | 04 | 2026-08-05 → 08-06 | 4 | Fair 5-fold source-group comparison ladder frozen |
| Multivariate-cut method design & software-validation canaries | 04 | 2026-08-06 → 08-07 | 9 | 54-family geometric-cut optimizer built, leakage-checked |
| 270-job production cut scan + nominal deployment candidate | 04 | 2026-08-07 | 3 | Modal winner selected in only 2/5 folds (0.40 frequency) → triggered stability escalation |
| Selection-stability bootstrap: draw registry → initial-200 → escalation-800 → all-1000 | 04 | 2026-08-08 → 08-10 | 20 | Low, non-dominant modal frequencies confirmed at full 1000-replica budget — instability characterized rigorously, not papered over |
| Final cut-baseline train performance, comparator, 16 publication figures | 04 | 2026-08-10 | 2 | `train_performance_summary.json` + figure set frozen |
| Track B Phase 2: released Sophon model validation & control-sample inference | 05 | 2026-08-01 → 08-06 | 11 | External model reproduced bit-for-bit; ZH/ZZ mass localization matched physical resonances |
| Track B Phase 3: Sophon-adapter integration contracts (paused) | 06 | 2026-08-06 | 6 | Bounded 4-file canary built; explicitly paused, refocused to the project's own SPA-Net baseline |
| Development snapshot 2026-08-21 (4-model comparison, native scaling adjudication, residual-background prototype) | 05 / 06 | 2026-08-21 | 4 | Continuous flavor info (K→KF) dominant gain; SPA-Net further gain; 2M→10M classification essentially flat |
| Four-model physical normalization archive (paper-facing) | 05 | 2026-08-24 | 20 | Canonical 68-row physically-normalized result set, 3-way cross-checked to ~1e-8 |
| Harvey FP32 score-precision investigation (v1→v2→v3) | 06 | 2026-09-06 → 09-07 | 3 | No special kinematic mode at FP32 quantization spikes |
| Harvey tail statistics & tail kinematics characterization | 06 | 2026-09-06 → 09-07 | 3 | MC-doubling resource needs estimated; tail feature ranking built |
| Harvey final question closure | 06 | 2026-09-07 | 1 | Terminal wrap-up: 5th-jet causal test, FP32 writeup, QCD/ttbar stability memos |
| Track F: postproduction pipeline freeze + evaluation-readiness protocol | 07 | 2026-09-08 → 09-10 | 2 | Join/preprocess/train pipeline verified executable; evaluation protocol pre-registered before the ParT2M result existed |
| Track F: ParT active20 integration + harm diagnosis | 07 | 2026-09-11 | 1 | Governing harm result, root-cause-diagnosed (see `docs/studies/07_pretrained_jet_representations/`) |
| Track G: JP-JEPA/SPA-Net compatibility prep | 07 | 2026-09-08 → 09-09 | 1 | Self-labeled exploratory, explicitly "not a production authorization" |

Early "Harvey update" at `docs/harvey_update_2026_07_08/` (2026-07-08) is
**not** part of the Sept 2026 SPA-Net reliability track above despite the
shared addressee (Prof. Harvey Newman) — it is an early status update on the
first Delphes HH4b baseline, two months earlier; classified under study 03.

## C. Other top-level directories

| Directory | Study | Status | Notes |
|---|---|---|---|
| `Results Summaries/` (25 files) | 01 (24 files), 02 (1 file) | historical/governing | bbWW-era write-ups, 2026-05/06. **Not moved** — 5 scripts hardcode `Path("Results Summaries")` as a write target. |
| `outputs/` (audits/, baselines/, plots/, spanet_hbb_assignment/, summaries/, ttbar_classifier/; 841 files) | 01, 02, 03, 04, 05 | mixed | Main bbWW/COLLIDE/classical-ML figure archive, 2026-06 → 2026-07-16; `outputs/summaries/*.md` is a dense sequential lab notebook of model-hierarchy decisions. Not moved (unreorganized this stage, per the root README's repository-map note); curated copies of its best figures are under the relevant `docs/studies/<NN>/figures/`. |
| `results/hh4b_cut_baseline_20260724_v1/` | 04 | superseded (by `artifacts/hh4b_cut_baseline/`) | Early, deliberately incomplete (train/val-only, no physical background normalization) cut-baseline package; already has its own `README.md`/`SOURCES.md`, not moved or duplicated. |
| `data_release/hh4b_delphes_analysis_v0_2026_07_08/` | 03 | governing | Self-contained, checksummed public release snapshot; already documented by its own `README.md`. |
| `metadata/` (120 files: `delphes/`, `production_plans/`) | 03 | historical/governing | Simulation-stage production bookkeeping. **Not moved** — production-orchestration scripts hardcode `metadata/delphes`, `metadata/production_plans`. |
| `cards/`, `configs/`, `config/` | 01, 03, 04 | governing | Physics generator cards and run configuration. `config/` (singular) and `configs/` (plural) are **two coexisting naming conventions from different eras**, not unified this stage — merging them risks live script paths for a cosmetic gain; documented here rather than forced. |
| `datasets/ak4ak8_v1/` | 03 | governing | Dataset manifest with its own `README.md`. |
| `event_files/`, `options_files/`, `data/spanet_hbb/` | 01 | historical | Original bbWW-era MadGraph run configs and SPA-Net-for-Hbb metadata (HDF5 payloads gitignored). |
| `docs/paper/jhep_hh4b_ml/` | 04, 05 | governing (train-only) | A real, in-progress paper-asset export (categorized BDT, single-head SPA-Net, checksum-verified c7q–c7u inputs) — not a compilable draft. `track_b_physical_normalization/` subdirectory is the four-model physical-normalization archive (study 05). |
| `docs/plans/track_b_track_a_integration_plan_20260801_v1.md` | 05 | historical | Defines the Track A (main CMS-style analysis) / Track B (external Sophon representation study) relationship. |
| `docs/research/background_inventory_and_ml_readiness_20260720.md` | 03 | historical | Background-sample merge-compatibility rules for the project's own Delphes production (despite the filename, not about COLLIDE). |
| `docs/updates/hh4b_validation_update_2026_07_07.md`, `docs/harvey_update_2026_07_08/` | 03 | historical | Early (one day apart) progress-update pair on the first Delphes HH4b samples/BDT sanity check, addressed to Prof. Harvey Newman. |
| `docs/references/fastjet.bib`, `docs/datasets/*.md` | cross_cutting_infra, 03 | reference | Literature/dataset-registry reference material. |
| `scripts/` (429 files), `tests/` (84 files) | all | governing | Paired roughly 1:1 (`scripts/x.py` ↔ `tests/test_x.py`). Not reorganized this stage; each study README links the handful of entry-point scripts most relevant to it rather than the full tree. `scripts/analysis/hh4b_plot_style.py` and `scripts/bootstrap_resampling.py` are shared cross-study infrastructure. |
| `artifacts/hh4b_cut_baseline/` | 04 | governing (train-only) + blocked (validation) | `final_status_blocked_validation/` documents the infrastructure-failure non-result explicitly (see below); everything else here is train-only. |
| `artifacts/hh4b_spanet_part_20260911/`, `artifacts/hh4b/pretrained_jet_representations/zero20_20260914/` | 05/07, 07 | governing | Already extensively documented by their own READMEs; not duplicated here. |

## D. Scientific integrity notes

- **Two frozen "cut baseline" numbers exist and do not fully agree in scope**:
  `results/hh4b_cut_baseline_20260724_v1` (2026-07-24, train/validation-only,
  no physical background normalization, explicitly marked PENDING) and
  `artifacts/hh4b_cut_baseline/` (2026-08-10/11, physically normalized,
  train-only, validation blocked by infrastructure). These are **not the
  same result** — the July bundle is an earlier, narrower snapshot the
  August bundle supersedes for anything citing a physically-normalized
  number. Both are kept; neither was silently preferred over the other by
  deleting the earlier one.
- **The cut-baseline validation campaign never produced a physics number.**
  `HH4B_CUT_BASELINE_FINAL_STATUS.md` and
  `artifacts/hh4b_cut_baseline/final_status_blocked_validation/` both state
  this explicitly: cluster `3795859`'s 116 jobs failed at a pre-access
  runtime probe before any validation event was opened. This is reported
  everywhere in this repository as an infrastructure non-result, never as a
  negative physics finding, and no downstream document should describe the
  cut baseline as "validated" or "tested out of sample."
- **"Track B" was reused as a label at least three times** with different
  meanings across the project's life (Aug 1–6: external Sophon-model
  validation; Aug 21 onward: SPA-Net physical-normalization/Harvey
  reliability track; a July 8 "Harvey update" predates both and is
  unrelated). This map and `docs/history/SURF_TIMELINE.md` disambiguate by
  date and content rather than by the label alone.
- **ParT terminology**: the pretrained jet-representation checkpoint used
  throughout study 07 is the official JetClass-**supervised** Particle
  Transformer, not a self-supervised model (see
  `docs/analysis_contracts/SPANET_PART_RESOURCE_AWARE_COMPARISON.md.save`).
  JP-JEPA (study 07, exploratory/future work) genuinely is self-supervised —
  do not conflate the two when reading `docs/checkpoints/track_g_jpjepa_*`.
- **Full128/full144 is feasibility planning, not a completed result** — see
  `docs/studies/07_pretrained_jet_representations/full144_scale_feasibility_20260826.md`.
  It is deliberately **not** included in `docs/results/model_landscape.md`,
  per that table's rule of only listing completed results.

## E. Branches

See [`docs/history/BRANCH_GUIDE.md`](../history/BRANCH_GUIDE.md) for the
full per-branch audit (12 branches + 2 remote-only refs).
