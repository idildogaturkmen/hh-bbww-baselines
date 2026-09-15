# Project story: bbWW → resolved HH→4b → SPA-Net/ParT

A concise scientific chronology of this repository's research program, intended
as the factual outline for writing the SURF paper. Each stage names the
evidence it rests on so claims can be traced back to a committed artifact.
For a more compact, table-form version of this same chronology, see
[`docs/history/SURF_TIMELINE.md`](../history/SURF_TIMELINE.md). For the
full per-topic writeup (question, method, what worked and what didn't,
figures, code, provenance) behind each stage below, see the corresponding
entry under [`docs/studies/`](../studies/).

## 1. Original baseline: HH → bbWW (2026-05 → 2026-06)

The project began with resolved **HH → bbWW** (one Higgs to b-quarks, one to
WW), including b-jet energy regression, Hbb jet-pairing (an early SPA-Net
application), and lepton+MET-based WW-side reconstruction. A comparison of a
tabular multiclass DNN against a family of Lorentz Boost Network (LBN)
variants (object-only, object+pair, object+auxiliary, full) was run against a
corrected-recoMET BDT reference; see
`docs/methods/ml_baselines/multiclass_dnn_comparison.md`. The best LBN variant
(obj+aux, no explicit pair features) reached a comparable but not clearly
superior significance to the BDT reference, motivating a broader
architecture/representation search rather than declaring an early winner.
Results write-ups from this era are collected under `Results Summaries/`
(kept in place this stage; a future pass may relocate it, see the migration
notes below). Full writeup, including the early SPA-Net attempt's negative
result: [`docs/studies/01_hh_bbww/`](../studies/01_hh_bbww/README.md).

## 1b. COLLIDE dataset studies (2026-05 → 2026-06, in parallel with §1)

The bbWW-era work above drew its background samples from **COLLIDE-1M**, an
external public simulated-event cache, not from this project's own
production. In parallel with the bbWW modeling work, the project inventoried
COLLIDE-1M's contents (156 parquet files, ~208GB, across 12 process groups)
and built cross-section/normalization metadata for it, then trained a
simple gradient-boosted classifier on it (ttbar-vs-HH AUC=0.712) as an
early separability check. This dataset was **not** carried forward into the
HH→4b-era work — once the project pivoted (§2) and built its own Delphes
production (§3), COLLIDE-1M was no longer used. Full writeup:
[`docs/studies/02_collide_dataset_studies/`](../studies/02_collide_dataset_studies/README.md).

## 2. The pivot to resolved HH → 4b (2026-07-01)

On 2026-07-01 the project pivoted to resolved **HH → 4b** (di-Higgs to four
b-quarks). The rationale — the bbWW channel's harder reconstruction problem
(lepton+MET ambiguity, WW combinatorics) versus HH→4b's cleaner, fully
hadronic final state that is more directly comparable to CMS's own
resolved-HH4b search — is recorded in
`docs/project_overview/physics_goal_and_pivot.md`. All subsequent work
described below is HH→4b work.

## 3. Simulation pipeline and classical ML baselines (2026-07 → 2026-08)

An MG5→Pythia8→Delphes event-production pipeline was built and iterated on
(`docs/methods/simulation/`, `cards/`, `scripts/delphes/`), including a
frozen v2 phase-1 generation-lineage record
(`docs/methods/simulation/frozen_v2_phase1_20260717/`). On top of this,
a cut-based baseline was optimized and frozen
(`HH4B_CUT_BASELINE_FINAL_STATUS.md`): the train-only optimized cut showed no
material significance improvement over the historical simple cut
(`R_HH(125,125) < 34`), and the planned validation campaign was blocked by an
infrastructure failure before any validation event was opened — a scientific
non-result, not a negative result, and reported as such.

In parallel, a BDT ladder was frozen
(`docs/checkpoints/hh4b_bdt_model_choice_20260727_v1/`): the primary
`global_v1_mass_aware` model reached a train-only OOF AUC of 0.774, with a
CMS-inspired categorized alternative showing a nominal-point improvement that
did not survive a source-member bootstrap stability check. Dense-DNN and
LBN-style comparisons for HH→4b follow the same experimental contract
established for bbWW but are less mature/frozen at the time of this
reorganization — see `docs/results/RESULTS_OVERVIEW.md` for exactly what is
and is not frozen.

## 4. Large-statistics dataset development and physical normalization (2026-07 → 2026-08)

Background campaigns (QCD/ttbar, `docs/provenance/background_campaigns/`) and
simulation-stage production bookkeeping
(`metadata/`, kept in place this stage — see migration notes) built up the
large-statistics HH→4b dataset. A dense sequence of 41+ dated checkpoints
under the `hh4b_physical_normalization_*` prefix (`docs/checkpoints/`)
establishes cross-section, luminosity, and generator-weight provenance so
that later results can be expressed as physical (Run 2, 138 fb⁻¹-equivalent)
yields rather than raw simulated counts.

## 5. SPA-Net symmetry-preserving reconstruction and classification (2026-08 → 2026-09)

SPA-Net (a symmetry-aware jet-assignment network) was adopted both as an
event classifier and as an HH-reconstruction (jet-pairing) model, following
the ablation ladder cut → BDT → categorized BDT → dense DNN → LBN-DNN →
SPA-Net (`docs/provenance/analysis_notes/hh4b_model_comparison_and_reporting_contract.md`).

## 6. 2M → 10M scaling study (2026-08-21 → 2026-09-11)

Training SPA-Net on the native 7-feature representation at 2M events versus
5x more data (10M events) was tested twice: an initial development-scaling
pass on 2026-08-21 (`docs/track_b/development_snapshot_20260821/spanet_10m_scaling/`)
and a refined, matched comparison frozen on 2026-09-11
(`artifacts/hh4b_spanet_part_20260911/README.md`). The refined result shows
the model has **saturated**, not improved, with 5x more training data
(all-background AUC 0.969281 at 2M vs. 0.969272 at 10M).

## 7. QCD-tail and numerical-reliability studies (2026-09-06 → 2026-09-07)

A series of "Harvey" review-response packages (`docs/checkpoints/track_b_harvey_*`)
subjected the frozen SPA-Net governing checkpoint to targeted reliability
checks: a genuine causal counterfactual test of whether extra (5th+) jets
perturb classification (they do — every affected event's score changes under
masking), an empirical (not projected) check of QCD tail-rate stability
across the two existing production lanes (stable at u>3.5, too few events to
judge at u>4.5), and a kinematic characterization of float32 quantization
spikes in the score distribution (no special kinematic mode found). These
establish the numerical and statistical reliability bounds of the governing
checkpoint's tail behavior.

## 8. Frozen ParT representation study and harm diagnosis (2026-09-11)

Appending 20 TRAIN-only, variance-selected, frozen ParT embedding dimensions
to the native SPA-Net input ("active20") significantly **degraded** both
classification (all-background AUC −0.025) and HH-reconstruction (exact-event
rate −0.371) relative to native SPA-Net at the same 2M scale. A read-only
root-cause audit
(`artifacts/hh4b_spanet_part_20260911/diagnosis/ROOT_CAUSE_DIAGNOSIS.md`)
ruled out implementation bugs and narrowed the cause to changed optimization
dynamics from the wider input embedding, compounded by a variance-only
feature-selection rule that discarded several more discriminative embedding
dimensions than it kept.

## 9. ZERO20 — a controlled width ablation, completed and diagnostic (2026-09-14)

The diagnosis in step 8 proposed a direct ablation: replace the 20 ParT
dimensions with 20 **zeroed** dimensions, isolating "wider input embedding"
from "ParT features specifically" as the cause of the harm
(`artifacts/hh4b_spanet_part_20260911/diagnosis/FOLLOWUP_DESIGNS.md`). This
ablation is now complete, matched-evaluated (final, 10,000-replicate paired
bootstrap), and frozen:
`artifacts/hh4b/pretrained_jet_representations/zero20_20260914/`.

The result: ZERO20 is statistically and practically indistinguishable from
native SPA-Net on both classification (ΔAUC −0.00017, ~30× below the
project's own pre-registered 0.005-AUC practical-effect floor) and
HH-reconstruction (McNemar p=0.52, not significant), while the original
ParT active20 result remains significantly and substantially degraded on
both. **This falsifies the "wider input layer alone" explanation** —
widening SPA-Net's input embedding from 7 to 27 channels does not, by
itself, reproduce the harm. The cause is tied to the specific selected
ParT values and/or their preprocessing/optimization interaction, not to
input width. Full narrative and statistics:
`docs/studies/07_pretrained_jet_representations/zero20_width_control.md`;
ranked, updated follow-up proposals (none authorized):
`docs/studies/07_pretrained_jet_representations/next_experiments.md`.

## 10. JP-JEPA / representation-learning direction — future work

A compatibility and numerical-validation prep track for JP-JEPA (a joint
predictive embedding architecture) exists under
`docs/checkpoints/track_g_jpjepa_spanet_compatibility_prep_20260908_v1/` and
the most recent commit on this branch ("Add JP-JEPA CPU batch invariance
artifact"). This is exploratory compatibility groundwork, not a result, and
should be framed in the paper as **future work**, not as part of the
headline result set.

---

## A note on this document's scope

This chronology reflects Stage 1 of a broader repository reorganization
(`repo-reorg/2026-09`). Several directories referenced above by their
current paths — notably `docs/checkpoints/`, `metadata/`, and
`Results Summaries/` — were proposed for relocation into `docs/provenance/`,
but a fresh dependency check run during this stage found live script/config
consumers of those exact paths that the original audit had not caught (see
the root `README.md`, "Reproducibility / provenance policy" section, for the
specifics). Those three moves were therefore skipped this stage and the
directories remain at their current top-level locations pending a follow-up
stage that also updates the consuming scripts in the same commit. This
document should be updated once that follow-up lands.
