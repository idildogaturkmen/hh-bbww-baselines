# Machine Learning for Resolved Higgs-Pair Reconstruction in HH → bbbb

This repository contains the complete SURF research program behind a study of
resolved di-Higgs production decaying to four b-quarks (HH → bbbb, "HH→4b"),
from the original H→bbWW baseline work it evolved out of, through Delphes
event simulation, classical and deep-learning baselines, and a
symmetry-preserving reconstruction network (SPA-Net) augmented with frozen
Particle Transformer (ParT) representations.

If you are new to this repository, read this file top to bottom before
descending into `docs/checkpoints/` — that directory is a 167-entry,
hash-stamped lab notebook (dated evidence for individual decisions), not an
onboarding guide. This README, `docs/project_overview/PROJECT_STORY.md`, and
`docs/results/RESULTS_OVERVIEW.md` are the onboarding path.

## Physics motivation

Higgs-pair (HH) production directly probes the Higgs self-coupling and is one
of the highest-priority searches for the HL-LHC era. This project targets the
**resolved** (non-boosted, four separate AK4 jets) HH → 4b final state in
Delphes-simulated, CMS Run 2-like events (AK4 jets, b-tagging, standard
QCD/ttbar/diboson backgrounds), comparing a physically-normalized cut-based
baseline against progressively more expressive learned models, up to a
symmetry-aware jet-assignment network (SPA-Net) and a study of whether
pretrained self-supervised jet representations (ParT) help it.

## Project evolution

1. **HH → bbWW baseline** (2026-05 → 06): the project's original channel —
   b-jet regression, Hbb jet-pairing, lepton+MET WW-side reconstruction, and
   an early DNN/LBN architecture comparison.
2. **Pivot to resolved HH → 4b** (2026-07-01): rationale in
   `docs/project_overview/physics_goal_and_pivot.md`.
3. **Simulation pipeline + classical baselines**: MG5→Pythia8→Delphes event
   production; a cut-based baseline; BDT, dense-DNN, and LBN comparisons.
4. **Large-statistics dataset + physical normalization**: background/signal
   production campaigns and cross-section/luminosity/generator-weight
   provenance work converting raw simulated yields to Run-2-equivalent
   physical yields.
5. **SPA-Net symmetry-preserving reconstruction and classification.**
6. **2M → 10M scaling study**: does more native training data help?
7. **QCD-tail / numerical-reliability studies**: causal and statistical
   robustness checks on the frozen governing checkpoint's rare-event tail.
8. **Frozen ParT representation study and harm diagnosis**: does appending
   pretrained ParT features help SPA-Net, and if not, why not?
9. **ZERO20**: a completed, controlled width-ablation diagnostic that
   isolates the cause of the ParT harm result — see below; input width
   alone is ruled out, the specific embedding content/preprocessing is
   implicated.
10. **JP-JEPA / representation-learning direction**: exploratory
    compatibility groundwork only — **future work**, not a current result.

Full narrative detail, with citations to the underlying frozen evidence, is
in `docs/project_overview/PROJECT_STORY.md`.

## Main contributions

- A reproducible MG5→Pythia8→Delphes HH→4b simulation pipeline with recorded
  generation lineage and physical-normalization provenance.
- A physically-normalized cut-based baseline, frozen and honestly reported
  including where its validation campaign was blocked by infrastructure
  failure rather than by a physics result.
- A BDT baseline with a stability-tested model choice (a simpler global model
  preferred over a categorized alternative whose apparent gain did not survive
  bootstrap resampling).
- A SPA-Net symmetry-preserving reconstruction/classification model, evaluated
  at two training scales (2M, 10M events) and subjected to causal and
  statistical reliability checks on its rare-event tail.
- A controlled, root-cause-diagnosed study of whether frozen pretrained ParT
  representations help SPA-Net (they significantly hurt, at this design
  point), completed by a width-control ablation (ZERO20) that falsifies
  "input width alone" as the cause and implicates the specific embedding
  content and/or its preprocessing/optimization interaction instead. See
  `docs/studies/pretrained_jet_representations/`.

## Headline results

See `docs/results/RESULTS_OVERVIEW.md` for the full set with citations to
frozen source artifacts. Top line:

| Result | Finding |
|---|---|
| SPA-Net native 2M vs. 10M scaling | Saturated: AUC 0.969281 (2M) vs. 0.969272 (10M) |
| Frozen ParT "active20" augmentation | Harm: ΔAUC −0.0251, Δ(exact-event HH reconstruction) −0.371, both diagnosed |
| HH→4b cut-based baseline | Train-only complete; optimized cut ≈ historical simple cut; validation **blocked** by infrastructure failure (no physics result) |
| HH→4b BDT baseline | Primary global model AUC (train-only OOF) 0.7737; categorized alternative's apparent gain not bootstrap-stable |
| ZERO20 width-control ablation | **Native-like**: ΔAUC vs. native −0.00017 (~30× below the pre-registered 0.005 practical floor), reconstruction statistically indistinguishable (McNemar p=0.52) — falsifies "input width alone" as the ParT-harm cause |

## Repository map

This reflects the tree as it exists **after this reorganization stage**.
Several directories that were candidates for relocation were intentionally
left in place — see "Reproducibility / provenance policy" below.

| What | Where |
|---|---|
| Project narrative, pivot rationale | `docs/project_overview/` |
| Stable methodology (simulation, normalization, ML baselines, paper prep) | `docs/methods/` |
| Result summary with citations to frozen evidence | `docs/results/RESULTS_OVERVIEW.md` |
| Deep-dive narrative for a specific study line (currently: pretrained jet representations — ParT active20, ZERO20) | `docs/studies/` |
| Dated provenance notes, research log, background-campaign records (moved this stage) | `docs/provenance/` |
| Full dated checkpoint ledger (167+ hash-stamped decision points; **not moved this stage**, see below) | `docs/checkpoints/` |
| Simulation-stage production bookkeeping (**not moved this stage**) | `metadata/` |
| bbWW-era result write-ups (**not moved this stage**) | `Results Summaries/` |
| Paper components (figures, tables, captions — not a compilable draft) | `docs/paper/jhep_hh4b_ml/` |
| All pipeline code (simulation, production orchestration, training, evaluation — unreorganized this stage) | `scripts/` |
| Run/experiment configuration (two coexisting conventions, not yet merged) | `configs/`, `config/` |
| Physics generator cards (MadGraph5, Delphes) | `cards/` |
| Unit tests (paired 1:1 with `scripts/` entry points) | `tests/` |
| Frozen, citable, lightweight result bundles (tables/plots/hashes, no raw data) | `artifacts/` |
| Forward convention for the next generation of SPA-Net/ParT bundles | `artifacts/hh4b/README.md` |
| Self-contained public data-release snapshot | `data_release/` |

Depth-2 tree of the current top level is available via `git ls-tree` or
`find . -maxdepth 2` if you want the literal current state rather than this
table.

## Reproducibility / provenance policy

Large binary artifacts — training datasets (HDF5), full model checkpoints,
event-level NPZ exports — are **deliberately not committed** to this
repository. They live in an external working area
(`/uscms_data/d3/iturkmen/hh4b_delphes/`, plus remote XRootD/EOS storage for
staged ROOT/HEPMC/LHE files) and are regenerable from the committed configs
and scripts. What *is* committed for every frozen result: a plain-language
`README.md`, lightweight machine-readable summaries (`tables/*.tsv`,
`metrics/*.json`, `plots/*.svg`), and a `SHA256SUMS` file recording the
hashes of the external large files the result depended on.

**This reorganization stage is documentation/provenance-only.** No script,
config, or hash-stamped bundle content was modified — only whole documentation
directories were relocated with `git mv` (history-preserving), each verified
beforehand to have no script-level path coupling.

Three directories that a prior audit proposed relocating were **deliberately
left in place** after a fresh dependency check, run immediately before each
proposed move, found live consumers the audit had missed:

- **`docs/checkpoints/`** — 35+ scripts, tests, and config files hardcode this
  path as a read or write default (e.g.
  `scripts/analysis/build_hh4b_cut_baseline_final_report.py`,
  `configs/baselines/hh4b_bdt_model_choice_v1.yaml`).
- **`metadata/`** — several production-orchestration scripts hardcode
  `repo / "metadata"` / `metadata/delphes` / `metadata/production_plans`
  (e.g. `scripts/production/plan_qcd_adaptive_checkpoint.py`,
  `scripts/production/run_waveb_priority_lhe_smoke.py`).
- **`Results Summaries/`** — five bbWW-era scripts hardcode
  `Path("Results Summaries")` as a write target (e.g.
  `scripts/train_pre_spa_pair_baselines.py`).

Moving any of these requires editing the consuming scripts in the same commit
as the move — deferred to a future stage, in line with this stage's
"documentation only, no live code paths touched" mandate.

## Current status

- The HH→4b cut baseline and BDT baseline are frozen (train-only).
- SPA-Net native-scaling and frozen-ParT-active20 results are frozen and
  matched (development/validation, not final blind test) as of 2026-09-11.
- **ZERO20 is complete and frozen** (final, 10,000-replicate matched
  evaluation, 2026-09-14): native-like on both classification and
  reconstruction, falsifying "input width alone" as the ParT-harm cause;
  see `docs/results/RESULTS_OVERVIEW.md` §8 and
  `docs/studies/pretrained_jet_representations/`.
- JP-JEPA compatibility prep is exploratory future work.
- This branch (`repo-reorg/2026-09`) is Stage 1 of a repository
  reorganization: documentation/provenance moves and publication-facing docs
  only. It has not been merged, and `scripts/`, `config/`/`configs/`, and the
  three directories named above have not yet been touched.

## Citation / project links

- GitHub: `idildogaturkmen/hh-bbww-baselines` (repository name predates the
  HH→4b pivot; a rename is a separate, coordinated exercise tracked outside
  this reorganization stage, given five active worktrees share one remote).
- Cite a specific result by its frozen artifact path and Git commit, e.g.
  `artifacts/hh4b_spanet_part_20260911/` at commit `73516e9`.
- For the narrative outline this project's paper will follow, see
  `docs/project_overview/PROJECT_STORY.md`.
