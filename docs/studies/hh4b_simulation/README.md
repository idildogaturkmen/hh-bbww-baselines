# HH→4b simulation

## Question

How do we generate a CMS-Run-2-like, resolved HH → 4b simulated dataset — signal
plus the relevant backgrounds — with enough statistics and correct physical
bookkeeping to serve as the foundation for every classification and
representation-learning study that follows?

## Method

MadGraph5 event generation (ggF HH → 4b, with VBF and additional diboson/Z+4b
background processes) → Pythia8 parton showering and hadronization → Delphes
CMS-like detector simulation (AK4/AK8 jet reconstruction, b-tagging), produced and
scaled across many HTCondor production campaigns, with per-campaign manifests,
checksums, and validation gates recorded at each stage.

## Dataset

The full training population totals **144,291,951 events** across 1,100 source
files: 69,750,093 signal events (a 2HDM H3VAR H1↔H2 sample, 500 files), 62,941,859
QCD events (400 files), and 11,599,999 ttbar events (200 files). A governing
80/20 file-level train/validation split is defined and reusable. A smaller,
citable public snapshot of an earlier production wave is also available (see
Frozen artifact below).

## Main result

Successfully produced, validated, and fully traced a multi-hundred-million-event
resolved HH → 4b simulated dataset, with generator cross sections, banners, and
production lineage recovered and frozen for every source, and cross-layer
validation (schema, checksum, and statistical closure checks) passing at each
production wave.

## Key figures/tables

- `metadata/production_plans/hh4b_canonical_signal200k_registry_20260723.{tsv,json}`
- `metadata/delphes/frozen_v2_phase1_20260717/phase1_process_summary.csv`
- `FULL144_PART_SPANET_SCALE_FEASIBILITY_20260826.md` (repository root) — the
  authoritative accounting of the full 144M-event population.

## Code/config pointers

- `cards/mg5/*.mg5`, `cards/delphes/*.tcl` — canonical, currently-used generator
  and detector cards.
- `scripts/production/` — production preparation, submission, and validation
  scripts (Condor-orchestrated).
- `scripts/delphes/` — Delphes-output inspection, cross-section checks, and
  input-building scripts.
- `configs/production/*.yaml`, `config/production/hh4b_final_production_v1.yaml`.

## Frozen artifact

- `data_release/hh4b_delphes_analysis_v0_2026_07_08/` — a self-contained, citable
  public snapshot (its own cards, scripts snapshot, parquet event summaries, and
  `checksums.sha256`).
- `metadata/` — the full campaign-by-campaign production bookkeeping ledger
  (manifests, submission/validation receipts, per-wave review decisions).

## Provenance

- `docs/delphes/AI_NATIVE_WORKFLOW.md`, `docs/delphes/frozen_v2_phase1_20260717/` —
  simulation methodology and the first frozen production phase.
- `docs/checkpoints/hh4b_3b_*_20260725_v1/`,
  `docs/checkpoints/hh4b_4b_real_parity_canary_20260725_v1/` — 3-tag/4-tag parity
  and builder-implementation checkpoints.
- `docs/checkpoints/hh4b_ttbar7_*`, `docs/checkpoints/hh4b_ttbar8_*`,
  `docs/checkpoints/hh4b_ttbar27_canonical72_*` — ttbar sample regeneration
  canaries and closures.
- `FULL144_PART_SPANET_SCALE_FEASIBILITY_20260826.{md,json}` (repository root).
