# Track B development paper-results snapshot — 2026-08-21

**DEVELOPMENT ONLY.** This snapshot freezes a publication-ready
DEVELOPMENT-stage comparison of four already-trained models —
BDT-K, BDT-KF, SPA-Net (2M events), SPA-Net (10M events) — ahead of
the final 144M-population BDT fits. It is **not a final/governing
result**, **not independent Stage-C inference**, and **not physically
normalized**. See `DEVELOPMENT_RESULTS_INTERPRETATION.md` for the full,
conservatively-worded interpretation and every caveat.

This snapshot is purely **additive and read-only**: it was built
entirely from already-frozen, hash-verified DEVELOPMENT artifacts
produced elsewhere in this project. No model was retrained, no
checkpoint was rescored, no independent inference/test data was
accessed, and no earlier artifact (Phase4AI, Step2j, or any other
prior phase) was modified. Every number in this snapshot is copied
verbatim from a source JSON field with a recorded (source file, SHA256)
pair — see `master_results/PROVENANCE_MAP.tsv` and
`SOURCE_PROVENANCE.tsv`.

## Contents

- `master_results/` — machine-readable master results.
  - `MASTER_DEVELOPMENT_RESULTS.json` / `.csv`: all four models' training
    population, feature/input definition, seed, training budget,
    checkpoint criterion, AUC (all-background/QCD/ttbar), all 6 working
    points (εS = 0.60/0.50/0.40/0.25/0.20/0.10) with thresholds,
    R_all-background/R_QCD/R_ttbar, raw survivor counts, and
    finite-support warnings, plus the 2M-vs-10M scaling comparison.
  - `PROVENANCE_MAP.tsv` — source file + SHA256 for every group of
    output fields.
- `tables/` — paper-ready tables, each in Markdown + CSV + LaTeX,
  each prominently labeled DEVELOPMENT ONLY:
  - `TABLE_MODEL_AUC_COMPARISON`
  - `TABLE_WORKING_POINTS`
  - `TABLE_SPANET_2M_10M_SCALING`
- `figures/` — paper-ready figures, each in PDF + SVG + PNG, clean
  HEP-publication aesthetic (CMS-inspired conventions; **no CMS / CMS
  Preliminary / CMS Simulation logo**, since this is not an official
  CMS result):
  - `fig_A_signal_eff_vs_all_background_rejection` — εS vs. R_B, log y,
    all 4 models.
  - `fig_B_signal_eff_vs_qcd_rejection` — εS vs. R_QCD, log y, all 4 models.
  - `fig_C_signal_eff_vs_ttbar_rejection` — εS vs. R_ttbar, log y, all 4
    models, finite-support/Poisson-limited points marked with an
    open-circle overlay + legend entry + footnote (rather than
    per-point arrow labels, to avoid visual clutter); working points
    with zero surviving background (rejection undefined) are omitted
    from the curve and counted in the footnote.
  - `fig_D_roc_curve_spanet_2M` — conventional ROC curve (vs.
    all-background/QCD/ttbar). **SPA-Net 2M only**: it is the only one
    of the four models with a frozen, exact per-event score array
    available (`val_signal_probs_400k.npy`); BDT-K/BDT-KF and SPA-Net
    10M only have summary statistics + 6-point working-point tables
    saved, not raw per-event scores, so a continuous ROC curve is not
    available for them without rescoring (not done, per instruction).
  - `fig_E1/E2/E3_epoch_vs_*` — the existing 2M-vs-10M training curves
    (`validation_average_jet_accuracy`, `val/loss/total_loss`,
    `loss/classification/EVENT/signal`).
  - **Panel F (normalized SPA-Net-10M / BDT-KF score distributions) was
    NOT produced.** Neither model has a frozen raw per-event score
    array available (10M's classification-evaluation script only saved
    summary statistics + working points; BDT-KF's matched-400k scoring
    likewise only saved a summary/working-point JSON) — per explicit
    instruction, this snapshot does not regenerate data merely to
    produce an optional plot.
- `DEVELOPMENT_RESULTS_INTERPRETATION.md` — conservative,
  publication-quality interpretation text covering: the dominant K→KF
  flavor-information gain; SPA-Net's competitive/directionally-stronger
  position vs. KF in some QCD-tail working points; the absence of a
  clear 2M→10M SPA-Net event-classification improvement; ttbar-tail
  finite-support limitations; the DEVELOPMENT/not-Stage-C/not-
  normalized status of every number here; and an explicit statement
  that no final model ranking should be claimed yet.
- `scripts/` — the three build scripts used to produce this snapshot
  (`build_master_results.py`, `build_tables.py`, `build_figures.py`),
  included for reproducibility. Each reads only already-frozen source
  artifacts (paths hardcoded at the top of each script) and system
  `python3` + `matplotlib` for plotting (the pinned SPA-Net pixi
  environment lacks `matplotlib`; plotting has no scientific dependency
  on the frozen training environment). Re-running them against
  unchanged sources reproduces this snapshot's outputs byte-for-byte
  (modulo matplotlib's own font-cache/backend nondeterminism in binary
  image metadata).

## Source artifacts (all pre-existing, read-only, unmodified by this snapshot)

- BDT-K / BDT-KF / SPA-Net-2M(transplanted) working points + AUC on the
  matched 400k cohort:
  `track_b_harvey_bdt_working_points_20260818_v1/step2j_interim_10m_convergence_20260820_v1/matched_400k/results/K_KF_SPANET_MATCHED_400K_COMPARISON.json`
  (cohort npz sha256 `9588d0fe79e32d455467c719eaaa57541fcbead9835b8c32bebb9d68b96a2d30`)
  and `.../HARVEY_SUMMARY_step2j_interim_10m_convergence_and_matched_400k_comparison.md`
  (BDT training-arm metadata: n_train, round budget, early-stopping behavior).
- SPA-Net 2M native classification evaluation (cross-check + raw score
  arrays for the ROC figure):
  `track_b_phase4_preflight_20260812/phase4AF_spanet_partial_events_population_correction_20260819_v1/spanet_2M_seed0_classification_evaluation_v1/work/{classification_evaluation_result.json, val_signal_probs_400k.npy, val_process_labels_400k.npy}`
- SPA-Net 10M primary + secondary classification evaluation:
  `track_b_phase4_preflight_20260812/phase4AI_spanet_10M_scaling_seed0_20260821_v1/classification_evaluation_10M/work/{classification_evaluation_10M_result.json, classification_evaluation_10M_secondary_diagnostic_result.json}`
- SPA-Net 2M-vs-10M training-scaling comparison + epoch histories:
  `track_b_phase4_preflight_20260812/phase4AI_spanet_10M_scaling_seed0_20260821_v1/scaling_study_2M_vs_10M/scaling_comparison_2M_vs_10M.json`,
  `.../full_run_attempt2_eaf_local_staging/work/training_10M_seed0_full_attempt2_result.json`,
  `phase4AF_.../spanet_2M_seed0_production_training_v1/work/training_2M_seed0_result.json`.
- Task D evaluation-boundary freeze (BDT-side scope discipline, quoted
  verbatim in the interpretation doc):
  `track_b_harvey_bdt_working_points_20260818_v1/step2k_full144m_final_fit_preflight_20260821_v1/results/TASK_D_EVALUATION_BOUNDARY_FREEZE.md`.

**No Phase4AI, Step2j, or any earlier artifact was modified to produce
this snapshot.**

## Not done

No training. No rescoring. No independent inference/test-data access.
No physical normalization. No 144M-population BDT fit started or
touched. No CMS/CMS-Preliminary/CMS-Simulation branding used anywhere
(this is not an official CMS result).

## Freeze

See `SHA256SUMS` (full recursive listing, `sha256sum -c`-verifiable
from this directory) and `SOURCE_PROVENANCE.tsv` for the claim/source/
verification-method record of this snapshot's construction.

**Final status: TRACK_B_DEVELOPMENT_PAPER_SNAPSHOT_FROZEN**
