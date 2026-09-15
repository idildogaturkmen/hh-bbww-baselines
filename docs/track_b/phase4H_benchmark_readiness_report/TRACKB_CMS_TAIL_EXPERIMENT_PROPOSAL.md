# Track B CMS-Tail Pairwise-Transformer Experiment Proposal (DESIGN ONLY -- NOT EXECUTED)

Status: proposal only. `HIGH_STAT_TRANSFER_EXECUTION_AUTHORIZED=false`. Nothing in
this document has been run. No model has been trained.

## Scientific question

Can a genuinely pairwise-aware Transformer operating on Track B's high-statistics
HH->4b benchmark approach the CMS-like classification regime referenced by:

- eps_S = 0.40 -> R_B ~ 100
- eps_S = 0.10 -> R_B ~ 1000

**These are reference values for comparison only.** They are not hyperparameter
optimization targets. See "Anti-target-chasing rule" below.

## Signal

- Primary: `gghh_kl1` (SM ggF HH->4b, kl=1), 2,538,851 stored events, an
  independent Track-B sample never touched by training_qcd file selection.
- Optional secondary (only if frozen before looking at any held split): an
  inclusive-HH definition combining `gghh_kl1` with one or more VBF samples
  (`vbf_cv1_c2v1_kl1` etc.), with the combination weights and the definition of
  "inclusive HH" written down and frozen in an addendum BEFORE any held-split
  data is examined. Absent that predeclaration, VBF is out of scope for this
  first experiment.

## Background

- `training_qcd` only, for this first experiment. No `inference_qcd_1`/`_2` and
  no `training_ttbar`.
- Deterministic file-disjoint partition from Phase-4F:
  - `development` (240 files, 37,759,585 events) for all fitting, cross-
    validation, hyperparameter selection, and diagnostic iteration.
  - `holdout_A` (80 files, 12,595,807 events) for a one-time internal
    method/family adjudication after development choices are frozen.
  - `holdout_B` (80 files, 12,586,467 events) sealed; no tuning or
    feature-family decisions after opening.

## Primary population

`TRACKB_CMS_INSPIRED_TAG4_SR_ANALOG` if reconstructible at the scale needed for
the chosen working points (it is constructible per Phase-4G; a full-sample
mass-plane scan across all three splits, not yet performed, would be needed
before launch to confirm SR/CR support at holdout scale -- see "Pre-launch
checklist" below). If the full-sample scan shows insufficient SR support at a
given eps_B working point, fall back to `tight_exact4` (no mass-plane cut),
with the absence of the mass cut explicitly documented in every plot and table
that uses the fallback.

## Model

- A genuine pairwise-aware Transformer: pairwise kinematic features (e.g.
  delta_R_ij, m_ij, k_T-style distances computed from the stored constituent
  4-vectors `part_px/py/pz/energy`) must enter the attention mechanism itself
  (e.g. as attention bias terms, as in ParT's original pair-embedding design),
  not be concatenated post-hoc as a per-token feature.
- Do **not** reuse the shallow Track-A `FiveJetParTNet` and relabel it "ParT".
  If a Track-A architecture is used as a starting point, the pairwise-attention
  modification must be implemented and ablated against the un-modified
  baseline before this counts as "genuinely pairwise-aware" for the purposes
  of this experiment.
- Architecture (layer count, embedding dims, attention heads, pairwise feature
  list) must be frozen in a version-controlled config file BEFORE `holdout_A`
  or `holdout_B` is opened. Changes after opening `holdout_A` require starting
  a fresh, newly-named experiment; they may not be applied retroactively to
  this one.

## Inputs (from `TRACKB_MODEL_INPUT_READINESS.tsv`)

All inputs are `READY_NOW`: `part_px/py/pz/energy`, `part_charge`, `part_pid`,
`part_d0val/d0err/dzval/dzerr`, `part_label` (AK4-jet membership),
`jet_pt/eta/phi/mass/energy`, `jet_sophonAK4_probB/probC/probL` (as a
predeclared ablation only -- see continuous-tagging rule below).

## Continuous-tagging anti-leakage rule (carried over from the frozen Phase-4G
protocol, `phase4G_transfer_and_roc_protocol_freeze`)

`jet_sophonAK4_probB/probC/probL` may be included as an explicit, predeclared
feature ablation, but the stored `pass_4j3b_selection`/`pass_4j2b_selection`
flags and any hard tag-count derived from them are **forbidden as model
inputs** for the HH-vs-QCD classification task, because the tag categories
themselves are threshold functions of the same continuous score. A model given
the hard flags could trivially reconstruct category membership rather than
learning genuine HH-vs-QCD discrimination.

## Metrics

Report at each of these signal efficiency working points:
`eps_S in {0.60, 0.50, 0.40, 0.25, 0.20, 0.10}`
and at each of these background efficiency working points:
`eps_B in {1e-2, 1e-3, 1e-4}`

- eps_B (background efficiency) at each eps_S
- R_B = 1/eps_B (background rejection) at each eps_S
- eps_S at each fixed eps_B
- ROC AUC (secondary metric only)
- raw surviving QCD event count at each working point (not weighted -- see
  "No physical yields" below)
- binomial (Clopper-Pearson) or bootstrap confidence intervals on every
  eps_B/eps_S estimate, since the tight_exact4/SR tail has raw counts as low
  as O(10-100) in the bounded canary
- score-tail support: number of raw background events surviving in the top
  1%, 0.1%, 0.01% of the classifier score, to diagnose whether high-rejection
  claims are supported by enough raw statistics to be meaningful

## No physical yields

**No physical (450 fb^-1 or 138 fb^-1) yield, significance, or cross-section
claim is authorized while `TRAINING_QCD_WEIGHT_CONTRACT_RESOLVED=false`.** All
metrics above are raw-count / efficiency-ratio metrics. This is compatible with
`RAW_COUNT_ML_BENCHMARK_READY=true` from the readiness report; it does not
require the weight contract to be closed.

## Anti-target-chasing rule

The CMS reference values (eps_S=0.40 -> R_B~100; eps_S=0.10 -> R_B~1000) exist
to give the reader a familiar comparison point. They must **not** be used to:
- select which architecture variant is reported,
- select which feature ablation is reported,
- stop training early because a checkpoint happens to cross a reference line,
- or retroactively adjust the mass-plane window, tag working point, or
  background sample after seeing how close a result lands to the reference.

Any result, whether it beats, matches, or falls short of the reference values,
is reported as-is with its statistical uncertainty. The reference values are a
comparison axis on the ROC plot, not a stopping criterion.

## Pre-launch checklist (must all be true before
`HIGH_STAT_TRANSFER_EXECUTION_AUTHORIZED` may be requested)

1. Full-sample (not bounded-canary) mass-plane scan of `training_qcd` to
   confirm SR/CR raw support at `development`/`holdout_A`/`holdout_B` scale,
   individually per split.
2. Equivalent mass-plane scan of `gghh_kl1` signal to confirm SR efficiency is
   non-degenerate (i.e. the analog SR is not accidentally signal-empty).
3. Architecture and feature-ablation list frozen in a versioned config file.
4. Written confirmation that `holdout_B` has not been examined in any form
   (no summary statistics, no plots) prior to the sealed final test.
5. This proposal document reviewed and explicitly re-authorized (this audit
   does not self-authorize execution).
