# HH4b global BDT apples-to-apples protocol v1

This train-only protocol freezes a global mass-aware XGBoost BDT, a global
mass-plane-blind ablation, and the unchanged strict comparator
`r_hh_125_125 < 34`. Validation and test payloads remain sealed.

The primary universe is exactly 1,042,397 physically authorized broad resolved
train rows from 441 source groups: 90,638 signal and 951,759 background rows.
Eligibility requires at least four selected AK4 jets with pT > 30 GeV and
|eta| < 2.5 and successful deterministic four-jet reconstruction. There is no
b-tag multiplicity or mass-plane requirement. The 128,675 rows from 23
classification-only auxiliary-QCD sources are excluded so classifier training,
the physical evaluation, and the historical comparator share one population.

The immutable five-fold `group_id` assignment is reused. For each outer fold,
the other four folds form four inner held-fold fits. Hyperparameters and the
fixed-efficiency score threshold are selected only from pooled inner-OOF
predictions. The outer fold is evaluated once, and five outer predictions are
pooled for the primary result. Event-random splitting is forbidden.

The mass-aware feature order contains the frozen 30 common reconstructed
kinematic/topology scalars followed by `mbb1`, `mbb2`, `delta_mbb`, and
`r_hh_125_125`. The mass-plane-blind model excludes exactly those four; it
retains `mhh`. Selected features must be finite. No row dropping, clipping, or
imputation is allowed.

Training uses positive hierarchical development weights only: equal total
signal/background weight, equal signal-mode totals, equal background-family
totals, equal source totals within strata, and equal rows within each source.
Weights are recomputed using each outer-training partition and normalized to
mean one. Signed physical weights never enter XGBoost. Evaluation uses the
authorized signed `resolved_selection_contribution_weight` at 138,000 pb^-1,
reported as a Run-2 expected-yield projection from Delphes simulation.

The estimator is XGBoost 2.1.4 `XGBClassifier` with the finite eight-candidate
registry in the JSON protocol. Candidate selection ranks pooled inner-OOF
training-weighted ROC AUC, then shallower depth, fewer trees, and candidate ID.
Early stopping is disabled because each candidate freezes its tree count.
The primary operating threshold targets weighted signal efficiency 0.585957
using development predictions only. Outer-fold information cannot select a
model, feature treatment, or threshold.

Primary reporting includes efficiencies, rejection, AUC, signed yields, sumw2,
effective MC counts, finite-MC uncertainty, S/B, and stat-only Asimov ZA.
Background Neff below 100 is flagged as finite-MC dominated. The historical
cut is never scanned and is recomputed fold by fold on the identical universe.

The later HHH-inspired categorized/specialist BDT is not implemented. Its
design remains pending recovery of the actual reference and project notes; all
physical background processes must propagate into any score-derived category.

Machine-readable authority is
`configs/baselines/hh4b_bdt_apples_to_apples_v1.json`. Full training and
`condor_submit` remain unauthorized.

The per-feature physics meaning, unit, treatment, mass-plane role, and model
membership are frozen in `docs/analysis/hh4b_bdt_apples_to_apples_v1_features.tsv`.
