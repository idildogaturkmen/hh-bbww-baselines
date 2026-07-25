# Canonical HH4b BDT-v1 grouped cross-validation

## Result

- Status: `hh4b_bdt_v1_grouped_cross_validation_pass`
- Train members and rows: 458 and 57326
- Frozen grouped folds: 5
- Hyperparameter configurations per variant: 24
- Search fold fits: 240
- Selected-configuration fold refits: 10
- Total XGBoost models trained: 250
- Missing/duplicate OOF rows and member leakage: 0/0/0

## Selected configurations

- `mass_aware`: trial 18, mean weighted fold AUC 0.777984
- `explicit_dijet_mass_plane_blind`: trial 18, mean weighted fold AUC 0.735628

Selection used exactly: highest mean held-out development-weighted ROC AUC,
highest worst-fold weighted ROC AUC, lowest fold AUC standard deviation,
lowest `n_estimators * (2 ** max_depth)`, then lowest trial ID. Validation,
test, physical significance, raw yields, and the central mass region played no
role.

## Scientific scope

The benchmark compares the frozen 34-feature mass-aware BDT with the frozen
30-feature **explicit dijet-mass-plane-blind ablation**. The latter is not
described as fully mass-decorrelated. No feature scaler was fitted, no
`r_hh_125_125 < 34` requirement or other signal-region selection was applied
to the BDT input, and every fit used the frozen hierarchical development
weights after a fold-local mean-one rescaling.

The optimized cut baseline is evaluated separately on the identical train
population. Balanced-efficiency and purity ratios are retained only as
explicitly nonphysical development proxies. No cross sections, luminosity,
generator/importance weights, physical yields, significance, or expected
limits were calculated.

Gain importance is a model diagnostic, not a causal attribution. Background
score–mass-plane diagnostics were not used in model selection.

## Data-access boundary

- Train candidate files opened: 458
- Validation candidate files opened / rows read / metrics: 0 / 0 / 0
- Test candidate files opened / rows read / metrics: 0 / 0 / 0
- Full-train models fitted: 0

The ten selected fold models and two row-level OOF prediction Parquets remain
only under the ignored runtime directory. No model, prediction, candidate
feature matrix, or serialized training dataset is in this checkpoint.

## Environment

- Python: `3.9.25`
- scikit-learn: `1.6.1`
- XGBoost: `2.1.4`
- NumPy: `1.26.4`
- SciPy: `1.13.1`
- mplhep used: `False`

All plots use the frozen nonofficial CMS-publication-inspired helper, with the
truthful “Delphes simulation” and 13 TeV annotations.

## Authorization boundary

- `grouped_cv_complete: true`
- `selected_hyperparameters_frozen: true`
- `train_oof_working_points_frozen: true`
- `final_full_train_models_fitted: false`
- `validation_evaluation_authorized: false`
- `test_access_authorized: false`
- `physical_significance_authorized: false`

## Next gate

`freeze_hh4b_bdt_v1_models_and_evaluate_validation_once`
