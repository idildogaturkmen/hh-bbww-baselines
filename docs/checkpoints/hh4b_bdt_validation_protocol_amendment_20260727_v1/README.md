# HH→4b BDT validation model-selection protocol amendment

Status: `hh4b_bdt_validation_model_selection_protocol_frozen`.

This checkpoint preserves
`docs/checkpoints/hh4b_bdt_model_choice_20260727_v1/` unchanged and supersedes only its validation-selection
policy. The global v1 mass-aware BDT remains the canonical global train-only
benchmark, and the categorized CMS-inspired mass-aware BDT remains the
categorized benchmark. Neither is selected as the final nominal BDT here.

Validation is frozen as the `model_selection_and_generalization_split`. In the
next gate, the two BDT candidates and the global mass-plane-blind diagnostic
are fit on the complete frozen train population without validation input.
Validation is then opened exactly once to compare the two candidates, select
the final nominal BDT under the predeclared five-condition rule, check the
optimized $R_{HH}<34$ cut, and evaluate the diagnostic. No validation-driven
feature, category, hyperparameter, weight, background-family, or reconstruction
change is authorized, and validation may not be reopened after inspection.

Both fixed train-OOF threshold generalization and equalized validation
signal-efficiency comparisons are required at targets 0.30, 0.50,
0.585957314769, and 0.70. The primary metric is combined
development-weighted validation background efficiency at target
0.585957314769; lower is better. Equalized validation thresholds are selection
diagnostics and do not replace the frozen train-OOF thresholds.

The source-member bootstrap uses seed 20260727 and 2000 replicas for
$\Delta\epsilon_B=\epsilon_B(\mathrm{categorized\ v2})
-\epsilon_B(\mathrm{global\ v1})$. Categorized v2 is selected only if its
point estimate is lower, its relative reduction is at least 0.02, at least
0.84 of replicas favor it, ggF and VBF efficiencies are each within 0.05 of
the target, and no integrity, category, or application failure occurs.
Otherwise global v1 is selected.

Test is frozen as the `final_unbiased_evaluation_split`. It remains inaccessible
during validation selection. A later gate may evaluate test exactly once after
validation selects the nominal model, but test results may never choose or
change that model.

Both BDT approaches, their train-only and validation comparisons, and the
category-specific ROC and feature-importance plots remain part of the
presentation. The final validation-selected nominal BDT must be labeled
explicitly, while the non-selected BDT remains a secondary benchmark.

All weights are development balancing, not physical normalization. No expected
yield, luminosity, significance, or expected limit is part of model selection.

This amendment opened zero validation and test candidate files, trained zero
models, wrote zero predictions, and ran zero hyperparameter trials.

Next gate: `fit_frozen_hh4b_bdt_candidates_and_select_on_validation_once`.
