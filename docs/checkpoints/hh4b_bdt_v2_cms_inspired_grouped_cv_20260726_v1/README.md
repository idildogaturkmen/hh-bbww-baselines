# CMS-inspired categorized HH4b BDT v2 grouped CV

Status: `hh4b_bdt_v2_cms_inspired_categorized_grouped_cv_pass`.

This is a train-only member-grouped cross-validation comparison, not a CMS reproduction. Category-local hierarchical weights were built from each fit's four training folds only; all metrics use the frozen global v1 development weights. No pooled ROC AUC was calculated from the two categories' uncalibrated scores. Ablation hyperparameters inherit the selected primary category configuration. Gain importance is a model diagnostic, not causal attribution. The member bootstrap is a development diagnostic, not a final confidence interval or systematic uncertainty.

Primary weighted ROC AUC: low mHH 0.753744, high mHH 0.805325. Cut-matched combined background efficiency: 0.183893.
