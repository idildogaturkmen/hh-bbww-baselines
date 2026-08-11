# Global BDT full-budget pilot attempt 1

Cluster `3797711` on `lpcschedd4.fnal.gov` started exactly once and failed at
Python module import because the transferred worker was located at Condor
scratch root rather than three directories below a repository root. No train,
validation, or test payload was opened and no model was fitted.

This is an infrastructure-only portable-path defect. The data universe,
features, folds, weights, hyperparameter registry, ranking, target efficiency,
threshold rule, and comparator are unchanged. Cluster 3797711 is permanently
`NEVER_RESUBMIT`. Its output is excluded from final scientific aggregation.
