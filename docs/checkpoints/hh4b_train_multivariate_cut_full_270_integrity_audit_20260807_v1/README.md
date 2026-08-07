# Full 270-job multivariate-cut train-scan integrity audit

This checkpoint freezes the completed train-only nested-OOF production scan.

Cluster 3755882 completed all 270 authorized jobs: 54 category-family
configurations across five outer folds. All 270 Condor completions, worker
receipts, returned bundle hashes, internal result hashes, canonical payload
hashes, execution provenance checks, fold-isolation checks, and category-family
matrix checks passed.

All 270 jobs passed the inner and pooled statistical-support gates. Thirty-five
job-level pooled inner-OOF operating points met the frozen 0.585957 signal-
efficiency feasibility requirement and 235 did not. Feasibility is a scientific
per-configuration outcome, not an execution failure.

This checkpoint does not select a cut family or thresholds and does not authorize
use of outer-fold performance for selection. Validation and test remain sealed.
The next analysis gate is train-only aggregation using pooled inner-OOF metrics
under the frozen deterministic ranking rule.
