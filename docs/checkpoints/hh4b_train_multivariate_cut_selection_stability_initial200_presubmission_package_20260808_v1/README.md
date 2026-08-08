# Initial-200 selection-stability pre-submission production package

This checkpoint freezes the complete transfer-safe package before the initial
200 selection-stability replicas are submitted.

The package contains exactly 200 bootstrap replicas, two b-tag categories, five
outer folds per category, and 27 structures per category-fold job: 2000 Condor
jobs and 54000 full-budget nested reoptimizations. The transfer layer uses 400
immutable replica-by-category payload archives, each reused by the five
outer-fold jobs for that replica/category. This changes transfer duplication
only and does not change the scientific architecture.

All 400 external payload archives were rehashed against the frozen payload
manifest at freeze time. The archives themselves remain external to Git; this
checkpoint freezes their paths, byte sizes, and SHA-256 identities through the
payload manifest.

The package is built and audited but has not been submitted. The eventual
submission must occur exactly once. If Condor submission succeeds but local
receipt/bookkeeping parsing fails, the campaign must not be resubmitted blindly.

Validation and test remain sealed. Replicas 200--999 are not production
authorized by this checkpoint. The nominal deployment candidate is unchanged.
