# Initial-200 selection-stability Condor submission

The frozen initial-200 production package was submitted exactly once.

- Cluster: 30002685
- Procedures: 0 through 1999
- Parsed jobs: 2000
- Expected nested structure evaluations: 54000
- Submission schedd: lpcschedd4.fnal.gov

The submission command returned zero and the full terse output parsed as one
contiguous 2000-procedure cluster. A durable submission-attempt marker was
written before invoking Condor, and an acceptance marker was written immediately
after Condor returned zero and before cluster parsing.

This cluster must never be resubmitted. If later auditing or bookkeeping fails,
the correct action is to inspect/recover the existing cluster and returned
artifacts, not to repeat the production submission.

Validation and test remain sealed. Replicas 200--999 are not authorized by this
checkpoint. Pilot transfer results do not enter the stability aggregation.
