# Selection-stability escalation archive-storage amendment

The authorized replicas 200--999 package build exhausted the user quota on
the NFS-backed `/uscms_data/d3` area after publishing 1,441 immutable payload
archives. This is a demonstrated infrastructure limitation, not a scientific
failure. The build stopped before submission, and validation and test remained
sealed.

All 1,441 completed archives were independently rehashed, copied into the
user's durable EOS area, rehashed again, and published under the same basename.
Only after target SHA-256 closure was the quota-limited duplicate removed. The
payload records were then atomically amended to the EOS paths. Archive bytes,
scientific execution HEAD, bootstrap semantics, search budget, job architecture,
resource requests, and nominal deployment candidate did not change.

The package builder and independent auditor now accept a separately bound
archive root. The remaining payloads must be built into the frozen EOS root.
Submission is still forbidden until the complete 1,600-archive package is
audited and frozen in a separate pre-submission checkpoint.
