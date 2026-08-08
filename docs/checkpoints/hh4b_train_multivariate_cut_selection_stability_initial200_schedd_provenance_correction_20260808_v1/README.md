# Initial-200 Condor schedd provenance correction

Cluster `30002685` was submitted exactly once. The LPC submission wrapper chose
`lpcschedd5.fnal.gov`, as preserved in the raw `condor_submit -terse` output.

The original generated submission receipt incorrectly recorded
`lpcschedd4.fnal.gov`. That frozen receipt is intentionally not rewritten.
A subsequent read-only reconciliation queried both candidate schedulers and
found all 2000 procedures on `lpcschedd5.fnal.gov` and none on
`lpcschedd4.fnal.gov`.

Therefore `lpcschedd5.fnal.gov` is authoritative for all future monitoring and
history queries for cluster `30002685`.

This correction changes bookkeeping provenance only. It does not alter the
submission, jobs, payloads, analysis, nominal selection, validation/test state,
or aggregation authorization. Cluster `30002685` must never be resubmitted.
