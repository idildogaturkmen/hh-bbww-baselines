# Escalation-800 selection-stability Condor submission

The frozen escalation-800 production package was submitted exactly once after
pre-submission checkpoint `5d730256593f346a9f47914980d9bb4796cbc23a` was
pushed, fetched, and verified clean.

- Cluster: `30020809`
- Authoritative schedd: `lpcschedd5.fnal.gov`
- Procedures: `0` through `7999`
- Parsed jobs: 8,000
- Expected nested structure evaluations: 216,000
- Submission command: `condor_submit -terse` with no `-name` override

Before the single submit call, the fail-closed driver reverified every frozen
input hash, the 8,000 empty return directories, the clean local/remote Git head,
and the absence of any package association in queue or history on all advertised
CMS LPC schedds. It persisted an attempt marker at `2026-08-10T18:31:13Z`.

Condor returned zero at `2026-08-10T18:38:09Z`. The driver persisted the raw
combined output and rc, then durably wrote
`CONDOR_ACCEPTED_DO_NOT_RESUBMIT.txt` before parsing the output. Parsing found one
contiguous 8,000-procedure cluster and the wrapper-selected schedd above. The
first read-only query on that schedd observed all 8,000 procedures in the queue,
initially idle, including endpoints 0 and 7999 with the correct return paths.
This queue snapshot was diagnostic and was not used as the acceptance gate.

Cluster `30020809` must never be resubmitted. Any parser, monitoring, return, or
later bookkeeping problem must be diagnosed against this accepted cluster and
its preserved evidence. The immutable never-resubmit registry is now
`3755882`, `3768139`, `30002685`, and `30020809`.

Validation and test remain sealed, pilot results remain excluded from stability
aggregation, and the nominal deployment candidate is unchanged.
