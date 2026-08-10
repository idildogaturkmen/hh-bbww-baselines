# Escalation-800 EOS/XRootD recovery canary

This checkpoint freezes the complete-return audit of existing accepted job
`30020809.0` after an in-place, transport-only recovery.

## Frozen outcome

- The original job was edited in place and released; no `condor_submit` call
  occurred and cluster `30020809` must never be resubmitted.
- The recovery wrapper downloaded the already-frozen replica-200 exact-3-tag
  payload through XRootD, verified its SHA-256, and delegated to the unchanged
  frozen scientific runner.
- The authoritative schedd records exactly one clean history completion for
  proc 0 with exit code zero and no signal exit.
- The exact return set is `job_receipt.json`, `result_bundle.tar.gz`, and
  `runner.log`.
- The independent audit passed all 27 structure results, the exact 82-file
  bundle inventory, internal checksums, canonical payload identities,
  execution provenance, and four-inner-fold coverage.
- All 27 pooled-support checks pass. Zero of 27 structures are feasible for
  this bootstrap/category/fold; infeasibility is a valid scientific result and
  was not used as an infrastructure gate.
- Validation payloads opened: **0**. Test payloads opened: **0**.

## Frozen boundary

The scientific execution head remains
`275a2aaebe83b2f1f5e7403a245b52daf30560b4`; only the data-transport path
changed. Pilot results remain excluded, the nominal selection is unchanged,
and no escalation aggregation or fold-level selection has occurred.

The next authorized action is a single fail-closed in-place transport recovery
of existing cluster `30020809` procs 1--7999, followed by read-only
monitoring. It must not submit or create another cluster.
