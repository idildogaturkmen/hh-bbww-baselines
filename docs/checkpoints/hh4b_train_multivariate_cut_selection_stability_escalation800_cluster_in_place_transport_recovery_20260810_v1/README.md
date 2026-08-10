# Escalation-800 cluster in-place transport recovery

This checkpoint freezes the exactly-once transport recovery of accepted
cluster `30020809` procs 1--7999 after the proc-0 XRootD canary passed.

## Frozen action

- No `condor_submit` call occurred. No new cluster was created.
- All 7,999 targets were already held with the common pre-execution EOS-FUSE
  transfer error `13:2`; the controlled-hold phase was therefore an audited
  no-op.
- The fail-closed driver verified exact proc coverage, zero scientific
  executable starts, original scientific arguments and working directories,
  original per-job transfer inputs, empty return directories, the sole clean
  proc-0 history record, and clean/pushed canary checkpoint HEAD
  `75898d4bd05158ddedd95fbc15bff53b54a35d72`.
- One `condor_qedit` invocation changed exactly two transport attributes on
  the existing procs: `Cmd` to the reviewed XRootD recovery wrapper and
  `TransferInput` to the unchanged frozen runner plus runtime.
- One `condor_release` invocation released only those edited held procs.
- Immediately after release, all 7,999 targets were idle with the edited
  transport attributes and unchanged scientific arguments/IWDs.

The full pre/hold/edit/release job-ad snapshots remain in the durable external
evidence root and are content-addressed by
`external_evidence_inventory.json`.

## Scientific boundary

The scientific execution head remains
`275a2aaebe83b2f1f5e7403a245b52daf30560b4`. The nominal selection is
unchanged, pilot results remain excluded, and validation/test remain sealed.
Cluster `30020809` must never be resubmitted.

The next step is read-only monitoring of this existing cluster, followed by
the complete 8,000-job/216,000-result return audit before any all-1000
aggregation.
