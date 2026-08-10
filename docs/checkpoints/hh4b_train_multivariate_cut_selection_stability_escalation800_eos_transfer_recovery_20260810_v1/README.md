# Escalation-800 EOS transfer-hold diagnosis and recovery preflight

Accepted cluster `30020809` on `lpcschedd5.fnal.gov` began holding jobs before
their executables started. Every diagnosed hold had HTCondor code/subcode
`13:2`: the schedd could not read a payload through the login-node
`/eos/uscms/...` FUSE path while preparing input transfer.

This is an infrastructure path-visibility defect, not a missing payload or a
scientific failure. At the frozen diagnosis point, all 864 held jobs had zero
starts; their 185 distinct referenced archives all existed at the frozen EOS
root and were all present with the same paths in the independently audited
1,600-entry manifest. The accepted cluster remains the only production cluster
and must never be resubmitted.

The schedd advertises curl/data transfer plugins but not an XRootD transfer
plugin, and the tested EOS WebDAV endpoints were not listening. Direct XRootD
access through `root://cmseos.fnal.gov//store/user/...` succeeded. A one-file
download-only smoke test reproduced the frozen replica-200 exact3tag archive
byte-for-byte: 14,792,039 bytes and SHA-256
`3c847483807a7e00599b45a784832acf56f0fa10acf67218c2d1dd6c3e47d135`.

The recovery wrapper added by this checkpoint changes transport only. It
downloads the already-frozen payload inside the execute sandbox, verifies the
expected archive SHA-256 supplied in the immutable job arguments, verifies the
unchanged frozen production runner SHA-256, and delegates to that runner. It
does not alter bootstrap draws, fold tables, search budgets, structures,
thresholds, scientific execution provenance, or output semantics.

After this checkpoint is committed and pushed, recovery must be validated on
held procedure `30020809.0` only by editing that existing job in place and
releasing only that procedure. No new cluster and no second `condor_submit` are
permitted. Cluster-wide in-place recovery is forbidden until the canary returns
a clean receipt, bundle, and runner log with validation/test counters still zero.
