# Escalation-800 cross-filesystem publication repair

This checkpoint diagnoses the failed package-resume attempt after the EOS
storage amendment and freezes the smallest implementation-only repair.

The forensic state is CASE 1: exactly 1,441 valid EOS archives and matching
payload records remain. All 800 materialization summaries exist, but 159
payloads are not published. The failed resume left 80 uncommitted NFS
`.partial` archives and no package manifest, build receipt, job matrix, queue,
submit file, package audit, return, log, or submission evidence.

The builder staged archives below the NFS package root and then called
`os.replace()` to publish them at the EOS archive root. The roots are on
different devices, so that atomic rename cannot succeed. The observed one
partial per missing replica/category-first build and zero newly published EOS
archives are exactly consistent with failure at that publication boundary.
No raw traceback from the manual command was preserved, so the errno itself is
recorded as an evidence-backed inference rather than a quoted error.

The repair changes one line: the temporary archive is now created beside its
final archive. The archive construction inputs and algorithm are unchanged.
A separate cross-filesystem replica-200 smoke build and full independent audit
passed. Both produced archive hashes are identical to the existing production
replica-200 hashes. The repair therefore changes no scientific semantics or
immutable payload bytes.

The 80 failed NFS partials remain untouched as diagnostic evidence. Production
submission remains forbidden until all 1,600 archives are present, the full
auditor passes, and a separate pre-submission checkpoint is pushed.
