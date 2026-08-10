# Escalation-800 selection-stability pre-submission production package

This checkpoint freezes the complete, independently audited production package
for authorized bootstrap replicas 200--999 before any scheduler submission.
The package contains exactly 800 replicas, two b-tag categories, five outer
folds per category, and 27 structures per category-fold job: 8,000 Condor jobs
and 216,000 full-budget nested reoptimizations. Its transfer layer contains
1,600 immutable replica-by-category archives, each reused by the five outer-fold
jobs for that replica/category.

The failed manual resume was forensically classified as CASE 1: 1,441 valid EOS
archives and matching payload records existed, while 159 payloads were absent.
The implementation had staged completed archives on NFS and attempted to
publish them to EOS with a cross-filesystem `os.replace()`. Commit
`d11cca4909fb4fcf1052cc9ec357b748efc0a616` made the one-line implementation-only
repair that stages each partial beside its final EOS archive. A cross-filesystem
smoke build and independent audit passed, and its replica-200 archive hashes
matched the existing production bytes exactly. Scientific execution provenance
remains `275a2aaebe83b2f1f5e7403a245b52daf30560b4`.

The repaired production build rehashed and reused the 1,441 existing immutable
archives, constructed only the 159 missing payloads, and finalized the package.
The full independent auditor then verified every outer archive SHA-256 and every
internal `SHA256SUMS` member, as well as provenance, fold/category mappings,
five-fold reuse, resource requests, the 8,000-row job matrix, empty return
directories, and all sealed counters.

All 1,600 archives remain external to Git at the frozen EOS archive root. This
checkpoint commits only compact deterministic evidence and submission inputs;
`payload_manifest.json` is the authoritative archive manifest binding every
path, byte size, and SHA-256 identity. The 80 legacy NFS `.partial` files from
the failed attempt remain preserved as forensic evidence and are neither
published archives nor members of the manifest.

The package has not been submitted. Validation and test remain sealed, pilot
results remain excluded from stability aggregation, and the nominal deployment
candidate is unchanged. The eventual submission must occur exactly once. If
Condor accepts it but receipt parsing or local bookkeeping fails, this package
must never be resubmitted blindly.
