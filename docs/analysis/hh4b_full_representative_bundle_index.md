# HH4b full representative production-bundle index

This gate extends the successful six-bundle archive-header canary to all remaining EOS
process-campaign combinations.

The operation is resumable. Each completed campaign writes a small progress fragment to
the `_incomplete` checkpoint. A network interruption therefore does not require already
completed bundles to be transferred again.

Every bundle is copied sequentially to non-workspace temporary scratch, checked against
its frozen byte size, hashed with SHA-256, indexed through TAR member headers, and deleted.
No archive member is extracted. ROOT, Parquet, HepMC, and LHE contents are not opened.

The final checkpoint combines the six canary campaigns with the 47 scale-out campaigns,
giving complete archive-header coverage for all 53 EOS campaigns. Small safe regular
non-payload files are recorded in a draft PN-b3 extraction allowlist, but extraction and
content review remain unauthorized.

The local legacy ttbar campaign is retained as an explicit blocking exception rather than
being silently normalized with the EOS ttbar campaigns.
