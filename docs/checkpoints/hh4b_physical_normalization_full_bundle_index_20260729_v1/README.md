# HH4b full representative production-bundle index

All 53 EOS process-campaign representative bundles now have byte-size and SHA-256 provenance and complete archive-header inventories. Six validated canaries were reused and the remaining 47 bundles were copied sequentially to temporary non-workspace scratch, indexed, and deleted.

No archive member was extracted. ROOT, Parquet, HepMC, and LHE contents were not opened. The resulting draft allowlist contains only small, regular, safe-path, non-payload, non-archive members with provenance-like names. Extraction remains unauthorized.

The legacy local ttbar campaign remains a blocking provenance exception. The next gate freezes the PN-b3 extraction allowlist and separately recovers the legacy ttbar generation artifacts.
