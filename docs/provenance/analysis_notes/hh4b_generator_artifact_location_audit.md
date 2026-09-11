# HH4b generator-artifact location audit

The physical-normalization workflow separates artifact discovery from artifact-content
review.

This gate performs metadata-only discovery:

1. verify one representative source locator for each process-campaign;
2. list each EOS campaign directory without downloading bundles;
3. inventory known local submission and return directories;
4. classify candidate provenance paths by filename;
5. freeze a one-bundle-per-campaign indexing plan for unresolved artifacts.

The gate does not open candidate Parquet files, ROOT files, HepMC files, or LHE files.
Bundle archives are not downloaded or indexed. Cross sections, generator-weight sums,
branching fractions, and physical yields remain unresolved.

A later gate may transfer one representative bundle per campaign to temporary scratch,
but only to list archive members and extract a frozen allowlist of small provenance files.
Event payload extraction remains prohibited.
