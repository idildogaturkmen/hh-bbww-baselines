# HH4b generator-artifact location audit

This checkpoint verifies the availability of representative production sources and
locates provenance sidecars by filename and filesystem metadata only.

It does not open ROOT, Parquet, HepMC, or LHE event payloads. It does not download or
extract bundle archives. It does not assign cross sections or calculate physical yields.

EOS campaign directories are listed, representative bundles are checked with `xrdfs
stat`, and known local campaign submission/return directories are inventoried. Missing
artifact classes are converted into a deterministic representative-bundle indexing plan.

The next gate is to inspect the member names of one representative archive per
process-campaign, extracting only small provenance files after an explicit allowlist is
frozen.
