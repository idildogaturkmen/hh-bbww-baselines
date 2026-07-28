# HH4b ttbar8 canary reconstruction retry1 contract

Status: `hh4b_ttbar8_exact_regeneration_canary_retry_contract_frozen`.

This checkpoint freezes exactly one reconstruction-only retry for `ttbar_100k_shard003`
using only the immutable ROOT returned by held job `3654710.0`. The ROOT SHA256
is `127c87510087d164038b7629ea9d5210008cc4cb1473aa9b069a5cd4b859629e` and its `Delphes` tree has 10,000
entries. The corrected payload contains the previously omitted isolated
Parquet writer at the required `/srv/payload/repo/scripts/delphes/` path.

The payload was created, fully listed, extracted, checksum-verified,
syntax/import checked, and exercised with non-physics synthetic data under the
frozen LCG 106 environment. Both Condor dry-run and dump contain exactly one
inert job (`requirements=False`, `hold=True`). No retry or scaleout job was
submitted, and source job `3654710.0` remains held.

The later submission gate must archive and remove the old held job exactly once
before submitting this single retry. `condor_release` is forbidden.
