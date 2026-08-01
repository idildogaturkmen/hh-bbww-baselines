# Train exactly-3b ROOT registry and production manifest

This checkpoint freezes the exact 441-source train population required for
the full exactly-three-b control reconstruction.

Source closure:

- 419 sources from the July 25 ROOT source map;
- 22 authoritative legacy ttbar ROOT sources from PN-c5p/PN-c5t;
- 441 unique transport IDs in total;
- 24 overlapping qcd_bbbb ROOT-map rows remain excluded;
- 15 ordinary ROOT-map rows not selected by the frozen transport remain
  excluded.

Candidate identity contract:

- candidate content identity is exact SHA-256 plus exact row count;
- the current transport candidate path is the operational locator;
- historical canonical EOS, local cache, and repaired-overlay locators are
  retained as provenance aliases when they identify the same immutable bytes;
- any candidate checksum or row-count disagreement fails closed;
- `candidate_identity_alias_audit.tsv` records all 441 path relations.

Access contract:

- 419 remote bundle-backed source members, each bound to its exact
  frozen evidence class from `source_access_audit.tsv`: bundle SHA-256 for
  modern archives, or EOS Adler-32 plus successful remote-stat byte size for
  legacy ggF archives, together with archive-member identity;
- 22 local direct ttbar ROOT files, bound by frozen ROOT SHA-256 and byte
  size;
- one source per job;
- temporary archives and extracted ROOT files must be deleted after output
  validation;
- full outputs are assigned to EOS at `/store/user/iturkmen/hh4b_delphes/run2_13tev/frozen_v2/threeb_control/train_full_20260731_v1`;
- no persistent full-production output on d3 is authorized.

This step does not transfer or deserialize any source payload, submit any
job, reconstruct any event, calculate any physical weight or yield, or open
validation/test data.

A deterministic five-source canary is authorized. Full 441-source
reconstruction remains unauthorized until that canary passes.
