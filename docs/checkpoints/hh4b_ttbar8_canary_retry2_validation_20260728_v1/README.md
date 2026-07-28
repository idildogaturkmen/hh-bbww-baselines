# HH4b ttbar retry2 canary validation

Retry2 job `3655099.0` completed successfully and reconstructed the
immutable 10,000-entry ROOT product for `ttbar_100k_shard003`.

## Canonical result

- Candidate rows: **38**
- Canonical columns: **72**
- Duplicate event keys: **0**
- Candidate SHA-256: `37fb80340351fb487b8bd142943a12c4ba73263ff6deca63d9ff8319991f5125`
- Scheduler exit code: **0**
- Returned checksum manifest: **pass**
- Portable Python environment: **exact**

## Legacy comparison

The predeclared nonsealed legacy product was used only as a validation
reference:

`/uscms_data/d3/iturkmen/hh4b_delphes/parquet/ttbar_100k_shard003_hh4b_candidates.parquet`

The legacy and regenerated canonical products are not event-aligned.
The canonical product contains 38 unique event keys, while the legacy
validation file contains 39 unique event keys. They share only 3 event
keys; 35 keys are canonical-only and 36 are legacy-only.

This proves that row-by-row value equality is not an applicable
validation criterion for these products. The legacy file is retained
only as a nonblocking schema and descriptive-distribution reference.
Shared columns are profiled independently, and finite values are
required, but no event-level equality claim is made.

The canonical-72 product reconstructed from the immutable 10,000-entry
ROOT is authoritative. The legacy 15-column product remains excluded
from the expanded canonical-v2 dataset.

## Recovery state

This canary raises the ttbar recovery state from 19 to 20 canonical-ready
members. Seven members still require exact full-chain regeneration.

No MadGraph, Pythia, or Delphes rerun occurred in retry2. No scaleout
job was submitted. No event-level product is committed in this
checkpoint. No sealed test member was opened. Physical normalization
remains out of scope.

Next gate:
`freeze_hh4b_ttbar7_exact_regeneration_scaleout_contract`
