# HH4b ttbar seven-member exact-regeneration scaleout

This checkpoint freezes the corrected full-chain scaleout contract for
the seven unresolved ttbar members.

The validated shard003 canary is excluded. The seven jobs retain their
fixed seeds and each generates exactly 10,000 events.

The corrected payload includes the canonical builder, isolated Parquet
writer, validated bootstrap, dedicated scaleout worker, canonical-72
schema, and reconstruction policy. The tested portable Python 3.9
environment is transferred separately for candidate reconstruction and
validation.

All seven return-directory trees have been precreated. The Condor
description is active but unsubmitted.

No scheduler action occurred. No sealed test member was opened.
Physical normalization remains out of scope.
