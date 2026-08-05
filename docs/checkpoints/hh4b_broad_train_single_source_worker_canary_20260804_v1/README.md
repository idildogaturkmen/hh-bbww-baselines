# Single-source broad-train worker canary

This checkpoint freezes the resumable one-source worker used to materialize
full train sources into deterministic Parquet outputs.

The worker was exercised on five complete train sources covering:

- ordinary signed exact-event sidecar transport;
- hard-QCD variable frozen-fragment transport;
- nested ggF signal archive extraction;
- local direct ROOT access;
- auxiliary-QCD exclusion from physical transport.

Each canary source reached exact full-source event closure. Re-running the
worker recognized and validated the existing output rather than rewriting it.
Physical luminosity weights were not applied. Validation and test remained
sealed.

The next gate is to create the HTCondor submission bundle and release all 464
train-source jobs.
