# HH4b ttbar8 exact-regeneration canary failure diagnosis

Status: `hh4b_ttbar8_exact_regeneration_canary_failure_diagnosed`

Primary classification: `canonical_reconstruction_failure`

Job `3654710.0` remains held on `lpcschedd4.fnal.gov`. Its payload exited normally with
code 1. Condor subsequently held it with transfer code/subcode 12/2 because the
canonical Parquet, success receipt, and checksum manifest were absent.

The earliest causal failure is:

`ERROR: missing isolated Parquet writer: /srv/payload/repo/scripts/delphes/write_parquet_from_pickle.py`

The immutable payload contains the frozen reconstruction builder but omits the
builder's required adjacent `write_parquet_from_pickle.py`. MadGraph produced a
10,000-event LHE, Pythia wrote 10,000 HepMC events, and Delphes exited normally.
The retained ROOT file has 10,000 entries by a diagnosis-only serial uproot
metadata check. Production ROOT validation, success-receipt creation, and
checksum-manifest creation were not reached after canonical reconstruction
failed.

The wrapper, expected-output registry, `--out` argument, transfer-output list,
and transfer remap agree on the exact Parquet path. No different or temporary
Parquet exists in the authorized canary return directory. This is not an output
path mismatch, cleanup failure, or transfer-configuration root cause.

The minimal correction is technical and physics-invariant: package and manifest
the already-required isolated writer beside the frozen builder, preflight that
dependency closure, and add fail-fast/output-existence checks. The member, seed,
event count, cards, versions, Delphes card, reconstruction features, thresholds,
and sample identity remain unchanged.

`corrected_retry_authorized` is false. No retry or scaleout job was submitted,
and no scheduler mutation was performed. The next gate is `freeze_hh4b_ttbar8_exact_regeneration_canary_retry_contract`.
