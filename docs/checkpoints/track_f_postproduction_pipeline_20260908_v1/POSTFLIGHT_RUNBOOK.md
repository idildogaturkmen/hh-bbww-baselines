# Postflight runbook — exact_2M ParT production

**Gate status as of this package's build time (2026-09-08T23:59Z):
NOT PASSED.** Latest known census (2026-09-08T23:18:46Z, from
`track_b_part_producer_v3_ihep_auth_fix_20260908_v1/receipt.json`):
**train 747/880, val 7/220.** The prepared resume (`bundle_v3`,
`run_worker_chunks_v3.sh`, a 346-shard 2-worker VAL-first partition) is
canary-verified but had **not been launched by anyone** as of this
package's build time. Do not trust these numbers tomorrow -- re-run Step 1
below fresh before anything else.

## What this gate checks (fail-closed on all four)

1. **Fresh EOS census** (`census_2m.py`, reused unchanged): every shard ID
   in `[0,880)` train / `[0,220)` val has both a `.h5` and matching
   `.receipt.json`, zero H5-only, zero receipt-only, zero duplicates/
   collisions, zero IDs outside the expected range, checkpoint SHA256 exact.
2. **Aggregate disjointness/uniqueness proof** (`aggregate_embedding_audit.py`,
   reused unchanged): the manifest's per-split `native_hdf5_row` windows
   are contiguous and non-overlapping across shards (O(n_shards), no global
   identity set), and no shard has an internal `(row, jet_slot)` duplicate.
3. **Manifest declared-count cross-check** (`cross_check_manifest_counts.py`,
   new): `SHARD_MANIFEST.json`'s own SHA256 and its declared shard/event
   counts per split still match this package's `FROZEN_FACTS.json`
   expectation (880/2,000,000 train, 220/400,000 val) -- catches a
   regenerated manifest with different windows/quotas.
4. **Checksum verification** (`verify_shard_checksums.py`, new): re-hashes
   shard `.h5` content against the receipt's own `output_sha256` (never
   trusts the receipt alone), for a sample by default, `--full` before an
   actual training freeze.

## Paste-ready command

```bash
PKG=/uscms_data/d3/iturkmen/hh4b_delphes/track_f_postproduction_pipeline_20260908_v1

# Routine check (fast: census + aggregate audit + manifest cross-check +
# a 20-shard checksum spot-check per split):
bash "$PKG/scripts/postflight/run_postflight_gate.sh"

# Full pre-freeze run (slow: every shard's content re-hashed, ~1100 xrdcp
# calls -- run this once, immediately before TRAIN_2M_PART_RUNBOOK.md's
# hash-freeze step, not routinely):
CHECKSUM_SAMPLE_N=full bash "$PKG/scripts/postflight/run_postflight_gate.sh"
```

Output lands in a fresh timestamped `logs/postflight_<UTC>/` directory
(or pass an explicit directory as the first argument) containing
`train_listing.txt`, `val_listing.txt`, `census.json`, `aggregate_audit.json`,
`manifest_cross_check.json`, `checksum_train.json`, `checksum_val.json`, and
one combined `GATE_RESULT.json` / `GATE_RESULT.txt`.

**Exit code 0 and `OVERALL_POSTFLIGHT_PASS=true` in `GATE_RESULT.json` is
the ONLY valid signal to proceed to `JOIN_AND_PREPROCESS_RUNBOOK.md`.**
Anything else — including a partial pass on 3 of 4 steps — means STOP.

## What this gate is read-only with respect to

`xrdfs ls` (train/val listings), `aggregate_embedding_audit.py`'s own
read-only per-shard `EMBEDDINGS`/`JOIN_KEY` reads, and `xrdcp` downloads to
a local temp file that is deleted immediately after each hash (checksum
step). **Nothing under the production's `shards/` directory is ever
written to, moved, or deleted by any script in this package.**

## Syntax/dry-run status (this session)

- `bash -n scripts/postflight/run_postflight_gate.sh` — PASS
- `python3 -m py_compile` on both new Python scripts — PASS
- The gate was **not** run against real EOS this session (would require
  live `xrdfs`/Kerberos/X.509 access this session does not hold, and the
  task explicitly says do not interfere with the live production).
