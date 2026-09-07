# STATUS -- v3 exact-Harvey-threshold audit

Created: 2026-09-06T21:32:22Z (UTC)

Purely additive, post-processing-only package. Does not modify, delete,
or re-derive anything in
`../track_b_harvey_fp32_score_precision_20260906_v2_governing10m/`.

## Source data identity (verified before any computation)

Sole event source:
```
../track_b_harvey_fp32_score_precision_20260906_v2_governing10m/SPA10M_EVENT_LOGITS_400K.parquet
```

SHA256 (`sha256sum`, this task):
```
bb282dac7717d8e5145e5c2f842beb9e1b6a6ac2d64c95ab04aa2a40ad0d5412
```

Matches the entry already recorded in that package's own `SHA256SUMS`
exactly -- confirms the source parquet is byte-identical to the one
produced by v2's authorized governing-10M inference run, and has not
been touched by this task.

## What this task does NOT do (verified, not merely asserted)

- No model inference (`torch`, `spanet`, `pytorch_lightning` are not
  imported anywhere in this package's scripts -- verified via an
  AST-based import check, not a naive substring grep, see
  `work/adversarial_self_check.json` and `receipt.json`).
- No checkpoint loaded (no `.ckpt` path referenced anywhere in this
  package's scripts).
- No HDF5 accessed (no `h5py` import, no `.h5` path referenced anywhere
  in this package's scripts).
- No training, no gradient computation.
- No EAF/GPU use.
- No Stage-C/holdout_B reference.
- No ParT code, checkpoint, or embedding referenced.
- No package installed, uninstalled, or upgraded.
- The existing v2 package is read-only input here; nothing in it was
  modified or deleted (verified: v2's own `SHA256SUMS` re-checked
  unchanged at the end of this task -- see `receipt.json`).

All computation in this package is pure `numpy`/`pyarrow`/`matplotlib`
post-processing of the 11 columns already present in the v2 parquet
(`row_index`, `process_label`, `z_background`, `z_signal`, `delta`,
`score_float32_production`, `score_float32_hex_be`, `k_lattice`,
`u_prob32`, `u_logit`, `sigmoid_delta_float64`).

Final status and all numeric answers: see
`HARVEY_EXACT_THRESHOLD_FP32_SUMMARY.md` and `receipt.json`.

## Post-completion finalization pass (this update)

A late file-writing bug was found and fixed: `task3_tail_census()`
wrote its CSV/JSON outputs to fixed filenames regardless of which
thresholds were passed to it, so the supplementary 0.9999/0.99999 call
silently overwrote `SPA10M_EXACT_THRESHOLD_TAIL_CENSUS.csv` and
`work/task3_tail_census.json` with the wrong (supplementary-threshold)
numbers. **No computed numerical result was ever wrong** -- the
in-memory dictionaries used for this report and `receipt.json` were
bound to distinct variables and stayed correct throughout; only those
two on-disk files were transiently wrong. Fixed by parameterizing the
output filenames; the analysis script was rerun end-to-end and every
number reproduced identically. Full account:
`receipt.json`'s `internal_provenance_note_csv_writing_bug` and
`HARVEY_EXACT_THRESHOLD_FP32_SUMMARY.md` Task 6, item 8.

Re-verified as part of this finalization pass:
- `SPA10M_EXACT_THRESHOLD_TAIL_CENSUS.csv` (>0.9997, >0.99997) and
  `SPA10M_SUPPLEMENTARY_0P9999_0P99999.csv` (>0.9999, >0.99999) are
  now confirmed **distinct files** (different SHA256) holding their
  own correct, intended threshold rows.
- `scripts/adversarial_self_check.py` rerun once more against the
  current, corrected files: all 6 checks **PASS**.
- `SHA256SUMS` regenerated LAST, after every other file was final, and
  immediately verified with `sha256sum -c SHA256SUMS`: every entry
  **PASS**.

## FINAL STATUS: COMPLETE

**READY_FOR_HARVEY = YES**

All 6 original tasks finished; the post-completion finalization pass
above re-confirmed every headline number and corrected the one
file-writing defect found. `receipt.json` and `SHA256SUMS` are final.
The v2 package this task read from is confirmed byte-identical to
before (its `SPA10M_EVENT_LOGITS_400K.parquet` SHA256 was re-hashed and
matches its own `SHA256SUMS` entry exactly) -- nothing outside this new
directory was modified.

**No git operation was run by this task** (no `git add`/`commit`/`push`),
per explicit instruction -- this package is left complete on disk for a
separate, later archival commit pass.
