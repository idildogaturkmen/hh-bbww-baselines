# STATUS -- v2 governing-10M FP32 tail rerun

Created: 2026-09-06T19:15:21Z (UTC, `date -u` at package creation time)

**UPDATE, same task, after completion:** Steps 1-5 below all PASSED as
recorded; the environment check that was still running when this file
was first written (last section) completed successfully -- see
`HARVEY_FP32_10M_FINAL_REPORT.md` for the resolved environment details,
the executed inference command, its exit code (0), and every
downstream result. Package status: **COMPLETE**.

## Step 1 -- governing SPA10M checkpoint located

```
/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4AI_spanet_10M_scaling_seed0_20260821_v1/full_run_attempt2_eaf_local_staging/spanet_10M_seed0/version_0/checkpoints/primary-epoch=47-step=234336-validation_average_jet_accuracy=0.4928.ckpt
```

## Step 2 -- SHA256 verified

Command: `sha256sum <checkpoint path above>`

```
fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c307102fa6943de5  primary-epoch=47-step=234336-validation_average_jet_accuracy=0.4928.ckpt
```

Matches the required governing hash
`fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c307102fa6943de5`
exactly. **PASS.**

## Step 3 -- frozen 400,000-event validation H5 located

```
/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4AF_spanet_partial_events_population_correction_20260819_v1/production_hdf5_build_v1/work/production_2M_val.h5
```

SHA256 (`sha256sum`, verified this task):

```
3c94bf8300d1dc3324e2cf25f9b6c618dffb94819cb3a297a43ac06320d5c2a2
```

This is the `HDF5_VAL` constant hard-coded in
`evaluate_classification.py` (the 2M sibling evaluation script that
`evaluate_classification_10M.py` imports and reuses verbatim) and
matches `HDF5_VAL_SHA256_EXPECTED` in that same file, and the
`hdf5_val_sha256` value already recorded in the frozen
`classification_evaluation_10M_result.json` from the v1 package's own
Task 1 trace. **PASS.**

## Step 4 -- proof this is the same validation cohort/order as the frozen evaluation

`evaluate_classification_10M.py` (read in the v1 package, path
`phase4AI_.../classification_evaluation_10M/evaluate_classification_10M.py`)
does `import evaluate_classification as base` (line 40) and never
reassigns `base.HDF5_VAL` -- it only overrides
`base.PRIMARY_CKPT`/`base.SECONDARY_CKPT` (lines 70-72). The dataset
object scored is `base.build_model_and_dataset()` -> `Options(EVENT_YAML,
HDF5_TRAIN, HDF5_VAL)` with `HDF5_VAL` fixed at import time to the exact
path/hash above, and the full-population loader
(`score_full_population`, `evaluate_classification.py:295-298`) uses
`DataLoader(model.validation_dataset, ..., shuffle=False,
drop_last=False)` -- deterministic row order, identical for any
checkpoint scored through this code path. This is therefore, by
construction (same file, same hash, same code, same `shuffle=False`
loader), the identical 400,000-event cohort **and** the identical row
order as the original frozen 10M evaluation and the v1 package's 2M
census. No new split, no resampling, no Stage-C/holdout_B file
referenced anywhere in this chain.

## Step 5 (this file)

Written immediately, before any inference command is run.

## Environment check (in progress at the time this file was first written)

Candidate pinned runtime:
```
/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4AE_spanet_10jet_literature_aligned_training_preflight_20260818_v1/gpu_canary_10jet_trainonlyweighted_v23exact_r2/.pixi/envs/default/bin/python
```
identified because `run_10M_seed0_full_50epoch_attempt2.sh` (the
governing checkpoint's own training wrapper) hard-codes exactly this
interpreter path (`PY="$R2_DIR/.pixi/envs/default/bin/python"`), and
`evaluate_classification.py`'s own `SPANET_REPO`/`SPANET_PINNED_COMMIT_EXPECTED`
constants point at the sibling `spanet_repo_v23exact` checkout in the
same directory. A first `import torch; import spanet; ...` smoke check
of this interpreter was still running past 120s at the time this file
was written (see the report for the resolved outcome -- this file is
not updated further; final status is in
`HARVEY_FP32_10M_FINAL_REPORT.md`).

No pip install/uninstall/upgrade was performed or will be performed to
resolve this. No training, no gradient update, no Stage-C/holdout_B
access, no EAF GPU allocation used for this environment check.
