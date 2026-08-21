# Phase 4AI: SPA-Net 10M-Scale Seed-0 Development Scaling Study

Additive-only. Does not modify any Phase4AF/AG/AH artifact (re-verified
this phase). The frozen 2M seed-0 result remains the current
development baseline and is not rerun.

## Status as of this phase

| item | status |
|---|---|
| 10M train HDF5 build | **COMPLETE and PASS** -- see below |
| Existing 400k val HDF5 | reused unchanged, re-verified (`sha256=3c94bf83...`) |
| One-epoch smoke test | **prepared, `AUTHORIZED_TO_RUN=True`, NOT YET EXECUTED** (no GPU in this session -- LPC has none) |
| Full 50-epoch run | **prepared, `AUTHORIZED_TO_RUN=False`** (per instruction, only to be flipped after smoke PASSes) |
| Classification evaluation | prepared, confirmed to correctly refuse (`NOT_YET_RUNNABLE`) since no checkpoint exists yet |
| Compute route | EAF A100 recommended for tonight; NRP A10 fallback audited, not pursued |

## 10M train HDF5 build -- real, executed, complete

`production_hdf5_build_10M_v1/build_hdf5_10M.py` was actually run this
phase (not merely prepared) against the existing frozen Stage-1 npz
population and `manifest_train_10M.tsv` -- **no ROOT re-read**. First
attempt was OOM-killed (confirmed via `dmesg`: anon-rss reached 10.4GB
against an ~11GB-RAM node); the script was redesigned around two
directly-verified facts (`source_sha256` constant within a file,
`source_entry_index` unique within a file) to check physical-identity
uniqueness via a memory-efficient per-file decomposition instead of a
10M-element global set, and to stream into preallocated arrays instead
of accumulating 880 per-file arrays before concatenating. Second
attempt: **`overall_pass: true`, wall time 68.97s.**

| check | result |
|---|---|
| exact quotas | signal=4,833,956, qcd=4,362,118, ttbar=803,926, total=10,000,000 -- exact match to the instruction |
| assignment-valid count | 2,520,203 |
| physical identity uniqueness | 10,000,000/10,000,000 unique -- zero duplicates (zero within-file, zero cross-file) |
| process/label invariant | 0 mismatches |
| dataset schema validation | 15/15 checks pass |
| output | `/eos/uscms/store/user/iturkmen/spanet_10M_scaling/production_hdf5_build_10M_v1/work/production_10M_train.h5`, 3,380,013,912 bytes, `sha256=2186ff1221eb7d42b26852167c29d61b91a646eaf4494314e5d55d427cd97790` (independently re-hashed this phase, matches) |
| existing val HDF5 | unchanged, re-verified `sha256=3c94bf8300d1dc3324e2cf25f9b6c618dffb94819cb3a297a43ac06320d5c2a2` |
| written to | **EOS**, not `d3` (per instruction -- `d3` is quota-constrained) |

## Frozen scientific configuration -- preserved, verified by diff

`full_run/launch_10M_seed0_full_50epoch.py` and
`smoke_test/launch_10M_seed0_smoke_one_epoch.py` were diffed directly
against the frozen 2M launcher
(`launch_2M_seed0_training.py`, `sha256=6e478f4b...`) while writing this
package: every `options.*` value, checkpoint filename/monitor/mode/
`save_top_k`/`save_last`, `SEED`/`BATCH_SIZE`,
`EXPECTED_WARNING_SUBSTRINGS`, and every `Trainer(...)` parameter are
**byte-identical** -- differing only in code comments (full-vs-2M) and,
for smoke-vs-full, only in the `MAX_EPOCHS` value (1 vs 50) and
`AUTHORIZED_TO_RUN`. No architecture or hyperparameter was changed for
the larger population or for any GPU difference. Both scripts
additionally hard-verify the GPU device name contains `"A100"` before
proceeding, per explicit instruction not to silently change EAF GPU
class.

## Execution blocker, disclosed

**No GPU training has executed.** This Claude Code session runs on
`cmslpc342.fnal.gov` (LPC), which has no GPU and no `kubectl` -- the
same limitation established and disclosed repeatedly throughout this
project. The one-epoch smoke test and the full 50-epoch run both
require an actual EAF GPU terminal (or, for the NRP fallback, cluster
access this session does not have). Exact commands:
`RUN_INSTRUCTIONS_AND_COMPARISON_PLAN.md`.

## Contents

- `production_hdf5_build_10M_v1/` -- the build script (executed) and
  its full result/provenance/logs.
- `compute_route_audit/COMPUTE_ROUTE_AUDIT.md` -- EAF vs. NRP A10 route
  comparison; EAF recommended for tonight.
- `smoke_test/`, `full_run/` -- launcher Python scripts + wrapper shell
  scripts, byte-diff-verified against the frozen 2M launcher.
- `classification_evaluation_10M/evaluate_classification_10M.py` --
  reuses the 2M evaluation's own functions (imported, not duplicated);
  confirmed to correctly report `NOT_YET_RUNNABLE` right now.
- `RUN_INSTRUCTIONS_AND_COMPARISON_PLAN.md` -- exact EAF commands plus
  a predeclared (written before any 10M result exists) comparison plan
  against the frozen 2M SPA-Net result; explicitly does **not** cite a
  Step2j BDT number without having read its actual frozen artifact.

## Explicitly not done

No 50-epoch training launched (no GPU access from this session). No
second seed. No 144M/full-statistics dataset built. No inference/test
data accessed. No Phase4AF/AG/AH artifact modified (re-verified this
phase: `phase4AF`'s `production_2M_val.h5` and `launch_2M_seed0_training.py`
hashes unchanged; `phase4AG`/`phase4AH` `SHA256SUMS` untouched).

## UPDATE 2026-08-21 (same session): smoke test attempt 1 -- FAILED at
preflight, `REVIEW_NEEDED` (not `SMOKE_PASS`)

Independently inspected, not taken on the user's own report per
explicit instruction. **`smoke_test/work/training_10M_seed0_smoke_result.json`
does not exist** -- the run never reached the point in `main()` that
writes it. `exit_status` file shows `EXIT_CODE=2`; the full 870-byte
`stderr.log` shows the GPU-identity check **passed**
(`nvidia-smi -L`/`torch.cuda.get_device_name` both independently
confirm `NVIDIA A100 80GB PCIe MIG 4g.40gb`) immediately followed by
`FATAL: train HDF5 not found:
/eos/uscms/store/user/iturkmen/spanet_10M_scaling/production_hdf5_build_10M_v1/work/production_10M_train.h5`.

**Root-caused, not merely reported**: independently re-verified from
this (LPC) session that the file genuinely exists at that exact path
with the exact matching SHA256
(`2186ff1221eb7d42b26852167c29d61b91a646eaf4494314e5d55d427cd97790`)
and a `Modify` timestamp ~26 minutes **before** the smoke attempt
started -- ruling out a build-not-finished-yet race. This is an
EAF-host-side EOS mount visibility/caching discrepancy, not a
missing/corrupted artifact and not a scientific or CUDA/GPU problem.

**Adjudication against all 10 predeclared smoke requirements**: actual
A100 GPU = PASS; exit code 0 = **FAIL** (got 2); exact 10M/400k
counts = NOT EVALUATED; finite losses = NOT EVALUATED;
assignment+classification loss keys = NOT EVALUATED; no CUDA OOM =
vacuously true, not meaningful; no schema/data error = **FAIL**;
checkpoint written = **FAIL** (none created, confirmed -- no
TensorBoard dir, no checkpoint dir); checkpoint reload-clean = NOT
EVALUATED; only-expected-warnings = NOT EVALUATED. Multiple explicit
failures -- per instruction, adjudicated **`REVIEW_NEEDED`**.

**Actions NOT taken**: no `SMOKE_PASS` evidence artifact created (see
`smoke_test/SMOKE_ATTEMPT_1_EVIDENCE.md` for the actual FAIL record
instead); `full_run/launch_10M_seed0_full_50epoch.py` **not** touched
(re-verified byte-unchanged, `sha256=991d35ca...`,
`AUTHORIZED_TO_RUN` still `False`); no EAF full-run command issued.

**Recommended next step (not performed by this session)**: from the
EAF terminal itself, confirm EOS visibility of the exact same
file/path before retrying the smoke test -- no script content needs to
change.

## UPDATE 2026-08-21 (same session): Attempt 2 (EAF-local-staging) prepared, additive

The user conclusively diagnosed Attempt 1's root cause from the EAF
host itself: `jupyter-iturkmen` has **no `/eos` mount at all**. The
10M HDF5 was never missing or corrupted -- only unreachable via the
LPC-only EOS-FUSE path. EAF has working `xrdfs`/`xrdcp` and reaches the
same file via XRootD; the user is staging a byte-identical local copy
to `/tmp/iturkmen_spanet_10M_20260821/production_10M_train.h5` and will
independently verify its SHA256 there before running anything.

**Attempt 1 preserved completely unchanged** -- all 20 pre-existing
`SHA256SUMS` entries re-verified OK both before and after this
amendment. New sibling directories
`smoke_test_attempt2_eaf_local_staging/` and
`full_run_attempt2_eaf_local_staging/` created, each derived from its
Attempt-1 counterpart with **only** `HDF5_TRAIN` changed to the local
staged path (`HDF5_VAL` unchanged), plus two disclosed, necessary
corollaries: `disk_before`/`disk_after` retargeted to the local staging
filesystem (the old `/eos/...` path would have crashed uncaught on EAF
once preflight passed), and two new explicit hard-fail pre-checks
(local-file readability, exact 10,000,000-row count via `h5py`) added
to `preflight()`, additive to the existing SHA256 check and the
unchanged SPA-Net-dataset-construction count assertion -- together
satisfying the required 5-point hard-fail checklist.

**Rigorously diff-verified, not merely asserted**: 9 separate diff
commands (per script pair) confirm **zero** scientific-semantics lines
differ -- every `options.*` value, checkpoint filename/monitor/mode,
`SEED=0`/`BATCH_SIZE=2048`/`MAX_EPOCHS` (1 for smoke, **50** for full,
confirmed **not** changed to 100), `EXPECTED_WARNING_SUBSTRINGS`, the
full `Trainer(...)` block, the A100-only hard GPU check, the
weight-loading patch, and the `DevelopmentJetReconstructionModel` class
body are all byte-identical to Attempt 1. Full itemized diff output:
`ATTEMPT2_DIFF_PROOF.md`. Both new Python scripts pass `ast.parse`;
both new wrapper shell scripts pass `bash -n`.

**Authorization state**: Attempt-2 smoke script's `AUTHORIZED_TO_RUN`
is `True` (remains authorized, per instruction). Attempt-2 full-run
script's `AUTHORIZED_TO_RUN` is `False` -- to be flipped only after the
Attempt-2 smoke test actually runs and is adjudicated PASS, mirroring
Attempt 1's own gate exactly.

**Exact EAF command for the Attempt-2 smoke test** (only step
authorized right now):
```bash
cd phase4AI_spanet_10M_scaling_seed0_20260821_v1/smoke_test_attempt2_eaf_local_staging
bash run_10M_seed0_smoke_one_epoch_attempt2.sh
```
(Prerequisite, per the user's own plan, not performed by this session:
`/tmp/iturkmen_spanet_10M_20260821/production_10M_train.h5` staged via
`xrdcp` and independently confirmed to have
`sha256=2186ff1221eb7d42b26852167c29d61b91a646eaf4494314e5d55d427cd97790`
on EAF before running the command above.)

Not done: no training launched, no GPU touched by this session, no
data transferred by this session (EAF-side staging is the user's own
action).

## UPDATE 2026-08-21 (same session): Attempt-2 smoke PASS -- full 50-epoch run authorized

Independently inspected `smoke_test_attempt2_eaf_local_staging/work/`
directly -- not taken on the user's characterization, per instruction.
`training_10M_seed0_smoke_attempt2_result.json` exists, `EXIT_CODE=0`,
`smoke_pass: true`. **All 11 predeclared requirements independently
re-verified from raw artifacts and found to strictly PASS**: actual
A100 GPU (both `nvidia-smi -L` and `torch.cuda.get_device_name`
confirm `NVIDIA A100 80GB PCIe MIG 4g.40gb`); exit code 0; exact
10,000,000/400,000 counts (Attempt-2's new early `h5py` check AND the
unchanged dataset-construction assertion both confirm); exact train
HDF5 SHA256 (`2186ff1221eb7d42b26852167c29d61b91a646eaf4494314e5d55d427cd97790`,
matches `build_10M_result.json` exactly); finite losses (directly
read from `epoch_history[0]`: `loss/total_loss=1.1471`,
`loss/h1/assignment_loss=0.4494`, `loss/h2/assignment_loss=0.4035`,
`loss/classification/EVENT/signal=0.3049`,
`validation_average_jet_accuracy=0.4762`, all finite); classification
and assignment loss keys both present and finite; no CUDA OOM
(`fit_exception: null`, independent log grep clean); no schema/data
error; checkpoint written (3 files, independently re-hashed from disk,
matched exactly); checkpoint reload-clean
(`checkpoint_reload_success: true`, zero missing/unexpected keys); only
previously understood warning classes (300 captured, exactly 3 unique
messages at 100 each -- the same numpy pair plus the same sklearn
message already root-caused for the frozen 2M run, no new warning
text). Full adjudication:
`smoke_test_attempt2_eaf_local_staging/SMOKE_ATTEMPT_2_PASS_EVIDENCE.md`.

**Full 50-epoch run authorized.** Changed **only**
`full_run_attempt2_eaf_local_staging/launch_10M_seed0_full_50epoch_attempt2.py`'s
`AUTHORIZED_TO_RUN` from `False` to `True` -- proved via a byte-level
diff against a pre-edit snapshot that this was the **only** line
changed (1-line diff). `SEED=0`, `BATCH_SIZE=2048`, `MAX_EPOCHS=50`
independently re-confirmed unchanged immediately after the edit --
**not** changed to 100. Attempt-1 evidence remains untouched.

**Exact EAF command for the full 50-epoch run:**
```bash
cd phase4AI_spanet_10M_scaling_seed0_20260821_v1/full_run_attempt2_eaf_local_staging
bash run_10M_seed0_full_50epoch_attempt2.sh
```
(Same local-staged `HDF5_TRAIN` as the passing smoke test; no new
prerequisite beyond what already staged/verified for the smoke run.)

Not launched by this session.

## UPDATE 2026-08-21 (same session): Full 50-epoch run completed on EAF -- adjudicated PASS; 2M-vs-10M scaling comparison and 400k classification evaluation performed

The Attempt-2 full 50-epoch run finished on EAF (`EXIT_CODE=0`,
`START_UTC=2026-08-21T06:52:24Z`, `END_UTC=2026-08-21T15:42:18Z`).
Independently inspected the complete run directly from raw artifacts
(exit-status file, full 24,160-byte stderr log, the complete
2,286,996-byte result JSON, all 3 checkpoint files re-hashed from disk
a third independent time, and checkpoint-internal metadata opened
directly via `torch.load`) -- not taken on the user's characterization,
per instruction. **All 12 predeclared full-run requirements
independently re-verified and found to strictly PASS**: exit code 0;
exactly 50 completed epochs (`completed_global_step=244100` = 4882 x
50 exactly); exact 10,000,000/400,000 event counts (cross-checked
against the early `h5py` preflight line and the SPA-Net
dataset-construction log lines); exact frozen HDF5 hashes
(`hdf5_train_sha256=2186ff12...` matches the build output;
`hdf5_val_sha256=3c94bf83...` matches the frozen 2M validation HDF5,
reused unchanged); exact A100 MIG GPU class (`NVIDIA A100 80GB PCIe MIG
4g.40gb`, 3-way agreement between `.exit_status`, `nvidia-smi -L`, and
`torch.cuda.get_device_name`); seed 0, batch size 2048; frozen
scientific configuration unchanged (full `options_snapshot` diffed
key-by-key against the frozen 2M run -- zero mismatches across all 17
architecture/optimizer/loss-scale keys); all logged
training/validation losses finite (scanned all 50 `epoch_history`
entries and `callback_metrics_final`, zero non-finite values); both
classification and assignment heads active and finite at epoch 47 and
epoch 49 (spot-checked directly); no CUDA OOM/traceback/data error
(`fit_exception=None`, independent `grep` of the full stderr log for
OOM/CUDA-error/Traceback/Exception signatures returned zero matches);
primary/secondary/last checkpoints exist and reload cleanly
(`checkpoint_reload_success=True`, zero missing/unexpected keys);
warnings contain no genuinely new class (`n_warnings_captured=15000`,
exactly the same 3 unique messages already documented for the 2M run
and the Attempt-2 smoke, at 5000 occurrences each).

**Checkpoint-semantic forensic investigation, per explicit
instruction ("Do NOT assume this is correct or incorrect. Open the
actual checkpoint metadata"):** `last.ckpt`'s SHA256
(`fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c307102fa6943de5`) is
byte-identical to the primary checkpoint's SHA256 (re-confirmed a
third independent time via direct `sha256sum` on disk), despite
training reaching epoch 49. Opened all 3 checkpoints' internal
metadata directly via `torch.load(..., weights_only=False)`: primary
stores `epoch=47, global_step=234336`; secondary stores `epoch=49,
global_step=244100`; `last.ckpt` stores `epoch=47, global_step=234336`
-- confirming this is a literal same-save-event artifact, not a
coincidental weight collision. Root-caused directly from the pinned
PyTorch Lightning v2.6.0 source
(`pytorch_lightning/callbacks/model_checkpoint.py`, read line-by-line):
`_save_last_checkpoint` is gated on
`self._last_global_step_saved == trainer.global_step`, which only
updates when the **primary** `ModelCheckpoint`'s own top-k logic
actually writes a new file that epoch; the primary's monitored metric
(`validation_average_jet_accuracy`) peaked at epoch 47 and did not
improve through epochs 48-49 (independently recomputed as the true
`max()` over all 50 epochs, matching the checkpoint's self-report
exactly), so no new primary file was written at epochs 48/49, so
`last.ckpt` (owned solely by the primary instance, which has
`save_last=True`; the secondary instance has `save_last=False`) was
never re-saved, and the `on_train_end` catch-up save is also skipped
once any save has occurred earlier in training
(`_last_checkpoint_saved` never resets). **Verdict: expected PyTorch
Lightning behavior under this checkpoint-callback configuration, not a
scientific-integrity problem or tooling bug, and does not affect
checkpoint selection** -- independently re-confirmed that epoch 47
genuinely is the true maximum `validation_average_jet_accuracy` over
all 50 epochs and epoch 49 genuinely is the true minimum
`val/loss/total_loss`, so the governing PRIMARY = maximum
`validation_average_jet_accuracy` rule (epoch 47) remains scientifically
correct and is unaffected by this finding. Full forensic write-up:
`full_run_attempt2_eaf_local_staging/ADJUDICATION_REPORT_10M_SEED0_50EPOCH.md`.

**Task A/B verdict: PASS.** Governing artifacts frozen/hashed (see
`SHA256SUMS` update below): primary checkpoint
(`fd9ea100...`), secondary checkpoint (`0a7f5504...`), the complete
result JSON, raw stdout/stderr logs, `last.ckpt`. Total runtime
31,727.16s (~8.81h); throughput 16,848.25 events/s (train-only); peak
GPU reserved 1,614,807,040 bytes (byte-identical to the 2M run,
reconfirming memory is architecture/batch-shape-bound, not
population-size-bound); peak host RSS 13,815,912 KiB. No earlier
artifact (Attempt-1 evidence, Attempt-2 smoke evidence, the 10M HDF5
build artifacts) was touched.

**Task C -- 2M vs. 10M training-scaling comparison** (read-only, uses
only the two already-frozen/completed result JSONs; system `python3` +
`matplotlib 3.4.3` for plotting, since the pinned pixi env lacks
`matplotlib` and plotting has no scientific dependency on the frozen
env): `scaling_study_2M_vs_10M/build_scaling_comparison.py`. All
identity cross-checks (architecture, seed, batch size, epoch budget,
validation HDF5) independently confirmed identical between the two
runs -- only training-population size differs (2,000,000 vs.
10,000,000 events). Summary:

| Metric | 2M seed-0 | 10M seed-0 |
|---|---|---|
| Primary best epoch | 49 | 47 |
| `validation_average_jet_accuracy` (primary) | 0.491227924823761 | 0.4928354024887085 |
| Secondary best `val/loss/total_loss` | 0.5106494426727295 | 0.4917766749858856 |
| Total runtime (s) | 8577.31 | 31727.16 |
| Events/s (train-only) | 15499.29 | 16848.25 |
| Peak GPU reserved (bytes) | 1,614,807,040 | 1,614,807,040 (identical) |
| Training population | 2,000,000 | 10,000,000 |

Full table + cross-checks: `scaling_study_2M_vs_10M/scaling_comparison_2M_vs_10M.json`.
Plots written to `scaling_study_2M_vs_10M/plots/`: epoch vs.
`validation_average_jet_accuracy`, epoch vs. `val/loss/total_loss`,
epoch vs. `loss/classification/EVENT/signal` (training). Per explicit
instruction, this comparison does **not** claim the 10M model is
better for event classification based on assignment accuracy or total
loss alone -- classification quality is compared separately in Task D
below.

**Task D -- exact 400k classification evaluation.** Evaluated the
frozen 10M PRIMARY checkpoint on the exact same frozen 400,000-event
development cohort already used for SPA-Net 2M, BDT-K, and BDT-KF
(`classification_evaluation_10M/evaluate_classification_10M.py`,
reusing the 2M evaluation's own functions via import, not duplicated).
Population reconstruction independently verified (`n_mismatch=0`
across `n_signal=193358, n_qcd=174485, n_ttbar=32157` -- exact match to
the BDT-K/BDT-KF study's own cohort composition); dataloader label
order independently asserted equal to the HDF5's own label order; all
probabilities finite and in `[0,1]`.

10M primary AUC: all-background `0.9692723276616946`, QCD
`0.9677556643819263`, ttbar `0.9775017953474364`. Full working-point
table (thresholds, `R_B`/`R_QCD`/`R_ttbar`, raw survivor counts at
epsS in {0.60, 0.50, 0.40, 0.25, 0.20, 0.10}) and the compact
comparison table against BDT-K, BDT-KF, and SPA-Net 2M (with the
BDT-K/BDT-KF caveats preserved verbatim -- development-population,
possible train/val leakage not controlled for, interim not
final-144M-population models, ttbar-tail comparison inconclusive at
tight working points):
`classification_evaluation_10M/COMPARISON_TABLE_400K_DEVELOPMENT.md`.
Disclosed finite-support caveat: at epsS=0.10, ttbar has zero
surviving background events out of 32,157 (rejection undefined); at
epsS=0.20, only 3 survivors -- same low-count-tail pattern already
documented for the 2M evaluation.

Secondary-checkpoint diagnostic evaluation (epoch 49) completed:
AUC all-background `0.9694592746462904`, QCD `0.9678946665362411`,
ttbar `0.9779488926480304` -- within ~1-4e-4 of the primary across all
three background definitions (statistically indistinguishable at this
precision), confirming the primary was not a cherry-picked outlier and
essentially any late-training checkpoint performs comparably on event
classification in this run. Per instruction, this diagnostic result
does **not** change checkpoint selection; the governing checkpoint
remains the primary (epoch 47, maximum
`validation_average_jet_accuracy`). Full comparison:
`classification_evaluation_10M/COMPARISON_TABLE_400K_DEVELOPMENT.md`.

All four tasks (A/B/C/D) are DEVELOPMENT ONLY, not final/governing
results, not independent Stage-C inference, and involve no physical
normalization. No 2M rerun. No 144M/full-statistics data built. No
second seed launched. No extension to 100 epochs. No independent
inference/test/Stage-C data accessed.
