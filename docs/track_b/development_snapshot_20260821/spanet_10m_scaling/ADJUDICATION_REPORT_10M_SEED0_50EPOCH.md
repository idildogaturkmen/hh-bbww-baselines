# Adjudication Report: 10M seed-0 SPA-Net, full 50-epoch run (Attempt 2, EAF-local-staging)

**Status: PASS. DEVELOPMENT ONLY -- not a final/governing result.**

This report independently re-verifies the completed Attempt-2 full
50-epoch run directly from raw artifacts (exit-status file, full
stderr log, the complete result JSON, checkpoint files re-hashed from
disk, and checkpoint-internal metadata opened via `torch.load`) -- not
from the user's characterization of the run, and not from the run
script's own self-reported summary booleans alone, per explicit
instruction.

Artifact root: `full_run_attempt2_eaf_local_staging/work/`
- `training_10M_seed0_full_attempt2.exit_status`
- `training_10M_seed0_full_attempt2_result.json` (2,286,996 bytes)
- `training_10M_seed0_full_attempt2.stderr.log` (24,160 bytes)
- `training_10M_seed0_full_attempt2.stdout.log`

## Top-line facts (independently confirmed)

| Fact | Value | Source |
|---|---|---|
| Exit code | `0` | `.exit_status` |
| Start / end (UTC) | `2026-08-21T06:52:24Z` / `2026-08-21T15:42:18Z` | `.exit_status` |
| GPU | `NVIDIA A100 80GB PCIe MIG 4g.40gb` | `.exit_status` + `stderr.log` `nvidia-smi -L` + `torch.cuda.get_device_name` (3-way agreement) |
| `exit_status` field | `COMPLETED` | result JSON |
| `fit_exception` | `None` | result JSON |
| `trainer_state_status` | `TrainerStatus.FINISHED` | result JSON |
| `all_epochs_completed` | `True` | result JSON |
| `completed_epoch` / `n_epochs_in_history` | `50` / `50` | result JSON, cross-checked against `len(epoch_history)` |
| `completed_global_step` | `244100` (= 4882 steps/epoch x 50, exact) | result JSON |
| Total wall time | `31727.155170865008` s (~8.81 h) | result JSON |
| Throughput | `16848.253262387578` events/s (train-only) | result JSON |
| Peak GPU reserved | `1614807040` bytes | result JSON -- byte-identical to the 2M run and the earlier GPU benchmark, reconfirming memory is architecture/batch-shape-bound, not population-size-bound |
| Peak host RSS (self) | `13815912` KiB | result JSON |

## Requirement-by-requirement adjudication (all 12 independently re-verified)

1. **Exit code 0.** PASS -- `.exit_status` file read directly.
2. **Exactly 50 completed epochs.** PASS -- `epoch_history` has 50 entries; epoch numbers independently confirmed contiguous `0..49`; `global_step` strictly increasing throughout.
3. **Exact 10,000,000 train / 400,000 validation events.** PASS -- `n_train_events=10000000`, `n_val_events=400000` in result JSON; independently cross-checked against `stderr.log`'s own early `h5py` row-count preflight line (`train HDF5 row count (h5py, pre-dataset-construction check): 10000000`) and the SPA-Net dataset-construction log lines (`Index Range: 0...9999999`, `Index Range: 0...399999`).
4. **Exact frozen HDF5 hashes.** PASS -- `hdf5_train_sha256=2186ff12...` matches the independently re-hashed `production_10M_train.h5` build output exactly; `hdf5_val_sha256=3c94bf83...` matches the frozen 2M validation HDF5 exactly (same file, reused unchanged).
5. **Exact A100 MIG GPU class.** PASS -- see top-line table; 3-way agreement (`.exit_status`, `nvidia-smi -L`, `torch.cuda.get_device_name`).
6. **Seed 0, batch size 2048.** PASS -- both fields read directly from result JSON.
7. **Frozen scientific configuration unchanged.** PASS -- full `options_snapshot` independently diffed key-by-key against the frozen 2M run's own `options_snapshot`: `batch_size=2048`, `partial_events=True`, `assignment_loss_scale=1.0`, `classification_loss_scale=1.0`, `detection_loss_scale=0.0`, `hidden_dim=32`, `transformer_dim=128`, `initial_embedding_dim=16`, `num_encoder_layers=8`, `num_branch_encoder_layers=2`, `num_classification_layers=1`, `dropout=0.0059`, `optimizer=AdamW`, `learning_rate=0.00659`, `l2_penalty=0.000374`, `gradient_clip=0.425`, `epochs=50`, `num_gpu=1` -- zero mismatches (also machine-cross-checked again in `scaling_study_2M_vs_10M/build_scaling_comparison.py`'s `architecture_identity_check`, which independently confirms `identical: true` across 14 architecture/optimizer keys).
8. **All logged training/validation losses finite.** PASS -- independently scanned every entry of `epoch_history` (50 entries) and `callback_metrics_final`; zero non-finite values found anywhere.
9. **Classification and assignment heads both active and finite.** PASS -- spot-checked at epoch 47 (`loss/classification/EVENT/signal=0.22315305471420288`, `loss/h1/assignment_loss=0.2649572491645813`, `loss/h2/assignment_loss=0.2722893953323364`) and epoch 49 (`0.22263303399085999`, `0.28970614075660706`, `0.23104579746723175`) -- all present and finite at both epochs.
10. **No CUDA OOM / traceback / scientific data error.** PASS -- `fit_exception=None`; independent `grep -inE "out of memory|CUDA error|Traceback|Exception"` (excluding the three already-known/expected warning-text substrings) against the full 24,160-byte stderr log returned zero matches.
11. **Primary, secondary, and last checkpoints exist and reload cleanly.** PASS with a disclosed and fully explained semantic observation -- see dedicated section below. All 3 files independently confirmed present on disk via `find`, independently re-hashed via `sha256sum` and matched exactly against the result JSON's self-reported hashes. `checkpoint_reload_success=True` with zero missing/unexpected keys (this check, per the existing script design inherited unchanged from the 2M/smoke evaluation pattern, reloads the primary checkpoint).
12. **Warnings contain no genuinely new class relative to the already-adjudicated 2M run and Attempt-2 smoke.** PASS -- `n_warnings_captured=15000`, exactly 3 unique messages at 5000 occurrences each: `"Mean of empty slice."` (expected), `"invalid value encountered in scalar divide"` (expected), `"Recall is ill-defined and being set to 0.0 due to no true samples..."` (not marked expected in `EXPECTED_WARNING_SUBSTRINGS` -- but this is the same pre-existing labeling gap already documented for the 2M run and the Attempt-2 smoke, not a new warning class introduced by the 10M scale-up).

**Overall Task A verdict: PASS. No scientific-integrity problem found.**

## Checkpoint-semantic forensic investigation: `last.ckpt` == primary checkpoint

**Observation as reported:** primary (epoch 47, score `0.4928354024887085`, sha256
`fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c307102fa6943de5`),
secondary (epoch 49, score `0.4917766749858856`, sha256
`0a7f5504c0ab11434bc8712bff8ff90cf45c30b4daa073e00cf37fb5e2e24ca1`),
`last.ckpt` reported sha256 identical to primary
(`fd9ea100...`) -- despite training reaching epoch 49 (`max_epochs=50`
reached).

**Independently re-derived, not assumed correct or incorrect, per explicit instruction:**

1. **Re-hashed all 3 files directly from disk** (a third independent
   hash computation, beyond the result JSON's own self-report and the
   `checkpoint_reload_success` check): `primary` -> `fd9ea100...`,
   `secondary` -> `0a7f5504...`, `last.ckpt` -> `fd9ea100...`. Confirms
   the byte-identity claim is real, not a JSON-authoring bug.
2. **Opened each checkpoint's internal metadata directly** via
   `torch.load(path, map_location='cpu', weights_only=False)` (pinned
   pixi env, not trusting any external bookkeeping):
   - `primary`: stored `epoch=47`, `global_step=234336`.
   - `secondary`: stored `epoch=49`, `global_step=244100`.
   - `last.ckpt`: stored `epoch=47`, `global_step=234336` -- **not**
     29/49, confirming this is a literal same-save-event artifact, not
     a coincidental weight collision at a different epoch.
   Each checkpoint's embedded `callbacks` dict was also inspected: the
   primary/`last.ckpt` files carry a stale loss-monitor callback state
   (`best_model_score=0.4951`, `epoch=46`) frozen from the last time the
   primary `ModelCheckpoint` instance actually wrote a file, whereas the
   secondary file's own loss-monitor callback state is current
   (`best_model_score=0.4918`, `epoch=49`) -- internally consistent with
   the root cause below.
3. **Root-caused directly from the pinned PyTorch Lightning source**
   (`pytorch_lightning/callbacks/model_checkpoint.py`, v2.6.0, read
   line-by-line, not inferred from documentation or memory):
   - `on_train_epoch_end` / `on_validation_end` (lines ~488-519) call
     `_save_topk_checkpoint(trainer, monitor_candidates)`
     **unconditionally** every epoch, but only call
     `_save_last_checkpoint(trainer, monitor_candidates)` if
     `self._last_global_step_saved == trainer.global_step` -- i.e. only
     if a new top-k file was **actually written this epoch**.
   - `_last_global_step_saved` is set (line ~601-602, inside
     `_save_checkpoint`) only when a file is genuinely saved to disk.
   - `on_train_end` (lines 535-539) has a catch-up save gated by
     `if self.save_last and not self._last_checkpoint_saved:` --
     `_last_checkpoint_saved` initializes to `""` (falsy) and becomes a
     truthy path string the **first** time any checkpoint is ever
     saved, and **never resets** afterward -- so this catch-up path is
     also skipped once any save has occurred.
   - `last.ckpt` is owned solely by the **primary** `ModelCheckpoint(save_last=True)`
     instance; the secondary instance has `save_last=False`.
   - The primary's monitored metric (`validation_average_jet_accuracy`)
     peaked at epoch 47 (`0.4928354024887085`) and **did not improve**
     through epochs 48 or 49 -- independently recomputed directly from
     all 50 `epoch_history` entries (`max()` over the full history),
     not merely trusted from the checkpoint's own self-report, and the
     result matches the checkpoint's self-reported best score exactly.
   - Consequence: the primary's `_save_topk_checkpoint` wrote no new
     file at epochs 48/49, so `_last_global_step_saved` remained frozen
     at epoch 47's `global_step` (234336), so `_save_last_checkpoint`
     was skipped at both epochs 48 and 49, and skipped again at
     `on_train_end` (since a save had already occurred earlier in
     training). `last.ckpt` therefore still holds the exact bytes
     written at epoch 47.

**Independent re-confirmation that this does not affect the governing
checkpoint-selection outcome:** directly recomputed
`max(validation_average_jet_accuracy)` over all 50 `epoch_history`
entries = epoch 47, value `0.4928354024887085` -- exact match to the
checkpoint's self-report, confirming epoch 47 genuinely is the true
maximum, not an artifact of stale bookkeeping. Directly recomputed
`min(val/loss/total_loss)` over all 50 entries = epoch 49, value
`0.4917766749858856` -- exact match, confirming epoch 49 genuinely is
the true minimum. **Per the governing selection rule (PRIMARY =
maximum `validation_average_jet_accuracy`, never overridden by a
secondary checkpoint's lower total loss), the primary checkpoint
(epoch 47) remains scientifically correct and is unaffected by this
finding.**

**Verdict: expected PyTorch Lightning behavior under this
checkpoint-callback configuration (two independent `ModelCheckpoint`
instances, one `save_last=True` and one `save_last=False`, both
monitoring different metrics with `save_top_k=1`), not a scientific-
integrity problem, not a tooling bug introduced by this project's own
launcher code, and not a reason to change checkpoint selection.**
`last.ckpt` is, in this specific run, a redundant copy of the primary
checkpoint rather than an independent "true final state" snapshot --
this is disclosed here explicitly rather than silently treating
`last.ckpt` as if it reflected epoch 49.

## Not done / explicitly out of scope for this report

- No training was (re-)launched by this adjudication.
- No checkpoint file was modified, moved, or deleted.
- Attempt-1 and Attempt-2-smoke evidence were not re-read or re-verified this update (already frozen from prior turns).
- Classification-AUC evaluation of the primary/secondary checkpoints is a separate deliverable (Task D, `classification_evaluation_10M/`), not part of this training-run adjudication.
