#!/usr/bin/env python3
"""PREPARED, NOT AUTHORIZED TO RUN.

SPA-Net+frozen-ParT training launcher, for EITHER the "active" (resource-
aware, train-frozen active-dim subset) or "all128" (sensitivity/control,
all 128 ParT dims) variant, selected by --variant on the command line.

This is a byte-level adaptation of the REAL, already-executed native
launcher (launch_2M_seed0_training.py, re-read in full this session --
see configs/FROZEN_FACTS.json:reuse_not_duplicate.native_launcher_to_adapt),
not a from-scratch rewrite. Every setting SPANET_PART_COMPARISON_CONTRACT_v2.md
requires to "remain identical" is reproduced UNCHANGED below (seed, batch
size, epoch budget, architecture, optimizer, loss scales, precision,
checkpoint monitor/mode, partial_events). The two intentional differences
from the native launcher are:
  1. HDF5_TRAIN/HDF5_VAL point at the augmented (7+K-feature) H5 built by
     build_augmented_input.py, not the native 7-feature H5.
  2. EVENT_YAML points at the matching augmented event YAML (also written
     by build_augmented_input.py), not trackb_hh4b.yaml.
Nothing about seed, architecture, optimizer, loss weighting, batch size,
epoch count, or checkpoint policy is touched.

AUTHORIZED_TO_RUN = False, hard-blocked further by a checksum/gate
preflight below. Both must independently agree before this can ever
execute. A human must edit AUTHORIZED_TO_RUN to True after reviewing the
preflight output for a REAL run -- this script will not flip it itself.

Usage (once every gate passes -- NOT run in this session):
    python3 launch_spa2m_part.py --variant active \
        --hdf5-train /path/spa2m_part_active_train.h5 \
        --hdf5-val   /path/spa2m_part_active_val.h5 \
        --event-yaml /path/part_augmented_active_hh4b.yaml \
        --run-name spanet_2M_part_active_seed0
"""
import argparse
import contextlib
import hashlib
import json
import os
import re
import resource
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import warnings
from collections import OrderedDict

AUTHORIZED_TO_RUN = False  # deliberate refusal; a human flips this after every preflight line below passes on real data

TRACK_B_ROOT = "/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812"
R2_CANARY_DIR = os.path.join(
    TRACK_B_ROOT, "phase4AE_spanet_10jet_literature_aligned_training_preflight_20260818_v1",
    "gpu_canary_10jet_trainonlyweighted_v23exact_r2",
)
SPANET_REPO = os.path.join(R2_CANARY_DIR, "spanet_repo_v23exact")
SPANET_PINNED_COMMIT_EXPECTED = "debbdc999bfb785eb110a36c5fd3eff211ebf234"

PART_CHECKPOINT_SHA256_EXPECTED = "61e752f80d7c237d4b18b97705df416a8518dd9e3d5a78a8bbdeebadd787fec0"
NATIVE_CONTROL_CHECKPOINT_SHA256 = "dc39cf76f0d8e40d07228f240fe58b1179e8134cd4f96c5c786a7d528d31ea8d"

SEED = 0
BATCH_SIZE = 2048
MAX_EPOCHS = 50

# ---- everything below copied UNCHANGED from launch_2M_seed0_training.py's
# frozen options -- see that file for the byte-for-byte source. ----
ARCHITECTURE = dict(hidden_dim=32, num_encoder_layers=8, num_branch_encoder_layers=2,
                     num_classification_layers=1, dropout=0.0059)
OPTIMIZER = dict(optimizer="AdamW", learning_rate=0.00659, l2_penalty=0.000374, gradient_clip=0.425)
LOSS_SCALES = dict(assignment_loss_scale=1.0, classification_loss_scale=1.0,
                    detection_loss_scale=0.0, kl_loss_scale=0.0, regression_loss_scale=0.0, balance_losses=False)
BALANCE_FLAGS = dict(balance_particles=False, balance_jets=False, balance_classifications=False)
PARTIAL_EVENTS = True
DATASET_LIMIT = 1.0
NUM_DATALOADER_WORKERS = 4
NUM_GPU = 1

EXPECTED_WARNING_SUBSTRINGS = ("Mean of empty slice", "invalid value encountered", "divide by zero encountered")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_nvidia_smi(cmd_args):
    try:
        return subprocess.check_output(["nvidia-smi"] + cmd_args, text=True, stderr=subprocess.STDOUT)
    except Exception as e:  # noqa: BLE001
        return f"<nvidia-smi error: {e}>"


def parse_epoch_from_ckpt_filename(path):
    if not path:
        return None
    m = re.search(r"epoch=(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


# ---- Same weight-loading patch as the native launcher (needed because
# this HDF5 also carries WEIGHT-bearing TARGETS, copied verbatim from the
# native H5 by build_augmented_input.py). Byte-for-byte unchanged. ----
def _patched_load_assignments(self, hdf5_file, limit_index):
    from spanet.dataset.types import SpecialKey
    import torch
    targets = OrderedDict()
    for event_particle, daughter_particles in self.event_info.product_particles.items():
        target_data = torch.empty(len(daughter_particles), self.num_events, dtype=torch.int64)
        for index, daughter in enumerate(daughter_particles):
            dataset = self.dataset(hdf5_file, [SpecialKey.Targets, event_particle], daughter)
            dataset.read_direct(target_data[index].numpy())
        for index, source in enumerate(daughter_particles.sources):
            if source >= 0:
                target_data[index] += self.source_offsets[source] * (target_data[index] >= 0)
        target_data = target_data.transpose(0, 1)
        try:
            target_mask = self.dataset(hdf5_file, [SpecialKey.Targets, event_particle], SpecialKey.Mask)
            target_mask = torch.from_numpy(target_mask[:]).bool()
        except KeyError:
            target_mask = (target_data >= 0).all(1)
        try:
            target_weight = self.dataset(hdf5_file, [SpecialKey.Targets, event_particle], SpecialKey.Weight)
            target_weight = torch.from_numpy(target_weight[:])
        except KeyError:
            print("Warning: no target weights in the dataset, creating ones weights")
            target_weight = torch.ones_like(target_mask, dtype=float)
        target_data = target_data[limit_index]
        target_mask = target_mask[limit_index]
        target_weight = target_weight[limit_index]
        targets[event_particle] = (target_data, target_mask, target_weight)
    return targets


def preflight(args, run_dir):
    import socket
    print("==== Preflight ====", file=sys.stderr)
    print(f"hostname: {socket.gethostname()}", file=sys.stderr)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(f"CUDA_VISIBLE_DEVICES: {cvd}", file=sys.stderr)
    nvidia_smi_L = run_nvidia_smi(["-L"])
    print(f"nvidia-smi -L:\n{nvidia_smi_L}", file=sys.stderr)
    if cvd in ("<unset>", ""):
        print("FATAL: CUDA_VISIBLE_DEVICES not set. No stale-UUID fallback permitted.", file=sys.stderr)
        sys.exit(6)

    import torch
    if not torch.cuda.is_available():
        print(json.dumps({"exit_status": "ERROR", "error": "torch.cuda.is_available() is False"}, indent=2))
        sys.exit(1)
    device_name = torch.cuda.get_device_name(0)
    total_mem_bytes = torch.cuda.get_device_properties(0).total_memory

    for name, path in [("train", args.hdf5_train), ("val", args.hdf5_val)]:
        if not os.path.isfile(path):
            print(f"FATAL: {name} HDF5 not found: {path}", file=sys.stderr)
            sys.exit(2)
        print(f"{name} HDF5 sha256: {sha256_file(path)}", file=sys.stderr)
    if not os.path.isfile(args.event_yaml):
        print(f"FATAL: event yaml not found: {args.event_yaml}", file=sys.stderr)
        sys.exit(2)

    try:
        commit = subprocess.check_output(["git", "-C", SPANET_REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception as e:  # noqa: BLE001
        commit = f"<error: {e}>"
    if commit != SPANET_PINNED_COMMIT_EXPECTED:
        print(f"FATAL: pinned commit mismatch -- expected {SPANET_PINNED_COMMIT_EXPECTED}, got {commit}", file=sys.stderr)
        sys.exit(4)
    try:
        status = subprocess.check_output(["git", "-C", SPANET_REPO, "status", "--porcelain"], text=True)
    except Exception as e:  # noqa: BLE001
        status = f"<error: {e}>"
    if status.strip():
        print(f"FATAL: pinned spanet_repo working tree not clean:\n{status}", file=sys.stderr)
        sys.exit(5)

    with open(args.build_receipt) as f:
        build_receipt = json.load(f)
    if build_receipt["variant"] != args.variant:
        print(f"FATAL: build receipt variant={build_receipt['variant']} != requested --variant={args.variant}", file=sys.stderr)
        sys.exit(8)
    if not build_receipt.get("all_finite"):
        print("FATAL: build receipt reports all_finite=False -- refusing to train on it.", file=sys.stderr)
        sys.exit(9)

    print("==== Preflight OK ====", file=sys.stderr)
    return dict(device_name=device_name, total_mem_bytes=total_mem_bytes, cuda_visible_devices=cvd,
                spanet_commit=commit, spanet_working_tree_clean=(status.strip() == ""),
                build_receipt=build_receipt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["active", "all128"])
    ap.add_argument("--hdf5-train", required=True)
    ap.add_argument("--hdf5-val", required=True)
    ap.add_argument("--event-yaml", required=True)
    ap.add_argument("--build-receipt", required=True, help="BUILD_RECEIPT_<variant>_train.json from build_augmented_input.py")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--run-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    if not AUTHORIZED_TO_RUN:
        print("FATAL: AUTHORIZED_TO_RUN is False. Deliberate refusal -- a human must review the preflight "
              "and flip this constant explicitly before this script will train anything.", file=sys.stderr)
        sys.exit(2)

    preflight_info = preflight(args, args.run_dir)

    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks.progress.rich_progress import _RICH_AVAILABLE
    from pytorch_lightning.callbacks import (
        ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor,
        RichProgressBar, RichModelSummary, ModelSummary, TQDMProgressBar, Callback,
    )
    sys.path.insert(0, SPANET_REPO)
    from spanet import JetReconstructionModel, Options
    from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset
    JetReconstructionDataset.load_assignments = _patched_load_assignments

    class DevelopmentJetReconstructionModel(JetReconstructionModel):
        def validation_step(self, batch, batch_idx):
            metrics = super().validation_step(batch, batch_idx)
            with torch.no_grad():
                outputs = self.forward(batch.sources)
                symmetric_losses, best_indices = self.symmetric_losses(
                    outputs.assignments, outputs.detections, batch.assignment_targets)
                permutations = self.event_permutation_tensor[best_indices].T
                masks = torch.stack([t.mask for t in batch.assignment_targets])
                masks = torch.gather(masks, 0, permutations)
                weights = torch.ones_like(symmetric_losses)
                masks_u = masks.unsqueeze(1)
                symmetric_losses = (weights * symmetric_losses).sum(-1) / torch.clamp(masks_u.sum(-1), 1, None)
                assignment_loss, detection_loss = torch.unbind(symmetric_losses, 1)
                total_loss_terms = []
                if self.options.assignment_loss_scale > 0:
                    total_loss_terms.append(assignment_loss)
                if self.options.detection_loss_scale > 0:
                    total_loss_terms.append(detection_loss)
                if self.options.classification_loss_scale > 0:
                    total_loss_terms = self.add_classification_loss(
                        total_loss_terms, outputs.classifications, batch.classification_targets)
                total_loss = torch.cat([term.view(-1) for term in total_loss_terms])
                self.log("val/loss/total_loss", total_loss.sum(), on_step=False, on_epoch=True, sync_dist=True)
            return metrics

    class TimingCallback(Callback):
        def __init__(self):
            self.train_epoch_wall_s, self.val_epoch_wall_s = [], []
            self._t_train_start = self._t_val_start = None

        def on_train_epoch_start(self, trainer, pl_module):
            self._t_train_start = time.perf_counter()

        def on_train_epoch_end(self, trainer, pl_module):
            if self._t_train_start is not None:
                self.train_epoch_wall_s.append(time.perf_counter() - self._t_train_start)

        def on_validation_epoch_start(self, trainer, pl_module):
            self._t_val_start = time.perf_counter()

        def on_validation_epoch_end(self, trainer, pl_module):
            if self._t_val_start is not None:
                self.val_epoch_wall_s.append(time.perf_counter() - self._t_val_start)

    class EpochHistoryCallback(Callback):
        def __init__(self):
            self.history = []

        def on_train_epoch_end(self, trainer, pl_module):
            snap = {k: (float(v) if torch.is_tensor(v) else v) for k, v in trainer.callback_metrics.items()}
            snap["epoch"] = trainer.current_epoch
            snap["global_step"] = trainer.global_step
            self.history.append(snap)

    pl.seed_everything(SEED, workers=True)

    options = Options(args.event_yaml, args.hdf5_train, args.hdf5_val)
    options.batch_size = BATCH_SIZE
    options.num_dataloader_workers = NUM_DATALOADER_WORKERS
    options.dataset_limit = DATASET_LIMIT
    options.partial_events = PARTIAL_EVENTS
    for k, v in {**LOSS_SCALES, **ARCHITECTURE, **OPTIMIZER, **BALANCE_FLAGS}.items():
        setattr(options, k, v)
    options.epochs = MAX_EPOCHS
    options.num_gpu = NUM_GPU

    options_snapshot = {k: v for k, v in vars(options).items()}
    print(json.dumps(options_snapshot, indent=2, default=str), file=sys.stderr)

    with contextlib.redirect_stdout(sys.stderr):
        model = DevelopmentJetReconstructionModel(options)

    n_train_events, n_val_events = len(model.training_dataset), len(model.validation_dataset)
    print(f"n_train_events={n_train_events} n_val_events={n_val_events} "
          f"input_width={7 + preflight_info['build_receipt']['n_retained_part_dims']}", file=sys.stderr)

    logger = TensorBoardLogger(save_dir=args.run_dir, name=args.run_name)
    checkpoint_dir_tag = "{epoch}-{step}"
    primary_ckpt_cb = ModelCheckpoint(verbose=True, filename="primary-" + checkpoint_dir_tag + "-{validation_average_jet_accuracy:.4f}",
                                       monitor="validation_average_jet_accuracy", mode="max", save_top_k=1, save_last=True)
    secondary_ckpt_cb = ModelCheckpoint(verbose=True, filename="secondary-" + checkpoint_dir_tag + "-{val/loss/total_loss:.4f}",
                                         monitor="val/loss/total_loss", mode="min", save_top_k=1, save_last=False)
    timing_cb, epoch_history_cb = TimingCallback(), EpochHistoryCallback()
    callbacks = [primary_ckpt_cb, secondary_ckpt_cb, LearningRateMonitor(), DeviceStatsMonitor(),
                 RichProgressBar() if _RICH_AVAILABLE else TQDMProgressBar(),
                 RichModelSummary(max_depth=1) if _RICH_AVAILABLE else ModelSummary(max_depth=1),
                 timing_cb, epoch_history_cb]

    torch.cuda.reset_peak_memory_stats()
    trainer = pl.Trainer(accelerator="gpu", devices=1, precision="32-true",
                          gradient_clip_val=options.gradient_clip if options.gradient_clip > 0 else None,
                          max_epochs=MAX_EPOCHS, num_sanity_val_steps=0, logger=logger, callbacks=callbacks,
                          deterministic=False)

    fit_exception, fit_traceback, caught_warnings = None, None, []
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trainer.fit(model)
            caught_warnings = [str(x.message) for x in w]
    except Exception as e:  # noqa: BLE001
        fit_exception, fit_traceback = repr(e), traceback.format_exc()
        print(f"FIT FAILED: {fit_exception}\n{fit_traceback}", file=sys.stderr)
    total_fit_wall_s = time.perf_counter() - t0

    n_unexpected_warnings = sum(1 for m in caught_warnings if not any(s in m for s in EXPECTED_WARNING_SUBSTRINGS))
    peak_allocated, peak_reserved = int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())

    result = dict(
        variant=args.variant, exit_status="FAILED" if fit_exception else "COMPLETED",
        fit_exception=fit_exception, fit_traceback=fit_traceback,
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        run_name=args.run_name, hdf5_train=args.hdf5_train, hdf5_val=args.hdf5_val, event_yaml=args.event_yaml,
        build_receipt_used=preflight_info["build_receipt"],
        spanet_pinned_commit=preflight_info["spanet_commit"], gpu_device_name=preflight_info["device_name"],
        seed=SEED, batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
        n_train_events=n_train_events, n_val_events=n_val_events,
        options_snapshot=options_snapshot,
        epoch_history=epoch_history_cb.history,
        gpu_peak_allocated_bytes=peak_allocated, gpu_peak_reserved_bytes=peak_reserved,
        gpu_total_memory_bytes=preflight_info["total_mem_bytes"],
        total_fit_wall_s=total_fit_wall_s,
        train_epoch_wall_s_including_validation=timing_cb.train_epoch_wall_s,
        val_epoch_wall_s=timing_cb.val_epoch_wall_s,
        n_unexpected_warnings=n_unexpected_warnings,
        primary_checkpoint_path=primary_ckpt_cb.best_model_path,
        primary_checkpoint_sha256=sha256_file(primary_ckpt_cb.best_model_path) if os.path.exists(primary_ckpt_cb.best_model_path or "") else None,
        primary_checkpoint_best_score=float(primary_ckpt_cb.best_model_score) if primary_ckpt_cb.best_model_score is not None else None,
        primary_checkpoint_best_epoch=parse_epoch_from_ckpt_filename(primary_ckpt_cb.best_model_path),
        host_peak_rss_kib_self=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    out_path = os.path.join(args.run_dir, "work", f"training_{args.run_name}_result.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))
    if fit_exception:
        sys.exit(7)


if __name__ == "__main__":
    main()
