"""v2 governing-10M full-cohort inference for Harvey's FP32 question.

Authorized, CPU-only, read-only forward pass of the GOVERNING SPA-Net
10M seed-0 primary checkpoint
(sha256 fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c307102fa6943de5)
over the FULL frozen 400,000-event development validation cohort
(production_2M_val.h5, sha256
3c94bf8300d1dc3324e2cf25f9b6c618dffb94819cb3a297a43ac06320d5c2a2 --
the exact file/row-order `evaluate_classification.py` and
`evaluate_classification_10M.py` both use).

Reuses `evaluate_classification.py`'s own functions verbatim (imported,
not duplicated) for preflight, process-label reconstruction,
model/dataset construction, and checkpoint loading -- exactly the same
code path the original frozen evaluations used, run through the
identical pinned environment
(.pixi/envs/default, spanet_repo_v23exact @ debbdc999bfb785eb110a36c5fd3eff211ebf234,
torch 2.8.0+cu128 CPU, pytorch_lightning 2.6.0, h5py 3.14.0).

model.eval() + torch.no_grad() throughout. options.num_gpu=0 (CPU-only,
inherited unchanged from build_model_and_dataset()). No optimizer, no
gradient, no Trainer.fit(). No Stage-C/holdout_B file referenced
anywhere. No ParT code imported or touched.

New relative to the frozen evaluation scripts: this script captures the
raw pre-softmax logits (z0, z1) per event -- something neither
evaluate_classification.py nor evaluate_classification_10M.py ever
persisted (see the v1 package's Task 1/4 findings) -- in addition to
the production float32 softmax score, so that Delta = z1 - z0 and a
numerically stable float64 u_logit = softplus(Delta)/ln(10) can be
computed and compared against u_prob32 = -log10(1 - score_f32).
"""
import csv
import json
import os
import struct
import sys
import time

PHASE4AF = ("/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/"
            "phase4AF_spanet_partial_events_population_correction_20260819_v1")
BASE_EVAL_DIR = os.path.join(PHASE4AF, "spanet_2M_seed0_classification_evaluation_v1")
sys.path.insert(0, BASE_EVAL_DIR)
import evaluate_classification as base  # noqa: E402 -- reuse verbatim, do not duplicate

GOVERNING_CKPT = (
    "/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/"
    "phase4AI_spanet_10M_scaling_seed0_20260821_v1/full_run_attempt2_eaf_local_staging/"
    "spanet_10M_seed0/version_0/checkpoints/"
    "primary-epoch=47-step=234336-validation_average_jet_accuracy=0.4928.ckpt"
)
GOVERNING_CKPT_SHA256_EXPECTED = "fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c307102fa6943de5"

OUT_DIR = "/uscms_data/d3/iturkmen/repos/hh-bbww-baselines-bdt/docs/checkpoints/track_b_harvey_fp32_score_precision_20260906_v2_governing10m"
WORK_DIR = os.path.join(OUT_DIR, "work")

SPACING = 2.0 ** -24


def f32_hex_be(x, np):
    return struct.pack(">f", np.float32(x)).hex()


def main():
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import pyarrow as pa
    import pyarrow.parquet as pq

    t_start = time.time()

    # ---- Step 1/2: re-verify governing checkpoint identity (defense in depth) ----
    actual_ckpt_sha = base.sha256_file(GOVERNING_CKPT)
    print(f"governing checkpoint sha256: {actual_ckpt_sha}", file=sys.stderr)
    if actual_ckpt_sha != GOVERNING_CKPT_SHA256_EXPECTED:
        print(f"FATAL: governing checkpoint sha256 mismatch -- expected "
              f"{GOVERNING_CKPT_SHA256_EXPECTED}, got {actual_ckpt_sha}", file=sys.stderr)
        sys.exit(2)

    # ---- Step 3/4: re-verify shared validation HDF5 identity (defense in depth) ----
    actual_val_sha = base.sha256_file(base.HDF5_VAL)
    print(f"HDF5_VAL sha256: {actual_val_sha}", file=sys.stderr)
    if actual_val_sha != base.HDF5_VAL_SHA256_EXPECTED:
        print(f"FATAL: HDF5_VAL sha256 mismatch -- expected "
              f"{base.HDF5_VAL_SHA256_EXPECTED}, got {actual_val_sha}", file=sys.stderr)
        sys.exit(2)

    # spanet pinned-commit / clean-tree checks, same as the frozen evaluation's own preflight()
    base.PRIMARY_CKPT = GOVERNING_CKPT
    base.PRIMARY_CKPT_SHA256_EXPECTED = GOVERNING_CKPT_SHA256_EXPECTED
    preflight_info = base.preflight()

    process_labels, per_process_file_count, n_manifest_rows = base.reconstruct_process_labels()

    import h5py
    with h5py.File(base.HDF5_VAL, "r") as f:
        hdf5_signal_label = f["CLASSIFICATIONS/EVENT/signal"][:]
    process_check = base.verify_process_reconstruction(process_labels, hdf5_signal_label)
    process_arr = np.array(process_labels)
    print(f"process reconstruction VERIFIED: {process_check}", file=sys.stderr)

    model = base.build_model_and_dataset()  # options.num_gpu=0 -- CPU only, unchanged
    n_val_events = len(model.validation_dataset)

    missing_p, unexpected_p = base.load_checkpoint_into(model, GOVERNING_CKPT, strict=True)
    print(f"checkpoint load: missing={missing_p} unexpected={unexpected_p}", file=sys.stderr)

    model.eval()
    full_loader = DataLoader(
        model.validation_dataset, batch_size=base.BATCH_SIZE, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=False,
    )

    all_z0, all_z1, all_probs32, all_true = [], [], [], []
    t0 = time.time()
    n_total = 0
    with torch.no_grad():
        for batch in full_loader:
            outputs = model.forward(batch.sources)
            logits = outputs.classifications["EVENT/signal"]  # [B, 2], float32
            true = batch.classification_targets["EVENT/signal"]

            probs32 = torch.softmax(logits, dim=-1)[:, 1]  # identical production code path

            all_z0.append(logits[:, 0].numpy())
            all_z1.append(logits[:, 1].numpy())
            all_probs32.append(probs32.numpy())
            all_true.append(true.numpy())
            n_total += logits.shape[0]
    wall_s = time.time() - t0
    print(f"forward pass wall time: {wall_s:.2f}s for {n_total} events "
          f"({n_total / wall_s:.1f} events/s)", file=sys.stderr)

    z0 = np.concatenate(all_z0)          # float32
    z1 = np.concatenate(all_z1)          # float32
    score32 = np.concatenate(all_probs32)  # float32 -- production softmax path, unchanged
    true = np.concatenate(all_true)

    assert n_total == n_val_events == 400_000, f"n_total={n_total} n_val_events={n_val_events}"
    assert np.array_equal(true, hdf5_signal_label), "dataloader label order != HDF5 label order -- ABORT"
    assert np.isfinite(score32).all(), "non-finite probability found"
    assert (score32 >= 0.0).all() and (score32 <= 1.0).all(), "probability outside [0,1]"
    assert np.isfinite(z0).all() and np.isfinite(z1).all(), "non-finite logit found"

    # ---- derived quantities, float64 throughout ----
    z0_64 = z0.astype(np.float64)
    z1_64 = z1.astype(np.float64)
    delta = z1_64 - z0_64  # Delta = z_signal - z_background

    def softplus_stable(x):
        return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))

    u_logit = softplus_stable(delta) / np.log(10.0)  # == -log10(1 - sigmoid(delta)) in exact arithmetic

    score64 = score32.astype(np.float64)
    with np.errstate(divide="ignore"):
        u_prob32 = -np.log10(1.0 - score64)  # +inf where score32 == 1.0 exactly, by construction

    k_round = np.zeros(len(score32), dtype=np.int64)
    below_one = score32 < np.float32(1.0)
    k_round[below_one] = np.round((1.0 - score64[below_one]) / SPACING).astype(np.int64)

    sigmoid_delta_f64 = 1.0 / (1.0 + np.exp(-delta))  # diagnostic, float64

    row_index = np.arange(n_total, dtype=np.int64)
    is_signal = process_arr == "signal"
    is_qcd = process_arr == "qcd"
    is_ttbar = process_arr == "ttbar"
    is_bg = ~is_signal

    # ---- write full per-event parquet (via pyarrow directly -- pandas is broken
    # in this pinned env: `import pandas` fails on missing pytz/dateutil. Not
    # fixed, not worked around by installing anything -- pyarrow alone is
    # sufficient and is already present in the pinned env.) ----
    score32_hex = np.array([f32_hex_be(v, np) for v in score32])
    table = pa.table({
        "row_index": row_index,
        "process_label": process_arr,
        "z_background": z0_64,
        "z_signal": z1_64,
        "delta": delta,
        "score_float32_production": score32.astype(np.float64),
        "score_float32_hex_be": score32_hex,
        "k_lattice": k_round,
        "u_prob32": u_prob32,
        "u_logit": u_logit,
        "sigmoid_delta_float64": sigmoid_delta_f64,
    })
    os.makedirs(OUT_DIR, exist_ok=True)
    pq.write_table(table, os.path.join(OUT_DIR, "SPA10M_EVENT_LOGITS_400K.parquet"))

    os.makedirs(WORK_DIR, exist_ok=True)
    meta = dict(
        governing_checkpoint=GOVERNING_CKPT,
        governing_checkpoint_sha256=actual_ckpt_sha,
        hdf5_val=base.HDF5_VAL,
        hdf5_val_sha256=actual_val_sha,
        spanet_commit=preflight_info["spanet_commit"],
        n_scored=int(n_total),
        n_signal=int(is_signal.sum()), n_qcd=int(is_qcd.sum()), n_ttbar=int(is_ttbar.sum()),
        n_all_background=int(is_bg.sum()),
        checkpoint_load_missing_keys=missing_p, checkpoint_load_unexpected_keys=unexpected_p,
        forward_pass_wall_s=wall_s,
        forward_pass_events_per_s=n_total / wall_s,
        total_wall_s=time.time() - t_start,
        process_reconstruction_check=process_check,
    )
    with open(os.path.join(WORK_DIR, "run_governing_10m_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"wrote {os.path.join(OUT_DIR, 'SPA10M_EVENT_LOGITS_400K.parquet')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
