"""Track D, Task 1 -- fifth-jet / extra-jet "leading-four-only" counterfactual.

Genuine causal A/B forward-pass test, requested by Harvey and specified
(but explicitly NOT executed) in
track_b_harvey_tail_kinematics_20260907_v2/HARVEY_FIFTH_JET_SUMMARY.md Sec.5.
This script executes that exact plan for the first time.

(A) Original input: the native SPA10M input tensor as scored by every
    prior frozen evaluation in this project (unchanged mask).
(B) Leading-four-only counterfactual: identical event, identical jet
    ordering/kinematics for slots 0-3, with mask[:, 4:] = False (jet
    slots 4-9, i.e. rank >= 5, hidden from the model). Jet DATA values
    are never modified, only the boolean mask.

Both A and B are scored through the *same* frozen governing SPA-Net 10M
checkpoint (sha256 fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c30
7102fa6943de5) via the identical model.forward() code path used by
run_governing_10m_tail_inference.py -- reused verbatim (imported), not
duplicated. CPU-only, model.eval() + torch.no_grad() throughout. No
optimizer, no gradient, no Trainer.fit(). No Stage-C/holdout_B file
referenced. No ParT code touched. No package installed.

u_logit = softplus(Delta)/ln(10), Delta = z1 - z0, exactly as specified
in the plan document -- NOT the FP32-saturating u = -log10(1-score).
"""
import json
import os
import sys
import time

GOVERNING10M_PKG = ("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines-bdt/docs/checkpoints/"
                     "track_b_harvey_fp32_score_precision_20260906_v2_governing10m")
sys.path.insert(0, os.path.join(GOVERNING10M_PKG, "scripts"))

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

OUT_DIR = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_final_question_closure_20260907_v1"
WORK_DIR = os.path.join(OUT_DIR, "work")

N_TOTAL_SLOTS = 10
LEADING_N = 4  # mask[:, LEADING_N:] = False for the counterfactual


def softplus_stable(x, np):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def main():
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import pyarrow as pa
    import pyarrow.parquet as pq
    from spanet.dataset.types import Source

    t_start = time.time()

    # ---- re-verify governing checkpoint + HDF5 identity (defense in depth) ----
    actual_ckpt_sha = base.sha256_file(GOVERNING_CKPT)
    print(f"governing checkpoint sha256: {actual_ckpt_sha}", file=sys.stderr)
    assert actual_ckpt_sha == GOVERNING_CKPT_SHA256_EXPECTED, "checkpoint sha256 mismatch"

    actual_val_sha = base.sha256_file(base.HDF5_VAL)
    print(f"HDF5_VAL sha256: {actual_val_sha}", file=sys.stderr)
    assert actual_val_sha == base.HDF5_VAL_SHA256_EXPECTED, "HDF5 sha256 mismatch"

    base.PRIMARY_CKPT = GOVERNING_CKPT
    base.PRIMARY_CKPT_SHA256_EXPECTED = GOVERNING_CKPT_SHA256_EXPECTED
    preflight_info = base.preflight()

    process_labels, per_process_file_count, n_manifest_rows = base.reconstruct_process_labels()

    import h5py
    with h5py.File(base.HDF5_VAL, "r") as f:
        hdf5_signal_label = f["CLASSIFICATIONS/EVENT/signal"][:]
        raw_pt = f["INPUTS/Source/pt"][:]
        raw_eta = f["INPUTS/Source/eta"][:]
        raw_phi = f["INPUTS/Source/phi"][:]
        raw_mask = f["INPUTS/Source/MASK"][:]
    process_check = base.verify_process_reconstruction(process_labels, hdf5_signal_label)
    process_arr = np.array(process_labels)
    print(f"process reconstruction VERIFIED: {process_check}", file=sys.stderr)

    n_selected_jets = raw_mask.sum(axis=1).astype(np.int64)

    model = base.build_model_and_dataset()  # options.num_gpu=0 -- CPU only
    n_val_events = len(model.validation_dataset)

    missing_p, unexpected_p = base.load_checkpoint_into(model, GOVERNING_CKPT, strict=True)
    print(f"checkpoint load: missing={missing_p} unexpected={unexpected_p}", file=sys.stderr)

    model.eval()
    full_loader = DataLoader(
        model.validation_dataset, batch_size=base.BATCH_SIZE, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=False,
    )

    all_z0_a, all_z1_a, all_z0_b, all_z1_b, all_true = [], [], [], [], []
    n_total = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in full_loader:
            # ---- (A) original, native input: exact frozen code path, unmodified ----
            outputs_a = model.forward(batch.sources)
            logits_a = outputs_a.classifications["EVENT/signal"]  # [B, 2]

            # ---- (B) leading-four-only counterfactual ----
            # batch.sources is Tuple[Source, ...]; this event topology has exactly
            # one source group ("Source" = jets). Clone the mask, never the data.
            src = batch.sources[0]
            mask_b = src.mask.clone()
            mask_b[:, LEADING_N:] = False
            sources_b = (Source(data=src.data, mask=mask_b),) + tuple(batch.sources[1:])
            outputs_b = model.forward(sources_b)
            logits_b = outputs_b.classifications["EVENT/signal"]

            true = batch.classification_targets["EVENT/signal"]

            all_z0_a.append(logits_a[:, 0].numpy())
            all_z1_a.append(logits_a[:, 1].numpy())
            all_z0_b.append(logits_b[:, 0].numpy())
            all_z1_b.append(logits_b[:, 1].numpy())
            all_true.append(true.numpy())
            n_total += logits_a.shape[0]
    wall_s = time.time() - t0
    print(f"forward pass (A+B) wall time: {wall_s:.2f}s for {n_total} events "
          f"({n_total / wall_s:.1f} events/s)", file=sys.stderr)

    z0_a = np.concatenate(all_z0_a).astype(np.float64)
    z1_a = np.concatenate(all_z1_a).astype(np.float64)
    z0_b = np.concatenate(all_z0_b).astype(np.float64)
    z1_b = np.concatenate(all_z1_b).astype(np.float64)
    true = np.concatenate(all_true)

    assert n_total == n_val_events == 400_000, f"n_total={n_total} n_val_events={n_val_events}"
    assert np.array_equal(true, hdf5_signal_label), "dataloader label order != HDF5 label order -- ABORT"
    assert np.isfinite(z0_a).all() and np.isfinite(z1_a).all(), "non-finite logit found (A)"
    assert np.isfinite(z0_b).all() and np.isfinite(z1_b).all(), "non-finite logit found (B)"

    delta_a = z1_a - z0_a
    delta_b = z1_b - z0_b
    u_logit_a = softplus_stable(delta_a, np) / np.log(10.0)
    u_logit_b = softplus_stable(delta_b, np) / np.log(10.0)

    # Sanity: for events with n_selected_jets <= 4, masking slots >=4 changes
    # nothing (those slots were already False), so A and B must be IDENTICAL.
    n_le4 = n_selected_jets <= LEADING_N
    max_abs_diff_le4 = float(np.max(np.abs(delta_b[n_le4] - delta_a[n_le4]))) if n_le4.any() else 0.0
    print(f"sanity: max|Delta_B-Delta_A| for n_selected_jets<=4 (n={int(n_le4.sum())}): "
          f"{max_abs_diff_le4:.3e} (must be ~0)", file=sys.stderr)
    assert max_abs_diff_le4 < 1e-4, "masking changed a <=4-jet event -- BUG"

    # ---- per-event kinematics needed for stratification, computed independently
    # of the model, directly from the raw HDF5 arrays (pt/eta/phi), no SPA-Net
    # normalization involved ----
    pt5 = np.full(n_total, np.nan, dtype=np.float64)
    min_dr5 = np.full(n_total, np.nan, dtype=np.float64)
    has5 = raw_mask[:, 4]
    pt5[has5] = raw_pt[has5, 4]

    def delta_r(eta1, phi1, eta2, phi2):
        dphi = np.abs(phi1 - phi2)
        dphi = np.minimum(dphi, 2 * np.pi - dphi)
        return np.sqrt((eta1 - eta2) ** 2 + dphi ** 2)

    idx5 = np.where(has5)[0]
    for lead in range(LEADING_N):
        dr = delta_r(raw_eta[idx5, 4], raw_phi[idx5, 4], raw_eta[idx5, lead], raw_phi[idx5, lead])
        if lead == 0:
            best = dr
        else:
            best = np.minimum(best, dr)
    min_dr5[idx5] = best

    is_signal = process_arr == "signal"
    is_qcd = process_arr == "qcd"
    is_ttbar = process_arr == "ttbar"

    table = pa.table({
        "row_index": np.arange(n_total, dtype=np.int64),
        "process_label": process_arr,
        "n_selected_jets": n_selected_jets,
        "pt5": pt5,
        "min_dR_j5_leading4": min_dr5,
        "z_background_A": z0_a, "z_signal_A": z1_a, "delta_A": delta_a, "u_logit_A": u_logit_a,
        "z_background_B": z0_b, "z_signal_B": z1_b, "delta_B": delta_b, "u_logit_B": u_logit_b,
        "delta_u_logit": u_logit_b - u_logit_a,
        "delta_logit_margin": delta_b - delta_a,
    })
    os.makedirs(OUT_DIR, exist_ok=True)
    pq.write_table(table, os.path.join(OUT_DIR, "FIFTH_JET_COUNTERFACTUAL_EVENTS_ALL400K.parquet"))

    os.makedirs(WORK_DIR, exist_ok=True)
    meta = dict(
        governing_checkpoint=GOVERNING_CKPT,
        governing_checkpoint_sha256=actual_ckpt_sha,
        hdf5_val=base.HDF5_VAL,
        hdf5_val_sha256=actual_val_sha,
        spanet_commit=preflight_info["spanet_commit"],
        n_scored=int(n_total),
        n_signal=int(is_signal.sum()), n_qcd=int(is_qcd.sum()), n_ttbar=int(is_ttbar.sum()),
        checkpoint_load_missing_keys=missing_p, checkpoint_load_unexpected_keys=unexpected_p,
        forward_pass_wall_s_A_and_B=wall_s,
        forward_pass_events_per_s_A_and_B=n_total / wall_s,
        total_wall_s=time.time() - t_start,
        process_reconstruction_check=process_check,
        sanity_max_abs_delta_diff_n_le4=max_abs_diff_le4,
        n_events_n_selected_jets_le4=int(n_le4.sum()),
        n_events_n_selected_jets_ge5=int((~n_le4).sum()),
        leading_n=LEADING_N,
        n_total_slots=N_TOTAL_SLOTS,
    )
    with open(os.path.join(WORK_DIR, "run_fifth_jet_counterfactual_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"wrote {os.path.join(OUT_DIR, 'FIFTH_JET_COUNTERFACTUAL_EVENTS_ALL400K.parquet')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
