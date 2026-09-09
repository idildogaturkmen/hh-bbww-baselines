"""Task 5 -- PREPARED, NOT LAUNCHED.

Tail-only rerun of the GOVERNING SPA-Net 10M primary checkpoint
(sha256 fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c307102fa6943de5)
restricted to the exact 1,256 event identities already known from the
frozen 2M sibling score archive (../TAIL_ONLY_RERUN_EVENT_IDENTITIES.csv,
row indices into the shared production_2M_val.h5 -- the SAME file and
SAME row order used by both the 2M and 10M classification evaluations;
verified from evaluate_classification_10M.py, which inherits
`base.HDF5_VAL` unchanged from evaluate_classification.py).

This does NOT touch Stage-C/holdout_B (still the exact frozen 400k
development validation cohort). It does NOT retrain anything (model.eval()
+ torch.no_grad() only, reusing the existing, unmodified
build_model_and_dataset()/load_checkpoint_into() functions verbatim).
It does NOT launch a broad inference job: only 1,256 of 400,000 rows
(0.314%) are scored, via a torch.utils.data.Subset, not the full loader.

Purpose: recover the raw pre-softmax classification logits (z0, z1)
for exactly the tail events already implicated in the FP32 lattice
pile-up (Tasks 2-3), for the checkpoint that actually governs the
project -- something no existing frozen archive contains (Task 1/4
finding: neither evaluate_classification.py nor
evaluate_classification_10M.py ever persists per-event logits).

DO NOT RUN AUTOMATICALLY. This file is a prepared plan only, per
explicit instruction ("Prepare the EAF command but DO NOT launch it").
"""
import csv
import json
import os
import sys

import numpy as np

PHASE4AF = ("/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/"
            "phase4AF_spanet_partial_events_population_correction_20260819_v1")
BASE_EVAL_DIR = os.path.join(PHASE4AF, "spanet_2M_seed0_classification_evaluation_v1")
sys.path.insert(0, BASE_EVAL_DIR)
import evaluate_classification as base  # noqa: E402 -- reuse verbatim, do not duplicate

PHASE4AI = ("/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/"
            "phase4AI_spanet_10M_scaling_seed0_20260821_v1")
GOVERNING_CKPT = os.path.join(
    PHASE4AI, "full_run_attempt2_eaf_local_staging", "spanet_10M_seed0", "version_0",
    "checkpoints", "primary-epoch=47-step=234336-validation_average_jet_accuracy=0.4928.ckpt",
)
GOVERNING_CKPT_SHA256_EXPECTED = "fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c307102fa6943de5"

EVENT_IDENTITIES_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "TAIL_ONLY_RERUN_EVENT_IDENTITIES.csv")
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "work_tail_rerun", "tail_only_rerun_result.json")


def main():
    assert base.sha256_file(GOVERNING_CKPT) == GOVERNING_CKPT_SHA256_EXPECTED, \
        "governing checkpoint hash mismatch -- ABORT, do not proceed"

    with open(EVENT_IDENTITIES_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    row_indices = [int(r["row_index_in_production_2M_val_h5"]) for r in rows]
    assert len(row_indices) == 1256

    import torch
    from torch.utils.data import DataLoader, Subset
    import torch.nn.functional as F

    base.PRIMARY_CKPT = GOVERNING_CKPT
    base.PRIMARY_CKPT_SHA256_EXPECTED = GOVERNING_CKPT_SHA256_EXPECTED
    base.preflight()

    model = base.build_model_and_dataset()
    missing_p, unexpected_p = base.load_checkpoint_into(model, GOVERNING_CKPT, strict=True)
    model.eval()

    subset = Subset(model.validation_dataset, row_indices)
    loader = DataLoader(subset, batch_size=256, shuffle=False, drop_last=False, num_workers=0)

    out_rows = []
    idx_ptr = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model.forward(batch.sources)
            logits = outputs.classifications["EVENT/signal"]  # [B, 2], float32
            z0 = logits[:, 0].double().numpy()
            z1 = logits[:, 1].double().numpy()
            delta = z1 - z0  # z_signal - z_background

            probs_f32_normal = torch.softmax(logits, dim=-1)[:, 1].numpy()  # production path, float32
            probs_f64_from_delta = 1.0 / (1.0 + np.exp(-delta))  # recomputed sigmoid(Delta), float64

            b = len(z0)
            for j in range(b):
                row_idx = row_indices[idx_ptr]
                out_rows.append(dict(
                    row_index=row_idx,
                    process_label=rows[idx_ptr]["process_label"],
                    score_2M_f32_reference=rows[idx_ptr]["score_2M_f32"],
                    z0=float(z0[j]), z1=float(z1[j]), delta=float(delta[j]),
                    softmax_probability_float32_production=float(probs_f32_normal[j]),
                    sigmoid_delta_float64_recomputed=float(probs_f64_from_delta[j]),
                ))
                idx_ptr += 1

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(dict(
            governing_checkpoint_sha256=GOVERNING_CKPT_SHA256_EXPECTED,
            n_events=len(out_rows),
            rows=out_rows,
        ), f, indent=2)
    print(f"wrote {OUT_PATH} ({len(out_rows)} events)")


if __name__ == "__main__":
    raise SystemExit(
        "This script is PREPARED but intentionally refuses to run automatically. "
        "Remove this guard only after explicit human authorization to launch the "
        "tail-only rerun described in the module docstring, then invoke main()."
    )
