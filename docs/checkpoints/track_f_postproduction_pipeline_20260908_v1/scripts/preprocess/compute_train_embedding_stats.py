#!/usr/bin/env python3
"""TRAIN-ONLY per-dimension ParT embedding diagnostics.

Per explicit instruction: the prior PREPROCESSING_DECISION.md (2026-09-07)
threshold (std<1e-3) and its 20-retained-dimension list were computed from
an incomplete 31-shard / 336,528-jet development sample and are NOT a
frozen scientific result -- they are a provisional METHOD only. This
script recomputes everything from the COMPLETE TRAIN population (never
val, never any dimension of val statistics), reports MULTIPLE candidate
thresholds so the 1e-3 choice can be justified rather than silently
re-canonized, and writes ONE frozen stats file that both build_augmented_input.py
variants (A: active-dim-only, B: all-128 eps-clamped) consume.

Reads directly from the TRAIN joined HDF5 produced by run_full_join.sh
(joined_features (N,10,135) = 7 native + 128 ParT, mask (N,10)) -- only
real jets (mask==True) contribute to any statistic. Never opens the val
joined HDF5.

Usage:
    python3 compute_train_embedding_stats.py \
        --joined-train-h5 joined_train.h5 \
        --out PART_TRAIN_STATS_FROZEN.json
"""
import argparse
import hashlib
import json
import sys

import h5py
import numpy as np

PART_DIM = 128
NATIVE_DIM = 7
CANDIDATE_THRESHOLDS = [1e-1, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5]
QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined-train-h5", required=True,
                     help="TRAIN split only -- refuse to run on anything with 'val' in the filename as a cheap guard")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-events", type=int, default=200_000,
                     help="process the (potentially large) train tensor in chunks to bound memory")
    args = ap.parse_args()

    if "val" in args.joined_train_h5.lower().split("/")[-1]:
        print("FATAL: refusing to compute 'train-only' stats from a file whose name contains 'val'. "
              "This is a cheap filename guard, not a substitute for passing the right path.", file=sys.stderr)
        sys.exit(2)

    with h5py.File(args.joined_train_h5, "r") as f:
        joined = f["joined_features"]
        mask = f["mask"][:]
        n_events, n_slots, width = joined.shape
        if width != NATIVE_DIM + PART_DIM:
            print(f"FATAL: expected width {NATIVE_DIM + PART_DIM}, got {width}", file=sys.stderr)
            sys.exit(3)

        n_real_jets = int(mask.sum())
        # running (count, sum, sumsq) per dim for mean/std; separately
        # collect the full real-jet embedding array for exact quantiles
        # (bounded: n_real_jets x 128 floats -- at the 2M-train scale this
        # is ~9.04M x 128 x 4 bytes =~ 4.6 GiB, the single largest
        # allocation in this script; chunked read keeps I/O bounded even
        # though the final quantile array itself is not chunked further).
        collected = np.empty((n_real_jets, PART_DIM), dtype=np.float32)
        collected_norm = np.empty(n_real_jets, dtype=np.float32)
        cursor = 0
        n_nonfinite_total = 0
        for lo in range(0, n_events, args.chunk_events):
            hi = min(lo + args.chunk_events, n_events)
            chunk = joined[lo:hi, :, NATIVE_DIM:]  # (hi-lo, 10, 128)
            chunk_mask = mask[lo:hi]               # (hi-lo, 10)
            real = chunk[chunk_mask]                # (n_real_in_chunk, 128)
            n_nonfinite_total += int((~np.isfinite(real)).sum())
            n_real_in_chunk = real.shape[0]
            collected[cursor:cursor + n_real_in_chunk] = real
            collected_norm[cursor:cursor + n_real_in_chunk] = np.linalg.norm(real, axis=1)
            cursor += n_real_in_chunk
        assert cursor == n_real_jets, f"chunked real-jet count {cursor} != mask.sum() {n_real_jets}"

    per_dim = {}
    stds = np.zeros(PART_DIM)
    for d in range(PART_DIM):
        col = collected[:, d]
        finite_col = col[np.isfinite(col)]
        mean = float(finite_col.mean()) if finite_col.size else float("nan")
        std = float(finite_col.std()) if finite_col.size else float("nan")
        stds[d] = std
        per_dim[f"dim_{d:03d}"] = {
            "mean": mean, "std": std,
            "min": float(finite_col.min()) if finite_col.size else None,
            "max": float(finite_col.max()) if finite_col.size else None,
            "quantiles": {str(q): float(np.quantile(finite_col, q)) for q in QUANTILES} if finite_col.size else None,
            "n_nonfinite": int((~np.isfinite(col)).sum()),
        }

    sorted_stds = np.sort(stds)
    # data-driven gap: the largest ratio jump between consecutive sorted
    # std values (log-scale gap), reported as a candidate -- NOT auto-applied.
    log_stds = np.log10(np.clip(sorted_stds, 1e-12, None))
    gaps = np.diff(log_stds)
    gap_idx = int(np.argmax(gaps)) if len(gaps) else None
    data_driven_gap_threshold = float(10 ** ((log_stds[gap_idx] + log_stds[gap_idx + 1]) / 2)) if gap_idx is not None else None

    threshold_scan = {}
    for t in CANDIDATE_THRESHOLDS:
        active = [d for d in range(PART_DIM) if stds[d] >= t]
        threshold_scan[str(t)] = {"n_active": len(active), "active_dims": active}
    if data_driven_gap_threshold is not None:
        active_gap = [d for d in range(PART_DIM) if stds[d] >= data_driven_gap_threshold]
        threshold_scan[f"data_driven_gap={data_driven_gap_threshold:.6g}"] = {"n_active": len(active_gap), "active_dims": active_gap}

    embedding_norm_stats = {
        "mean": float(collected_norm.mean()), "std": float(collected_norm.std()),
        "min": float(collected_norm.min()), "max": float(collected_norm.max()),
        "quantiles": {str(q): float(np.quantile(collected_norm, q)) for q in QUANTILES},
    }

    result = {
        "_purpose": "TRAIN-ONLY frozen ParT embedding statistics. Consumed unchanged by build_augmented_input.py for BOTH train and val (val stats are NEVER computed or used for standardization).",
        "input_h5": args.joined_train_h5,
        "input_h5_sha256": sha256_file(args.joined_train_h5),
        "n_train_events": n_events,
        "n_real_jets_train": n_real_jets,
        "n_nonfinite_values_total": n_nonfinite_total,
        "all_finite": n_nonfinite_total == 0,
        "per_dim_stats": per_dim,
        "embedding_l2_norm_stats_train_real_jets": embedding_norm_stats,
        "threshold_scan_NOT_A_DECISION_a_data_diagnostic_only": threshold_scan,
        "data_driven_gap_threshold_recommended_for_review": data_driven_gap_threshold,
        "prior_provisional_decision_for_comparison": {
            "source": "track_b_part_postflight_2m_20260907_v1/preprocessing_decision/PREPROCESSING_DECISION.md",
            "threshold_used_there": 1e-3,
            "sample_size_there": "336528 jets / 31 of 1100 shards (incomplete)",
            "n_active_there": 20,
        },
        "REMINDER": "This file reports diagnostics at multiple candidate thresholds. It does NOT itself pick a threshold. The actual active-dimension freeze decision is a human/project decision to be recorded in build_augmented_input.py's --active-threshold argument at the time variant A is built, justified against this file's threshold_scan.",
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"n_train_events={n_events} n_real_jets_train={n_real_jets} all_finite={result['all_finite']}")
    print(f"data_driven_gap_threshold_recommended_for_review={data_driven_gap_threshold}")
    for t, v in threshold_scan.items():
        print(f"  threshold={t}: n_active={v['n_active']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
