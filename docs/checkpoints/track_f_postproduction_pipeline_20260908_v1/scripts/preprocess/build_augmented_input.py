#!/usr/bin/env python3
"""Build the SPA-Net-facing augmented input H5 from a joined native+ParT
HDF5, applying TRAIN-FROZEN standardization constants unchanged to
whichever split is being written (train or val -- val is NEVER used to
compute a mean/std, only to be transformed by train's).

Two independent variants, per explicit instruction (do not let a result
be attributable to cherry-picked pruning):

  A. "active" -- native 7 + only the ParT dims whose TRAIN std clears
     --active-threshold (default None: must be passed explicitly and is
     echoed into the receipt with its justification/threshold_scan
     context, traceable to compute_train_embedding_stats.py -- never a
     silent default).
  B. "all128" -- native 7 + all 128 ParT dims, eps-clamped standardization
     (z = (x-mean)/max(std,eps)) for every dimension, none dropped. This
     is the sensitivity/control variant.

SCHEMA (matches the native production_2M_{train,val}.h5's own layout, per
direct h5py inspection of that file -- SPA-Net's JetReconstructionDataset
reads one NAMED (N,10) dataset per Source key, not one packed (N,W,10)
tensor):
  INPUTS/Source/{pt,eta,phi,mass,probB,probC,probL}  -- copied UNCHANGED
      from the joined H5's native columns (themselves copied unchanged
      from the original native H5 by streaming_native_part_join.py)
  INPUTS/Source/part_emb_NNN  (one per retained dim)  -- NEW, standardized
  INPUTS/Source/MASK  -- copied unchanged
  TARGETS/*, CLASSIFICATIONS/*  -- copied UNCHANGED from --native-h5
      (the ORIGINAL production_2M_{train,val}.h5), since the joined H5
      never carries these groups at all (streaming_native_part_join.py
      only ever reads INPUTS/Source/* from native). Training needs these
      to exist for the assignment/classification losses to be computable.

Never overwrites the native production_2M_{train,val}.h5 or the joined
intermediate -- always writes to a new, versioned output path.

Usage:
    python3 build_augmented_input.py \
        --joined-h5 joined_train.h5 --native-h5 production_2M_train.h5 --split train \
        --train-stats PART_TRAIN_STATS_FROZEN.json \
        --variant active --active-threshold 1e-3 \
        --event-yaml-template trackb_hh4b.yaml \
        --out-h5 spa2m_part_active_train.h5 --out-yaml part_augmented_active_hh4b.yaml \
        --out-receipt BUILD_RECEIPT_active_train.json
"""
import argparse
import hashlib
import json
import sys

import h5py
import numpy as np
import yaml

NATIVE_DIM = 7
PART_DIM = 128
NATIVE_FEATURE_NAMES = ["pt", "eta", "phi", "mass", "probB", "probC", "probL"]
COPY_GROUPS_FROM_NATIVE = ["TARGETS", "CLASSIFICATIONS"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_train_stats(path):
    with open(path) as f:
        stats = json.load(f)
    means = np.array([stats["per_dim_stats"][f"dim_{d:03d}"]["mean"] for d in range(PART_DIM)], dtype=np.float64)
    stds = np.array([stats["per_dim_stats"][f"dim_{d:03d}"]["std"] for d in range(PART_DIM)], dtype=np.float64)
    return stats, means, stds


def build_event_yaml(template_path, retained_dims, out_path):
    with open(template_path) as f:
        cfg = yaml.safe_load(f)
    source = cfg["INPUTS"]["SEQUENTIAL"]["Source"]
    for d in retained_dims:
        source[f"part_emb_{d:03d}"] = "none"  # already standardized by this script -- no further SPA-Net-side normalization
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return list(source.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined-h5", required=True)
    ap.add_argument("--native-h5", required=True,
                     help="the ORIGINAL production_2M_{train,val}.h5 -- TARGETS/CLASSIFICATIONS are copied from here, never from the joined H5")
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--train-stats", required=True, help="PART_TRAIN_STATS_FROZEN.json -- always TRAIN-derived, even when --split val")
    ap.add_argument("--variant", required=True, choices=["active", "all128"])
    ap.add_argument("--active-threshold", type=float, default=None,
                     help="required for --variant active; a TRAIN std cutoff, must be justified against the stats file's threshold_scan")
    ap.add_argument("--eps", type=float, default=1e-3, help="denominator floor for z=(x-mean)/max(std,eps)")
    ap.add_argument("--event-yaml-template", required=True)
    ap.add_argument("--out-h5", required=True)
    ap.add_argument("--out-yaml", required=True)
    ap.add_argument("--out-receipt", required=True)
    ap.add_argument("--chunk-events", type=int, default=200_000)
    args = ap.parse_args()

    if args.variant == "active" and args.active_threshold is None:
        print("FATAL: --variant active requires --active-threshold explicitly (no silent default). "
              "Justify it against compute_train_embedding_stats.py's threshold_scan.", file=sys.stderr)
        sys.exit(2)

    stats, train_means, train_stds = load_train_stats(args.train_stats)

    if args.variant == "active":
        retained_dims = [d for d in range(PART_DIM) if train_stds[d] >= args.active_threshold]
    else:
        retained_dims = list(range(PART_DIM))
    eps_used = args.eps

    n_retained = len(retained_dims)
    if n_retained == 0:
        print("FATAL: zero retained dimensions -- threshold too strict.", file=sys.stderr)
        sys.exit(3)

    n_nonfinite_written = 0
    with h5py.File(args.joined_h5, "r") as fin, h5py.File(args.native_h5, "r") as fnat:
        joined_in = fin["joined_features"]
        mask = fin["mask"][:]
        n_events, n_slots, width = joined_in.shape
        if width != NATIVE_DIM + PART_DIM:
            print(f"FATAL: expected joined width {NATIVE_DIM + PART_DIM}, got {width}", file=sys.stderr)
            sys.exit(4)
        if fnat["INPUTS/Source/pt"].shape[0] != n_events:
            print(f"FATAL: native-h5 event count {fnat['INPUTS/Source/pt'].shape[0]} != joined event count {n_events}", file=sys.stderr)
            sys.exit(4)

        with h5py.File(args.out_h5, "w") as fout:
            # 1. Native features -- copied unchanged, one named (N,10) dataset per key.
            for i, name in enumerate(NATIVE_FEATURE_NAMES):
                ds = fout.create_dataset(f"INPUTS/Source/{name}", shape=(n_events, n_slots), dtype=np.float32,
                                          chunks=(min(4096, n_events), n_slots))
                for lo in range(0, n_events, args.chunk_events):
                    hi = min(lo + args.chunk_events, n_events)
                    ds[lo:hi] = joined_in[lo:hi, :, i]
            fout.create_dataset("INPUTS/Source/MASK", data=mask)

            # 2. Standardized ParT columns -- one named (N,10) dataset per retained dim.
            part_ds = {
                d: fout.create_dataset(f"INPUTS/Source/part_emb_{d:03d}", shape=(n_events, n_slots), dtype=np.float32,
                                        chunks=(min(4096, n_events), n_slots))
                for d in retained_dims
            }
            denom = np.maximum(train_stds[retained_dims], eps_used)
            for lo in range(0, n_events, args.chunk_events):
                hi = min(lo + args.chunk_events, n_events)
                part_chunk = joined_in[lo:hi, :, NATIVE_DIM:][:, :, retained_dims]  # (n, 10, n_retained)
                chunk_mask = mask[lo:hi]
                standardized = (part_chunk - train_means[retained_dims]) / denom
                standardized = np.where(chunk_mask[:, :, None], standardized, 0.0).astype(np.float32)
                n_nonfinite_written += int((~np.isfinite(standardized)).sum())
                for j, d in enumerate(retained_dims):
                    part_ds[d][lo:hi] = standardized[:, :, j]

            # 3. TARGETS/CLASSIFICATIONS -- copied verbatim from the ORIGINAL native H5
            #    (never present in the joined intermediate at all).
            for group in COPY_GROUPS_FROM_NATIVE:
                if group in fnat:
                    fnat.copy(group, fout)
                else:
                    print(f"WARNING: native H5 has no group '{group}' -- nothing copied for it", file=sys.stderr)

    retained_yaml_keys = build_event_yaml(args.event_yaml_template, retained_dims, args.out_yaml)

    receipt = {
        "variant": args.variant,
        "split": args.split,
        "input_joined_h5": args.joined_h5,
        "input_joined_h5_sha256": sha256_file(args.joined_h5),
        "input_native_h5": args.native_h5,
        "input_native_h5_sha256": sha256_file(args.native_h5),
        "train_stats_source": args.train_stats,
        "train_stats_source_sha256": sha256_file(args.train_stats),
        "train_stats_computed_from_train_only": True,
        "active_threshold_used": args.active_threshold if args.variant == "active" else None,
        "eps_used": eps_used,
        "n_retained_part_dims": n_retained,
        "retained_dims": retained_dims,
        "n_events": n_events,
        "output_width_features_per_jet": NATIVE_DIM + n_retained,
        "output_feature_names": NATIVE_FEATURE_NAMES + [f"part_emb_{d:03d}" for d in retained_dims],
        "output_h5": args.out_h5,
        "output_h5_sha256": sha256_file(args.out_h5),
        "output_event_yaml": args.out_yaml,
        "output_event_yaml_keys": retained_yaml_keys,
        "copied_groups_from_native": COPY_GROUPS_FROM_NATIVE,
        "n_nonfinite_in_output": n_nonfinite_written,
        "all_finite": n_nonfinite_written == 0,
        "note_masked_slots": "masked (padding) jet slots written as exact 0.0 in every appended ParT column, matching native zero-padding",
    }
    with open(args.out_receipt, "w") as f:
        json.dump(receipt, f, indent=2)

    print(json.dumps({k: v for k, v in receipt.items() if k not in ("retained_dims", "output_feature_names")}, indent=2))
    if n_nonfinite_written:
        print(f"WARNING: {n_nonfinite_written} non-finite values written to output", file=sys.stderr)
        sys.exit(5)


if __name__ == "__main__":
    main()
