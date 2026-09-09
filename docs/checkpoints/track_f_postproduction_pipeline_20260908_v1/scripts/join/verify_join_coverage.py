#!/usr/bin/env python3
"""Fail-closed post-join verification.

streaming_native_part_join.py (reused unchanged from
track_b_part_postproduction_tools_20260826_v2/code/) will happily produce a
PARTIAL join if some shards aren't available locally yet -- that's
intentional for its own development/testing use, but it means "the join
script ran with exit 0" is NOT sufficient evidence the joined output is
usable for training. This script is the explicit gate on top of it:

  - 100% shard coverage for the split (n_shards_covered_locally ==
    n_shards_in_manifest, and that count matches FROZEN_FACTS.json's
    expected shard count for the split)
  - n_real_jets_covered == n_real_jets_expected_from_native_mask (exact,
    zero tolerance)
  - re-open the joined HDF5 itself and independently recompute
    mask.sum() and part_coverage_mask.sum() (native h5py reduction, not
    trusting the join script's own self-reported stats twice removed)
  - assert every row in [0, n_events) has mask -> part_coverage_mask
    implication verified in both directions where the split is claimed
    fully covered (mask==True implies part_coverage_mask==True; the
    converse is already enforced inside join_one_shard as a hard assert)

Writes ONE machine-readable join receipt. Exits nonzero if anything here
does not hold -- this is the "produce a machine-readable join receipt"
deliverable for the exact event-level JOIN package.

Usage:
    python3 verify_join_coverage.py \
        --joined-h5 joined_train.h5 --coverage-json joined_train.h5.coverage.json \
        --frozen-facts FROZEN_FACTS.json --split train \
        --out JOIN_RECEIPT_train.json
"""
import argparse
import hashlib
import json
import sys

import h5py
import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined-h5", required=True)
    ap.add_argument("--coverage-json", required=True,
                     help="the .coverage.json sidecar written by streaming_native_part_join.py")
    ap.add_argument("--frozen-facts", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.coverage_json) as f:
        cov = json.load(f)
    with open(args.frozen_facts) as f:
        facts = json.load(f)

    expected = facts["part_production"]["expected"][args.split]
    checks = {}

    checks["shard_count_full"] = (cov["n_shards_covered_locally"] == cov["n_shards_in_manifest"] == expected["shards"])
    checks["event_count_matches_frozen_facts"] = (cov["n_events_total_split"] == expected["events"])
    checks["real_jet_count_matches_frozen_facts"] = (cov["n_real_jets_expected_from_native_mask"] == expected["real_jets"])
    checks["real_jets_fully_covered_per_join_script"] = (cov["n_real_jets_covered"] == cov["n_real_jets_expected_from_native_mask"])
    checks["no_partial_shards_reported"] = all(
        s.get("status") == "joined" for s in cov.get("per_shard", [])
    ) if cov.get("per_shard") else False

    with h5py.File(args.joined_h5, "r") as f:
        mask = f["mask"][:]
        covered = f["part_coverage_mask"][:]
        joined = f["joined_features"]
        n_events, n_slots, width = joined.shape
        emb_slice = joined[: min(2048, n_events), :, 7:]  # spot-check a bounded slice, never the whole tensor
        finite_spot_check = bool(np.isfinite(emb_slice).all())

    checks["reopened_mask_sum_matches_expected_real_jets"] = (int(mask.sum()) == expected["real_jets"])
    checks["reopened_coverage_sum_matches_expected_real_jets"] = (int(covered.sum()) == expected["real_jets"])
    checks["mask_true_implies_covered_true_everywhere"] = bool(np.all(covered[mask]))
    checks["joined_tensor_shape_matches_events_and_slots"] = (n_events == expected["events"] and n_slots == 10)
    checks["spot_checked_embedding_columns_finite"] = finite_spot_check

    overall_pass = all(checks.values())

    receipt = {
        "split": args.split,
        "joined_h5_path": args.joined_h5,
        "joined_h5_sha256": sha256_file(args.joined_h5),
        "joined_tensor_shape": [n_events, n_slots, width],
        "checks": checks,
        "coverage_json_input": cov,
        "OVERALL_JOIN_VERIFICATION_PASS": overall_pass,
    }
    with open(args.out, "w") as f:
        json.dump(receipt, f, indent=2)

    print(json.dumps({k: v for k, v in receipt.items() if k not in ("coverage_json_input",)}, indent=2))
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
