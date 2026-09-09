#!/usr/bin/env python3
"""Fail-closed checksum verification for ParT production shards.

Neither census_2m.py (ID presence only) nor aggregate_embedding_audit.py
(disjointness/uniqueness only) re-hashes shard file *content* against the
receipt's own output_sha256. This script does exactly that -- the same
discipline the EAF-side check_shard_complete.py already applies per-shard
during production, run here independently, after the fact, from a
read-only vantage point.

Two modes:
  --sample-only N   fetch+hash a random sample of N complete shards per
                     split (fast, xrdcp over the network) -- a spot check,
                     not a full guarantee.
  --full            fetch+hash EVERY complete shard in both splits (slow:
                     ~1100 xrdcp calls at full production scale; run this
                     once, right before freezing hashes for training
                     launch, not routinely).

Never writes to EOS. Downloads to a local temp file which is deleted
immediately after hashing, one shard at a time (bounded disk/memory).

Usage:
    python3 verify_shard_checksums.py \
        --manifest SHARD_MANIFEST.json --split train \
        --eos-root root://cmseos.fnal.gov//store/.../shards \
        --sample-only 20 --out checksum_spot_check_train.json
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def xrdcp_to_tmp(remote_path, tmp_dir):
    local_path = os.path.join(tmp_dir, os.path.basename(remote_path))
    subprocess.run(["xrdcp", "-f", "-s", remote_path, local_path], check=True,
                    capture_output=True, text=True, timeout=300)
    return local_path


def fetch_receipt(eos_root, split, shard_id, tmp_dir):
    remote = f"{eos_root}/{split}/{shard_id}.receipt.json"
    local = xrdcp_to_tmp(remote, tmp_dir)
    with open(local) as f:
        receipt = json.load(f)
    os.remove(local)
    return receipt


def check_one_shard(eos_root, split, shard_id, tmp_dir):
    receipt = fetch_receipt(eos_root, split, shard_id, tmp_dir)
    expected_sha = receipt.get("output_sha256")
    remote_h5 = f"{eos_root}/{split}/{shard_id}.h5"
    local_h5 = xrdcp_to_tmp(remote_h5, tmp_dir)
    try:
        actual_sha = sha256_file(local_h5)
    finally:
        os.remove(local_h5)
    return {
        "shard_id": shard_id,
        "receipt_status": receipt.get("status"),
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "pass": (expected_sha == actual_sha) and receipt.get("status") == "complete",
        "all_finite_per_receipt": receipt.get("all_finite"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--eos-root", required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample-only", type=int, help="spot-check N random complete shards")
    group.add_argument("--full", action="store_true", help="check every complete shard (slow)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    all_ids = sorted(s["shard_id"] for s in manifest["shards"] if s["split"] == args.split)

    if args.sample_only:
        rng = random.Random(args.seed)
        ids_to_check = sorted(rng.sample(all_ids, min(args.sample_only, len(all_ids))))
        mode = f"sample_only_n={args.sample_only}_seed={args.seed}"
    else:
        ids_to_check = all_ids
        mode = "full"

    results = []
    with tempfile.TemporaryDirectory(prefix="checksum_verify_") as tmp_dir:
        for shard_id in ids_to_check:
            try:
                results.append(check_one_shard(args.eos_root, args.split, shard_id, tmp_dir))
            except Exception as e:  # noqa: BLE001 -- record and keep going; any failure fails the gate
                results.append({"shard_id": shard_id, "pass": False, "error": repr(e)})

    n_pass = sum(1 for r in results if r.get("pass"))
    overall = {
        "split": args.split, "mode": mode, "n_checked": len(results), "n_pass": n_pass,
        "n_fail": len(results) - n_pass,
        "OVERALL_CHECKSUM_PASS": (n_pass == len(results) and len(results) > 0),
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(overall, f, indent=2)
    print(f"{args.split}: {n_pass}/{len(results)} checksum-verified ({mode})")
    print(f"OVERALL_CHECKSUM_PASS = {overall['OVERALL_CHECKSUM_PASS']}")
    sys.exit(0 if overall["OVERALL_CHECKSUM_PASS"] else 1)


if __name__ == "__main__":
    main()
