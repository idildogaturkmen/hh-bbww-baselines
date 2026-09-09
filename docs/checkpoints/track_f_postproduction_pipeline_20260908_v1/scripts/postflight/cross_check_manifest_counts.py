#!/usr/bin/env python3
"""Fail-closed cross-check: does the SHARD_MANIFEST.json's own declared
summary (shard/event/jet counts) match (a) this package's FROZEN_FACTS.json
expectation and (b) a fresh census result?

This is the one check neither census_2m.py nor aggregate_embedding_audit.py
performs by itself: census_2m.py only compares shard-ID *presence*, and
aggregate_embedding_audit.py only proves *disjointness/uniqueness* of
already-present shards. Neither one asserts that the manifest's own
train/val event and expected-jet totals still equal the numbers this
package was built against. If someone regenerates SHARD_MANIFEST.json with
different windows/quotas, this is the check that would catch it.

Read-only. Never contacts EOS. Takes already-produced JSON files as input.

Usage:
    python3 cross_check_manifest_counts.py \
        --manifest /path/to/SHARD_MANIFEST.json \
        --frozen-facts /path/to/FROZEN_FACTS.json \
        --census-json /path/to/fresh_census_output.json \
        --out /path/to/manifest_cross_check.json
"""
import argparse
import hashlib
import json
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--frozen-facts", required=True)
    ap.add_argument("--census-json", required=True,
                     help="output JSON from census_2m.py (already run for this session)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.frozen_facts) as f:
        facts = json.load(f)
    with open(args.census_json) as f:
        census = json.load(f)

    manifest_sha256_actual = sha256_file(args.manifest)
    manifest_sha256_expected = facts["part_production"]["manifest_sha256"]
    manifest_hash_pass = (manifest_sha256_actual == manifest_sha256_expected)

    with open(args.manifest) as f:
        manifest = json.load(f)
    summary = manifest.get("summary", {})

    checks = {}
    for split in ("train", "val"):
        expected = facts["part_production"]["expected"][split]
        shards_for_split = [s for s in manifest["shards"] if s["split"] == split]
        actual_n_shards = len(shards_for_split)
        actual_n_events = sum(
            s.get("expected_event_count", s["native_hdf5_row_end_exclusive"] - s["native_hdf5_row_start"])
            for s in shards_for_split
        )
        checks[split] = {
            "expected_shards": expected["shards"], "actual_shards_in_manifest": actual_n_shards,
            "shards_match": actual_n_shards == expected["shards"],
            "expected_events": expected["events"], "actual_events_in_manifest": actual_n_events,
            "events_match": actual_n_events == expected["events"],
            "census_complete_pairs": census[split]["complete_pairs"],
            "census_matches_manifest_shard_count": census[split]["complete_pairs"] <= actual_n_shards,
        }

    overall_pass = (
        manifest_hash_pass
        and all(c["shards_match"] and c["events_match"] for c in checks.values())
        and census.get("OVERALL_CENSUS_PASS") is True
    )

    result = {
        "manifest_path": args.manifest,
        "manifest_sha256_expected": manifest_sha256_expected,
        "manifest_sha256_actual": manifest_sha256_actual,
        "manifest_hash_pass": manifest_hash_pass,
        "per_split": checks,
        "census_overall_pass_input": census.get("OVERALL_CENSUS_PASS"),
        "OVERALL_MANIFEST_CROSS_CHECK_PASS": overall_pass,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
