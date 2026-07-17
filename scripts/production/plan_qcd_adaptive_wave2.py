#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PTHAT_BINS = [
    (50, 75),
    (75, 100),
    (100, 200),
    (200, 300),
    (300, 500),
    (500, 700),
    (700, 1000),
    (1000, 0),
]

MAX_EVENTS_PER_JOB = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan but do not submit the second adaptive "
            "physical-QCD campaign."
        )
    )

    parser.add_argument("--campaign", required=True)
    parser.add_argument("--allocation-json", type=Path, required=True)
    parser.add_argument("--previous-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument(
        "--split-salt",
        default="qcd-adaptive-wave2-whole-shard-v2",
    )

    return parser.parse_args()


def hash_fraction(
    split_salt: str,
    campaign: str,
    bin_id: int,
    shard_id: int,
) -> float:
    payload = (
        f"{split_salt}\0{campaign}\0{bin_id}\0{shard_id}"
    ).encode()

    value = int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        "big",
    )

    return value / 2**64


def deterministic_split(
    split_salt: str,
    campaign: str,
    bin_id: int,
    shard_id: int,
) -> str:
    fraction = hash_fraction(
        split_salt,
        campaign,
        bin_id,
        shard_id,
    )

    if fraction < 0.70:
        return "train"

    if fraction < 0.85:
        return "validation"

    return "test"


def deterministic_train_validation_split(
    split_salt: str,
    campaign: str,
    bin_id: int,
    shard_id: int,
) -> str:
    fraction = hash_fraction(
        split_salt + "-non-test",
        campaign,
        bin_id,
        shard_id,
    )

    # Preserve the original 70:15 ratio after removing test.
    return (
        "train"
        if fraction < 70.0 / 85.0
        else "validation"
    )


def load_prior_identities(
    repo: Path,
    store: Path,
) -> tuple[set[int], set[int]]:
    seeds: set[int] = set()
    shards: set[int] = set()

    roots = [
        store / "condor_submit",
        repo / "outputs/agent_runs",
        repo / "metadata/delphes",
    ]

    for root in roots:
        if not root.is_dir():
            continue

        for path in root.rglob("*manifest.csv"):
            try:
                with path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
            except (OSError, csv.Error, UnicodeDecodeError):
                continue

            for row in rows:
                try:
                    if row.get("seed") not in (None, ""):
                        seeds.add(int(row["seed"]))

                    if row.get("shard_id") not in (None, ""):
                        shards.add(int(row["shard_id"]))
                except ValueError as error:
                    raise SystemExit(
                        f"ERROR: invalid identity in {path}"
                    ) from error

    receipt_root = store / "condor_return"

    if receipt_root.is_dir():
        for path in receipt_root.rglob("*_receipt.json"):
            try:
                receipt = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue

            seed = receipt.get("seed")
            shard = receipt.get("shard_id")

            if type(seed) is int:
                seeds.add(seed)

            if type(shard) is int:
                shards.add(shard)

    return seeds, shards


def choose_shard_base(
    campaign: str,
    jobs: int,
    prior_seeds: set[int],
    prior_shards: set[int],
) -> int:
    lower = 100
    highest_base = 9_999 - jobs
    width = highest_base - lower + 1

    start = (
        lower
        + int(hashlib.sha256(campaign.encode()).hexdigest(), 16)
        % width
    )

    for offset in range(width):
        base = lower + ((start - lower + offset) % width)

        candidate_shards = set(range(base, base + jobs))

        if candidate_shards & prior_shards:
            continue

        candidate_seeds = {
            1_200_000 + bin_id * 10_000 + shard_id
            for bin_id in range(8)
            for shard_id in candidate_shards
        }

        if candidate_seeds & prior_seeds:
            continue

        return base

    raise SystemExit(
        "ERROR: no collision-free shard range available"
    )


def main() -> None:
    args = parse_args()

    repo = args.repo.resolve()
    store = args.store.resolve()
    allocation_path = args.allocation_json.resolve()
    previous_manifest_path = (
        args.previous_manifest.resolve()
    )
    output_dir = args.output_dir.resolve()

    allocation_document = json.loads(
        allocation_path.read_text()
    )

    assert (
        allocation_document["status"]
        == "proposal_not_authorized_for_submission"
    )
    assert not allocation_document["submission_authorized"]

    allocation = {
        int(key): int(value)
        for key, value
        in allocation_document["events_by_bin"].items()
    }

    required_test_bins = {
        int(value)
        for value in allocation_document[
            "required_new_sealed_test_bins"
        ]
    }

    assert sorted(allocation) == list(range(8))
    assert sum(allocation.values()) == 500_000
    assert required_test_bins == {1, 4, 5, 6, 7}

    with previous_manifest_path.open(newline="") as handle:
        previous_rows = list(csv.DictReader(handle))

    previous_test_bins = {
        int(row["bin_id"])
        for row in previous_rows
        if row["dataset_split"] == "test"
    }

    assert previous_test_bins == {0, 2, 3}, (
        f"Unexpected previous test bins: "
        f"{sorted(previous_test_bins)}"
    )

    jobs_per_bin = {
        bin_id: math.ceil(
            allocation[bin_id] / MAX_EVENTS_PER_JOB
        )
        for bin_id in range(8)
    }

    total_jobs = sum(jobs_per_bin.values())

    prior_seeds, prior_shards = load_prior_identities(
        repo,
        store,
    )

    shard_base = choose_shard_base(
        args.campaign,
        total_jobs,
        prior_seeds,
        prior_shards,
    )

    jobs: list[dict[str, Any]] = []
    job_id = 0

    for bin_id in range(8):
        remaining = allocation[bin_id]
        local_index = 0

        while remaining > 0:
            n_events = min(
                remaining,
                MAX_EVENTS_PER_JOB,
            )

            shard_id = shard_base + job_id
            seed = (
                1_200_000
                + bin_id * 10_000
                + shard_id
            )

            split = deterministic_split(
                args.split_salt,
                args.campaign,
                bin_id,
                shard_id,
            )

            reason = "deterministic_hash_70_15_15"

            if (
                bin_id in required_test_bins
                and local_index == 0
            ):
                split = "test"
                reason = (
                    "predeclared_sealed_test_for_missing_stratum"
                )

            jobs.append(
                {
                    "campaign": args.campaign,
                    "job_id": job_id,
                    "bin_id": bin_id,
                    "pthat_min_GeV": PTHAT_BINS[bin_id][0],
                    "pthat_max_GeV": PTHAT_BINS[bin_id][1],
                    "n_events": n_events,
                    "shard_id": shard_id,
                    "seed": seed,
                    "dataset_split": split,
                    "split_assignment_reason": reason,
                    "split_salt": args.split_salt,
                }
            )

            remaining -= n_events
            local_index += 1
            job_id += 1

    # Every bin must retain at least one non-test job.
    for bin_id in range(8):
        group = [
            job
            for job in jobs
            if job["bin_id"] == bin_id
        ]

        if all(
            job["dataset_split"] == "test"
            for job in group
        ):
            candidates = [
                job
                for job in group
                if job["split_assignment_reason"]
                != "predeclared_sealed_test_for_missing_stratum"
            ]

            assert candidates, (
                f"Bin {bin_id} has no non-forced shard "
                "available for training or validation"
            )

            selected = candidates[-1]

            selected["dataset_split"] = (
                deterministic_train_validation_split(
                    args.split_salt,
                    args.campaign,
                    bin_id,
                    int(selected["shard_id"]),
                )
            )

            selected["split_assignment_reason"] = (
                "deterministic_non_test_coverage_guard"
            )

    assert len(jobs) == total_jobs
    assert sum(job["n_events"] for job in jobs) == 500_000

    seeds = [int(job["seed"]) for job in jobs]
    shards = [int(job["shard_id"]) for job in jobs]

    assert len(seeds) == len(set(seeds))
    assert len(shards) == len(set(shards))
    assert not (set(seeds) & prior_seeds)
    assert not (set(shards) & prior_shards)

    forced_test_bins = {
        int(job["bin_id"])
        for job in jobs
        if job["split_assignment_reason"]
        == "predeclared_sealed_test_for_missing_stratum"
    }

    assert forced_test_bins == required_test_bins

    new_test_bins = {
        int(job["bin_id"])
        for job in jobs
        if job["dataset_split"] == "test"
    }

    combined_test_bins = (
        previous_test_bins | new_test_bins
    )

    assert combined_test_bins == set(range(8))

    split_jobs = Counter(
        str(job["dataset_split"])
        for job in jobs
    )

    split_events: dict[str, int] = defaultdict(int)

    for job in jobs:
        split_events[str(job["dataset_split"])] += int(
            job["n_events"]
        )

    events_by_bin: dict[int, int] = defaultdict(int)
    jobs_by_bin: Counter[int] = Counter()

    for job in jobs:
        bin_id = int(job["bin_id"])
        events_by_bin[bin_id] += int(job["n_events"])
        jobs_by_bin[bin_id] += 1

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = (
        output_dir / f"{args.campaign}_proposal_manifest.csv"
    )
    summary_path = (
        output_dir / f"{args.campaign}_proposal_summary.json"
    )

    fields = list(jobs[0].keys())

    with manifest_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(jobs)

    summary = {
        "schema_version": 1,
        "status": "proposal_not_authorized_for_submission",
        "campaign": args.campaign,
        "allocation_source": str(allocation_path),
        "previous_manifest": str(
            previous_manifest_path
        ),
        "total_jobs": len(jobs),
        "total_events": 500_000,
        "shard_id_base": shard_base,
        "jobs_by_bin": {
            str(key): jobs_by_bin[key]
            for key in range(8)
        },
        "events_by_bin": {
            str(key): events_by_bin[key]
            for key in range(8)
        },
        "split_job_counts": dict(split_jobs),
        "split_event_counts": dict(split_events),
        "previous_test_bins": sorted(
            previous_test_bins
        ),
        "new_test_bins": sorted(new_test_bins),
        "combined_test_bins": sorted(
            combined_test_bins
        ),
        "forced_sealed_test_bins": sorted(
            forced_test_bins
        ),
        "forced_test_assignments_fixed_before_generation": True,
        "allocation_used_final_test": False,
        "seed_reuse_count": 0,
        "shard_reuse_count": 0,
        "submission_authorized": False,
        "five_million_authorized": False,
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print("QCD_WAVE2_PROPOSAL_VALID")
    print("jobs:", len(jobs))
    print("events:", sum(events_by_bin.values()))
    print("jobs by bin:", dict(jobs_by_bin))
    print("events by bin:", dict(events_by_bin))
    print("split jobs:", dict(split_jobs))
    print("split events:", dict(split_events))
    print(
        "combined test bins:",
        sorted(combined_test_bins),
    )
    print("manifest:", manifest_path)
    print("summary:", summary_path)
    print("NO_SUBMISSION_CREATED")


if __name__ == "__main__":
    main()
