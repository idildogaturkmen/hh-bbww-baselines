#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


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

REGIONS = [
    "hh_candidate",
    "rhh_lt80",
    "rhh_lt50",
]

MAX_EVENTS_PER_JOB = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan a train+validation-only adaptive physical-QCD "
            "checkpoint without submitting jobs."
        )
    )

    parser.add_argument("--split-summary", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument(
        "--total-events",
        type=int,
        default=1_500_000,
    )
    parser.add_argument(
        "--minimum-per-bin",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--split-salt",
        default="qcd-adaptive-wave3-trainval-v1",
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def finite_float(value: Any, default: float = 0.0) -> float:
    number = float(value)

    if not math.isfinite(number):
        return default

    return number


def weighted_average(
    values: list[float],
    weights: list[int],
) -> float:
    selected = [
        (float(value), int(weight))
        for value, weight in zip(values, weights)
        if math.isfinite(float(value)) and int(weight) > 0
    ]

    if not selected:
        return 0.0

    denominator = sum(weight for _, weight in selected)

    return sum(
        value * weight
        for value, weight in selected
    ) / denominator


def hash_fraction(
    split_salt: str,
    campaign: str,
    bin_id: int,
    shard_id: int,
) -> float:
    payload = (
        f"{split_salt}\0{campaign}\0"
        f"{bin_id}\0{shard_id}"
    ).encode()

    integer = int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        "big",
    )

    return integer / 2**64


def deterministic_trainval_split(
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

    # Preserve the original train:validation ratio of 70:15
    # after deliberately excluding additional test shards.
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
        repo / "metadata/delphes",
        repo / "outputs/agent_runs",
        store / "condor_submit",
    ]

    for root in roots:
        if not root.is_dir():
            continue

        for path in root.rglob("*manifest.csv"):
            try:
                with path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
            except (
                OSError,
                csv.Error,
                UnicodeDecodeError,
            ):
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
        + int(
            hashlib.sha256(
                campaign.encode()
            ).hexdigest(),
            16,
        )
        % width
    )

    for offset in range(width):
        base = (
            lower
            + (
                start
                - lower
                + offset
            )
            % width
        )

        candidate_shards = set(
            range(base, base + jobs)
        )

        if candidate_shards & prior_shards:
            continue

        candidate_seeds = {
            1_200_000
            + bin_id * 10_000
            + shard_id
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

    if args.total_events <= 0:
        raise SystemExit(
            "ERROR: total events must be positive"
        )

    if args.total_events % MAX_EVENTS_PER_JOB:
        raise SystemExit(
            "ERROR: total events must be divisible by 10000"
        )

    if args.minimum_per_bin <= 0:
        raise SystemExit(
            "ERROR: minimum per bin must be positive"
        )

    if args.minimum_per_bin % MAX_EVENTS_PER_JOB:
        raise SystemExit(
            "ERROR: minimum per bin must be divisible by 10000"
        )

    frame = pd.read_csv(args.split_summary)

    required_columns = {
        "dataset_split",
        "bin_id",
        "region",
        "n_generated",
        "n_selected",
        "sigma_gen_pb_event_count_weighted",
        "weighted_yield_pb",
        "conditional_mc_variance_pb2",
    }

    missing = required_columns - set(frame.columns)

    if missing:
        raise SystemExit(
            "ERROR: split summary lacks columns: "
            f"{sorted(missing)}"
        )

    # Final test is excluded from all allocation calculations.
    frame = frame[
        frame["dataset_split"].isin(
            ["train", "validation"]
        )
    ].copy()

    if set(frame["bin_id"].astype(int)) != set(range(8)):
        raise SystemExit(
            "ERROR: train+validation data lack a pThat bin"
        )

    combined: dict[
        tuple[int, str],
        dict[str, float | int],
    ] = {}

    for bin_id in range(8):
        for region in REGIONS:
            rows = frame[
                (
                    frame["bin_id"].astype(int)
                    == bin_id
                )
                & (frame["region"] == region)
            ].copy()

            # Some bins have only train or only validation.
            # One or two rows are both valid.
            if len(rows) not in {1, 2}:
                raise SystemExit(
                    "ERROR: unexpected train/validation "
                    f"coverage for bin={bin_id}, "
                    f"region={region}: rows={len(rows)}"
                )

            generated = [
                int(value)
                for value in rows["n_generated"]
            ]

            selected = [
                int(value)
                for value in rows["n_selected"]
            ]

            n_generated = sum(generated)
            n_selected = sum(selected)

            if n_generated <= 0:
                raise SystemExit(
                    f"ERROR: empty bin {bin_id}"
                )

            sigma_pb = weighted_average(
                [
                    finite_float(value)
                    for value in rows[
                        "sigma_gen_pb_event_count_weighted"
                    ]
                ],
                generated,
            )

            observed_yield = weighted_average(
                [
                    finite_float(value)
                    for value in rows[
                        "weighted_yield_pb"
                    ]
                ],
                generated,
            )

            variance_coefficients = [
                finite_float(variance)
                * n_events
                for variance, n_events in zip(
                    rows[
                        "conditional_mc_variance_pb2"
                    ],
                    generated,
                )
            ]

            observed_coefficient = weighted_average(
                variance_coefficients,
                generated,
            )

            # Jeffreys regularization prevents a zero-survivor
            # bin from being interpreted as zero variance.
            smoothed_probability = (
                n_selected + 0.5
            ) / (
                n_generated + 1.0
            )

            bernoulli_floor = (
                sigma_pb**2
                * smoothed_probability
                * (1.0 - smoothed_probability)
            )

            variance_coefficient = max(
                observed_coefficient,
                bernoulli_floor,
            )

            yield_proxy = max(
                observed_yield,
                sigma_pb * smoothed_probability,
            )

            combined[(bin_id, region)] = {
                "n_generated": n_generated,
                "n_selected": n_selected,
                "sigma_pb": sigma_pb,
                "observed_yield_pb": observed_yield,
                "smoothed_probability": (
                    smoothed_probability
                ),
                "yield_proxy_pb": yield_proxy,
                "observed_variance_coefficient": (
                    observed_coefficient
                ),
                "bernoulli_coefficient_floor": (
                    bernoulli_floor
                ),
                "variance_coefficient": (
                    variance_coefficient
                ),
            }

    current_events = {
        bin_id: int(
            combined[
                (bin_id, "hh_candidate")
            ]["n_generated"]
        )
        for bin_id in range(8)
    }

    total_yield_proxies = {
        region: sum(
            float(
                combined[
                    (bin_id, region)
                ]["yield_proxy_pb"]
            )
            for bin_id in range(8)
        )
        for region in REGIONS
    }

    if not all(
        value > 0
        for value in total_yield_proxies.values()
    ):
        raise SystemExit(
            "ERROR: nonpositive yield proxy"
        )

    objective_coefficient: dict[
        int,
        float,
    ] = defaultdict(float)

    for bin_id in range(8):
        for region in REGIONS:
            coefficient = float(
                combined[
                    (bin_id, region)
                ]["variance_coefficient"]
            )

            objective_coefficient[bin_id] += (
                coefficient
                / total_yield_proxies[region] ** 2
                / len(REGIONS)
            )

    allocation = {
        bin_id: args.minimum_per_bin
        for bin_id in range(8)
    }

    remaining = (
        args.total_events
        - sum(allocation.values())
    )

    if remaining < 0:
        raise SystemExit(
            "ERROR: minimum allocation exceeds total"
        )

    while remaining > 0:
        best_bin = None
        best_improvement = -1.0

        for bin_id in range(8):
            before = (
                current_events[bin_id]
                + allocation[bin_id]
            )

            after = (
                before
                + MAX_EVENTS_PER_JOB
            )

            coefficient = objective_coefficient[
                bin_id
            ]

            improvement = (
                coefficient / before
                - coefficient / after
            )

            if improvement > best_improvement:
                best_improvement = improvement
                best_bin = bin_id

        if best_bin is None:
            raise SystemExit(
                "ERROR: allocation optimization failed"
            )

        allocation[best_bin] += MAX_EVENTS_PER_JOB
        remaining -= MAX_EVENTS_PER_JOB

    if sum(allocation.values()) != args.total_events:
        raise SystemExit(
            "ERROR: allocation total mismatch"
        )

    projected_regions = {}

    for region in REGIONS:
        projected_variance = sum(
            float(
                combined[
                    (bin_id, region)
                ]["variance_coefficient"]
            )
            / (
                current_events[bin_id]
                + allocation[bin_id]
            )
            for bin_id in range(8)
        )

        projected_error = math.sqrt(
            max(projected_variance, 0.0)
        )

        projected_regions[region] = {
            "trainval_yield_proxy_pb": (
                total_yield_proxies[region]
            ),
            "projected_standard_error_pb": (
                projected_error
            ),
            "projected_relative_standard_error": (
                projected_error
                / total_yield_proxies[region]
            ),
            "expected_additional_raw_selected": sum(
                allocation[bin_id]
                * float(
                    combined[
                        (bin_id, region)
                    ]["smoothed_probability"]
                )
                for bin_id in range(8)
            ),
        }

    with args.registry.open(newline="") as handle:
        registry_rows = list(
            csv.DictReader(handle)
        )

    if not registry_rows:
        raise SystemExit(
            "ERROR: canonical registry is empty"
        )

    existing_test_bins = {
        int(row["bin_id"])
        for row in registry_rows
        if row["dataset_split"] == "test"
    }

    if existing_test_bins != set(range(8)):
        raise SystemExit(
            "ERROR: canonical sealed test lacks a bin"
        )

    prior_seeds, prior_shards = (
        load_prior_identities(
            args.repo.resolve(),
            args.store.resolve(),
        )
    )

    total_jobs = (
        args.total_events
        // MAX_EVENTS_PER_JOB
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
        n_jobs = (
            allocation[bin_id]
            // MAX_EVENTS_PER_JOB
        )

        for local_index in range(n_jobs):
            shard_id = shard_base + job_id

            seed = (
                1_200_000
                + bin_id * 10_000
                + shard_id
            )

            split = deterministic_trainval_split(
                args.split_salt,
                args.campaign,
                bin_id,
                shard_id,
            )

            jobs.append(
                {
                    "campaign": args.campaign,
                    "job_id": job_id,
                    "bin_id": bin_id,
                    "pthat_min_GeV": (
                        PTHAT_BINS[bin_id][0]
                    ),
                    "pthat_max_GeV": (
                        PTHAT_BINS[bin_id][1]
                    ),
                    "n_events": (
                        MAX_EVENTS_PER_JOB
                    ),
                    "shard_id": shard_id,
                    "seed": seed,
                    "dataset_split": split,
                    "split_assignment_reason": (
                        "deterministic_trainval_hash"
                    ),
                    "split_salt": args.split_salt,
                }
            )

            job_id += 1

    # Coverage guard: a bin with one new job is assigned
    # to training. Bins with at least two jobs retain both
    # train and validation coverage.
    for bin_id in range(8):
        group = [
            job
            for job in jobs
            if job["bin_id"] == bin_id
        ]

        if not group:
            raise SystemExit(
                f"ERROR: bin {bin_id} has no jobs"
            )

        if len(group) == 1:
            group[0]["dataset_split"] = "train"
            group[0]["split_assignment_reason"] = (
                "single_shard_training_coverage_guard"
            )

        elif all(
            job["dataset_split"] == "train"
            for job in group
        ):
            group[-1]["dataset_split"] = "validation"
            group[-1]["split_assignment_reason"] = (
                "validation_coverage_guard"
            )

        elif all(
            job["dataset_split"] == "validation"
            for job in group
        ):
            group[-1]["dataset_split"] = "train"
            group[-1]["split_assignment_reason"] = (
                "training_coverage_guard"
            )

    if len(jobs) != total_jobs:
        raise SystemExit(
            "ERROR: job-count mismatch"
        )

    seeds = {
        int(job["seed"])
        for job in jobs
    }

    shards = {
        int(job["shard_id"])
        for job in jobs
    }

    if len(seeds) != len(jobs):
        raise SystemExit(
            "ERROR: duplicate new seeds"
        )

    if len(shards) != len(jobs):
        raise SystemExit(
            "ERROR: duplicate new shards"
        )

    if seeds & prior_seeds:
        raise SystemExit(
            "ERROR: seed collision"
        )

    if shards & prior_shards:
        raise SystemExit(
            "ERROR: shard collision"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    allocation_path = (
        args.output_dir
        / "trainval_only_allocation.json"
    )

    manifest_path = (
        args.output_dir
        / "reviewed_manifest.csv"
    )

    summary_path = (
        args.output_dir
        / "reviewed_summary.json"
    )

    allocation_document = {
        "schema_version": 1,
        "status": (
            "checkpoint_proposal_not_authorized_"
            "for_submission"
        ),
        "campaign": args.campaign,
        "allocation_uses_splits": [
            "train",
            "validation",
        ],
        "allocation_excludes_test": True,
        "total_new_events": args.total_events,
        "events_by_bin": {
            str(key): allocation[key]
            for key in range(8)
        },
        "current_trainval_events_by_bin": {
            str(key): current_events[key]
            for key in range(8)
        },
        "projected_regions": (
            projected_regions
        ),
        "full_remaining_3990000_authorized": False,
        "checkpoint_1500000_authorized": False,
        "submission_authorized": False,
    }

    allocation_path.write_text(
        json.dumps(
            allocation_document,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    with manifest_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(jobs[0]),
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(jobs)

    jobs_by_bin = Counter(
        int(job["bin_id"])
        for job in jobs
    )

    split_jobs = Counter(
        str(job["dataset_split"])
        for job in jobs
    )

    split_events: dict[str, int] = defaultdict(int)

    for job in jobs:
        split_events[
            str(job["dataset_split"])
        ] += int(job["n_events"])

    summary = {
        "schema_version": 1,
        "status": (
            "proposal_not_authorized_for_submission"
        ),
        "campaign": args.campaign,
        "total_jobs": len(jobs),
        "total_events": args.total_events,
        "events_by_bin": {
            str(key): allocation[key]
            for key in range(8)
        },
        "jobs_by_bin": {
            str(key): jobs_by_bin[key]
            for key in range(8)
        },
        "split_job_counts": dict(split_jobs),
        "split_event_counts": dict(split_events),
        "previous_test_bins": sorted(
            existing_test_bins
        ),
        "new_test_bins": [],
        "combined_test_bins": list(range(8)),
        "forced_sealed_test_bins": [],
        "allocation_used_final_test": False,
        "allocation_excludes_test": True,
        "shard_id_base": shard_base,
        "seed_reuse_count": 0,
        "shard_reuse_count": 0,
        "split_salt": args.split_salt,
        "allocation_file": str(
            allocation_path
        ),
        "allocation_sha256": sha256_file(
            allocation_path
        ),
        "source_split_summary": str(
            args.split_summary.resolve()
        ),
        "source_split_summary_sha256": (
            sha256_file(args.split_summary)
        ),
        "source_registry": str(
            args.registry.resolve()
        ),
        "source_registry_sha256": (
            sha256_file(args.registry)
        ),
        "submission_authorized": False,
        "checkpoint_1500000_authorized": False,
        "full_remaining_3990000_authorized": False,
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

    print(
        "QCD_WAVE3_CHECKPOINT_PROPOSAL_VALID"
    )

    print(
        "jobs:",
        len(jobs),
    )

    print(
        "events:",
        args.total_events,
    )

    print(
        "events by bin:",
        summary["events_by_bin"],
    )

    print(
        "split events:",
        summary["split_event_counts"],
    )

    print(
        "shard base:",
        shard_base,
    )

    print("projected regions:")

    for region, values in projected_regions.items():
        print(
            region,
            "relative_error=",
            f"{values['projected_relative_standard_error']:.4f}",
            "expected_new_raw=",
            f"{values['expected_additional_raw_selected']:.1f}",
        )

    print(
        "manifest:",
        manifest_path,
    )

    print(
        "summary:",
        summary_path,
    )

    print("NO_SUBMISSION_PERFORMED")


if __name__ == "__main__":
    main()
