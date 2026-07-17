#!/usr/bin/env python3
"""Prepare one fail-closed, allocation-driven QCD Condor campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
from typing import Any


EXPECTED_CARD_SHA256 = (
    "1b2041245162de8360defdc502404e0696496c50773d4aa7f043798651a9517c"
)
REQUIRED_ANCESTOR = "4ca7d754972b30dda56df302d7f9635b86f7565e"
EXPECTED_BRANCH = "delphes-hh4b-production"
EXPECTED_TOTAL = 500_000
MAX_EVENTS_PER_JOB = 10_000
MAX_PROJECTED_TIB = 0.10
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--pilot-summary-csv", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument(
        "--split-salt", default="qcd-adaptive-whole-shard-v1"
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(
            f"/uscms_data/d3/{os.environ.get('USER', '')}/repos/hh-bbww-baselines"
        ),
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=Path(
            f"/uscms_data/d3/{os.environ.get('USER', '')}/hh4b_delphes"
        ),
    )
    parser.add_argument(
        "--eos-base",
        default=(
            f"/store/user/{os.environ.get('USER', '')}/hh4b_delphes/"
            "run2_13tev/frozen_v2"
        ),
    )
    return parser.parse_args()


def run(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_payload_manifest(payload: Path) -> dict[str, str]:
    with tarfile.open(payload, "r:gz") as archive:
        member = archive.getmember("payload/manifest.txt")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise SystemExit("ERROR: payload manifest is unreadable")
        lines = extracted.read().decode("utf-8").splitlines()
    result: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if not separator or not key or key in result:
            raise SystemExit("ERROR: malformed payload manifest")
        result[key] = value
    return result


def deterministic_split(
    split_salt: str, campaign: str, bin_id: int, shard_id: int
) -> str:
    payload = f"{split_salt}\0{campaign}\0{bin_id}\0{shard_id}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    fraction = value / 2**64
    if fraction < 0.70:
        return "train"
    if fraction < 0.85:
        return "validation"
    return "test"


def load_prior_identities(repo: Path, store: Path) -> tuple[set[int], set[int], list[str]]:
    seeds: set[int] = set()
    shards: set[int] = set()
    sources: list[str] = []
    manifest_roots = [store / "condor_submit", repo / "outputs/agent_runs"]
    for root in manifest_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*manifest.csv"):
            try:
                with path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
            except (OSError, csv.Error, UnicodeDecodeError):
                continue
            if not rows or "seed" not in rows[0]:
                continue
            used = False
            for row in rows:
                try:
                    seeds.add(int(row["seed"]))
                    if row.get("shard_id") not in (None, ""):
                        shards.add(int(row["shard_id"]))
                    used = True
                except (TypeError, ValueError):
                    raise SystemExit(f"ERROR: invalid identity in prior manifest: {path}")
            if used:
                sources.append(str(path))
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
    return seeds, shards, sorted(sources)


def choose_shard_base(
    campaign: str,
    jobs: int,
    prior_seeds: set[int],
    prior_shards: set[int],
) -> int:
    lower = 100
    highest_base = 9_999 - jobs
    width = highest_base - lower + 1
    start = lower + int(hashlib.sha256(campaign.encode()).hexdigest(), 16) % width
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
    raise SystemExit("ERROR: no collision-free shard-ID range is available")


def load_allocation(path: Path) -> tuple[dict[int, int], dict[str, Any]]:
    document = json.loads(path.read_text())
    if document.get("status") != "pass":
        raise SystemExit("ERROR: source audit is not a production pass")
    roles = document.get("sample_roles", {})
    if roles.get("physics_weighted_inference") != (
        "inclusive pTHat-stratified HardQCD with physical cross-section weights"
    ):
        raise SystemExit("ERROR: source audit has the wrong physical sample role")
    if not roles.get("ml_enrichment_excluded_from_physical_qcd_normalization"):
        raise SystemExit("ERROR: ML-enrichment exclusion is not asserted")
    physics = document["adaptive_recommendations"]["physics"]
    allocation = {int(key): int(value) for key, value in physics["events_by_bin"].items()}
    if set(allocation) != set(range(8)):
        raise SystemExit("ERROR: physics allocation does not contain exactly bins 0--7")
    if sum(allocation.values()) != EXPECTED_TOTAL:
        raise SystemExit("ERROR: physics allocation does not sum to 500,000")
    if not physics.get("zero_candidate_bins_retained"):
        raise SystemExit("ERROR: zero-candidate bins are not retained")
    return allocation, document


def load_bytes_per_event(path: Path) -> dict[int, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {
        int(row["bin_id"]): float(row["bytes_per_generated_event"])
        for row in rows
    }
    if set(result) != set(range(8)) or any(
        not math.isfinite(value) or value <= 0 for value in result.values()
    ):
        raise SystemExit("ERROR: invalid per-bin storage measurements")
    return result


def ensure_eos_absent(eos_directory: str, proxy: Path) -> None:
    environment = os.environ.copy()
    environment["X509_USER_PROXY"] = str(proxy)
    result = subprocess.run(
        ["xrdfs", "root://cmseos.fnal.gov", "stat", eos_directory],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    if result.returncode == 0:
        raise SystemExit(f"ERROR: EOS campaign namespace already exists: {eos_directory}")
    diagnostic = result.stdout + result.stderr
    if "No such file or directory" not in diagnostic and "[3011]" not in diagnostic:
        raise SystemExit(f"ERROR: EOS namespace preflight failed: {diagnostic.strip()}")


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    store = args.store.resolve()
    audit_path = args.audit_json.resolve()
    pilot_summary_path = args.pilot_summary_csv.resolve()
    proxy = args.proxy.resolve()
    source_payload = args.payload.resolve()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]+", args.campaign):
        raise SystemExit("ERROR: invalid campaign name")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.split_salt):
        raise SystemExit("ERROR: invalid split salt")
    for required in (audit_path, pilot_summary_path, proxy, source_payload):
        if not required.is_file() or required.stat().st_size == 0:
            raise SystemExit(f"ERROR: missing input: {required}")

    branch = run(["git", "branch", "--show-current"], cwd=repo)
    if branch != EXPECTED_BRANCH:
        raise SystemExit(f"ERROR: current branch is {branch}, expected {EXPECTED_BRANCH}")
    if run(["git", "status", "--porcelain"], cwd=repo):
        raise SystemExit("ERROR: repository worktree or index is dirty")
    git_head = run(["git", "rev-parse", "HEAD"], cwd=repo)
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, git_head],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["openssl", "x509", "-in", str(proxy), "-noout", "-checkend", "3600"],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    allocation, audit = load_allocation(audit_path)
    bytes_per_event = load_bytes_per_event(pilot_summary_path)
    projected_bytes = sum(
        allocation[bin_id] * bytes_per_event[bin_id] for bin_id in range(8)
    )
    projected_tib = projected_bytes / 2**40
    if projected_tib >= MAX_PROJECTED_TIB:
        raise SystemExit(
            f"ERROR: projected compressed storage {projected_tib:.6f} TiB "
            f"is not below {MAX_PROJECTED_TIB:.2f} TiB"
        )

    jobs_per_bin = {
        bin_id: math.ceil(events / MAX_EVENTS_PER_JOB)
        for bin_id, events in allocation.items()
    }
    total_jobs = sum(jobs_per_bin.values())
    prior_seeds, prior_shards, identity_sources = load_prior_identities(repo, store)
    shard_base = choose_shard_base(
        args.campaign, total_jobs, prior_seeds, prior_shards
    )

    jobs: list[dict[str, Any]] = []
    job_id = 0
    for bin_id in range(8):
        remaining = allocation[bin_id]
        while remaining:
            n_events = min(remaining, MAX_EVENTS_PER_JOB)
            shard_id = shard_base + job_id
            seed = 1_200_000 + bin_id * 10_000 + shard_id
            dataset_split = deterministic_split(
                args.split_salt, args.campaign, bin_id, shard_id
            )
            jobs.append(
                {
                    "job_id": job_id,
                    "bin_id": bin_id,
                    "shard_id": shard_id,
                    "seed": seed,
                    "n_events": n_events,
                    "dataset_split": dataset_split,
                }
            )
            remaining -= n_events
            job_id += 1

    if len(jobs) != total_jobs or sum(job["n_events"] for job in jobs) != EXPECTED_TOTAL:
        raise SystemExit("ERROR: internal shard-planning mismatch")
    seeds = [job["seed"] for job in jobs]
    shards = [job["shard_id"] for job in jobs]
    if len(seeds) != len(set(seeds)) or set(seeds) & prior_seeds:
        raise SystemExit("ERROR: new seed collision")
    if len(shards) != len(set(shards)) or set(shards) & prior_shards:
        raise SystemExit("ERROR: new shard-ID collision")
    if any(job["n_events"] > MAX_EVENTS_PER_JOB for job in jobs):
        raise SystemExit("ERROR: a planned job exceeds 10,000 events")
    split_jobs = {
        split: sum(job["dataset_split"] == split for job in jobs)
        for split in ("train", "validation", "test")
    }
    split_events = {
        split: sum(
            job["n_events"]
            for job in jobs
            if job["dataset_split"] == split
        )
        for split in ("train", "validation", "test")
    }
    if any(split_jobs[split] == 0 or split_events[split] == 0 for split in split_jobs):
        raise SystemExit("ERROR: deterministic whole-shard split left a split empty")

    submit_dir = store / "condor_submit" / args.campaign
    log_dir = store / "condor_logs" / args.campaign
    return_dir = store / "condor_return" / args.campaign / "receipts"
    eos_directory = f"{args.eos_base}/bundles/{args.campaign}"
    for path in (submit_dir, log_dir, return_dir.parent):
        if path.exists():
            raise SystemExit(f"ERROR: refusing to reuse local campaign path: {path}")
    ensure_eos_absent(eos_directory, proxy)

    payload_sha256 = sha256_file(source_payload)
    payload_name = f"{args.campaign}_inputs_{payload_sha256[:16]}.tar.gz"
    wrapper = repo / "scripts/production/run_qcd_importance_bundle.sh"
    wrapper_sha256 = sha256_file(wrapper)
    payload_manifest = load_payload_manifest(source_payload)
    if payload_manifest.get("git_head") != git_head:
        raise SystemExit("ERROR: payload git head does not match current checkout")
    if payload_manifest.get("worker_wrapper_sha256") != wrapper_sha256:
        raise SystemExit("ERROR: payload and worker wrapper hashes disagree")
    card = repo / "cards/delphes/delphes_card_CMS_lpc_ak4ak8_run2_frozen_v2.tcl"
    card_sha256 = sha256_file(card)
    if card_sha256 != EXPECTED_CARD_SHA256:
        raise SystemExit("ERROR: frozen-v2 card hash mismatch")

    submit_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    return_dir.mkdir(parents=True)
    campaign_payload = submit_dir / payload_name
    shutil.copy2(source_payload, campaign_payload)
    if sha256_file(campaign_payload) != payload_sha256:
        raise SystemExit("ERROR: immutable payload copy hash mismatch")
    campaign_payload.chmod(0o444)

    submit_file = submit_dir / f"{args.campaign}.sub"
    manifest_file = submit_dir / f"{args.campaign}_manifest.csv"
    summary_file = submit_dir / f"{args.campaign}_summary.json"

    submit_file.write_text(
        "\n".join(
            [
                "universe = vanilla",
                f"executable = {wrapper}",
                (
                    "arguments = $(bin_id) $(shard_id) $(n_events) "
                    f"{args.campaign} $(Cluster) {payload_name} {eos_directory} "
                    f"{payload_sha256} $(dataset_split) {args.split_salt}"
                ),
                "",
                (
                    f"output = {log_dir}/{args.campaign}_$(Cluster)_"
                    "job$(job_id)_bin$(bin_id)_shard$(shard_id).out"
                ),
                (
                    f"error = {log_dir}/{args.campaign}_$(Cluster)_"
                    "job$(job_id)_bin$(bin_id)_shard$(shard_id).err"
                ),
                f"log = {log_dir}/{args.campaign}_$(Cluster).log",
                "",
                "should_transfer_files = YES",
                "when_to_transfer_output = ON_EXIT",
                f"transfer_input_files = {campaign_payload}",
                "transfer_output_files = job_receipt.json",
                (
                    "transfer_output_remaps = \"job_receipt.json = "
                    f"{return_dir}/{args.campaign}_$(Cluster)_job$(job_id)_"
                    "bin$(bin_id)_shard$(shard_id)_receipt.json\""
                ),
                "",
                "request_cpus = 1",
                "request_memory = 4GB",
                "request_disk = 20GB",
                "",
                "getenv = False",
                'environment = "LC_ALL=C LANG=C"',
                "use_x509userproxy = True",
                f"x509userproxy = {proxy}",
                "",
                f'+JobBatchName = "{args.campaign}"',
                '+DesiredOS = "EL9"',
                '+QCDDatasetSplit = "$(dataset_split)"',
                "on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)",
                "",
                "queue job_id,bin_id,shard_id,n_events,dataset_split from (",
                *[
                    (
                        f"{job['job_id']} {job['bin_id']} {job['shard_id']} "
                        f"{job['n_events']} {job['dataset_split']}"
                    )
                    for job in jobs
                ],
                ")",
                "",
            ]
        )
    )

    manifest_fields = [
        "campaign",
        "job_id",
        "bin_id",
        "pthat_min_GeV",
        "pthat_max_GeV",
        "n_events",
        "shard_id",
        "seed",
        "dataset_split",
        "split_salt",
        "payload_sha256",
        "wrapper_sha256",
        "card_sha256",
        "git_head",
        "eos_directory",
        "allocation_audit_sha256",
    ]
    audit_sha256 = sha256_file(audit_path)
    with manifest_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for job in jobs:
            pthat_min, pthat_max = PTHAT_BINS[job["bin_id"]]
            writer.writerow(
                {
                    "campaign": args.campaign,
                    **job,
                    "pthat_min_GeV": pthat_min,
                    "pthat_max_GeV": pthat_max,
                    "split_salt": args.split_salt,
                    "payload_sha256": payload_sha256,
                    "wrapper_sha256": wrapper_sha256,
                    "card_sha256": card_sha256,
                    "git_head": git_head,
                    "eos_directory": eos_directory,
                    "allocation_audit_sha256": audit_sha256,
                }
            )

    summary = {
        "schema_version": 1,
        "campaign": args.campaign,
        "authorization": "one physics/control-aware 500K wave; no 5M",
        "allocation_source": str(audit_path),
        "allocation_audit_sha256": audit_sha256,
        "allocation_objective": audit["adaptive_recommendations"]["physics"][
            "objective"
        ],
        "events_by_bin": {str(key): allocation[key] for key in range(8)},
        "jobs_by_bin": {str(key): jobs_per_bin[key] for key in range(8)},
        "total_events": EXPECTED_TOTAL,
        "total_jobs": total_jobs,
        "max_events_per_job": MAX_EVENTS_PER_JOB,
        "shard_id_base": shard_base,
        "shard_ids": shards,
        "seeds_by_bin": {
            str(bin_id): [job["seed"] for job in jobs if job["bin_id"] == bin_id]
            for bin_id in range(8)
        },
        "prior_identity_manifest_count": len(identity_sources),
        "prior_seed_count": len(prior_seeds),
        "prior_shard_id_count": len(prior_shards),
        "seed_reuse_count": 0,
        "shard_id_reuse_count": 0,
        "split_assignment_unit": "whole_shard",
        "split_algorithm": (
            "SHA256(split_salt,campaign,bin_id,shard_id): "
            "70% train / 15% validation / 15% test"
        ),
        "split_salt": args.split_salt,
        "split_job_counts": split_jobs,
        "split_event_counts": split_events,
        "final_test_excluded_from_allocation": True,
        "payload_sha256": payload_sha256,
        "wrapper_sha256": wrapper_sha256,
        "card_sha256": card_sha256,
        "git_head": git_head,
        "projected_compressed_bytes": projected_bytes,
        "projected_compressed_TiB": projected_tib,
        "storage_gate_TiB": MAX_PROJECTED_TIB,
        "storage_gate_pass": projected_tib < MAX_PROJECTED_TIB,
        "local_persistent_extraction_planned": False,
        "eos_directory": eos_directory,
        "eos_namespace_absent_at_preparation": True,
        "five_million_authorized": False,
        "submit_file": str(submit_file),
        "manifest_file": str(manifest_file),
    }
    summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Prepared submit file: {submit_file}")
    print(f"Campaign manifest: {manifest_file}")
    print(f"Campaign summary: {summary_file}")
    print(f"Payload SHA-256: {payload_sha256}")
    print(f"Worker wrapper SHA-256: {wrapper_sha256}")
    print(f"Jobs: {total_jobs}")
    print(f"Events: {EXPECTED_TOTAL}")
    print(f"Projected compressed TiB: {projected_tib:.9f}")
    print(f"EOS directory: {eos_directory}")
    print(f"Submit only once with: condor_submit {submit_file}")


if __name__ == "__main__":
    main()
