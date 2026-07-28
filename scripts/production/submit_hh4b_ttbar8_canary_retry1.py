#!/usr/bin/env python3
"""Safely remove the failed HH4b canary and submit one reconstruction-only retry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[2]

BRANCH = "delphes-hh4b-production"
REQUIRED_ANCESTOR = "24813782dc78236ff1d6ca1adf43392f005b51f5"

SCHEDD = "lpcschedd4.fnal.gov"
OLD_JOB = "3654710.0"
OLD_CLUSTER = "3654710"
OLD_PROC = "0"
OLD_SEED = "105003"
OLD_BATCH = "hh4b_ttbar8_exact_regeneration_canary_submission_20260727_v1"

CONTRACT_CHECKPOINT = (
    REPO
    / "docs/checkpoints/hh4b_ttbar8_canary_retry1_contract_20260727_v1"
)
CONTRACT_SUMMARY = CONTRACT_CHECKPOINT / "summary.json"
CONTRACT_CHECKSUMS = CONTRACT_CHECKPOINT / "SHA256SUMS"

SUBMIT_FILE = (
    REPO
    / "outputs/agent_runs/hh4b_ttbar8_canary_retry1_prepare_20260727_v1"
    / "hh4b_ttbar8_exact_regeneration_canary_retry1.sub"
)

PAYLOAD = (
    REPO
    / "outputs/agent_runs/hh4b_ttbar8_canary_retry1_prepare_20260727_v1"
    / "hh4b_ttbar8_exact_regeneration_canary_retry1_payload.tar.gz"
)
EXPECTED_PAYLOAD_SHA256 = (
    "3cc059b75cb63d27b57c51dfbd6643397fcb794e8e184ce689b4d905f53fad97"
)

RETURNED_ROOT = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/condor_return/"
    "hh4b_ttbar8_exact_regeneration_20260727_v1/"
    "members/ttbar_100k_shard003/root/"
    "ttbar_100k_shard003_pythia8_delphes.root"
)
EXPECTED_ROOT_SHA256 = (
    "127c87510087d164038b7629ea9d5210008cc4cb1473aa9b069a5cd4b859629e"
)

RUNTIME_DIR = (
    REPO
    / "outputs/agent_runs/hh4b_ttbar8_canary_retry1_submission_20260727_v1"
)
STATE_FILE = RUNTIME_DIR / "submission_record.json"

CONFIRM_TOKEN = "REMOVE_3654710_AND_SUBMIT_ONE_RETRY"

REQUIRED_ROOT_BRANCHES = (
    "Jet.PT",
    "Jet.Eta",
    "Jet.Phi",
    "Jet.Mass",
    "Jet.BTag",
    "Jet.Flavor",
)

REQUIRED_PAYLOAD_SUFFIXES = (
    "repo/scripts/delphes/reconstruct_hh4b_candidates_v2.py",
    "repo/scripts/delphes/write_parquet_from_pickle.py",
    "repo/configs/production/hh4b_rich_v2_reconstruction_policy_v1.yaml",
)

FORBIDDEN_SUBMIT_TOKENS = (
    "run_hh4b_ttbar8_exact_regeneration_member.sh",
    "generate_events",
    "lhe_to_hepmc3",
    "DelphesHepMC",
)


class GateError(RuntimeError):
    """Raised when a safety or provenance gate fails."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(
    args: Iterable[str],
    *,
    cwd: Path = REPO,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        list(args),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise GateError(
            f"Command failed ({proc.returncode}): {' '.join(args)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise GateError(f"Expected JSON object: {path}")
    return data


def git_output(*args: str) -> str:
    return run(("git", *args)).stdout.strip()


def verify_git_state() -> dict[str, str]:
    branch = git_output("branch", "--show-current")
    head = git_output("rev-parse", "HEAD")
    origin = git_output("rev-parse", f"origin/{BRANCH}")

    if branch != BRANCH:
        raise GateError(f"Wrong branch: {branch!r}; expected {BRANCH!r}")

    if head != origin:
        raise GateError(f"HEAD and origin differ: {head} != {origin}")

    tracked = run(("git", "status", "--porcelain")).stdout
    if tracked.strip():
        raise GateError(
            "Worktree is not clean. Commit or remove unrelated changes first:\n"
            + tracked
        )

    ancestor = run(
        ("git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, head),
        check=False,
    )
    if ancestor.returncode != 0:
        raise GateError(
            f"Required retry-contract commit {REQUIRED_ANCESTOR} "
            f"is not an ancestor of HEAD {head}"
        )

    return {"branch": branch, "head": head, "origin": origin}


def verify_contract() -> dict[str, Any]:
    if not CONTRACT_SUMMARY.is_file():
        raise GateError(f"Missing contract summary: {CONTRACT_SUMMARY}")

    summary = load_json(CONTRACT_SUMMARY)

    expected_status = (
        "hh4b_ttbar8_exact_regeneration_canary_retry_contract_frozen"
    )
    if summary.get("status") != expected_status:
        raise GateError(
            f"Contract status is {summary.get('status')!r}, "
            f"expected {expected_status!r}"
        )

    counts = summary.get("counts", {})
    expected_counts = {
        "retry_jobs_described": 1,
        "retry_jobs_submitted": 0,
        "scaleout_jobs_submitted": 0,
    }
    for key, expected in expected_counts.items():
        if counts.get(key) != expected:
            raise GateError(
                f"Contract count {key}={counts.get(key)!r}, expected {expected}"
            )

    if summary.get("retry_member") != "ttbar_100k_shard003":
        raise GateError("Retry member differs from frozen contract")

    if summary.get("retry_seed_provenance") != 105003:
        raise GateError("Retry seed provenance differs from frozen contract")

    if (
        summary.get("retry_mode")
        != "canonical72_reconstruction_only_from_returned_root"
    ):
        raise GateError("Retry mode is not reconstruction-only")

    checksum = run(
        ("sha256sum", "-c", CONTRACT_CHECKSUMS.name),
        cwd=CONTRACT_CHECKPOINT,
    )
    if "FAILED" in checksum.stdout or "FAILED" in checksum.stderr:
        raise GateError("Contract checkpoint checksum verification failed")

    return summary


def verify_returned_root() -> dict[str, Any]:
    if not RETURNED_ROOT.is_file():
        raise GateError(f"Returned ROOT file is missing: {RETURNED_ROOT}")

    actual_sha = sha256_file(RETURNED_ROOT)
    if actual_sha != EXPECTED_ROOT_SHA256:
        raise GateError(
            f"Returned ROOT SHA256 mismatch:\n"
            f"expected {EXPECTED_ROOT_SHA256}\n"
            f"actual   {actual_sha}"
        )

    try:
        import uproot
        from uproot.source.futures import TrivialExecutor
    except Exception as exc:
        raise GateError(f"Unable to import uproot serial reader: {exc}") from exc

    executor = TrivialExecutor()
    with uproot.open(
        RETURNED_ROOT,
        decompression_executor=executor,
        interpretation_executor=executor,
    ) as root_file:
        if "Delphes" not in root_file:
            raise GateError("Returned ROOT file has no Delphes tree")

        tree = root_file["Delphes"]
        entries = int(tree.num_entries)
        if entries != 10000:
            raise GateError(f"Delphes entries={entries}, expected 10000")

        missing = [name for name in REQUIRED_ROOT_BRANCHES if name not in tree]
        if missing:
            raise GateError(f"Missing required ROOT branches: {missing}")

    return {
        "path": str(RETURNED_ROOT),
        "bytes": RETURNED_ROOT.stat().st_size,
        "sha256": actual_sha,
        "tree": "Delphes",
        "entries": 10000,
        "required_branches": list(REQUIRED_ROOT_BRANCHES),
    }


def verify_payload() -> dict[str, Any]:
    if not PAYLOAD.is_file():
        raise GateError(f"Retry payload is missing: {PAYLOAD}")

    actual_sha = sha256_file(PAYLOAD)
    if actual_sha != EXPECTED_PAYLOAD_SHA256:
        raise GateError(
            f"Payload SHA256 mismatch:\n"
            f"expected {EXPECTED_PAYLOAD_SHA256}\n"
            f"actual   {actual_sha}"
        )

    with tarfile.open(PAYLOAD, "r:gz") as archive:
        members = [member.name.lstrip("./") for member in archive.getmembers()]

    missing: list[str] = []
    for suffix in REQUIRED_PAYLOAD_SUFFIXES:
        if not any(name.endswith(suffix) for name in members):
            missing.append(suffix)

    if missing:
        raise GateError(f"Payload is missing required members: {missing}")

    return {
        "path": str(PAYLOAD),
        "bytes": PAYLOAD.stat().st_size,
        "sha256": actual_sha,
        "member_count": len(members),
        "required_members": list(REQUIRED_PAYLOAD_SUFFIXES),
        "missing_writer_present": True,
    }


def count_queued_jobs(submit_text: str) -> int:
    total = 0

    for raw_line in submit_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        match = re.fullmatch(r"queue(?:\s+(\d+))?", line, flags=re.IGNORECASE)
        if match:
            total += int(match.group(1) or "1")
            continue

        if re.match(r"^queue\b", line, flags=re.IGNORECASE):
            raise GateError(
                "Unsupported non-scalar queue statement in frozen submit file: "
                + line
            )

    return total


def verify_submit_description() -> dict[str, Any]:
    if not SUBMIT_FILE.is_file():
        raise GateError(f"Retry submit file is missing: {SUBMIT_FILE}")

    text = SUBMIT_FILE.read_text(encoding="utf-8")

    jobs = count_queued_jobs(text)
    if jobs != 1:
        raise GateError(f"Submit description queues {jobs} jobs, expected 1")

    for token in FORBIDDEN_SUBMIT_TOKENS:
        if token in text:
            raise GateError(
                f"Retry submit description contains forbidden full-chain token: {token}"
            )

    required_basenames = (RETURNED_ROOT.name, PAYLOAD.name)
    for basename in required_basenames:
        if basename not in text:
            raise GateError(
                f"Retry submit description does not reference required input: {basename}"
            )

    return {
        "path": str(SUBMIT_FILE),
        "bytes": SUBMIT_FILE.stat().st_size,
        "sha256": sha256_file(SUBMIT_FILE),
        "queued_jobs": jobs,
        "returned_root_referenced": True,
        "corrected_payload_referenced": True,
        "full_chain_tokens_absent": True,
    }


def condor_attr(job: str, attr: str) -> str:
    proc = run(
        ("condor_q", "-name", SCHEDD, job, "-af", attr),
        check=False,
    )
    if proc.returncode != 0:
        raise GateError(
            f"condor_q failed for {job} {attr}:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def query_old_job() -> dict[str, str]:
    cluster = condor_attr(OLD_JOB, "ClusterId")
    if not cluster:
        raise GateError(f"Original held job {OLD_JOB} is not in condor_q")

    attrs = {
        "ClusterId": cluster,
        "ProcId": condor_attr(OLD_JOB, "ProcId"),
        "JobStatus": condor_attr(OLD_JOB, "JobStatus"),
        "ExactRegenerationCanary": condor_attr(
            OLD_JOB, "ExactRegenerationCanary"
        ),
        "CanarySeed": condor_attr(OLD_JOB, "CanarySeed"),
        "JobBatchName": condor_attr(OLD_JOB, "JobBatchName"),
        "HoldReasonCode": condor_attr(OLD_JOB, "HoldReasonCode"),
        "HoldReasonSubCode": condor_attr(OLD_JOB, "HoldReasonSubCode"),
        "HoldReason": condor_attr(OLD_JOB, "HoldReason"),
        "UserLog": condor_attr(OLD_JOB, "UserLog"),
        "Out": condor_attr(OLD_JOB, "Out"),
        "Err": condor_attr(OLD_JOB, "Err"),
    }

    if attrs["ClusterId"] != OLD_CLUSTER or attrs["ProcId"] != OLD_PROC:
        raise GateError(f"Old job identity mismatch: {attrs}")

    if attrs["JobStatus"] != "5":
        raise GateError(
            f"Old job is not held: JobStatus={attrs['JobStatus']!r}"
        )

    if attrs["CanarySeed"] != OLD_SEED:
        raise GateError(
            f"Old job seed is {attrs['CanarySeed']!r}, expected {OLD_SEED}"
        )

    if attrs["ExactRegenerationCanary"].lower() != "true":
        raise GateError("Old job is not marked ExactRegenerationCanary=true")

    if attrs["JobBatchName"].strip('"') != OLD_BATCH:
        raise GateError(
            f"Old job batch is {attrs['JobBatchName']!r}, expected {OLD_BATCH!r}"
        )

    return attrs


def archive_old_job(attrs: dict[str, str]) -> dict[str, Any]:
    archive_dir = RUNTIME_DIR / "original_held_job_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    classad = run(
        ("condor_q", "-name", SCHEDD, OLD_JOB, "-long")
    ).stdout
    classad_path = archive_dir / "job_3654710_0.classad.txt"
    classad_path.write_text(classad, encoding="utf-8")

    archived_files: list[dict[str, Any]] = [
        {
            "kind": "classad",
            "source": "condor_q -long",
            "archived_path": str(classad_path),
            "bytes": classad_path.stat().st_size,
            "sha256": sha256_file(classad_path),
        }
    ]

    for attr in ("UserLog", "Out", "Err"):
        raw = attrs.get(attr, "").strip().strip('"')
        if not raw or raw == "undefined":
            continue

        source = Path(raw)
        if not source.is_file():
            raise GateError(f"Expected old-job log is missing: {source}")

        target = archive_dir / f"{attr.lower()}_{source.name}"
        shutil.copy2(source, target)

        archived_files.append(
            {
                "kind": attr,
                "source": str(source),
                "archived_path": str(target),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )

    return {
        "timestamp_utc": utc_now(),
        "job": OLD_JOB,
        "attributes": attrs,
        "files": archived_files,
    }


def parse_terse_submission(text: str) -> tuple[int, int]:
    match = re.search(r"\b(\d+)\.(\d+)\b", text)
    if not match:
        raise GateError(f"Unable to parse condor_submit -terse output: {text!r}")
    return int(match.group(1)), int(match.group(2))


def wait_until_job_absent(job: str, timeout_seconds: int = 45) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        output = condor_attr(job, "ClusterId")
        if not output:
            return
        time.sleep(2)
    raise GateError(f"Job {job} remained in condor_q after removal")


def wait_for_new_job(job: str, timeout_seconds: int = 45) -> dict[str, str]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        cluster = condor_attr(job, "ClusterId")
        if cluster:
            return {
                "ClusterId": cluster,
                "ProcId": condor_attr(job, "ProcId"),
                "JobStatus": condor_attr(job, "JobStatus"),
                "JobBatchName": condor_attr(job, "JobBatchName"),
                "Iwd": condor_attr(job, "Iwd"),
                "Cmd": condor_attr(job, "Cmd"),
                "Args": condor_attr(job, "Args"),
                "TransferInput": condor_attr(job, "TransferInput"),
                "TransferOutput": condor_attr(job, "TransferOutput"),
                "TransferOutputRemaps": condor_attr(
                    job, "TransferOutputRemaps"
                ),
                "RequestCpus": condor_attr(job, "RequestCpus"),
                "RequestMemory": condor_attr(job, "RequestMemory"),
                "RequestDisk": condor_attr(job, "RequestDisk"),
                "Out": condor_attr(job, "Out"),
                "Err": condor_attr(job, "Err"),
                "UserLog": condor_attr(job, "UserLog"),
            }
        time.sleep(2)
    raise GateError(f"New retry job {job} did not appear in condor_q")


def save_record(record: dict[str, Any]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, STATE_FILE)


def preflight() -> dict[str, Any]:
    result = {
        "timestamp_utc": utc_now(),
        "mode": "preflight",
        "git": verify_git_state(),
        "contract": verify_contract(),
        "returned_root": verify_returned_root(),
        "payload": verify_payload(),
        "submit_description": verify_submit_description(),
        "old_job": query_old_job(),
        "checks": {
            "old_job_held": True,
            "old_job_removed": False,
            "retry_submitted": False,
            "scaleout_submitted": False,
            "madgraph_rerun": False,
            "pythia_rerun": False,
            "delphes_rerun": False,
        },
        "status": "preflight_pass",
    }
    return result


def execute_submission(confirm_token: str) -> dict[str, Any]:
    if confirm_token != CONFIRM_TOKEN:
        raise GateError(
            "Incorrect confirmation token. No scheduler change was made."
        )

    if STATE_FILE.exists():
        existing = load_json(STATE_FILE)
        if existing.get("submission", {}).get("completed") is True:
            raise GateError(
                f"A retry submission is already recorded in {STATE_FILE}; "
                "refusing a duplicate submission."
            )

    result = preflight()
    result["mode"] = "execute"
    result["execution_started_utc"] = utc_now()

    archive = archive_old_job(result["old_job"])
    result["original_job_archive"] = archive
    save_record(result)

    removal_command = (
        "condor_rm",
        "-name",
        SCHEDD,
        OLD_JOB,
    )
    removal = run(removal_command)

    result["removal"] = {
        "completed": True,
        "timestamp_utc": utc_now(),
        "command": list(removal_command),
        "stdout": removal.stdout,
        "stderr": removal.stderr,
        "returncode": removal.returncode,
    }
    save_record(result)

    wait_until_job_absent(OLD_JOB)

    submit_command = (
        "condor_submit",
        "-name",
        SCHEDD,
        "-terse",
        str(SUBMIT_FILE),
    )
    submitted = run(submit_command)

    cluster, proc = parse_terse_submission(
        submitted.stdout + "\n" + submitted.stderr
    )
    new_job = f"{cluster}.{proc}"
    snapshot = wait_for_new_job(new_job)

    result["submission"] = {
        "completed": True,
        "timestamp_utc": utc_now(),
        "command": list(submit_command),
        "stdout": submitted.stdout,
        "stderr": submitted.stderr,
        "returncode": submitted.returncode,
        "scheduler": SCHEDD,
        "cluster_id": cluster,
        "process_id": proc,
        "job_id": new_job,
        "jobs_submitted": 1,
        "scaleout_jobs_submitted": 0,
        "member": "ttbar_100k_shard003",
        "seed_provenance": 105003,
        "source_job": OLD_JOB,
        "source_root_path": str(RETURNED_ROOT),
        "source_root_sha256": EXPECTED_ROOT_SHA256,
        "retry_mode": "canonical72_reconstruction_only_from_returned_root",
        "scheduler_snapshot": snapshot,
    }
    result["checks"]["old_job_removed"] = True
    result["checks"]["retry_submitted"] = True
    result["status"] = (
        "hh4b_ttbar8_exact_regeneration_canary_retry_submitted"
    )
    result["next_gate"] = (
        "validate_hh4b_ttbar8_exact_regeneration_canary_retry1"
    )
    result["execution_finished_utc"] = utc_now()

    save_record(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("preflight", "execute"),
        required=True,
    )
    parser.add_argument(
        "--confirm-token",
        default="",
        help="Required only for --mode execute.",
    )
    args = parser.parse_args()

    try:
        if args.mode == "preflight":
            record = preflight()
        else:
            record = execute_submission(args.confirm_token)
    except GateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
