#!/usr/bin/env python3
"""Freeze the failed, zero-access outcome of validation cluster 3795859."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pandas as pd


CLUSTER = 3795859
SCHEDD = "lpcschedd4.fnal.gov"
EXPECTED_JOBS = 116
REMOTE_PARENT = "/store/user/iturkmen/hh4b_cut_baseline/validation_once_20260811_v1"
REMOTE_RESULTS = f"{REMOTE_PARENT}/results"
EXPECTED_TRACE = "ModuleNotFoundError: No module named 'awkward'"
EXPECTED_FAILURE = "message=runtime probe failed"


class FailureAuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FailureAuditError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def run(arguments: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def checked(arguments: list[str]) -> bytes:
    result = run(arguments)
    if result.returncode != 0:
        raise FailureAuditError(
            f"command failed rc={result.returncode}: {arguments!r}; "
            f"stdout={result.stdout!r}; stderr={result.stderr!r}"
        )
    return result.stdout


def query_scheduler() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    queue = json.loads(
        checked(
            [
                "/usr/bin/bash",
                "/usr/local/bin/condor_q",
                "-name",
                SCHEDD,
                str(CLUSTER),
                "-json",
            ]
        )
    )
    history = json.loads(
        checked(
            [
                "/usr/bin/bash",
                "/usr/local/bin/condor_history",
                "-name",
                SCHEDD,
                "-constraint",
                f"ClusterId == {CLUSTER}",
                "-match",
                "1000",
                "-json",
                "-attributes",
                "ClusterId,ProcId,JobStatus,ExitCode,ExitBySignal,NumJobStarts,"
                "CompletionDate,Cmd,Args,TransferInput,Iwd",
            ]
        )
    )
    return queue, history


def validate_history(history: list[dict[str, object]]) -> list[dict[str, object]]:
    require(len(history) == EXPECTED_JOBS, f"history count changed: {len(history)}")
    by_proc = {int(row["ProcId"]): row for row in history}
    require(set(by_proc) == set(range(EXPECTED_JOBS)), "history proc coverage changed")
    for proc, row in by_proc.items():
        require(row.get("ClusterId") == CLUSTER, f"history cluster changed for proc {proc}")
        require(row.get("JobStatus") == 4, f"history status changed for proc {proc}")
        require(row.get("ExitCode") == 1, f"history exit code changed for proc {proc}")
        require(row.get("ExitBySignal") is False, f"proc {proc} exited by signal")
        require(row.get("NumJobStarts") == 1, f"proc {proc} start count changed")
    return [by_proc[proc] for proc in range(EXPECTED_JOBS)]


def load_queue_items(path: Path) -> dict[int, tuple[int, str]]:
    items: dict[int, tuple[int, str]] = {}
    for proc, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        fields = line.split()
        require(len(fields) == 2, f"bad queue line: {line!r}")
        items[proc] = (int(fields[0]), fields[1])
    require(set(items) == set(range(EXPECTED_JOBS)), "queue item coverage changed")
    return items


def validate_logs(
    log_root: Path, items: dict[int, tuple[int, str]]
) -> tuple[list[dict[str, object]], list[Path]]:
    require(log_root.is_dir() and not log_root.is_symlink(), "log root is invalid")
    cluster_log = log_root / f"validation.{CLUSTER}.log"
    require(cluster_log.is_file(), "cluster event log is missing")
    rows = []
    files = [cluster_log]
    for proc in range(EXPECTED_JOBS):
        row_index, source_uid = items[proc]
        stdout = log_root / f"validation.{CLUSTER}.{proc}.row{row_index}.out"
        stderr = log_root / f"validation.{CLUSTER}.{proc}.row{row_index}.err"
        require(stdout.is_file() and stderr.is_file(), f"scheduler log pair missing for proc {proc}")
        out_text = stdout.read_text(encoding="utf-8")
        err_text = stderr.read_text(encoding="utf-8")
        require("worker.py: OK" in out_text, f"common bundle check missing for proc {proc}")
        require("VALIDATION_RUNTIME=PASS" not in out_text, f"runtime unexpectedly passed for proc {proc}")
        require(EXPECTED_TRACE in err_text, f"runtime traceback changed for proc {proc}")
        require(EXPECTED_FAILURE in err_text, f"runner failure changed for proc {proc}")
        require("marker_created=FALSE" in err_text, f"marker state changed for proc {proc}")
        require("worker_started=FALSE" in err_text, f"worker state changed for proc {proc}")
        rows.append(
            {
                "proc_id": proc,
                "production_row_index": row_index,
                "source_uid": source_uid,
                "stdout_bytes": stdout.stat().st_size,
                "stdout_sha256": sha256(stdout),
                "stderr_bytes": stderr.stat().st_size,
                "stderr_sha256": sha256(stderr),
                "exit_code": 1,
                "durable_marker_created": False,
                "validation_worker_started": False,
                "failure_stage": "runtime_probe_before_marker_and_worker",
            }
        )
        files.extend((stdout, stderr))
    actual = {path.resolve() for path in log_root.glob(f"validation.{CLUSTER}*") if path.is_file()}
    require(actual == {path.resolve() for path in files}, "scheduler log file closure changed")
    return rows, files


def validate_remote() -> tuple[str, str]:
    listing = checked(["xrdfs", "root://cmseos.fnal.gov", "ls", REMOTE_PARENT])
    entries = {line.strip().rstrip("/") for line in listing.decode().splitlines()}
    require(entries == {f"{REMOTE_PARENT}/runtime"}, f"remote campaign namespace changed: {entries}")
    require(REMOTE_RESULTS not in entries, "validation result namespace exists")
    return listing.decode(), sha256_bytes(listing)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--campaign-package", type=Path, required=True)
    parser.add_argument("--submission-checkpoint", type=Path, required=True)
    parser.add_argument("--transport-recovery-checkpoint", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--runtime-bundle", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    require(repo == args.repo, "repository path is not canonical")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    remote_head = subprocess.check_output(
        ["git", "rev-parse", "origin/delphes-hh4b-production"], cwd=repo, text=True
    ).strip()
    require(head == remote_head, "local/remote HEAD mismatch")
    for checkpoint in (args.submission_checkpoint, args.transport_recovery_checkpoint):
        result = subprocess.run(
            ["sha256sum", "-c", "SHA256SUMS"],
            cwd=checkpoint,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        require(result.returncode == 0, f"checkpoint checksum failure: {checkpoint}")
    queue_items = load_queue_items(args.campaign_package / "validation_queue.items")
    queue, history = query_scheduler()
    require(queue == [], f"validation queue is not empty: {len(queue)}")
    history = validate_history(history)
    log_rows, log_files = validate_logs(args.log_root.resolve(), queue_items)
    remote_listing, remote_listing_sha = validate_remote()
    runtime_members = checked(["tar", "-tzf", str(args.runtime_bundle.resolve())]).decode().splitlines()
    require(
        any(member.startswith("./lib/python3.9/site-packages/awkward/") for member in runtime_members),
        "authorized runtime does not contain awkward at the diagnosed path",
    )
    runner_text = (args.campaign_package / "run_validation_source.sh").read_text()
    require(
        'export PYTHONPATH="$SCRATCH/runtime/site-packages"' in runner_text,
        "frozen runner no longer has the diagnosed PYTHONPATH",
    )

    output = args.output_checkpoint if args.output_checkpoint.is_absolute() else repo / args.output_checkpoint
    output = output.resolve()
    require(output.is_relative_to(repo), "output escapes repository")
    require(output.parent.is_dir() and not output.exists(), "output path invalid/exists")
    build = output.parent / f".{output.name}.build.{os.getpid()}"
    require(not build.exists(), "temporary output exists")
    build.mkdir()
    evidence = build / "evidence"
    scheduler_logs = evidence / "scheduler_logs"
    scheduler_logs.mkdir(parents=True)
    for path in log_files:
        shutil.copy2(path, scheduler_logs / path.name)
    (evidence / "condor_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (evidence / "remote_parent_listing.txt").write_text(remote_listing, encoding="utf-8")
    pd.DataFrame(log_rows).to_csv(
        evidence / "validation_preaccess_failure_inventory.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    summary = {
        "schema_version": 1,
        "status": "blocked_validation_campaign_failed_before_payload_access",
        "repository_head_at_audit": head,
        "cluster_id": CLUSTER,
        "authoritative_schedd": SCHEDD,
        "queue_jobs": 0,
        "history_jobs": EXPECTED_JOBS,
        "exit_code_1_jobs": EXPECTED_JOBS,
        "runtime_probe_failure_jobs": EXPECTED_JOBS,
        "durable_source_attempt_markers_created": 0,
        "validation_workers_started": 0,
        "validation_sources_opened": 0,
        "validation_evaluation_cycles": 0,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
        "remote_result_namespace_absent": True,
        "remote_parent_listing_sha256": remote_listing_sha,
        "failure_stage": "runtime_probe_before_marker_and_worker",
        "failure_exception": EXPECTED_TRACE,
        "root_cause": (
            "frozen runner exported $SCRATCH/runtime/site-packages while the "
            "authorized archive stores packages under lib/python3.9/site-packages"
        ),
        "condor_submit_calls": 1,
        "do_not_resubmit_cluster": True,
        "second_validation_submission_authorized": False,
        "nominal_selection_changed": False,
        "validation_metrics_available": False,
        "next": "requires_new_explicit_user_authority_before_any_fresh_validation_submission",
    }
    (build / "validation_preaccess_failure_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (build / "README.md").write_text(
        "# HH→4b validation pre-access failure audit\n\n"
        "All 116 accepted jobs failed the frozen runtime probe before creating "
        "a source marker or starting the validation worker. Validation and test "
        "payload counters therefore remain zero. Cluster 3795859 must not be resubmitted.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in build.rglob("*") if path.is_file())
    (build / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(build).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    os.replace(build, output)
    print("VALIDATION_PREACCESS_FAILURE_AUDIT=PASS")
    print("CLUSTER_ID=3795859")
    print("FAILED_RUNTIME_PROBE_JOBS=116")
    print("DURABLE_SOURCE_MARKERS_CREATED=0")
    print("VALIDATION_WORKERS_STARTED=0")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")
    print("SECOND_VALIDATION_SUBMISSION_AUTHORIZED=FALSE")


if __name__ == "__main__":
    main()
