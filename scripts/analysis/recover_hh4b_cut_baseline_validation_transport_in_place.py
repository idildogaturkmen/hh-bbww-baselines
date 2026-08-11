#!/usr/bin/env python3
"""Fail-closed in-place transport recovery for validation cluster 3795859.

The accepted jobs never reached their executable because the remote schedd
could not read a submit-client-local /tmp path.  This driver permits one
evidence-first edit of only Cmd and TransferInput on the existing held jobs,
followed by one release.  It never calls condor_submit and refuses to operate
after any scientific executable start or validation result creation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


REPO = Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines")
BRANCH = "delphes-hh4b-production"
SCHEDD = "lpcschedd4.fnal.gov"
CLUSTER = 3795859
TARGET_COUNT = 116
PRESUBMISSION = REPO / (
    "docs/checkpoints/"
    "hh4b_cut_baseline_validation_campaign_presubmission_20260811_v1"
)
SUBMISSION = REPO / (
    "docs/checkpoints/hh4b_cut_baseline_validation_submission_20260811_v1"
)
FROZEN_PACKAGE = PRESUBMISSION / "evidence/campaign_package"
FROZEN_RUNNER = FROZEN_PACKAGE / "run_validation_source.sh"
FROZEN_COMMON = FROZEN_PACKAGE / "validation_common_bundle.tar.gz"
FROZEN_QUEUE = FROZEN_PACKAGE / "validation_queue.items"
ORIGINAL_PACKAGE = Path(
    "/tmp/hh4b_cut_baseline_validation_campaign_package_20260811_v2"
)
ORIGINAL_RUNNER = ORIGINAL_PACKAGE / "run_validation_source.sh"
ORIGINAL_COMMON = ORIGINAL_PACKAGE / "validation_common_bundle.tar.gz"
ORIGINAL_TRANSFER_INPUT = str(ORIGINAL_COMMON)
RECOVERY_TRANSFER_INPUT = str(FROZEN_COMMON)
REMOTE_RESULT_ROOT = (
    "/store/user/iturkmen/hh4b_cut_baseline/"
    "validation_once_20260811_v1/results"
)
LOG_ROOT = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/condor_submit/"
    "hh4b_cut_baseline_validation_once_20260811_v1/logs"
)
EVIDENCE = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/condor_submit/"
    "hh4b_cut_baseline_validation_once_20260811_v1/"
    "transport_recovery_evidence_v1"
)


class RecoveryGateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryGateError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_write(path: Path, payload: bytes, mode: int = 0o600) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def run(arguments: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        arguments,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def checked(arguments: list[str], cwd: Path | None = None) -> bytes:
    result = run(arguments, cwd)
    if result.returncode != 0:
        raise RecoveryGateError(
            f"command failed rc={result.returncode}: {arguments!r}; "
            f"stdout={result.stdout!r}; stderr={result.stderr!r}"
        )
    return result.stdout


def record_command(name: str, result: subprocess.CompletedProcess[bytes]) -> None:
    durable_write(EVIDENCE / f"{name}.stdout", result.stdout)
    durable_write(EVIDENCE / f"{name}.stderr", result.stderr)
    durable_write(EVIDENCE / f"{name}.rc", f"{result.returncode}\n".encode())


def verify_checkpoint(path: Path) -> None:
    result = run(["sha256sum", "-c", "SHA256SUMS"], path)
    require(
        result.returncode == 0,
        f"checkpoint checksum failure: {path}: {result.stdout!r}{result.stderr!r}",
    )


def query_queue() -> list[dict[str, Any]]:
    payload = checked(
        [
            "/usr/bin/bash",
            "/usr/local/bin/condor_q",
            "-name",
            SCHEDD,
            str(CLUSTER),
            "-json",
            "-attributes",
            "ClusterId,ProcId,JobStatus,HoldReasonCode,HoldReasonSubCode,"
            "HoldReason,Cmd,Args,TransferInput,Iwd,NumJobStarts,"
            "NumShadowStarts,NumSystemHolds,Owner",
        ]
    )
    return json.loads(payload)


def query_history() -> list[dict[str, Any]]:
    payload = checked(
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
            "ClusterId,ProcId,JobStatus,ExitCode,Cmd,TransferInput,CompletionDate",
        ]
    )
    return json.loads(payload)


def load_queue() -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for proc, line in enumerate(FROZEN_QUEUE.read_text(encoding="utf-8").splitlines()):
        fields = line.split()
        require(len(fields) == 2, f"malformed frozen queue item: {line!r}")
        row_index, source_uid = fields
        rows[proc] = {
            "row_index": int(row_index),
            "source_uid": source_uid,
            "arguments": f"{row_index} {source_uid} {CLUSTER} {proc}",
        }
    require(set(rows) == set(range(TARGET_COUNT)), "frozen queue coverage changed")
    return rows


def verify_repository(expected_head: str) -> dict[str, Any]:
    require(bool(re.fullmatch(r"[0-9a-f]{40}", expected_head)), "bad expected HEAD")
    head = checked(["git", "rev-parse", "HEAD"], REPO).decode().strip()
    remote = checked(["git", "rev-parse", f"origin/{BRANCH}"], REPO).decode().strip()
    require(head == remote == expected_head, f"local/remote/expected HEAD mismatch: {head} {remote}")
    driver = Path(__file__).resolve()
    relative = driver.relative_to(REPO).as_posix()
    committed = checked(["git", "show", f"{head}:{relative}"], REPO)
    require(sha256_bytes(committed) == sha256(driver), "recovery driver differs from committed bytes")
    verify_checkpoint(PRESUBMISSION)
    verify_checkpoint(SUBMISSION)
    receipt = json.loads(
        (SUBMISSION / "evidence/submission/validation_submission_receipt.json").read_text()
    )
    require(receipt.get("cluster_id") == CLUSTER, "submission receipt cluster changed")
    require(receipt.get("schedd") == SCHEDD, "submission receipt schedd changed")
    require(receipt.get("expected_job_count") == TARGET_COUNT, "submission job count changed")
    require(receipt.get("do_not_resubmit_cluster") is True, "never-resubmit flag changed")
    campaign = json.loads((FROZEN_PACKAGE / "campaign_summary.json").read_text())
    require(sha256(FROZEN_RUNNER) == campaign["runner_sha256"], "frozen runner SHA changed")
    require(
        sha256(FROZEN_COMMON) == campaign["validation_common_bundle_sha256"],
        "frozen common bundle SHA changed",
    )
    require(FROZEN_RUNNER.stat().st_mode & 0o111, "frozen runner is not executable")
    return {
        "repository_head": head,
        "driver_sha256": sha256(driver),
        "runner_sha256": sha256(FROZEN_RUNNER),
        "common_bundle_sha256": sha256(FROZEN_COMMON),
        "submission_receipt_sha256": sha256(
            SUBMISSION / "evidence/submission/validation_submission_receipt.json"
        ),
    }


def verify_remote_absence() -> dict[str, Any]:
    parent = str(Path(REMOTE_RESULT_ROOT).parent)
    result = run(["xrdfs", "root://cmseos.fnal.gov", "ls", parent])
    require(result.returncode == 0, f"cannot audit EOS parent: {result.stderr!r}")
    entries = {line.strip().rstrip("/") for line in result.stdout.decode().splitlines()}
    require(REMOTE_RESULT_ROOT not in entries, "validation result namespace already exists")
    return {
        "remote_result_namespace_absent": True,
        "remote_parent": parent,
        "remote_parent_listing_sha256": sha256_bytes(result.stdout),
    }


def validate_original_held(
    ads: list[dict[str, Any]], rows: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    require(len(ads) == TARGET_COUNT, f"expected {TARGET_COUNT} queued jobs, found {len(ads)}")
    by_proc = {int(ad["ProcId"]): ad for ad in ads}
    require(set(by_proc) == set(rows), "queued proc coverage changed")
    for proc, ad in by_proc.items():
        require(ad.get("ClusterId") == CLUSTER, f"cluster mismatch for proc {proc}")
        require(ad.get("Owner") == "iturkmen", f"owner mismatch for proc {proc}")
        require(ad.get("JobStatus") == 5, f"proc {proc} is not held")
        require(
            (ad.get("HoldReasonCode"), ad.get("HoldReasonSubCode")) == (13, 2),
            f"unexpected hold reason for proc {proc}",
        )
        require(ad.get("NumJobStarts") == 0, f"proc {proc} already started")
        require(ad.get("Cmd") == str(ORIGINAL_RUNNER), f"original Cmd changed for proc {proc}")
        require(
            ad.get("TransferInput") == ORIGINAL_TRANSFER_INPUT,
            f"original TransferInput changed for proc {proc}",
        )
        require(ad.get("Args") == rows[proc]["arguments"], f"arguments changed for proc {proc}")
    return {
        "queued_jobs": len(ads),
        "held_jobs": len(ads),
        "hold_reason_13_2_jobs": len(ads),
        "scientific_executable_starts": 0,
        "proc_min": min(by_proc),
        "proc_max": max(by_proc),
    }


def validate_edited_held(
    ads: list[dict[str, Any]], rows: dict[int, dict[str, Any]]
) -> None:
    require(len(ads) == TARGET_COUNT, "post-edit queue count changed")
    by_proc = {int(ad["ProcId"]): ad for ad in ads}
    require(set(by_proc) == set(rows), "post-edit proc coverage changed")
    for proc, ad in by_proc.items():
        require(ad.get("JobStatus") == 5, f"proc {proc} left hold before release")
        require(ad.get("NumJobStarts") == 0, f"proc {proc} started before release")
        require(ad.get("Cmd") == str(FROZEN_RUNNER), f"Cmd edit failed for proc {proc}")
        require(
            ad.get("TransferInput") == RECOVERY_TRANSFER_INPUT,
            f"TransferInput edit failed for proc {proc}",
        )
        require(ad.get("Args") == rows[proc]["arguments"], f"arguments changed for proc {proc}")


def preflight(expected_head: str) -> dict[str, Any]:
    require(not EVIDENCE.exists(), f"recovery evidence exists; refusing to repeat: {EVIDENCE}")
    require(EVIDENCE.parent.is_dir(), "recovery evidence parent is absent")
    repository = verify_repository(expected_head)
    rows = load_queue()
    ads = query_queue()
    history = query_history()
    scheduler = validate_original_held(ads, rows)
    require(history == [], f"validation cluster already has history rows: {len(history)}")
    require(LOG_ROOT.is_dir() and not LOG_ROOT.is_symlink(), "log root is invalid")
    require(not list(LOG_ROOT.glob("validation.*.out")), "validation stdout already exists")
    require(not list(LOG_ROOT.glob("validation.*.err")), "validation stderr already exists")
    remote = verify_remote_absence()
    return {
        "schema_version": 1,
        "status": "pass_validation_cluster_in_place_transport_recovery_preflight",
        "cluster_id": CLUSTER,
        "authoritative_schedd": SCHEDD,
        "target_count": TARGET_COUNT,
        "repository": repository,
        "scheduler": scheduler,
        **remote,
        "original_cmd": str(ORIGINAL_RUNNER),
        "recovery_cmd": str(FROZEN_RUNNER),
        "original_transfer_input": ORIGINAL_TRANSFER_INPUT,
        "recovery_transfer_input": RECOVERY_TRANSFER_INPUT,
        "condor_submit_planned": False,
        "scientific_arguments_changed": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }


def execute(expected_head: str) -> None:
    gate = preflight(expected_head)
    rows = load_queue()
    attempted_utc = utc_now()
    os.mkdir(EVIDENCE, 0o700)
    fsync_directory(EVIDENCE.parent)
    durable_write(EVIDENCE / "recovery_driver.py", Path(__file__).read_bytes(), 0o700)
    durable_write(EVIDENCE / "pre_recovery_gate.json", json_bytes(gate))
    durable_write(EVIDENCE / "pre_recovery_job_ads.json", json_bytes(query_queue()))
    durable_write(
        EVIDENCE / "RECOVERY_ATTEMPTED_DO_NOT_REPEAT.txt",
        (
            "status=validation_cluster_3795859_in_place_transport_recovery_attempted\n"
            f"attempted_utc={attempted_utc}\n"
            f"repository_head={expected_head}\n"
            "rule=DO_NOT_CALL_CONDOR_SUBMIT\n"
            "rule=DO_NOT_REPEAT_THIS_RECOVERY_DRIVER\n"
            "rule=ONLY_EXISTING_CLUSTER_3795859_PROCS_0_115_ARE_TARGETED\n"
        ).encode(),
    )

    constraint = (
        f"ClusterId == {CLUSTER} && JobStatus == 5 && NumJobStarts == 0 && "
        f"Cmd == {json.dumps(str(ORIGINAL_RUNNER))} && "
        f"TransferInput == {json.dumps(ORIGINAL_TRANSFER_INPUT)}"
    )
    edit = run(
        [
            "/usr/bin/bash",
            "/usr/local/bin/condor_qedit",
            "-name",
            SCHEDD,
            "-constraint",
            constraint,
            "Cmd",
            json.dumps(str(FROZEN_RUNNER)),
            "TransferInput",
            json.dumps(RECOVERY_TRANSFER_INPUT),
        ]
    )
    record_command("condor_qedit_transport_attributes", edit)
    require(edit.returncode == 0, "transport qedit failed; jobs remain held")
    edited = query_queue()
    validate_edited_held(edited, rows)
    durable_write(EVIDENCE / "post_edit_pre_release_job_ads.json", json_bytes(edited))

    release_constraint = (
        f"ClusterId == {CLUSTER} && JobStatus == 5 && NumJobStarts == 0 && "
        f"Cmd == {json.dumps(str(FROZEN_RUNNER))} && "
        f"TransferInput == {json.dumps(RECOVERY_TRANSFER_INPUT)}"
    )
    release = run(
        [
            "/usr/bin/bash",
            "/usr/local/bin/condor_release",
            "-name",
            SCHEDD,
            "-reason",
            "Release accepted validation cluster after audited shared-path transport repair; no resubmission",
            "-constraint",
            release_constraint,
        ]
    )
    record_command("condor_release_recovered_targets", release)
    require(release.returncode == 0, "release failed after verified edits")

    released_utc = utc_now()
    post_ads = query_queue()
    require(len(post_ads) == TARGET_COUNT, "post-release queue count changed unexpectedly")
    for ad in post_ads:
        proc = int(ad["ProcId"])
        require(ad.get("Cmd") == str(FROZEN_RUNNER), f"post-release Cmd changed for proc {proc}")
        require(
            ad.get("TransferInput") == RECOVERY_TRANSFER_INPUT,
            f"post-release TransferInput changed for proc {proc}",
        )
        require(ad.get("Args") == rows[proc]["arguments"], f"post-release arguments changed for proc {proc}")
    durable_write(EVIDENCE / "post_release_job_ads.json", json_bytes(post_ads))
    status_counts: dict[str, int] = {}
    for ad in post_ads:
        key = str(ad.get("JobStatus"))
        status_counts[key] = status_counts.get(key, 0) + 1
    receipt = {
        "schema_version": 1,
        "status": "pass_validation_cluster_in_place_transport_recovery_edited_and_released",
        "cluster_id": CLUSTER,
        "authoritative_schedd": SCHEDD,
        "target_count": TARGET_COUNT,
        "repository_head": expected_head,
        "attempted_utc": attempted_utc,
        "released_utc": released_utc,
        "post_release_status_counts": status_counts,
        "condor_submit_called": False,
        "never_resubmit_cluster": True,
        "transport_recovery_only": True,
        "scientific_arguments_changed": False,
        "nominal_selection_changed": False,
        "validation_payloads_opened_before_release": 0,
        "test_payloads_opened": 0,
    }
    durable_write(EVIDENCE / "transport_recovery_receipt.json", json_bytes(receipt))
    durable_write(
        EVIDENCE / "RECOVERY_RELEASED_MONITOR_ONLY.txt",
        (
            "status=validation_cluster_3795859_transport_edited_and_released\n"
            f"released_utc={released_utc}\n"
            "rule=MONITOR_EXISTING_CLUSTER_ONLY\n"
            "rule=DO_NOT_RESUBMIT_CLUSTER_3795859\n"
        ).encode(),
    )
    print("VALIDATION_CLUSTER_IN_PLACE_TRANSPORT_RECOVERY=RELEASED")
    print(f"CLUSTER_ID={CLUSTER}")
    print(f"TARGET_COUNT={TARGET_COUNT}")
    print("CONDOR_SUBMIT_CALLED=FALSE")
    print("DO_NOT_RESUBMIT=TRUE")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute-in-place", action="store_true")
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    if args.preflight_only:
        print(json.dumps(preflight(args.expected_head), indent=2, sort_keys=True))
    else:
        execute(args.expected_head)


if __name__ == "__main__":
    try:
        main()
    except RecoveryGateError as error:
        raise SystemExit(f"RECOVERY_GATE_ERROR: {error}") from error
