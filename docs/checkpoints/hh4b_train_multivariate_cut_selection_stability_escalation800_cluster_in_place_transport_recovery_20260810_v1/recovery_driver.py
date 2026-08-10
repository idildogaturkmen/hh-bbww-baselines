#!/usr/bin/env python3
"""Exactly-once in-place transport recovery for cluster 30020809 procs 1..7999."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


REPO = Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines")
EXPECTED_REPOSITORY_HEAD = "75898d4bd05158ddedd95fbc15bff53b54a35d72"
PACKAGE = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/baselines/"
    "multivariate_cut_selection_stability_escalation800_production_package_v1_20260809"
)
SCHEDD = "lpcschedd5.fnal.gov"
CLUSTER = 30020809
CANARY_PROC = 0
TARGET_COUNT = 7999
EVIDENCE = (
    PACKAGE / "evidence/production_submission_v1/eos_xrootd_recovery_v1/"
    "cluster_wide_in_place_v1"
)
CANARY_CHECKPOINT = (
    REPO / "docs/checkpoints/"
    "hh4b_train_multivariate_cut_selection_stability_escalation800_eos_recovery_canary_20260810_v1"
)
CANARY_AUDIT = CANARY_CHECKPOINT / "proc0_complete_return_audit.json"
RECOVERY = REPO / "scripts/production/run_hh4b_stability_escalation800_eos_recovery.sh"
ORIGINAL_RUNNER = PACKAGE / "submission/run_escalation800_category_fold_job.sh"
QUEUE_ITEMS = PACKAGE / "submission/escalation800_queue.items"
RUNTIME = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/baselines/"
    "multivariate_cut_bounded_six_job_transfer_pilot_v8_r1_20260807T004537Z/"
    "archives/hh4b_python_runtime_v8_r1.tar.gz"
)
RECOVERY_TRANSFER_INPUT = f"{ORIGINAL_RUNNER},{RUNTIME}"
EXPECTED_RECOVERY_SHA = "ed687c9cda7dc27c51c65b1a789479c1612ce402dcb2ebd59a7cdeca67bd4821"
EXPECTED_RUNNER_SHA = "b5a0f96950f1b2cba99e1ed173b33a70630f10f2e1c516090920dea7137c15e5"
EXPECTED_RUNTIME_SHA = "4a9c4061e1d975f276326022454e578c5f0de3c094b9b56345f192cccc654112"


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def command(arguments: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(arguments, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def checked(arguments: list[str], cwd: Path | None = None) -> bytes:
    result = command(arguments, cwd)
    if result.returncode != 0:
        raise GateError(
            f"command failed rc={result.returncode}: {arguments!r}; "
            f"stdout={result.stdout!r}; stderr={result.stderr!r}"
        )
    return result.stdout


def record_command(name: str, result: subprocess.CompletedProcess[bytes]) -> None:
    durable_write(EVIDENCE / f"{name}.stdout", result.stdout)
    durable_write(EVIDENCE / f"{name}.stderr", result.stderr)
    durable_write(EVIDENCE / f"{name}.rc", f"{result.returncode}\n".encode())


def query_queue() -> list[dict[str, object]]:
    output = checked([
        "/usr/bin/bash", "/usr/local/bin/condor_q", "-name", SCHEDD,
        str(CLUSTER), "-json", "-attributes",
        "ClusterId,ProcId,JobStatus,HoldReasonCode,HoldReasonSubCode,HoldReason,"
        "Cmd,Arguments,TransferInput,Iwd,NumJobStarts,NumShadowStarts,"
        "NumSystemHolds,Owner",
    ])
    return json.loads(output)


def query_history() -> list[dict[str, object]]:
    output = checked([
        "/usr/bin/bash", "/usr/local/bin/condor_history", "-name", SCHEDD,
        "-constraint", f"ClusterId == {CLUSTER}", "-match", "10000", "-json",
        "-attributes", "ClusterId,ProcId,JobStatus,ExitCode,ExitBySignal,"
        "Cmd,TransferInput,Iwd,CompletionDate",
    ])
    return json.loads(output)


def load_queue_items() -> dict[int, dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    with QUEUE_ITEMS.open(newline="") as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            require(len(fields) == 10, f"queue item has {len(fields)} fields: {fields!r}")
            job_index, replica, outer_fold, category, head, payload_archive, payload_basename, payload_sha, runtime_sha, return_dir = fields
            proc = int(job_index)
            require(proc not in rows, f"duplicate queue proc {proc}")
            rows[proc] = {
                "replica": int(replica),
                "outer_fold": int(outer_fold),
                "category_id": category,
                "execution_head": head,
                "payload_archive": payload_archive,
                "payload_basename": payload_basename,
                "payload_sha256": payload_sha,
                "runtime_sha256": runtime_sha,
                "return_dir": return_dir,
                "arguments": " ".join([
                    job_index, replica, outer_fold, category, head,
                    payload_basename, payload_sha, runtime_sha,
                ]),
                "original_transfer_input": f"{payload_archive},{RUNTIME}",
            }
    require(set(rows) == set(range(8000)), "queue item proc coverage is not exactly 0..7999")
    return rows


def validate_canary_and_git() -> dict[str, object]:
    require(EXPECTED_REPOSITORY_HEAD != "__CANARY_CHECKPOINT_HEAD__", "driver HEAD placeholder was not frozen")
    local = checked(["git", "rev-parse", "HEAD"], REPO).decode().strip()
    remote = checked(["git", "rev-parse", "origin/delphes-hh4b-production"], REPO).decode().strip()
    status = checked(["git", "status", "--porcelain=v1", "--untracked-files=all"], REPO)
    require(local == EXPECTED_REPOSITORY_HEAD, f"local HEAD mismatch: {local}")
    require(remote == EXPECTED_REPOSITORY_HEAD, f"remote HEAD mismatch: {remote}")
    require(status == b"", f"repository is not clean: {status!r}")

    hashes = {
        "recovery_sha256": sha256_file(RECOVERY),
        "runner_sha256": sha256_file(ORIGINAL_RUNNER),
        "runtime_sha256": sha256_file(RUNTIME),
    }
    expected_hashes = {
        "recovery_sha256": EXPECTED_RECOVERY_SHA,
        "runner_sha256": EXPECTED_RUNNER_SHA,
        "runtime_sha256": EXPECTED_RUNTIME_SHA,
    }
    require(hashes == expected_hashes, f"transport input hashes changed: {hashes!r}")

    require(CANARY_AUDIT.is_file(), f"committed canary audit absent: {CANARY_AUDIT}")
    canary = json.loads(CANARY_AUDIT.read_text())
    require(
        canary.get("status") == "pass_escalation800_proc0_eos_xrootd_recovery_canary_complete_return_audit",
        f"canary audit did not pass: {canary.get('status')!r}",
    )
    require(canary.get("cluster_id") == CLUSTER and canary.get("proc_id") == CANARY_PROC, "canary identity mismatch")
    require(canary.get("condor_submit_called") is False, "canary audit submit flag mismatch")
    require(canary.get("never_resubmit_cluster") is True, "canary no-resubmit flag mismatch")
    require(canary.get("transport_recovery_only") is True, "canary transport-only flag mismatch")
    require(canary.get("nominal_selection_changed") is False, "canary nominal-selection flag mismatch")
    require(canary.get("validation_payloads_opened") == 0 and canary.get("test_payloads_opened") == 0, "canary sealed counters changed")
    require(canary["return_audit"]["structure_results_audited"] == 27, "canary structure audit count mismatch")
    return {
        "repository_head": local,
        "transport_input_sha256": hashes,
        "canary_audit_path": str(CANARY_AUDIT),
        "canary_audit_sha256": sha256_file(CANARY_AUDIT),
    }


def validate_scheduler_preflight(
    ads: list[dict[str, object]],
    history: list[dict[str, object]],
    items: dict[int, dict[str, object]],
) -> dict[str, object]:
    require(len(ads) == TARGET_COUNT, f"expected {TARGET_COUNT} queued ads after canary: {len(ads)}")
    by_proc = {int(ad["ProcId"]): ad for ad in ads}
    require(len(by_proc) == TARGET_COUNT, "duplicate proc ids in queue ads")
    require(set(by_proc) == set(range(1, 8000)), "queued proc coverage is not exactly 1..7999")

    status_counts = {1: 0, 2: 0, 5: 0}
    hold_reason_counts: dict[str, int] = {}
    for proc, ad in by_proc.items():
        item = items[proc]
        require(ad.get("ClusterId") == CLUSTER, f"cluster mismatch for proc {proc}")
        require(ad.get("Owner") == "iturkmen", f"owner mismatch for proc {proc}")
        status = int(ad.get("JobStatus", -1))
        require(status in status_counts, f"unexpected JobStatus {status} for proc {proc}")
        status_counts[status] += 1
        require(int(ad.get("NumJobStarts", -1)) == 0, f"proc {proc} started scientific executable unexpectedly")
        require(ad.get("Cmd") == str(ORIGINAL_RUNNER), f"original Cmd mismatch for proc {proc}")
        require(ad.get("Arguments") == item["arguments"], f"scientific Arguments mismatch for proc {proc}")
        require(ad.get("TransferInput") == item["original_transfer_input"], f"original TransferInput mismatch for proc {proc}")
        require(ad.get("Iwd") == item["return_dir"], f"Iwd mismatch for proc {proc}")
        return_dir = Path(str(item["return_dir"]))
        require(return_dir.is_dir(), f"return directory absent for proc {proc}")
        require(not any(return_dir.iterdir()), f"return directory is nonempty for untouched proc {proc}")
        if status == 5:
            code = int(ad.get("HoldReasonCode", -1))
            subcode = int(ad.get("HoldReasonSubCode", -1))
            require((code, subcode) == (13, 2), f"unexpected hold cause for proc {proc}: {(code, subcode)}")
            key = f"{code}:{subcode}"
            hold_reason_counts[key] = hold_reason_counts.get(key, 0) + 1

    require(len(history) == 1, f"expected exactly canary in history: {len(history)} rows")
    canary = history[0]
    require(canary.get("ProcId") == 0 and canary.get("ClusterId") == CLUSTER, "history is not solely proc0")
    require(canary.get("JobStatus") == 4 and canary.get("ExitCode") == 0, "canary history not complete/zero")
    require(canary.get("ExitBySignal") is False, "canary exited by signal")
    require(canary.get("Cmd") == str(RECOVERY), "canary history recovery Cmd mismatch")
    require(canary.get("TransferInput") == RECOVERY_TRANSFER_INPUT, "canary history transfer input mismatch")
    return {
        "queued_target_count": len(ads),
        "queued_proc_min": min(by_proc),
        "queued_proc_max": max(by_proc),
        "status_counts": {str(key): value for key, value in status_counts.items()},
        "held_reason_counts": hold_reason_counts,
        "target_return_directories_empty": TARGET_COUNT,
        "history_count": 1,
        "canary_history_ad": canary,
    }


def require_all_held_original(ads: list[dict[str, object]], items: dict[int, dict[str, object]]) -> None:
    require(len(ads) == TARGET_COUNT, f"post-hold ad count mismatch: {len(ads)}")
    require({int(ad["ProcId"]) for ad in ads} == set(range(1, 8000)), "post-hold proc coverage mismatch")
    for ad in ads:
        proc = int(ad["ProcId"])
        require(ad.get("JobStatus") == 5, f"proc {proc} did not enter controlled held state")
        require(ad.get("Cmd") == str(ORIGINAL_RUNNER), f"proc {proc} Cmd changed before qedit")
        require(ad.get("TransferInput") == items[proc]["original_transfer_input"], f"proc {proc} TransferInput changed before qedit")
        require(int(ad.get("NumJobStarts", -1)) == 0, f"proc {proc} executable start count changed")


def require_all_held_edited(ads: list[dict[str, object]], items: dict[int, dict[str, object]]) -> None:
    require(len(ads) == TARGET_COUNT, f"post-edit ad count mismatch: {len(ads)}")
    require({int(ad["ProcId"]) for ad in ads} == set(range(1, 8000)), "post-edit proc coverage mismatch")
    for ad in ads:
        proc = int(ad["ProcId"])
        require(ad.get("JobStatus") == 5, f"proc {proc} not held after qedit")
        require(ad.get("Cmd") == str(RECOVERY), f"proc {proc} recovery Cmd edit did not close")
        require(ad.get("TransferInput") == RECOVERY_TRANSFER_INPUT, f"proc {proc} recovery TransferInput edit did not close")
        require(ad.get("Arguments") == items[proc]["arguments"], f"proc {proc} scientific Arguments changed")
        require(ad.get("Iwd") == items[proc]["return_dir"], f"proc {proc} Iwd changed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run all read-only gates and print their summary without creating evidence or editing jobs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(not EVIDENCE.exists(), f"cluster-wide recovery evidence already exists; refusing to repeat: {EVIDENCE}")
    require(EVIDENCE.parent.is_dir(), f"recovery evidence parent absent: {EVIDENCE.parent}")
    git_canary = validate_canary_and_git()
    items = load_queue_items()
    pre_ads = query_queue()
    pre_history = query_history()
    scheduler_gate = validate_scheduler_preflight(pre_ads, pre_history, items)

    if args.preflight_only:
        print(json.dumps({
            "status": "pass_escalation800_cluster_in_place_transport_recovery_read_only_preflight",
            "cluster_id": CLUSTER,
            "target_proc_range": [1, 7999],
            "git_and_canary": git_canary,
            "scheduler": scheduler_gate,
            "condor_submit_planned": False,
            "nominal_selection_changed": False,
            "validation_payloads_opened": 0,
            "test_payloads_opened": 0,
        }, indent=2, sort_keys=True))
        return 0

    attempted_utc = utc_now()
    os.mkdir(EVIDENCE, 0o700)
    fsync_directory(EVIDENCE.parent)
    durable_write(EVIDENCE / "recovery_driver.py", Path(__file__).read_bytes(), 0o700)
    durable_write(EVIDENCE / "pre_recovery_job_ads.json", json_bytes(pre_ads))
    durable_write(EVIDENCE / "pre_recovery_history.json", json_bytes(pre_history))
    durable_write(EVIDENCE / "pre_recovery_gate.json", json_bytes({
        "status": "pass_escalation800_cluster_in_place_transport_recovery_preflight",
        "attempted_utc": attempted_utc,
        "cluster_id": CLUSTER,
        "target_proc_range": [1, 7999],
        "git_and_canary": git_canary,
        "scheduler": scheduler_gate,
        "condor_submit_planned": False,
        "nominal_selection_changed": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }))
    durable_write(EVIDENCE / "CLUSTER_RECOVERY_ATTEMPTED_DO_NOT_REPEAT.txt", (
        "status=cluster_30020809_procs_1_7999_in_place_recovery_attempt_marker\n"
        f"attempted_utc={attempted_utc}\n"
        f"repository_head={EXPECTED_REPOSITORY_HEAD}\n"
        "rule=DO_NOT_CALL_CONDOR_SUBMIT\n"
        "rule=DO_NOT_REPEAT_THIS_RECOVERY_DRIVER\n"
        "rule=ONLY_EXISTING_CLUSTER_30020809_PROCS_1_7999_ARE_TARGETED\n"
    ).encode())

    hold_constraint = (
        f"ClusterId == {CLUSTER} && ProcId != {CANARY_PROC} && "
        "(JobStatus == 1 || JobStatus == 2)"
    )
    active_before_hold = (
        int(scheduler_gate["status_counts"]["1"])
        + int(scheduler_gate["status_counts"]["2"])
    )
    if active_before_hold:
        hold = command([
            "/usr/bin/bash", "/usr/local/bin/condor_hold", "-name", SCHEDD,
            "-reason", "Controlled hold for audited EOS-FUSE-to-XRootD transport-only recovery; no resubmission",
            "-constraint", hold_constraint,
        ])
    else:
        hold = subprocess.CompletedProcess(
            args=["condor_hold", "SKIPPED_ALL_TARGETS_ALREADY_HELD"],
            returncode=0,
            stdout=(
                f"SKIPPED=TRUE\nREASON=ALL_{TARGET_COUNT}_TARGETS_ALREADY_HELD\n"
            ).encode(),
            stderr=b"",
        )
    record_command("condor_hold_idle_and_transferring", hold)
    require(hold.returncode == 0, "controlled cluster hold failed after durable attempt marker")

    deadline = time.monotonic() + 45.0
    while True:
        held_ads = query_queue()
        if len(held_ads) == TARGET_COUNT and all(ad.get("JobStatus") == 5 for ad in held_ads):
            break
        require(time.monotonic() < deadline, "not all targets entered held state within 45 seconds; inspect durable evidence")
        time.sleep(2.0)
    require_all_held_original(held_ads, items)
    durable_write(EVIDENCE / "post_controlled_hold_job_ads.json", json_bytes(held_ads))

    edit_constraint = f"ClusterId == {CLUSTER} && ProcId != {CANARY_PROC} && JobStatus == 5"
    edit = command([
        "/usr/bin/bash", "/usr/local/bin/condor_qedit", "-name", SCHEDD,
        "-constraint", edit_constraint,
        "Cmd", json.dumps(str(RECOVERY)),
        "TransferInput", json.dumps(RECOVERY_TRANSFER_INPUT),
    ])
    record_command("condor_qedit_transport_attributes", edit)
    require(edit.returncode == 0, "cluster transport qedit failed; targets remain held")

    edited_ads = query_queue()
    require_all_held_edited(edited_ads, items)
    durable_write(EVIDENCE / "post_edit_pre_release_job_ads.json", json_bytes(edited_ads))

    release_constraint = (
        f"ClusterId == {CLUSTER} && ProcId != {CANARY_PROC} && JobStatus == 5 && "
        f"Cmd == {json.dumps(str(RECOVERY))}"
    )
    release = command([
        "/usr/bin/bash", "/usr/local/bin/condor_release", "-name", SCHEDD,
        "-reason", "Release existing audited cluster after transport-only XRootD recovery; no resubmission",
        "-constraint", release_constraint,
