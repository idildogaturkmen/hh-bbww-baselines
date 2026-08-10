#!/usr/bin/env python3
"""Fail-closed, exactly-once submission driver for the audited HH4b escalation800 package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


REPO = Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines")
PACKAGE = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/baselines/"
    "multivariate_cut_selection_stability_escalation800_production_package_v1_20260809"
)
EVIDENCE = PACKAGE / "evidence" / "production_submission_v1"
SUBMIT_FILE = PACKAGE / "submission" / "escalation800_category_fold.sub"
PRESUBMISSION_COMMIT = "5d730256593f346a9f47914980d9bb4796cbc23a"
EXECUTION_HEAD = "275a2aaebe83b2f1f5e7403a245b52daf30560b4"
TOOLING_HEAD = "d11cca4909fb4fcf1052cc9ec357b748efc0a616"
EXPECTED_JOBS = 8000
EXPECTED_EVALUATIONS = 216000
PACKAGE_TOKEN = (
    "multivariate_cut_selection_stability_escalation800_production_package_v1_20260809"
)
SCHEDD_RE = re.compile(r"^lpcschedd[0-9]+\.fnal\.gov$")
RANGE_RE = re.compile(r"^(\d+)\.(\d+)\s*-\s*(\d+)\.(\d+)\s*$", re.MULTILINE)
SELECTED_SCHEDD_RE = re.compile(
    r"^Attempting to submit jobs to (lpcschedd[0-9]+\.fnal\.gov)\s*$",
    re.MULTILINE,
)

EXPECTED_HASHES = {
    PACKAGE / "build" / "build_state.json":
        "3ada726955516a97a139f3dd2b985d369e420294778e42c30bea8ddc4e7ff1c5",
    PACKAGE / "evidence" / "escalation800_package_build_receipt.json":
        "387da6ff6a2b1910e40acf30f859ea657dfe01852e0e4e81dfb01098573697e4",
    PACKAGE / "evidence" / "payload_manifest.json":
        "2d29023c1fb00e18c159c35716876266e0694bcaad3c2a3de4b725a7a38634fd",
    PACKAGE / "evidence" / "payload_manifest.tsv":
        "5e24f8795552d50482401b9582d3b3ded85649f05a27743a4c17c05f56ef4ab4",
    PACKAGE / "evidence" / "escalation800_production_package_audit.json":
        "d52fb2effa8af6acbe7dc140a13fa0ac5705825f4668c9f1a9835120df91e81a",
    PACKAGE / "evidence" / "escalation800_production_package_audit.tsv":
        "ee7586a01ddb84ba674ec69620e85eea50e5a5fc521df23a112f85046fa277f6",
    PACKAGE / "submission" / "escalation800_job_matrix.tsv":
        "730853b84ca172572b15a26203115808b652ef3044125d0196cd3e67a1b53888",
    PACKAGE / "submission" / "escalation800_queue.items":
        "7fbf206d0b55627500cd11ca6bc62affb171932eead5f87c1167a99a1f633daf",
    PACKAGE / "submission" / "run_escalation800_category_fold_job.sh":
        "b5a0f96950f1b2cba99e1ed173b33a70630f10f2e1c516090920dea7137c15e5",
    SUBMIT_FILE:
        "631c32bfb32be59220d6ba14b2ac0899d8fba489a5596a0131d25d955a8c87c6",
    Path(
        "/uscms_data/d3/iturkmen/hh4b_delphes/baselines/"
        "multivariate_cut_bounded_six_job_transfer_pilot_v8_r1_20260807T004537Z/"
        "archives/hh4b_python_runtime_v8_r1.tar.gz"
    ):
        "4a9c4061e1d975f276326022454e578c5f0de3c094b9b56345f192cccc654112",
    REPO / "docs" / "checkpoints" /
        "hh4b_train_multivariate_cut_selection_stability_escalation800_"
        "presubmission_package_20260810_v1" / "presubmission_package_freeze.json":
        "1a62a955fb36247f21b4700ae4c20195c458cdf938ca522d868045a7b3733974",
}


class GateError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_write_exclusive(path: Path, payload: bytes, mode: int = 0o600) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def run_checked(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise GateError(
            f"command failed rc={result.returncode}: {command!r}; "
            f"stdout={result.stdout!r}; stderr={result.stderr!r}"
        )
    return result


def parse_submit_output(raw: bytes, advertised_schedds: list[str]) -> dict[str, object]:
    text = raw.decode("utf-8")
    ranges = RANGE_RE.findall(text)
    if len(ranges) != 1:
        raise GateError(f"expected one terse cluster range, found {len(ranges)}")
    cluster_a, first_proc, cluster_b, last_proc = (int(value) for value in ranges[0])
    if cluster_a != cluster_b:
        raise GateError("terse output spans two cluster identifiers")
    parsed_jobs = last_proc - first_proc + 1
    if first_proc != 0 or last_proc != EXPECTED_JOBS - 1 or parsed_jobs != EXPECTED_JOBS:
        raise GateError(
            f"unexpected procedure range {first_proc}-{last_proc} ({parsed_jobs} jobs)"
        )
    schedd_matches = SELECTED_SCHEDD_RE.findall(text)
    if len(schedd_matches) != 1:
        raise GateError(f"expected one wrapper-selected schedd, found {schedd_matches!r}")
    schedd = schedd_matches[0]
    if schedd not in advertised_schedds:
        raise GateError(f"selected schedd {schedd!r} was not advertised in preflight")
    return {
        "cluster_id": cluster_a,
        "first_proc": first_proc,
        "last_proc": last_proc,
        "parsed_job_count": parsed_jobs,
        "schedd": schedd,
    }


def scan_json_for_prior_submission() -> int:
    scanned = 0
    paths = list((PACKAGE / "build").rglob("*.json"))
    paths.extend((PACKAGE / "evidence").glob("*.json"))
    for path in paths:
        scanned += 1
        value = json.loads(path.read_text())
        stack = [value]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for key, child in item.items():
                    if key in {"production_submission_performed", "condor_submission_performed"} and child is True:
                        raise GateError(f"prior submission=true evidence in {path}")
                    if key in {"cluster_id", "ClusterId"}:
                        raise GateError(f"prior cluster association in {path}: {child!r}")
                    stack.append(child)
            elif isinstance(item, list):
                stack.extend(item)
    return scanned


def scheduler_preflight() -> tuple[list[str], list[dict[str, object]]]:
    status = run_checked([
        "condor_status",
        "-schedd",
        "-constraint",
        'FERMIHTC_SCHEDD_TYPE =?= "CMSLPC"',
        "-af",
        "Name",
    ])
    schedds = sorted(line.strip() for line in status.stdout.decode().splitlines() if line.strip())
    if not schedds or any(not SCHEDD_RE.fullmatch(name) for name in schedds):
        raise GateError(f"unexpected advertised schedd list: {schedds!r}")

    constraint = f'regexp("{PACKAGE_TOKEN}/returns/job_[0-9]+", Iwd)'
    scans: list[dict[str, object]] = []
    for schedd in schedds:
        for command_kind, command in (
            (
                "queue",
                ["/usr/bin/bash", "/usr/local/bin/condor_q", "-name", schedd,
                 "-constraint", constraint, "-af",
                 "ClusterId", "ProcId", "JobStatus", "Iwd"],
            ),
            (
                "history",
                ["/usr/bin/bash", "/usr/local/bin/condor_history", "-name", schedd,
                 "-constraint", constraint,
                 "-match", "10000", "-af", "ClusterId", "ProcId", "JobStatus", "Iwd"],
            ),
        ):
            result = run_checked(command)
            if result.stdout.strip():
                raise GateError(
                    f"prior {command_kind} association on {schedd}: {result.stdout!r}"
                )
            scans.append({
                "kind": command_kind,
                "schedd": schedd,
                "row_count": 0,
                "stdout_sha256": sha256_bytes(result.stdout),
                "stderr_sha256": sha256_bytes(result.stderr),
            })
    return schedds, scans


def preflight() -> dict[str, object]:
    if EVIDENCE.exists():
        raise GateError(f"submission evidence already exists; refusing to run: {EVIDENCE}")
    # Confirm that the package filesystem supports the directory fsync used by
    # every durable marker before the first mutation occurs.
    fsync_directory(EVIDENCE.parent)

    head = run_checked(["git", "rev-parse", "HEAD"], cwd=REPO).stdout.decode().strip()
    remote = run_checked(
        ["git", "rev-parse", "origin/delphes-hh4b-production"], cwd=REPO
    ).stdout.decode().strip()
    status = run_checked(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=REPO
    ).stdout
    if head != PRESUBMISSION_COMMIT or remote != PRESUBMISSION_COMMIT or status:
        raise GateError(
            f"Git gate failed: head={head}, remote={remote}, status={status!r}"
        )

    observed_hashes: dict[str, str] = {}
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise GateError(f"missing frozen input: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise GateError(f"SHA mismatch for {path}: {observed} != {expected}")
        observed_hashes[str(path)] = observed

    audit = json.loads(
        (PACKAGE / "evidence" / "escalation800_production_package_audit.json").read_text()
    )
    required_audit = {
        "status": "pass_escalation800_transfer_safe_production_package_pre_submission_audit",
        "replica_count": 800,
        "payload_archive_count": 1600,
        "condor_job_count": EXPECTED_JOBS,
        "expected_structure_evaluations": EXPECTED_EVALUATIONS,
        "production_submission_performed": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
        "nominal_selection_changed": False,
        "pilot_results_may_enter_stability_aggregation": False,
    }
    for key, expected in required_audit.items():
        if audit.get(key) != expected:
            raise GateError(f"audit gate mismatch {key}: {audit.get(key)!r} != {expected!r}")

    return_root = PACKAGE / "returns"
    return_dirs = sorted(path for path in return_root.iterdir() if path.is_dir())
    if len(return_dirs) != EXPECTED_JOBS:
        raise GateError(f"return directory count mismatch: {len(return_dirs)}")
    nonempty_returns = [str(path) for path in return_dirs if any(path.iterdir())]
    if nonempty_returns:
        raise GateError(f"nonempty return directories before submission: {nonempty_returns[:5]}")
    log_files = [str(path) for path in (PACKAGE / "logs").iterdir()]
    if log_files:
        raise GateError(f"log directory is not empty before submission: {log_files[:5]}")

    json_files_scanned = scan_json_for_prior_submission()
    schedds, scheduler_scans = scheduler_preflight()
    return {
        "advertised_cms_lpc_schedds": schedds,
        "expected_condor_jobs": EXPECTED_JOBS,
        "expected_structure_evaluations": EXPECTED_EVALUATIONS,
        "frozen_input_sha256": observed_hashes,
        "git_clean": True,
        "json_files_scanned_for_prior_submission": json_files_scanned,
        "local_and_remote_head": head,
        "no_prior_scheduler_association": True,
        "package_root": str(PACKAGE),
        "return_directory_count": len(return_dirs),
        "scheduler_scans": scheduler_scans,
        "status": "pass_escalation800_exactly_once_submission_preflight",
        "test_payloads_opened": 0,
        "validation_payloads_opened": 0,
    }


def execute_once() -> int:
    gate = preflight()
    driver_bytes = Path(__file__).read_bytes()
    driver_sha = sha256_bytes(driver_bytes)
    attempt_utc = utc_now()

    os.mkdir(EVIDENCE, 0o700)
    fsync_directory(EVIDENCE.parent)
    durable_write_exclusive(EVIDENCE / "submission_driver.py", driver_bytes, 0o700)
    durable_write_exclusive(EVIDENCE / "pre_submit_gate.json", json_bytes(gate))

    attempt = (
        "status=submission_attempt_marker_created_before_condor_submit\n"
        f"attempt_utc={attempt_utc}\n"
        f"repository_head={PRESUBMISSION_COMMIT}\n"
        f"scientific_execution_head={EXECUTION_HEAD}\n"
        f"tooling_head={TOOLING_HEAD}\n"
        f"submit_file={SUBMIT_FILE}\n"
        f"submit_file_sha256={EXPECTED_HASHES[SUBMIT_FILE]}\n"
        f"queue_items_sha256={EXPECTED_HASHES[PACKAGE / 'submission' / 'escalation800_queue.items']}\n"
        f"expected_jobs={EXPECTED_JOBS}\n"
        "schedd_policy=dynamic_lpc_wrapper_no_name_override\n"
        f"submission_driver_sha256={driver_sha}\n"
        "rule=DO_NOT_RERUN_WITHOUT_INSPECTING_CONDOR_ACCEPTANCE_STATE\n"
    ).encode()
    durable_write_exclusive(EVIDENCE / "SUBMISSION_ATTEMPTED_DO_NOT_RERUN.txt", attempt)

    # This is the single authorized submission call. The durable attempt marker exists first.
    result = subprocess.run(
        ["/usr/bin/bash", "/usr/local/bin/condor_submit", "-terse", str(SUBMIT_FILE)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    raw = result.stdout
    durable_write_exclusive(EVIDENCE / "condor_submit_terse_output.txt", raw)
    durable_write_exclusive(EVIDENCE / "condor_submit_rc.txt", f"{result.returncode}\n".encode())
    raw_sha = sha256_bytes(raw)
    accepted_utc = utc_now()

    if result.returncode != 0:
        sys.stdout.buffer.write(raw)
        sys.stdout.flush()
        raise GateError(
            "condor_submit returned nonzero after the durable attempt marker; "
            "do not rerun without determining scheduler acceptance state"
        )

    # Acceptance is made durable immediately after rc=0 and before any parsing.
    acceptance = (
        "status=condor_submit_returned_zero_acceptance_marker\n"
        f"accepted_utc={accepted_utc}\n"
        f"condor_submit_rc={result.returncode}\n"
        f"raw_submit_output_sha256={raw_sha}\n"
        f"repository_head={PRESUBMISSION_COMMIT}\n"
        f"expected_jobs={EXPECTED_JOBS}\n"
        "rule=DO_NOT_RESUBMIT_EVEN_IF_CLUSTER_PARSING_OR_LATER_BOOKKEEPING_FAILS\n"
    ).encode()
    durable_write_exclusive(EVIDENCE / "CONDOR_ACCEPTED_DO_NOT_RESUBMIT.txt", acceptance)

    try:
        parsed = parse_submit_output(raw, gate["advertised_cms_lpc_schedds"])
    except Exception as error:
        failure = (
            "status=post_acceptance_parse_failed\n"
            f"failed_utc={utc_now()}\n"
            f"error={type(error).__name__}: {error}\n"
            "rule=DO_NOT_RESUBMIT; recover the accepted cluster from raw output/scheduler state\n"
        ).encode()
        durable_write_exclusive(
            EVIDENCE / "POST_ACCEPTANCE_PARSE_FAILED_DO_NOT_RESUBMIT.txt", failure
        )
        sys.stdout.buffer.write(raw)
        sys.stdout.flush()
        raise

    parse_record = {
        **parsed,
        "schema_version": 1,
        "status": "pass_single_contiguous_escalation800_condor_submission_parse",
    }
    durable_write_exclusive(EVIDENCE / "condor_submit_parse.json", json_bytes(parse_record))

    attempt_sha = sha256_file(EVIDENCE / "SUBMISSION_ATTEMPTED_DO_NOT_RERUN.txt")
    acceptance_sha = sha256_file(EVIDENCE / "CONDOR_ACCEPTED_DO_NOT_RESUBMIT.txt")
    parse_sha = sha256_file(EVIDENCE / "condor_submit_parse.json")
    receipt = {
        **parsed,
        "authorized_replicas": [200, 999],
        "condor_acceptance_marker_sha256": acceptance_sha,
        "condor_submission_performed": True,
        "condor_submit_parse_json_sha256": parse_sha,
        "condor_submit_rc": result.returncode,
        "do_not_resubmit_cluster": True,
        "expected_job_count": EXPECTED_JOBS,
        "expected_structure_evaluations": EXPECTED_EVALUATIONS,
        "never_resubmit_clusters": [3755882, 3768139, 30002685, parsed["cluster_id"]],
        "next": "freeze_submission_receipt_then_monitor_cluster_read_only",
        "nominal_selection_changed": False,
        "package_audit_sha256": EXPECTED_HASHES[
            PACKAGE / "evidence" / "escalation800_production_package_audit.json"
        ],
        "payload_manifest_sha256": EXPECTED_HASHES[
            PACKAGE / "evidence" / "payload_manifest.json"
        ],
        "pilot_results_may_enter_stability_aggregation": False,
        "presubmission_checkpoint": PRESUBMISSION_COMMIT,
        "production_submission_performed": True,
        "queue_items_sha256": EXPECTED_HASHES[
            PACKAGE / "submission" / "escalation800_queue.items"
        ],
        "raw_condor_submit_output_sha256": raw_sha,
        "repository_head_at_submission": PRESUBMISSION_COMMIT,
        "runner_sha256": EXPECTED_HASHES[
            PACKAGE / "submission" / "run_escalation800_category_fold_job.sh"
        ],
        "runtime_archive_sha256": EXPECTED_HASHES[next(
            path for path in EXPECTED_HASHES if path.name == "hh4b_python_runtime_v8_r1.tar.gz"
        )],
        "schema_version": 1,
        "scientific_execution_repository_head": EXECUTION_HEAD,
        "status": "pass_escalation800_selection_stability_condor_submission_accepted_and_parsed",
        "submission_accepted_utc": accepted_utc,
        "submission_attempt_marker_sha256": attempt_sha,
        "submission_attempt_utc": attempt_utc,
        "submission_driver_sha256": driver_sha,
        "submit_file_sha256": EXPECTED_HASHES[SUBMIT_FILE],
        "test_payloads_opened": 0,
        "tooling_repository_head": TOOLING_HEAD,
        "validation_payloads_opened": 0,
    }
    durable_write_exclusive(EVIDENCE / "escalation800_submission_receipt.json", json_bytes(receipt))

    sys.stdout.buffer.write(raw)
    sys.stdout.flush()
    print(f"EXACTLY_ONCE_SUBMISSION=PASS")
    print(f"CLUSTER_ID={parsed['cluster_id']}")
    print(f"FIRST_PROC={parsed['first_proc']}")
    print(f"LAST_PROC={parsed['last_proc']}")
    print(f"JOB_COUNT={parsed['parsed_job_count']}")
    print(f"AUTHORITATIVE_SCHEDD={parsed['schedd']}")
    print("DO_NOT_RESUBMIT=TRUE")
    return 0


def self_test() -> int:
    synthetic = (
        b"30009999.0 - 30009999.7999\n"
        b"Querying the CMS LPC pool and trying to find an available schedd...\n\n"
        b"Attempting to submit jobs to lpcschedd6.fnal.gov\n\n"
    )
    parsed = parse_submit_output(synthetic, ["lpcschedd4.fnal.gov", "lpcschedd6.fnal.gov"])
    if parsed != {
        "cluster_id": 30009999,
        "first_proc": 0,
        "last_proc": 7999,
        "parsed_job_count": 8000,
        "schedd": "lpcschedd6.fnal.gov",
    }:
        raise GateError(f"parser self-test mismatch: {parsed!r}")
    with tempfile.TemporaryDirectory(prefix="hh4b_escalation800_submit_selftest_") as tmp:
        target = Path(tmp) / "exclusive.txt"
        durable_write_exclusive(target, b"durable\n")
        if target.read_bytes() != b"durable\n":
            raise GateError("durable-write self-test mismatch")
        try:
            durable_write_exclusive(target, b"forbidden overwrite\n")
        except FileExistsError:
            pass
        else:
            raise GateError("exclusive-write self-test allowed an overwrite")
    print("EXACTLY_ONCE_SUBMISSION_DRIVER_SELF_TEST=PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute-exactly-once", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        return self_test()
    if arguments.preflight_only:
        print(json.dumps(preflight(), indent=2, sort_keys=True))
        return 0
    return execute_once()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as error:
        print(f"GATE_ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
