#!/usr/bin/env python3
"""Fail-closed exactly-once submitter for the sealed 116-source validation.

The sole condor_submit call is guarded by a durable attempt marker.  A second
durable acceptance marker is written immediately after return code zero and
before parsing the scheduler response.
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
import sys
import tempfile
from typing import Any

from audit_hh4b_cut_baseline_validation_campaign import (
    audit_local_package,
    verify_remote_runtime_and_absent_output,
)
from prepare_hh4b_cut_baseline_validation_campaign import (
    committed_bytes,
    git,
    git_is_ancestor,
    sha256,
    sha256_bytes,
    verify_repository_and_authorization_checkpoint,
)


EXPECTED_JOBS = 116
SCHEDD_RE = re.compile(r"^lpcschedd[0-9]+\.fnal\.gov$")
RANGE_RE = re.compile(r"^(\d+)\.(\d+)\s*-\s*(\d+)\.(\d+)\s*$", re.MULTILINE)
SELECTED_SCHEDD_RE = re.compile(
    r"^Attempting to submit jobs to (lpcschedd[0-9]+\.fnal\.gov)\s*$",
    re.MULTILINE,
)
NEVER_RESUBMIT_EXISTING = [3754344, 3755882, 3768139, 30002685, 30020809]


class SubmissionGateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SubmissionGateError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


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


def run_checked(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise SubmissionGateError(
            f"command failed rc={result.returncode}: {command!r}; "
            f"stdout={result.stdout!r}; stderr={result.stderr!r}"
        )
    return result


def parse_submit_output(raw: bytes, advertised_schedds: list[str]) -> dict[str, Any]:
    text = raw.decode("utf-8")
    ranges = RANGE_RE.findall(text)
    require(len(ranges) == 1, f"expected one terse cluster range, found {len(ranges)}")
    cluster_a, first_proc, cluster_b, last_proc = (int(value) for value in ranges[0])
    require(cluster_a == cluster_b, "terse output spans two clusters")
    jobs = last_proc - first_proc + 1
    require(
        first_proc == 0 and last_proc == EXPECTED_JOBS - 1 and jobs == EXPECTED_JOBS,
        f"unexpected validation proc range {first_proc}-{last_proc} ({jobs} jobs)",
    )
    matches = SELECTED_SCHEDD_RE.findall(text)
    require(len(matches) == 1, f"expected one wrapper-selected schedd, found {matches!r}")
    schedd = matches[0]
    require(schedd in advertised_schedds, f"selected schedd was not advertised: {schedd}")
    return {
        "cluster_id": cluster_a,
        "first_proc": first_proc,
        "last_proc": last_proc,
        "parsed_job_count": jobs,
        "schedd": schedd,
    }


def verify_committed_checkpoint(
    repo: Path,
    checkpoint: Path,
    summary_path: Path,
    commit: str,
    head: str,
) -> dict[str, Any]:
    require(bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "audit checkpoint commit is invalid")
    require(git(repo, "rev-parse", commit) == commit, "audit checkpoint commit is not exact")
    require(git_is_ancestor(repo, commit, head), "audit checkpoint is not an ancestor of submission HEAD")
    sums = checkpoint / "SHA256SUMS"
    for path in (summary_path, sums):
        require(path.is_file() and not path.is_symlink(), f"missing audit checkpoint file: {path}")
        require(
            sha256_bytes(committed_bytes(repo, commit, path)) == sha256(path),
            f"audit checkpoint differs from committed bytes: {path}",
        )
    result = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=checkpoint,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(result.returncode == 0, f"audit checkpoint checksum failure: {result.stdout}{result.stderr}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def validate_presubmission_freeze(
    freeze: dict[str, Any], audit: dict[str, Any]
) -> None:
    require(
        freeze.get("status")
        == "pass_exactly_once_validation_campaign_frozen_before_submission",
        "validation campaign pre-submission freeze status changed",
    )
    implementation = freeze.get("implementation_sha256")
    require(
        isinstance(implementation, str)
        and len(implementation) == 64
        and all(character in "0123456789abcdef" for character in implementation),
        "validation campaign pre-submission freezer SHA is invalid",
    )
    require(freeze.get("condor_jobs_prepared") == EXPECTED_JOBS, "frozen validation job count changed")
    require(freeze.get("durable_marker_precedes_source_access") is True, "frozen durable marker gate changed")
    require(
        freeze.get("automatic_validation_source_rerun_authorized") is False,
        "frozen campaign authorizes a validation source rerun",
    )
    require(freeze.get("remote_output_namespace_absent") is True, "frozen validation output namespace was present")
    require(freeze.get("production_submission_performed") is False, "frozen campaign reports submission")
    require(freeze.get("validation_payloads_opened") == 0, "frozen campaign reports validation access")
    require(freeze.get("test_payloads_opened") == 0, "frozen campaign reports test access")
    require(
        freeze.get("campaign_package_sha256s_sha256")
        == audit.get("campaign_package_sha256s_sha256"),
        "frozen package differs from the frozen independent audit",
    )


def scheduler_preflight(package: Path) -> tuple[list[str], list[dict[str, Any]]]:
    status = run_checked(
        [
            "condor_status",
            "-schedd",
            "-constraint",
            'FERMIHTC_SCHEDD_TYPE =?= "CMSLPC"',
            "-af",
            "Name",
        ]
    )
    schedds = sorted(line.strip() for line in status.stdout.decode().splitlines() if line.strip())
    require(schedds and all(SCHEDD_RE.fullmatch(name) for name in schedds), f"unexpected LPC schedd list: {schedds!r}")
    runner = package / "run_validation_source.sh"
    escaped_runner = str(runner).replace('"', '\\"')
    constraint = f'Cmd == "{escaped_runner}" || JobBatchName == "hh4b_cut_baseline_one_time_validation_116"'
    scans = []
    for schedd in schedds:
        commands = (
            (
                "queue",
                [
                    "/usr/bin/bash",
                    "/usr/local/bin/condor_q",
                    "-name",
                    schedd,
                    "-constraint",
                    constraint,
                    "-af",
                    "ClusterId",
                    "ProcId",
                    "JobStatus",
                    "Cmd",
                    "JobBatchName",
                ],
            ),
            (
                "history",
                [
                    "/usr/bin/bash",
                    "/usr/local/bin/condor_history",
                    "-name",
                    schedd,
                    "-constraint",
                    constraint,
                    "-match",
                    "1000",
                    "-af",
                    "ClusterId",
                    "ProcId",
                    "JobStatus",
                    "Cmd",
                    "JobBatchName",
                ],
            ),
        )
        for kind, command in commands:
            result = run_checked(command)
            require(not result.stdout.strip(), f"prior validation {kind} association on {schedd}: {result.stdout!r}")
            scans.append(
                {
                    "kind": kind,
                    "schedd": schedd,
                    "row_count": 0,
                    "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
                    "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
                }
            )
    return schedds, scans


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    head = verify_repository_and_authorization_checkpoint(
        repo,
        args.authorization_checkpoint.resolve(),
        args.authorization.resolve(),
        args.authorization_checkpoint_commit,
    )
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    require(
        authorization.get("authorized_sha256", {}).get("validation_campaign_submitter")
        == sha256(Path(__file__).resolve()),
        "validation campaign submitter is not the authorized code",
    )
    audit_checkpoint_report = verify_committed_checkpoint(
        repo,
        args.campaign_audit_checkpoint.resolve(),
        args.campaign_audit_summary.resolve(),
        args.campaign_audit_checkpoint_commit,
        head,
    )
    require(
        audit_checkpoint_report.get("status")
        == "pass_exactly_once_cut_baseline_validation_campaign_pre_submission_audit",
        "frozen validation campaign audit status changed",
    )
    require(audit_checkpoint_report.get("production_submission_performed") is False, "frozen audit reports submission")
    require(audit_checkpoint_report.get("validation_payloads_opened") == 0, "frozen audit reports validation access")
    require(audit_checkpoint_report.get("test_payloads_opened") == 0, "frozen audit reports test access")
    freeze_path = (
        args.campaign_audit_checkpoint.resolve()
        / "validation_campaign_presubmission_freeze.json"
    )
    require(freeze_path.is_file() and not freeze_path.is_symlink(), "pre-submission freeze summary is missing")
    presubmission_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    validate_presubmission_freeze(presubmission_freeze, audit_checkpoint_report)
    require(
        presubmission_freeze.get("implementation_sha256")
        == authorization.get("authorized_sha256", {}).get(
            "validation_campaign_presubmission_freezer"
        ),
        "pre-submission freeze was not produced by the master-authorized freezer",
    )
    frozen_audit_sums = (
        args.campaign_audit_checkpoint.resolve()
        / "evidence/campaign_audit/SHA256SUMS"
    )
    require(
        presubmission_freeze.get("campaign_audit_sha256s_sha256")
        == sha256(frozen_audit_sums),
        "pre-submission freeze differs from its campaign audit evidence",
    )

    package = args.campaign_package.resolve()
    local = audit_local_package(
        package,
        args.authorization.resolve(),
        args.source_access_manifest.resolve(),
        args.validation_runtime_bundle.resolve(),
        args.validation_campaign_auditor.resolve(),
    )
    for field in (
        "campaign_package_sha256s_sha256",
        "campaign_runner_sha256",
        "campaign_submit_file_sha256",
        "campaign_queue_items_sha256",
        "campaign_common_bundle_sha256",
    ):
        require(local[field] == audit_checkpoint_report.get(field), f"campaign differs from frozen audit: {field}")
    remote = verify_remote_runtime_and_absent_output(local)
    proxy = Path(local["x509_proxy"])
    require(proxy.is_file() and not proxy.is_symlink(), "validation X509 proxy is missing/nonregular")
    proxy_lifetime = run_checked(
        ["voms-proxy-info", "-file", str(proxy), "-timeleft"]
    )
    try:
        proxy_seconds = int(proxy_lifetime.stdout.decode("utf-8").strip())
    except ValueError as error:
        raise SubmissionGateError("validation X509 proxy lifetime is not an integer") from error
    require(proxy_seconds >= 43200, f"validation X509 proxy lifetime is below 12 hours: {proxy_seconds}")
    log_root = Path(local["local_log_root"])
    require(log_root.is_dir() and not log_root.is_symlink(), "validation log root is invalid")
    require(
        not list(log_root.glob("validation.*")),
        "prior validation scheduler logs exist",
    )
    evidence = args.evidence_dir.resolve()
    require(evidence.parent.is_dir() and not evidence.exists(), "submission evidence already exists or parent is missing")
    fsync_directory(evidence.parent)
    schedds, scans = scheduler_preflight(package)
    return {
        "schema_version": 1,
        "status": "pass_exactly_once_cut_baseline_validation_submission_preflight",
        "repository_head": head,
        "authorization_checkpoint_commit": args.authorization_checkpoint_commit,
        "campaign_audit_checkpoint_commit": args.campaign_audit_checkpoint_commit,
        "validation_campaign_presubmission_freeze_sha256": sha256(freeze_path),
        "campaign_package": str(package),
        "submit_file": str(package / "validation_116.submit"),
        "submit_file_sha256": local["campaign_submit_file_sha256"],
        "queue_items_sha256": local["campaign_queue_items_sha256"],
        "runner_sha256": local["campaign_runner_sha256"],
        "expected_jobs": EXPECTED_JOBS,
        "advertised_cms_lpc_schedds": schedds,
        "scheduler_scans": scans,
        "x509_proxy": str(proxy),
        "x509_proxy_timeleft_seconds": proxy_seconds,
        "local_log_root": str(log_root),
        "no_prior_scheduler_association": True,
        **remote,
        "production_submission_performed": False,
        "validation_payloads_opened_before_submission": 0,
        "test_payloads_opened": 0,
    }


def execute_once(args: argparse.Namespace) -> int:
    gate = preflight(args)
    evidence = args.evidence_dir.resolve()
    package = args.campaign_package.resolve()
    submit_file = package / "validation_116.submit"
    attempt_utc = utc_now()
    driver_bytes = Path(__file__).read_bytes()
    driver_sha = hashlib.sha256(driver_bytes).hexdigest()

    os.mkdir(evidence, 0o700)
    fsync_directory(evidence.parent)
    durable_write_exclusive(evidence / "submission_driver.py", driver_bytes, 0o700)
    durable_write_exclusive(evidence / "pre_submit_gate.json", json_bytes(gate))
    attempt = (
        "status=validation_submission_attempt_marker_created_before_condor_submit\n"
        f"attempt_utc={attempt_utc}\n"
        f"repository_head={gate['repository_head']}\n"
        f"submit_file={submit_file}\n"
        f"submit_file_sha256={gate['submit_file_sha256']}\n"
        f"queue_items_sha256={gate['queue_items_sha256']}\n"
        f"expected_jobs={EXPECTED_JOBS}\n"
        "schedd_policy=dynamic_lpc_wrapper_no_name_override\n"
        f"submission_driver_sha256={driver_sha}\n"
        "rule=DO_NOT_RERUN_WITHOUT_INSPECTING_CONDOR_ACCEPTANCE_STATE\n"
    ).encode("utf-8")
    durable_write_exclusive(evidence / "SUBMISSION_ATTEMPTED_DO_NOT_RERUN.txt", attempt)

    # The only authorized submission call.  Do not add a -name override.
    result = subprocess.run(
        ["/usr/bin/bash", "/usr/local/bin/condor_submit", "-terse", str(submit_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    raw = result.stdout
    durable_write_exclusive(evidence / "condor_submit_terse_output.txt", raw)
    durable_write_exclusive(evidence / "condor_submit_rc.txt", f"{result.returncode}\n".encode("utf-8"))
    raw_sha = hashlib.sha256(raw).hexdigest()
    accepted_utc = utc_now()
    if result.returncode != 0:
        sys.stdout.buffer.write(raw)
        sys.stdout.flush()
        raise SubmissionGateError(
            "condor_submit returned nonzero after durable attempt marker; do not rerun until scheduler acceptance is resolved"
        )

    acceptance = (
        "status=condor_submit_returned_zero_validation_acceptance_marker\n"
        f"accepted_utc={accepted_utc}\n"
        f"condor_submit_rc={result.returncode}\n"
        f"raw_submit_output_sha256={raw_sha}\n"
        f"repository_head={gate['repository_head']}\n"
        f"expected_jobs={EXPECTED_JOBS}\n"
        "rule=DO_NOT_RESUBMIT_EVEN_IF_CLUSTER_PARSING_OR_LATER_BOOKKEEPING_FAILS\n"
    ).encode("utf-8")
    durable_write_exclusive(evidence / "CONDOR_ACCEPTED_DO_NOT_RESUBMIT.txt", acceptance)

    try:
        parsed = parse_submit_output(raw, gate["advertised_cms_lpc_schedds"])
    except Exception as error:
        durable_write_exclusive(
            evidence / "POST_ACCEPTANCE_PARSE_FAILED_DO_NOT_RESUBMIT.txt",
            (
                "status=post_acceptance_parse_failed\n"
                f"failed_utc={utc_now()}\n"
                f"error={type(error).__name__}: {error}\n"
                "rule=DO_NOT_RESUBMIT; recover accepted cluster from raw scheduler evidence\n"
            ).encode("utf-8"),
        )
        sys.stdout.buffer.write(raw)
        sys.stdout.flush()
        raise

    durable_write_exclusive(
        evidence / "condor_submit_parse.json",
        json_bytes(
            {
                "schema_version": 1,
                "status": "pass_single_contiguous_validation_116_condor_submission_parse",
                **parsed,
            }
        ),
    )
    receipt = {
        "schema_version": 1,
        "status": "pass_one_time_cut_baseline_validation_condor_submission_accepted_and_parsed",
        **parsed,
        "repository_head_at_submission": gate["repository_head"],
        "authorization_checkpoint_commit": gate["authorization_checkpoint_commit"],
        "campaign_audit_checkpoint_commit": gate["campaign_audit_checkpoint_commit"],
        "condor_submission_performed": True,
        "expected_job_count": EXPECTED_JOBS,
        "do_not_resubmit_cluster": True,
        "never_resubmit_clusters": [*NEVER_RESUBMIT_EXISTING, parsed["cluster_id"]],
        "submission_attempt_utc": attempt_utc,
        "submission_accepted_utc": accepted_utc,
        "raw_condor_submit_output_sha256": raw_sha,
        "submission_attempt_marker_sha256": sha256(evidence / "SUBMISSION_ATTEMPTED_DO_NOT_RERUN.txt"),
        "condor_acceptance_marker_sha256": sha256(evidence / "CONDOR_ACCEPTED_DO_NOT_RESUBMIT.txt"),
        "submit_file_sha256": gate["submit_file_sha256"],
        "queue_items_sha256": gate["queue_items_sha256"],
        "runner_sha256": gate["runner_sha256"],
        "validation_payloads_opened_before_submission": 0,
        "validation_execution_authorized": True,
        "test_payloads_opened": 0,
        "test_access_authorized": False,
        "next": "monitor_this_validation_cluster_read_only_and_never_rerun_a_source_with_a_durable_marker",
    }
    durable_write_exclusive(evidence / "validation_submission_receipt.json", json_bytes(receipt))
    sys.stdout.buffer.write(raw)
    sys.stdout.flush()
    print("EXACTLY_ONCE_VALIDATION_SUBMISSION=PASS")
    print(f"CLUSTER_ID={parsed['cluster_id']}")
    print(f"FIRST_PROC={parsed['first_proc']}")
    print(f"LAST_PROC={parsed['last_proc']}")
    print(f"JOB_COUNT={parsed['parsed_job_count']}")
    print(f"AUTHORITATIVE_SCHEDD={parsed['schedd']}")
    print("DO_NOT_RESUBMIT=TRUE")
    print("TEST_PAYLOADS_OPENED=0")
    return 0


def self_test() -> int:
    raw = (
        b"30009999.0 - 30009999.115\n"
        b"Querying the CMS LPC pool and trying to find an available schedd...\n\n"
        b"Attempting to submit jobs to lpcschedd6.fnal.gov\n\n"
    )
    parsed = parse_submit_output(raw, ["lpcschedd4.fnal.gov", "lpcschedd6.fnal.gov"])
    require(
        parsed
        == {
            "cluster_id": 30009999,
            "first_proc": 0,
            "last_proc": 115,
            "parsed_job_count": 116,
            "schedd": "lpcschedd6.fnal.gov",
        },
        f"parser self-test mismatch: {parsed!r}",
    )
    with tempfile.TemporaryDirectory(prefix="hh4b_validation_submit_selftest_") as directory:
        target = Path(directory) / "exclusive.txt"
        durable_write_exclusive(target, b"durable\n")
        try:
            durable_write_exclusive(target, b"forbidden\n")
        except FileExistsError:
            pass
        else:
            raise SubmissionGateError("exclusive durable write permitted overwrite")
    print("EXACTLY_ONCE_VALIDATION_SUBMISSION_DRIVER_SELF_TEST=PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute-exactly-once", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--authorization-checkpoint", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-checkpoint-commit")
    parser.add_argument("--source-access-manifest", type=Path)
    parser.add_argument("--validation-runtime-bundle", type=Path)
    parser.add_argument("--validation-campaign-auditor", type=Path)
    parser.add_argument("--campaign-package", type=Path)
    parser.add_argument("--campaign-audit-checkpoint", type=Path)
    parser.add_argument("--campaign-audit-summary", type=Path)
    parser.add_argument("--campaign-audit-checkpoint-commit")
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    required = (
        "repo",
        "authorization_checkpoint",
        "authorization",
        "authorization_checkpoint_commit",
        "source_access_manifest",
        "validation_runtime_bundle",
        "validation_campaign_auditor",
        "campaign_package",
        "campaign_audit_checkpoint",
        "campaign_audit_summary",
        "campaign_audit_checkpoint_commit",
        "evidence_dir",
    )
    for name in required:
        require(getattr(args, name) is not None, f"--{name.replace('_', '-')} is required")
    if args.preflight_only:
        print(json.dumps(preflight(args), indent=2, sort_keys=True, allow_nan=False))
        return 0
    return execute_once(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SubmissionGateError as error:
        print(f"SUBMISSION_GATE_ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
