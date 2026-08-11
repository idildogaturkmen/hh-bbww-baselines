#!/usr/bin/env python3
"""Freeze the accepted exactly-once validation submission before monitoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


EXPECTED_FILES = {
    "submission_driver.py",
    "pre_submit_gate.json",
    "SUBMISSION_ATTEMPTED_DO_NOT_RERUN.txt",
    "condor_submit_terse_output.txt",
    "condor_submit_rc.txt",
    "CONDOR_ACCEPTED_DO_NOT_RESUBMIT.txt",
    "condor_submit_parse.json",
    "validation_submission_receipt.json",
}
EXISTING_CLUSTERS = [3754344, 3755882, 3768139, 30002685, 30020809]


class ValidationSubmissionFreezeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationSubmissionFreezeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_marker(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        require("=" in line, f"invalid durable marker line: {path}: {line!r}")
        key, value = line.split("=", 1)
        require(key and key not in rows, f"duplicate/empty durable marker key: {path}: {key!r}")
        rows[key] = value
    return rows


def validate_submission_payloads(
    gate: dict[str, Any], parsed: dict[str, Any], receipt: dict[str, Any]
) -> None:
    require(
        gate.get("status") == "pass_exactly_once_cut_baseline_validation_submission_preflight",
        "validation submission preflight status changed",
    )
    require(gate.get("expected_jobs") == 116, "validation preflight job count changed")
    require(gate.get("no_prior_scheduler_association") is True, "prior validation scheduler association existed")
    require(gate.get("remote_output_namespace_absent") is True, "validation output namespace existed before submission")
    require(gate.get("production_submission_performed") is False, "preflight reports an earlier submission")
    require(gate.get("validation_payloads_opened_before_submission") == 0, "validation was opened before submission")
    require(gate.get("test_payloads_opened") == 0, "test was opened before submission")
    require(
        parsed.get("status") == "pass_single_contiguous_validation_116_condor_submission_parse",
        "validation submit parse status changed",
    )
    require(
        parsed.get("first_proc") == 0
        and parsed.get("last_proc") == 115
        and parsed.get("parsed_job_count") == 116,
        "validation proc range changed",
    )
    require(
        receipt.get("status")
        == "pass_one_time_cut_baseline_validation_condor_submission_accepted_and_parsed",
        "validation submission receipt status changed",
    )
    require(receipt.get("cluster_id") == parsed.get("cluster_id"), "validation cluster parse/receipt mismatch")
    require(receipt.get("schedd") == parsed.get("schedd"), "validation schedd parse/receipt mismatch")
    require(bool(re.fullmatch(r"lpcschedd[0-9]+\.fnal\.gov", str(receipt.get("schedd")))), "validation schedd is invalid")
    require(receipt.get("expected_job_count") == 116, "validation receipt job count changed")
    require(receipt.get("do_not_resubmit_cluster") is True, "validation cluster is not sealed against resubmission")
    cluster = int(receipt["cluster_id"])
    require(
        receipt.get("never_resubmit_clusters") == [*EXISTING_CLUSTERS, cluster],
        "validation never-resubmit registry changed",
    )
    require(receipt.get("validation_payloads_opened_before_submission") == 0, "validation opened before submission")
    require(receipt.get("validation_execution_authorized") is True, "validation execution was not authorized")
    require(receipt.get("test_access_authorized") is False, "test access was authorized")
    require(receipt.get("test_payloads_opened") == 0, "test was opened")
    for field in ("submit_file_sha256", "queue_items_sha256", "runner_sha256"):
        require(receipt.get(field) == gate.get(field), f"submission gate/receipt mismatch: {field}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines"))
    parser.add_argument("--submission-evidence", type=Path, required=True)
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("docs/checkpoints/hh4b_cut_baseline_validation_submission_20260810_v1"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    require(repo == args.repo, "repository path is not canonical")
    require(
        subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
        == "delphes-hh4b-production",
        "branch changed",
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    remote = subprocess.check_output(
        ["git", "rev-parse", "origin/delphes-hh4b-production"], cwd=repo, text=True
    ).strip()
    require(head == remote, "local/remote HEAD mismatch")

    source = args.submission_evidence.resolve()
    require(source.is_dir() and not source.is_symlink(), "validation submission evidence is invalid")
    require(not any(path.is_symlink() for path in source.iterdir()), "validation submission evidence contains symlink")
    require({path.name for path in source.iterdir() if path.is_file()} == EXPECTED_FILES, "validation submission evidence file set changed")
    require(not (source / "POST_ACCEPTANCE_PARSE_FAILED_DO_NOT_RESUBMIT.txt").exists(), "post-acceptance parse failed")
    gate = json.loads((source / "pre_submit_gate.json").read_text(encoding="utf-8"))
    parsed = json.loads((source / "condor_submit_parse.json").read_text(encoding="utf-8"))
    receipt = json.loads((source / "validation_submission_receipt.json").read_text(encoding="utf-8"))
    validate_submission_payloads(gate, parsed, receipt)
    require((source / "condor_submit_rc.txt").read_text(encoding="utf-8") == "0\n", "condor_submit return code changed")
    require(
        receipt.get("raw_condor_submit_output_sha256")
        == sha256(source / "condor_submit_terse_output.txt"),
        "raw validation submit output SHA changed",
    )
    require(
        receipt.get("submission_attempt_marker_sha256")
        == sha256(source / "SUBMISSION_ATTEMPTED_DO_NOT_RERUN.txt"),
        "validation submission attempt marker SHA changed",
    )
    require(
        receipt.get("condor_acceptance_marker_sha256")
        == sha256(source / "CONDOR_ACCEPTED_DO_NOT_RESUBMIT.txt"),
        "validation acceptance marker SHA changed",
    )
    attempt = parse_marker(source / "SUBMISSION_ATTEMPTED_DO_NOT_RERUN.txt")
    acceptance = parse_marker(source / "CONDOR_ACCEPTED_DO_NOT_RESUBMIT.txt")
    require(
        attempt.get("status")
        == "validation_submission_attempt_marker_created_before_condor_submit",
        "validation submission attempt marker status changed",
    )
    require(
        attempt.get("submission_driver_sha256") == sha256(source / "submission_driver.py"),
        "validation submission driver differs from durable attempt marker",
    )
    require(attempt.get("submit_file_sha256") == gate.get("submit_file_sha256"), "attempt marker submit SHA changed")
    require(attempt.get("queue_items_sha256") == gate.get("queue_items_sha256"), "attempt marker queue SHA changed")
    require(attempt.get("expected_jobs") == "116", "attempt marker job count changed")
    require(
        acceptance.get("status")
        == "condor_submit_returned_zero_validation_acceptance_marker",
        "validation acceptance marker status changed",
    )
    require(acceptance.get("condor_submit_rc") == "0", "validation acceptance marker return code changed")
    require(
        acceptance.get("raw_submit_output_sha256")
        == receipt.get("raw_condor_submit_output_sha256"),
        "validation acceptance marker raw-output SHA changed",
    )
    require(acceptance.get("expected_jobs") == "116", "acceptance marker job count changed")

    output = args.output_checkpoint if args.output_checkpoint.is_absolute() else repo / args.output_checkpoint
    output = output.resolve()
    require(output.is_relative_to(repo), "validation submission checkpoint is outside repository")
    require(output.parent.is_dir() and not output.exists(), "validation submission checkpoint output exists/invalid")
    build = output.parent / f".{output.name}.build.{os.getpid()}"
    require(not build.exists(), "validation submission checkpoint build path exists")
    evidence = build / "evidence/submission"
    evidence.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.is_file():
            shutil.copy2(path, evidence / path.name)
    freeze = {
        "schema_version": 1,
        "status": "pass_exactly_once_validation_submission_frozen_before_monitoring",
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "repository_parent_head": head,
        "repository_head_at_submission": receipt["repository_head_at_submission"],
        "cluster_id": receipt["cluster_id"],
        "authoritative_schedd": receipt["schedd"],
        "jobs_submitted": 116,
        "first_proc": 0,
        "last_proc": 115,
        "do_not_resubmit_cluster": True,
        "never_resubmit_clusters": receipt["never_resubmit_clusters"],
        "submission_evidence": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(source.iterdir()) if path.is_file()
        },
        "validation_payloads_opened_before_submission": 0,
        "test_access_authorized": False,
        "test_payloads_opened": 0,
        "next": "commit_and_push_then_monitor_this_validation_cluster_read_only",
    }
    (build / "validation_submission_freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (build / "README.md").write_text(
        "# HH→4b exactly-once validation submission\n\n"
        "This checkpoint freezes the durable attempt marker, immediate Condor-acceptance "
        "marker, raw submit output, dynamic schedd parse, and never-resubmit receipt before "
        "read-only monitoring. Validation had not yet opened and test remains sealed.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in build.rglob("*") if path.is_file())
    (build / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(build).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    os.replace(build, output)
    print("HH4B_CUT_BASELINE_VALIDATION_SUBMISSION_FREEZE=PASS")
    print(f"CLUSTER_ID={receipt['cluster_id']}")
    print(f"AUTHORITATIVE_SCHEDD={receipt['schedd']}")
    print("JOBS_SUBMITTED=116")
    print("DO_NOT_RESUBMIT=TRUE")
    print("VALIDATION_PAYLOADS_OPENED_BEFORE_SUBMISSION=0")
    print("TEST_PAYLOADS_OPENED=0")


if __name__ == "__main__":
    main()
