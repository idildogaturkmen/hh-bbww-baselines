#!/usr/bin/env python3
"""Freeze the audited 116-job validation package before its one submit call."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


class ValidationCampaignFreezeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationCampaignFreezeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256s(root: Path) -> None:
    require(root.is_dir() and not root.is_symlink(), f"invalid evidence root: {root}")
    require((root / "SHA256SUMS").is_file(), f"missing SHA256SUMS: {root}")
    result = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(result.returncode == 0, f"checksum failure: {root}: {result.stdout}{result.stderr}")


def validate_campaign_freeze_inputs(
    package: dict[str, Any], audit: dict[str, Any]
) -> None:
    require(
        package.get("status")
        == "pass_prepared_exactly_once_cut_baseline_validation_campaign",
        "validation campaign package status changed",
    )
    require(
        audit.get("status")
        == "pass_exactly_once_cut_baseline_validation_campaign_pre_submission_audit",
        "validation campaign audit status changed",
    )
    require(package.get("condor_jobs_prepared") == audit.get("condor_jobs") == 116, "validation job count changed")
    require(package.get("authorized_physical_validation_sources") == 116, "physical validation source count changed")
    require(package.get("auxiliary_qcd_validation_sources_in_queue") == 0, "auxiliary QCD entered queue")
    require(package.get("durable_marker_created_before_each_source_access") is True, "durable marker contract changed")
    require(package.get("automatic_source_rerun_authorized") is False, "source rerun was authorized")
    require(audit.get("durable_marker_precedes_source_access") is True, "audited durable marker contract changed")
    require(audit.get("automatic_validation_rerun_authorized") is False, "audit authorized a validation rerun")
    require(audit.get("remote_output_namespace_absent") is True, "remote validation namespace is not absent")
    for payload, label in ((package, "package"), (audit, "audit")):
        require(payload.get("production_submission_performed") is False, f"{label} reports submission")
        require(payload.get("validation_payloads_opened") == 0, f"{label} reports validation access")
        require(payload.get("test_payloads_opened") == 0, f"{label} reports test access")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines"))
    parser.add_argument("--campaign-package", type=Path, required=True)
    parser.add_argument("--campaign-audit", type=Path, required=True)
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("docs/checkpoints/hh4b_cut_baseline_validation_campaign_presubmission_20260810_v1"),
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
    package_root = args.campaign_package.resolve()
    audit_root = args.campaign_audit.resolve()
    verify_sha256s(package_root)
    verify_sha256s(audit_root)
    require(not any(path.is_symlink() for path in package_root.rglob("*")), "campaign package contains symlink")
    require(not any(path.is_symlink() for path in audit_root.rglob("*")), "campaign audit contains symlink")
    package = json.loads((package_root / "campaign_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((audit_root / "validation_campaign_audit.json").read_text(encoding="utf-8"))
    validate_campaign_freeze_inputs(package, audit)
    require(
        audit.get("campaign_package_sha256s_sha256") == sha256(package_root / "SHA256SUMS"),
        "campaign package differs from audit",
    )

    output = args.output_checkpoint if args.output_checkpoint.is_absolute() else repo / args.output_checkpoint
    output = output.resolve()
    require(output.is_relative_to(repo), "checkpoint output is outside repository")
    require(output.parent.is_dir() and not output.exists(), "checkpoint output path is invalid/exists")
    build = output.parent / f".{output.name}.build.{os.getpid()}"
    require(not build.exists(), "checkpoint build path exists")
    build.mkdir()
    evidence = build / "evidence"
    evidence.mkdir()
    shutil.copytree(package_root, evidence / "campaign_package")
    shutil.copytree(audit_root, evidence / "campaign_audit")
    freeze = {
        "schema_version": 1,
        "status": "pass_exactly_once_validation_campaign_frozen_before_submission",
        "repository_parent_head": head,
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "campaign_package_sha256s_sha256": sha256(package_root / "SHA256SUMS"),
        "campaign_audit_sha256s_sha256": sha256(audit_root / "SHA256SUMS"),
        "campaign_summary_sha256": sha256(package_root / "campaign_summary.json"),
        "campaign_audit_summary_sha256": sha256(audit_root / "validation_campaign_audit.json"),
        "condor_jobs_prepared": 116,
        "durable_marker_precedes_source_access": True,
        "automatic_validation_source_rerun_authorized": False,
        "remote_output_namespace_absent": True,
        "production_submission_performed": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
        "next": "commit_and_push_then_run_exactly_once_validation_submission_preflight_and_one_submit_call",
    }
    (build / "validation_campaign_presubmission_freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (build / "README.md").write_text(
        "# HH→4b one-time validation campaign pre-submission freeze\n\n"
        "The 116-source package and independent audit are frozen here before "
        "the only authorized `condor_submit` call. Every source requires an "
        "exclusive durable marker before access; automatic reruns are forbidden; "
        "test remains sealed.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in build.rglob("*") if path.is_file())
    (build / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(build).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    os.replace(build, output)
    print("CUT_BASELINE_VALIDATION_CAMPAIGN_PRESUBMISSION_FREEZE=PASS")
    print("CONDOR_JOBS_PREPARED=116")
    print("PRODUCTION_SUBMISSION_PERFORMED=FALSE")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")


if __name__ == "__main__":
    main()
