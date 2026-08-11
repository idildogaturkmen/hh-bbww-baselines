#!/usr/bin/env python3
"""Independently audit the sealed one-time HH->4b validation campaign.

This audit is metadata-only.  It verifies that the remote result namespace does
not yet exist and never reads any validation event payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
from typing import Any
import zlib

import pandas as pd

from prepare_hh4b_cut_baseline_validation_campaign import (
    COMMON_MEMBERS,
    CampaignPreparationError,
    git_is_ancestor,
    render_runner,
    render_submit,
    sha256,
    truthy,
    validate_remote_path,
    verify_repository_and_authorization_checkpoint,
)


class CampaignAuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignAuditError(message)


def local_adler32(path: Path) -> str:
    checksum = 1
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum = zlib.adler32(block, checksum)
    return f"{checksum & 0xffffffff:08x}"


def check_sha256s(root: Path) -> None:
    result = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(result.returncode == 0, f"campaign checksum failure: {result.stdout}{result.stderr}")


def read_common_bundle(path: Path) -> dict[str, bytes]:
    expected = set(COMMON_MEMBERS) | {"COMMON_SHA256SUMS"}
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        require({member.name for member in members} == expected, "common bundle member set changed")
        require(all(member.isfile() for member in members), "common bundle contains a non-file member")
        require(
            all(
                not member.name.startswith("/")
                and ".." not in Path(member.name).parts
                for member in members
            ),
            "unsafe common bundle member",
        )
        payloads = {
            member.name: archive.extractfile(member).read()
            for member in members
        }
    manifest = payloads["COMMON_SHA256SUMS"].decode("utf-8").splitlines()
    observed = {}
    for line in manifest:
        expected_sha, name = line.split("  ", 1)
        require(name in payloads and name != "COMMON_SHA256SUMS", f"bad common checksum member: {name}")
        require(hashlib.sha256(payloads[name]).hexdigest() == expected_sha, f"common member checksum failed: {name}")
        observed[name] = expected_sha
    require(set(observed) == expected - {"COMMON_SHA256SUMS"}, "common checksum closure changed")
    return payloads


def audit_local_package(
    package: Path,
    authorization_path: Path,
    source_access_manifest: Path,
    runtime_bundle: Path,
    campaign_auditor: Path,
) -> dict[str, Any]:
    require(package.is_dir() and not package.is_symlink(), "campaign package is invalid")
    expected_files = {
        "SHA256SUMS",
        "artifact_manifest.tsv",
        "campaign_summary.json",
        "run_validation_source.sh",
        "validation_116.submit",
        "validation_common_bundle.tar.gz",
        "validation_queue.items",
    }
    actual_files = {path.name for path in package.iterdir() if path.is_file()}
    require(actual_files == expected_files, "campaign package file set changed or submission evidence leaked in")
    check_sha256s(package)

    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    summary = json.loads((package / "campaign_summary.json").read_text(encoding="utf-8"))
    require(
        summary.get("status")
        == "pass_prepared_exactly_once_cut_baseline_validation_campaign",
        "campaign preparation status changed",
    )
    require(summary.get("production_submission_performed") is False, "campaign reports submission")
    require(summary.get("validation_payloads_opened") == 0, "validation reports opened payloads")
    require(summary.get("test_payloads_opened") == 0, "test reports opened payloads")
    require(summary.get("condor_jobs_prepared") == 116, "prepared job count changed")
    require(summary.get("durable_marker_created_before_each_source_access") is True, "durable marker contract changed")
    require(summary.get("automatic_source_rerun_authorized") is False, "automatic validation rerun was authorized")
    require(summary.get("authorization_sha256") == sha256(authorization_path), "authorization SHA changed")
    authorized_hashes = authorization.get("authorized_sha256", {})
    require(
        authorized_hashes.get("source_access_manifest") == sha256(source_access_manifest),
        "authorized source access manifest changed",
    )
    require(
        authorized_hashes.get("validation_runtime_bundle") == sha256(runtime_bundle),
        "authorized validation runtime changed",
    )
    require(
        authorized_hashes.get("validation_campaign_auditor") == sha256(campaign_auditor),
        "authorized campaign auditor changed",
    )
    require(summary.get("validation_runtime_bundle_sha256") == sha256(runtime_bundle), "runtime package SHA changed")

    access = pd.read_csv(source_access_manifest, sep="\t", keep_default_na=False)
    physical = access.loc[access["physical_evaluation_eligible"].map(truthy)].copy()
    auxiliary = access.loc[~access["physical_evaluation_eligible"].map(truthy)].copy()
    require(len(access) == 121 and len(physical) == 116 and len(auxiliary) == 5, "validation source closure changed")
    require(auxiliary["auxiliary_qcd"].map(truthy).all(), "nonphysical validation source is not auxiliary QCD")
    expected_queue = [
        f"{int(row.production_row_index)} {row.source_uid}"
        for row in physical.sort_values("production_row_index").itertuples(index=False)
    ]
    observed_queue = (package / "validation_queue.items").read_text(encoding="utf-8").splitlines()
    require(observed_queue == expected_queue, "validation queue/source identity changed")
    require(len(observed_queue) == len(set(observed_queue)) == 116, "validation queue is not unique")

    common_path = package / "validation_common_bundle.tar.gz"
    common = read_common_bundle(common_path)
    expected_common_hashes = summary.get("validation_common_member_sha256", {})
    for name in set(COMMON_MEMBERS):
        require(
            hashlib.sha256(common[name]).hexdigest() == expected_common_hashes.get(name),
            f"campaign summary common-member SHA changed: {name}",
        )
    require(common["validation_authorization.json"] == authorization_path.read_bytes(), "common authorization bytes changed")
    require(common["source_access_manifest.tsv"] == source_access_manifest.read_bytes(), "common source manifest bytes changed")

    expected_runner = render_runner(
        execution_head=summary["repository_head"],
        authorization_head=summary["authorization_repository_head"],
        authorization_sha=summary["authorization_sha256"],
        common_sha=summary["validation_common_bundle_sha256"],
        runtime_sha=summary["validation_runtime_bundle_sha256"],
        runtime_remote_path=summary["validation_runtime_remote_path"],
        remote_output_root=summary["remote_output_root"],
    )
    runner_path = package / "run_validation_source.sh"
    require(runner_path.read_text(encoding="utf-8") == expected_runner, "validation runner is not the authorized deterministic rendering")
    syntax = subprocess.run(
        ["bash", "-n", str(runner_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(syntax.returncode == 0, f"validation runner shell syntax failed: {syntax.stderr}")
    marker_gate = 'xrdcp --nopbar --cksum adler32 "$LOCAL_MARKER"'
    worker_start = 'python3 "$SCRATCH/common/worker.py"'
    require(marker_gate in expected_runner and worker_start in expected_runner, "marker/worker gate missing")
    require(expected_runner.index(marker_gate) < expected_runner.index("MARKER_CREATED=TRUE") < expected_runner.index(worker_start), "marker gate ordering changed")
    require('xrdcp -f --nopbar --cksum adler32 "$LOCAL_MARKER"' not in expected_runner, "marker overwrite was enabled")

    expected_submit = render_submit(
        package_root=package.resolve(),
        log_root=Path(summary["local_log_root"]),
        x509_proxy=Path(summary["x509_proxy"]),
        apptainer_image=summary["apptainer_image"],
    )
    submit_path = package / "validation_116.submit"
    require(submit_path.read_text(encoding="utf-8") == expected_submit, "validation submit file is not deterministic")
    submit_text = expected_submit.lower()
    require("queue row_index, source_uid" in submit_text, "validation queue statement changed")
    require("max_retries" not in submit_text and "retry" not in submit_text, "validation retry directive detected")
    command_directives = []
    for line in submit_text.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() in {"executable", "arguments"}:
            command_directives.append(value.strip())
    require(
        not any(
            re.search(r"(^|[/\s])condor_submit(?:\s|$)", value)
            for value in command_directives
        ),
        "nested submission command detected",
    )

    artifact_manifest = pd.read_csv(package / "artifact_manifest.tsv", sep="\t", keep_default_na=False)
    expected_manifest_names = expected_files - {"SHA256SUMS", "artifact_manifest.tsv"}
    require(set(artifact_manifest["relative_path"]) == expected_manifest_names, "campaign artifact manifest closure changed")
    for row in artifact_manifest.itertuples(index=False):
        path = package / row.relative_path
        require(int(row.bytes) == path.stat().st_size and row.sha256 == sha256(path), f"artifact manifest mismatch: {path}")

    return {
        "repository_head": summary["repository_head"],
        "authorization_checkpoint_commit": summary["authorization_checkpoint_commit"],
        "remote_output_root": validate_remote_path(summary["remote_output_root"], "validation output root"),
        "runtime_remote_path": validate_remote_path(summary["validation_runtime_remote_path"], "runtime remote path"),
        "x509_proxy": summary["x509_proxy"],
        "local_log_root": summary["local_log_root"],
        "runtime_local_adler32": local_adler32(runtime_bundle),
        "campaign_package_sha256s_sha256": sha256(package / "SHA256SUMS"),
        "campaign_runner_sha256": sha256(package / "run_validation_source.sh"),
        "campaign_submit_file_sha256": sha256(package / "validation_116.submit"),
        "campaign_queue_items_sha256": sha256(package / "validation_queue.items"),
        "campaign_common_bundle_sha256": sha256(package / "validation_common_bundle.tar.gz"),
        "condor_jobs": 116,
        "physical_validation_sources": 116,
        "auxiliary_qcd_sources": 5,
        "production_submission_performed": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }


def xrdfs(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["xrdfs", "root://cmseos.fnal.gov", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def verify_remote_runtime_and_absent_output(local: dict[str, Any]) -> dict[str, Any]:
    checksum = xrdfs("query", "checksum", local["runtime_remote_path"])
    require(checksum.returncode == 0, f"remote runtime checksum query failed: {checksum.stderr}")
    tokens = checksum.stdout.strip().lower().split()
    require(len(tokens) >= 2 and tokens[-2] == "adler32", "remote runtime checksum response changed")
    remote_adler = tokens[-1].removeprefix("0x").zfill(8)
    require(remote_adler == local["runtime_local_adler32"], "remote runtime differs from authorized local bundle")

    remote_output = local["remote_output_root"]
    parent = str(Path(remote_output).parent)
    listing = xrdfs("ls", parent)
    require(listing.returncode == 0, f"cannot prove validation output namespace absence: {listing.stderr}")
    entries = {line.strip().rstrip("/") for line in listing.stdout.splitlines() if line.strip()}
    require(remote_output not in entries, "validation output namespace already exists")
    return {
        "runtime_remote_checksum_query": checksum.stdout.strip(),
        "remote_runtime_adler32": remote_adler,
        "remote_output_parent": parent,
        "remote_output_namespace_absent": True,
        "remote_output_parent_listing_sha256": hashlib.sha256(listing.stdout.encode("utf-8")).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--authorization-checkpoint", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-checkpoint-commit", required=True)
    parser.add_argument("--source-access-manifest", type=Path, required=True)
    parser.add_argument("--validation-runtime-bundle", type=Path, required=True)
    parser.add_argument("--campaign-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    head = verify_repository_and_authorization_checkpoint(
        repo,
        args.authorization_checkpoint.resolve(),
        args.authorization.resolve(),
        args.authorization_checkpoint_commit,
    )
    local = audit_local_package(
        args.campaign_package.resolve(),
        args.authorization.resolve(),
        args.source_access_manifest.resolve(),
        args.validation_runtime_bundle.resolve(),
        Path(__file__).resolve(),
    )
    require(
        git_is_ancestor(repo, local["repository_head"], head),
        "campaign preparation head is not an ancestor of audit head",
    )
    remote = verify_remote_runtime_and_absent_output(local)

    output = args.output_dir.resolve()
    require(output.parent.is_dir() and not output.exists(), "audit output path is invalid or already exists")
    output.mkdir()
    report = {
        "schema_version": 1,
        "status": "pass_exactly_once_cut_baseline_validation_campaign_pre_submission_audit",
        "audit_repository_head": head,
        **local,
        **remote,
        "durable_marker_precedes_source_access": True,
        "automatic_validation_rerun_authorized": False,
        "production_submission_performed": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }
    (output / "validation_campaign_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    inventory = []
    for path in sorted(args.campaign_package.resolve().iterdir()):
        if path.is_file():
            inventory.append(
                {"relative_path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    pd.DataFrame(inventory).to_csv(
        output / "validation_campaign_inventory.tsv", sep="\t", index=False, lineterminator="\n"
    )
    files = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    print("CUT_BASELINE_VALIDATION_CAMPAIGN_AUDIT=PASS")
    print("REMOTE_OUTPUT_NAMESPACE_ABSENT=TRUE")
    print("CONDOR_JOBS=116")
    print("PRODUCTION_SUBMISSION_PERFORMED=FALSE")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")


if __name__ == "__main__":
    try:
        main()
    except CampaignPreparationError as error:
        raise CampaignAuditError(str(error)) from error
