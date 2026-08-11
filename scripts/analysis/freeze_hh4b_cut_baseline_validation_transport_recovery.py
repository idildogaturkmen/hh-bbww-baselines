#!/usr/bin/env python3
"""Freeze the one allowed in-place transport repair of validation cluster 3795859."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


CLUSTER = 3795859
SCHEDD = "lpcschedd4.fnal.gov"
TARGET_COUNT = 116
EXPECTED_FILES = {
    "RECOVERY_ATTEMPTED_DO_NOT_REPEAT.txt",
    "RECOVERY_RELEASED_MONITOR_ONLY.txt",
    "condor_qedit_transport_attributes.rc",
    "condor_qedit_transport_attributes.stderr",
    "condor_qedit_transport_attributes.stdout",
    "condor_release_recovered_targets.rc",
    "condor_release_recovered_targets.stderr",
    "condor_release_recovered_targets.stdout",
    "post_edit_pre_release_job_ads.json",
    "post_release_job_ads.json",
    "pre_recovery_gate.json",
    "pre_recovery_job_ads.json",
    "recovery_driver.py",
    "transport_recovery_receipt.json",
}


class RecoveryFreezeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryFreezeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_ads(
    ads: list[dict[str, Any]], *, expected_cmd: str, expected_transfer: str
) -> dict[str, int]:
    require(len(ads) == TARGET_COUNT, f"job-ad count changed: {len(ads)}")
    by_proc = {int(ad["ProcId"]): ad for ad in ads}
    require(set(by_proc) == set(range(TARGET_COUNT)), "job-ad proc coverage changed")
    status_counts: dict[str, int] = {}
    for proc, ad in by_proc.items():
        require(ad.get("ClusterId") == CLUSTER, f"cluster changed for proc {proc}")
        require(ad.get("Cmd") == expected_cmd, f"Cmd changed for proc {proc}")
        require(ad.get("TransferInput") == expected_transfer, f"TransferInput changed for proc {proc}")
        require(ad.get("NumJobStarts") == 0, f"proc {proc} started before recovery snapshot")
        key = str(ad.get("JobStatus"))
        status_counts[key] = status_counts.get(key, 0) + 1
    return status_counts


def validate_evidence(repo: Path, evidence: Path) -> dict[str, Any]:
    require(evidence.is_dir() and not evidence.is_symlink(), "recovery evidence root is invalid")
    require(not any(path.is_symlink() for path in evidence.rglob("*")), "recovery evidence contains symlink")
    files = {path.name for path in evidence.iterdir() if path.is_file()}
    require(files == EXPECTED_FILES, f"recovery evidence file closure changed: {files ^ EXPECTED_FILES}")

    gate = read_json(evidence / "pre_recovery_gate.json")
    receipt = read_json(evidence / "transport_recovery_receipt.json")
    require(
        gate.get("status") == "pass_validation_cluster_in_place_transport_recovery_preflight",
        "recovery preflight status changed",
    )
    require(
        receipt.get("status")
        == "pass_validation_cluster_in_place_transport_recovery_edited_and_released",
        "recovery receipt status changed",
    )
    for payload, label in ((gate, "gate"), (receipt, "receipt")):
        require(payload.get("cluster_id") == CLUSTER, f"{label} cluster changed")
        require(payload.get("authoritative_schedd") == SCHEDD, f"{label} schedd changed")
        require(payload.get("target_count") == TARGET_COUNT, f"{label} target count changed")
        require(payload.get("test_payloads_opened") == 0, f"test opened in {label}")
    require(gate.get("validation_payloads_opened") == 0, "validation opened before recovery")
    require(gate.get("condor_submit_planned") is False, "recovery gate planned a submission")
    require(receipt.get("validation_payloads_opened_before_release") == 0, "validation opened before release")
    require(receipt.get("condor_submit_called") is False, "recovery called condor_submit")
    require(receipt.get("never_resubmit_cluster") is True, "never-resubmit flag changed")
    require(receipt.get("transport_recovery_only") is True, "recovery is not transport-only")
    require(receipt.get("scientific_arguments_changed") is False, "scientific arguments changed")
    require(receipt.get("nominal_selection_changed") is False, "nominal selection changed")
    require((evidence / "condor_qedit_transport_attributes.rc").read_text().strip() == "0", "qedit failed")
    require((evidence / "condor_release_recovered_targets.rc").read_text().strip() == "0", "release failed")
    qedit_stdout = (evidence / "condor_qedit_transport_attributes.stdout").read_text()
    require(qedit_stdout.count("for 116 matching jobs") == 2, "qedit target closure changed")
    release_stdout = (evidence / "condor_release_recovered_targets.stdout").read_text()
    require("have been released" in release_stdout, "release acceptance evidence changed")

    original_cmd = gate["original_cmd"]
    original_transfer = gate["original_transfer_input"]
    recovery_cmd = gate["recovery_cmd"]
    recovery_transfer = gate["recovery_transfer_input"]
    pre = read_json(evidence / "pre_recovery_job_ads.json")
    edited = read_json(evidence / "post_edit_pre_release_job_ads.json")
    released = read_json(evidence / "post_release_job_ads.json")
    pre_counts = validate_ads(pre, expected_cmd=original_cmd, expected_transfer=original_transfer)
    edited_counts = validate_ads(edited, expected_cmd=recovery_cmd, expected_transfer=recovery_transfer)
    released_counts = validate_ads(released, expected_cmd=recovery_cmd, expected_transfer=recovery_transfer)
    require(pre_counts == {"5": TARGET_COUNT}, "pre-recovery jobs were not all held")
    require(edited_counts == {"5": TARGET_COUNT}, "edited jobs left hold before release")
    require(released_counts == {"1": TARGET_COUNT}, "post-release snapshot was not all idle")

    recovery_head = receipt["repository_head"]
    relative_driver = "scripts/analysis/recover_hh4b_cut_baseline_validation_transport_in_place.py"
    committed_driver = subprocess.check_output(
        ["git", "show", f"{recovery_head}:{relative_driver}"], cwd=repo
    )
    require(
        sha256_bytes(committed_driver) == sha256(evidence / "recovery_driver.py"),
        "external recovery driver differs from committed bytes",
    )
    return {
        "gate_sha256": sha256(evidence / "pre_recovery_gate.json"),
        "receipt_sha256": sha256(evidence / "transport_recovery_receipt.json"),
        "driver_sha256": sha256(evidence / "recovery_driver.py"),
        "pre_recovery_status_counts": pre_counts,
        "post_edit_status_counts": edited_counts,
        "post_release_status_counts": released_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    require(repo == args.repo, "repository path is not canonical")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    remote = subprocess.check_output(
        ["git", "rev-parse", "origin/delphes-hh4b-production"], cwd=repo, text=True
    ).strip()
    require(head == remote, "local/remote HEAD mismatch")
    evidence = args.evidence_dir.resolve()
    audit = validate_evidence(repo, evidence)

    output = args.output_checkpoint if args.output_checkpoint.is_absolute() else repo / args.output_checkpoint
    output = output.resolve()
    require(output.is_relative_to(repo), "output escapes repository")
    require(output.parent.is_dir() and not output.exists(), "output path is invalid/exists")
    build = output.parent / f".{output.name}.build.{os.getpid()}"
    require(not build.exists(), "checkpoint build path exists")
    build.mkdir()
    (build / "evidence").mkdir()
    shutil.copytree(evidence, build / "evidence/transport_recovery")
    inventory = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(evidence.iterdir())
        if path.is_file()
    }
    (build / "external_evidence_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    freeze = {
        "schema_version": 1,
        "status": "pass_validation_cluster_in_place_transport_recovery_checkpoint_freeze",
        "repository_parent_head": head,
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "cluster_id": CLUSTER,
        "authoritative_schedd": SCHEDD,
        "target_count": TARGET_COUNT,
        **audit,
        "condor_submit_called": False,
        "do_not_resubmit_cluster": True,
        "transport_recovery_only": True,
        "scientific_arguments_changed": False,
        "nominal_selection_changed": False,
        "validation_payloads_opened_before_release": 0,
        "test_payloads_opened": 0,
        "next": "monitor_existing_cluster_read_only_then_audit_all_116_returns",
    }
    (build / "validation_transport_recovery_freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (build / "README.md").write_text(
        "# HH→4b validation in-place transport recovery\n\n"
        "The accepted 116-job cluster was repaired in place after a pre-execution "
        "client-local `/tmp` transfer failure. Only `Cmd` and `TransferInput` were "
        "changed to byte-identical frozen shared copies; no new submission occurred.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in build.rglob("*") if path.is_file())
    (build / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(build).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    os.replace(build, output)
    print("VALIDATION_TRANSPORT_RECOVERY_FREEZE=PASS")
    print(f"CLUSTER_ID={CLUSTER}")
    print("CONDOR_SUBMIT_CALLED=FALSE")
    print("DO_NOT_RESUBMIT=TRUE")
    print("VALIDATION_PAYLOADS_OPENED_BEFORE_RELEASE=0")
    print("TEST_PAYLOADS_OPENED=0")


if __name__ == "__main__":
    main()
