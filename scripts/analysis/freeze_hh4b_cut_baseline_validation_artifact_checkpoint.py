#!/usr/bin/env python3
"""Freeze each post-access validation artifact as a separate Git checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


ROLE_CONTRACTS = {
    "validation_returns": {
        "evidence_name": "validation_returns",
        "summary": "validation_return_audit.json",
        "status": "pass_complete_one_time_cut_baseline_validation_return_audit",
        "next": "commit_and_push_then_aggregate_the_one_frozen_validation_cycle",
    },
    "validation_performance": {
        "evidence_name": "validation_performance",
        "summary": "validation_performance_summary.json",
        "status": "pass_one_time_hh4b_cut_baseline_validation_aggregation",
        "next": "commit_and_push_then_render_figure17_from_frozen_validation_results",
    },
    "publication_validation": {
        "evidence_name": "publication_validation",
        "summary": "manifests/figure17_validation_vs_train_oof_provenance.json",
        "status": "pass_cut_baseline_validation_publication_figure",
        "next": "commit_and_push_then_build_the_final_cut_baseline_status_report",
    },
}


class ValidationArtifactFreezeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationArtifactFreezeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_role_summary(role: str, summary: dict[str, Any]) -> None:
    contract = ROLE_CONTRACTS[role]
    require(summary.get("status") == contract["status"], f"{role} status changed")
    require(summary.get("validation_payloads_opened") == 1, f"validation cycle count changed in {role}")
    require(summary.get("test_payloads_opened") == 0, f"test opened in {role}")
    if role == "validation_returns":
        for field in (
            "clean_history_jobs",
            "source_attempt_markers",
            "source_summaries",
            "source_distributions",
            "job_receipts",
            "physical_validation_sources_opened_once",
        ):
            require(summary.get(field) == 116, f"validation return closure changed: {field}")
        require(summary.get("distribution_rows") == 116 * 1360, "validation distribution row count changed")
        require(summary.get("auxiliary_qcd_validation_sources_opened") == 0, "auxiliary QCD validation was opened")
        require(summary.get("auxiliary_qcd_validation_sources_remain_sealed") == 5, "auxiliary QCD seal count changed")
        require(summary.get("validation_evaluation_cycles") == 1, "validation evaluation cycle count changed")
        require(summary.get("source_payload_reruns") == 0, "validation source was rerun")
        require(summary.get("do_not_resubmit_cluster") is True, "validation cluster is not sealed against resubmission")
    elif role == "validation_performance":
        require(summary.get("validation_sources_opened") == 116, "validation source count changed")
        require(summary.get("validation_source_payloads_each_opened_exactly_once") is True, "validation source open closure changed")
        require(summary.get("validation_payload_reruns") == 0, "validation source was rerun")
        require(summary.get("auxiliary_qcd_validation_sources_excluded_and_unopened") == 5, "auxiliary QCD validation seal changed")
        require(summary.get("nominal_selection_changed") is False, "validation changed the nominal cut")
        require(summary.get("official_cms_result") is False, "validation result claims official CMS status")
        require(isinstance(summary.get("pooled_metrics"), dict), "validation pooled metrics are missing")
    else:
        outputs = summary.get("outputs")
        require(isinstance(outputs, dict) and len(outputs) == 3, "Figure 17 output closure changed")
        suffixes = sorted(Path(path).suffix for path in outputs)
        require(suffixes == [".pdf", ".png", ".tsv"], "Figure 17 PDF/PNG/TSV closure changed")
        require(summary.get("official_cms_result") is False, "Figure 17 claims official CMS status")
    for field in ("cut_scan_performed", "threshold_adjustment_performed", "family_adjustment_performed"):
        if field in summary:
            require(summary.get(field) is False, f"forbidden validation action reported: {field}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines"))
    parser.add_argument("--role", choices=sorted(ROLE_CONTRACTS), required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
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

    source = args.source_dir if args.source_dir.is_absolute() else repo / args.source_dir
    source = source.resolve()
    require(source.is_relative_to(repo), "validation artifact source is outside repository")
    require(source.is_dir() and not source.is_symlink(), "validation artifact source is invalid")
    require(not any(path.is_symlink() for path in source.rglob("*")), "validation artifact contains symlink")
    sums = source / "SHA256SUMS"
    require(sums.is_file() and not sums.is_symlink(), "validation artifact lacks SHA256SUMS")
    check = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=source,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(check.returncode == 0, f"validation artifact checksum failure: {check.stdout}{check.stderr}")
    actual_files = {
        path.relative_to(repo).as_posix() for path in source.rglob("*") if path.is_file()
    }
    tracked_files = set(
        subprocess.check_output(
            ["git", "ls-files", "--", str(source.relative_to(repo))], cwd=repo, text=True
        ).splitlines()
    )
    require(actual_files <= tracked_files, "validation artifact source contains uncommitted/ignored files")
    relative_source = source.relative_to(repo).as_posix()
    require(
        subprocess.run(["git", "diff", "--quiet", "--", relative_source], cwd=repo).returncode
        == 0,
        "validation artifact source has unstaged changes",
    )
    require(
        subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", relative_source], cwd=repo
        ).returncode
        == 0,
        "validation artifact source has staged changes",
    )

    contract = ROLE_CONTRACTS[args.role]
    summary_path = source / contract["summary"]
    require(summary_path.is_file() and not summary_path.is_symlink(), "validation artifact summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_role_summary(args.role, summary)

    output = args.output_checkpoint if args.output_checkpoint.is_absolute() else repo / args.output_checkpoint
    output = output.resolve()
    require(output.is_relative_to(repo), "checkpoint output is outside repository")
    require(output.parent.is_dir() and not output.exists(), "checkpoint output path is invalid/exists")
    build = output.parent / f".{output.name}.build.{os.getpid()}"
    require(not build.exists(), "checkpoint build path exists")
    build.mkdir()
    evidence = build / "evidence" / contract["evidence_name"]
    evidence.parent.mkdir()
    shutil.copytree(source, evidence)
    freeze = {
        "schema_version": 1,
        "status": f"pass_{args.role}_artifact_checkpoint_frozen",
        "role": args.role,
        "repository_parent_head": head,
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "source_path": str(source.relative_to(repo)),
        "source_sha256s_sha256": sha256(sums),
        "summary_path": str(summary_path.relative_to(repo)),
        "summary_sha256": sha256(summary_path),
        "validation_evaluation_cycles": 1,
        "validation_sources_opened_once": 116,
        "validation_payload_reruns": 0,
        "nominal_selection_changed": False,
        "test_payloads_opened": 0,
        "next": contract["next"],
    }
    (build / "artifact_checkpoint_freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (build / "README.md").write_text(
        f"# HH→4b {args.role.replace('_', ' ')} checkpoint\n\n"
        "This checkpoint records the single authorized fixed-cut validation cycle. "
        "No retuning or rerun occurred, auxiliary QCD validation sources remained sealed, "
        "and test remained unopened.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in build.rglob("*") if path.is_file())
    (build / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(build).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    os.replace(build, output)
    print("HH4B_CUT_BASELINE_VALIDATION_ARTIFACT_FREEZE=PASS")
    print(f"ROLE={args.role}")
    print("VALIDATION_EVALUATION_CYCLES=1")
    print("VALIDATION_SOURCE_RERUNS=0")
    print("TEST_PAYLOADS_OPENED=0")


if __name__ == "__main__":
    main()
