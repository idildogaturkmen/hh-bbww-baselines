#!/usr/bin/env python3
"""Freeze the complete cluster-30020809 return audit before aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


SHARD_NAMES = [
    f"structure_result_inventory_replicas_{low:04d}_{low + 99:04d}.tsv"
    for low in range(200, 1000, 100)
]
EXPECTED_FILES = {
    "complete_return_audit.json",
    "complete_return_audit.txt",
    "final_history_snapshot.txt",
    "final_history_snapshot.txt.stderr",
    "final_queue_snapshot.txt",
    "final_queue_snapshot.txt.stderr",
    "job_return_audit.tsv",
    "structure_result_inventory_manifest.json",
    *SHARD_NAMES,
}
CHECKPOINT_COPY_FILES = {
    "complete_return_audit.json",
    "complete_return_audit.txt",
    "structure_result_inventory_manifest.json",
}


class FreezeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def validate_audit_summary(audit: dict[str, Any], manifest: dict[str, Any]) -> None:
    require(
        audit.get("status")
        == "pass_complete_escalation800_8000_job_216000_structure_return_audit",
        "complete-return audit status changed",
    )
    require(audit.get("cluster_id") == 30020809, "cluster changed")
    require(audit.get("authoritative_schedd") == "lpcschedd5.fnal.gov", "schedd changed")
    require(audit.get("jobs_audited") == 8000, "audited job count changed")
    require(audit.get("structure_results_audited") == 216000, "structure-result count changed")
    for key in (
        "all_job_receipts_clean",
        "all_runner_logs_match_receipt_sha256",
        "all_structure_internal_sha256s_verified",
        "all_structure_canonical_payload_sha256s_recomputed",
        "all_inner_crossfit_four_fold_coverage_verified",
        "all_execution_provenance_sha256s_match_frozen_payload_manifest",
        "all_outer_fold_used_for_selection_false",
        "all_validation_payloads_opened_zero",
        "all_test_payloads_opened_zero",
        "exact_structure_set_per_category_fold_verified",
        "never_resubmit_cluster",
    ):
        require(audit.get(key) is True, f"complete-return true gate failed: {key}")
    for key in (
        "aggregation_performed",
        "fold_level_winners_selected",
        "replica_level_stability_statistics_computed",
        "pilot_results_enter_stability_aggregation",
        "nominal_selection_changed",
    ):
        require(audit.get(key) is False, f"complete-return false gate failed: {key}")
    require(audit.get("validation_payloads_opened") == 0, "validation was opened")
    require(audit.get("test_payloads_opened") == 0, "test was opened")
    require(
        manifest.get("status")
        == "pass_escalation800_structure_result_inventory_sharded",
        "structure inventory manifest status changed",
    )
    require(manifest.get("total_rows_excluding_headers") == 216000, "inventory row total changed")
    require(manifest.get("shard_count") == 8, "inventory shard count changed")
    require(manifest.get("validation_payloads_opened") == 0, "inventory reports validation access")
    require(manifest.get("test_payloads_opened") == 0, "inventory reports test access")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines"))
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path(
            "docs/checkpoints/"
            "hh4b_train_multivariate_cut_selection_stability_escalation800_"
            "complete_return_audit_20260810_v1"
        ),
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

    source = args.audit_output.resolve()
    require(source.is_dir() and not source.is_symlink(), "audit output is invalid")
    actual_files = {path.name for path in source.iterdir() if path.is_file()}
    require(actual_files == EXPECTED_FILES, "audit output file set changed")
    require(not any(path.is_symlink() for path in source.iterdir()), "audit output contains symlink")
    audit = json.loads((source / "complete_return_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (source / "structure_result_inventory_manifest.json").read_text(encoding="utf-8")
    )
    validate_audit_summary(audit, manifest)
    require((source / "final_queue_snapshot.txt").stat().st_size == 0, "final queue is not empty")
    require((source / "final_queue_snapshot.txt.stderr").stat().st_size == 0, "queue stderr is not empty")
    require((source / "final_history_snapshot.txt.stderr").stat().st_size == 0, "history stderr is not empty")
    require(line_count(source / "final_history_snapshot.txt") == 8000, "history line count changed")
    require(line_count(source / "job_return_audit.tsv") == 8001, "job audit line count changed")
    shard_by_name = {row["path"]: row for row in manifest["shards"]}
    require(set(shard_by_name) == set(SHARD_NAMES), "inventory shard names changed")
    for name in SHARD_NAMES:
        path = source / name
        row = shard_by_name[name]
        require(line_count(path) == 27001, f"inventory shard row count changed: {name}")
        require(path.stat().st_size == row["bytes"], f"inventory shard size changed: {name}")
        require(sha256(path) == row["sha256"], f"inventory shard SHA changed: {name}")

    output = args.output_checkpoint
    if not output.is_absolute():
        output = repo / output
    output = output.resolve()
    require(output.is_relative_to(repo), "checkpoint is outside repository")
    require(output.parent.is_dir() and not output.exists(), "checkpoint output path is invalid/exists")
    build = output.parent / f".{output.name}.build.{os.getpid()}"
    require(not build.exists(), "checkpoint build path exists")
    build.mkdir()
    for name in sorted(CHECKPOINT_COPY_FILES):
        shutil.copy2(source / name, build / name)
    freeze = {
        "schema_version": 1,
        "status": "pass_escalation800_complete_return_audit_frozen_before_all1000_aggregation",
        "repository_parent_head": head,
        "cluster_id": 30020809,
        "authoritative_schedd": "lpcschedd5.fnal.gov",
        "queue_rows": 0,
        "history_rows": 8000,
        "clean_history_rows": 8000,
        "bad_history_rows": 0,
        "receipts": 8000,
        "bundles": 8000,
        "runner_logs": 8000,
        "complete_triplets": 8000,
        "partial_triplets": 0,
        "jobs_audited": 8000,
        "structure_results_audited": 216000,
        "structure_inventory_shards": 8,
        "structure_rows_per_shard": 27000,
        "scientific_outcome_counts_are_not_infrastructure_failure_gates": True,
        "aggregation_performed": False,
        "pilot_results_enter_stability_aggregation": False,
        "nominal_selection_changed": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
        "do_not_resubmit_cluster": True,
        "checkpoint_copy_files": sorted(CHECKPOINT_COPY_FILES),
        "large_source_artifacts_bound_by_sha256_not_copied": True,
        "source_artifacts": {
            name: {"bytes": (source / name).stat().st_size, "sha256": sha256(source / name)}
            for name in sorted(EXPECTED_FILES)
        },
        "next": "commit_and_push_then_run_predeclared_all1000_aggregation",
    }
    (build / "complete_return_audit_freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (build / "README.md").write_text(
        "# HH→4b escalation-800 complete-return audit\n\n"
        "This checkpoint freezes the clean 8,000-job / 216,000-structure audit "
        "for cluster `30020809` before all-1000 aggregation. Scientific "
        "feasibility is diagnostic, the nominal cut is unchanged, validation "
        "and test remain sealed, and the cluster must never be resubmitted. "
        "The large job/history/inventory tables are not duplicated here; all "
        "16 validated source artifacts are bound by byte count and SHA256 in "
        "`complete_return_audit_freeze.json`.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in build.iterdir() if path.is_file())
    (build / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    os.replace(build, output)
    print("ESCALATION800_COMPLETE_RETURN_AUDIT_FREEZE=PASS")
    print("JOBS_AUDITED=8000")
    print("STRUCTURE_RESULTS_AUDITED=216000")
    print("AGGREGATION_PERFORMED=FALSE")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")
    print("DO_NOT_RESUBMIT_CLUSTER_30020809=TRUE")


if __name__ == "__main__":
    main()
