#!/usr/bin/env python3
"""Freeze all-1000, train-performance, or train-figure artifacts independently.

The all-1000 checkpoint is intentionally compact.  It copies every downstream
consumer table but binds the ten large, already committed ranking ledgers by
path, byte count, and SHA-256 instead of duplicating them.
"""

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
    "all1000": {
        "evidence_name": "aggregation",
        "summary": "all1000_stability_summary.json",
        "status": "pass_all1000_predeclared_selection_stability_aggregation",
    },
    "train_performance": {
        "evidence_name": "train_performance",
        "summary": "train_performance_summary.json",
        "status": "pass_hh4b_cut_baseline_train_performance_and_fixed_comparator",
    },
    "publication_train_only": {
        "evidence_name": "publication_train_only",
        "summary": "publication_figure_summary.json",
        "status": "pass_hh4b_cut_baseline_train_only_publication_figures",
    },
}
ALL1000_AUDIT_STATUS = "pass_independent_all1000_row_level_audit_and_byte_identical_rerun"
ALL1000_RANKED_SHARD_PREFIX = "tables/ranked_structure_results_replicas_"


class ArtifactFreezeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactFreezeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256s(root: Path) -> dict[str, dict[str, int | str]]:
    sums = root / "SHA256SUMS"
    require(sums.is_file() and not sums.is_symlink(), f"artifact lacks SHA256SUMS: {root}")
    bindings: dict[str, dict[str, int | str]] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and fields[1].startswith("./"), "malformed SHA256SUMS line")
        digest, relative = fields[0], fields[1][2:]
        require(
            len(digest) == 64 and all(character in "0123456789abcdef" for character in digest),
            f"malformed SHA-256 for {relative}",
        )
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"invalid checksummed artifact: {relative}")
        require(sha256(path) == digest, f"artifact checksum mismatch: {relative}")
        bindings[relative] = {"bytes": path.stat().st_size, "sha256": digest}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    require(actual == set(bindings) | {"SHA256SUMS"}, "artifact checksum file set mismatch")
    return dict(sorted(bindings.items()))


def should_copy_all1000_file(relative: str) -> bool:
    return relative != "SHA256SUMS" and not relative.startswith(ALL1000_RANKED_SHARD_PREFIX)


def validate_all1000_audit(audit: dict[str, Any], source_sums_sha256: str) -> None:
    require(audit.get("status") == ALL1000_AUDIT_STATUS, "all1000 independent audit status changed")
    require(audit.get("ranked_structure_results_audited") == 270000, "audit ranked count changed")
    require(audit.get("fold_level_winners_independently_selected") == 10000,
            "audit winner count changed")
    require(audit.get("replica_category_reductions_independently_recomputed") == 2000,
            "audit reduction count changed")
    require(audit.get("initial200_ranked_rows_proven_identical") == 54000,
            "audit initial200 ranking proof changed")
    require(audit.get("initial200_replica_category_reductions_proven_identical") == 400,
            "audit initial200 reduction proof changed")
    rerun = audit.get("deterministic_rerun", {})
    require(
        rerun.get("status") == "pass_every_output_file_byte_identical"
        and rerun.get("file_count") == 29,
        "audit deterministic rerun proof changed",
    )
    require(audit.get("source_sha256s_sha256") == source_sums_sha256,
            "audit/source SHA256SUMS binding changed")
    require(audit.get("pilot_results_used") is False, "audit used pilot results")
    require(audit.get("nominal_deployment_candidate_changed") is False,
            "audit changed nominal candidate")
    require(audit.get("validation_payloads_opened") == 0, "audit opened validation")
    require(audit.get("test_payloads_opened") == 0, "audit opened test")


def require_committed_unchanged(repo: Path, root: Path) -> None:
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    require(bool(paths), f"empty artifact root: {root}")
    for path in paths:
        relative = path.relative_to(repo).as_posix()
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        require(tracked.returncode == 0, f"artifact is not committed: {relative}")
    require(subprocess.run(["git", "diff", "--quiet", "--", root.relative_to(repo)], cwd=repo).returncode == 0,
            f"artifact has unstaged changes: {root}")
    require(subprocess.run(["git", "diff", "--cached", "--quiet", "--", root.relative_to(repo)],
                           cwd=repo).returncode == 0,
            f"artifact has staged changes: {root}")


def copy_compact_all1000(source: Path, evidence: Path) -> int:
    copied = 0
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source).as_posix()
        if not should_copy_all1000_file(relative):
            continue
        destination = evidence / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied += 1
    shutil.copy2(source / "SHA256SUMS", evidence / "FULL_SOURCE_SHA256SUMS")
    return copied + 1


def validate_role_summary(role: str, summary: dict[str, Any]) -> None:
    contract = ROLE_CONTRACTS[role]
    require(summary.get("status") == contract["status"], f"{role} status changed")
    require(summary.get("validation_payloads_opened") == 0, f"validation opened in {role}")
    require(summary.get("test_payloads_opened") == 0, f"test opened in {role}")
    if role == "all1000":
        require(summary.get("fold_level_winner_count") == 10000, "all1000 winner count changed")
        require(summary.get("replica_category_result_count") == 2000, "all1000 reduction count changed")
        require(summary.get("ranked_structure_result_count") == 270000, "all1000 result count changed")
        require(summary.get("nominal_deployment_candidate_changed") is False, "all1000 changed nominal candidate")
    elif role == "train_performance":
        require(summary.get("primary_generalization_estimate") == "pooled_nested_outer_oof", "primary estimate changed")
        require(summary.get("fixed_deployment_cut_called_primary_generalization_estimate") is False, "fixed train diagnostic mislabeled primary")
        require(summary.get("historical_comparator_threshold_scan_performed") is False, "historical comparator was tuned")
        require(summary.get("historical_comparator_threshold") == {"r_hh_125_125_lt": 34.0}, "historical comparator changed")
        require(summary.get("official_cms_result") is False, "train result claims official CMS status")
    else:
        require(summary.get("figures") == summary.get("pdfs") == summary.get("pngs") == 16, "figure count changed")
        require(summary.get("figure_data_sidecars") == 16, "figure sidecar count changed")
        require(summary.get("official_cms_status_claimed") is False, "figures claim official CMS status")


def checkpoint_readme(role: str) -> str:
    lines = [
        f"# HH→4b {role.replace('_', ' ')} checkpoint",
        "",
        "This checkpoint was frozen before validation. The nominal cut is unchanged, "
        "validation and test payload counts are zero, and no official CMS status is claimed.",
    ]
    if role == "all1000":
        lines.extend([
            "",
            "For the all-1000 role, the ten large ranked-result shards remain in the committed "
            "source artifact and are bound here by exact path, byte count, and SHA-256. Every "
            "downstream-consumed table and the independent byte-identical-rerun audit are copied "
            "into this compact checkpoint.",
        ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines"))
    parser.add_argument("--role", choices=sorted(ROLE_CONTRACTS), required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--independent-audit-dir",
        type=Path,
        default=Path("artifacts/hh4b_cut_baseline/all1000_stability_independent_audit"),
        help="Required committed independent audit for the all1000 role.",
    )
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
    require(source.is_relative_to(repo), "artifact source is outside repository")
    require(source.is_dir() and not source.is_symlink(), "artifact source is invalid")
    require_committed_unchanged(repo, source)
    sums = source / "SHA256SUMS"
    source_bindings = verify_sha256s(source)
    contract = ROLE_CONTRACTS[args.role]
    summary_path = source / contract["summary"]
    require(summary_path.is_file(), f"artifact summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_role_summary(args.role, summary)
    require(not any(path.is_symlink() for path in source.rglob("*")), "artifact source contains symlink")

    audit_root: Path | None = None
    audit_path: Path | None = None
    audit_sums: Path | None = None
    audit: dict[str, Any] | None = None
    if args.role == "all1000":
        audit_root = (
            args.independent_audit_dir
            if args.independent_audit_dir.is_absolute()
            else repo / args.independent_audit_dir
        ).resolve()
        require(audit_root.is_relative_to(repo) and audit_root.is_dir() and not audit_root.is_symlink(),
                "all1000 independent audit root is invalid")
        require_committed_unchanged(repo, audit_root)
        verify_sha256s(audit_root)
        audit_path = audit_root / "all1000_stability_independent_audit.json"
        audit_sums = audit_root / "SHA256SUMS"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        validate_all1000_audit(audit, sha256(sums))

    output = args.output_checkpoint if args.output_checkpoint.is_absolute() else repo / args.output_checkpoint
    output = output.resolve()
    require(output.is_relative_to(repo), "checkpoint output is outside repository")
    require(output.parent.is_dir() and not output.exists(), "checkpoint output path is invalid/exists")
    build = output.parent / f".{output.name}.build.{os.getpid()}"
    require(not build.exists(), "checkpoint build path exists")
    build.mkdir()
    evidence = build / "evidence" / contract["evidence_name"]
    evidence.parent.mkdir()
    if args.role == "all1000":
        copied_evidence_files = copy_compact_all1000(source, evidence)
        assert audit_root is not None and audit is not None and audit_path is not None and audit_sums is not None
        shutil.copytree(audit_root, build / "evidence" / "independent_audit")
        implementation = build / "implementation"
        implementation.mkdir()
        for relative in (
            "scripts/analysis/aggregate_hh4b_all1000_selection_stability.py",
            "scripts/analysis/audit_hh4b_all1000_selection_stability.py",
            "scripts/analysis/freeze_hh4b_cut_baseline_train_artifact_checkpoint.py",
            "tests/test_aggregate_hh4b_all1000_selection_stability.py",
            "tests/test_audit_hh4b_all1000_selection_stability.py",
            "tests/test_freeze_hh4b_cut_baseline_train_artifact_checkpoint.py",
        ):
            shutil.copy2(repo / relative, implementation / Path(relative).name)
    else:
        shutil.copytree(source, evidence)
        copied_evidence_files = len([path for path in evidence.rglob("*") if path.is_file()])
    ranked_bindings = {
        relative: binding
        for relative, binding in source_bindings.items()
        if relative.startswith(ALL1000_RANKED_SHARD_PREFIX)
    }
    freeze = {
        "schema_version": 1,
        "status": f"pass_{args.role}_artifact_checkpoint_frozen",
        "role": args.role,
        "repository_parent_head": head,
        "source_path": str(source.relative_to(repo)),
        "source_sha256s_sha256": sha256(sums),
        "source_file_count_excluding_sha256s": len(source_bindings),
        "source_total_bytes_excluding_sha256s": sum(int(item["bytes"]) for item in source_bindings.values()),
        "source_artifacts": source_bindings,
        "source_artifacts_committed_and_unchanged": True,
        "summary_path": str(summary_path.relative_to(repo)),
        "summary_sha256": sha256(summary_path),
        "checkpoint_evidence_file_count": copied_evidence_files,
        "large_ranked_result_shards_bound_by_sha256_not_copied": args.role == "all1000",
        "large_ranked_result_shard_count": len(ranked_bindings),
        "large_ranked_result_shards": ranked_bindings,
        "independent_audit": None if audit is None else {
            "path": str(audit_path.relative_to(repo)),
            "sha256": sha256(audit_path),
            "sha256s_sha256": sha256(audit_sums),
            "status": audit["status"],
            "ranked_structure_results_audited": audit["ranked_structure_results_audited"],
            "fold_level_winners_independently_selected": audit[
                "fold_level_winners_independently_selected"
            ],
            "replica_category_reductions_independently_recomputed": audit[
                "replica_category_reductions_independently_recomputed"
            ],
            "deterministic_rerun": audit["deterministic_rerun"],
        },
        "nominal_selection_changed": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
        "next": {
            "all1000": "commit_and_push_then_build_train_performance_and_fixed_comparator",
            "train_performance": "commit_and_push_then_render_train_only_publication_figures",
            "publication_train_only": "commit_and_push_then_freeze_master_train_only_checkpoint",
        }[args.role],
    }
    (build / "artifact_checkpoint_freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (build / "README.md").write_text(checkpoint_readme(args.role), encoding="utf-8")
    files = sorted(path for path in build.rglob("*") if path.is_file())
    (build / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(build).as_posix()}\n" for path in files
        ),
        encoding="utf-8",
    )
    os.replace(build, output)
    print("HH4B_CUT_BASELINE_TRAIN_ARTIFACT_FREEZE=PASS")
    print(f"ROLE={args.role}")
    print("NOMINAL_SELECTION_CHANGED=FALSE")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")


if __name__ == "__main__":
    main()
