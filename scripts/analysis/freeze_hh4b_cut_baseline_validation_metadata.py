#!/usr/bin/env python3
"""Freeze metadata-only validation access and normalization inputs atomically."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pandas as pd


class ValidationMetadataFreezeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationMetadataFreezeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def require_committed_unchanged(repo: Path, path: Path) -> None:
    relative = path.relative_to(repo).as_posix()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(tracked.returncode == 0, f"required input is not committed: {relative}")
    require(
        subprocess.run(["git", "diff", "--quiet", "--", relative], cwd=repo).returncode == 0,
        f"required input has unstaged changes: {relative}",
    )
    require(
        subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", relative], cwd=repo
        ).returncode
        == 0,
        f"required input has staged changes: {relative}",
    )


def validate_metadata_freeze(
    remote: dict[str, Any], metadata: dict[str, Any], access: pd.DataFrame, coefficients: pd.DataFrame
) -> None:
    require(
        remote.get("status") == "pass_metadata_only_validation_remote_bundle_checksum_closure",
        "remote validation checksum closure status changed",
    )
    require(remote.get("remote_bundles") == 116, "remote validation bundle count changed")
    require(remote.get("independent_checksum_queries_per_bundle") == 2, "independent checksum query count changed")
    require(remote.get("all_independent_queries_match") is True, "independent remote checksums disagree")
    require(remote.get("new_metadata_checksum_closures") == 97, "new remote checksum closure count changed")
    require(remote.get("preexisting_frozen_checksums_verified") == 19, "preexisting checksum verification count changed")
    require(metadata.get("status") == "pass_metadata_only", "validation metadata status changed")
    require(metadata.get("physical_evaluation_sources") == 116, "physical validation source count changed")
    require(metadata.get("auxiliary_qcd_sources") == 5, "auxiliary QCD source count changed")
    require(metadata.get("validation_sources") == 121, "validation source count changed")
    require(metadata.get("validation_generated_events") == 1_002_584, "validation generated-event count changed")
    require(metadata.get("remote_bundle_checksum_closure") == "pass_116_of_116", "remote checksum closure changed")
    require(metadata.get("event_payload_files_opened") == 0, "event payload opened during metadata freeze")
    require(metadata.get("validation_access_authorized") is False, "metadata freeze authorized validation")
    require(metadata.get("validation_payloads_opened") == 0, "validation opened during metadata freeze")
    require(metadata.get("test_payloads_opened") == 0, "test opened during metadata freeze")
    require(len(access) == len(coefficients) == 121, "source/coefficient registry row count changed")
    require(access["source_uid"].is_unique and coefficients["source_uid"].is_unique, "source UID is not unique")
    require(access["source_uid"].tolist() == coefficients["source_uid"].tolist(), "source/coefficient registry order changed")
    physical = access["physical_evaluation_eligible"].map(truthy)
    auxiliary = access["auxiliary_qcd"].map(truthy)
    require(int(physical.sum()) == 116 and int(auxiliary.sum()) == 5, "validation source partition changed")
    require(physical.eq(~auxiliary).all(), "physical/auxiliary validation partition overlaps")
    require(not access["validation_content_opened"].map(truthy).any(), "validation content was opened")
    require(not access["test_content_opened"].map(truthy).any(), "test content was opened")
    require(
        coefficients.loc[physical, "run2_yield_coefficient_per_generator_weight"]
        .astype(float)
        .gt(0.0)
        .all(),
        "physical validation coefficient is invalid",
    )
    require(
        coefficients.loc[~physical, "run2_yield_coefficient_per_generator_weight"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"", "nan", "none", "null"})
        .all(),
        "auxiliary QCD received a physical coefficient",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines"))
    parser.add_argument("--remote-checksum-query-root", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, default=Path("docs/checkpoints/hh4b_full_5m_200k_expanded_manifest_split_20260728_v1/development_manifest.tsv"))
    parser.add_argument("--root-source-map", type=Path, default=Path("docs/checkpoints/hh4b_3b_root_source_map_20260725_v1/member_level_root_source_map.tsv"))
    parser.add_argument("--ordinary-coefficients", type=Path, default=Path("docs/checkpoints/hh4b_physical_normalization_run2_coefficient_analytical_closure_freeze_20260731_v2/ordinary_process_run2_coefficient_registry.tsv"))
    parser.add_argument("--hard-qcd-coefficients", type=Path, default=Path("docs/checkpoints/hh4b_physical_normalization_run2_coefficient_analytical_closure_freeze_20260731_v2/hard_qcd_shard_run2_coefficient_registry.tsv"))
    parser.add_argument("--ttbar-registry", type=Path, default=Path("docs/checkpoints/hh4b_physical_normalization_ttbar_independence_inclusion_freeze_20260730_v1/unique_ttbar_member_registry.tsv"))
    parser.add_argument("--legacy-root", type=Path, default=Path("/uscms_data/d3/iturkmen/hh4b_delphes"))
    parser.add_argument("--output-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_cut_baseline_validation_metadata_20260810_v1"))
    args = parser.parse_args()

    repo = args.repo.resolve()
    require(repo == args.repo, "repository path is not canonical")
    require(
        subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
        == "delphes-hh4b-production",
        "branch changed",
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    require(
        head
        == subprocess.check_output(
            ["git", "rev-parse", "origin/delphes-hh4b-production"], cwd=repo, text=True
        ).strip(),
        "local/remote HEAD mismatch",
    )
    remote_root = args.remote_checksum_query_root.resolve()
    require(remote_root.is_dir() and not remote_root.is_symlink(), "remote checksum query root is invalid")
    remote_sums = remote_root / "SHA256SUMS"
    check = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"], cwd=remote_root,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    require(check.returncode == 0, f"remote checksum evidence failed: {check.stdout}{check.stderr}")
    remote_registry = remote_root / "validation_116_remote_bundle_checksum_registry.tsv"
    remote_summary_path = remote_root / "validation_remote_checksum_summary.json"
    remote_summary = json.loads(remote_summary_path.read_text(encoding="utf-8"))

    def resolve_input(value: Path) -> Path:
        path = value if value.is_absolute() else repo / value
        path = path.resolve()
        require(path.is_file() and not path.is_symlink(), f"frozen metadata input is invalid: {path}")
        require(path.is_relative_to(repo), f"frozen metadata input is outside repository: {path}")
        return path

    inputs = {
        "development_manifest": resolve_input(args.development_manifest),
        "root_source_map": resolve_input(args.root_source_map),
        "ordinary_coefficients": resolve_input(args.ordinary_coefficients),
        "hard_qcd_coefficients": resolve_input(args.hard_qcd_coefficients),
        "ttbar_registry": resolve_input(args.ttbar_registry),
    }
    preparer = repo / "scripts/analysis/prepare_hh4b_cut_baseline_validation_metadata.py"
    freezer = Path(__file__).resolve()
    for path in [*inputs.values(), preparer, freezer]:
        require_committed_unchanged(repo, path)
    output = args.output_checkpoint if args.output_checkpoint.is_absolute() else repo / args.output_checkpoint
    output = output.resolve()
    require(output.is_relative_to(repo), "metadata checkpoint is outside repository")
    require(output.parent.is_dir() and not output.exists(), "metadata checkpoint output exists/invalid")
    build = output.parent / f".{output.name}.build.{os.getpid()}"
    require(not build.exists(), "metadata checkpoint build path exists")
    evidence = build / "evidence/remote_checksum_queries"
    evidence.mkdir(parents=True)
    for path in sorted(remote_root.iterdir()):
        if path.is_file():
            shutil.copy2(path, evidence / path.name)
    stable_registry_label = (
        output.relative_to(repo) / "evidence/remote_checksum_queries" / remote_registry.name
    ).as_posix()
    command = [
        "python3", str(preparer),
        "--development-manifest", str(inputs["development_manifest"].relative_to(repo)),
        "--root-source-map", str(inputs["root_source_map"].relative_to(repo)),
        "--ordinary-coefficients", str(inputs["ordinary_coefficients"].relative_to(repo)),
        "--hard-qcd-coefficients", str(inputs["hard_qcd_coefficients"].relative_to(repo)),
        "--ttbar-registry", str(inputs["ttbar_registry"].relative_to(repo)),
        "--remote-checksum-registry", str(evidence / remote_registry.name),
        "--remote-checksum-registry-label", stable_registry_label,
        "--legacy-root", str(args.legacy_root.resolve()),
        "--output-dir", str(build),
    ]
    result = subprocess.run(command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    require(result.returncode == 0, f"metadata preparation failed: {result.stdout}{result.stderr}")
    metadata_path = build / "validation_metadata_preparation_summary.json"
    access_path = build / "validation_121_source_access_manifest.tsv"
    coefficients_path = build / "validation_121_physical_coefficient_registry.tsv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    access = pd.read_csv(access_path, sep="\t", keep_default_na=False)
    coefficients = pd.read_csv(coefficients_path, sep="\t", keep_default_na=False)
    validate_metadata_freeze(remote_summary, metadata, access, coefficients)
    require(metadata["input_sha256"].get(stable_registry_label) == sha256(evidence / remote_registry.name), "stable remote registry provenance missing")
    require(not any(".build." in path for path in metadata["input_sha256"]), "temporary build path leaked into metadata provenance")
    freeze = {
        "schema_version": 1,
        "status": "pass_cut_baseline_validation_metadata_checkpoint_frozen",
        "repository_parent_head": head,
        "remote_checksum_evidence_sha256s_sha256": sha256(evidence / "SHA256SUMS"),
        "remote_checksum_registry_sha256": sha256(evidence / remote_registry.name),
        "source_access_manifest_sha256": sha256(access_path),
        "physical_coefficient_registry_sha256": sha256(coefficients_path),
        "physical_evaluation_sources": 116,
        "auxiliary_qcd_sources": 5,
        "event_payload_files_opened": 0,
        "validation_access_authorized": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
        "next": "commit_and_push_then_bind_this_metadata_in_the_master_train_only_freeze",
    }
    (build / "validation_metadata_checkpoint_freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (build / "README.md").write_text(
        "# HH→4b cut-baseline validation metadata checkpoint\n\n"
        "This is a metadata-only freeze of 121 source identities, 116 independently "
        "checksum-closed physical sources, five sealed auxiliary-QCD sources, and physical "
        "normalization coefficients. No event payload was opened and validation remains unauthorized.\n",
        encoding="utf-8",
    )
    checksum_paths = sorted(
        path for path in build.rglob("*")
        if path.is_file() and path != build / "SHA256SUMS"
    )
    (build / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(build).as_posix()}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    os.replace(build, output)
    print("HH4B_CUT_BASELINE_VALIDATION_METADATA_FREEZE=PASS")
    print("PHYSICAL_EVALUATION_SOURCES=116")
    print("AUXILIARY_QCD_SOURCES=5")
    print("EVENT_PAYLOAD_FILES_OPENED=0")
    print("VALIDATION_ACCESS_AUTHORIZED=FALSE")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")


if __name__ == "__main__":
    main()
