#!/usr/bin/env python3
"""Create the explicit one-time cut-baseline validation authorization.

This gate is metadata-only and may run only after the master train-only
checkpoint is committed, pushed, and is an ancestor of the remote-matched
repository HEAD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd


BRANCH = "delphes-hh4b-production"
NOMINAL_THRESHOLDS = {
    "exact3tag": {
        "r_hh_125_125": 36.40814019639858,
        "ht_candidate_jets": 176.5458068847656,
    },
    "ge4tag": {
        "r_hh_125_125": 33.92808917804956,
        "mhh": 164.73708096689654,
        "abs_h_delta_eta": 6.904302164473993,
    },
}


class AuthorizationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorizationError(message)


def clean(value: Any) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def truthy(value: Any) -> bool:
    return clean(value).lower() in {"true", "1", "yes", "y"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def verify_repo(repo: Path, master_commit: str) -> str:
    require(repo.resolve() == repo, "repository path is not canonical")
    require(git(repo, "branch", "--show-current") == BRANCH, "branch changed")
    head = git(repo, "rev-parse", "HEAD")
    remote = git(repo, "rev-parse", f"origin/{BRANCH}")
    require(head == remote, "local and remote HEAD differ")
    require(git(repo, "rev-parse", master_commit) == master_commit, "master commit is not exact")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", master_commit, head], cwd=repo
    )
    require(ancestor.returncode == 0, "master checkpoint commit is not an ancestor")
    return head


def verify_checkpoint(checkpoint: Path) -> None:
    require(checkpoint.is_dir() and not checkpoint.is_symlink(), f"bad checkpoint: {checkpoint}")
    sums = checkpoint / "SHA256SUMS"
    require(sums.is_file() and not sums.is_symlink(), f"missing SHA256SUMS: {checkpoint}")
    result = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=checkpoint,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(result.returncode == 0, f"checkpoint checksum failure: {result.stdout}{result.stderr}")


def validate_master_summary(summary: dict[str, Any]) -> None:
    require(
        summary.get("status") == "pass_master_train_only_hh4b_cut_baseline_freeze",
        "master train-only status changed",
    )
    require(summary.get("validation_payloads_opened") == 0, "validation was opened before master freeze")
    require(summary.get("test_payloads_opened") == 0, "test was opened before master freeze")
    require(summary.get("nominal_selection_changed") is False, "nominal selection changed")
    require(summary.get("nominal_thresholds") == NOMINAL_THRESHOLDS, "master nominal thresholds changed")
    require(summary.get("cut_family_frozen") is True, "cut family is not frozen")
    require(summary.get("variables_frozen") is True, "cut variables are not frozen")
    require(summary.get("thresholds_frozen") is True, "cut thresholds are not frozen")
    require(summary.get("reporting_choices_frozen") is True, "reporting choices are not frozen")


def verify_master_code_manifest(repo: Path, checkpoint: Path) -> Path:
    manifest_path = checkpoint / "code_manifest.tsv"
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "master code manifest is missing")
    manifest = pd.read_csv(manifest_path, sep="\t", keep_default_na=False)
    require(
        list(manifest.columns) == ["relative_path", "bytes", "sha256"],
        "master code manifest schema changed",
    )
    require(len(manifest) > 0 and manifest["relative_path"].is_unique, "master code manifest is empty/nonunique")
    for row in manifest.itertuples(index=False):
        path = (repo / str(row.relative_path)).resolve()
        require(path.is_relative_to(repo), f"master code path escapes repository: {row.relative_path}")
        require(path.is_file() and not path.is_symlink(), f"master-frozen code is missing: {path}")
        require(path.stat().st_size == int(row.bytes), f"master-frozen code size changed: {path}")
        require(sha256(path) == str(row.sha256), f"master-frozen code SHA changed: {path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--master-checkpoint", type=Path, required=True)
    parser.add_argument("--master-summary", type=Path, required=True)
    parser.add_argument("--master-checkpoint-commit", required=True)
    parser.add_argument("--validation-metadata-checkpoint", type=Path, required=True)
    parser.add_argument("--source-access-manifest", type=Path, required=True)
    parser.add_argument("--physical-coefficient-registry", type=Path, required=True)
    parser.add_argument("--broad-feature-extractor", type=Path, required=True)
    parser.add_argument("--candidate-reconstruction-module", type=Path, required=True)
    parser.add_argument("--validation-source-worker", type=Path, required=True)
    parser.add_argument("--validation-aggregator", type=Path, required=True)
    parser.add_argument("--validation-plotter", type=Path, required=True)
    parser.add_argument("--validation-campaign-preparer", type=Path, required=True)
    parser.add_argument("--validation-campaign-auditor", type=Path, required=True)
    parser.add_argument("--validation-campaign-presubmission-freezer", type=Path, required=True)
    parser.add_argument("--validation-campaign-submitter", type=Path, required=True)
    parser.add_argument("--validation-submission-freezer", type=Path, required=True)
    parser.add_argument("--validation-return-auditor", type=Path, required=True)
    parser.add_argument("--validation-artifact-freezer", type=Path, required=True)
    parser.add_argument("--final-report-builder", type=Path, required=True)
    parser.add_argument("--validation-runtime-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    head = verify_repo(repo, args.master_checkpoint_commit)
    expected_code_paths = {
        "broad_feature_extractor": "scripts/analysis/hh4b_broad_feature_extractor_v1.py",
        "candidate_reconstruction_module": "scripts/analysis/hh4b_train_common_table_single_source_worker_v1.py",
        "validation_source_worker": "scripts/analysis/run_hh4b_cut_baseline_validation_source.py",
        "validation_aggregator": "scripts/analysis/aggregate_hh4b_cut_baseline_validation.py",
        "validation_plotter": "scripts/analysis/plot_hh4b_cut_baseline_validation.py",
        "validation_campaign_preparer": "scripts/analysis/prepare_hh4b_cut_baseline_validation_campaign.py",
        "validation_campaign_auditor": "scripts/analysis/audit_hh4b_cut_baseline_validation_campaign.py",
        "validation_campaign_presubmission_freezer": "scripts/analysis/freeze_hh4b_cut_baseline_validation_campaign_presubmission.py",
        "validation_campaign_submitter": "scripts/analysis/submit_hh4b_cut_baseline_validation_once.py",
        "validation_submission_freezer": "scripts/analysis/freeze_hh4b_cut_baseline_validation_submission.py",
        "validation_return_auditor": "scripts/analysis/audit_hh4b_cut_baseline_validation_returns.py",
        "validation_artifact_freezer": "scripts/analysis/freeze_hh4b_cut_baseline_validation_artifact_checkpoint.py",
        "final_report_builder": "scripts/analysis/build_hh4b_cut_baseline_final_report.py",
    }
    for argument_name, relative in expected_code_paths.items():
        require(
            getattr(args, argument_name).resolve() == (repo / relative).resolve(),
            f"authorization code path changed: {argument_name}",
        )
    verify_checkpoint(args.master_checkpoint)
    verify_checkpoint(args.validation_metadata_checkpoint)
    master_code_manifest = verify_master_code_manifest(repo, args.master_checkpoint.resolve())
    metadata_root = args.validation_metadata_checkpoint.resolve()
    for path in (args.source_access_manifest, args.physical_coefficient_registry):
        require(
            path.resolve().is_relative_to(metadata_root),
            f"validation metadata input is outside its frozen checkpoint: {path}",
        )
    require(args.master_summary.is_file(), "master summary is missing")
    master = json.loads(args.master_summary.read_text(encoding="utf-8"))
    validate_master_summary(master)
    for path in (args.master_summary, args.master_checkpoint / "SHA256SUMS"):
        relative = path.resolve().relative_to(repo).as_posix()
        committed = subprocess.check_output(
            ["git", "show", f"{args.master_checkpoint_commit}:{relative}"],
            cwd=repo,
        )
        require(
            sha256_bytes(committed) == sha256(path),
            f"master checkpoint file differs from committed bytes: {relative}",
        )

    inputs = [
        args.source_access_manifest,
        args.physical_coefficient_registry,
        args.broad_feature_extractor,
        args.candidate_reconstruction_module,
        args.validation_source_worker,
        args.validation_aggregator,
        args.validation_plotter,
        args.validation_campaign_preparer,
        args.validation_campaign_auditor,
        args.validation_campaign_presubmission_freezer,
        args.validation_campaign_submitter,
        args.validation_submission_freezer,
        args.validation_return_auditor,
        args.validation_artifact_freezer,
        args.final_report_builder,
        args.validation_runtime_bundle,
    ]
    for path in inputs:
        require(path.is_file() and not path.is_symlink(), f"missing/nonregular authorization input: {path}")

    access = pd.read_csv(args.source_access_manifest, sep="\t", keep_default_na=False)
    coefficients = pd.read_csv(
        args.physical_coefficient_registry, sep="\t", keep_default_na=False
    )
    require(len(access) == 121 and access["source_uid"].nunique() == 121, "validation metadata source closure changed")
    physical = access["physical_evaluation_eligible"].map(truthy)
    auxiliary = access["auxiliary_qcd"].map(truthy)
    require(int(physical.sum()) == 116, "physical validation source count changed")
    require(int(auxiliary.sum()) == 5, "auxiliary validation source count changed")
    require(physical.eq(~auxiliary).all(), "physical/auxiliary source partition changed")
    require(not access["validation_content_opened"].map(truthy).any(), "validation content already opened")
    require(not access["validation_access_authorized"].map(truthy).any(), "metadata pre-authorized validation")
    require(not access["test_content_opened"].map(truthy).any(), "test content opened")
    require(len(coefficients) == 121 and coefficients["source_uid"].is_unique, "coefficient registry closure changed")
    require(access["source_uid"].tolist() == coefficients["source_uid"].tolist(), "source/coefficient registry identity changed")
    require(
        coefficients.loc[physical, "run2_yield_coefficient_per_generator_weight"]
        .astype(float)
        .gt(0.0)
        .all(),
        "physical validation coefficient invalid",
    )
    require(
        coefficients.loc[~physical, "run2_yield_coefficient_per_generator_weight"]
        .astype(str)
        .map(clean)
        .eq("")
        .all(),
        "auxiliary QCD received a physical coefficient",
    )

    require(not args.output_dir.exists(), f"authorization output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=False)
    authorization = {
        "schema_version": 1,
        "status": "authorized_one_time_cut_baseline_validation",
        "repository_head": head,
        "master_train_only_checkpoint_commit": args.master_checkpoint_commit,
        "master_train_only_checkpoint_path": str(args.master_checkpoint),
        "master_train_only_checkpoint_sha256s": sha256(
            args.master_checkpoint / "SHA256SUMS"
        ),
        "master_train_only_code_manifest_sha256": sha256(master_code_manifest),
        "validation_metadata_checkpoint_path": str(
            args.validation_metadata_checkpoint
        ),
        "validation_metadata_checkpoint_sha256s": sha256(
            args.validation_metadata_checkpoint / "SHA256SUMS"
        ),
        "nominal_thresholds": NOMINAL_THRESHOLDS,
        "cut_family_frozen": True,
        "variables_frozen": True,
        "thresholds_frozen": True,
        "reporting_choices_frozen": True,
        "cut_scan_authorized": False,
        "threshold_adjustment_authorized": False,
        "family_adjustment_authorized": False,
        "authorized_validation_sources": 116,
        "auxiliary_qcd_validation_sources_authorized": 0,
        "auxiliary_qcd_validation_sources_remain_unopened": 5,
        "authorized_generated_events": int(
            pd.to_numeric(access.loc[physical, "generated_events"], errors="raise").sum()
        ),
        "source_payload_access_policy": (
            "each of 116 physical validation sources may be opened exactly once; "
            "a durable do-not-rerun marker must precede access"
        ),
        "validation_access_authorized": True,
        "validation_payloads_opened_before_authorization": 0,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
        "test_access_authorized": False,
        "authorized_sha256": {
            "source_access_manifest": sha256(args.source_access_manifest),
            "physical_coefficient_registry": sha256(
                args.physical_coefficient_registry
            ),
            "broad_feature_extractor": sha256(args.broad_feature_extractor),
            "candidate_reconstruction_module": sha256(
                args.candidate_reconstruction_module
            ),
            "validation_source_worker": sha256(args.validation_source_worker),
            "validation_aggregator": sha256(args.validation_aggregator),
            "validation_plotter": sha256(args.validation_plotter),
            "validation_campaign_preparer": sha256(
                args.validation_campaign_preparer
            ),
            "validation_campaign_auditor": sha256(
                args.validation_campaign_auditor
            ),
            "validation_campaign_presubmission_freezer": sha256(
                args.validation_campaign_presubmission_freezer
            ),
            "validation_campaign_submitter": sha256(
                args.validation_campaign_submitter
            ),
            "validation_submission_freezer": sha256(
                args.validation_submission_freezer
            ),
            "validation_return_auditor": sha256(
                args.validation_return_auditor
            ),
            "validation_artifact_freezer": sha256(
                args.validation_artifact_freezer
            ),
            "final_report_builder": sha256(args.final_report_builder),
            "validation_runtime_bundle": sha256(
                args.validation_runtime_bundle
            ),
        },
        "next": (
            "commit_and_push_this_authorization_then_prepare_exactly_once_"
            "116_source_validation_execution"
        ),
    }
    authorization_path = args.output_dir / "validation_authorization.json"
    authorization_path.write_text(
        json.dumps(authorization, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "VALIDATION_AUTHORIZED_TEST_SEALED").write_text(
        "VALIDATION_ACCESS_AUTHORIZED=TRUE\n"
        "VALIDATION_PAYLOADS_OPENED=0\n"
        "TEST_ACCESS_AUTHORIZED=FALSE\n"
        "TEST_PAYLOADS_OPENED=0\n",
        encoding="utf-8",
    )
    files = sorted(path for path in args.output_dir.iterdir() if path.is_file())
    (args.output_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )

    print("CUT_BASELINE_VALIDATION_AUTHORIZATION=PASS")
    print("AUTHORIZED_VALIDATION_SOURCES=116")
    print("AUXILIARY_QCD_VALIDATION_SOURCES_AUTHORIZED=0")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_ACCESS_AUTHORIZED=FALSE")
    print("TEST_PAYLOADS_OPENED=0")


if __name__ == "__main__":
    main()
