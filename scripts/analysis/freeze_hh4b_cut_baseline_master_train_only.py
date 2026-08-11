#!/usr/bin/env python3
"""Freeze the master train-only HH->4b cut-baseline contract.

This is the final gate before validation authorization.  It binds every
scientific choice, train-side result, publication sidecar, and validation
execution component while validation and test remain unopened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd


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


class MasterFreezeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MasterFreezeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def verify_repo(repo: Path) -> str:
    require(repo.resolve() == repo, "repository path is not canonical")
    require(git(repo, "branch", "--show-current") == "delphes-hh4b-production", "branch changed")
    head = git(repo, "rev-parse", "HEAD")
    require(head == git(repo, "rev-parse", "origin/delphes-hh4b-production"), "local/remote HEAD mismatch")
    return head


def verify_checkpoint(path: Path) -> None:
    require(path.is_dir() and not path.is_symlink(), f"invalid checkpoint: {path}")
    sums = path / "SHA256SUMS"
    require(sums.is_file() and not sums.is_symlink(), f"checkpoint lacks SHA256SUMS: {path}")
    result = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(result.returncode == 0, f"checkpoint checksum failure: {path}: {result.stdout}{result.stderr}")
    listed = {
        line.split("  ", 1)[1].removeprefix("./")
        for line in sums.read_text(encoding="utf-8").splitlines()
    }
    actual = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file()
    }
    require(actual == listed | {"SHA256SUMS"}, f"checkpoint file closure changed: {path}")


def require_committed_unchanged(repo: Path, path: Path) -> None:
    relative = path.relative_to(repo).as_posix()
    tracked = set(git(repo, "ls-files", "--", relative).splitlines())
    actual = (
        {item.relative_to(repo).as_posix() for item in path.rglob("*") if item.is_file()}
        if path.is_dir()
        else {relative}
    )
    require(actual <= tracked, f"master input contains uncommitted files: {path}")
    require(
        subprocess.run(["git", "diff", "--quiet", "--", relative], cwd=repo).returncode
        == 0,
        f"master input has unstaged changes: {path}",
    )
    require(
        subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", relative], cwd=repo
        ).returncode
        == 0,
        f"master input has staged changes: {path}",
    )


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing/nonregular JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON root is not an object: {path}")
    return payload


def require_sealed(payload: dict[str, Any], label: str) -> None:
    require(payload.get("validation_payloads_opened") == 0, f"validation opened in {label}")
    require(payload.get("test_payloads_opened") == 0, f"test opened in {label}")


def validate_summary_contracts(
    full270: dict[str, Any],
    nested: dict[str, Any],
    deployment: dict[str, Any],
    initial200: dict[str, Any],
    escalation800: dict[str, Any],
    all1000: dict[str, Any],
    train: dict[str, Any],
    publication: dict[str, Any],
    validation_metadata: dict[str, Any],
) -> None:
    expected_status = {
        "full270": (full270, "pass_full_270_returned_result_integrity_audit"),
        "nested": (nested, "pass_outer_fold_category_family_selection"),
        "deployment": (deployment, "train_only_deployment_candidate_frozen"),
        "initial200": (initial200, "pass_initial200_predeclared_selection_stability_aggregation"),
        "escalation800": (
            escalation800,
            "pass_complete_escalation800_8000_job_216000_structure_return_audit",
        ),
        "all1000": (all1000, "pass_all1000_predeclared_selection_stability_aggregation"),
        "train": (train, "pass_hh4b_cut_baseline_train_performance_and_fixed_comparator"),
        "publication": (
            publication,
            "pass_hh4b_cut_baseline_train_only_publication_figures",
        ),
        "validation_metadata": (validation_metadata, "pass_metadata_only"),
    }
    for label, (payload, status) in expected_status.items():
        require(payload.get("status") == status, f"{label} status changed")
        require_sealed(payload, label)

    require(nested.get("winner_count") == 10, "nested outer-OOF winner count changed")
    require(nested.get("outer_fold_results_used_for_selection") is False, "outer-fold result leakage")
    require(deployment.get("nominal_selection_is_changed_by_stability_escalation") is False, "deployment allowed stability retuning")
    observed_thresholds = {
        category: deployment["categories"][category][
            "deployment_thresholds_coordinatewise_median"
        ]
        for category in ("exact3tag", "ge4tag")
    }
    require(observed_thresholds == NOMINAL_THRESHOLDS, "nominal deployment thresholds changed")
    require(initial200.get("nominal_deployment_candidate_changed") is False, "initial200 changed nominal candidate")
    require(escalation800.get("jobs_audited") == 8000, "escalation job count changed")
    require(escalation800.get("structure_results_audited") == 216000, "escalation result count changed")
    require(escalation800.get("nominal_selection_changed") is False, "escalation changed nominal candidate")
    require(all1000.get("fold_level_winner_count") == 10000, "all1000 winner count changed")
    require(all1000.get("replica_category_result_count") == 2000, "all1000 reduction count changed")
    require(all1000.get("ranked_structure_result_count") == 270000, "all1000 structure count changed")
    require(all1000.get("nominal_deployment_candidate_changed") is False, "all1000 changed nominal candidate")
    require(train.get("primary_generalization_estimate") == "pooled_nested_outer_oof", "primary train estimate changed")
    require(train.get("fixed_deployment_cut_called_primary_generalization_estimate") is False, "fixed train diagnostic mislabeled primary")
    require(train.get("historical_comparator_threshold_scan_performed") is False, "historical comparator was tuned")
    require(train.get("historical_comparator_threshold") == {"r_hh_125_125_lt": 34.0}, "historical comparator changed")
    require(train.get("official_cms_result") is False, "train output claims official CMS status")
    require(publication.get("figures") == publication.get("pdfs") == publication.get("pngs") == 16, "train figure closure changed")
    require(publication.get("figure_data_sidecars") == 16, "figure sidecar count changed")
    require(publication.get("official_cms_status_claimed") is False, "figures claim official CMS status")
    require(validation_metadata.get("event_payload_files_opened") == 0, "validation event payload opened during metadata preparation")
    require(validation_metadata.get("validation_access_authorized") is False, "validation metadata pre-authorized access")
    require(validation_metadata.get("physical_evaluation_sources") == 116, "validation physical-source count changed")
    require(validation_metadata.get("auxiliary_qcd_sources") == 5, "validation auxiliary-source count changed")
    require(validation_metadata.get("remote_bundle_checksum_closure") == "pass_116_of_116", "validation checksum closure changed")


def resolve(repo: Path, value: Path) -> Path:
    path = value if value.is_absolute() else repo / value
    path = path.resolve()
    require(path.is_relative_to(repo), f"checkpoint is outside repository: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines"))
    parser.add_argument("--full270-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_train_multivariate_cut_full_270_integrity_audit_20260807_v1"))
    parser.add_argument("--nested-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_train_multivariate_cut_train_only_nested_oof_aggregation_20260807_v1"))
    parser.add_argument("--deployment-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_train_multivariate_cut_train_only_deployment_candidate_20260807_v1"))
    parser.add_argument("--initial200-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_initial200_aggregation_20260809_v1"))
    parser.add_argument("--escalation800-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_escalation800_complete_return_audit_20260810_v1"))
    parser.add_argument("--all1000-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_all1000_aggregation_20260810_v1"))
    parser.add_argument("--train-performance-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_train_cut_baseline_performance_and_comparator_20260810_v1"))
    parser.add_argument("--publication-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_train_cut_baseline_publication_figures_20260810_v1"))
    parser.add_argument("--validation-metadata-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_cut_baseline_validation_metadata_20260810_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/checkpoints/hh4b_cut_baseline_master_train_only_freeze_20260810_v1"))
    args = parser.parse_args()

    repo = args.repo.resolve()
    head = verify_repo(repo)
    checkpoints = {
        "full270": resolve(repo, args.full270_checkpoint),
        "nested": resolve(repo, args.nested_checkpoint),
        "deployment": resolve(repo, args.deployment_checkpoint),
        "initial200": resolve(repo, args.initial200_checkpoint),
        "escalation800": resolve(repo, args.escalation800_checkpoint),
        "all1000": resolve(repo, args.all1000_checkpoint),
        "train": resolve(repo, args.train_performance_checkpoint),
        "publication": resolve(repo, args.publication_checkpoint),
        "validation_metadata": resolve(repo, args.validation_metadata_checkpoint),
    }
    for checkpoint in checkpoints.values():
        verify_checkpoint(checkpoint)
        require_committed_unchanged(repo, checkpoint)
    summaries = {
        "full270": checkpoints["full270"] / "evidence/full_270_returned_result_integrity_audit.json",
        "nested": checkpoints["nested"] / "evidence/outer_fold_category_winners.json",
        "deployment": checkpoints["deployment"] / "train_only_deployment_candidate.json",
        "initial200": checkpoints["initial200"] / "evidence/aggregation/initial200_stability_summary.json",
        "escalation800": checkpoints["escalation800"] / "complete_return_audit.json",
        "all1000": checkpoints["all1000"] / "evidence/aggregation/all1000_stability_summary.json",
        "train": checkpoints["train"] / "evidence/train_performance/train_performance_summary.json",
        "publication": checkpoints["publication"] / "evidence/publication_train_only/publication_figure_summary.json",
        "validation_metadata": checkpoints["validation_metadata"] / "validation_metadata_preparation_summary.json",
    }
    payloads = {name: load_json(path) for name, path in summaries.items()}
    validate_summary_contracts(**payloads)

    code_paths = [
        "scripts/analysis/capture_hh4b_escalation800_final_scheduler_evidence.py",
        "scripts/analysis/audit_hh4b_escalation800_complete_returns.py",
        "scripts/analysis/freeze_hh4b_escalation800_complete_return_audit.py",
        "scripts/analysis/aggregate_hh4b_all1000_selection_stability.py",
        "scripts/analysis/build_hh4b_cut_baseline_train_performance.py",
        "scripts/analysis/plot_hh4b_cut_baseline_publication.py",
        "scripts/analysis/freeze_hh4b_cut_baseline_train_artifact_checkpoint.py",
        "scripts/analysis/hh4b_broad_feature_extractor_v1.py",
        "scripts/analysis/hh4b_train_common_table_single_source_worker_v1.py",
        "scripts/analysis/query_hh4b_cut_validation_remote_checksums.py",
        "scripts/analysis/prepare_hh4b_cut_baseline_validation_metadata.py",
        "scripts/analysis/freeze_hh4b_cut_baseline_validation_metadata.py",
        "scripts/analysis/freeze_hh4b_cut_baseline_master_train_only.py",
        "scripts/analysis/authorize_hh4b_cut_baseline_validation.py",
        "scripts/analysis/prepare_hh4b_cut_baseline_validation_campaign.py",
        "scripts/analysis/audit_hh4b_cut_baseline_validation_campaign.py",
        "scripts/analysis/freeze_hh4b_cut_baseline_validation_campaign_presubmission.py",
        "scripts/analysis/submit_hh4b_cut_baseline_validation_once.py",
        "scripts/analysis/freeze_hh4b_cut_baseline_validation_submission.py",
        "scripts/analysis/run_hh4b_cut_baseline_validation_source.py",
        "scripts/analysis/audit_hh4b_cut_baseline_validation_returns.py",
        "scripts/analysis/aggregate_hh4b_cut_baseline_validation.py",
        "scripts/analysis/plot_hh4b_cut_baseline_validation.py",
        "scripts/analysis/freeze_hh4b_cut_baseline_validation_artifact_checkpoint.py",
        "scripts/analysis/build_hh4b_cut_baseline_final_report.py",
    ]
    code_rows = []
    for relative in code_paths:
        path = repo / relative
        require(path.is_file() and not path.is_symlink(), f"missing master-freeze code: {path}")
        require_committed_unchanged(repo, path)
        code_rows.append({"relative_path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})

    final_output = resolve(repo, args.output_dir)
    require(not final_output.exists(), f"master checkpoint already exists: {final_output}")
    output = final_output.parent / f".{final_output.name}.build.{os.getpid()}"
    require(not output.exists(), f"master checkpoint build path exists: {output}")
    output.mkdir()
    checkpoint_rows = []
    for name in checkpoints:
        checkpoint_rows.append(
            {
                "role": name,
                "checkpoint_path": str(checkpoints[name].relative_to(repo)),
                "checkpoint_sha256s_sha256": sha256(checkpoints[name] / "SHA256SUMS"),
                "summary_path": str(summaries[name].relative_to(repo)),
                "summary_sha256": sha256(summaries[name]),
                "summary_status": payloads[name]["status"],
                "validation_payloads_opened": 0,
                "test_payloads_opened": 0,
            }
        )
    pd.DataFrame(checkpoint_rows).to_csv(
        output / "input_checkpoint_manifest.tsv", sep="\t", index=False, lineterminator="\n"
    )
    pd.DataFrame(code_rows).to_csv(
        output / "code_manifest.tsv", sep="\t", index=False, lineterminator="\n"
    )
    methodology = r"""# Frozen HH→4b cut-baseline methodology

- The primary train-side generalization estimate is pooled five-fold nested outer-OOF performance.
- The median deployment cut evaluated on train is a fixed diagnostic, not the primary generalization estimate.
- The historical comparator is fixed at $R_{HH}(125,125)<34$ and was not optimized here.
- The 1,000-replica selection-stability bootstrap is diagnostic and cannot change the nominal cut.
- The 2,000-replica paired source-group metric bootstrap evaluates frozen selections and does not rerun selection optimization.
- Expected yields use $\sqrt{s}=13$ TeV and 138 fb$^{-1}$ equivalent (138000 pb$^{-1}$ internally).
- Results are Delphes simulation projections and are not an official CMS measurement.
- Validation may be evaluated once only with the frozen nominal cut after a separate authorization checkpoint.
- Test remains sealed for the later frozen cross-model comparison.
"""
    (output / "METHODOLOGY.md").write_text(methodology, encoding="utf-8")
    master = {
        "schema_version": 1,
        "status": "pass_master_train_only_hh4b_cut_baseline_freeze",
        "repository_head": head,
        "nominal_thresholds": NOMINAL_THRESHOLDS,
        "nominal_selection_changed": False,
        "cut_family_frozen": True,
        "variables_frozen": True,
        "thresholds_frozen": True,
        "search_budget_frozen": True,
        "reporting_choices_frozen": True,
        "primary_generalization_estimate": "pooled_nested_outer_oof",
        "fixed_train_deployment_cut_role": "diagnostic_only",
        "historical_comparator": {"r_hh_125_125_lt": 34.0, "optimized_here": False},
        "selection_stability_replicas": 1000,
        "selection_stability_changes_nominal_candidate": False,
        "paired_metric_bootstrap_replicas": 2000,
        "paired_metric_bootstrap_seed": 20260727,
        "sqrt_s_tev": 13.0,
        "luminosity_fb_inverse": 138.0,
        "luminosity_pb_inverse_internal": 138000.0,
        "yield_interpretation": "Run-2 expected-yield projection",
        "official_cms_result": False,
        "train_pooled_metrics": payloads["train"]["pooled_metrics"],
        "initial200_category_diagnostics": payloads["initial200"]["categories"],
        "all1000_category_diagnostics": payloads["all1000"]["categories"],
        "train_publication_figures": 16,
        "train_publication_figure_data_sidecars": 16,
        "validation_metadata_sources": 121,
        "validation_physical_sources_authorizable_later": 116,
        "validation_auxiliary_qcd_sources_remain_unopened": 5,
        "validation_access_authorized": False,
        "validation_payloads_opened": 0,
        "test_access_authorized": False,
        "test_payloads_opened": 0,
        "next": "commit_and_push_master_checkpoint_then_create_explicit_one_time_validation_authorization",
    }
    (output / "master_train_only_summary.json").write_text(
        json.dumps(master, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    files = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    os.replace(output, final_output)
    print("HH4B_CUT_BASELINE_MASTER_TRAIN_ONLY_FREEZE=PASS")
    print("NOMINAL_SELECTION_CHANGED=FALSE")
    print("VALIDATION_ACCESS_AUTHORIZED=FALSE")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_ACCESS_AUTHORIZED=FALSE")
    print("TEST_PAYLOADS_OPENED=0")


if __name__ == "__main__":
    main()
