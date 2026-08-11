#!/usr/bin/env python3
"""Build the final HH->4b cut-baseline report after pre-access failure.

This is deliberately separate from the successful-validation final-report
builder.  It accepts no validation-performance input and fails closed if any
validation or test payload access, scientific validation metric, or retry
authorization is reported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from typing import Any


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
EXPECTED_RUNTIME_SHA256 = "4a9c4061e1d975f276326022454e578c5f0de3c094b9b56345f192cccc654112"
EXPECTED_FROZEN_RUNNER_SHA256 = "20c41377c2aaadf5d2de26c3698bac570022e81c81105a30889af6c50e3e59c1"
VALIDATION_CLUSTER = 3795859
VALIDATION_SCHEDD = "lpcschedd4.fnal.gov"
NEXT_DECISION = (
    "determine separately whether a replacement validation campaign is "
    "scientifically/procedurally permissible given that the failed campaign "
    "never opened validation payloads"
)


class BlockedFinalReportError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BlockedFinalReportError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing/nonregular JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON root is not an object: {path}")
    return payload


def verify_sha256s(root: Path) -> None:
    require(root.is_dir() and not root.is_symlink(), f"invalid checksummed root: {root}")
    require((root / "SHA256SUMS").is_file(), f"missing SHA256SUMS: {root}")
    result = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(
        result.returncode == 0,
        f"checksum failure: {root}: {result.stdout}{result.stderr}",
    )


def require_committed_unchanged(repo: Path, root: Path) -> None:
    relative = root.relative_to(repo).as_posix()
    actual = {
        path.relative_to(repo).as_posix() for path in root.rglob("*") if path.is_file()
    }
    tracked = set(git(repo, "ls-files", "--", relative).splitlines())
    require(actual <= tracked, f"artifact contains uncommitted/ignored files: {root}")
    require(
        subprocess.run(["git", "diff", "--quiet", "--", relative], cwd=repo).returncode == 0,
        f"artifact has unstaged changes: {root}",
    )
    require(
        subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", relative], cwd=repo
        ).returncode
        == 0,
        f"artifact has staged changes: {root}",
    )


def checkpoint_record(repo: Path, root: Path) -> dict[str, Any]:
    verify_sha256s(root)
    require_committed_unchanged(repo, root)
    relative = root.relative_to(repo).as_posix()
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "path": relative,
        "files": len(files),
        "sha256s_sha256": sha256(root / "SHA256SUMS"),
        "last_checkpoint_commit": git(repo, "log", "-1", "--format=%H", "--", relative),
    }


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def validate_blocked_summaries(summaries: dict[str, dict[str, Any]]) -> None:
    expected_statuses = {
        "initial200": "pass_initial200_predeclared_selection_stability_aggregation",
        "all1000": "pass_all1000_predeclared_selection_stability_aggregation",
        "train": "pass_hh4b_cut_baseline_train_performance_and_fixed_comparator",
        "publication_train": "pass_hh4b_cut_baseline_train_only_publication_figures",
        "master": "pass_master_train_only_hh4b_cut_baseline_freeze",
        "authorization": "authorized_one_time_cut_baseline_validation",
        "campaign": "pass_exactly_once_validation_campaign_frozen_before_submission",
        "submission": "pass_exactly_once_validation_submission_frozen_before_monitoring",
        "transport": "pass_validation_cluster_in_place_transport_recovery_checkpoint_freeze",
        "failure": "blocked_validation_campaign_failed_before_payload_access",
    }
    for role, expected in expected_statuses.items():
        require(summaries[role].get("status") == expected, f"input status changed: {role}")
        require(summaries[role].get("test_payloads_opened") == 0, f"test opened in {role}")

    for role in (
        "initial200",
        "all1000",
        "train",
        "publication_train",
        "master",
        "authorization",
        "campaign",
        "failure",
    ):
        require(
            summaries[role].get("validation_payloads_opened") == 0,
            f"validation payload opened in {role}",
        )
    require(
        summaries["submission"].get("validation_payloads_opened_before_submission") == 0,
        "validation opened before submission",
    )
    require(
        summaries["transport"].get("validation_payloads_opened_before_release") == 0,
        "validation opened before transport recovery release",
    )
    require(summaries["master"].get("nominal_thresholds") == NOMINAL_THRESHOLDS, "nominal cut changed")
    require(summaries["master"].get("nominal_selection_changed") is False, "master selection changed")
    require(summaries["all1000"].get("nominal_deployment_candidate_changed") is False, "stability changed nominal cut")
    require(summaries["train"].get("primary_generalization_estimate") == "pooled_nested_outer_oof", "primary train estimate changed")
    require(summaries["train"].get("historical_comparator_threshold_scan_performed") is False, "historical comparator was tuned")
    require(summaries["train"].get("historical_comparator_threshold") == {"r_hh_125_125_lt": 34.0}, "historical comparator changed")
    require(summaries["train"].get("official_cms_result") is False, "train result claims official CMS status")
    require(summaries["publication_train"].get("official_cms_status_claimed") is False, "figures claim official CMS status")
    require(summaries["publication_train"].get("figures") == 16, "train figure count changed")

    submission = summaries["submission"]
    transport = summaries["transport"]
    failure = summaries["failure"]
    for role, payload in (("submission", submission), ("transport", transport), ("failure", failure)):
        require(payload.get("cluster_id") == VALIDATION_CLUSTER, f"cluster changed in {role}")
        require(payload.get("authoritative_schedd") == VALIDATION_SCHEDD, f"schedd changed in {role}")
        require(payload.get("do_not_resubmit_cluster") is True, f"resubmission allowed in {role}")
    require(submission.get("jobs_submitted") == 116, "submitted job count changed")
    require(transport.get("condor_submit_called") is False, "transport recovery resubmitted")
    require(transport.get("scientific_arguments_changed") is False, "transport recovery changed science")
    require(failure.get("queue_jobs") == 0, "validation queue is not empty")
    require(failure.get("history_jobs") == 116, "validation history coverage changed")
    require(failure.get("exit_code_1_jobs") == 116, "validation exit coverage changed")
    require(failure.get("runtime_probe_failure_jobs") == 116, "runtime-probe failure coverage changed")
    require(failure.get("durable_source_attempt_markers_created") == 0, "validation marker was created")
    require(failure.get("validation_workers_started") == 0, "validation worker started")
    require(failure.get("validation_sources_opened") == 0, "validation source opened")
    require(failure.get("validation_evaluation_cycles") == 0, "validation evaluation occurred")
    require(failure.get("validation_metrics_available") is False, "validation metric was reported")
    require(failure.get("remote_result_namespace_absent") is True, "validation result namespace exists")
    require(failure.get("second_validation_submission_authorized") is False, "validation retry was authorized")
    require(failure.get("nominal_selection_changed") is False, "failure changed nominal selection")
    require(failure.get("failure_stage") == "runtime_probe_before_marker_and_worker", "failure stage changed")
    require(failure.get("failure_exception") == "ModuleNotFoundError: No module named 'awkward'", "failure exception changed")
    require(
        failure.get("root_cause")
        == "frozen runner exported $SCRATCH/runtime/site-packages while the authorized archive stores packages under lib/python3.9/site-packages",
        "validation root cause changed",
    )


def metric_row(label: str, metrics: dict[str, Any]) -> str:
    fields = (
        "signal_physical_efficiency",
        "background_physical_efficiency",
        "background_rejection",
        "signal_over_background",
        "asimov_significance_stat_only",
        "signal_selected_effective_events",
        "background_selected_effective_events",
    )
    values = [f"{metrics[field]:.8g}" for field in fields]
    return "| " + " | ".join([label, *values]) + " |"


def render_markdown(report: dict[str, Any]) -> str:
    nested = report["train_only"]["pooled_nested_outer_oof"]
    historical = report["train_only"]["historical_rhh125125_lt34"]
    stability = report["train_only"]["all1000_selection_stability"]
    validation = report["validation"]
    checkpoint_rows = "\n".join(
        f"| `{row['path']}` | `{row['sha256s_sha256']}` | `{row['last_checkpoint_commit']}` |"
        for row in report["checkpoints"]
    )
    cluster_rows = "\n".join(
        f"| {row['cluster_id']} | {row['role']} | `{row['authoritative_schedd']}` | YES |"
        for row in report["clusters"]
    )
    stability_rows = "\n".join(
        "| "
        + " | ".join(
            [
                category,
                values["modal_replica_selected_structure_id"],
                f"{values['modal_replica_selected_frequency']:.3f}",
                f"{values['nominal_structure_recovery_frequency']:.3f}",
                f"{values['top_two_frequency_gap']:.3f}",
                f"{values['tie_resolution_replica_category_fraction']:.3f}",
                f"{values['infeasible_fold_winner_frequency']:.4f}",
            ]
        )
        + " |"
        for category, values in stability["categories"].items()
    )
    return f"""# Resolved HH→4b cut-baseline final status

Status: **BLOCKED AT VALIDATION BY PRE-ACCESS INFRASTRUCTURE FAILURE**

- Train-only cut-baseline study: **COMPLETE**.
- Train-only methodology and results: **FROZEN**.
- Validation attempt: infrastructure failure before event access.
- Validation scientific result: **NOT AVAILABLE**.
- Validation payloads opened: **0**.
- Test payloads opened: **0**.
- Corrected validation retry authorized: **FALSE**.

This is not a scientific failure of the cut. No scheduler failure is interpreted as a physics observation, no partial validation performance is derived, and no threshold, variable, family, search budget, nominal rule, methodology, or reporting choice was changed.

## Frozen nominal deployment candidate

- `exact3tag`: `r_hh_125_125 < 36.40814019639858` and `HT_candidate_jets > 176.5458068847656 GeV`.
- `ge4tag`: `r_hh_125_125 < 33.92808917804956`, `mHH > 164.73708096689654 GeV`, and `|Delta eta_HH| < 6.904302164473993`.

The historical primary simple cut reference remains `R_HH(125,125) < 34`, including for subsequent apples-to-apples model comparisons.

## Train-only performance

Primary generalization estimate: pooled five-fold nested outer-OOF. Values are a Run-2 expected-yield projection from Delphes simulation at `sqrt(s)=13 TeV`, `138 fb^-1` equivalent. These are not official CMS results and include no systematic uncertainties.

| Selection/scope | epsS | epsB | background rejection | S/B | stat-only ZA | signal Neff | background Neff |
|---|---:|---:|---:|---:|---:|---:|---:|
{metric_row('nested outer-OOF / exact3tag', nested['exact3tag'])}
{metric_row('nested outer-OOF / ge4tag', nested['ge4tag'])}
{metric_row('nested outer-OOF / combined', nested['combined'])}
{metric_row('historical R_HH<34 / combined', historical['combined'])}

The frozen optimized deployment cut has no material stat-only significance improvement over the historical simple cut. In the frozen 2,000-draw paired source-group bootstrap, the combined fixed-cut-minus-historical `ZA` difference has median `{report['train_only']['paired_bootstrap_combined_za_difference']['difference_median']}` and 95% interval `[{report['train_only']['paired_bootstrap_combined_za_difference']['difference_p2p5']}, {report['train_only']['paired_bootstrap_combined_za_difference']['difference_p97p5']}]`, spanning zero.

## All-1000 selection stability

The full study contains 270,000 ranked structure evaluations, 10,000 fold winners, and 2,000 replica/category reductions. The detailed optimized structures are not especially stable under resampling; this did not change the frozen nominal cut.

| Category | modal structure | modal frequency | nominal recovery | top-two gap | tie fraction | infeasible fold-winner fraction |
|---|---|---:|---:|---:|---:|---:|
{stability_rows}

## Validation and test boundary

Exactly one campaign submission was accepted: cluster `{validation['cluster_id']}` on `{validation['authoritative_schedd']}`. All 116 jobs subsequently exited during the frozen runtime probe because the runner exported `$SCRATCH/runtime/site-packages`, while the authorized runtime archive stores `awkward` and the other packages under `lib/python3.9/site-packages`. The common exception was `ModuleNotFoundError: No module named 'awkward'`.

The audit proves queue=0, history=116, runtime-probe failures=116, durable markers=0, scientific workers=0, validation sources opened=0, validation evaluation cycles=0, validation payloads opened=0, and test payloads opened=0. Consequently no validation aggregation or validation-performance figure is scientifically allowed.

**Figure 17 / validation-comparison products were intentionally not produced because no validation event payload was opened and no valid validation performance measurement exists.** The completed publication set contains 16 train-only figures, each with PDF, PNG, and machine-readable sidecar.

Cluster 3795859 is permanently NEVER RESUBMIT. A corrected replacement validation campaign is not authorized by this report or this session.

## Immutable production registry

| Cluster | Role | Schedd | Never resubmit |
|---:|---|---|---|
{cluster_rows}

## Checkpoint closure

| Checkpoint | SHA256SUMS SHA256 | Last checkpoint commit |
|---|---|---|
{checkpoint_rows}

The pre-access failure checkpoint is `{validation['failure_audit_checkpoint']}` with SHA256SUMS hash `{validation['failure_audit_checkpoint_sha256s_sha256']}`. The frozen runner hash is `{validation['frozen_runner_sha256']}`; the runtime bundle hash is `{validation['runtime_bundle_sha256']}`.

## Remaining decision

`NEXT_DECISION_REQUIRED={report['next_decision_required']}`

No cross-model final test evaluation may occur before that separate decision and any newly authorized, provenance-complete validation procedure. `VALIDATION_SCIENTIFIC_RESULT_AVAILABLE=FALSE`, `VALIDATION_PAYLOADS_OPENED=0`, `TEST_PAYLOADS_OPENED=0`, and `CORRECTED_VALIDATION_RETRY_AUTHORIZED=FALSE`.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines"),
    )
    parser.add_argument(
        "--runtime-bundle",
        type=Path,
        default=Path(
            "/uscms_data/d3/iturkmen/hh4b_delphes/baselines/"
            "multivariate_cut_bounded_six_job_transfer_pilot_v8_r1_20260807T004537Z/"
            "archives/hh4b_python_runtime_v8_r1.tar.gz"
        ),
    )
    parser.add_argument("--output-json", type=Path, default=Path("HH4B_CUT_BASELINE_FINAL_STATUS.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("HH4B_CUT_BASELINE_FINAL_STATUS.md"))
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/hh4b_cut_baseline/final_status_blocked_validation"),
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    require(repo == args.repo, "repository path is not canonical")
    require(git(repo, "branch", "--show-current") == BRANCH, "branch changed")
    head = git(repo, "rev-parse", "HEAD")
    require(head == git(repo, "rev-parse", f"origin/{BRANCH}"), "local/remote HEAD mismatch")

    def resolve(value: Path) -> Path:
        path = value if value.is_absolute() else repo / value
        path = path.resolve()
        require(path.is_relative_to(repo), f"path outside repository: {path}")
        return path

    summary_paths = {
        "initial200": repo / "docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_initial200_aggregation_20260809_v1/evidence/aggregation/initial200_stability_summary.json",
        "all1000": repo / "docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_all1000_aggregation_20260810_v1/evidence/aggregation/all1000_stability_summary.json",
        "train": repo / "docs/checkpoints/hh4b_train_cut_baseline_performance_and_comparator_20260810_v1/evidence/train_performance/train_performance_summary.json",
        "publication_train": repo / "docs/checkpoints/hh4b_train_cut_baseline_publication_figures_20260810_v1/evidence/publication_train_only/publication_figure_summary.json",
        "master": repo / "docs/checkpoints/hh4b_cut_baseline_master_train_only_freeze_20260811_v3/master_train_only_summary.json",
        "authorization": repo / "docs/checkpoints/hh4b_cut_baseline_validation_authorization_20260811_v3/validation_authorization.json",
        "campaign": repo / "docs/checkpoints/hh4b_cut_baseline_validation_campaign_presubmission_20260811_v1/validation_campaign_presubmission_freeze.json",
        "submission": repo / "docs/checkpoints/hh4b_cut_baseline_validation_submission_20260811_v1/validation_submission_freeze.json",
        "transport": repo / "docs/checkpoints/hh4b_cut_baseline_validation_transport_recovery_20260811_v1/validation_transport_recovery_freeze.json",
        "failure": repo / "docs/checkpoints/hh4b_cut_baseline_validation_preaccess_failure_audit_20260811_v1/validation_preaccess_failure_audit.json",
    }
    summaries = {role: load_json(path) for role, path in summary_paths.items()}
    validate_blocked_summaries(summaries)

    checkpoint_root = repo / "docs/checkpoints"
    prefixes = (
        "hh4b_train_multivariate_cut_",
        "hh4b_train_cut_baseline_",
        "hh4b_train_historical_rhh125125_lt34_",
        "hh4b_cut_baseline_",
    )
    checkpoint_paths = sorted(
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir() and path.name.startswith(prefixes) and (path / "SHA256SUMS").is_file()
    )
    # Summary files can be nested several levels below their checkpoint roots.
    for summary_path in summary_paths.values():
        matches = [root for root in checkpoint_paths if summary_path.is_relative_to(root)]
        require(len(matches) == 1, f"summary is not bound to one checkpoint: {summary_path}")
    checkpoints = [checkpoint_record(repo, path) for path in checkpoint_paths]

    publication_root = repo / "artifacts/hh4b_cut_baseline/publication_train_only"
    for artifact_root in (
        repo / "artifacts/hh4b_cut_baseline/initial200_stability",
        repo / "artifacts/hh4b_cut_baseline/all1000_stability",
        repo / "artifacts/hh4b_cut_baseline/all1000_stability_independent_audit",
        repo / "artifacts/hh4b_cut_baseline/train_performance",
        publication_root,
    ):
        verify_sha256s(artifact_root)
        require_committed_unchanged(repo, artifact_root)
    pdfs = sorted((publication_root / "figures/pdf").glob("*.pdf"))
    pngs = sorted((publication_root / "figures/png").glob("*.png"))
    sidecars = sorted((publication_root / "figure_data").glob("*.tsv"))
    require(len(pdfs) == len(pngs) == len(sidecars) == 16, "train Figure 1-16 closure changed")
    require(not (repo / "artifacts/hh4b_cut_baseline/publication_validation").exists(), "validation publication artifact unexpectedly exists")
    require(not (repo / "artifacts/hh4b_cut_baseline/validation_performance").exists(), "validation performance artifact unexpectedly exists")

    campaign_root = repo / "docs/checkpoints/hh4b_cut_baseline_validation_campaign_presubmission_20260811_v1"
    runner = campaign_root / "evidence/campaign_package/run_validation_source.sh"
    campaign_summary = load_json(campaign_root / "evidence/campaign_package/campaign_summary.json")
    require(sha256(runner) == EXPECTED_FROZEN_RUNNER_SHA256, "frozen runner hash changed")
    require(campaign_summary.get("runner_sha256") == EXPECTED_FROZEN_RUNNER_SHA256, "campaign runner binding changed")
    runner_text = runner.read_text(encoding="utf-8")
    require('export PYTHONPATH="$SCRATCH/runtime/site-packages"' in runner_text, "diagnosed frozen runner mismatch absent")

    runtime = args.runtime_bundle.resolve()
    require(runtime.is_file() and not runtime.is_symlink(), "runtime bundle is missing/nonregular")
    require(sha256(runtime) == EXPECTED_RUNTIME_SHA256, "runtime bundle hash changed")
    require(campaign_summary.get("validation_runtime_bundle_sha256") == EXPECTED_RUNTIME_SHA256, "campaign runtime binding changed")
    with tarfile.open(runtime, "r:gz") as archive:
        names = set(archive.getnames())
    require("./lib/python3.9/site-packages/awkward/__init__.py" in names, "authorized runtime layout changed")
    require("./site-packages/awkward/__init__.py" not in names, "unexpected top-level runtime site-packages exists")

    paired_rows = read_tsv(
        repo / "docs/checkpoints/hh4b_train_cut_baseline_performance_and_comparator_20260810_v1/evidence/train_performance/tables/paired_bootstrap_difference_summary.tsv"
    )
    combined_za = [
        row
        for row in paired_rows
        if row["scope"] == "combined" and row["metric"] == "asimov_significance_stat_only"
    ]
    require(len(combined_za) == 1, "paired combined ZA row closure changed")
    combined_za_row = combined_za[0]

    failure_root = repo / "docs/checkpoints/hh4b_cut_baseline_validation_preaccess_failure_audit_20260811_v1"
    failure_record = next(row for row in checkpoints if row["path"] == failure_root.relative_to(repo).as_posix())
    clusters = [
        {"cluster_id": 3754344, "role": "bounded_six_job_transfer_pilot", "jobs": 6, "authoritative_schedd": "lpcschedd4.fnal.gov", "never_resubmit": True},
        {"cluster_id": 3755882, "role": "original_270_scan", "jobs": 270, "authoritative_schedd": "lpcschedd4.fnal.gov", "never_resubmit": True},
        {"cluster_id": 3768139, "role": "four_job_transfer_pilot", "jobs": 4, "authoritative_schedd": "lpcschedd4.fnal.gov", "never_resubmit": True},
        {"cluster_id": 30002685, "role": "initial200_stability", "jobs": 2000, "authoritative_schedd": "lpcschedd5.fnal.gov", "never_resubmit": True},
        {"cluster_id": 30020809, "role": "escalation800_stability", "jobs": 8000, "authoritative_schedd": "lpcschedd5.fnal.gov", "never_resubmit": True},
        {"cluster_id": VALIDATION_CLUSTER, "role": "one_time_validation_preaccess_failure", "jobs": 116, "authoritative_schedd": VALIDATION_SCHEDD, "never_resubmit": True},
    ]
    require(len({row["cluster_id"] for row in clusters}) == 6, "never-resubmit registry is not unique")

    train = summaries["train"]
    all1000 = summaries["all1000"]
    failure = summaries["failure"]
    report = {
        "schema_version": 1,
        "status": "blocked_validation_preaccess_infrastructure_failure",
        "branch": BRANCH,
        "repository_parent_head": head,
        "report_builder": {
            "path": Path(__file__).resolve().relative_to(repo).as_posix(),
            "sha256": sha256(Path(__file__).resolve()),
            "commit": git(repo, "log", "-1", "--format=%H", "--", Path(__file__).resolve().relative_to(repo).as_posix()),
        },
        "train_only": {
            "scientific_status": "complete",
            "methodology_status": "frozen",
            "nominal_thresholds": NOMINAL_THRESHOLDS,
            "primary_generalization_estimate": "pooled_five_fold_nested_outer_oof",
            "pooled_nested_outer_oof": train["pooled_metrics"]["nested_outer_oof"],
            "historical_primary_simple_cut_reference": {"r_hh_125_125_lt": 34.0},
            "historical_rhh125125_lt34": train["pooled_metrics"]["historical_rhh125125_lt34"],
            "fixed_nominal_deployment_cut": train["pooled_metrics"]["fixed_nominal_deployment_cut"],
            "paired_bootstrap": train["paired_metric_bootstrap"],
            "paired_bootstrap_combined_za_difference": combined_za_row,
            "conclusion": "no_material_stat_only_significance_improvement_over_historical_rhh125125_lt34",
            "all1000_selection_stability": {
                "ranked_structure_result_count": all1000["ranked_structure_result_count"],
                "fold_level_winner_count": all1000["fold_level_winner_count"],
                "replica_category_result_count": all1000["replica_category_result_count"],
                "categories": all1000["categories"],
                "detailed_structure_stability": "not_especially_stable_under_resampling",
                "nominal_deployment_candidate_changed": False,
            },
        },
        "normalization_and_wording": {
            "sqrt_s_tev": 13.0,
            "luminosity_fb_inverse_equivalent": 138.0,
            "yield_interpretation": "Run-2 expected-yield projection",
            "simulation": "Delphes simulation",
            "official_cms_result": False,
            "systematic_uncertainties_included": False,
        },
        "validation": {
            "attempt_status": "infrastructure_failure_before_event_access",
            "scientific_result_available": False,
            "infrastructure_failure_is_cut_performance_observation": False,
            "cluster_id": VALIDATION_CLUSTER,
            "authoritative_schedd": VALIDATION_SCHEDD,
            "submitted_jobs": 116,
            "queue_jobs": failure["queue_jobs"],
            "history_jobs": failure["history_jobs"],
            "exit_code_1_jobs": failure["exit_code_1_jobs"],
            "runtime_probe_failure_jobs": failure["runtime_probe_failure_jobs"],
            "failure_exception": failure["failure_exception"],
            "failure_stage": failure["failure_stage"],
            "root_cause": failure["root_cause"],
            "durable_source_attempt_markers_created": 0,
            "scientific_workers_started": 0,
            "validation_sources_opened": 0,
            "validation_evaluation_cycles": 0,
            "validation_payloads_opened": 0,
            "validation_metrics_available": False,
            "validation_aggregation_performed": False,
            "figure_17_validation_comparison_produced": False,
            "figure_17_omission_intentional": True,
            "test_payloads_opened": 0,
            "corrected_validation_retry_authorized": False,
            "never_resubmit": True,
            "runtime_bundle_sha256": EXPECTED_RUNTIME_SHA256,
            "frozen_runner_sha256": EXPECTED_FROZEN_RUNNER_SHA256,
            "preaccess_failure_auditor_sha256": sha256(repo / "scripts/analysis/audit_hh4b_cut_baseline_validation_preaccess_failure.py"),
            "failure_audit_checkpoint": failure_record["path"],
            "failure_audit_checkpoint_sha256s_sha256": failure_record["sha256s_sha256"],
        },
        "test": {
            "status": "sealed_not_opened",
            "access_authorized": False,
            "payloads_opened": 0,
        },
        "figures": {
            "train_only_figures": 16,
            "pdf_files": 16,
            "png_files": 16,
            "machine_readable_sidecars": 16,
            "figure_17_validation_comparison_produced": False,
            "inventory": [
                {
                    "path": path.relative_to(repo).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in [*pdfs, *pngs, *sidecars]
            ],
        },
        "clusters": clusters,
        "never_resubmit_registry": [row["cluster_id"] for row in clusters],
        "checkpoints": checkpoints,
        "validation_scientific_result_available": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
        "corrected_validation_retry_authorized": False,
        "next_decision_required": NEXT_DECISION,
    }

    if args.validate_only:
        print("BLOCKED_FINAL_REPORT_INPUT_AUDIT=PASS")
        print(f"REPOSITORY_PARENT_HEAD={head}")
        print("VALIDATION_SCIENTIFIC_RESULT_AVAILABLE=FALSE")
        print("VALIDATION_PAYLOADS_OPENED=0")
        print("TEST_PAYLOADS_OPENED=0")
        print("CORRECTED_VALIDATION_RETRY_AUTHORIZED=FALSE")
        return

    output_json = resolve(args.output_json)
    output_markdown = resolve(args.output_markdown)
    artifact_root = resolve(args.artifact_root)
    for output in (output_json, output_markdown):
        require(not output.exists(), f"refusing to overwrite report: {output}")
    require(not artifact_root.exists(), f"refusing to overwrite artifact root: {artifact_root}")
    artifact_root.mkdir(parents=True)
    json_text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    markdown_text = render_markdown(report)
    output_json.write_text(json_text, encoding="utf-8")
    output_markdown.write_text(markdown_text, encoding="utf-8")
    artifact_json = artifact_root / output_json.name
    artifact_markdown = artifact_root / output_markdown.name
    shutil.copyfile(output_json, artifact_json)
    shutil.copyfile(output_markdown, artifact_markdown)
    sums = artifact_root / "SHA256SUMS"
    sums.write_text(
        f"{sha256(artifact_json)}  {artifact_json.name}\n"
        f"{sha256(artifact_markdown)}  {artifact_markdown.name}\n",
        encoding="utf-8",
    )
    verify_sha256s(artifact_root)
    require(output_json.read_bytes() == artifact_json.read_bytes(), "JSON report copy changed")
    require(output_markdown.read_bytes() == artifact_markdown.read_bytes(), "Markdown report copy changed")

    print("BLOCKED_FINAL_REPORT=PASS")
    print(f"REPOSITORY_PARENT_HEAD={head}")
    print("TRAIN_ONLY_CUT_BASELINE_STUDY=COMPLETE")
    print("VALIDATION_SCIENTIFIC_RESULT_AVAILABLE=FALSE")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")
    print("CORRECTED_VALIDATION_RETRY_AUTHORIZED=FALSE")
    print("TRAIN_ONLY_FIGURES=16")
    print("FIGURE_17_INTENTIONALLY_OMITTED=TRUE")
    print(f"OUTPUT_JSON={output_json}")
    print(f"OUTPUT_MARKDOWN={output_markdown}")
    print(f"ARTIFACT_ROOT={artifact_root}")


if __name__ == "__main__":
    main()
